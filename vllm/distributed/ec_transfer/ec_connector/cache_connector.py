# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ECCacheConnector — turn the L3 encoder-cache store from pure storage into a
BOUNDED cache with its own capacity + eviction (P1: byte-bounded LRU).

Design invariants (see docs/L3_cache_改造spec.md):
  * L3 evicts ONLY on its OWN byte capacity (in save_caches). It NEVER
    subscribes to L2 (encoder_cache_manager) eviction — L2 and L3 are two
    independent caches. (fork's L2->L3 cascade is deliberately NOT ported.)
  * Producer (E) is the single writer + single evictor. Consumers (PD) only
    read. Cross-process read/delete safety = atomic write (.tmp->replace),
    an eviction grace window, and miss-is-recoverable loads (a consumer that
    loses the race just recomputes locally, never crashes).
  * Eviction ordering is delegated to `_evict_candidate()` — LRU here, a
    value-density (p*c/m) policy can drop in later without touching the rest.
"""
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import safetensors

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)

_DEFAULT_STORAGE = "/dev/shm/ec_cache"
_DEFAULT_MAX_BYTES = 4 * 1024**3  # 4 GiB
_DEFAULT_GRACE_SEC = 10.0


@dataclass
class MMMeta:
    mm_hash: str
    num_token: int


@dataclass
class _Entry:
    """Producer-side index record for one cached embedding."""

    size_bytes: int
    last_access: float
    ref: int = 0
    hits: int = 0  # reuse counter — feeds value-density later (p)


class ECCacheConnectorMetadata(ECConnectorMetadata):
    def __init__(self):
        self.mm_datas: list[MMMeta] = []

    def add_mm_data(self, mm_hash: str, num_token: int) -> None:
        self.mm_datas.append(MMMeta(mm_hash=mm_hash, num_token=num_token))


class ECCacheConnector(ECConnectorBase):
    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        cfg = vllm_config.ec_transfer_config
        if cfg is None:
            raise ValueError("ec_transfer_config must be set for ECCacheConnector")
        self._storage_path = cfg.get_from_extra_config(
            "shared_storage_path", _DEFAULT_STORAGE
        )
        self._capacity_bytes = int(
            cfg.get_from_extra_config("ec_cache_max_bytes", _DEFAULT_MAX_BYTES)
        )
        self._grace_sec = float(
            cfg.get_from_extra_config("ec_cache_evict_grace_sec", _DEFAULT_GRACE_SEC)
        )
        os.makedirs(self._storage_path, exist_ok=True)

        # Producer-side cache index (writer/evictor owns capacity accounting).
        self._lock = threading.Lock()
        self._index: OrderedDict[str, _Entry] = OrderedDict()  # LRU: front = oldest
        self._used_bytes = 0

        # Consumer-side: mm_hash -> num_token that this step must load.
        self._need_loads: dict[str, int] = {}

        # Observability counters (prove cache-not-storage: saves/evicts/hits/miss).
        self._n_save = 0
        self._n_evict = 0
        self._n_hit = 0
        self._n_miss = 0

        logger.info(
            "ECCacheConnector role=%s path=%s cap=%.2fGiB grace=%.1fs",
            role,
            self._storage_path,
            self._capacity_bytes / 1024**3,
            self._grace_sec,
        )

    # ==============================
    # Producer (E): write + evict
    # ==============================
    def save_caches(self, encoder_cache, mm_hash, **kwargs) -> None:
        if not self.is_producer:
            return
        ec = encoder_cache[mm_hash]
        tensors = {"ec_cache": ec.detach().cpu()}
        size = tensors["ec_cache"].numel() * tensors["ec_cache"].element_size()

        with self._lock:
            if mm_hash in self._index:  # already stored — refresh LRU only
                self._touch(mm_hash)
                return
            self._ensure_capacity(size)
            self._atomic_write(mm_hash, tensors)
            self._index[mm_hash] = _Entry(size_bytes=size, last_access=time.monotonic())
            self._used_bytes += size
            self._n_save += 1
        if self._n_save % 10 == 0:
            logger.info(
                "[L3 stats] save=%d evict=%d hit=%d miss=%d used=%.0fMB/%.0fMB entries=%d",
                self._n_save, self._n_evict, self._n_hit, self._n_miss,
                self._used_bytes / 1e6, self._capacity_bytes / 1e6, len(self._index))

    def _ensure_capacity(self, incoming: int) -> None:
        """Evict LRU, unreferenced, out-of-grace entries until `incoming` fits.
        If nothing is evictable we allow a transient over-fill rather than block
        the encoder (correctness > strict bound)."""
        while self._used_bytes + incoming > self._capacity_bytes:
            victim = self._evict_candidate()
            if victim is None:
                logger.warning(
                    "L3 over capacity (used=%d cap=%d) but no evictable entry",
                    self._used_bytes, self._capacity_bytes)
                return
            self._delete_entry(victim)

    def _evict_candidate(self) -> str | None:
        """Pluggable eviction choice. P1 = LRU: oldest entry with ref==0 and
        past the grace window. (Swap this body for value-density p*c/m later.)"""
        now = time.monotonic()
        for mm_hash, e in self._index.items():  # LRU order, oldest first
            if e.ref == 0 and (now - e.last_access) >= self._grace_sec:
                return mm_hash
        return None

    def _delete_entry(self, mm_hash: str) -> None:
        try:
            os.remove(self._filename(mm_hash))
        except FileNotFoundError:
            pass
        e = self._index.pop(mm_hash, None)
        if e is not None:
            self._used_bytes -= e.size_bytes
            self._n_evict += 1

    def _atomic_write(self, mm_hash: str, tensors) -> None:
        final = self._filename(mm_hash)
        os.makedirs(os.path.dirname(final), exist_ok=True)
        tmp = f"{final}.tmp.{os.getpid()}"
        safetensors.torch.save_file(tensors, tmp)
        os.replace(tmp, final)  # atomic on same fs -> readers never see torn files

    # ==============================
    # Consumer (PD): reserve + load
    # ==============================
    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        mm_hash = request.mm_features[index].identifier
        if not self.is_consumer or not self.has_cache_item(mm_hash):
            return
        self._need_loads[mm_hash] = request.get_num_encoder_embeds(index)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> ECConnectorMetadata:
        meta = ECCacheConnectorMetadata()
        for mm_hash, ntok in self._need_loads.items():
            meta.add_mm_data(mm_hash, ntok)
        self._need_loads.clear()
        return meta

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        from vllm.platforms import current_platform

        meta = self._get_connector_metadata()
        assert isinstance(meta, ECCacheConnectorMetadata)
        for mm in meta.mm_datas:
            if mm.mm_hash in encoder_cache:
                continue
            fn = self._filename(mm.mm_hash)
            try:
                ec = safetensors.torch.load_file(
                    fn, device=current_platform.device_type
                )["ec_cache"]
            except (FileNotFoundError, OSError) as ex:
                # Recoverable miss: evicted/torn between check and load.
                # Leave it absent -> PD re-encodes locally. Never crash.
                self._n_miss += 1
                logger.debug("L3 miss for %s (%s) -> local recompute", mm.mm_hash, ex)
                continue
            encoder_cache[mm.mm_hash] = ec
            self._n_hit += 1
            if self._n_hit % 10 == 0:
                logger.info("[L3 stats] load-hit=%d miss=%d", self._n_hit, self._n_miss)
            self._on_hit(mm.mm_hash)

    def has_cache_item(self, identifier: str) -> bool:
        # Filesystem is the cross-process source of truth (index is per-process).
        return os.path.exists(self._filename(identifier))

    # ==============================
    # Helpers
    # ==============================
    def _on_hit(self, mm_hash: str) -> None:
        """Producer-local bookkeeping on a served hit (LRU refresh + p counter).
        No-op on a pure consumer whose index doesn't hold the entry."""
        with self._lock:
            if mm_hash in self._index:
                self._index[mm_hash].hits += 1
                self._touch(mm_hash)

    def _touch(self, mm_hash: str) -> None:
        e = self._index.get(mm_hash)
        if e is not None:
            e.last_access = time.monotonic()
            self._index.move_to_end(mm_hash)  # LRU: most-recent at back

    def _filename(self, mm_hash: str) -> str:
        return os.path.join(self._storage_path, mm_hash, "encoder_cache.safetensors")
