# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
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


@dataclass
class MMMeta:
    mm_hash: str
    num_token: int

    @staticmethod
    def make_meta(mm_hash, num_token) -> "MMMeta":
        return MMMeta(mm_hash=mm_hash, num_token=num_token)


@dataclass
class ECExampleConnectorMetadata(ECConnectorMetadata):
    mm_datas: list[MMMeta]

    def __init__(self):
        self.mm_datas = []

    def add_mm_data(self, mm_data: MMMeta):
        self.mm_datas.append(mm_data)


class ECExampleConnector(ECConnectorBase):
    # NOTE: This is Simple debug implementation of the EC connector.
    # It save / load the EC cache to / from the disk.

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        # req_id -> index
        self._mm_datas_need_loads: dict[str, int] = {}
        # Deferred deletion (producer side). In disagg E/PD the producer frees
        # an mm_hash as soon as it is encoded (E has no decode phase), so an
        # aggressive policy can evict+delete a shared-store file while the
        # consumer (PD) for the SAME request is still about to read it ->
        # FileNotFoundError and a dead engine. We split deletion in two:
        #   * logical: a queued hash is reported absent by has_cache_item, so
        #     the producer re-encodes immediately (policy effect, no bias);
        #   * physical: the file is only unlinked after a grace window (>> the
        #     PD read latency), so the in-flight transfer completes first.
        # PD is a separate connector instance (never a producer) so its
        # _pending_deletes stays empty and it still sees the physical file.
        # Tune the window via VLLM_EC_DELETE_GRACE_SEC (seconds).
        self._pending_deletes: dict[str, float] = {}
        self._delete_grace_s: float = float(
            os.environ.get("VLLM_EC_DELETE_GRACE_SEC", "10")
        )
        # Consume-on-load: make the shared store a single-use E->PD transfer
        # buffer. The consumer (PD) unlinks each file right after it has loaded
        # the embed into its in-GPU encoder_cache, so the store never grows
        # append-only (root cause of the /dev/shm host-RAM OOM that kills PD).
        # This does NOT change the research metrics: the producer (E) never
        # reads the store (has_cache_item is consumer-only), so E's policy
        # cache alone governs hit-rate / encode load; deleting the transfer
        # file only means a *different* PD that later needs the same hash
        # re-encodes locally (no effect on E-side encoder_forward / hit stats).
        # Set EC_STORE_CONSUME_ON_LOAD=0 to restore the append-only store.
        self._consume_on_load: bool = (
            os.environ.get("EC_STORE_CONSUME_ON_LOAD", "1") != "0"
        )
        transfer_config = vllm_config.ec_transfer_config
        if transfer_config is not None:
            self._storage_path = transfer_config.get_from_extra_config(
                "shared_storage_path", "/tmp"
            )
            logger.debug(transfer_config)
            logger.debug("Shared storage path is %s", self._storage_path)
        else:
            raise ValueError("ec_transfer_config must be set for ECConnectorBase")

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        """
        Start loading the cache from the connector into vLLM's encoder cache.

        This method loads the encoder cache based on metadata provided by the scheduler.
        It is called before `_gather_mm_embeddings` for the EC Connector. For EC,
        the `encoder_cache` and `mm_hash` are stored in `kwargs`.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            kwargs (dict): Additional keyword arguments for the connector.
        """
        from vllm.platforms import current_platform

        # Get the metadata
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ECExampleConnectorMetadata)
        assert encoder_cache is not None
        if metadata is None:
            logger.warning(
                "In connector.start_load_caches, but the connector metadata is None"
            )
            return
        # Load the EC for each mm data. Time it so the proxy's pd_first_chunk
        # can be decomposed: this embed load (file read + H2D) happens inside
        # the PD execute step, before prefill, and is not covered by any metric.
        import time as _ec_t
        _ec_t0 = _ec_t.perf_counter()
        _ec_loaded = 0
        for mm_data in metadata.mm_datas:
            if mm_data.mm_hash in encoder_cache:
                continue
            filename = self._generate_filename_debug(mm_data.mm_hash)
            ec_cache = safetensors.torch.load_file(
                filename, device=current_platform.device_type
            )["ec_cache"]
            encoder_cache[mm_data.mm_hash] = ec_cache
            _ec_loaded += 1
            logger.debug("Success load encoder cache for hash %s", mm_data.mm_hash)
            # Single-use transfer buffer: the embed now lives in PD's in-GPU
            # encoder_cache, so the on-disk copy is free to go. Unlink it (and
            # its folder) directly — do NOT go through _generate_filename_debug,
            # which would re-create the directory it is trying to remove.
            if self._consume_on_load:
                folder = os.path.join(self._storage_path, mm_data.mm_hash)
                try:
                    os.remove(os.path.join(folder, "encoder_cache.safetensors"))
                    os.rmdir(folder)
                except OSError:
                    pass
        if metadata.mm_datas:
            logger.info(
                "[ECLoad] loaded=%d of=%d ms=%.1f",
                _ec_loaded,
                len(metadata.mm_datas),
                (_ec_t.perf_counter() - _ec_t0) * 1000.0,
            )

    def save_caches(self, encoder_cache, mm_hash, **kwargs) -> None:
        """
        Save the encoder cache to the connector.

        This method saves the encoder cache from the worker's local storage
        to shared storage or another external connector.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            mm_hash (str): The hash of the multimodal data whose cache is being saved.
            kwargs (dict): Additional keyword arguments for the connector.
        """
        # Return if it is PD Instance
        if not self.is_producer:
            return
        # A re-encode of a hash that was queued for deletion cancels that
        # pending delete — the file is being (re)written right now.
        self._pending_deletes.pop(mm_hash, None)
        filename = self._generate_filename_debug(mm_hash)
        ec_cache = encoder_cache[mm_hash]
        tensors = {"ec_cache": ec_cache.detach().cpu()}
        safetensors.torch.save_file(tensors, filename)
        logger.debug("Save cache successful for mm_hash %s", mm_hash)
        # Make deferred physical deletions progress even if evictions pause.
        self._flush_pending_deletes()

    def has_cache_item(
        self,
        identifier: str,
    ) -> bool:
        """
        Check if cache exist externally for the media

        Args:
            identifier (str): the identifier of the media.

        Returns:
            Bool indicate that media exists in cache or not
        """
        return self._found_match_for_mm_data(identifier)

    def delete_caches(self, mm_hashes: list[str]) -> None:
        """Remove persisted embeddings for evicted mm_hashes (producer only).

        Makes the shared store track the encoder-cache policy instead of
        growing append-only; see ECConnectorBase.delete_caches.

        Deletion is split logical/physical to avoid a load/delete race that
        kills the engine in disagg E/PD (see __init__): the hash is queued
        (logically gone immediately for has_cache_item) and the file is only
        unlinked after a grace window so an in-flight consumer read finishes.
        """
        if not self.is_producer:
            return
        now = time.monotonic()
        for mm_hash in mm_hashes:
            self._pending_deletes.setdefault(mm_hash, now)
        self._flush_pending_deletes()

    def _flush_pending_deletes(self) -> None:
        """Physically unlink queued files whose grace window has elapsed."""
        if not self._pending_deletes:
            return
        now = time.monotonic()
        due = [
            h
            for h, queued_at in self._pending_deletes.items()
            if now - queued_at >= self._delete_grace_s
        ]
        for mm_hash in due:
            self._pending_deletes.pop(mm_hash, None)
            foldername = self._generate_foldername_debug(mm_hash, create_folder=False)
            filename = os.path.join(foldername, "encoder_cache.safetensors")
            try:
                if os.path.exists(filename):
                    os.remove(filename)
                if os.path.isdir(foldername):
                    os.rmdir(foldername)
                logger.debug("Deleted shared-store cache for mm_hash %s", mm_hash)
            except OSError as e:
                logger.warning(
                    "Failed to delete shared-store cache for %s: %s", mm_hash, e
                )

    def update_state_after_alloc(
        self,
        request: "Request",
        index: int,
    ) -> None:
        """
        Update ECConnector state after encoder cache allocation.
        """
        mm_hash = request.mm_features[index].identifier
        # Only load cache if it is consumer and cache exists
        if not self.is_consumer or not self.has_cache_item(mm_hash):
            return
        num_encoder_token = request.get_num_encoder_embeds(index)
        self._mm_datas_need_loads[mm_hash] = num_encoder_token

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ECConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.
        This only build for load mm_data only
        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = ECExampleConnectorMetadata()
        for mm_hash, num_encoder_token in self._mm_datas_need_loads.items():
            meta.add_mm_data(MMMeta.make_meta(mm_hash, num_encoder_token))
        self._mm_datas_need_loads.clear()
        return meta

    # ==============================
    # Helper functions
    # ==============================

    def _found_match_for_mm_data(self, mm_hash) -> bool:
        """Check if the cache is hit for the request.

        A hash queued for deletion is reported absent so the producer
        re-encodes it (the policy evicted it); the physical file may still
        linger during its grace window purely so an in-flight consumer read
        can complete. PD instances never queue deletes, so they still hit.
        """
        if mm_hash in self._pending_deletes:
            return False
        filename = self._generate_filename_debug(mm_hash)
        return os.path.exists(filename)

    def _generate_foldername_debug(
        self,
        mm_hash: str,
        create_folder: bool = True,  # <- now defaults to True
    ) -> str:
        """
        Return the folder in which the cache for this mm_hash lives.
        If `create_folder` is True (default) the directory is created
        recursively the first time it is needed.
        """
        foldername = os.path.join(self._storage_path, mm_hash)
        if create_folder:
            os.makedirs(foldername, exist_ok=True)
        return foldername

    def _generate_filename_debug(self, mm_hash: str) -> str:
        """
        Return the full path of the safetensors file for this mm_hash.
        Ensures the parent directory exists because
        `_generate_foldername_debug` is called with its default
        (`create_folder=True`).
        """
        foldername = self._generate_foldername_debug(mm_hash)  # <- folder auto-created
        return os.path.join(foldername, "encoder_cache.safetensors")
