# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import time
from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Optional

from vllm.logger import init_logger
from vllm.v1.core.encoder_cache_policy import (
    EncoderCachePolicy,
    FIFOEncoderCachePolicy,
    build_policy_from_env,
)
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import SchedulerConfig

logger = init_logger(__name__)


class EncoderCacheManager:
    """Manages caching of encoder outputs for multimodal models in vLLM V1.

    The EncoderCacheManager handles the lifecycle of multimodal encoder outputs
    (such as vision embeddings from images) during request processing. It
    provides memory-aware caching to avoid recomputing encoder outputs when the
    same multimodal inputs appear in different stages of request processing.

    This manager is particularly important for:
    - Vision-language models (e.g., LLaVA) where image encoder outputs are
      cached
    - Any multimodal model where encoder computation is expensive and
      cacheable

    The cache operates at the granularity of individual multimodal input items
    within requests, allowing for fine-grained memory management and enabling
    chunked processing of multimodal inputs.

    Cache is enabled to share embeddings of same multimodal data
    item (identified by their hash value) between different requests,
    and eviction takes place at allocation time when there's no free
    space for new embeddings.

    Eviction order over the freeable queue is delegated to an
    :class:`EncoderCachePolicy` (``policy``). By default the legacy FIFO
    behavior is preserved. Policies can pin freeable entries for a number
    of mm-item ticks; pinned entries are skipped during eviction unless
    the cache is fully pinned, in which case the earliest-inserted pinned
    entry is force-evicted and ``forced_unpin_evictions`` is incremented.

    Args:
        cache_size: Limit the size of the cache, measured by the number of
                    encoder embeddings from the input sequence.
        policy: Eviction-decision strategy. Defaults to
            ``FIFOEncoderCachePolicy`` for legacy behavior.

    Attributes:
        cache_size: Total cache capacity in encoder embeddings.
        num_free_slots: Current available cache capacity in encoder embeddings.
        num_freeable_slots: Capacity that can be immediately reclaimed by
            evicting entries with zero references (in encoder embeddings).
        cached: Mapping from mm_hash to a set of request IDs that currently
            reference the cached entry. If the set is empty, the entry exists
            but is not referenced by any request and is eligible for
            reclamation.
        freeable: List of tuples (mm_hash, num_encoder_embeds) representing entries
            whose no current running request is needed and that can be freed to
            make space when needed.
        freed: List of mm_hash strings that were actually evicted since the
            last call to get_freed_mm_hashes(). This list is cleared on return.
        policy: Active eviction policy.
        current_tick: Monotonically increasing counter of mm-item arrival
            events (both hits and successful allocations). Drives pin
            timestamps so they are independent of wall clock.
        hits / misses / forced_unpin_evictions: Cumulative counters surfaced
            via :meth:`get_stats` for benchmarking.
    """

    def __init__(
        self,
        cache_size: int,
        policy: Optional[EncoderCachePolicy] = None,
    ):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.num_freeable_slots = cache_size

        # mm_hash of mm_data => ids of requests that reference the mm_data
        self.cached: dict[str, set[str]] = {}

        # mm_hash of mm_data => num_encoder_embeds of the mm_data
        self.freeable: OrderedDict[str, int] = OrderedDict()
        self.freed: list[str] = []

        self.policy: EncoderCachePolicy = policy or FIFOEncoderCachePolicy()
        # Per-mm_hash absolute tick before which this entry is protected
        # from eviction. Missing key == 0 == evictable now. Populated
        # at the moment an entry transitions to the freeable queue —
        # not at arrival — so the pin actually covers the freeable
        # window rather than expiring while the request is in flight.
        self._unlock_tick: dict[str, int] = {}
        # Pin horizon (in ticks) most recently requested by the policy
        # for each mm_hash. Stored at arrival, applied when the entry
        # enters freeable.
        self._pin_horizon: dict[str, int] = {}

        # Metrics: mm-item-level arrivals and forced-eviction events.
        self.current_tick: int = 0
        self.hits: int = 0
        self.misses: int = 0
        self.forced_unpin_evictions: int = 0

        # Periodic stats logging for benchmark observability.
        try:
            self._stats_log_interval = float(
                os.environ.get("VLLM_ENCODER_CACHE_STATS_INTERVAL_SEC", "0")
            )
        except ValueError:
            self._stats_log_interval = 0.0
        self._last_stats_log_ts = time.monotonic()

    def reset(self) -> None:
        """Reset the encoder cache to its initial state.

        This clears all cached encoder outputs and resets capacity tracking.
        Called when model weights are updated to invalidate stale embeddings.
        """
        self.cached.clear()
        self.freeable.clear()
        self.freed.clear()
        self._unlock_tick.clear()
        self._pin_horizon.clear()
        self.num_free_slots = self.cache_size
        self.num_freeable_slots = self.cache_size
        self.current_tick = 0
        self.hits = 0
        self.misses = 0
        self.forced_unpin_evictions = 0
        self.policy.reset()

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        """Check if encoder output for a specific multimodal input is cached.

        If the encoder output is cached, update `cached` to add the request id
        to the set of request ids that reference the cached encoder output.
        If the encoder output was previously not referenced by any request,
        update `freeable` and `num_freeable_slots` accordingly.

        Args:
            request: The request containing the multimodal input
            input_id: Index of the multimodal input within the request

        Returns:
            True if the encoder output for this input is already cached
        """
        mm_hash = request.mm_features[input_id].identifier
        # Not cached at all
        if mm_hash not in self.cached:
            return False

        # Cached but currently not referenced by any request
        if not self.cached[mm_hash]:
            num_encoder_embeds = self.freeable.pop(mm_hash)
            self.num_freeable_slots -= num_encoder_embeds

        self.cached[mm_hash].add(request.request_id)
        # Hit: account for the arrival and let policy refresh the pin.
        self.hits += 1
        self.current_tick += 1
        num_embeds = request.get_num_encoder_embeds(input_id)
        # Policy returns a horizon (in ticks). We do not apply it now
        # because the entry has just become referenced and is no
        # longer in the freeable queue; the horizon will be applied
        # to ``_unlock_tick`` when refs drop to 0 again.
        horizon = self.policy.on_arrival(mm_hash, num_embeds, self.current_tick)
        if horizon > 0:
            self._pin_horizon[mm_hash] = horizon
        else:
            self._pin_horizon.pop(mm_hash, None)
        return True

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        """Check if there's sufficient cache space for a multimodal input.
        If there is, return True and update EncoderCacheManager state.

        If there is not enough free space in `num_free_slots` but there is
        enough reclaimable space in `num_freeable_slots`, entries will be
        evicted from `freeable` (their mm_hash appended to `freed`) until
        enough space is available, and then this method returns True.
        The active :class:`EncoderCachePolicy` controls which entry to
        evict next: pinned entries are skipped while any unpinned candidate
        remains; if no unpinned candidate exists, the FIFO-front pinned
        entry is force-evicted and ``forced_unpin_evictions`` is incremented.

        Returns False only if the requested number of tokens exceeds both
        the free and reclaimable capacities combined.

        Args:
            request: The request containing the multimodal input.
            input_id: Index of the multimodal input within the request.
            encoder_compute_budget: Number of encoder embeddings allowed to be
                computed when this method is invoked.
            num_embeds_to_schedule: Number of encoder embeddings already scheduled to be
                allocated with cache space when this method is invoked.

        Returns:
            True if there's enough capacity to hold the encoder output for this
            input (possibly after reclaiming `freeable` entries); otherwise
            False.

        Note: This method does not allocate physical memory for the encoder
        output but only the state of EncoderCacheManager.
        """
        num_embeds = request.get_num_encoder_embeds(input_id)

        # Not enough compute budget
        if num_embeds > encoder_compute_budget:
            return False

        num_embeds += num_embeds_to_schedule

        # Enough free slots
        if num_embeds <= self.num_free_slots:
            return True

        # Not enough reclaimable slots
        if num_embeds > self.num_freeable_slots:
            return False

        # Not enough free slots but enough reclaimable slots.
        # NOTE: Eviction takes place here, but physical memory is not freed
        # until model runner is notified by the scheduler output.
        while num_embeds > self.num_free_slots:
            mm_hash = self._select_eviction_candidate(forced=False)
            if mm_hash is None:
                # All freeable entries are pinned; the policy must yield.
                mm_hash = self._select_eviction_candidate(forced=True)
                if mm_hash is None:
                    # Should not happen since num_freeable_slots >= num_embeds.
                    return False
                self.forced_unpin_evictions += 1
            self._evict(mm_hash)
        return True

    def _select_eviction_candidate(self, forced: bool) -> Optional[str]:
        """Pick the next mm_hash to evict from the freeable queue.

        With ``forced=False`` evicts the FIFO-front *unpinned* entry,
        skipping entries whose unlock tick is still in the future. Returns
        ``None`` when every freeable entry is pinned.

        With ``forced=True`` every freeable entry is pinned and one must be
        given up. Rather than blindly dropping the FIFO-front entry (which
        may be a hot, expensive-to-recompute image), evict the entry the
        policy values least — ``policy.eviction_value`` — so we free space
        at the lowest expected recompute cost. Ties are broken by FIFO
        insertion order (the earliest-inserted of the cheapest entries is
        evicted first). For policies that do not override ``eviction_value``
        (FIFO/no-cache) every value is 0.0, so this reduces to FIFO order
        and the legacy behavior is preserved.
        """
        if not forced:
            for mm_hash in self.freeable:
                if self._unlock_tick.get(mm_hash, 0) <= self.current_tick:
                    return mm_hash
            return None

        best_hash: Optional[str] = None
        best_value = float("inf")
        for mm_hash, num_embeds in self.freeable.items():
            value = self.policy.eviction_value(mm_hash, num_embeds)
            if value < best_value:
                best_value = value
                best_hash = mm_hash
        return best_hash

    def _evict(self, mm_hash: str) -> None:
        """Physically evict an entry from the cache bookkeeping."""
        num_free_embeds = self.freeable.pop(mm_hash)
        del self.cached[mm_hash]
        self._unlock_tick.pop(mm_hash, None)
        self.freed.append(mm_hash)
        self.num_free_slots += num_free_embeds

    def allocate(self, request: Request, input_id: int) -> None:
        """Allocate cache space for a multimodal input's encoder output.

        This reserves cache space for storing the encoder output of the
        specified multimodal input. The actual encoder output storage happens in
        the model runner; this method updates the manager's bookkeeping.

        Note:
            This method assumes can_allocate() returned True for the same input.
        """

        mm_hash = request.mm_features[input_id].identifier
        request_id = request.request_id
        if mm_hash not in self.cached:
            self.cached[mm_hash] = set()

        num_encoder_embeds = request.get_num_encoder_embeds(input_id)

        # NOTE: Encoder cache should always have enough space for encoder inputs
        # that are scheduled since eviction takes place at can_allocate().
        assert self.num_free_slots >= num_encoder_embeds
        assert self.num_freeable_slots >= num_encoder_embeds

        self.cached[mm_hash].add(request_id)
        self.num_free_slots -= num_encoder_embeds
        self.num_freeable_slots -= num_encoder_embeds

        # Miss: this is the request that triggered allocation. Count
        # exactly once per unique mm_item -> drives policy's view of d_i.
        self.misses += 1
        self.current_tick += 1
        # See note in check_and_update_cache: the policy returns a
        # horizon (ticks) that we stash here and apply to
        # ``_unlock_tick`` when refs eventually drop to 0.
        horizon = self.policy.on_arrival(
            mm_hash, num_encoder_embeds, self.current_tick
        )
        if horizon > 0:
            self._pin_horizon[mm_hash] = horizon
        else:
            self._pin_horizon.pop(mm_hash, None)

    def get_cached_input_ids(self, request: Request) -> set[int]:
        """Get all cached multimodal input IDs for a request.

        Returns the set of input IDs whose `mm_hash` exists in the cache map.
        This includes entries that are currently unreferenced (and thus present
        in `freeable`); for such entries, freeing for this request will be a
        no-op.
        """
        return {
            input_id
            for input_id in range(len(request.mm_features))
            if request.mm_features[input_id].identifier in self.cached
        }

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        """Free the request's reference to the encoder input (`mm_data`)

        When the reference set for the corresponding `mm_hash` becomes empty,
        the entry is appended to `freeable` and `num_freeable_slots` is
        increased by the number of encoder embeddings for that input. If the
        active policy declares ``evict_on_unreference`` (e.g. the no-cache
        ablation), the entry is dropped immediately instead.

        Outside of ``evict_on_unreference``, the entry is NOT physically freed
        until capacity is needed (e.g., by `can_allocate`).
        """
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        # The mm_hash not in cache or the req_id set is empty
        if not self.cached.get(mm_hash, None):
            return
        self.cached[mm_hash].discard(req_id)
        if self.cached[mm_hash]:
            return
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        if self.policy.evict_on_unreference:
            # Ablation: drop immediately, no freeable buffer.
            del self.cached[mm_hash]
            self._unlock_tick.pop(mm_hash, None)
            self._pin_horizon.pop(mm_hash, None)
            self.freed.append(mm_hash)
            self.num_free_slots += num_encoder_embeds
            # num_freeable_slots tracks num_free_slots in this mode since
            # the freeable queue is never populated.
            self.num_freeable_slots += num_encoder_embeds
        else:
            # Entry transitions to freeable: NOW is when the pin
            # actually starts counting. Materialize the previously
            # requested horizon into an absolute unlock tick.
            horizon = self._pin_horizon.pop(mm_hash, 0)
            if horizon > 0:
                self._unlock_tick[mm_hash] = self.current_tick + horizon
            else:
                self._unlock_tick.pop(mm_hash, None)
            self.freeable[mm_hash] = num_encoder_embeds
            self.num_freeable_slots += num_encoder_embeds

    def free(self, request: Request) -> None:
        """Free all encoder input cache reference held by *request*.

        For each cached input ID, `free_encoder_input` is invoked.
        The data stays in memory until eviction is triggered by a future
        attempt allocation called by 'can_allocate'.

        Typically called when a request is finished, cancelled, or aborted.
        """
        input_ids = self.get_cached_input_ids(request)
        for input_id in input_ids:
            self.free_encoder_input(request, input_id)

    def get_freed_mm_hashes(self) -> list[str]:
        """Get and clear the list of recently freed encoder cache entries.

        Returns:
            List of mm_hash strings that were actually evicted since the last
            call to be used by the scheduler to notify workers about which
            encoder outputs can be removed from their caches. The internal
            list is cleared after this call.
        """
        freed = self.freed
        self.freed = []
        return freed

    def maybe_log_stats(self) -> None:
        """Emit a one-line summary at most every
        ``VLLM_ENCODER_CACHE_STATS_INTERVAL_SEC`` seconds.

        Disabled (no-op) when the interval is 0 or negative. Designed
        to be called from the scheduler's per-step ``make_stats``.
        """
        if self._stats_log_interval <= 0.0:
            return
        now = time.monotonic()
        if now - self._last_stats_log_ts < self._stats_log_interval:
            return
        self._last_stats_log_ts = now
        stats = self.get_stats()
        # Surface known vs unknown mm_hash classification from policies
        # that maintain a pool registry; useful for diagnosing hash
        # mismatch between offline pool generation and server-side
        # hashing (manifests as all arrivals being 'unknown').
        known = getattr(self.policy, "known_hits", None)
        unknown = getattr(self.policy, "unknown_hits", None)
        pool_suffix = ""
        if known is not None and unknown is not None:
            pool_suffix = f" pool_known={known} pool_unknown={unknown}"
        logger.info(
            "encoder_cache policy=%s hits=%d misses=%d hit_rate=%.4f "
            "forced_unpin=%d pinned=%d freeable=%d free_slots=%d%s",
            stats["policy"],
            stats["hits"],
            stats["misses"],
            stats["hit_rate"],
            stats["forced_unpin_evictions"],
            stats["num_pinned"],
            stats["num_freeable"],
            stats["num_free_slots"],
            pool_suffix,
        )

    def get_stats(self) -> dict[str, int | float | str]:
        """Snapshot of cumulative cache metrics for benchmarking."""
        total = self.hits + self.misses
        return {
            "policy": type(self.policy).__name__,
            "hits": self.hits,
            "misses": self.misses,
            "arrivals": total,
            "hit_rate": (self.hits / total) if total else 0.0,
            "forced_unpin_evictions": self.forced_unpin_evictions,
            "num_pinned": sum(
                1
                for mm_hash in self.freeable
                if self._unlock_tick.get(mm_hash, 0) > self.current_tick
            ),
            "num_freeable": len(self.freeable),
            "num_free_slots": self.num_free_slots,
        }


def compute_mm_encoder_budget(
    scheduler_config: "SchedulerConfig",
    mm_max_toks_per_item: Mapping[str, int],
) -> tuple[int, int]:
    """Compute the encoder cache budget based on the model and scheduler
    configurations for a multimodal model.

    Args:
        scheduler_config: Scheduler configuration.
        mm_max_toks_per_item: The maximum number of tokens per item for each
            non-text modality.

    Returns:
        - Compute budget for encoder execution, measured in number of tokens
            from the input sequence.
        - Space budget for encoder cache size, measured in number of tokens
            from the input sequence.
    """

    if not mm_max_toks_per_item:
        logger.warning(
            "All non-text modalities supported by the model have been "
            "explicitly disabled via limit_mm_per_prompt. Encoder cache will "
            "not be initialized."
        )
        return 0, 0

    max_tokens_per_mm_item = max(mm_max_toks_per_item.values())

    if (
        scheduler_config.disable_chunked_mm_input
        and max_tokens_per_mm_item > scheduler_config.max_num_batched_tokens
    ):
        raise ValueError(
            "Chunked MM input disabled but max_tokens_per_mm_item "
            f"({max_tokens_per_mm_item}) is larger than max_num_batched_tokens"
            f" ({scheduler_config.max_num_batched_tokens}). Please increase "
            "max_num_batched_tokens."
        )

    encoder_compute_budget = max(
        scheduler_config.max_num_encoder_input_tokens, max_tokens_per_mm_item
    )
    encoder_cache_size = max(
        scheduler_config.encoder_cache_size, max_tokens_per_mm_item
    )

    return encoder_compute_budget, encoder_cache_size


# NOTE (NickLucche): Temporary implementation for encoder-decoder models that only
# use the manager for scheduling purposes. Encoder-decoder models will eventually
# utilize the cache and this class will fold into EncoderCacheManager, as
# differences with MM models shrink.
class EncoderDecoderCacheManager(EncoderCacheManager):
    def __init__(
        self,
        cache_size: int,
        policy: Optional[EncoderCachePolicy] = None,
    ):
        # Intentionally do not call super().__init__: this class uses a
        # different state shape.
        del policy  # encoder-decoder path does not use the cross-request cache
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.allocated: list[str] = []
        self.to_free: list[str] = []

    def reset(self) -> None:
        """Reset the encoder cache to its initial state."""
        self.num_free_slots = self.cache_size
        self.allocated.clear()
        self.to_free.clear()

    def check_and_update_cache(self, request: Request, input_id: int) -> bool:
        return False

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        # Not enough compute budget
        if num_encoder_embeds > encoder_compute_budget:
            return False

        num_encoder_embeds += num_embeds_to_schedule
        # Enough free slots
        return num_encoder_embeds <= self.num_free_slots

    def allocate(self, request: Request, input_id: int) -> None:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        self.num_free_slots -= num_encoder_embeds

        mm_hash = request.mm_features[input_id].identifier
        self.allocated.append(mm_hash)

    def free(self, request: Request) -> None:
        for input_id in range(len(request.mm_features)):
            self.free_encoder_input(request, input_id)

    def get_cached_input_ids(self, request: Request) -> set[int]:
        return set(range(len(request.mm_features)))

    def get_freed_mm_hashes(self) -> list[str]:
        # As encoder cache is not used for enc-dec models, we can free the entries here
        # The actual free happens in the runner, *before* the model is executed.
        # Therefore, `freeable` acts as a buffer to free the entries only after the
        # model is executed, mimicking the state transition of `EncoderCacheManager`.
        to_free = self.to_free
        self.to_free = self.allocated
        self.allocated = []
        return to_free

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        num_encoder_embeds = request.get_num_encoder_embeds(input_id)
        self.num_free_slots += num_encoder_embeds


__all__ = [
    "EncoderCacheManager",
    "EncoderDecoderCacheManager",
    "compute_mm_encoder_budget",
    "build_policy_from_env",
]
