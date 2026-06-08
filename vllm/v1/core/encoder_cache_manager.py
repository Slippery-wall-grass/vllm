# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
import random
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.logger import init_logger
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
    Oldest cached embeddings with no request referenced will be first evicted.

    NOTE: The EncoderCacheManager operates on the level of multimodal embeddings
    instead of encoder tokens (i.e. all tokens that represent the multimodal data
    in the input sequence). This means all break/text tokens in-between multimodal
    embeddings are not considered with respect to the cache size and the number
    of free slots.

    Args:
        cache_size: Limit the size of the cache, measured by the number of
                    encoder embeddings from the input sequence.

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
    """

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.num_free_slots = cache_size
        self.num_freeable_slots = cache_size
        # Single greppable line for benchmarks / sanity checks: confirms
        # vLLM's effective encoder cache size at startup. Useful when
        # offline tools (solve_lambda.py) assumed a different B and the
        # runtime re-solves with this value instead.
        logger.info(
            "EncoderCacheManagerInit class=%s cache_size=%d",
            type(self).__name__, cache_size,
        )

        # mm_hash of mm_data => ids of requests that reference the mm_data
        self.cached: dict[str, set[str]] = {}

        # mm_hash of mm_data => num_encoder_embeds of the mm_data
        self.freeable: OrderedDict[str, int] = OrderedDict()
        self.freed: list[str] = []

        # Hit / miss counters used by external benchmarks. Logged on reset()
        # and cleared so each measurement window has clean numbers.
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    def reset(self) -> None:
        """Reset the encoder cache to its initial state.

        This clears all cached encoder outputs and resets capacity tracking.
        Called when model weights are updated to invalidate stale embeddings.
        """
        # Log hit/miss stats before clearing them so external tools (e.g.
        # the encoder-cache benchmark) can grep them out of the worker log.
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        logger.info(
            "Encoder cache stats before reset: "
            "hits=%d misses=%d total=%d hit_rate=%.4f",
            self.cache_hits, self.cache_misses, total, hit_rate,
        )

        self.cached.clear()
        self.freeable.clear()
        self.freed.clear()
        self.num_free_slots = self.cache_size
        self.num_freeable_slots = self.cache_size
        self.cache_hits = 0
        self.cache_misses = 0

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
            self.cache_misses += 1
            self._maybe_log_trace(request, mm_hash, hit=False)
            return False

        # Cached but currently not referenced by any request
        if not self.cached[mm_hash]:
            num_encoder_embeds = self.freeable.pop(mm_hash)
            self.num_freeable_slots -= num_encoder_embeds

        self.cached[mm_hash].add(request.request_id)
        self.cache_hits += 1
        self._maybe_log_trace(request, mm_hash, hit=True)
        return True

    def _maybe_log_trace(
        self, request: Request, mm_hash: str, hit: bool,
    ) -> None:
        """Emit one INFO line per cache check when VLLM_ENCODER_CACHE_TRACE
        is enabled. Benchmarks parse these to reconstruct per-request
        hit/miss curves. Format is stable; do not change without updating
        parse_encoder_trace.py.
        """
        # Lazy-imported to avoid circular import overhead in hot path.
        import vllm.envs as envs
        if not envs.VLLM_ENCODER_CACHE_TRACE:
            return
        logger.info(
            "EncoderCacheTrace req_id=%s mm_hash=%s hit=%d "
            "cum_hits=%d cum_misses=%d",
            request.request_id, mm_hash, int(hit),
            self.cache_hits, self.cache_misses,
        )

    def _maybe_log_occupancy(self, event: str) -> None:
        """Emit one INFO line whenever pinned/freeable/free occupancy
        changes (controlled by VLLM_ENCODER_CACHE_TRACE).

        The three categories are derived from the manager's internal
        counters:
            pinned   = cache_size - num_freeable_slots
                       (held by in-flight requests, cannot evict)
            freeable = num_freeable_slots - num_free_slots
                       (no live ref but still resident; evictable)
            free     = num_free_slots
                       (physically empty)

        `event` is one of "allocate" | "free" | "evict" so we can see
        which transition produced this snapshot.
        """
        import vllm.envs as envs
        if not envs.VLLM_ENCODER_CACHE_TRACE:
            return
        # `_occupancy_t0` is set on first call; later timestamps are
        # relative to it so the log is timezone- and clock-skew-free.
        import time
        t0 = getattr(self, "_occupancy_t0", None)
        if t0 is None:
            t0 = time.monotonic()
            self._occupancy_t0 = t0
        t_rel = time.monotonic() - t0
        pinned = self.cache_size - self.num_freeable_slots
        freeable_only = self.num_freeable_slots - self.num_free_slots
        free = self.num_free_slots
        logger.info(
            "EncoderCacheOccupancy event=%s t=%.6f cache_size=%d "
            "pinned=%d freeable=%d free=%d num_pinned_entries=%d "
            "num_freeable_entries=%d",
            event, t_rel, self.cache_size,
            pinned, freeable_only, free,
            sum(1 for v in self.cached.values() if v),
            len(self.freeable),
        )

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
        Older entries are evicted first.

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

        # Not enough free slots but enough reclaimable slots
        # NOTE: Eviction takes place here, but physical memory is not freed
        # until model runner is notified by the scheduler output.
        evicted_any = False
        while num_embeds > self.num_free_slots:
            mm_hash, num_free_embeds = self.freeable.popitem(last=False)
            del self.cached[mm_hash]
            self.freed.append(mm_hash)
            self.num_free_slots += num_free_embeds
            evicted_any = True
        if evicted_any:
            self._maybe_log_occupancy("evict")
        return True

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
        self._maybe_log_occupancy("allocate")

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
        increased by the number of encoder embeddings for that input.

        The entry is NOT physically freed until capacity is needed (e.g., by
        `can_allocate`).
        """
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        # The mm_hash not in cache or the req_id set is empty
        if not self.cached.get(mm_hash, None):
            return
        self.cached[mm_hash].discard(req_id)
        if not self.cached[mm_hash]:
            num_encoder_embeds = request.get_num_encoder_embeds(input_id)
            self.freeable[mm_hash] = num_encoder_embeds
            self.num_freeable_slots += num_encoder_embeds
            # Pinned shrunk; freeable grew. Log so plot_occupancy.py
            # can render the over-time stacked-area chart.
            self._maybe_log_occupancy("free")

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


@dataclass
class TypeMetadata:
    """Metadata for a data type used in distribution-aware cache management.

    Attributes:
        type_id: Unique identifier for this data type.
        p_i: Access probability of this data type.
        m_i: Memory cost in number of encoder embeddings.
        c_i: Computation cost (encoder time in seconds).
    """
    type_id: str
    p_i: float
    m_i: int
    c_i: float


class DistributionAwareCacheManager(EncoderCacheManager):
    """Distribution-aware encoder cache manager.

    Instead of FIFO eviction, this manager uses knowledge of the access
    distribution to decide which cache entries to keep vs evict.

    The algorithm solves for the optimal dual variable lambda* that maximizes:
        D(lambda) = (M - B) * lambda
                    - sum_i p_i * E[(m_i * d_i * lambda - c_i)^+]
    where:
        M = sum of all m_i
        B = cache size (memory budget)
        d_i ~ Geometric(p_i), i.e. P[d_i = d] = p_i * (1-p_i)^d

    When an entry becomes unreferenced, we sample its next arrival time d
    from the geometric distribution and check if m_i * lambda* * d - c_i >= 0.
    If so, the entry is marked evictable (cost of keeping > cost of recomputing).
    Otherwise, it is marked non-evictable (should be kept in cache).

    When eviction is needed, evictable entries are evicted first (FIFO among
    evictables). If not enough, non-evictable entries are evicted as fallback.
    """

    def __init__(self, cache_size: int):
        super().__init__(cache_size)
        # Distribution configuration
        self.type_metadata: dict[str, TypeMetadata] = {}
        self.hash_to_type: dict[str, str] = {}
        self.lambda_star: float = 0.0
        self._configured = False

        # Per-entry evictability: mm_hash -> is_evictable
        self.evictability: dict[str, bool] = {}

        # cache_hits / cache_misses are inherited from the base class.

        self._rng = random.Random(42)

    def reset(self) -> None:
        super().reset()
        self.evictability.clear()
        # Counters are reset by the base class.

    def configure_distribution(
        self,
        type_metadata: dict[str, TypeMetadata],
        hash_to_type: dict[str, str],
    ) -> None:
        """Configure the distribution parameters and solve for lambda*.

        Args:
            type_metadata: Mapping from type_id to TypeMetadata.
            hash_to_type: Mapping from mm_hash to type_id.
        """
        self.type_metadata = type_metadata
        self.hash_to_type = hash_to_type
        self.lambda_star = self._solve_lambda()
        self._configured = True
        # High-level summary
        M = sum(meta.m_i for meta in type_metadata.values())
        logger.info(
            "Distribution-aware cache configured: lambda*=%.6f, "
            "cache_size=%d, sum_m_i=%d (fits=%s), "
            "%d types, %d hash mappings",
            self.lambda_star, self.cache_size, M,
            "yes" if M <= self.cache_size else "no",
            len(type_metadata), len(hash_to_type),
        )
        # Per-type detail. The "keep_threshold_d" is the smallest geom
        # sample `d` at which evictability flips from False to True
        # (cost >= 0), i.e. how many "waits" before this type's entry
        # becomes safe to evict. Lower = evicted sooner.
        # E[d] = (1-p)/p, so we also print that as a workload-only
        # baseline.
        for tid, meta in type_metadata.items():
            if self.lambda_star * meta.m_i > 0:
                d0 = max(0, int(math.ceil(
                    meta.c_i / (meta.m_i * self.lambda_star))))
            else:
                d0 = -1  # never evict
            e_d = ((1 - meta.p_i) / meta.p_i) if meta.p_i > 0 else float("inf")
            logger.info(
                "DistAwareTypeConfig type=%s p_i=%.4f m_i=%d c_i=%.4f "
                "E[d]=%.1f keep_threshold_d=%d",
                tid, meta.p_i, meta.m_i, meta.c_i, e_d, d0,
            )

    @classmethod
    def from_config_file(cls, cache_size: int,
                         config_path: str) -> "DistributionAwareCacheManager":
        """Create a DistributionAwareCacheManager from a JSON config file.

        Expected JSON format:
        {
            "types": {
                "type_0": {"p_i": 0.3, "m_i": 576, "c_i": 0.05},
                ...
            },
            "hash_to_type": {
                "abc123...": "type_0",
                ...
            }
        }
        """
        manager = cls(cache_size)
        with open(config_path) as f:
            config = json.load(f)

        type_metadata = {}
        for type_id, meta in config["types"].items():
            type_metadata[type_id] = TypeMetadata(
                type_id=type_id,
                p_i=meta["p_i"],
                m_i=meta["m_i"],
                c_i=meta["c_i"],
            )

        hash_to_type = config.get("hash_to_type", {})
        manager.configure_distribution(type_metadata, hash_to_type)
        return manager

    def _expected_positive_part(self, m_i: int, c_i: float, p_i: float,
                                lam: float) -> float:
        """Compute E[(m_i * d * lam - c_i)^+] where d ~ Geometric(p_i).

        P[d = d] = p_i * (1-p_i)^d for d = 0, 1, 2, ...

        E[(m*d*lam - c)^+] = sum_{d >= d0} p*(1-p)^d * (m*d*lam - c)
        where d0 = ceil(c / (m*lam)) if m*lam > 0, else infinity.

        Using geometric tail sums:
          sum_{d=d0}^inf q^d = q^d0 / (1-q) = q^d0 / p
          sum_{d=d0}^inf d*q^d = d0*q^d0/(1-q) + q^(d0+1)/(1-q)^2
        """
        if lam <= 0 or m_i * lam <= 0:
            return 0.0

        d0 = max(0, math.ceil(c_i / (m_i * lam)))
        q = 1 - p_i

        if q <= 0 or q >= 1:
            # Degenerate: p_i=1 means d is always 0
            if p_i >= 1.0:
                val = m_i * 0 * lam - c_i
                return max(0.0, val) * 1.0
            return 0.0

        q_d0 = q ** d0

        # sum_{d=d0}^inf p*q^d = p * q^d0 / (1-q) = q^d0
        tail_prob = q_d0  # = p * q^d0 / p

        # sum_{d=d0}^inf d * p * q^d
        # = p * [d0 * q^d0 / (1-q) + q^(d0+1) / (1-q)^2]
        # = d0 * q^d0 + q^(d0+1) / (1-q)
        # = d0 * q^d0 + q^(d0+1) / p
        tail_d_weighted = d0 * q_d0 + q ** (d0 + 1) / p_i

        # E[(m*d*lam - c)^+] = m*lam * tail_d_weighted - c * tail_prob
        result = m_i * lam * tail_d_weighted - c_i * tail_prob
        return max(0.0, result)

    def _dual_function(self, lam: float) -> float:
        """Compute D(lambda) = (M - B)*lambda - sum_i p_i*E[(m_i*d_i*lam - c_i)^+]."""
        M = sum(meta.m_i for meta in self.type_metadata.values())
        B = self.cache_size

        val = (M - B) * lam
        for meta in self.type_metadata.values():
            val -= meta.p_i * self._expected_positive_part(
                meta.m_i, meta.c_i, meta.p_i, lam
            )
        return val

    def _solve_lambda(self) -> float:
        """Solve for lambda* that maximizes D(lambda) using scipy."""
        if not self.type_metadata:
            return 0.0

        try:
            from scipy.optimize import minimize_scalar
        except ImportError:
            logger.warning(
                "scipy not available, falling back to grid search for lambda*"
            )
            return self._solve_lambda_grid()

        result = minimize_scalar(
            lambda l: -self._dual_function(l),
            bounds=(0, 10000),
            method="bounded",
        )
        return result.x

    def _solve_lambda_grid(self) -> float:
        """Fallback grid search for lambda* when scipy is unavailable."""
        best_lam = 0.0
        best_val = self._dual_function(0.0)
        for exp in range(-6, 5):
            for mantissa in [1, 2, 5]:
                lam = mantissa * (10.0 ** exp)
                val = self._dual_function(lam)
                if val > best_val:
                    best_val = val
                    best_lam = lam
        # Refine around best
        for delta in [x * best_lam * 0.01 for x in range(-50, 51)]:
            lam = best_lam + delta
            if lam <= 0:
                continue
            val = self._dual_function(lam)
            if val > best_val:
                best_val = val
                best_lam = lam
        return best_lam

    def _sample_geometric(self, p_i: float) -> int:
        """Sample from geometric distribution P[d=d] = p_i * (1-p_i)^d."""
        if p_i >= 1.0:
            return 0
        if p_i <= 0.0:
            return 0
        # Use inverse CDF: d = floor(log(U) / log(1-p_i))
        u = self._rng.random()
        if u == 0:
            return 0
        return int(math.log(u) / math.log(1 - p_i))

    def _compute_evictability(self, mm_hash: str,
                              num_encoder_embeds: int) -> bool:
        """Determine if an entry should be marked as evictable.

        Returns True if the entry is evictable (can be removed when needed).
        """
        if not self._configured:
            return True  # Fall back to FIFO behavior

        type_id = self.hash_to_type.get(mm_hash)
        if type_id is None or type_id not in self.type_metadata:
            return True  # Unknown type, treat as evictable

        meta = self.type_metadata[type_id]
        d = self._sample_geometric(meta.p_i)
        cost = meta.m_i * self.lambda_star * d - meta.c_i
        return cost >= 0  # evictable if keeping cost >= recompute cost

    # check_and_update_cache: inherited from base class which now maintains
    # cache_hits / cache_misses for all manager flavors.

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        mm_hash = request.mm_features[input_id].identifier
        was_referenced = bool(self.cached.get(mm_hash))

        super().free_encoder_input(request, input_id)

        # If entry just moved to freeable, compute evictability
        if was_referenced and mm_hash in self.freeable:
            num_encoder_embeds = request.get_num_encoder_embeds(input_id)
            is_evictable = self._compute_evictability(
                mm_hash, num_encoder_embeds
            )
            self.evictability[mm_hash] = is_evictable

    def can_allocate(
        self,
        request: Request,
        input_id: int,
        encoder_compute_budget: int,
        num_embeds_to_schedule: int,
    ) -> bool:
        num_embeds = request.get_num_encoder_embeds(input_id)

        if num_embeds > encoder_compute_budget:
            return False

        num_embeds += num_embeds_to_schedule

        if num_embeds <= self.num_free_slots:
            return True

        if num_embeds > self.num_freeable_slots:
            return False

        # Phase 1: evict entries marked as evictable first
        evictable_hashes = [
            h for h in self.freeable
            if self.evictability.get(h, True)
        ]
        evicted_any = False
        for mm_hash in evictable_hashes:
            if num_embeds <= self.num_free_slots:
                break
            num_free_embeds = self.freeable.pop(mm_hash)
            del self.cached[mm_hash]
            self.evictability.pop(mm_hash, None)
            self.freed.append(mm_hash)
            self.num_free_slots += num_free_embeds
            evicted_any = True

        # Phase 2: if still not enough, fall back to FIFO on non-evictables
        while num_embeds > self.num_free_slots:
            mm_hash, num_free_embeds = self.freeable.popitem(last=False)
            del self.cached[mm_hash]
            self.evictability.pop(mm_hash, None)
            self.freed.append(mm_hash)
            self.num_free_slots += num_free_embeds
            evicted_any = True

        if evicted_any:
            self._maybe_log_occupancy("evict")
        return True

    def get_hit_rate(self) -> float:
        """Return the cache hit rate."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def get_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.get_hit_rate(),
            "lambda_star": self.lambda_star,
            "configured": self._configured,
            "num_types": len(self.type_metadata),
            "evictable_count": sum(
                1 for v in self.evictability.values() if v
            ),
            "non_evictable_count": sum(
                1 for v in self.evictability.values() if not v
            ),
        }


class NoCacheEncoderManager(EncoderCacheManager):
    """Encoder cache manager that disables cross-request caching.

    Used as a baseline for benchmarking. Entries are physically removed
    from the cache as soon as no request references them, so a follow-up
    request for the same multimodal input always misses and recomputes.

    Concurrent requests for the same input within a single scheduler step
    can still share the in-flight encoder output (this is unavoidable
    without major scheduler changes), but back-to-back requests separated
    in time will always recompute.
    """

    # __init__: inherited; cache_hits / cache_misses are set up in the base
    # class. reset() is also inherited and now logs stats before clearing.
    # check_and_update_cache: inherited; the base class maintains counters.

    def free_encoder_input(self, request: Request, input_id: int) -> None:
        req_id = request.request_id
        mm_hash = request.mm_features[input_id].identifier
        if not self.cached.get(mm_hash, None):
            return
        self.cached[mm_hash].discard(req_id)
        if not self.cached[mm_hash]:
            # Immediately physically free instead of keeping in `freeable`.
            num_encoder_embeds = request.get_num_encoder_embeds(input_id)
            del self.cached[mm_hash]
            self.num_free_slots += num_encoder_embeds
            self.num_freeable_slots += num_encoder_embeds
            self.freed.append(mm_hash)

    def get_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def get_stats(self) -> dict:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.get_hit_rate(),
            "policy": "none",
        }


def create_encoder_cache_manager(cache_size: int) -> EncoderCacheManager:
    """Factory that selects an encoder cache manager from env vars.

    Reads `VLLM_ENCODER_CACHE_POLICY` (one of "fifo", "none",
    "distribution_aware") and, when applicable,
    `VLLM_ENCODER_CACHE_CONFIG_PATH` for the distribution config.
    """
    import vllm.envs as envs

    policy = (envs.VLLM_ENCODER_CACHE_POLICY or "fifo").lower()

    if policy == "none":
        logger.info("Encoder cache policy: none (caching disabled)")
        return NoCacheEncoderManager(cache_size=cache_size)

    if policy == "distribution_aware":
        config_path = envs.VLLM_ENCODER_CACHE_CONFIG_PATH
        if not config_path:
            logger.warning(
                "VLLM_ENCODER_CACHE_POLICY=distribution_aware but "
                "VLLM_ENCODER_CACHE_CONFIG_PATH not set; falling back to FIFO."
            )
            return EncoderCacheManager(cache_size=cache_size)
        logger.info(
            "Encoder cache policy: distribution_aware (config=%s)",
            config_path,
        )
        return DistributionAwareCacheManager.from_config_file(
            cache_size=cache_size, config_path=config_path
        )

    if policy != "fifo":
        logger.warning(
            "Unknown VLLM_ENCODER_CACHE_POLICY=%s; falling back to FIFO.",
            policy,
        )
    return EncoderCacheManager(cache_size=cache_size)


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
    def __init__(self, cache_size: int):
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
