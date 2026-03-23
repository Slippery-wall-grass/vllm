# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.encoder_cache_manager import (
    DistributionAwareCacheManager,
    EncoderCacheManager,
    EncoderDecoderCacheManager,
    TypeMetadata,
)

pytestmark = pytest.mark.cpu_test


# ------------------ Mock Classes ------------------ #
class MockRequest:
    def __init__(self, request_id, mm_hashes, token_counts):
        self.request_id = request_id
        self._token_counts = token_counts
        self.mm_features = []
        for i, mm_hash in enumerate(mm_hashes):
            feature = MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=mm_hash,
                mm_position=PlaceholderRange(offset=0, length=self._token_counts[i]),
            )
            self.mm_features.append(feature)

    def get_num_encoder_embeds(self, input_id: int) -> int:
        return self._token_counts[input_id]


# ------------------ Unit Tests ------------------ #
def test_basic_allocate_and_reuse():
    cache = EncoderCacheManager(cache_size=10)
    req = MockRequest("r1", ["imgA"], [4])

    assert not cache.check_and_update_cache(req, 0)
    assert cache.can_allocate(req, 0, int(1e9), 0)

    cache.allocate(req, 0)

    assert cache.check_and_update_cache(req, 0)
    assert "r1" in cache.cached["imgA"]
    assert cache.num_free_slots == 6

    # Free twice to bring refcount to 0.
    cache.free_encoder_input(req, 0)
    cache.free_encoder_input(req, 0)

    assert not cache.cached["imgA"]
    assert "imgA" in cache.freeable
    assert cache.num_freeable_slots == 10
    assert cache.num_free_slots == 6


def test_freeing_decreases_refcount_and_moves_to_freeable():
    manager = EncoderCacheManager(cache_size=10)
    req = MockRequest("req2", ["img3"], [5])

    assert manager.can_allocate(req, 0, int(1e9), 0)
    manager.allocate(req, 0)

    assert len(manager.cached["img3"]) == 1

    manager.free_encoder_input(req, 0)

    assert not manager.cached["img3"]
    assert "img3" in manager.freeable
    assert manager.num_freeable_slots == 10


def test_free_request_frees_all_inputs():
    manager = EncoderCacheManager(cache_size=10)
    req = MockRequest("req3", ["a", "b"], [2, 3])

    assert manager.can_allocate(req, 0, int(1e9), 0)
    manager.allocate(req, 0)

    assert manager.can_allocate(req, 1, int(1e9), 0)
    manager.allocate(req, 1)

    assert len(manager.cached["a"]) == 1
    assert len(manager.cached["b"]) == 1

    manager.free(req)

    assert not manager.cached["a"]
    assert not manager.cached["b"]
    assert "a" in manager.freeable
    assert "b" in manager.freeable
    assert manager.num_freeable_slots == 10


def test_eviction_when_cache_is_full():
    manager = EncoderCacheManager(cache_size=10)

    req1 = MockRequest("req1", ["x"], [6])
    req2 = MockRequest("req2", ["y"], [5])

    assert manager.can_allocate(req1, 0, int(1e9), 0)
    manager.allocate(req1, 0)
    manager.free_encoder_input(req1, 0)

    assert manager.can_allocate(req2, 0, int(1e9), 0)
    manager.allocate(req2, 0)

    # 'x' should have been evicted.
    assert "x" not in manager.cached
    assert "x" in manager.get_freed_mm_hashes()


def test_get_cached_input_ids():
    manager = EncoderCacheManager(cache_size=10)
    req = MockRequest("reqX", ["m", "n", "o"], [2, 4, 3])

    assert manager.can_allocate(req, 0, int(1e9), 0)
    manager.allocate(req, 0)

    assert manager.can_allocate(req, 2, int(1e9), 0)
    manager.allocate(req, 2)

    cached_ids = manager.get_cached_input_ids(req)
    assert cached_ids == {0, 2}


def test_has_cache_restores_from_freeable():
    manager = EncoderCacheManager(cache_size=10)
    req = MockRequest("reqY", ["imgZ"], [4])

    assert manager.can_allocate(req, 0, int(1e9), 0)
    manager.allocate(req, 0)

    manager.free_encoder_input(req, 0)

    # Should restore from freeable.
    assert manager.check_and_update_cache(req, 0)
    assert len(manager.cached["imgZ"]) == 1
    assert "imgZ" not in manager.freeable
    assert manager.num_freeable_slots == 6


def test_get_freed_mm_hashes_clears_freed_list():
    manager = EncoderCacheManager(cache_size=10)
    req1 = MockRequest("reqA", ["a"], [5])
    req2 = MockRequest("reqB", ["b"], [6])

    assert manager.can_allocate(req1, 0, int(1e9), 0)
    manager.allocate(req1, 0)
    manager.free_encoder_input(req1, 0)

    # Should trigger eviction of 'a'.
    assert manager.can_allocate(req2, 0, int(1e9), 0)
    manager.allocate(req2, 0)

    freed = manager.get_freed_mm_hashes()
    assert "a" in freed
    assert manager.get_freed_mm_hashes() == []


def test_schedule_request_multi_images_respect_space_limit():
    manager = EncoderCacheManager(cache_size=10)
    req = MockRequest("reqA", ["a", "b"], [5, 6])
    compute_budget = 100

    num_tokens_to_schedule = 0
    assert manager.can_allocate(req, 0, compute_budget, num_tokens_to_schedule)
    num_tokens_to_schedule += req.get_num_encoder_embeds(0)
    compute_budget -= req.get_num_encoder_embeds(0)

    assert not manager.can_allocate(req, 1, compute_budget, num_tokens_to_schedule)


def test_schedule_request_multi_images_respect_compute_limit():
    manager = EncoderCacheManager(cache_size=100)
    req = MockRequest("reqA", ["a", "b"], [5, 6])
    compute_budget = 10
    num_tokens_to_schedule = 0
    assert manager.can_allocate(req, 0, compute_budget, num_tokens_to_schedule)
    num_tokens_to_schedule += req.get_num_encoder_embeds(0)
    compute_budget -= req.get_num_encoder_embeds(0)

    assert not manager.can_allocate(req, 1, compute_budget, num_tokens_to_schedule)


def test_encoder_cache_with_is_embed_mask():
    class MockRequestWithMask(MockRequest):
        def get_num_encoder_embeds(self, input_id: int) -> int:
            return self.mm_features[input_id].mm_position.get_num_embeds()

    is_embed = torch.zeros(100, dtype=torch.bool)
    is_embed[torch.tensor([5, 15, 25, 35, 45, 55, 65, 75])] = True

    request = MockRequestWithMask("r1", ["img1"], [100])
    request.mm_features[0] = MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier="img1",
        mm_position=PlaceholderRange(offset=0, length=100, is_embed=is_embed),
    )

    manager = EncoderCacheManager(cache_size=100)
    manager.allocate(request, 0)

    assert manager.num_free_slots == 92
    assert "img1" in manager.cached

    old_size = 100
    new_size = request.mm_features[0].mm_position.get_num_embeds()
    assert new_size == 8
    savings_ratio = old_size / new_size
    assert savings_ratio == 12.5


def test_encoder_cache_mask_based_retrieval():
    class MockRequestWithMask(MockRequest):
        def get_num_encoder_embeds(self, input_id: int) -> int:
            return self.mm_features[input_id].mm_position.get_num_embeds()

    is_embed = torch.tensor(
        [False, False, True, True, False, True, True, True, False, False]
    )

    request = MockRequestWithMask("r1", ["img1"], [10])
    request.mm_features[0] = MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier="img1",
        mm_position=PlaceholderRange(offset=0, length=10, is_embed=is_embed),
    )

    manager = EncoderCacheManager(cache_size=50)
    manager.allocate(request, 0)

    assert request.mm_features[0].mm_position.get_num_embeds() == 5

    start_idx = 2
    end_idx = 8
    num_embeds_before = is_embed[:start_idx].sum().item()
    num_embeds_in_range = is_embed[start_idx:end_idx].sum().item()

    assert num_embeds_before == 0
    assert num_embeds_in_range == 5

    start_idx = 0
    end_idx = 5
    num_embeds_before = is_embed[:start_idx].sum().item() if start_idx > 0 else 0
    num_embeds_in_range = is_embed[start_idx:end_idx].sum().item()

    assert num_embeds_before == 0
    assert num_embeds_in_range == 2


def test_reset_clears_all_state():
    """Test that reset() clears all cached entries and restores capacity."""
    manager = EncoderCacheManager(cache_size=20)

    req1 = MockRequest("req1", ["img1", "img2"], [5, 3])
    req2 = MockRequest("req2", ["img3"], [4])

    manager.allocate(req1, 0)
    manager.allocate(req1, 1)
    manager.allocate(req2, 0)
    manager.free_encoder_input(req1, 0)

    req3 = MockRequest("req3", ["img4"], [10])
    manager.free_encoder_input(req1, 1)
    manager.free_encoder_input(req2, 0)
    manager.can_allocate(req3, 0, int(1e9), 0)
    manager.allocate(req3, 0)

    assert len(manager.cached) > 0
    assert manager.num_free_slots < 20

    manager.reset()

    assert len(manager.cached) == 0
    assert len(manager.freeable) == 0
    assert len(manager.freed) == 0
    assert manager.num_free_slots == 20
    assert manager.num_freeable_slots == 20


def test_reset_allows_fresh_allocations():
    manager = EncoderCacheManager(cache_size=10)

    req1 = MockRequest("req1", ["img1"], [10])
    manager.allocate(req1, 0)
    assert manager.num_free_slots == 0

    manager.reset()

    req2 = MockRequest("req2", ["img2"], [8])
    assert manager.can_allocate(req2, 0, int(1e9), 0)
    manager.allocate(req2, 0)

    assert manager.num_free_slots == 2
    assert "img2" in manager.cached
    assert "img1" not in manager.cached


def test_encoder_decoder_cache_manager_reset():
    manager = EncoderDecoderCacheManager(cache_size=20)

    req1 = MockRequest("req1", ["img1"], [5])
    req2 = MockRequest("req2", ["img2"], [3])

    manager.allocate(req1, 0)
    manager.allocate(req2, 0)
    manager.free(req1)
    manager.get_freed_mm_hashes()

    assert manager.num_free_slots < 20

    manager.reset()

    assert len(manager.allocated) == 0
    assert len(manager.to_free) == 0
    assert manager.num_free_slots == 20


def test_encoder_decoder_cache_manager_reset_allows_fresh_allocations():
    manager = EncoderDecoderCacheManager(cache_size=10)

    req1 = MockRequest("req1", ["img1"], [10])
    manager.allocate(req1, 0)
    assert manager.num_free_slots == 0

    manager.reset()

    req2 = MockRequest("req2", ["img2"], [8])
    assert manager.can_allocate(req2, 0, int(1e9), 0)
    manager.allocate(req2, 0)

    assert manager.num_free_slots == 2
    assert "img2" in manager.allocated


# ---------- DistributionAwareCacheManager Tests ---------- #

def _make_distribution_aware_cache(cache_size, types_config, hash_to_type):
    """Helper to create a configured DistributionAwareCacheManager."""
    manager = DistributionAwareCacheManager(cache_size)
    type_metadata = {}
    for type_id, cfg in types_config.items():
        type_metadata[type_id] = TypeMetadata(
            type_id=type_id,
            p_i=cfg["p_i"],
            m_i=cfg["m_i"],
            c_i=cfg["c_i"],
        )
    manager.configure_distribution(type_metadata, hash_to_type)
    return manager


def test_distribution_aware_no_config_falls_back_to_fifo():
    """Without configuration, behaves like FIFO."""
    manager = DistributionAwareCacheManager(cache_size=10)
    req1 = MockRequest("r1", ["imgA"], [6])
    req2 = MockRequest("r2", ["imgB"], [5])

    assert manager.can_allocate(req1, 0, int(1e9), 0)
    manager.allocate(req1, 0)
    manager.free_encoder_input(req1, 0)

    assert manager.can_allocate(req2, 0, int(1e9), 0)
    manager.allocate(req2, 0)

    assert "imgA" not in manager.cached
    assert "imgA" in manager.get_freed_mm_hashes()


def test_distribution_aware_prefers_evictable():
    """Evictable entries are evicted before non-evictable ones."""
    # Setup: high p_i for type_a (frequent access => non-evictable)
    # low p_i for type_b (rare access => evictable)
    types_config = {
        "type_a": {"p_i": 0.9, "m_i": 5, "c_i": 10.0},  # high c, frequent
        "type_b": {"p_i": 0.01, "m_i": 5, "c_i": 0.001},  # low c, rare
    }
    hash_to_type = {"imgA": "type_a", "imgB": "type_b"}

    manager = _make_distribution_aware_cache(
        cache_size=15, types_config=types_config, hash_to_type=hash_to_type
    )

    req1 = MockRequest("r1", ["imgA"], [5])
    req2 = MockRequest("r2", ["imgB"], [5])
    req3 = MockRequest("r3", ["imgC"], [6])

    # Allocate and free both
    manager.allocate(req1, 0)
    manager.allocate(req2, 0)
    manager.free_encoder_input(req1, 0)
    manager.free_encoder_input(req2, 0)

    # Both are now in freeable. imgB (rare, low cost) should be evictable,
    # imgA (frequent, high cost) should be non-evictable.
    # When we need space for imgC, imgB should be evicted first.
    assert manager.can_allocate(req3, 0, int(1e9), 0)
    manager.allocate(req3, 0)

    freed = manager.get_freed_mm_hashes()
    # imgB (evictable) should be evicted, imgA (non-evictable) should remain
    assert "imgB" in freed


def test_distribution_aware_fallback_to_fifo():
    """When no evictable entries, falls back to FIFO."""
    # All types have very high computation cost => all non-evictable
    types_config = {
        "type_a": {"p_i": 0.5, "m_i": 5, "c_i": 1000.0},
        "type_b": {"p_i": 0.5, "m_i": 5, "c_i": 1000.0},
    }
    hash_to_type = {"imgA": "type_a", "imgB": "type_b"}

    manager = _make_distribution_aware_cache(
        cache_size=10, types_config=types_config, hash_to_type=hash_to_type
    )

    req1 = MockRequest("r1", ["imgA"], [5])
    req2 = MockRequest("r2", ["imgB"], [5])
    req3 = MockRequest("r3", ["imgC"], [6])

    manager.allocate(req1, 0)
    manager.allocate(req2, 0)
    manager.free_encoder_input(req1, 0)
    manager.free_encoder_input(req2, 0)

    # Both non-evictable, but we still need space => fallback to FIFO
    assert manager.can_allocate(req3, 0, int(1e9), 0)
    manager.allocate(req3, 0)

    freed = manager.get_freed_mm_hashes()
    # At least one should be evicted to make space
    assert len(freed) >= 1


def test_distribution_aware_hit_rate_tracking():
    """Cache hit/miss counters work correctly."""
    manager = DistributionAwareCacheManager(cache_size=20)

    req = MockRequest("r1", ["imgA"], [5])
    # Miss
    assert not manager.check_and_update_cache(req, 0)
    assert manager.cache_misses == 1
    assert manager.cache_hits == 0

    manager.allocate(req, 0)
    # Hit
    assert manager.check_and_update_cache(req, 0)
    assert manager.cache_hits == 1
    assert manager.cache_misses == 1
    assert manager.get_hit_rate() == 0.5


def test_lambda_solver():
    """Verify the lambda solver produces a reasonable result."""
    manager = DistributionAwareCacheManager(cache_size=10)
    types_config = {
        "type_a": {"p_i": 0.3, "m_i": 5, "c_i": 0.05},
        "type_b": {"p_i": 0.7, "m_i": 5, "c_i": 0.02},
    }
    type_metadata = {
        tid: TypeMetadata(type_id=tid, **cfg)
        for tid, cfg in types_config.items()
    }
    manager.configure_distribution(type_metadata, {})

    # lambda* should be non-negative
    assert manager.lambda_star >= 0
    assert manager._configured


def test_distribution_aware_reset():
    """Reset clears distribution-aware state too."""
    manager = DistributionAwareCacheManager(cache_size=10)
    req = MockRequest("r1", ["imgA"], [5])

    manager.allocate(req, 0)
    manager.check_and_update_cache(req, 0)

    manager.reset()

    assert manager.cache_hits == 0
    assert manager.cache_misses == 0
    assert len(manager.evictability) == 0
    assert manager.num_free_slots == 10


def test_distribution_aware_get_stats():
    """get_stats returns expected structure."""
    manager = DistributionAwareCacheManager(cache_size=10)
    stats = manager.get_stats()
    assert "cache_hits" in stats
    assert "cache_misses" in stats
    assert "hit_rate" in stats
    assert "lambda_star" in stats
    assert stats["configured"] is False
