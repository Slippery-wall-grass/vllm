# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the pluggable encoder-cache eviction policy.

Covers three concerns end-to-end on top of :class:`EncoderCacheManager`:

1. Closed-form / numerical correctness of :func:`solve_lambda_star` and
   :func:`expected_positive_part` (Lagrangian dual math).
2. The pin -> unlock state machine: pinned entries are skipped during
   eviction until their unlock tick passes, and ``forced_unpin_evictions``
   ticks up only when every freeable entry is pinned.
3. The ``no-cache`` ablation drops entries on the moment their reference
   count reaches zero.
"""

import math
import random

import pytest

from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.v1.core.encoder_cache_lambda import (
    TypeStats,
    dual_value,
    expected_positive_part,
    solve_lambda_star,
)
from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
from vllm.v1.core.encoder_cache_policy import (
    EncoderCachePolicy,
    FIFOEncoderCachePolicy,
    NoCacheEncoderCachePolicy,
    OfflineLagrangianEncoderCachePolicy,
    _PoolEntry,
    build_policy_from_env,
)

pytestmark = pytest.mark.cpu_test


class MockRequest:
    """Minimal duck-typed request for cache-manager tests."""

    def __init__(self, request_id: str, mm_hashes: list[str], token_counts: list[int]):
        self.request_id = request_id
        self._token_counts = token_counts
        self.mm_features = [
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=h,
                mm_position=PlaceholderRange(offset=0, length=token_counts[i]),
            )
            for i, h in enumerate(mm_hashes)
        ]

    def get_num_encoder_embeds(self, input_id: int) -> int:
        return self._token_counts[input_id]


# ---------------------------------------------------------------------------
# Closed form: monte-carlo cross-check
# ---------------------------------------------------------------------------


def _mc_expected_positive_part(p, m, c, lam, n=200_000, seed=0):
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n):
        u = max(rng.random(), 1e-300)
        d = int(math.floor(math.log(u) / math.log(1.0 - p)))
        total += max(0.0, m * lam * d - c)
    return total / n


@pytest.mark.parametrize(
    "p,m,c,lam",
    [
        (0.5, 100.0, 0.01, 1e-4),
        (0.1, 200.0, 0.05, 5e-4),
        (0.02, 500.0, 0.1, 1e-3),
        (0.3, 50.0, 0.02, 0.0),  # trivial
    ],
)
def test_expected_positive_part_matches_monte_carlo(p, m, c, lam):
    closed = expected_positive_part(p, m, c, lam)
    if lam == 0.0:
        assert closed == 0.0
        return
    mc = _mc_expected_positive_part(p, m, c, lam)
    # Loose tolerance — MC has noise.
    assert math.isfinite(closed)
    assert abs(closed - mc) < 0.05 * max(abs(mc), 1e-6) + 1e-4


def test_solve_lambda_star_under_pressure():
    # Pressure: sum m_i = 600, capacity = 200.
    types = [
        TypeStats(p=0.5, m=100.0, c=0.05),
        TypeStats(p=0.3, m=200.0, c=0.10),
        TypeStats(p=0.2, m=300.0, c=0.20),
    ]
    lam, d_opt = solve_lambda_star(types, cache_capacity=200.0)
    # Sanity: lambda* must be strictly positive when pressure exists.
    assert lam > 0.0
    # Optimality: nearby points should not beat lam.
    for delta in (-0.3, -0.1, 0.1, 0.3):
        nearby = max(1e-12, lam * (1.0 + delta))
        assert dual_value(types, 200.0, nearby) <= d_opt + 1e-6


def test_solve_lambda_star_no_pressure_returns_zero():
    # sum m_i = 60, capacity = 1000 -> no constraint, lambda*=0.
    types = [TypeStats(p=0.5, m=20.0, c=0.05), TypeStats(p=0.5, m=40.0, c=0.10)]
    lam, _ = solve_lambda_star(types, cache_capacity=1000.0)
    assert lam == 0.0


# ---------------------------------------------------------------------------
# Pin state machine on the manager
# ---------------------------------------------------------------------------


class _AlwaysPinPolicy(EncoderCachePolicy):
    """Test double: pin every arrival for ``horizon`` ticks starting
    from the moment the entry enters the freeable queue."""

    def __init__(self, horizon: int):
        self.horizon = horizon

    def on_arrival(self, mm_hash, num_embeds, current_tick):
        return self.horizon  # ticks; applied at freeable transition


class _NeverPinPolicy(EncoderCachePolicy):
    def on_arrival(self, mm_hash, num_embeds, current_tick):
        return 0


def _alloc_and_release(mgr: EncoderCacheManager, req_id: str, mm_hash: str, n: int):
    req = MockRequest(req_id, [mm_hash], [n])
    assert mgr.can_allocate(req, 0, int(1e9), 0)
    mgr.allocate(req, 0)
    mgr.free_encoder_input(req, 0)
    return req


def test_pinned_entries_are_skipped_for_eviction():
    # Capacity 10. Insert one pinned 4-slot entry (A) then a non-pinned
    # 4-slot entry (B). A third 4-slot allocation should evict B first
    # because A is still pinned.
    mgr = EncoderCacheManager(cache_size=10, policy=_NeverPinPolicy())
    # Use the AlwaysPin policy only for A by toggling.
    pin_policy = _AlwaysPinPolicy(horizon=100)
    mgr.policy = pin_policy
    _alloc_and_release(mgr, "r1", "A", 4)

    mgr.policy = _NeverPinPolicy()
    _alloc_and_release(mgr, "r2", "B", 4)

    assert "A" in mgr.freeable
    assert "B" in mgr.freeable

    # Trigger eviction: allocate 4-slot C; should evict B (unpinned),
    # leaving A pinned.
    req_c = MockRequest("r3", ["C"], [4])
    assert mgr.can_allocate(req_c, 0, int(1e9), 0)
    mgr.allocate(req_c, 0)

    assert "A" in mgr.cached
    assert "B" not in mgr.cached
    assert mgr.forced_unpin_evictions == 0


def test_pin_expires_after_unlock_horizon():
    mgr = EncoderCacheManager(cache_size=8, policy=_AlwaysPinPolicy(horizon=3))
    _alloc_and_release(mgr, "r1", "A", 4)
    # Pin tick set at current_tick=1 (one allocate), unlock at tick 4.
    # current_tick advances by 1 per arrival (allocate or hit).
    # Drive ticks by allocating + immediately releasing an unrelated entry.
    mgr.policy = _NeverPinPolicy()
    for i in range(4):
        _alloc_and_release(mgr, f"r{i + 10}", f"X{i}", 1)
    # Now A's unlock tick has long since passed; it should be evictable.
    assert mgr._unlock_tick.get("A", 0) <= mgr.current_tick
    # Allocate a request that requires eviction of A.
    req = MockRequest("r99", ["NEW"], [8])
    assert mgr.can_allocate(req, 0, int(1e9), 0)
    mgr.allocate(req, 0)
    assert mgr.forced_unpin_evictions == 0
    assert "A" not in mgr.cached


def test_forced_unpin_eviction_when_all_pinned():
    mgr = EncoderCacheManager(cache_size=8, policy=_AlwaysPinPolicy(horizon=10_000))
    _alloc_and_release(mgr, "r1", "A", 4)
    _alloc_and_release(mgr, "r2", "B", 4)
    # Cache is full; both entries pinned with huge horizons.
    req = MockRequest("r3", ["C"], [4])
    assert mgr.can_allocate(req, 0, int(1e9), 0)
    mgr.allocate(req, 0)
    # Exactly one forced eviction (A is FIFO-front).
    assert mgr.forced_unpin_evictions == 1
    assert "A" not in mgr.cached
    assert "B" in mgr.cached
    assert "C" in mgr.cached


# ---------------------------------------------------------------------------
# No-cache policy
# ---------------------------------------------------------------------------


def test_nocache_policy_evicts_on_unreference():
    mgr = EncoderCacheManager(cache_size=8, policy=NoCacheEncoderCachePolicy())
    req = MockRequest("r1", ["A"], [4])
    assert mgr.can_allocate(req, 0, int(1e9), 0)
    mgr.allocate(req, 0)
    assert "A" in mgr.cached
    mgr.free_encoder_input(req, 0)
    # Dropped immediately, not parked in freeable.
    assert "A" not in mgr.cached
    assert "A" not in mgr.freeable
    assert mgr.num_free_slots == 8


# ---------------------------------------------------------------------------
# Offline policy: hash-to-type lookup
# ---------------------------------------------------------------------------


def test_offline_policy_unknown_hash_falls_back_to_no_pin():
    pol = OfflineLagrangianEncoderCachePolicy(
        pool={
            "known": _PoolEntry(type_idx=0, p=0.5, m=100, c=0.05, unlock_horizon=5),
        },
        lambda_star=1e-3,
        seed=42,
    )
    assert pol.on_arrival("unknown_hash", 100, current_tick=10) == 0
    assert pol.unknown_hits == 1
    # Known hash may or may not be pinned depending on the sampled d;
    # invariant we can check: result is either 0 (no pin) or the
    # configured horizon (applied at freeable transition).
    out = pol.on_arrival("known", 100, current_tick=10)
    assert out in (0, 5)


def test_build_policy_from_env(monkeypatch):
    monkeypatch.delenv("VLLM_ENCODER_CACHE_POLICY", raising=False)
    assert isinstance(build_policy_from_env(), FIFOEncoderCachePolicy)
    monkeypatch.setenv("VLLM_ENCODER_CACHE_POLICY", "nocache")
    assert isinstance(build_policy_from_env(), NoCacheEncoderCachePolicy)
    monkeypatch.setenv("VLLM_ENCODER_CACHE_POLICY", "offline")
    monkeypatch.delenv("VLLM_ENCODER_CACHE_POLICY_CONFIG", raising=False)
    with pytest.raises(ValueError):
        build_policy_from_env()


def test_offline_policy_from_json(tmp_path):
    cfg = {
        "version": 1,
        "lambda_star": 1e-3,
        "entries": {
            "h_a": {"type_idx": 0, "p": 0.5, "m": 100, "c": 0.05},
            "h_b": {"type_idx": 1, "p": 0.1, "m": 300, "c": 0.20},
        },
    }
    import json

    cfg_path = tmp_path / "mm_pool.json"
    cfg_path.write_text(json.dumps(cfg))
    pol = OfflineLagrangianEncoderCachePolicy.from_json_file(str(cfg_path), seed=0)
    # T_a = ceil(0.05 / (100 * 1e-3)) = ceil(0.5) = 1
    # T_b = ceil(0.20 / (300 * 1e-3)) = ceil(0.667) = 1
    assert pol._pool["h_a"].unlock_horizon == 1
    assert pol._pool["h_b"].unlock_horizon == 1


# ---------------------------------------------------------------------------
# Stats surface
# ---------------------------------------------------------------------------


def test_stats_counters():
    mgr = EncoderCacheManager(cache_size=8, policy=FIFOEncoderCachePolicy())
    req = MockRequest("r1", ["A"], [4])
    mgr.can_allocate(req, 0, int(1e9), 0)
    mgr.allocate(req, 0)  # miss
    assert mgr.misses == 1
    # Second request for same hash -> hit.
    req2 = MockRequest("r2", ["A"], [4])
    assert mgr.check_and_update_cache(req2, 0)
    assert mgr.hits == 1
    stats = mgr.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)
    assert stats["policy"] == "FIFOEncoderCachePolicy"
