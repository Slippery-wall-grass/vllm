# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pluggable eviction policies for :class:`EncoderCacheManager`.

The manager owns physical cache bookkeeping; the policy decides

1. whether an entry should survive past the moment its last referencing
   request releases it (``evict_on_unreference``), and
2. for entries that linger as freeable, until which global tick they
   are protected from eviction (``on_arrival`` -> ``unlock_tick``).

Three concrete policies are provided:

- ``FIFOEncoderCachePolicy``: legacy behavior. No pins, FIFO over the
  freeable queue.
- ``NoCacheEncoderCachePolicy``: ablation. An entry is dropped the
  moment its reference count reaches zero; the cache only deduplicates
  concurrently in-flight requests.
- ``OfflineLagrangianEncoderCachePolicy``: solves a Lagrangian dual
  offline (see ``tools/precompute_mm_pool.py``). At each arrival of a
  known type i it draws d ~ Geom(p_i) and pins the entry until
  ``current_tick + ceil(c_i / (m_i * lambda*))`` iff
  ``m_i * lambda* * d - c_i < 0``.

An online estimator can later subclass ``EncoderCachePolicy`` and plug
in via ``build_policy_from_env``.
"""

from __future__ import annotations

import abc
import json
import math
import os
import random
from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)


class EncoderCachePolicy(abc.ABC):
    """Strategy hook for :class:`EncoderCacheManager`.

    Implementations are expected to be cheap: ``on_arrival`` is called
    once per scheduling event, on the scheduler's hot path.
    """

    @property
    def evict_on_unreference(self) -> bool:
        """If True, the manager evicts an entry immediately when its
        reference count drops to zero. Otherwise the entry lingers in
        the freeable queue until physical space is needed."""
        return False

    @abc.abstractmethod
    def on_arrival(
        self,
        mm_hash: str,
        num_embeds: int,
        current_tick: int,
    ) -> int:
        """Called for every arrival (hit or miss) of an mm_item.

        Returns the absolute tick value before which the entry must
        not be evicted. A value ``<= current_tick`` means the entry is
        immediately evictable (not pinned).
        """

    def reset(self) -> None:
        """Override to clear policy-local state when the manager
        resets (e.g. after model weight reload)."""


class FIFOEncoderCachePolicy(EncoderCachePolicy):
    """Legacy behavior: no pinning, FIFO over freeable queue order."""

    def on_arrival(
        self,
        mm_hash: str,
        num_embeds: int,
        current_tick: int,
    ) -> int:
        return 0


class NoCacheEncoderCachePolicy(EncoderCachePolicy):
    """Ablation that disables cross-request reuse.

    The entry survives only while at least one request references it.
    Useful as a lower bound for cache-policy ablations.
    """

    @property
    def evict_on_unreference(self) -> bool:
        return True

    def on_arrival(
        self,
        mm_hash: str,
        num_embeds: int,
        current_tick: int,
    ) -> int:
        return 0


@dataclass
class _PoolEntry:
    """Per-type parameters consumed by the Lagrangian policy."""

    type_idx: int
    p: float  # geometric parameter; equals type frequency in the request stream.
    m: float  # memory cost in encoder embedding tokens (a.k.a. m_i).
    c: float  # encoder compute cost in seconds (a.k.a. c_i).
    unlock_horizon: int  # ceil(c / (m * lambda*)), 0 means never pin.


class OfflineLagrangianEncoderCachePolicy(EncoderCachePolicy):
    """Lagrangian dual policy with offline-solved ``lambda*``.

    For each arrival of a known ``mm_hash`` -> type i, samples
    ``d ~ Geom(p_i)`` and pins the entry until
    ``current_tick + T_i`` iff ``m_i * lambda* * d - c_i < 0`` where
    ``T_i = ceil(c_i / (m_i * lambda*))``.

    Unknown mm_hashes (not in the pool config) fall back to "never
    pinned" so cache pressure from out-of-pool traffic is absorbed by
    FIFO without disrupting the policy state for in-pool traffic.
    """

    def __init__(
        self,
        pool: dict[str, _PoolEntry],
        lambda_star: float,
        seed: int = 0,
    ):
        self._pool = pool
        self._lambda = float(lambda_star)
        self._rng = random.Random(seed)
        self._seed = seed
        self._known_hits = 0
        self._unknown_hits = 0
        logger.info(
            "OfflineLagrangianEncoderCachePolicy: %d pool entries, lambda*=%.6g",
            len(pool),
            self._lambda,
        )

    @classmethod
    def from_json_file(
        cls,
        path: str,
        seed: int = 0,
    ) -> "OfflineLagrangianEncoderCachePolicy":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        lam = float(cfg["lambda_star"])
        pool: dict[str, _PoolEntry] = {}
        for mm_hash, entry in cfg["entries"].items():
            p = float(entry["p"])
            m = float(entry["m"])
            c = float(entry["c"])
            if lam > 0 and m > 0:
                horizon = max(1, math.ceil(c / (m * lam)))
            else:
                horizon = 0
            pool[mm_hash] = _PoolEntry(
                type_idx=int(entry.get("type_idx", -1)),
                p=p,
                m=m,
                c=c,
                unlock_horizon=horizon,
            )
        return cls(pool=pool, lambda_star=lam, seed=seed)

    def on_arrival(
        self,
        mm_hash: str,
        num_embeds: int,
        current_tick: int,
    ) -> int:
        entry = self._pool.get(mm_hash)
        if entry is None:
            self._unknown_hits += 1
            return 0
        self._known_hits += 1
        if entry.unlock_horizon == 0 or self._lambda <= 0.0:
            return 0
        # Inverse-CDF sample d ~ Geometric on {0,1,2,...}:
        # P[d >= k] = (1-p)^k  =>  d = floor(log(u) / log(1-p))
        p = entry.p
        if p >= 1.0:
            d = 0
        else:
            u = max(self._rng.random(), 1e-300)
            d = int(math.floor(math.log(u) / math.log(1.0 - p)))
        # Pin iff predicted memory cost is below recomputation savings.
        if entry.m * self._lambda * d - entry.c < 0:
            return current_tick + entry.unlock_horizon
        return 0

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._known_hits = 0
        self._unknown_hits = 0

    @property
    def known_hits(self) -> int:
        return self._known_hits

    @property
    def unknown_hits(self) -> int:
        return self._unknown_hits


def build_policy_from_env() -> EncoderCachePolicy:
    """Construct a policy from the process environment.

    Reads:
      - ``VLLM_ENCODER_CACHE_POLICY``: ``fifo`` (default), ``nocache``,
        or ``offline``.
      - ``VLLM_ENCODER_CACHE_POLICY_CONFIG``: path to JSON config,
        required for ``offline``.
      - ``VLLM_ENCODER_CACHE_POLICY_SEED``: optional RNG seed for the
        offline policy. Defaults to 0.
    """
    mode = os.environ.get("VLLM_ENCODER_CACHE_POLICY", "fifo").strip().lower()
    cfg_path = os.environ.get("VLLM_ENCODER_CACHE_POLICY_CONFIG")
    seed = int(os.environ.get("VLLM_ENCODER_CACHE_POLICY_SEED", "0"))

    if mode == "fifo":
        return FIFOEncoderCachePolicy()
    if mode in ("nocache", "no-cache", "no_cache"):
        return NoCacheEncoderCachePolicy()
    if mode in ("offline", "lagrangian"):
        if not cfg_path:
            raise ValueError(
                "VLLM_ENCODER_CACHE_POLICY=offline requires "
                "VLLM_ENCODER_CACHE_POLICY_CONFIG to point at a JSON file."
            )
        return OfflineLagrangianEncoderCachePolicy.from_json_file(
            cfg_path, seed=seed
        )
    raise ValueError(
        f"Unknown VLLM_ENCODER_CACHE_POLICY: {mode!r}. "
        "Expected one of: fifo, nocache, offline."
    )
