# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Closed-form Lagrangian dual for the encoder cache pinning policy.

Notation matches ``encoder_cache_policy.py``:

- K item types indexed by i in {0, ..., K-1}.
- p_i: probability that a uniformly sampled mm_item is of type i. The
  inter-arrival gap d_i between two consecutive type-i mm_items is then
  Geom(p_i) on support {0, 1, 2, ...} with pmf ``p_i * (1-p_i)**d``.
- m_i: memory cost in encoder embedding tokens (a.k.a. ``num_embeds``).
- c_i: encoder compute cost in seconds.
- M = sum_i m_i: aggregate cache demand if everything were pinned.
- B: encoder cache capacity in encoder embedding tokens.

The Lagrangian dual function used by the policy is

    D(lambda) = (M - B) * lambda
              - sum_i p_i * E_d~Geom(p_i)[ (m_i * lambda * d - c_i)+ ]

and we want ``lambda*`` = argmax_{lambda >= 0} D(lambda).

The expectation admits a closed form. Let ``a = m_i * lambda``,
``b = c_i``, ``q = 1 - p_i``, and ``d0 = floor(b/a) + 1`` (the smallest
integer d at which a*d > b). Then::

    E[(a*d - b)+] = q**d0 * ( a * (d0 + q/p_i) - b )

valid for ``a > 0``. When ``a = 0`` the expectation is identically 0.

``D`` is piecewise linear in lambda (knots at lambda = c_i / (m_i * k)
for integer k >= 1) and concave on lambda >= 0, so a 1D bounded
maximization with sufficient evaluations is exact for our purposes.
We use golden-section search over a log-spaced bracket and refine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TypeStats:
    p: float
    m: float
    c: float


def expected_positive_part(p: float, m: float, c: float, lam: float) -> float:
    """Closed-form ``E[(m*lam*d - c)+]`` for ``d ~ Geom(p)``.

    Geometric support is ``{0, 1, 2, ...}`` with pmf ``p*(1-p)**d``.
    """
    if lam <= 0.0 or m <= 0.0:
        return 0.0
    a = m * lam
    b = c
    if b <= 0.0:
        # (a*d - b)+ = a*d - b a.s. -> closed form is mean shifted.
        # E[a*d] = a * (1-p)/p; E[(a*d - b)+] when b<=0 is a*(1-p)/p - b.
        return a * (1.0 - p) / p - b
    q = 1.0 - p
    if q <= 0.0:
        # All mass at d=0; positive part is max(0, -b) = 0 since b > 0.
        return 0.0
    d0 = math.floor(b / a) + 1
    # q ** d0 can underflow for large d0; that just means a vanishing
    # contribution, which is fine.
    coef = a * (d0 + q / p) - b
    if coef <= 0.0:
        # Numerically; analytically coef > 0 for d >= d0.
        return 0.0
    return (q**d0) * coef


def dual_value(
    types: list[TypeStats],
    cache_capacity: float,
    lam: float,
) -> float:
    """Evaluate ``D(lambda)``."""
    total_m = sum(t.m for t in types)
    excess = (total_m - cache_capacity) * lam
    penalty = 0.0
    for t in types:
        penalty += t.p * expected_positive_part(t.p, t.m, t.c, lam)
    return excess - penalty


def _golden_section_max(
    f,
    lo: float,
    hi: float,
    tol: float = 1e-9,
    max_iter: int = 200,
) -> float:
    """Golden-section search for the argmax of a unimodal ``f`` on
    ``[lo, hi]``. Returns the argmax.

    For our concave ``D`` the function is unimodal, so this is exact
    up to ``tol``.
    """
    phi = (math.sqrt(5.0) - 1.0) / 2.0  # ~0.618
    a, b = lo, hi
    c_pt = b - phi * (b - a)
    d_pt = a + phi * (b - a)
    fc = f(c_pt)
    fd = f(d_pt)
    for _ in range(max_iter):
        if abs(b - a) < tol * max(1.0, abs(a) + abs(b)):
            break
        if fc > fd:
            b, d_pt, fd = d_pt, c_pt, fc
            c_pt = b - phi * (b - a)
            fc = f(c_pt)
        else:
            a, c_pt, fc = c_pt, d_pt, fd
            d_pt = a + phi * (b - a)
            fd = f(d_pt)
    return 0.5 * (a + b)


def solve_lambda_star(
    types: list[TypeStats],
    cache_capacity: float,
    upper_factor: float = 4.0,
) -> tuple[float, float]:
    """Return ``(lambda_star, D(lambda_star))``.

    ``upper_factor`` scales the heuristic upper bound; the true optimum
    is bounded by ``c_max / m_min`` since beyond that even ``d=1`` would
    not justify caching, and unilaterally raising lambda only decreases
    the dual once every type unpins.
    """
    if not types:
        return 0.0, 0.0
    total_m = sum(t.m for t in types)
    if total_m <= cache_capacity:
        # No cache pressure; the unconstrained optimum is lambda = 0.
        return 0.0, dual_value(types, cache_capacity, 0.0)

    c_max = max(t.c for t in types)
    m_min = min(t.m for t in types if t.m > 0.0)
    if c_max <= 0.0 or m_min <= 0.0:
        return 0.0, dual_value(types, cache_capacity, 0.0)
    hi = upper_factor * c_max / m_min
    lo = 1e-12 * c_max / m_min

    def f(lam: float) -> float:
        return dual_value(types, cache_capacity, lam)

    lam_star = _golden_section_max(f, lo, hi)
    return lam_star, f(lam_star)
