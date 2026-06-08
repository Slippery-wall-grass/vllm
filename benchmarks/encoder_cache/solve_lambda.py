# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Solve for the optimal lambda* that maximizes the dual function D(lambda).

D(lambda) = (M - B) * lambda - sum_i p_i * E[(m_i * d_i * lambda - c_i)^+]

where M = sum(m_i), B = cache budget, d_i ~ Geometric(p_i).

Usage:
    python solve_lambda.py \
        --profile-path /tmp/encoder_cache_test_images/profile.json \
        --distribution '{"type_0": 0.3, "type_1": 0.2, "type_2": 0.15, \
                         "type_3": 0.2, "type_4": 0.15}' \
        --cache-budget 1000 \
        --output-path /tmp/encoder_cache_test_images/lambda_config.json
"""

import argparse
import json
import math

import numpy as np
from scipy.optimize import minimize_scalar


def expected_positive_part(m_i: int, c_i: float, p_i: float,
                           lam: float) -> float:
    """Compute E[(m_i * d * lam - c_i)^+] where d ~ Geometric(p_i).

    P[d = d] = p_i * (1-p_i)^d for d = 0, 1, 2, ...

    Using geometric tail sums:
      sum_{d=d0}^inf q^d = q^d0 / p
      sum_{d=d0}^inf d*q^d = d0*q^d0/(1-q) + q^(d0+1)/(1-q)^2
    """
    if lam <= 0 or m_i * lam <= 0:
        return 0.0

    d0 = max(0, math.ceil(c_i / (m_i * lam)))
    q = 1 - p_i

    if q <= 0 or q >= 1:
        if p_i >= 1.0:
            return max(0.0, -c_i)
        return 0.0

    q_d0 = q ** d0

    # Tail probability: sum_{d=d0}^inf p*q^d = q^d0
    tail_prob = q_d0

    # Weighted tail: sum_{d=d0}^inf d*p*q^d = d0*q^d0 + q^(d0+1)/p
    tail_d_weighted = d0 * q_d0 + q ** (d0 + 1) / p_i

    result = m_i * lam * tail_d_weighted - c_i * tail_prob
    return max(0.0, result)


def dual_function(lam: float, types: list[dict], cache_budget: int) -> float:
    """Compute D(lambda) = (M - B)*lambda - sum_i p_i * E[(...)^+]."""
    M = sum(t["m_i"] for t in types)
    B = cache_budget

    val = (M - B) * lam
    for t in types:
        val -= t["p_i"] * expected_positive_part(
            t["m_i"], t["c_i"], t["p_i"], lam
        )
    return val


def solve_lambda(types: list[dict], cache_budget: int) -> dict:
    """Solve for lambda* and compute per-type decision thresholds.

    Returns dict with lambda_star and per-type info.
    """
    def neg_D(lam):
        return -dual_function(lam, types, cache_budget)

    result = minimize_scalar(neg_D, bounds=(0, 100000), method="bounded")
    lambda_star = result.x
    D_star = -result.fun

    # Compute per-type eviction threshold:
    # d_threshold_i = ceil(c_i / (m_i * lambda*))
    # If d < d_threshold_i => non-evictable (keep in cache)
    type_info = []
    for t in types:
        if lambda_star > 0 and t["m_i"] * lambda_star > 0:
            d_threshold = math.ceil(t["c_i"] / (t["m_i"] * lambda_star))
        else:
            d_threshold = 0

        # Expected keep probability:
        # P[d < d_threshold] = 1 - (1-p_i)^d_threshold
        q = 1 - t["p_i"]
        keep_prob = 1 - q ** d_threshold if q < 1 else 0

        type_info.append({
            "type_id": t["type_id"],
            "p_i": t["p_i"],
            "m_i": t["m_i"],
            "c_i": t["c_i"],
            "d_threshold": d_threshold,
            "keep_probability": keep_prob,
        })

    return {
        "lambda_star": lambda_star,
        "D_star": D_star,
        "cache_budget": cache_budget,
        "M_total": sum(t["m_i"] for t in types),
        "types": type_info,
    }


def print_analysis(result: dict) -> None:
    """Print a human-readable analysis of the solution."""
    print("=" * 60)
    print("Distribution-Aware Cache Optimization Results")
    print("=" * 60)
    print(f"lambda*       = {result['lambda_star']:.6f}")
    print(f"D(lambda*)    = {result['D_star']:.6f}")
    print(f"Cache budget  = {result['cache_budget']}")
    print(f"Total M       = {result['M_total']}")
    print()
    print(f"{'Type':<10} {'p_i':<8} {'m_i':<8} {'c_i':<10} "
          f"{'d_thresh':<10} {'keep_prob':<10}")
    print("-" * 60)
    for t in result["types"]:
        print(f"{t['type_id']:<10} {t['p_i']:<8.3f} {t['m_i']:<8} "
              f"{t['c_i']:<10.4f} {t['d_threshold']:<10} "
              f"{t['keep_probability']:<10.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Solve for optimal lambda* for encoder cache policy"
    )
    parser.add_argument("--profile-path", type=str, required=True,
                        help="Path to profile JSON from profile_encoder.py")
    parser.add_argument("--distribution", type=str, required=True,
                        help="JSON dict mapping type_id -> p_i probability")
    parser.add_argument("--cache-budget", type=int, default=None,
                        help="Cache budget B in number of encoder embeddings. "
                        "INFORMATIONAL ONLY: the runtime cache manager "
                        "re-solves lambda* using vLLM's actual encoder "
                        "cache size, so this value only affects the offline "
                        "report printed here. If omitted, the offline solve "
                        "uses B = sum(m_i) // 2 as a placeholder.")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Path for output config JSON")
    parser.add_argument(
        "--c-i-scale", type=float, default=1.0,
        help="Multiplicative correction applied to every c_i loaded "
             "from --profile-path. Useful when profile_encoder used a "
             "different attn implementation / dtype than vLLM at "
             "runtime (e.g. HF eager vs vLLM flash-attn). For a "
             "Qwen2.5-VL profile in fp16 vs runtime in bf16+flash, "
             "passing --c-i-scale=0.3 brings c_i in line with what the "
             "runtime measures. Set to 1.0 (default) to keep raw "
             "profile values.",
    )
    args = parser.parse_args()

    with open(args.profile_path) as f:
        profile = json.load(f)

    if args.c_i_scale != 1.0:
        print(f"NOTE: scaling profile c_i by {args.c_i_scale} to "
              f"align with runtime encoder speed.")
        for t in profile.values():
            t["c_i"] = t["c_i"] * args.c_i_scale

    distribution = json.loads(args.distribution)

    # Build types list
    types = []
    for type_id, p_i in distribution.items():
        if type_id not in profile:
            print(f"WARNING: {type_id} in distribution but not in profile, "
                  "skipping")
            continue
        prof = profile[type_id]
        types.append({
            "type_id": type_id,
            "p_i": p_i,
            "m_i": prof["m_i"],
            "c_i": prof["c_i"],
        })

    # Normalize probabilities
    total_p = sum(t["p_i"] for t in types)
    if abs(total_p - 1.0) > 0.01:
        print(f"WARNING: probabilities sum to {total_p}, normalizing to 1.0")
        for t in types:
            t["p_i"] /= total_p

    cache_budget = args.cache_budget
    if cache_budget is None:
        # Placeholder only: the runtime manager re-solves with vLLM's real
        # cache size. Using M // 2 gives a reasonable offline preview.
        M = sum(t["m_i"] for t in types)
        cache_budget = max(1, M // 2)
        print(f"NOTE: --cache-budget not provided; using placeholder "
              f"B = M//2 = {cache_budget} for offline reporting only. "
              "Runtime lambda* is re-computed with vLLM's actual cache size.")

    result = solve_lambda(types, cache_budget)
    print_analysis(result)

    # Save output config
    output_path = args.output_path
    if output_path is None:
        from pathlib import Path
        output_path = str(
            Path(args.profile_path).parent / "lambda_config.json"
        )

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Config written to {output_path}")


if __name__ == "__main__":
    main()
