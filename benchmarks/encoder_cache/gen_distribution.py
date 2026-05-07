# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate a request-distribution JSON for the encoder-cache benchmarks.

The benchmark scripts accept a JSON dict mapping `type_id -> probability`
via the `--distribution` flag (or the `DISTRIBUTION` env var in
run_cache_comparison.sh). This helper produces such a dict for several
canonical workload shapes so you don't have to hand-craft them, and so
runs are reproducible across people / machines.

Presets:

    uniform         every type equally likely (1/K)
    zipf            classic Zipf (s=1.0): p_i ∝ 1/(i+1)
    zipf-light      Zipf with smaller exponent (s=0.5): mildly skewed
    zipf-heavy      Zipf with s=2.0: strongly skewed (long tail)
    skewed          alias for zipf-heavy
    extreme         very hot head (one type ≈ 90%)
    pareto-80-20    top 20% of types share 80% of the mass
    bimodal         half "hot" (mass = 0.8/k_hot), half "cold"
    custom          Zipf with --zipf-s X (any positive float)

Usage:
    # Print to stdout — pipe straight into DISTRIBUTION:
    DISTRIBUTION="$(python gen_distribution.py --preset zipf --num-types 5)"

    # Save to a file:
    python gen_distribution.py --preset pareto-80-20 --num-types 10 \
        -o /tmp/dist_p8020.json
"""

import argparse
import json
import sys


def _zipf_weights(num_types: int, s: float) -> list[float]:
    """Unnormalised Zipf-like weights: w_i = 1/(i+1)^s, i = 0..K-1."""
    return [1.0 / (i + 1) ** s for i in range(num_types)]


def _normalise(ws: list[float]) -> list[float]:
    total = sum(ws)
    if total <= 0:
        raise ValueError("weights sum to non-positive; cannot normalise")
    return [w / total for w in ws]


def _bimodal(num_types: int, hot_frac: float, hot_mass: float) -> list[float]:
    """Bimodal: top `hot_frac` of types share `hot_mass` of probability,
    the rest split (1-hot_mass).
    """
    k_hot = max(1, int(round(num_types * hot_frac)))
    k_cold = max(1, num_types - k_hot)
    hot_each = hot_mass / k_hot
    cold_each = (1.0 - hot_mass) / k_cold
    return [hot_each] * k_hot + [cold_each] * k_cold


def _extreme(num_types: int, head_mass: float = 0.9) -> list[float]:
    """Single hot head, uniform tail."""
    if num_types == 1:
        return [1.0]
    head = head_mass
    tail_each = (1.0 - head) / (num_types - 1)
    return [head] + [tail_each] * (num_types - 1)


def build(preset: str, num_types: int, zipf_s: float = 1.0) -> dict[str, float]:
    if num_types <= 0:
        raise ValueError(f"num_types must be positive, got {num_types}")

    preset = preset.lower()
    if preset == "uniform":
        ws = [1.0] * num_types
    elif preset == "zipf":
        ws = _zipf_weights(num_types, 1.0)
    elif preset == "zipf-light":
        ws = _zipf_weights(num_types, 0.5)
    elif preset in ("zipf-heavy", "skewed"):
        ws = _zipf_weights(num_types, 2.0)
    elif preset == "extreme":
        ws = _extreme(num_types, head_mass=0.9)
    elif preset == "pareto-80-20":
        ws = _bimodal(num_types, hot_frac=0.20, hot_mass=0.80)
    elif preset == "bimodal":
        ws = _bimodal(num_types, hot_frac=0.50, hot_mass=0.80)
    elif preset == "custom":
        ws = _zipf_weights(num_types, zipf_s)
    else:
        raise ValueError(
            f"unknown preset {preset!r}. Available: uniform, zipf, "
            f"zipf-light, zipf-heavy/skewed, extreme, pareto-80-20, "
            f"bimodal, custom")

    probs = _normalise(ws)
    return {f"type_{i}": p for i, p in enumerate(probs)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="zipf",
        choices=("uniform", "zipf", "zipf-light", "zipf-heavy", "skewed",
                 "extreme", "pareto-80-20", "bimodal", "custom"),
        help="Workload shape (default: zipf)",
    )
    parser.add_argument("--num-types", "-k", type=int, required=True,
                        help="Number of distinct media types K")
    parser.add_argument("--zipf-s", type=float, default=1.0,
                        help="Exponent for --preset custom (default 1.0). "
                             "Higher = more skewed.")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to write JSON to (default: stdout)")
    parser.add_argument("--pretty", action="store_true",
                        help="Pretty-print the JSON")
    args = parser.parse_args()

    dist = build(args.preset, args.num_types, args.zipf_s)
    text = (
        json.dumps(dist, indent=2)
        if args.pretty
        else json.dumps(dist, separators=(",", ":"))
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
            f.write("\n")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(text)

    # Echo a brief summary to stderr so the operator sees what was made.
    sorted_items = sorted(dist.items(), key=lambda kv: -kv[1])
    print("Preview (sorted by mass):", file=sys.stderr)
    for tid, p in sorted_items[:10]:
        print(f"  {tid}: {p * 100:.2f}%", file=sys.stderr)
    if len(sorted_items) > 10:
        print(f"  ... ({len(sorted_items) - 10} more)", file=sys.stderr)


if __name__ == "__main__":
    main()
