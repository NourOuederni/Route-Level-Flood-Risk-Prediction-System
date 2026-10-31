"""Analytic Hierarchy Process: expert weights for the susceptibility factors (guide 6.4, 7.2).

Usage (CLI):
    python -m app.engine.ahp experts/*.yaml            # aggregate experts, print weights + CR
    python -m app.engine.ahp experts/*.yaml --mc 2000  # plus Monte-Carlo stability check

Each expert file holds the upper triangle of a Saaty pairwise matrix:
    factors: [depression, drainage_position, low_slope, impervious, crossing, past_flood]
    pairs:
      depression>drainage_position: 1      # 1..9, or 1/3 style fractions as strings
      depression>low_slope: 3
      ...
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml

# Saaty random consistency index, n = 1..10
RANDOM_INDEX = [0.0, 0.0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]


def matrix_from_pairs(factors: list[str], pairs: dict[str, float | str]) -> np.ndarray:
    n = len(factors)
    idx = {f: i for i, f in enumerate(factors)}
    m = np.ones((n, n))
    for key, value in pairs.items():
        a, b = key.split(">")
        v = float(Fraction(str(value)))
        if v <= 0:
            raise ValueError(f"{key}: comparisons must be positive")
        i, j = idx[a.strip()], idx[b.strip()]
        m[i, j], m[j, i] = v, 1.0 / v
    return m


def weights_and_cr(m: np.ndarray) -> tuple[np.ndarray, float]:
    """Principal-eigenvector weights and Saaty consistency ratio (CR < 0.10 is acceptable)."""
    n = m.shape[0]
    vals, vecs = np.linalg.eig(m)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    lam = vals[k].real
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX[n - 1] if n <= len(RANDOM_INDEX) else 1.49
    return w, (ci / ri if ri > 0 else 0.0)


def aggregate(matrices: list[np.ndarray]) -> np.ndarray:
    """Aggregate individual judgements by element-wise geometric mean (AIJ)."""
    return np.exp(np.mean([np.log(m) for m in matrices], axis=0))


def monte_carlo(m: np.ndarray, runs: int = 1000, spread: float = 0.3,
                seed: int = 0) -> dict[str, np.ndarray]:
    """Perturb judgements log-normally and report the weight distribution."""
    rng = np.random.default_rng(seed)
    n = m.shape[0]
    out = []
    for _ in range(runs):
        p = np.ones_like(m)
        for i in range(n):
            for j in range(i + 1, n):
                v = m[i, j] * float(np.exp(rng.normal(0, spread)))
                v = float(np.clip(v, 1 / 9, 9))
                p[i, j], p[j, i] = v, 1 / v
        out.append(weights_and_cr(p)[0])
    arr = np.array(out)
    return {"mean": arr.mean(0), "p05": np.percentile(arr, 5, 0), "p95": np.percentile(arr, 95, 0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--mc", type=int, default=0, help="Monte-Carlo runs (0 = skip)")
    args = ap.parse_args()

    factors: list[str] | None = None
    mats = []
    for f in args.files:
        doc = yaml.safe_load(f.read_text(encoding="utf-8"))
        factors = factors or doc["factors"]
        if doc["factors"] != factors:
            raise SystemExit(f"{f}: factor order differs from the first file")
        m = matrix_from_pairs(factors, doc["pairs"])
        w, cr = weights_and_cr(m)
        flag = "OK" if cr < 0.10 else "INCONSISTENT - ask the expert to revise"
        print(f"{f.name}: CR = {cr:.3f} ({flag})")
        mats.append(m)

    agg = aggregate(mats)
    w, cr = weights_and_cr(agg)
    print(f"\nAggregated ({len(mats)} experts), CR = {cr:.3f}")
    print("susceptibility:\n  weights:")
    for name, value in zip(factors, w):
        print(f"    {name}: {value:.3f}")
    if args.mc:
        res = monte_carlo(agg, runs=args.mc)
        print(f"\nMonte-Carlo ({args.mc} runs): mean [p05, p95]")
        for i, name in enumerate(factors):
            print(f"  {name:18s} {res['mean'][i]:.3f} [{res['p05'][i]:.3f}, {res['p95'][i]:.3f}]")


if __name__ == "__main__":
    main()
