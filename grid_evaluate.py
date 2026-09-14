"""
Evaluate sampler output (Pasqal, Qiskit, SA, anything) on a grid-expansion instance.

Every bitstring IS a valid expansion plan — feasibility is 100% by construction, so the
one-hot failure mode that dominated our earlier logistics study is structurally absent. The questions left are
the honest ones:

  1. TRUE cost — every sample is re-scored against the exact DC objective, never the
     QUBO surrogate. The surrogate is only the sampling landscape.
  2. Preference — did the sampler PREFER cheap plans, or merely cover the space?
     Decided by qubo_diagnostics.sampling_quality with its pre-registered 5% effect
     floor, p <= 0.05, and MDE reporting (never read a null without its MDE).
  3. Anytime yield — per-shot probability of optimum / within-5% / within-15% plans,
     the metrics a fixed-runtime deployment (the challenge's few-hours runtime requirement) actually buys.
  4. Surrogate alignment — rank corr(QUBO energy, true F) over the sampled set: is the
     landscape the device saw pointing at the truth in the region it actually visited?

Consumes one results JSON schema for every platform ({"metadata": ..., "results":
{bitstring: count}}), so Pasqal and Qiskit outputs flow through one evaluator.

Usage:
    python grid_evaluate.py --instance inst33 --results inst33_pasqal_results.json
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from qubo_diagnostics import sampling_quality, print_sampling_quality   # noqa: E402
from grid_expansion_qubo import (                                       # noqa: E402
    load_instance, exact_objective, qubo_value, GROUND_TRUTH_MAX,
)
import itertools                                                         # noqa: E402


def evaluate(instance, results_file, save=True):
    meta, net, cands, costs, Q = load_instance(instance)
    n = meta["n_candidates"]
    const = meta["qubo_constant"]
    F = exact_objective(net, cands, costs, meta["lambda"], meta["mu"], meta["budget"])

    with open(results_file) as fh:
        data = json.load(fh)
    res_meta = data.get("metadata", {})
    counts = data["results"]

    # ---- stamp guard: refuse to score a mismatched run ------------------------------
    stamped_sha = res_meta.get("qubo_sha256_16")
    if stamped_sha and stamped_sha != meta["qubo_sha256_16"]:
        print(f"✗ Results were produced from QUBO sha {stamped_sha}, but instance "
              f"'{meta['name']}' has sha {meta['qubo_sha256_16']}. Wrong instance.")
        return None
    widths = {len(b) for b in counts}
    if widths != {n}:
        print(f"✗ Bitstring width {widths} != {n} candidates. Wrong instance.")
        return None

    total_shots = sum(counts.values())
    print("=" * 70)
    print(f"Grid expansion evaluation — instance '{meta['name']}' "
          f"[{meta.get('network', 'synth')}]")
    print("=" * 70)
    print(f"  {meta['n_bus']} buses, {n} candidates, lambda={meta['lambda']:g}; "
          f"solver: {res_meta.get('solver', '?')} on {res_meta.get('device', '?')}")
    print(f"  {total_shots} shots, {len(counts)} distinct plans "
          f"(feasibility 100% by construction — no decoder, no repair)")

    # ---- true cost of every sampled plan ---------------------------------------------
    rows = []                                    # (bitstring, trueF, quboE, shots)
    for bs, c in counts.items():
        x = [int(ch) for ch in bs]
        sel = tuple(i for i, v in enumerate(x) if v)
        rows.append((bs, float(F(sel)), qubo_value(Q, const, x), int(c)))
    rows.sort(key=lambda r: r[1])

    # ---- ground truth / baseline pool -------------------------------------------------
    gt = meta.get("ground_truth")
    pool = None
    if n <= GROUND_TRUTH_MAX:
        subsets = [tuple(s) for k in range(n + 1)
                   for s in itertools.combinations(range(n), k)]
        pool = np.array([F(s) for s in subsets])
        opt_val = float(pool.min())
    elif gt:
        opt_val = gt["optimal_value"]
    else:
        opt_val = rows[0][1]                     # best sampled = weakest claim
    build_none = float(F(()))

    print(f"\n  Best sampled plan:  F = {rows[0][1]:.4f}  "
          f"(optimum {opt_val:.4f}, build-nothing {build_none:.4f})")
    print(f"  Best-plan gap:      {100 * (rows[0][1] - opt_val) / abs(opt_val):+.2f}%")
    print(f"\n  Top sampled plans (true DC cost, not QUBO energy):")
    print(f"  {'plan':<{max(n, 4) + 2}} {'true F':>10} {'QUBO E':>10} {'shots':>6}")
    for bs, f_true, qe, c in rows[:8]:
        mark = " ← optimum" if abs(f_true - opt_val) < 1e-9 else ""
        print(f"  {bs:<{max(n, 4) + 2}} {f_true:>10.4f} {qe:>10.4f} {c:>6}{mark}")

    # ---- preference test (pre-registered 5% floor, p <= 0.05, MDE reported) ----------
    sq = sampling_quality(
        [(bs, f_true, c) for bs, f_true, _qe, c in rows],
        optimal_cost=opt_val,
        feasible_pool_costs=pool,
    )
    if sq:
        print()
        print_sampling_quality(sq, show=20)

    # ---- anytime yield per shot (challenge runtime requirement: fixed budget, what do you get?) ----
    def yield_within(pct):
        thr = opt_val + abs(opt_val) * pct / 100.0
        return sum(c for _b, f_true, _q, c in rows if f_true <= thr) / total_shots

    hit_opt = sum(c for _b, f_true, _q, c in rows
                  if abs(f_true - opt_val) < 1e-9) / total_shots
    y5, y15 = yield_within(5.0), yield_within(15.0)
    print(f"\n  Anytime yield per shot:  optimum {100 * hit_opt:.2f}%   "
          f"within-5% {100 * y5:.2f}%   within-15% {100 * y15:.2f}%")
    if pool is not None:
        u_opt = float((pool <= opt_val + 1e-9).sum()) / len(pool)
        u5 = float((pool <= opt_val + abs(opt_val) * 0.05).sum()) / len(pool)
        u15 = float((pool <= opt_val + abs(opt_val) * 0.15).sum()) / len(pool)
        print(f"  Uniform-random plan:     optimum {100 * u_opt:.2f}%   "
              f"within-5% {100 * u5:.2f}%   within-15% {100 * u15:.2f}%")
        print(f"  Lift over random:        "
              f"optimum {hit_opt / u_opt if u_opt else float('inf'):.1f}x   "
              f"within-5% {y5 / u5 if u5 else float('inf'):.1f}x   "
              f"within-15% {y15 / u15 if u15 else float('inf'):.1f}x")

    # ---- surrogate alignment in the visited region ------------------------------------
    align = None
    if len(rows) >= 3:
        fr = np.array([r[1] for r in rows])
        qr = np.array([r[2] for r in rows])
        align = float(np.corrcoef(np.argsort(np.argsort(fr)),
                                  np.argsort(np.argsort(qr)))[0, 1])
        agree = rows[int(np.argmin(qr))][1] == rows[0][1]
        print(f"\n  Surrogate alignment on sampled set: rank corr {align:.3f}; "
              f"QUBO-best {'IS' if agree else 'is NOT'} true-best among samples")

    analysis = {
        "instance": meta["name"], "network": meta.get("network"),
        "qubo_sha256_16": meta["qubo_sha256_16"],
        "results_file": str(results_file), "solver": res_meta.get("solver"),
        "device": res_meta.get("device"), "total_shots": total_shots,
        "distinct_plans": len(counts),
        "optimal_value": opt_val, "build_none_value": build_none,
        "best_sampled_F": rows[0][1],
        "best_gap_pct": 100 * (rows[0][1] - opt_val) / abs(opt_val),
        "yield_optimum": hit_opt, "yield_within5": y5, "yield_within15": y15,
        "uniform_yields": ([u_opt, u5, u15] if pool is not None else None),
        "surrogate_rank_corr_sampled": align,
        "sampling_quality": (
            {**sq, "solutions": sq["solutions"][:50],
             "solutions_truncated_to": 50} if sq else None),
        "top_plans": [{"plan": b, "true_F": f, "qubo_E": q, "shots": c}
                      for b, f, q, c in rows[:20]],
    }
    if save:
        out = Path(results_file).with_name(
            Path(results_file).stem + "_analysis.json")
        with open(out, "w") as fh:
            json.dump(analysis, fh, indent=2)
        print(f"\n  ✓ Analysis saved to {out}")
    return analysis


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--instance", default="inst33",
                   help="instance name (loads <name>_problem.json + <name>_qubo.npy)")
    p.add_argument("--results", required=True, help="sampler results JSON")
    args = p.parse_args()
    return 0 if evaluate(args.instance, args.results) is not None else 1


if __name__ == "__main__":
    sys.exit(main())
