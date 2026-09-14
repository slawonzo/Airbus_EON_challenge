"""
Iterative local-surrogate refinement — the hybrid loop, tested classically.

Motivation (measured, instB33): the GLOBAL quadratic surrogate has an argmin true-gap
of +8.79% that is UNCHANGED when training data grows 5x — a model-class bias, not
variance. Congestion saturates (clip at 0), so the set function is not globally
quadratic. But locally, around an incumbent plan, quadratic is a far better model.

The loop:  fit a quadratic surrogate on subsets NEAR the incumbent  ->  minimise it
(here Gurobi MIQP; on hardware, the QPU sampler)  ->  re-score the argmin on the TRUE
DC objective  ->  accept if better, recentre, tighten the neighbourhood.

This is exactly the hybrid quantum-classical workflow of the proposal with the sampler
swapped for an exact QUBO solver, so it measures the CEILING the loop offers: if it
closes the gap here, a sampler that reaches the surrogate's low-energy set inherits it.

Usage:
    python surrogate_refine.py --instance instB33 --rounds 6
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from grid_expansion_qubo import load_instance, exact_objective   # noqa: E402
from milp_ground_truth import solve_qubo_miqp                    # noqa: E402


def fit_local_quadratic(F, center, n, n_train, radius, rng):
    """Ridge-LSQ quadratic fit on subsets within ~radius Hamming flips of center."""
    X = np.empty((n_train, n), dtype=float)
    y = np.empty(n_train)
    flip_p = radius / n
    for t in range(n_train):
        x = center.copy()
        mask = rng.random(n) < flip_p
        x[mask] ^= 1
        X[t] = x
        y[t] = F(tuple(np.flatnonzero(x)))
    iu, ju = np.triu_indices(n, k=1)
    A = np.hstack([np.ones((n_train, 1)), X, X[:, iu] * X[:, ju]])
    p = A.shape[1]
    beta = np.linalg.solve(A.T @ A + 1e-8 * np.eye(p), A.T @ y)
    Q = np.zeros((n, n))
    Q[np.arange(n), np.arange(n)] = beta[1:n + 1]
    Q[iu, ju] = beta[n + 1:] / 2.0
    Q[ju, iu] = beta[n + 1:] / 2.0
    return Q, float(beta[0]), int(n_train)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--instance", default="instB33")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--n-train", type=int, default=1500, help="samples per round")
    p.add_argument("--radius", type=float, default=6.0,
                   help="initial mean Hamming distance of training samples")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    meta, net, cands, costs, Q0 = load_instance(args.instance)
    n = meta["n_candidates"]
    rng = np.random.default_rng(args.seed)
    F = exact_objective(net, cands, costs, meta["lambda"], meta["mu"], meta["budget"])

    # certified optimum if available (for reporting only — never used by the loop)
    gt_path = Path(f"{meta['name']}_ground_truth.json")
    opt = None
    if gt_path.exists():
        with open(gt_path) as fh:
            opt = json.load(fh)["true_milp"]["objective"]

    # round 0: global surrogate argmin = the published starting point
    r0 = solve_qubo_miqp(meta, Q0)
    center = np.array([int(c) for c in r0["plan"]])
    best = float(F(tuple(np.flatnonzero(center))))
    true_evals = 1

    def gap(v):
        return f"{100 * (v - opt) / abs(opt):+.3f}%" if opt else "n/a"

    print(f"Local-surrogate refinement on '{meta['name']}': {n} candidates, "
          f"{args.rounds} rounds x {args.n_train} true evals")
    if opt:
        print(f"  certified optimum F* = {opt:.6f} (reporting only)")
    print(f"  round 0 (global surrogate argmin): F = {best:.6f}  gap {gap(best)}")

    trajectory = [{"round": 0, "F": best, "gap_pct":
                   100 * (best - opt) / abs(opt) if opt else None}]
    radius = args.radius
    for r in range(1, args.rounds + 1):
        Qc, c0, used = fit_local_quadratic(F, center, n, args.n_train, radius, rng)
        true_evals += used
        meta_local = {"qubo_constant": c0}
        sol = solve_qubo_miqp(meta_local, Qc)
        xs = np.array([int(c) for c in sol["plan"]])
        v = float(F(tuple(np.flatnonzero(xs))))
        true_evals += 1
        moved = v < best - 1e-12
        if moved:
            best, center = v, xs
        else:
            radius = max(radius * 0.7, 2.0)     # shrink the trust region
        print(f"  round {r}: argmin true F = {v:.6f}  gap {gap(v)}  "
              f"{'ACCEPT' if moved else f'reject (radius -> {radius:.1f})'}")
        trajectory.append({"round": r, "F": v, "accepted": moved,
                           "radius": radius,
                           "gap_pct": 100 * (v - opt) / abs(opt) if opt else None})

    print(f"\n  Final: F = {best:.6f}  gap {gap(best)}  "
          f"[{true_evals} true evals total]")
    out = Path(f"{meta['name']}_refine.json")
    with open(out, "w") as fh:
        json.dump({"instance": meta["name"],
                   "qubo_sha256_16": meta["qubo_sha256_16"],
                   "certified_optimum": opt, "final_F": best,
                   "final_plan": "".join(map(str, center)),
                   "true_evals": true_evals, "n_train_per_round": args.n_train,
                   "trajectory": trajectory}, fh, indent=2)
    print(f"  ✓ Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
