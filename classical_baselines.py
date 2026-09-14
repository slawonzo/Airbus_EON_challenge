"""
Classical baselines for a grid-expansion instance — the numbers a quantum claim must beat.

Baselines must be TUNED, not strawmen. Four methods, each
reporting true-DC cost, evaluation budget and wall time:

  greedy       forward selection on the true objective (the planner's default).
  sa_true      simulated annealing directly on the true objective, multi-restart,
               T0 auto-tuned from the single-flip |dF| distribution (95th pct).
  sa_surrogate SA on the QUBO surrogate (cheap), elites re-scored against truth —
               the CLASSICAL TWIN of the quantum pipeline (sample surrogate, re-score
               truth). Its gap vs sa_true prices what the surrogate costs anyone.
  random       uniform plans with the same true-eval budget as sa_true (sanity floor).

--emit-results writes the random sampler's output in the standard results-JSON schema,
which doubles as an end-to-end test harness for grid_evaluate.py.

Usage:
    python classical_baselines.py --instance inst33
    python classical_baselines.py --instance instB --restarts 20 --sweeps 300
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from grid_expansion_qubo import load_instance, exact_objective, qubo_value  # noqa: E402


class Counter:
    """Wrap F so every method's evaluation budget is accounted, not guessed."""

    def __init__(self, F):
        self.F, self.n = F, 0

    def __call__(self, sel):
        self.n += 1
        return self.F(sel)


def x_to_sel(x):
    return tuple(i for i, v in enumerate(x) if v)


def greedy(F, n):
    x = np.zeros(n, dtype=int)
    best = F(())
    improved = True
    while improved:
        improved = False
        for i in np.flatnonzero(x == 0):
            x[i] = 1
            v = F(x_to_sel(x))
            if v < best - 1e-12:
                best, improved = v, True
            else:
                x[i] = 0
    return best, x


def tune_T0(energy, n, rng, probes=200):
    """95th percentile of single-flip |dE| at random states — accepts most moves at
    the start without being blind, the standard non-strawman initialisation."""
    deltas = []
    for _ in range(probes):
        x = rng.integers(0, 2, n)
        i = rng.integers(n)
        e0 = energy(x)
        x[i] ^= 1
        deltas.append(abs(energy(x) - e0))
    return float(np.percentile(deltas, 95)) + 1e-12


def sa(energy, n, rng, restarts, sweeps, T0):
    """Multi-restart single-flip Metropolis with geometric cooling to T0/1000.
    Returns (best_value, best_x, samples) where samples collects (x, E) visits."""
    best_v, best_x = np.inf, None
    alpha = (1e-3) ** (1.0 / max(sweeps - 1, 1))
    for _r in range(restarts):
        x = rng.integers(0, 2, n)
        e = energy(x)
        T = T0
        for _s in range(sweeps):
            for i in rng.permutation(n):
                x[i] ^= 1
                e2 = energy(x)
                if e2 <= e or rng.random() < np.exp(-(e2 - e) / T):
                    e = e2
                else:
                    x[i] ^= 1
            T *= alpha
            if e < best_v:
                best_v, best_x = e, x.copy()
    return best_v, best_x


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--instance", default="inst33")
    p.add_argument("--restarts", type=int, default=10)
    p.add_argument("--sweeps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--emit-results", action="store_true",
                   help="also write <instance>_random_results.json in the standard "
                        "results schema (test harness for grid_evaluate.py)")
    args = p.parse_args()

    meta, net, cands, costs, Q = load_instance(args.instance)
    n = meta["n_candidates"]
    const = meta["qubo_constant"]
    rng = np.random.default_rng(args.seed)
    gt = meta.get("ground_truth")
    opt = gt["optimal_value"] if gt else None

    print(f"Baselines for '{meta['name']}' [{meta.get('network', 'synth')}]: "
          f"{n} candidates, lambda={meta['lambda']:g}")
    if gt:
        print(f"  known optimum F = {opt:.4f} (build-nothing "
              f"{gt['build_none_value']:.4f})")

    report = {"instance": meta["name"], "qubo_sha256_16": meta["qubo_sha256_16"],
              "seed": args.seed, "methods": {}}

    def record(name, best, x, evals, wall, extra=None):
        gap = 100.0 * (best - opt) / abs(opt) if opt is not None else None
        report["methods"][name] = {
            "best_F": float(best), "plan": "".join(map(str, x)),
            "gap_pct": gap, "true_evals": int(evals), "wall_s": round(wall, 3),
            **(extra or {})}
        gtxt = f"gap {gap:+.2f}%" if gap is not None else "no ground truth"
        print(f"  {name:<13} F = {best:.4f}  {gtxt}  "
              f"[{evals} true evals, {wall:.2f}s]")

    # greedy on truth
    Fc = Counter(exact_objective(net, cands, costs, meta["lambda"], meta["mu"],
                                 meta["budget"]))
    t = time.perf_counter()
    v, x = greedy(Fc, n)
    record("greedy", v, x, Fc.n, time.perf_counter() - t)

    # tuned SA on truth
    Fc = Counter(exact_objective(net, cands, costs, meta["lambda"], meta["mu"],
                                 meta["budget"]))
    e_true = lambda xx: Fc(x_to_sel(xx))                          # noqa: E731
    t = time.perf_counter()
    T0 = tune_T0(e_true, n, rng)
    v, x = sa(e_true, n, rng, args.restarts, args.sweeps, T0)
    sa_true_evals = Fc.n
    record("sa_true", v, x, Fc.n, time.perf_counter() - t, {"T0": T0})

    # SA on the QUBO surrogate, elite re-scored on truth (the quantum pipeline's twin)
    Fc = Counter(exact_objective(net, cands, costs, meta["lambda"], meta["mu"],
                                 meta["budget"]))
    e_q = lambda xx: qubo_value(Q, const, xx)                     # noqa: E731
    t = time.perf_counter()
    T0q = tune_T0(e_q, n, rng)
    _vq, xq = sa(e_q, n, rng, args.restarts, args.sweeps, T0q)
    v = Fc(x_to_sel(xq))                                          # ONE true re-score
    record("sa_surrogate", v, xq, Fc.n, time.perf_counter() - t,
           {"T0": T0q, "note": "surrogate evals are free; 1 true re-score"})

    # random with sa_true's budget
    Fc = Counter(exact_objective(net, cands, costs, meta["lambda"], meta["mu"],
                                 meta["budget"]))
    t = time.perf_counter()
    best_v, best_x, rand_counts = np.inf, None, {}
    for _ in range(sa_true_evals):
        xx = rng.integers(0, 2, n)
        vv = Fc(x_to_sel(xx))
        bs = "".join(map(str, xx))
        rand_counts[bs] = rand_counts.get(bs, 0) + 1
        if vv < best_v:
            best_v, best_x = vv, xx.copy()
    record("random", best_v, best_x, Fc.n, time.perf_counter() - t)

    out = Path(f"{meta['name']}_baselines.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"  ✓ Saved {out}")

    if args.emit_results:
        rf = Path(f"{meta['name']}_random_results.json")
        with open(rf, "w") as fh:
            json.dump({"metadata": {"instance": meta["name"],
                                    "qubo_sha256_16": meta["qubo_sha256_16"],
                                    "solver": "uniform_random",
                                    "device": "local",
                                    "total_samples": sum(rand_counts.values()),
                                    "unique_solutions": len(rand_counts)},
                       "results": rand_counts}, fh, indent=2)
        print(f"  ✓ Saved {rf} (evaluator test harness)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
