"""
Grid-expansion QUBO builder — E.ON Global Quantum + AI Challenge 2026.

Problem (challenge statement §4.1/§5.1): given a distribution network and a set of
candidate new lines, choose the subset to BUILD that minimises

    F(S) = sum_e cost_e  +  lambda * C(S)

where C(S) is the congestion amount (sum over lines and scenarios of thermal-rating
overload) after adding S, evaluated by DC power flow. One binary variable per candidate
line: n candidates -> n qubits. No one-hot rows, no slack registers — the encoding family
that our earlier logistics study showed fights the analog drive is structurally absent here.

QUBO construction — pairwise interpolation of the congestion set-function:
    C(S) ~ C0 + sum_e D_e x_e + sum_{e<f} I_ef x_e x_f
with D_e = C({e}) - C0 and I_ef = C({e,f}) - C({e}) - C({f}) + C0.
Exact for |S| <= 2 BY CONSTRUCTION; an approximation beyond, because adding lines changes
the network admittance matrix nonlinearly. The approximation is MEASURED, not assumed:
`fit_report()` samples random subsets and compares the fitted C-hat against exact DC
re-solves (RMSE, max error, rank correlation), and the numbers are stamped into the
instance metadata. The off-diagonal I_ef terms are the physical interference between two
added lines (loop-flow coupling) — the quadratic structure is the physics, not an
encoding artefact. Sign-mixed I_ef is the frustration that drives classical hardness — and
it is exactly what Pasqal's analog solver cannot represent (repulsive Rydberg interaction
=> off-diagonals must be >= 0). `--offdiag nonneg` fits a constrained surrogate for that
path only; its fidelity cost is measured and stamped (see build_qubo docstring).

Discipline carried over from the earlier logistics study:
  * every instance is seeded and its metadata stamped (schema, seed, fit error, signal);
  * `--selftest` must pass before any instance is trusted;
  * ground truth by exhaustive enumeration of the TRUE objective (exact DC per subset)
    whenever n_cand <= GROUND_TRUTH_MAX — the QUBO is then scored against it honestly;
  * `signal = feasible-band / max|Q|` is reported (necessary-not-sufficient screen).

Outputs (one contract, consumed by all platforms):
    <name>_qubo.npy      symmetric Q, minimise x^T Q x
    <name>_problem.json  network, candidates, scenarios, fit stats, ground truth, stamps

Usage:
    python grid_expansion_qubo.py --selftest
    python grid_expansion_qubo.py --n-bus 12 --n-cand 12 --lam 50 --seed 7 --name instA
    python grid_expansion_qubo.py --network case33bw --n-cand 33 --scenarios 9 \
        --workers 32 --name instB33s9          # scenario-count probe, many-core box
"""

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "eon-grid-qubo-1"
GROUND_TRUTH_MAX = 14      # exhaustive exact-DC enumeration cap (2^14 solves)
QUBO_BRUTE_MAX = 22        # exhaustive QUBO argmin cap


# ----------------------------------------------------------------------------- network --

class Network:
    """DC-power-flow network. Bus 0 is slack. Lines are (i, j, b, rating)."""

    def __init__(self, n_bus, lines, injections, scenario_weights, coords=None):
        self.n_bus = n_bus
        self.lines = list(lines)                       # existing lines
        self.injections = np.asarray(injections)       # (n_scen, n_bus), rows sum to 0
        self.weights = np.asarray(scenario_weights, dtype=float)
        self.weights = self.weights / self.weights.sum()
        self.coords = coords

    def flows(self, extra_lines=()):
        """DC flows for existing + extra lines. Returns (lines_used, f) where f is
        (n_scen, n_line). One factorisation serves all scenarios (RHS stacked)."""
        lines = self.lines + list(extra_lines)
        n = self.n_bus
        B = np.zeros((n, n))
        for (i, j, b, _r) in lines:
            B[i, i] += b
            B[j, j] += b
            B[i, j] -= b
            B[j, i] -= b
        Bred = B[1:, 1:]
        P = self.injections[:, 1:].T                   # (n_bus-1, n_scen)
        theta = np.zeros((n, P.shape[1]))
        theta[1:, :] = np.linalg.solve(Bred, P)
        f = np.array([[b * (theta[i, s] - theta[j, s]) for (i, j, b, _r) in lines]
                      for s in range(P.shape[1])])     # (n_scen, n_line)
        return lines, f

    def congestion(self, extra_lines=()):
        """Scenario-weighted total thermal overload (the §4.1 'congestion amount')."""
        lines, f = self.flows(extra_lines)
        ratings = np.array([r for (_i, _j, _b, r) in lines])
        over = np.clip(np.abs(f) - ratings[None, :], 0.0, None)
        return float(self.weights @ over.sum(axis=1))


def synthetic_network(n_bus=12, n_chords=3, n_scen=3, seed=7, rating_frac=0.9,
                      stress=1.45):
    """Seeded synthetic MV-style feeder: spanning tree + chords, congested under stress.

    Scenarios: [mild 0.8x, nominal 1.0x, stress] load multipliers; the stress scenario
    also flips two load buses into exporters (DER 'solar noon') so flow directions
    reverse — the loop-flow frustration source. Ratings are set to rating_frac of the
    NOMINAL |flow|, so nominal is near-loaded and stress overloads several lines.
    """
    if n_scen > 3:
        raise ValueError("synthetic_network supports <= 3 scenarios; use a pandapower "
                         "network (--network case33bw/cigre_mv/mv_oberrhein) for more")
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 1, size=(n_bus, 2))

    # minimum spanning tree by Euclidean distance (Prim)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    in_tree, edges = {0}, []
    while len(in_tree) < n_bus:
        best = min(((i, j) for i in in_tree for j in range(n_bus) if j not in in_tree),
                   key=lambda e: d[e])
        edges.append(best)
        in_tree.add(best[1])
    # chords
    all_pairs = [(i, j) for i in range(n_bus) for j in range(i + 1, n_bus)
                 if (i, j) not in edges and (j, i) not in edges]
    rng.shuffle(all_pairs)
    edges += all_pairs[:n_chords]
    used = set(edges)

    # injections: bus 0 slack; 2 generators; rest loads
    gens = list(rng.choice(range(1, n_bus), size=2, replace=False))
    base = np.zeros(n_bus)
    for g in gens:
        base[g] = rng.uniform(0.8, 1.2)
    loads = [i for i in range(1, n_bus) if i not in gens]
    for l in loads:
        base[l] = -rng.uniform(0.2, 0.5)
    base[0] = -base.sum()                              # slack balances

    scen = []
    for mult in (0.8, 1.0, stress):
        v = base.copy()
        v[loads] *= mult
        if mult == stress and len(loads) >= 2:         # DER flip: two loads export
            flip = loads[:2]
            v[flip] = -0.6 * v[flip]
        v[0] = -(v[1:].sum())
        scen.append(v)
    scen = np.array(scen[:n_scen])

    # provisional net (generous ratings) to measure nominal flows, then tighten
    prov = [(i, j, 1.0 / (d[i, j] + 0.05), 1e9) for (i, j) in edges]
    net0 = Network(n_bus, prov, scen, np.ones(n_scen))
    _, f = net0.flows()
    nominal = np.abs(f[min(1, n_scen - 1)])            # the 1.0x scenario
    lines = [(i, j, b, max(rating_frac * fl, 0.05))
             for (i, j, b, _), fl in zip(prov, nominal)]
    weights = [0.25, 0.5, 0.25][:n_scen]
    return Network(n_bus, lines, scen, weights, coords=pts), used, d


def make_candidates(net, used, d, n_cand, seed):
    """Candidate new lines: unused bus pairs (shortest first gets priority mix) plus
    parallels of the most congested existing lines. Cost ~ length; rating moderate."""
    rng = np.random.default_rng(seed + 1)
    lines, f = net.flows()
    ratings = np.array([r for (_i, _j, _b, r) in lines])
    over = np.clip(np.abs(f) - ratings[None, :], 0, None).sum(axis=0)
    hot = list(np.argsort(-over)[:max(2, n_cand // 3)])

    cands = []
    for k in hot:                                       # parallels of congested lines
        i, j, b, r = net.lines[k]
        cands.append((i, j, b, r))
    pool = [(i, j) for i in range(net.n_bus) for j in range(i + 1, net.n_bus)
            if (i, j) not in used and (j, i) not in used]
    pool.sort(key=lambda e: d[e])
    picks = pool[:2 * n_cand]
    rng.shuffle(picks)
    for (i, j) in picks:
        if len(cands) >= n_cand:
            break
        cands.append((i, j, 1.0 / (d[i, j] + 0.05), rng.uniform(0.3, 0.8)))
    cands = cands[:n_cand]
    length = np.array([d[c[0], c[1]] + 0.05 for c in cands])
    costs = 1.0 * length / length.mean()                # unit-mean build cost
    return cands, costs


# -------------------------------------------------------------- pandapower networks --

PP_NETWORKS = ("case33bw", "cigre_mv", "mv_oberrhein")


def from_pandapower(name, n_cand, n_scen=3, rating_frac=0.9, stress=1.45, seed=7,
                    parallel_cost="unit"):
    """Build a Network + candidate set from a pandapower reference grid.

    Networks: case33bw (IEEE 33-bus Baran-Wu — its 5 out-of-service tie lines are the
    natural candidate pool), cigre_mv (CIGRE MV with full DER), mv_oberrhein (real
    German MV grid, 179 buses — instance-C scale).

    Modelling choices, recorded here because they are choices:
      * DC approximation: line b = 1/x_pu; transformers included as lines with
        x_pu = vk% (network stays connected, trafo flows are checked too).
      * Scenarios: (load_mult, der_mult) = (0.8, 0.3), (1.0, 0.6), (stress, 1.2) —
        mild / nominal / stressed-with-DER-export. Networks without DER degrade to
        pure load scaling. n_scen > 3 switches to a load x DER grid with uniform
        weights (see the code comment) — the scenario-count hardness probe.
      * Ratings are SYNTHESISED as rating_frac x nominal |flow| (same rule as the
        synthetic generator) so the expansion problem is non-degenerate; the grids'
        own max_i_ka ratings are generous enough that nothing congests otherwise.
      * Candidates: out-of-service lines first (real ties), then parallels of the
        most-loaded in-service lines. Costs proportional to line length for ties;
        parallels cost `parallel_cost="unit"` (1.0 each — the frozen instC179 rule,
        which makes every mv_oberrhein candidate cost the same) or `"length"` (the
        paralleled line's own km; transformers = median line length) — the
        heterogeneous-cost axis of the hardness screen (added 2026-09-07).
    """
    if parallel_cost not in ("unit", "length"):
        raise ValueError("parallel_cost must be 'unit' or 'length'")
    import pandapower.networks as pn
    rng = np.random.default_rng(seed)
    factory = {"case33bw": pn.case33bw,
               "cigre_mv": lambda: pn.create_cigre_network_mv(with_der="all"),
               "mv_oberrhein": pn.mv_oberrhein}
    net = factory[name]()
    s_base = net.sn_mva

    # bus mapping with the slack at index 0
    slack = int(net.ext_grid.bus.iloc[0])
    order = [slack] + [b for b in net.bus.index if b != slack]
    idx = {b: i for i, b in enumerate(order)}
    n_bus = len(order)

    def line_b(row):
        vn = net.bus.at[row.from_bus, "vn_kv"]
        x_pu = row.x_ohm_per_km * row.length_km / (vn ** 2 / s_base)
        return 1.0 / max(x_pu, 1e-6)

    existing, existing_len, cand_pool, lengths = [], [], [], []
    for _, r in net.line.iterrows():
        rec = (idx[r.from_bus], idx[r.to_bus], line_b(r), 1e9)
        if r.in_service:
            existing.append(rec)
            existing_len.append(float(r.length_km))
        else:
            cand_pool.append((rec, float(r.length_km)))
    median_len = float(np.median(existing_len)) if existing_len else 1.0
    for _, t in getattr(net, "trafo", __import__("pandas").DataFrame()).iterrows():
        x_pu = (t.vk_percent / 100.0) * (s_base / t.sn_mva)
        existing.append((idx[t.hv_bus], idx[t.lv_bus], 1.0 / max(x_pu, 1e-6), 1e9))
        existing_len.append(median_len)

    # injections (per-unit): generation minus load per bus
    base = np.zeros(n_bus)
    der = np.zeros(n_bus)
    for _, l in net.load.iterrows():
        base[idx[l.bus]] -= l.p_mw / s_base
    for tbl in ("sgen", "gen"):
        df = getattr(net, tbl, None)
        if df is not None:
            for _, g in df.iterrows():
                der[idx[g.bus]] += g.p_mw / s_base

    def inj(lm, dm):
        v = lm * base + dm * der
        v[0] -= v.sum()                                 # slack balances
        return v

    if n_scen <= 3:
        pairs = [(0.8, 0.3), (1.0, 0.6), (stress, 1.2)][:n_scen]
        weights = [0.25, 0.5, 0.25][:n_scen]
    else:
        # Scenario-count probe (>3): load x DER grid, load-major order, uniform weights.
        # 3 DER levels (0.3 / 0.75 / 1.2) x ceil(n/3) load levels from 0.8 to `stress`;
        # the first n_scen combinations are used. More scenarios = more distinct flow
        # patterns one plan must satisfy at once — the scenario-count hardness knob.
        n_der = 3
        loads = np.linspace(0.8, stress, -(-n_scen // n_der))
        ders = np.linspace(0.3, 1.2, n_der)
        pairs = [(float(lm), float(dm)) for lm in loads for dm in ders][:n_scen]
        weights = [1.0 / n_scen] * n_scen
    scen = [inj(lm, dm) for lm, dm in pairs]
    network = Network(n_bus, existing, np.array(scen), weights)

    # tighten ratings around the NOMINAL case (load 1.0, DER 0.6), same rule as
    # synthetic_network. For n_scen <= 3 that is scenario index 1 — bit-compatible with
    # every instance generated before the >3 extension (inst33 sha fb00e67d7eb38a5f).
    _, f = network.flows()
    if n_scen <= 3:
        nominal = np.abs(f[min(1, n_scen - 1)])
    else:
        _, fn = Network(n_bus, existing, np.array([inj(1.0, 0.6)]), [1.0]).flows()
        nominal = np.abs(fn[0])
    network.lines = [(i, j, b, max(rating_frac * fl, 0.02))
                     for (i, j, b, _), fl in zip(network.lines, nominal)]

    # candidates: real ties first, then parallels of hottest lines
    cands = [rec for rec, _ln in cand_pool][:n_cand]
    lengths = [ln for _rec, ln in cand_pool][:n_cand]
    if len(cands) < n_cand:
        over = np.argsort(-np.abs(f).sum(axis=0))
        for k in over:
            if len(cands) >= n_cand:
                break
            i, j, b, r = network.lines[k]
            cands.append((i, j, b, max(r, 0.1)))
            lengths.append(1.0 if parallel_cost == "unit" else existing_len[k])
    lengths = np.array(lengths[: len(cands)]) + 0.05
    costs = lengths / lengths.mean()
    return network, cands[:n_cand], costs[:n_cand]


# -------------------------------------------------------------------------------- QUBO --

def _congestion_chunk(args):
    n_bus, lines, injections, weights, cands, chunk = args
    net = Network(n_bus, lines, injections, weights)
    return [net.congestion([cands[e] for e in sel]) for sel in chunk]


def congestion_batch(net, cands, subsets, workers=1):
    """Exact DC congestion for many subsets. Bit-identical for any `workers` (each
    subset is a pure function), so parallelism never changes an instance's hash —
    it only makes the 120-candidate fits (29k DC solves) feasible on a many-core box."""
    if workers is None or workers <= 1 or len(subsets) < 64:
        return np.array([net.congestion([cands[e] for e in sel]) for sel in subsets])
    import os
    if not any(os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                            "MKL_NUM_THREADS")):
        # Learned the hard way on the 224-core H200 box: 120 workers x one BLAS thread per
        # core = load average 3400. The DC solves are 33-179 x 179 — single-threaded is faster.
        print(f"  ⚠ --workers {workers} without OMP_NUM_THREADS=1: every worker will spawn one "
              f"BLAS thread per core (oversubscription). Set OMP_NUM_THREADS=1 "
              f"OPENBLAS_NUM_THREADS=1 in the environment.", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    n_chunks = min(len(subsets), workers * 8)
    chunks = [subsets[i::n_chunks] for i in range(n_chunks)]      # strided split
    spec = (net.n_bus, net.lines, net.injections, net.weights, cands)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_congestion_chunk, [spec + (c,) for c in chunks]))
    y = np.empty(len(subsets))
    for i, part in enumerate(parts):
        y[i::n_chunks] = part                                      # re-interleave
    return y


def build_qubo(net, cands, costs, lam=50.0, mu=0.0, budget=0, method="regress",
               n_train=None, rng=None, offdiag="free", workers=1):
    """Quadratic congestion surrogate -> QUBO. Returns (Q, const, parts) with
    F-hat(x) = const + x^T Q x. Optional soft budget mu*(sum x - budget)^2.

    method='anchor': exact pairwise interpolation from C(∅), C({e}), C({e,f}).
        Exact for |S|<=2 by construction — but congestion SATURATES at zero as lines
        are added, so extrapolation from the ∅ end is poor exactly where the optimum
        lives (measured: rank corr 0.18, argmin true-rank 146/4096 on the seed-7
        instance). Kept for the self-test and as a cautionary baseline.
    method='regress' (default): least-squares fit of (const, D, I) over subsets
        sampled uniformly across ALL sizes — the best quadratic in the region that
        matters, standard surrogate-model practice. Quality is measured out-of-sample
        by fit_report() and stamped into the metadata; sampled solutions are ALWAYS
        re-scored against the TRUE objective (exact DC), never the surrogate energy —
        the same energy-vs-cost separation our earlier logistics study enforced.

    offdiag='free' (default): unconstrained fit; I_ef takes either sign (the physics:
        loop-flow coupling can relieve or aggravate). Measured on case33bw: 42-58 % of
        the couplings come out NEGATIVE.
    offdiag='nonneg': the same least-squares fit with the pairwise coefficients
        constrained to I_ef >= 0 (bounded LSQ, scipy lsq_linear). Needed for Pasqal's
        analog neutral-atom path, whose Rydberg 1/r^6 interaction is purely repulsive:
        qubosolver raises "Quantum solver does not handle off-diagonal negative
        coefficients" on a sign-mixed Q (submission of inst33 on 2026-09-07). The
        diagonal stays free (detuning map). Gate-based QAOA has no such restriction, so
        this is a modelling concession made ONLY for the analog cross-check, and its
        fidelity cost is measured by fit_report()/ground truth like any other fit. A
        lossless variable flip (x -> 1-x, which flips the sign of I_ef when exactly one
        of e,f is flipped) was checked first: on inst33 it removes at most 19 of 38
        negatives, so it cannot replace the constraint.
    """
    if offdiag not in ("free", "nonneg"):
        raise ValueError(f"offdiag must be 'free' or 'nonneg', got {offdiag!r}")
    n = len(cands)
    C0 = net.congestion()

    if method == "anchor":
        C1 = np.array([net.congestion([cands[e]]) for e in range(n)])
        D = C1 - C0
        I = np.zeros((n, n))
        for e in range(n):
            for f_ in range(e + 1, n):
                cef = net.congestion([cands[e], cands[f_]])
                I[e, f_] = I[f_, e] = cef - C1[e] - C1[f_] + C0
        c_const = C0
        if offdiag == "nonneg":
            I = np.maximum(I, 0.0)          # anchor mode is a baseline only: clip
    else:
        rng = rng or np.random.default_rng(0)
        p = 1 + n + n * (n - 1) // 2
        n_train = n_train or max(4 * p, 300)
        # anchors (all sizes 0..2) plus random subsets uniform over size
        subsets = [()] + [(e,) for e in range(n)] + \
                  list(itertools.combinations(range(n), 2))
        while len(subsets) < n_train:
            k = int(rng.integers(0, n + 1))
            subsets.append(tuple(sorted(rng.choice(n, size=k, replace=False))))
        A = np.zeros((len(subsets), p))
        pair_idx = {pr: 1 + n + t
                    for t, pr in enumerate(itertools.combinations(range(n), 2))}
        for r, sel in enumerate(subsets):
            A[r, 0] = 1.0
            for e in sel:
                A[r, 1 + e] = 1.0
            for pr in itertools.combinations(sel, 2):
                A[r, pair_idx[pr]] = 1.0
        y = congestion_batch(net, cands, subsets, workers)
        ridge = 1e-8
        if offdiag == "free":
            beta = np.linalg.solve(A.T @ A + ridge * np.eye(p), A.T @ y)
        else:
            # Same ridge objective ||A b - y||^2 + ridge ||b||^2, written as an
            # augmented least-squares system so the bound constraint can be applied:
            # const and D free, every pair coefficient >= 0.
            from scipy.optimize import lsq_linear
            A_aug = np.vstack([A, np.sqrt(ridge) * np.eye(p)])
            y_aug = np.concatenate([y, np.zeros(p)])
            lb = np.full(p, -np.inf)
            lb[1 + n:] = 0.0
            res = lsq_linear(A_aug, y_aug, bounds=(lb, np.full(p, np.inf)),
                             method="bvls" if p <= 1500 else "trf", tol=1e-12)
            beta = res.x
            beta[1 + n:] = np.maximum(beta[1 + n:], 0.0)   # exact zeros, not -1e-17
        c_const, D = float(beta[0]), beta[1:1 + n]
        I = np.zeros((n, n))
        for pr, col in pair_idx.items():
            I[pr[0], pr[1]] = I[pr[1], pr[0]] = beta[col]

    Q = np.diag(costs + lam * D) + lam * I / 2.0        # symmetric: off-diag split
    const = lam * c_const
    if mu > 0:
        Q += mu * (np.ones((n, n)) - np.eye(n))         # cross terms of (sum x)^2
        Q += np.diag(np.full(n, mu * (1 - 2 * budget)))
        const += mu * budget ** 2
    off = Q[~np.eye(n, dtype=bool)]
    return Q, const, {"C0": C0, "method": method, "offdiag": offdiag,
                      "I_max": float(np.abs(I).max()),
                      "offdiag_negative": int((off < 0).sum()),
                      "analog_representable": bool((off >= 0).all())}


def exact_objective(net, cands, costs, lam, mu=0.0, budget=0):
    """True F(S) with an exact DC re-solve — the yardstick the QUBO is scored against."""
    def F(sel):
        S = [cands[e] for e in sel]
        v = costs[list(sel)].sum() if sel else 0.0
        v += lam * net.congestion(S)
        if mu > 0:
            v += mu * (len(sel) - budget) ** 2
        return float(v)
    return F


def qubo_value(Q, const, x):
    x = np.asarray(x, dtype=float)
    return float(const + x @ Q @ x)


def brute_force_qubo(Q, const):
    n = Q.shape[0]
    best, bx = np.inf, None
    for bits in itertools.product([0, 1], repeat=n):
        v = qubo_value(Q, const, bits)
        if v < best:
            best, bx = v, bits
    return best, bx


def fit_report(net, cands, costs, Q, const, lam, rng, n_samples=200, mu=0.0, budget=0):
    """Measure the pairwise approximation on random subsets of ALL sizes."""
    n = len(cands)
    F = exact_objective(net, cands, costs, lam, mu, budget)
    exact, fitted = [], []
    for _ in range(n_samples):
        k = rng.integers(0, n + 1)
        sel = tuple(sorted(rng.choice(n, size=k, replace=False))) if k else ()
        x = np.zeros(n)
        x[list(sel)] = 1
        exact.append(F(sel))
        fitted.append(qubo_value(Q, const, x))
    exact, fitted = np.array(exact), np.array(fitted)
    err = fitted - exact
    denom = exact.std() if exact.std() > 0 else 1.0
    rank = float(np.corrcoef(np.argsort(np.argsort(exact)),
                             np.argsort(np.argsort(fitted)))[0, 1])
    return {"n_samples": int(n_samples),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "max_abs_err": float(np.abs(err).max()),
            "rmse_over_spread": float(np.sqrt((err ** 2).mean()) / denom),
            "rank_corr": rank}


def qubo_stats(Q):
    off = Q[~np.eye(Q.shape[0], dtype=bool)]
    nz = off[np.abs(off) > 1e-12]
    return {"max_abs": float(np.abs(Q).max()),
            "offdiag_density": float(len(nz) / max(len(off), 1)),
            "offdiag_sign_mix": float((nz > 0).mean()) if len(nz) else 0.0,
            # strict '< 0' is exactly the test qubosolver applies before an analog run
            "offdiag_negative": int((off < 0).sum()),
            "analog_representable": bool((off >= 0).all()),
            "diag_uniform": bool(np.allclose(np.diag(Q), np.diag(Q)[0]))}


# ---------------------------------------------------------------------------- selftest --

def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'✓' if cond else '✗'} {name}")
        ok = ok and cond

    print("=" * 70 + "\nGrid QUBO self-test\n" + "=" * 70)

    # 1. two-bus analytic case: P=1 through b=2 line -> flow exactly 1, overload 0.6
    net2 = Network(2, [(0, 1, 2.0, 0.4)], [[-1.0, 1.0]], [1.0])
    _, f = net2.flows()
    check("2-bus DC flow analytic (|f|=1)", abs(abs(f[0][0]) - 1.0) < 1e-9)
    check("2-bus overload = 0.6", abs(net2.congestion() - 0.6) < 1e-9)

    # 2. flow conservation on synthetic net
    net, used, d = synthetic_network(seed=3)
    lines, f = net.flows()
    inj = net.injections[0].copy()
    for (li, (i, j, b, r)) in enumerate(lines):
        inj[i] -= f[0][li]
        inj[j] += f[0][li]
    check("flow conservation (KCL) at every bus", np.abs(inj).max() < 1e-8)

    # 3. anchor mode: exact for |S|<=2 by construction
    cands, costs = make_candidates(net, used, d, 8, seed=3)
    Qa, ca, _ = build_qubo(net, cands, costs, lam=50.0, method="anchor")
    F = exact_objective(net, cands, costs, lam=50.0)
    worst = 0.0
    for sel in [(), (0,), (3,), (0, 1), (2, 5), (6, 7)]:
        x = np.zeros(8)
        x[list(sel)] = 1
        worst = max(worst, abs(qubo_value(Qa, ca, x) - F(sel)))
    check("anchor-mode QUBO exact on all |S|<=2 probes", worst < 1e-9)

    # 4. regression mode: out-of-sample quality must beat the anchor pathology
    rng = np.random.default_rng(11)
    Q, const, _ = build_qubo(net, cands, costs, lam=50.0, rng=rng)
    fr = fit_report(net, cands, costs, Q, const, 50.0, rng)
    fa = fit_report(net, cands, costs, Qa, ca, 50.0, np.random.default_rng(11))
    check(f"regression fit rank corr > 0.8 (got {fr['rank_corr']:.3f}; "
          f"anchor gives {fa['rank_corr']:.3f})", fr["rank_corr"] > 0.8)
    check("regression beats anchor out-of-sample",
          fr["rmse"] < fa["rmse"])

    # 5. structure
    check("Q symmetric", np.allclose(Q, Q.T))
    check("diag NOT uniform (DMM is live here, unlike logistics)",
          not np.allclose(np.diag(Q), np.diag(Q)[0]))

    # 6. analog-representable fit: no negative off-diagonal, cost measured not hidden
    Qn, cn, pn = build_qubo(net, cands, costs, lam=50.0, rng=np.random.default_rng(11),
                            offdiag="nonneg")
    offn = Qn[~np.eye(8, dtype=bool)]
    check("nonneg fit: every off-diagonal >= 0 (qubosolver's strict '< 0' test)",
          pn["analog_representable"] and (offn >= 0).all())
    check("nonneg fit: symmetric, diagonal still free (some negative D allowed)",
          np.allclose(Qn, Qn.T))
    fn = fit_report(net, cands, costs, Qn, cn, 50.0, np.random.default_rng(11))
    check(f"nonneg fit still ranks better than anchor (rank corr {fn['rank_corr']:.3f} "
          f"vs {fa['rank_corr']:.3f}); free fit {fr['rank_corr']:.3f} is the reference",
          fn["rank_corr"] > fa["rank_corr"])

    print(f"\n  {'✓ Self-test passed.' if ok else '✗ SELF-TEST FAILED'}")
    return 0 if ok else 1


# -------------------------------------------------------------------------------- main --

def load_instance(name_or_json):
    """Reload a generated instance: returns (meta, net, cands, costs, Q).

    Verifies the saved QUBO against the sha256 stamped into the problem JSON, so an
    evaluator or submitter can never be pointed at a mismatched pair (the failure
    mode that produced a spurious '0% valid' in the earlier logistics study).
    """
    p = Path(name_or_json)
    meta_path = p if p.suffix == ".json" else Path(f"{name_or_json}_problem.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    qubo_path = meta_path.with_name(meta_path.name.replace("_problem.json", "_qubo.npy"))
    Q = np.load(qubo_path)
    sha = hashlib.sha256(Q.tobytes()).hexdigest()[:16]
    if sha != meta["qubo_sha256_16"]:
        raise ValueError(f"{qubo_path.name} sha256 {sha} != stamped "
                         f"{meta['qubo_sha256_16']} — mismatched QUBO/problem pair")
    net = Network(meta["n_bus"],
                  [(l["from"], l["to"], l["b"], l["rating"])
                   for l in meta["existing_lines"]],
                  np.array(meta["injections"]), meta["scenario_weights"])
    cands = [(c["from"], c["to"], c["b"], c["rating"]) for c in meta["candidates"]]
    costs = np.array([c["cost"] for c in meta["candidates"]])
    return meta, net, cands, costs, Q


def generate(args):
    rng = np.random.default_rng(args.seed + 2)
    if args.network == "synth":
        net, used, d = synthetic_network(args.n_bus, args.n_chords, args.scenarios,
                                         args.seed, args.rating_frac, args.stress)
        cands, costs = make_candidates(net, used, d, args.n_cand, args.seed)
    else:
        net, cands, costs = from_pandapower(args.network, args.n_cand,
                                            args.scenarios, args.rating_frac,
                                            args.stress, args.seed,
                                            parallel_cost=args.parallel_cost)
    n_bus = net.n_bus
    Q, const, parts = build_qubo(net, cands, costs, args.lam, args.mu, args.budget,
                                 method=args.fit, n_train=args.n_train, rng=rng,
                                 offdiag=args.offdiag, workers=args.workers)
    fit = fit_report(net, cands, costs, Q, const, args.lam, rng, mu=args.mu,
                     budget=args.budget)
    stats = qubo_stats(Q)

    print(f"Instance '{args.name}' [{args.network}]: {n_bus} buses, "
          f"{len(net.lines)} lines, "
          f"{len(cands)} candidates -> {len(cands)} qubits, "
          f"{args.scenarios} scenarios, lambda={args.lam:g}, "
          f"fit={args.fit}/{args.offdiag}")
    print(f"  Base congestion C0:      {parts['C0']:.4f}  "
          f"({'no congestion — DEGENERATE, reseed' if parts['C0'] < 1e-9 else 'ok'})")
    print(f"  Pairwise fit:            RMSE {fit['rmse']:.4f} "
          f"({100 * fit['rmse_over_spread']:.1f}% of spread), "
          f"max {fit['max_abs_err']:.4f}, rank corr {fit['rank_corr']:.3f}")
    print(f"  Off-diagonal:            density {100 * stats['offdiag_density']:.0f}%, "
          f"sign-mix {100 * stats['offdiag_sign_mix']:.0f}% positive "
          f"(frustration indicator); {stats['offdiag_negative']} negative -> "
          f"{'analog-representable (Pasqal OK)' if stats['analog_representable'] else 'Pasqal analog will REJECT (use --offdiag nonneg); gate-based QAOA unaffected'}")

    gt = None
    n_c = len(cands)
    if n_c <= GROUND_TRUTH_MAX:
        F = exact_objective(net, cands, costs, args.lam, args.mu, args.budget)
        subsets = [tuple(s) for k in range(n_c + 1)
                   for s in itertools.combinations(range(n_c), k)]
        vals = np.array([F(s) for s in subsets])
        order = np.argsort(vals)
        opt_sel, opt_val = subsets[order[0]], float(vals[order[0]])
        qv, qx = brute_force_qubo(Q, const)
        q_sel = tuple(i for i in range(n_c) if qx[i])
        gt_rank = int(np.where([subsets[o] == q_sel for o in order])[0][0])
        # the two metrics that matter for a SAMPLER (argmin identity does not):
        # how good, in TRUE cost, is what the surrogate points at — and how deep in
        # the surrogate's energy ordering the true optimum sits (top-k reachability).
        gap_pct = 100.0 * (F(q_sel) - opt_val) / abs(opt_val) if opt_val else 0.0
        qe = np.array([qubo_value(Q, const,
                                  [1 if i in s else 0 for i in range(n_c)])
                       for s in subsets])
        opt_qrank = int((qe < qe[subsets.index(opt_sel)]).sum())
        gt = {"optimal_subset": list(opt_sel), "optimal_value": opt_val,
              "build_none_value": float(F(())),
              "qubo_argmin_subset": list(q_sel),
              "qubo_argmin_true_value": float(F(q_sel)),
              "qubo_argmin_true_gap_pct": gap_pct,
              "qubo_argmin_rank_in_truth": gt_rank,
              "optimum_rank_in_qubo_energy": opt_qrank,
              "qubo_matches_truth": q_sel == opt_sel,
              "n_subsets": len(subsets)}
        print(f"  Ground truth (exhaustive {len(subsets)} exact DC solves):")
        print(f"    optimal build set {list(opt_sel)}  F = {opt_val:.4f}  "
              f"(build-nothing F = {gt['build_none_value']:.4f})")
        print(f"    QUBO argmin {list(q_sel)} -> true rank {gt_rank}, "
              f"true gap {gap_pct:+.2f}% "
              f"{'✓ MATCHES optimum' if gt['qubo_matches_truth'] else ''}")
        print(f"    true optimum sits at QUBO-energy rank {opt_qrank} "
              f"(top-k a sampler must reach)")
        if opt_sel in ((), tuple(range(n_c))):
            print(f"    ⚠ DEGENERATE optimum (build none/all) — retune lam or reseed.")

        # signal screen over the top decile band (same check as in the earlier logistics study)
        band = float(np.ptp(vals[order[: max(2, len(vals) // 10)]]))
        print(f"    signal (top-decile band / max|Q|): {band / stats['max_abs']:.4f}")

    out_q = Path(f"{args.name}_qubo.npy")
    np.save(out_q, Q)
    meta = {
        "schema": SCHEMA_VERSION, "name": args.name, "seed": args.seed,
        "network": args.network,
        "n_bus": n_bus, "n_candidates": len(cands),
        "scenarios": args.scenarios, "lambda": args.lam,
        "mu": args.mu, "budget": args.budget,
        "rating_frac": args.rating_frac, "stress": args.stress,
        "qubo_constant": const, "base_congestion": parts["C0"],
        "fit_method": parts["method"], "offdiag_constraint": parts["offdiag"],
        "parallel_cost": args.parallel_cost,
        "candidates": [{"from": int(i), "to": int(j), "b": float(b),
                        "rating": float(r), "cost": float(c)}
                       for (i, j, b, r), c in zip(cands, costs)],
        "existing_lines": [{"from": int(i), "to": int(j), "b": float(b),
                            "rating": float(r)} for (i, j, b, r) in net.lines],
        "injections": net.injections.tolist(),
        "scenario_weights": net.weights.tolist(),
        "fit": fit, "qubo_stats": stats, "ground_truth": gt,
        "qubo_sha256_16": hashlib.sha256(Q.tobytes()).hexdigest()[:16],
    }
    out_m = Path(f"{args.name}_problem.json")
    with open(out_m, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  ✓ Saved {out_q} and {out_m}  (Q sha256 {meta['qubo_sha256_16']})")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--name", default="instA")
    p.add_argument("--network", default="synth",
                   choices=("synth",) + PP_NETWORKS,
                   help="grid source: 'synth' (built-in generator) or a pandapower "
                        "reference network (case33bw / cigre_mv / mv_oberrhein); "
                        "pandapower networks ignore --n-bus/--n-chords")
    p.add_argument("--n-bus", type=int, default=12)
    p.add_argument("--n-cand", type=int, default=12)
    p.add_argument("--n-chords", type=int, default=3)
    p.add_argument("--scenarios", type=int, default=3)
    p.add_argument("--lam", type=float, default=50.0,
                   help="congestion weight; sweep it to trace the cost-vs-congestion "
                        "Pareto front (challenge ref [6] framing)")
    p.add_argument("--mu", type=float, default=0.0, help="soft budget weight (off = 0)")
    p.add_argument("--budget", type=int, default=0)
    p.add_argument("--n-train", type=int, default=None,
                   help="training subsets for --fit regress (default 4x n_params)")
    p.add_argument("--fit", choices=["regress", "anchor"], default="regress",
                   help="surrogate fit: 'regress' (LSQ over all subset sizes, default) "
                        "or 'anchor' (|S|<=2 interpolation — poor, kept as baseline)")
    p.add_argument("--parallel-cost", choices=["unit", "length"], default="unit",
                   help="build cost of parallel-line candidates: 'unit' (default; the frozen "
                        "instC179 rule — every mv_oberrhein candidate costs 1.0) or 'length' "
                        "(the paralleled line's km; heterogeneous costs = hardness axis)")
    p.add_argument("--offdiag", choices=["free", "nonneg"], default="free",
                   help="'free' (default): sign-mixed couplings, fine for QAOA/classical. "
                        "'nonneg': constrain every off-diagonal >= 0 (bounded LSQ) so the "
                        "QUBO is representable on Pasqal's analog neutral-atom solver, "
                        "whose Rydberg interaction is purely repulsive — qubosolver rejects "
                        "sign-mixed Q. The fidelity cost is measured and stamped like any fit.")
    p.add_argument("--rating-frac", type=float, default=0.9)
    p.add_argument("--stress", type=float, default=1.45)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--workers", type=int, default=1,
                   help="processes for the surrogate training DC solves (result is "
                        "bit-identical for any value; use many on the H200 box)")
    args = p.parse_args()
    return selftest() if args.selftest else generate(args)


if __name__ == "__main__":
    sys.exit(main())
