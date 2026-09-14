"""
Problem-agnostic QUBO checks for the Pasqal analog pipeline.

Nothing here knows about TSP. This is the distilled, reusable outcome of our earlier
logistics (TSP) study on Pasqal hardware — import it for any later QUBO problem, such as
the E.ON grid expansion problem, rather than rediscovering the same traps.

The three findings worth carrying forward, each earned the hard way:

1. ASYMMETRY IS SILENT AND FATAL.  `x^T Q x == x^T Q^T x`, so an asymmetric Q gives the
   right objective value and every classical check passes. The quantum path does NOT
   error — it reads off-diagonals to place atoms and builds a garbage register. A 25-var
   TSP with max|Q - Q^T| = 20.0 returned 0 valid solutions out of 223; symmetrising it
   took the same problem on the same device to 100% valid. Run `check_symmetry` before
   every submission.

2. `--no-dmm` IS A NO-OP WHEN THE DIAGONAL IS UNIFORM.  `qubosolver/pipeline/drive.py`
   computes `spread = d_max - d_min` over `d = -0.5 * diag(Q_normalised)` and sets
   `use_dmm = False` itself when `spread <= 1e-15`. So on a QUBO whose diagonal is
   constant the flag changes nothing in either direction, and any A/B built on it is
   measuring noise. Run `check_dmm_effective` to find out whether the flag can matter.

3. FEASIBLE IS NOT OPTIMISED.  A sampler that covers the whole feasible set "finds the
   optimum" at +0.00% gap while carrying no optimisation signal at all. On a small
   instance that is indistinguishable from success by gap alone. `sampling_quality`
   separates the two: rank-correlate solution cost against shot count and compare the
   expected sampled cost to a uniform baseline. Check this BEFORE believing a gap number.

Deliberately NOT a go/no-go gate: the geometric embeddability statistics in
`coupling_stats`. Neutral atoms realise couplings as U_ij = C6/r_ij^6, which really does
mean N atoms offer only 2N spatial degrees of freedom for N(N-1)/2 couplings, that
U_ij > 0 always so exactly-zero couplings are unsatisfiable, and that dynamic range maps
to distance only as the 6th root. All true — and all of it predicted that the 25-var TSP
could not work, which turned out to be wrong. Treat these numbers as descriptive context,
never as grounds to reject a formulation without testing it.
"""

import numpy as np


# ----------------------------------------------------------------------------------
# Finding 1 — symmetry
# ----------------------------------------------------------------------------------

def check_symmetry(Q, tol=1e-9):
    """
    Report whether Q is symmetric, and by how much it is not.

    Returns (ok, max_asymmetry, message).
    """
    Q = np.asarray(Q, dtype=float)
    max_asym = float(np.abs(Q - Q.T).max()) if Q.size else 0.0
    ok = max_asym <= tol
    if ok:
        return True, max_asym, "✓ Q is symmetric"
    return (
        False,
        max_asym,
        f"✗ Q is ASYMMETRIC (max|Q - Q.T| = {max_asym:g}). The objective is unaffected "
        f"(x^T Q x == x^T Q^T x) so this will not error, but the quantum backend reads "
        f"off-diagonals to place atoms and will build a wrong register. Fix with "
        f"Q = (Q + Q.T) / 2 in the generator.",
    )


def symmetrize(Q):
    """Exact fix: (Q + Q.T)/2 leaves x^T Q x unchanged for every x."""
    Q = np.asarray(Q, dtype=float)
    return (Q + Q.T) / 2


# ----------------------------------------------------------------------------------
# Finding 2 — whether the DMM can do anything at all
# ----------------------------------------------------------------------------------

def check_dmm_effective(Q, tol=1e-15):
    """
    Will the DMM actually be used, or will the library silently switch it off?

    Mirrors `HeuristicDriveShaper.generate` in qubosolver/pipeline/drive.py: the local
    detuning targets are `d = -0.5 * diag(Q / max|Q|)`, and the DMM is disabled when
    `d.max() - d.min() <= 1e-15`. When disabled, the drive falls back to a single global
    detuning `delta_g(T) = mean(d)` and the diagonal carries no per-atom information.

    Returns (dmm_can_matter, spread, message).
    """
    Q = np.asarray(Q, dtype=float)
    scale = np.abs(Q).max()
    if scale == 0:
        return False, 0.0, "✗ Q is all zeros"

    d = -0.5 * np.diag(Q / scale)
    spread = float(d.max() - d.min())

    if spread <= tol:
        return (
            False,
            spread,
            f"⚠ DMM cannot matter: diagonal is uniform (spread = {spread:g}). The library "
            f"disables the DMM itself, so --no-dmm / --dmm change nothing here. Any A/B "
            f"comparison built on that flag for this QUBO is measuring noise.",
        )
    return (
        True,
        spread,
        f"✓ DMM is live: diagonal spread = {spread:.4f} (d in [{d.min():.4f}, {d.max():.4f}]). "
        f"With DMM the final global detuning is d_max = {d.max():.4f}; without it, "
        f"mean(d) = {d.mean():.4f} — which also sets how many atoms get excited.",
    )


# ----------------------------------------------------------------------------------
# Descriptive only — see the module docstring on why this is not a gate
# ----------------------------------------------------------------------------------

def coupling_stats(Q):
    """
    Geometric-embeddability statistics. DESCRIPTIVE CONTEXT ONLY.

    These numbers said the 25-variable TSP was impossible; it subsequently ran at 100%
    feasibility. Report them, do not gate on them.
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    iu = np.triu_indices(n, k=1)
    pairs = Q[iu]
    nz = np.abs(pairs[np.abs(pairs) > 1e-9])
    dof = 2 * n
    ratio = float(nz.max() / nz.min()) if nz.size else float("nan")
    return {
        "n_variables": n,
        "n_pairs": int(pairs.size),
        "nonzero_pairs": int(nz.size),
        "zero_pairs": int(pairs.size - nz.size),
        "spatial_dof": dof,
        "over_determined_factor": round(pairs.size / dof, 2) if dof else float("nan"),
        "coupling_ratio": round(ratio, 2),
        "distance_ratio_needed": round(ratio ** (1 / 6), 3) if nz.size else float("nan"),
        "diagonal_spread": float(np.ptp(np.diag(Q))),
    }


def preflight(Q, verbose=True):
    """
    Run every pre-submission check. Call this before spending cloud time.

    Returns (ok_to_submit, report). `ok_to_submit` is False only for symmetry, which is
    a genuine correctness bug; everything else is advisory.
    """
    sym_ok, max_asym, sym_msg = check_symmetry(Q)
    dmm_live, spread, dmm_msg = check_dmm_effective(Q)
    stats = coupling_stats(Q)

    report = {
        "symmetric": sym_ok,
        "max_asymmetry": max_asym,
        "dmm_can_matter": dmm_live,
        "diagonal_spread_normalised": spread,
        "coupling_stats": stats,
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"QUBO preflight ({stats['n_variables']} variables)")
        print(f"{'='*70}")
        print(f"  {sym_msg}")
        print(f"  {dmm_msg}")
        print(f"  couplings {stats['nonzero_pairs']}/{stats['n_pairs']} nonzero, "
              f"{stats['zero_pairs']} require exactly zero, ratio {stats['coupling_ratio']}x "
              f"(descriptive only — not predictive of success)")

    return sym_ok, report


# ----------------------------------------------------------------------------------
# Finding 3 — optimisation vs enumeration
# ----------------------------------------------------------------------------------

def _avg_ranks(values):
    """Ranks with ties averaged, so tied costs don't fabricate a correlation."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < order.size:
        j = i
        while j + 1 < order.size and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _advantage_null_p(costs, shots, pool_costs=None, trials=20000, seed=0):
    """
    P(expected sampled cost <= observed | shots distributed uniformly over the solutions).

    This is the test that matters, and it is far more powerful than the rank correlation,
    because it uses how much cheaper the preferred solutions are rather than only their
    order. On the `custom_3x4` run the rank correlation gave p = 0.056 ("not significant")
    while this gave p < 1e-5 on the same 95 shots — the run concentrated 70% of its shots on
    the cheapest 6 of 14 solutions, which a rank statistic largely throws away.

    pool_costs: the FULL feasible set if the caller knows it (the stronger null — it does not
    credit the sampler for the set it happened to reach). Defaults to the solutions found.
    """
    n_shots = int(shots.sum())
    if n_shots <= 0 or len(costs) < 2:
        return 1.0
    observed = float((np.asarray(costs) * np.asarray(shots)).sum() / n_shots)
    pool = np.asarray(pool_costs if pool_costs is not None else costs, dtype=float)
    if len(pool) < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n_shots, np.full(len(pool), 1.0 / len(pool)), size=trials)
    sim = (draws * pool).sum(axis=1) / n_shots
    return float((sim <= observed + 1e-12).mean())


def _uniform_null_p(costs, shots, observed_rho, trials=20000, seed=0):
    """
    P(rank correlation <= observed | shots distributed uniformly over these solutions).

    The null is multinomial rather than a label permutation, because shot-count noise is
    the thing that actually fools us: at 14 shots the counts are near-random whatever the
    dynamics did, and a permutation test would return the same p-value for 14 shots as for
    14,000. Conservative by construction — the null uses only the solutions that were
    actually found, so a run that missed part of the feasible set is not penalised twice.
    """
    n_shots = int(shots.sum())
    k = len(costs)
    if n_shots <= 0 or k < 3:
        return 1.0
    rng = np.random.default_rng(seed)
    cost_ranks = _avg_ranks(costs)
    draws = rng.multinomial(n_shots, np.full(k, 1.0 / k), size=trials)
    hits = 0
    counted = 0
    for d in draws:
        seen = d > 0
        if seen.sum() < 3:            # too few to rank-correlate; not evidence either way
            continue
        counted += 1
        r = np.corrcoef(_avg_ranks(cost_ranks[seen]), _avg_ranks(d[seen]))[0, 1]
        if r <= observed_rho + 1e-12:
            hits += 1
    return float(hits / counted) if counted else 1.0


def sampling_quality(solutions, optimal_cost=None, feasible_pool_costs=None):
    """
    Did the sampler PREFER good solutions, or just cover the feasible set?

    Args:
        solutions: iterable of (label, cost, shots) for DISTINCT feasible solutions.
            The caller is responsible for collapsing symmetry-equivalent encodings
            first (e.g. a TSP cycle has n rotations x 2 directions); counting them
            separately inflates diversity and distorts the baseline.
        optimal_cost: known optimum, for gap reporting. Defaults to the best sampled.

    Returns None if fewer than two distinct solutions were found (nothing to compare),
    otherwise a dict whose two decisive fields are:
        cost_shot_rank_corr    strongly negative => cheaper solutions sampled more often
        advantage_over_uniform_pct   how much better than blind sampling of what it found
    """
    rows = [(str(label), float(cost), float(shots)) for label, cost, shots in solutions]
    rows = [r for r in rows if r[2] > 0]
    if len(rows) < 2:
        return None

    costs = np.array([r[1] for r in rows])
    shots = np.array([r[2] for r in rows])
    best = float(costs.min())
    opt = float(optimal_cost) if optimal_cost is not None else best

    rho = float(np.corrcoef(_avg_ranks(costs), _avg_ranks(shots))[0, 1])
    expected = float((costs * shots).sum() / shots.sum())
    # Baseline and its significance test must use the SAME null, or the reported effect size
    # and p-value describe different questions. Prefer the full feasible set when the caller
    # knows it: uniform over only the solutions found is the weaker claim.
    pool = np.asarray(feasible_pool_costs, dtype=float) \
        if feasible_pool_costs is not None else costs
    uniform = float(pool.mean())

    # A rank correlation is a point estimate over a handful of solutions, and shot counts
    # are themselves noisy. On a run with 14 feasible shots across 5 solutions, rho = -0.456
    # looks like optimisation and is produced by uniform sampling roughly one time in five.
    # So test it: resample the observed feasible shots uniformly across the solutions found
    # and ask how often chance alone beats what we measured.
    p_value = _uniform_null_p(costs, shots, rho)

    # The decisive test. The rank correlation only sees order; this sees how much cheaper the
    # preferred solutions actually are, and is much more powerful on a modest shot budget.
    # Where the two disagree, trust this one — see _advantage_null_p.
    p_advantage = _advantage_null_p(costs, shots, feasible_pool_costs)

    # Optimising requires the cost advantage to be BOTH statistically real and large enough
    # to matter. Significance alone is not enough: at 353 shots the cpu_small run cleared
    # p = 0.009 on a +0.4% advantage while its most-sampled solution was the worst feasible
    # one. Effect size alone is not enough either — that was the 14-shot false positive.
    advantage_pct = 100.0 * (uniform - expected) / uniform if uniform else 0.0
    MIN_EFFECT_PCT = 5.0
    optimising = p_advantage <= 0.05 and advantage_pct >= MIN_EFFECT_PCT
    enumerating = not optimising

    # How large an advantage could this many feasible shots have detected at all? Without
    # this, a null result reads as "no effect" when it may only mean "no power" — the
    # custom_2x6 c=15 run reported +2.4% n.s. on 92 shots, but 92 shots cannot resolve
    # anything below ~7%, so that verdict was a statement about the shot budget, not the
    # sampler. Reported alongside every null so the two are never confused again.
    n_feas = float(shots.sum())
    sd = float(pool.std(ddof=1)) if pool.size > 1 else 0.0
    mde_pct = (100.0 * 1.645 * sd / np.sqrt(n_feas) / uniform
               if n_feas > 0 and uniform else float("nan"))
    # Only meaningful when the measured effect is not itself significant.
    underpowered = bool(p_advantage > 0.05 and advantage_pct < mde_pct)

    return {
        "distinct_solutions": len(rows),
        "feasible_shots": int(shots.sum()),
        "cost_shot_rank_corr": rho,
        "rank_corr_p_value": p_value,
        "advantage_p_value": p_advantage,
        "advantage_null": "full feasible set" if feasible_pool_costs is not None
                          else "solutions found",
        "min_effect_pct": MIN_EFFECT_PCT,
        "detectable_advantage_pct": mde_pct,
        "underpowered": underpowered,
        "_pool_sd": sd,
        "significant": bool(p_advantage <= 0.05),
        "expected_sampled_cost": expected,
        "uniform_baseline_cost": uniform,
        "best_sampled_cost": best,
        "expected_gap_pct": 100.0 * (expected - opt) / opt if opt else float("nan"),
        "uniform_gap_pct": 100.0 * (uniform - opt) / opt if opt else float("nan"),
        "best_gap_pct": 100.0 * (best - opt) / opt if opt else float("nan"),
        "advantage_over_uniform_pct": 100.0 * (uniform - expected) / uniform if uniform else 0.0,
        "is_enumerating": bool(enumerating),
        "solutions": [
            {"label": l, "cost": c, "shots": int(s)}
            for l, c, s in sorted(rows, key=lambda r: r[1])
        ],
    }


def print_sampling_quality(sq, show=None):
    """Human-readable verdict for `sampling_quality`. `show` caps the solution table."""
    if sq is None:
        print("\n(only one distinct feasible solution — no sampling-quality signal)")
        return

    print(f"\n{'='*70}")
    print(f"Sampling quality — is the objective actually being optimised?")
    print(f"{'='*70}")
    print(f"  Distinct feasible solutions:    {sq['distinct_solutions']}")
    print(f"  Feasible shots behind them:     {sq['feasible_shots']:,}")
    print(f"  Cost-vs-shots rank correlation: {sq['cost_shot_rank_corr']:+.3f}   "
          f"(strongly negative = optimising)")
    print(f"    vs uniform-sampling null:     p = {sq['rank_corr_p_value']:.3f}   "
          f"{'SIGNIFICANT' if sq['rank_corr_p_value'] <= 0.05 else 'NOT significant'}"
          f"   (weak test — order only)")
    print(f"  Expected sampled cost:          {sq['expected_sampled_cost']:.4f}  "
          f"({sq['expected_gap_pct']:+.2f}% vs optimal)")
    print(f"  Uniform over {sq['advantage_null']:<18}: {sq['uniform_baseline_cost']:.4f}  "
          f"({sq['uniform_gap_pct']:+.2f}% vs optimal)")
    print(f"  Advantage over uniform:         {sq['advantage_over_uniform_pct']:+.1f}%")
    print(f"    vs uniform-sampling null:     p = {sq['advantage_p_value']:.5f}   "
          f"{'SIGNIFICANT' if sq['advantage_p_value'] <= 0.05 else 'NOT significant'}"
          f"   [null: {sq['advantage_null']}]")
    print(f"    (this is the decisive test — it uses cost magnitudes, not just order)")

    rows = sq["solutions"] if show is None else sq["solutions"][:show]
    best = sq["best_sampled_cost"]
    print(f"\n  {'cost':>10} {'gap':>8} {'shots':>7}  solution")
    for r in rows:
        gap = 100.0 * (r["cost"] - best) / best if best else 0.0
        print(f"  {r['cost']:10.4f} {gap:+7.2f}% {r['shots']:7d}  {r['label']}")
    if show is not None and len(sq["solutions"]) > show:
        print(f"  ... and {len(sq['solutions']) - show} more")

    if sq["feasible_shots"] < 30:
        print(f"\n  ⚠ ONLY {sq['feasible_shots']} FEASIBLE SHOTS. Every number above is")
        print(f"    dominated by shot noise and none of them supports a conclusion in")
        print(f"    either direction. Raise the shot count or fix feasibility first.")

    adv, p_adv = sq["advantage_over_uniform_pct"], sq["advantage_p_value"]
    if not sq["is_enumerating"]:
        print(f"\n  ✓ OPTIMISING. Cheaper solutions are sampled more often than chance "
              f"allows:")
        print(f"    advantage {adv:+.1f}% at p = {p_adv:.5f}. The objective is driving the "
              f"dynamics.")
        if sq["rank_corr_p_value"] > 0.05:
            print(f"    Note the rank correlation alone would have missed this "
                  f"(p = {sq['rank_corr_p_value']:.3f}) —")
            print(f"    it discards cost magnitudes, which is where this signal lives.")
    elif p_adv <= 0.05:
        mde = sq.get("detectable_advantage_pct", float("nan"))
        print(f"\n  ⚠ REAL BUT TOO SMALL. The {adv:+.1f}% cost advantage is statistically "
              f"solid (p = {p_adv:.5f}),")
        print(f"    and this run resolves ~{mde:.1f}%, so it is not a large-n artefact — the "
              f"objective")
        print(f"    IS influencing the sampler. But {adv:+.1f}% is below the "
              f"{sq['min_effect_pct']:.0f}% bar for a result that")
        print(f"    would matter at scale. Treat the optimum as covered rather than sought,")
        print(f"    and do NOT relax the threshold to accommodate a near miss.")
    elif sq.get("underpowered"):
        thr = sq["min_effect_pct"]
        print(f"\n  ⚠ NO EFFECT DETECTED — but this run could not have detected one.")
        print(f"    Advantage {adv:+.1f}% (p = {p_adv:.3f}), and {sq['feasible_shots']:,} "
              f"feasible shots only")
        print(f"    resolve advantages above ~{sq['detectable_advantage_pct']:.1f}%. That is a "
              f"statement about the")
        print(f"    shot budget, NOT about the sampler: a real {thr:.0f}% effect would be "
              f"invisible here.")
        sd, base = sq.get("_pool_sd"), sq["uniform_baseline_cost"]
        if sd and base:
            need = int(np.ceil((1.645 * sd / (thr / 100.0 * base)) ** 2))
            print(f"    Resolving the {thr:.0f}% threshold needs ~{need:,} feasible shots. "
                  f"Fix feasibility first.")
    else:
        print(f"\n  ⚠ NOT optimising (advantage {adv:+.1f}%, p = {p_adv:.3f}; rank p = "
              f"{sq['rank_corr_p_value']:.3f}).")
        print(f"    Not distinguishable from uniform sampling, and the run had the power to")
        print(f"    see a {MIN_EFFECT_PCT:.0f}% effect (resolves ~"
              f"{sq['detectable_advantage_pct']:.1f}%), so any optimum here was")
        print(f"    found by ENUMERATION — which only works while the feasible set stays")
        print(f"    tiny. A good gap number under these conditions means nothing.")


def feasibility_report(samples, decode):
    """
    Split raw samples into feasible and infeasible using a caller-supplied decoder.

    Args:
        samples: {bitstring: shots}
        decode:  bitstring -> solution object, or None when constraints are violated.

    Feasibility and optimality are separate questions; this answers only the first.
    """
    feasible_shots = infeasible_shots = 0
    feasible, infeasible = {}, {}
    for bs, shots in samples.items():
        sol = decode(bs)
        if sol is None:
            infeasible[bs] = shots
            infeasible_shots += shots
        else:
            feasible[bs] = (sol, shots)
            feasible_shots += shots

    total = feasible_shots + infeasible_shots
    return {
        "total_shots": total,
        "feasible_shots": feasible_shots,
        "infeasible_shots": infeasible_shots,
        "feasible_pct": 100.0 * feasible_shots / total if total else 0.0,
        "unique_total": len(samples),
        "unique_feasible": len(feasible),
        "feasible": feasible,
        "infeasible": infeasible,
    }
