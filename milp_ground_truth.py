"""
Exact ground truth for a grid-expansion instance via Gurobi or HiGHS — and the classical-hardness screen.

Solves the TRUE problem (not the QUBO surrogate) as a MILP:

    min  cost^T x  +  lambda * sum_s w_s sum_l overload_{s,l}   (+ mu-budget term)
    s.t. DC power flow per scenario, candidate lines switched by big-M,
         overload_{s,l} >= |f_{s,l}| - rating_l,  overload >= 0.

The overload slacks make congestion exactly the clip(|f|-r, 0) of the simulator, so the
MILP optimum IS the optimum of grid_expansion_qubo.exact_objective — and the script
proves it: the reported objective is cross-checked against exact_objective(x*) and the
run aborts loudly on any mismatch (that check also guards the big-M choice).

Big-M (since 2026-09-07 evening): rigorous per-scenario, per-line bounds from the base
network's effective resistances (maximum principle + Rayleigh monotonicity, see
_flow_bounds). They are valid for every plan, so no optimum can be cut off, and they are
orders of magnitude tighter than the earlier "50x observed spread" heuristic — which had
left HiGHS with 22-50% gaps after an hour on the 120-candidate instC179.

--qubo additionally solves the QUBO surrogate as a MIQP, giving the surrogate's argmin
at sizes brute force cannot reach — the surrogate-quality metric at B/C scale.

Two solver backends build the SAME model:
  * gurobi  — pip gurobipy carries a size-limited licence (~2000 vars/constrs): instB33
              fits, instC179 (≈2500 vars) does not.
  * highs   — scipy.optimize.milp (HiGHS, bundled with scipy ≥1.9, no licence). This is
              the backend for the H200 box, which has no Gurobi licence (checked
              2026-09-07). Linear objective only: the mu-budget quadratic needs gurobi.
  --solver auto (default) uses gurobi when importable and falls back to highs.

--lam overrides the instance's lambda for the TRUE problem only (network, candidates and
costs unchanged) — the cost-vs-congestion Pareto sweep and the hardness probe at zero
instance-generation cost. Output goes to <name>_ground_truth_lam<value>.json so the
stamped ground truth of the instance itself is never overwritten.

Usage:
    python milp_ground_truth.py --instance instB33 --qubo
    python milp_ground_truth.py --instance instC179 --time-limit 3600 --solver highs   # H200
    python milp_ground_truth.py --instance instC179 --lam 2 --solver highs             # Pareto point
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from grid_expansion_qubo import load_instance, exact_objective   # noqa: E402


def _big_m_heuristic(net, cands):
    """Legacy big-M (used for every run before 2026-09-07 evening): 50x the observed
    angle spread of the two extreme plans. Valid in practice but so loose that HiGHS
    left 22-50% gaps after an hour on instC179. Kept only for the comparison print."""
    _, f0 = net.flows()
    _, f1 = net.flows(extra_lines=cands)
    theta_max = 0.0
    for f, lines in ((f0, net.lines), (f1, net.lines + list(cands))):
        for (i, j, b, _r), col in zip(lines, np.abs(f).max(axis=0)):
            theta_max = max(theta_max, col / b)
    return 50.0 * max(theta_max, 1e-3)


def _flow_bounds(net, cands, margin=1.01):
    """Rigorous per-scenario angle / flow bounds that hold for EVERY build plan.

    DC flow is a resistive network with conductances b. By superposition
    theta = sum_k p_k phi^k with B phi^k = e_k - e_0 (unit injection at bus k, withdrawn
    at the slack). Maximum principle: 0 <= phi^k_i <= R(k,0) for every bus i; reciprocity
    plus the maximum principle: |phi^k_i - phi^k_j| <= min(R(k,0), R(i,j)), where R is
    the effective resistance. Rayleigh monotonicity: adding a candidate line (a positive
    conductance) never increases any effective resistance, so the BASE network's R
    bounds every plan. Hence, for every plan and scenario s,
        |theta_i|            <= sum_k |p_sk| R(k,0)               =: TH[s]
        |theta_i - theta_j|  <= sum_k |p_sk| min(R(k,0), R(i,j)) =: D[s, line(i,j)]
        |f_line|             <= b_line * D[s, line].
    For a candidate that parallels an existing line (all of instC179) R(i,j) <= 1/b_exist,
    which is what makes these bounds orders of magnitude tighter than the heuristic.
    `margin` only absorbs floating-point slack; the exact_objective cross-check in main()
    still guards the result. Returns (TH (n_scen,), D (n_scen, n_existing + n_cand))."""
    n_bus = net.n_bus
    B = np.zeros((n_bus, n_bus))
    for i, j, b, _r in net.lines:
        B[i, i] += b
        B[j, j] += b
        B[i, j] -= b
        B[j, i] -= b
    G = np.zeros((n_bus, n_bus))
    G[1:, 1:] = np.linalg.inv(B[1:, 1:])              # Laplacian grounded at slack 0
    d = np.diag(G).copy()                              # R(k, 0)
    R = d[:, None] + d[None, :] - 2.0 * G              # R(i, j)
    P = np.abs(np.asarray(net.injections, dtype=float))
    P[:, 0] = 0.0                                      # slack balances; not a source
    TH = P @ d
    all_lines = list(net.lines) + list(cands)
    D = np.empty((P.shape[0], len(all_lines)))
    for k, (i, j, _b, _r) in enumerate(all_lines):
        D[:, k] = P @ np.minimum(d, R[i, j])
    return margin * TH, margin * D


def solve_true_milp_highs(meta, net, cands, costs, time_limit=600, verbose=False,
                          lam=None):
    """Same MILP as solve_true_milp, assembled as sparse matrices for scipy's HiGHS."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import coo_matrix

    n_bus = meta["n_bus"]
    lam = meta["lambda"] if lam is None else lam
    mu, budget = meta["mu"], meta["budget"]
    if mu and budget:
        raise ValueError("mu-budget term is quadratic — use --solver gurobi")
    n_scen = len(net.weights)
    n_c = len(cands)
    all_lines = list(net.lines) + list(cands)
    nE, nL = len(net.lines), len(all_lines)
    TH, D = _flow_bounds(net, cands)
    bvec = np.array([b for (_i, _j, b, _r) in all_lines])
    M = D * bvec[None, :]                              # (n_scen, nL) flow bounds

    # variable layout: x | th[s,i] | f[s,k] | ov[s,k]
    o_th = n_c
    o_f = o_th + n_scen * n_bus
    o_ov = o_f + n_scen * nL
    N = o_ov + n_scen * nL
    ith = lambda s, i: o_th + s * n_bus + i          # noqa: E731
    iff = lambda s, k: o_f + s * nL + k              # noqa: E731
    iov = lambda s, k: o_ov + s * nL + k             # noqa: E731

    lb = np.full(N, -np.inf)
    ub = np.full(N, np.inf)
    lb[:n_c], ub[:n_c] = 0.0, 1.0
    for s in range(n_scen):
        lb[o_th + s * n_bus: o_th + (s + 1) * n_bus] = -TH[s]
        ub[o_th + s * n_bus: o_th + (s + 1) * n_bus] = TH[s]
        lb[ith(s, 0)] = ub[ith(s, 0)] = 0.0            # slack angle
        lb[o_f + s * nL: o_f + (s + 1) * nL] = -M[s]
        ub[o_f + s * nL: o_f + (s + 1) * nL] = M[s]
    lb[o_ov:] = 0.0
    integrality = np.zeros(N)
    integrality[:n_c] = 1

    rows, cols, vals, c_lo, c_hi = [], [], [], [], []
    r = 0

    def add(entries, lo, hi):
        nonlocal r
        for col, v in entries:
            rows.append(r)
            cols.append(col)
            vals.append(v)
        c_lo.append(lo)
        c_hi.append(hi)
        r += 1

    for s in range(n_scen):
        for k, (i, j, b, rating) in enumerate(all_lines):
            base = [(iff(s, k), 1.0), (ith(s, i), -b), (ith(s, j), b)]   # f - b(th_i-th_j)
            if k < nE:
                add(base, 0.0, 0.0)
            else:
                xc = k - nE
                Mk = M[s, k]
                add(base + [(xc, Mk)], -np.inf, Mk)          # f - expr <=  M(1-x)
                add(base + [(xc, -Mk)], -Mk, np.inf)         # f - expr >= -M(1-x)
                add([(iff(s, k), 1.0), (xc, -Mk)], -np.inf, 0.0)     # f <=  M x
                add([(iff(s, k), 1.0), (xc, Mk)], 0.0, np.inf)       # f >= -M x
            add([(iov(s, k), 1.0), (iff(s, k), -1.0)], -rating, np.inf)   # ov >=  f - r
            add([(iov(s, k), 1.0), (iff(s, k), 1.0)], -rating, np.inf)    # ov >= -f - r
        # KCL at every bus except slack
        incid = {bus: [] for bus in range(1, n_bus)}
        for k, (i, j, b, rating) in enumerate(all_lines):
            if i >= 1:
                incid[i].append((iff(s, k), -1.0))
            if j >= 1:
                incid[j].append((iff(s, k), 1.0))
        for bus in range(1, n_bus):
            add(incid[bus], -net.injections[s, bus], -net.injections[s, bus])

    A = coo_matrix((vals, (rows, cols)), shape=(r, N)).tocsr()
    c = np.zeros(N)
    c[:n_c] = costs
    for s in range(n_scen):
        c[o_ov + s * nL: o_ov + (s + 1) * nL] = lam * net.weights[s]

    t0 = time.perf_counter()
    res = milp(c, constraints=LinearConstraint(A, np.array(c_lo), np.array(c_hi)),
               integrality=integrality, bounds=Bounds(lb, ub),
               options={"disp": bool(verbose), "time_limit": float(time_limit),
                        "mip_rel_gap": 0.0})
    wall = time.perf_counter() - t0
    status = {0: "OPTIMAL", 1: "TIME_LIMIT", 2: "INFEASIBLE", 3: "UNBOUNDED"}.get(
        res.status, f"HIGHS_{res.status}")
    if res.x is None:
        return {"status": status, "objective": None, "bound": float(res.mip_dual_bound)
                if res.mip_dual_bound is not None else None, "mip_gap": None,
                "wall_s": round(wall, 2), "nodes": int(res.mip_node_count or 0),
                "n_vars": int(N), "n_constrs": int(r), "plan": None, "subset": None,
                "solver": "highs", "lambda_used": float(lam)}
    xs = np.array([int(round(v)) for v in res.x[:n_c]])
    return {
        "status": status, "objective": float(res.fun),
        "bound": float(res.mip_dual_bound), "mip_gap": float(res.mip_gap),
        "wall_s": round(wall, 2), "nodes": int(res.mip_node_count),
        "n_vars": int(N), "n_constrs": int(r),
        "plan": "".join(map(str, xs)),
        "subset": [int(i) for i in np.flatnonzero(xs)],
        "solver": "highs", "lambda_used": float(lam),
        "big_m": {"kind": "effective-resistance (rigorous)", "theta_max": float(TH.max()),
                  "flow_M_max": float(M.max()), "cand_M_max": float(M[:, nE:].max()),
                  "heuristic_theta_bound": float(_big_m_heuristic(net, cands))},
    }


def solve_true_milp(meta, net, cands, costs, time_limit=600, verbose=False, lam=None):
    import gurobipy as gp
    from gurobipy import GRB

    n_bus = meta["n_bus"]
    lam = meta["lambda"] if lam is None else lam
    mu, budget = meta["mu"], meta["budget"]
    n_scen = len(net.weights)
    n_c = len(cands)

    all_lines = list(net.lines) + list(cands)
    nE = len(net.lines)
    TH, D = _flow_bounds(net, cands)                   # same rigorous bounds as highs
    fmax = {(s, k): b * D[s, k] for s in range(n_scen)
            for k, (i, j, b, r) in enumerate(all_lines)}

    m = gp.Model("grid_true")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = 0.0

    x = m.addVars(n_c, vtype=GRB.BINARY, name="x")
    th = m.addVars(n_scen, n_bus, lb=[-TH[s] for s in range(n_scen) for _ in range(n_bus)],
                   ub=[TH[s] for s in range(n_scen) for _ in range(n_bus)], name="th")
    f = m.addVars(n_scen, len(all_lines),
                  lb=[-fmax[s, k] for s in range(n_scen) for k in range(len(all_lines))],
                  ub=[fmax[s, k] for s in range(n_scen) for k in range(len(all_lines))],
                  name="f")
    ov = m.addVars(n_scen, len(all_lines), lb=0.0, name="ov")

    for s in range(n_scen):
        m.addConstr(th[s, 0] == 0.0)
        # flow definition: existing exact, candidate big-M switched
        for k, (i, j, b, r) in enumerate(all_lines):
            expr = b * (th[s, i] - th[s, j])
            if k < nE:
                m.addConstr(f[s, k] == expr)
            else:
                xc = x[k - nE]
                M = fmax[s, k]
                m.addConstr(f[s, k] - expr <= M * (1 - xc))
                m.addConstr(f[s, k] - expr >= -M * (1 - xc))
                m.addConstr(f[s, k] <= M * xc)
                m.addConstr(f[s, k] >= -M * xc)
            # overload >= |f| - rating
            m.addConstr(ov[s, k] >= f[s, k] - r)
            m.addConstr(ov[s, k] >= -f[s, k] - r)
        # KCL at every bus except slack (slack absorbs by construction of injections)
        for bus in range(1, n_bus):
            inflow = gp.LinExpr()
            for k, (i, j, b, r) in enumerate(all_lines):
                if i == bus:
                    inflow -= f[s, k]
                elif j == bus:
                    inflow += f[s, k]
            m.addConstr(inflow + net.injections[s, bus] == 0.0)

    obj = gp.quicksum(costs[e] * x[e] for e in range(n_c))
    for s in range(n_scen):
        obj += lam * net.weights[s] * gp.quicksum(
            ov[s, k] for k in range(len(all_lines)))
    if mu and budget:
        t = gp.quicksum(x[e] for e in range(n_c)) - budget
        obj += mu * t * t
    m.setObjective(obj, GRB.MINIMIZE)

    t0 = time.perf_counter()
    m.optimize()
    wall = time.perf_counter() - t0
    status = {2: "OPTIMAL", 9: "TIME_LIMIT"}.get(m.Status, str(m.Status))
    xs = np.array([int(round(x[e].X)) for e in range(n_c)])
    return {
        "status": status, "objective": float(m.ObjVal),
        "bound": float(m.ObjBound), "mip_gap": float(m.MIPGap),
        "wall_s": round(wall, 2), "nodes": int(m.NodeCount),
        "n_vars": m.NumVars, "n_constrs": m.NumConstrs,
        "plan": "".join(map(str, xs)),
        "subset": [int(i) for i in np.flatnonzero(xs)],
        "solver": "gurobi", "lambda_used": float(lam),
        "big_m": {"kind": "effective-resistance (rigorous)", "theta_max": float(TH.max()),
                  "flow_M_max": float(max(fmax.values())),
                  "cand_M_max": float(max(v for (s, k), v in fmax.items() if k >= nE)),
                  "heuristic_theta_bound": float(_big_m_heuristic(net, cands))},
    }


def solve_qubo_miqp(meta, Q, time_limit=600):
    import gurobipy as gp
    from gurobipy import GRB
    n = Q.shape[0]
    m = gp.Model("grid_qubo")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = 0.0
    x = m.addVars(n, vtype=GRB.BINARY)
    obj = gp.QuadExpr()
    for i in range(n):
        obj += Q[i, i] * x[i]
        for j in range(i + 1, n):
            if abs(Q[i, j]) > 1e-15:
                obj += 2.0 * Q[i, j] * x[i] * x[j]
    m.setObjective(obj + meta["qubo_constant"], GRB.MINIMIZE)
    t0 = time.perf_counter()
    m.optimize()
    xs = np.array([int(round(x[i].X)) for i in range(n)])
    return {"status": {2: "OPTIMAL", 9: "TIME_LIMIT"}.get(m.Status, str(m.Status)),
            "qubo_energy": float(m.ObjVal), "mip_gap": float(m.MIPGap),
            "wall_s": round(time.perf_counter() - t0, 2),
            "nodes": int(m.NodeCount),
            "plan": "".join(map(str, xs)),
            "subset": [int(i) for i in np.flatnonzero(xs)]}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--instance", default="instB33")
    p.add_argument("--time-limit", type=float, default=600)
    p.add_argument("--qubo", action="store_true",
                   help="also solve the QUBO surrogate exactly (MIQP, gurobi only)")
    p.add_argument("--solver", choices=["auto", "gurobi", "highs"], default="auto",
                   help="MILP backend; 'auto' = gurobi if importable else highs")
    p.add_argument("--lam", type=float, default=None,
                   help="override lambda for the TRUE problem (Pareto/hardness sweep); "
                        "writes <name>_ground_truth_lam<value>.json instead")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    solver = args.solver
    if solver == "auto":
        try:
            import gurobipy  # noqa: F401
            solver = "gurobi"
        except ImportError:
            solver = "highs"

    meta, net, cands, costs, Q = load_instance(args.instance)
    lam = meta["lambda"] if args.lam is None else args.lam
    F = exact_objective(net, cands, costs, lam, meta["mu"], meta["budget"])
    print(f"MILP ground truth for '{meta['name']}' [{meta.get('network', 'synth')}]: "
          f"{meta['n_candidates']} candidates, lambda={lam:g}"
          f"{' (OVERRIDE; instance has ' + format(meta['lambda'], 'g') + ')' if args.lam is not None else ''}, "
          f"solver={solver}")

    solve = solve_true_milp if solver == "gurobi" else solve_true_milp_highs
    res = solve(meta, net, cands, costs, args.time_limit, args.verbose, lam=lam)
    if res["objective"] is None:
        print(f"  TRUE problem:  {res['status']} — no feasible solution within the limit "
              f"(bound {res['bound']}). {res['wall_s']}s, {res['nodes']} nodes.")
        return 1
    print(f"  TRUE problem:  {res['status']}  F* = {res['objective']:.6f}  "
          f"(bound {res['bound']:.6f}, gap {100 * res['mip_gap']:.3f}%)")
    print(f"    plan {res['plan']}  build {res['subset']}  "
          f"({len(res['subset'])}/{meta['n_candidates']} lines)")
    print(f"    {res['wall_s']}s, {res['nodes']} nodes, "
          f"{res['n_vars']} vars / {res['n_constrs']} constrs")
    bm = res["big_m"]
    print(f"    big-M: rigorous theta_max {bm['theta_max']:.4g}, candidate flow M max "
          f"{bm['cand_M_max']:.4g}  (legacy heuristic theta bound was "
          f"{bm['heuristic_theta_bound']:.4g})")

    # the self-check that makes the number trustworthy (also guards big-M)
    check = float(F(tuple(res["subset"])))
    if abs(check - res["objective"]) > 1e-5 * max(1.0, abs(check)):
        print(f"  ✗ MISMATCH: simulator re-score {check:.6f} != MILP "
              f"{res['objective']:.6f} — big-M or model bug. DO NOT USE.")
        return 1
    print(f"  ✓ cross-check: exact_objective(plan) = {check:.6f} matches MILP")
    res["true_F_recheck"] = check
    if res["status"] != "OPTIMAL":
        print(f"  ⚠ NOT PROVEN OPTIMAL — incumbent only; the bound is the certificate.")

    out = {"instance": meta["name"], "qubo_sha256_16": meta["qubo_sha256_16"],
           "lambda_used": lam, "solver": solver, "true_milp": res}

    if args.qubo and solver != "gurobi":
        print("  (--qubo skipped: the surrogate MIQP needs the gurobi backend)")
    if args.qubo and solver == "gurobi":
        rq = solve_qubo_miqp(meta, Q, args.time_limit)
        tq = float(F(tuple(rq["subset"])))
        rq["true_F_of_argmin"] = tq
        gap = 100.0 * (tq - res["objective"]) / abs(res["objective"])
        rq["argmin_true_gap_pct"] = gap
        print(f"  QUBO surrogate: {rq['status']}  E* = {rq['qubo_energy']:.6f}  "
              f"({rq['wall_s']}s, {rq['nodes']} nodes)")
        print(f"    argmin true F = {tq:.6f}  ->  true gap {gap:+.3f}% "
              f"(surrogate quality at {meta['n_candidates']}q)")
        out["qubo_miqp"] = rq

    suffix = "" if args.lam is None else f"_lam{args.lam:g}"
    path = Path(f"{meta['name']}_ground_truth{suffix}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  ✓ Saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
