# Quantum + AI Grid Expansion Planning for Distribution Networks

Code, instances and raw results behind Team DXC Quantum's Phase 1 concept proposal for the
E.ON problem statement *Quantum-Enabled Grid Expansion Planning for Distribution System Energy
Networks* (2026 Global Quantum + AI Challenge). Everything quoted in the proposal was produced
by the scripts in this repository on the instance files committed here.

## The problem in one paragraph

A distribution system operator must choose which candidate reinforcements (tie lines, parallel
lines, transformer upgrades) to build so that the grid is not overloaded across a set of operating
scenarios, at minimum capital cost. One binary variable per candidate line, no auxiliary
variables, so `n` candidates map to `n` qubits. The true objective is

```
F(x) = sum_e c_e x_e  +  lambda * sum_s w_s sum_l max(|f_l(x, s)| - r_l, 0)
```

with `f_l(x, s)` the DC power flow on line `l` in scenario `s` for build plan `x`, `r_l` the line
rating, `c_e` the build cost and `w_s` the scenario weight. A scenario is one operating state of
the grid (load and distributed-generation output at every bus). A quadratic surrogate (QUBO) of
`F` is learned from the power-flow simulator; quantum and classical samplers search the surrogate;
every sampled plan is re-scored on the true `F`. The proposal describes the method, the measured
limitation of the global surrogate and the trust-region refit loop that fixes it.

## Files

| File | Role |
|---|---|
| `grid_expansion_qubo.py` | Instance builder from pandapower networks (any scenario count), DC multi-scenario simulator, regression surrogate, exhaustive ground truth for n <= 14, `--selftest` |
| `milp_ground_truth.py` | Exact MILP of the true objective (flow bounds from network effective resistances, overload slacks) in HiGHS or Gurobi, cross-checked against the simulator; `--lam` overrides lambda for Pareto sweeps and hardness probes |
| `classical_baselines.py` | Greedy, tuned simulated annealing on the true objective, SA on the surrogate, uniform random at matched evaluation budget |
| `surrogate_refine.py` | Trust-region local refit loop (accept / shrink), trajectory logged |
| `qiskit_qaoa.py` | QAOA reference in Qiskit (Ising mapping with endianness self-test), common results schema |
| `grid_submit.py` | Pasqal submission of the same instance; thin wrapper around `submit_to_pasqal.py` |
| `grid_evaluate.py` | One evaluator for every platform: SHA-256 hash guard, true re-scoring, preference test against a uniform null with a pre-registered 5 % floor and minimum detectable effect, per-shot anytime yields, surrogate alignment |
| `qubo_diagnostics.py` | Problem-agnostic QUBO checks behind the evaluator: symmetry check, detuning-map effectiveness check, sampling-quality preference test against a uniform baseline, coupling statistics |
| `requirements.txt` | Pinned package versions used for all reported numbers |

The proposal's Table A1 lists the seven scripts that are run directly. `submit_to_pasqal.py` and
`qubo_diagnostics.py` are library modules imported by `grid_submit.py` and `grid_evaluate.py`; they
are shared with our earlier logistics study and ship here so the repository is self-contained.

Every instance is a pair `<name>_qubo.npy` (symmetric Q, minimise x^T Q x) + `<name>_problem.json`
(network, candidates, scenarios, fit statistics, ground truth, stamps). The SHA-256 prefix of the
QUBO is stored in the JSON and checked by every downstream script, so a results file can never be
scored against the wrong instance. Use the committed files as they are; regenerating them with
other library versions can change the surrogate fit and hence the hash.

## Instances

| Name | Network | Candidates (qubits) | Scenarios | lambda | Notes |
|---|---|---|---|---|---|
| `inst33` | IEEE 33-bus feeder (`case33bw`) | 12 | 3 | 50 | Correctness anchor; optimum by exhaustive enumeration of all 4,096 plans |
| `inst33p` | same | 12 | 3 | 50 | Same problem, surrogate fitted with non-negative off-diagonal couplings (`--offdiag nonneg`) for the analog neutral-atom path; fit statistics stamped in the JSON |
| `instB33` | `case33bw` | 33 | 3 | 50 | Ties plus parallel reinforcements; exact optimum proved by MILP |
| `instC179` | `mv_oberrhein`, real German medium-voltage grid, 179 buses | 120 | 3 | 50 | Utility-scale instance; hardness map over lambda in `instC179_ground_truth_lam*.json` |
| `h200_results/scenario_probe/instB33s{3,6,9,12,18,24}` | `case33bw` | 33 | 3 to 24 | 10, 50 | Scenario-count study |
| `h200_results/scenario_probe/instC179s{6,9,12}` | `mv_oberrhein` | 120 | 6 to 12 | 2, 50 | Scenario-count study on the real grid |

Scenarios are (load multiplier, DER multiplier) pairs applied to the network's nominal loads and
generators: mild (0.8, 0.3), nominal (1.0, 0.6) and stressed with DER export (1.45, 1.2) with
weights 0.25 / 0.5 / 0.25; for more than three scenarios a load x DER grid with uniform weights.
Line ratings are synthesised as 0.9 x nominal flow because the bundled ratings never congest.
Candidates are the network's out-of-service tie lines (cost proportional to length) plus parallels
of the most-loaded in-service lines (unit cost). All of this is stated in the proposal wherever a
number appears.

## Result files and where they appear in the proposal

| Proposal | Files |
|---|---|
| Table 1, Table B1 (baselines, certified optima, refit loop) | `inst33_baselines.json`, `instB33_baselines.json`, `instC179_baselines.json`, `inst33_ground_truth.json`, `instB33_ground_truth.json`, `instC179_ground_truth.json`, `instB33_refine.json` |
| Section 2.3 (QAOA, noiseless simulator) | `inst33_qaoa_results.json` + `_analysis.json`, `inst33p_qaoa_results.json` + `_analysis.json` |
| Validation protocol, known-null check | `inst33_random_results.json` + `_analysis.json` |
| Table C1 (hardness map by lambda and scenario count) | `instC179_ground_truth_lam{0.5,1,2,5,10,20,100}.json`, `h200_results/sweep_lam*.log`, `h200_results/instC179_gt_lam50_highs_v2.log`, `h200_results/scenario_probe/` (instances, ground truths, logs, driver script) |

Solver logs name the host class. Three-scenario lambda sweeps except lambda = 50 ran on a 20-core
laptop with seven solves in parallel (wall-clock inflated, ordering robust); everything else ran
one solve at a time on a 224-core compute node. HiGHS 1.12 throughout; Gurobi confirmation of the
open instances is the first task of the proposed sprint.

## Quick start

```bash
pip install -r requirements.txt
python grid_expansion_qubo.py --selftest
python qiskit_qaoa.py --selftest

# exact ground truth (HiGHS; add --solver gurobi if licensed), Pareto point at another lambda
python milp_ground_truth.py --instance instB33
python milp_ground_truth.py --instance instC179 --lam 2 --solver highs --time-limit 1800

# tuned classical baselines, then the trust-region refit loop
python classical_baselines.py --instance instB33
python surrogate_refine.py --instance instB33 --rounds 6

# QAOA on the noiseless Aer simulator, then score it with the common evaluator
python qiskit_qaoa.py --instance inst33 --reps 2 --shots 2000
python grid_evaluate.py --instance inst33 --results inst33_qaoa_results.json

# Pasqal emulator on the analog-compatible surrogate (prompts for credentials, nothing stored)
python grid_submit.py --instance inst33p --device EMU_FREE --shots 2000
python grid_evaluate.py --instance inst33p --results inst33p_pasqal_results.json

# build a new instance (example: real grid, 120 candidates, 9 scenarios, 16 worker processes)
python grid_expansion_qubo.py --network mv_oberrhein --n-cand 120 --scenarios 9 --lam 50 --seed 7 --workers 16 --name instC179s9
```

Run every command from the repository root: scripts resolve `<name>_problem.json` relative to the
working directory. Instances inside `h200_results/scenario_probe/` can be addressed by path
(`--instance h200_results/scenario_probe/instC179s9_problem.json`).

## Modelling choices and limits

- DC power flow inside the loop; the proposal commits to AC re-checks of winning plans.
- Ratings are synthesised (fraction of nominal flow); a DSO's real ratings replace them without
  code changes.
- Every quantum number so far is a noiseless-simulator result, an upper bound on hardware.
- The analog neutral-atom path can only realise non-negative couplings, hence the constrained
  surrogate `inst33p`; the gate-based and classical paths use the unconstrained surrogate.
- Reported hardness is "HiGHS-open after the stated time limit with the stated bound"; a weak
  relaxation looks exactly like hardness, which is why the bound and the solver are always named.

## Licence and contact

Apache License 2.0 (see `LICENSE`). Networks come from the open pandapower library; no E.ON-internal
or third-party confidential data is used.

Team DXC Quantum, team lead Dr. Sławomir Folwarski (slawomir.folwarski@dxc.com).
