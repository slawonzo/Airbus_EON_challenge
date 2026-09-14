"""
Qiskit QAOA reference for a grid-expansion instance — the Qiskit-compatibility layer the challenge requires.

The challenge (§5.3) requires E.ON to be able to execute and validate results in
Qiskit. This script is that path: it consumes the SAME <name>_qubo.npy / _problem.json
pair as the Pasqal pipeline and emits the SAME results-JSON schema, so grid_evaluate.py
scores both platforms with identical discipline. The QUBO matrix is the contract.

Local backend is qiskit-aer (exact simulation to ~30 qubits); swapping in IBM hardware
is a one-line backend change (qiskit-ibm-runtime), kept out of this file so nothing
here ever needs credentials.

Usage:
    python qiskit_qaoa.py --selftest
    python qiskit_qaoa.py --instance inst33 --reps 2 --shots 2000
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from grid_expansion_qubo import load_instance, qubo_value   # noqa: E402


def qubo_to_ising(Q):
    """Q (symmetric) -> (pauli_terms, offset) with x_i = (1 - z_i)/2.

    Returns terms as {(i,): h_i, (i, j): J_ij} for a SparsePauliOp, and the constant
    offset, such that  x^T Q x  ==  sum h_i z_i + sum J_ij z_i z_j + offset.
    """
    n = Q.shape[0]
    h = -Q.diagonal() / 2.0 - (Q.sum(axis=1) - Q.diagonal()) / 2.0
    offset = Q.diagonal().sum() / 2.0 + (Q.sum() - Q.diagonal().sum()) / 4.0
    terms = {}
    for i in range(n):
        if abs(h[i]) > 1e-15:
            terms[(i,)] = float(h[i])
    for i in range(n):
        for j in range(i + 1, n):
            if abs(Q[i, j]) > 1e-15:
                terms[(i, j)] = float(Q[i, j] / 2.0)   # Q symmetric: 2Q_ij x_i x_j -> Q_ij z_i z_j / 2
    return terms, float(offset)


def ising_energy(terms, offset, x):
    z = 1.0 - 2.0 * np.asarray(x, dtype=float)
    e = offset
    for idx, c in terms.items():
        e += c * np.prod(z[list(idx)])
    return e


def build_pauli_op(terms, n):
    from qiskit.quantum_info import SparsePauliOp
    labels, coeffs = [], []
    for idx, c in terms.items():
        s = ["I"] * n
        for q in idx:
            s[n - 1 - q] = "Z"          # Qiskit label order: qubit 0 rightmost
        labels.append("".join(s))
        coeffs.append(c)
    return SparsePauliOp(labels, np.array(coeffs))


def run_qaoa(Q, const, reps=2, shots=2000, maxiter=200, seed=0):
    """Optimise QAOA(reps) on Aer and sample. Returns (counts, info).

    counts keys use OUR convention: leftmost char = candidate 0 (keys are reversed
    from Qiskit's little-endian order at the boundary, nowhere else).
    """
    from scipy.optimize import minimize
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from qiskit_aer.primitives import EstimatorV2, SamplerV2

    n = Q.shape[0]
    terms, offset = qubo_to_ising(Q)
    op = build_pauli_op(terms, n)
    ansatz = QAOAAnsatz(cost_operator=op, reps=reps)
    backend = AerSimulator(seed_simulator=seed)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1,
                                      seed_transpiler=seed)
    circ = pm.run(ansatz)
    op_t = op.apply_layout(circ.layout)

    est = EstimatorV2()
    n_evals = [0]

    def energy(params):
        n_evals[0] += 1
        ev = est.run([(circ, op_t, [params])]).result()[0].data.evs
        return float(np.atleast_1d(ev)[0])

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, np.pi / 4, ansatz.num_parameters)
    t0 = time.perf_counter()
    opt = minimize(energy, x0, method="COBYLA",
                   options={"maxiter": maxiter, "rhobeg": 0.3})
    wall_opt = time.perf_counter() - t0

    meas = circ.copy()
    meas.measure_all()
    sampler = SamplerV2(seed=seed)
    raw = sampler.run([(meas, opt.x)], shots=shots).result()[0]
    key = list(raw.data.keys())[0]                  # measure_all register name
    qiskit_counts = getattr(raw.data, key).get_counts()
    counts = {k[::-1]: v for k, v in qiskit_counts.items()}   # to our convention

    info = {"reps": reps, "shots": shots, "optimizer": "COBYLA",
            "estimator_evals": n_evals[0],
            "opt_energy_ising": float(opt.fun),
            "opt_energy_qubo": float(opt.fun) + const - offset * 0,
            "wall_s_optimize": round(wall_opt, 2),
            "params": [float(v) for v in opt.x], "seed": seed}
    return counts, info


def selftest():
    print("=" * 60)
    print("qiskit_qaoa self-test")
    print("=" * 60)
    rng = np.random.default_rng(1)
    n = 5
    Q = rng.normal(size=(n, n))
    Q = (Q + Q.T) / 2
    terms, offset = qubo_to_ising(Q)
    for k in range(2 ** n):
        x = [(k >> i) & 1 for i in range(n)]
        a = qubo_value(Q, 0.0, x)
        b = ising_energy(terms, offset, x)
        assert abs(a - b) < 1e-9, (x, a, b)
    print(f"  ✓ Ising mapping == QUBO on all {2**n} states")

    # endianness: a QUBO that only rewards x_0 = 1 must sample plans starting with '1'
    Q2 = np.zeros((3, 3))
    Q2[0, 0] = -5.0
    counts, _ = run_qaoa(Q2, 0.0, reps=1, shots=400, maxiter=60, seed=2)
    top = max(counts, key=counts.get)
    assert top[0] == "1", f"endianness broken: top sample {top}"
    print(f"  ✓ endianness: top sample '{top}' has candidate 0 built")
    print("\n  ✓ Self-test passed.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--instance", default="inst33")
    p.add_argument("--reps", type=int, default=2, help="QAOA depth p")
    p.add_argument("--shots", type=int, default=2000)
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None,
                   help="Results JSON. Default: <instance>_qaoa_results.json")
    args = p.parse_args()
    if args.selftest:
        return selftest()

    meta, _net, _cands, _costs, Q = load_instance(args.instance)
    n = meta["n_candidates"]
    print(f"QAOA(p={args.reps}) on '{meta['name']}' [{meta.get('network', 'synth')}]: "
          f"{n} qubits, {args.shots} shots, Aer simulator")
    if n > 30:
        print(f"  ⚠ {n} qubits is beyond comfortable local Aer simulation — "
              f"this is an IBM-hardware / H200 job.")
        return 1

    counts, info = run_qaoa(Q, meta["qubo_constant"], args.reps, args.shots,
                            args.maxiter, args.seed)
    print(f"  optimiser: {info['estimator_evals']} energy evaluations, "
          f"{info['wall_s_optimize']}s; sampled {sum(counts.values())} shots, "
          f"{len(counts)} distinct plans")

    out = args.output or f"{meta['name']}_qaoa_results.json"
    with open(out, "w") as fh:
        json.dump({"metadata": {
            "instance": meta["name"], "network": meta.get("network"),
            "n_variables": n, "qubo_sha256_16": meta["qubo_sha256_16"],
            "solver": "qiskit_qaoa", "device": "AerSimulator",
            "total_samples": sum(counts.values()),
            "unique_solutions": len(counts), **info,
        }, "results": counts}, fh, indent=2)
    print(f"  ✓ Saved {out}")
    print(f"\nNext:")
    print(f"  python grid_evaluate.py --instance {meta['name']} --results {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
