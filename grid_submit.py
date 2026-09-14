"""
Submit a grid-expansion QUBO to Pasqal.

Deliberately thin: everything that touches credentials, the SDK or device selection is
imported from submit_to_pasqal.py (looked up in this folder, then one level up), so
there is exactly ONE copy of the interactive-credential path (the password is never
handled by our code — the Pasqal SDK prompts for it directly with hidden input).

Differences from the logistics family that matter here:
  * The diagonal is NOT uniform (costs differ per candidate line), so the DMM is live
    for the first time — --no-dmm is a real experimental variable, not a no-op.
  * Every bitstring is a feasible plan: no one-hot constraint, no excitation-density
    fight with the analog drive. The mechanism found in our earlier logistics study predicts this family suits the
    hardware; this script produces the datapoint that tests it.

Usage:
    python grid_submit.py --instance inst33 --device EMU_FREE --shots 2000
    python grid_submit.py --instance inst33 --use-classical     # local, no cloud
"""

import sys
import json
import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from submit_to_pasqal import (          # noqa: E402
    get_credentials_interactive,
    get_device_info,
    submit_qubo_to_pasqal,
    save_pasqal_results,
)
from grid_expansion_qubo import load_instance   # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Submit a grid-expansion QUBO to Pasqal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Notes:
  Unlike the logistics family, this QUBO's diagonal is non-uniform, so the DMM is
  active by default and --no-dmm is a real ablation (worth one free run eventually).

  Evaluate with:  python grid_evaluate.py --instance <name> --results <file>
        """,
    )
    parser.add_argument("--instance", default="inst33",
                        help="instance name (loads <name>_problem.json + <name>_qubo.npy)")
    parser.add_argument("--device", default=None,
                        help="EMU_FREE | EMU_SV | EMU_MPS. Auto-selects if omitted.")
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--username", default=None,
                        help="Optional; prompted interactively if omitted (keeps it out "
                             "of shell history)")
    parser.add_argument("--project-id", default=None, help="Optional; prompted if omitted")
    parser.add_argument("--use-classical", action="store_true",
                        help="Local CPLEX instead of the cloud — sanity-check an instance "
                             "without spending anything")
    parser.add_argument("--postprocess", action="store_true",
                        help="Enable qubosolver classical repair (default OFF — we measure "
                             "the device, not the repair)")
    parser.add_argument("--no-dmm", action="store_true",
                        help="Disable the DMM (real ablation here — non-uniform diagonal)")
    parser.add_argument("--output", default=None,
                        help="Results JSON. Default: <instance>_pasqal_results.json")
    args = parser.parse_args()

    try:
        meta, _net, _cands, _costs, Q = load_instance(args.instance)
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ {e}")
        return 1

    n = meta["n_candidates"]
    gt = meta.get("ground_truth")
    print(f"Loaded grid instance '{meta['name']}' [{meta.get('network', 'synth')}]")
    print(f"  {meta['n_bus']} buses, {n} candidate lines -> {n} qubits, "
          f"lambda={meta['lambda']:g}, Q sha {meta['qubo_sha256_16']}")
    if gt:
        print(f"  Known optimum: build {gt['optimal_subset']} at F = "
              f"{gt['optimal_value']:.4f} (build-nothing {gt['build_none_value']:.4f}); "
              f"optimum sits at QUBO-energy rank {gt['optimum_rank_in_qubo_energy']}")

    # Fail before authenticating, not after, if the analog solver cannot represent Q.
    # qubosolver's quantum path raises on ANY strictly negative off-diagonal (Rydberg
    # interactions are repulsive); this reproduces its exact test locally.
    if not args.use_classical:
        import numpy as np
        off = Q[np.triu_indices(n, 1)]            # one entry per coupling pair
        n_neg = int((off < 0).sum())
        if n_neg:
            print(f"\n✗ {n_neg} of {off.size} couplings (off-diagonal pairs) are negative "
                  f"(min {off.min():+.4f}); Pasqal's analog solver rejects this "
                  f"(\"Quantum solver does not handle off-diagonal negative coefficients\").")
            print(f"  Rebuild an analog-representable surrogate of the same network and "
                  f"submit that instead, e.g.:")
            print(f"    python grid_expansion_qubo.py --network {meta.get('network', 'synth')} "
                  f"--n-cand {n} --scenarios {meta['scenarios']} --lam {meta['lambda']:g} "
                  f"--rating-frac {meta['rating_frac']:g} --stress {meta['stress']:g} "
                  f"--seed {meta['seed']} --offdiag nonneg --name {meta['name']}p")
            print(f"  Gate-based QAOA (qiskit_qaoa.py) and every classical baseline still "
                  f"accept the sign-mixed '{meta['name']}' unchanged.")
            return 1

    # Fail before authenticating, not after, if the device cannot hold the problem.
    if args.device and not args.use_classical:
        devices = get_device_info()
        if args.device in devices:
            practical = devices[args.device]["practical"]
            if n > practical:
                print(f"\n✗ {n} atoms exceeds {args.device}'s practical limit "
                      f"of {practical}. Use EMU_MPS above 20 atoms.")
                return 1

    username = args.username
    project_id = args.project_id
    if not args.use_classical and (not username or not project_id):
        username, project_id = get_credentials_interactive()
        if not username or not project_id:
            return 1

    result = submit_qubo_to_pasqal(
        Q,
        username=username,
        project_id=project_id,
        use_quantum=not args.use_classical,
        num_shots=args.shots,
        device_name=args.device,
        do_postprocessing=args.postprocess,
        use_dmm=not args.no_dmm,
    )
    if result is None:
        return 1
    solution, _connection = result

    out = args.output or f"{meta['name']}_pasqal_results.json"
    save_pasqal_results(solution, out)

    # Stamp so grid_evaluate.py can never score a mismatched instance.
    with open(out) as f:
        data = json.load(f)
    data["metadata"].update({
        "instance": meta["name"],
        "network": meta.get("network"),
        "n_variables": n,
        "qubo_sha256_16": meta["qubo_sha256_16"],
        "device": args.device or "auto",
        "shots_requested": args.shots,
        "postprocess": args.postprocess,
        "dmm": not args.no_dmm,
        "solver": "classical_cplex" if args.use_classical else "pasqal_quantum",
    })
    with open(out, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nNext:")
    print(f"  python grid_evaluate.py --instance {meta['name']} --results {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
