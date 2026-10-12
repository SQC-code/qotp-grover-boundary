"""Run transpile-only scaling for large n (14, 16).

Skip statevector simulation to avoid OOM. Report depth/CX/1Q only.
Output: scaling_results_large.json + scaling_table_rows_large.tex (with dagger marker).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trapdoor_grover import QOTPGroverProtocol
from qiskit import transpile


def keys_for_n(n):
    N = 2 ** n
    return [1, N - 1]


def run_n_transpile_only(n):
    keys = keys_for_n(n)
    proto = QOTPGroverProtocol(n_qubits=n, keys=keys)
    qc, n_iter = proto.build_circuit()
    tqc = transpile(qc, basis_gates=["u3", "u2", "u1", "cx", "id"], optimization_level=2)
    counts = dict(tqc.count_ops())
    return {
        "n": n, "N": 2 ** n, "M": 2, "n_iter": n_iter,
        "depth": tqc.depth(),
        "gate_counts": {k: int(v) for k, v in counts.items()},
        "cx_count": int(counts.get("cx", 0)),
        "single_qubit_count": int(sum(v for k, v in counts.items() if k in ("u1", "u2", "u3", "id"))),
        "sim_time_s": None,
        "integrity_ratio": None,
        "simulated": False,
    }


def main():
    out_dir = Path(__file__).resolve().parent
    results = []
    print("=== Transpile-only scaling: n in {14, 16} ===")
    for n in [14, 16]:
        print(f"--- starting n={n} ---", flush=True)
        t0 = time.perf_counter()
        try:
            r = run_n_transpile_only(n)
            results.append(r)
            elapsed = time.perf_counter() - t0
            print(
                f"n={n:2d}  depth={r['depth']:8d}  CX={r['cx_count']:8d}  "
                f"1Q={r['single_qubit_count']:8d}  n_iter={r['n_iter']:3d}  "
                f"(transpile {elapsed:.1f}s)",
                flush=True,
            )
        except Exception as exc:
            print(f"n={n} FAILED: {type(exc).__name__}: {exc}", flush=True)
            results.append({"n": n, "error": str(exc)})

    (out_dir / "scaling_results_large.json").write_text(json.dumps(results, indent=2))

    lines = []
    for r in results:
        if "error" in r:
            continue
        sim_cell = "--"
        int_cell = "--"
        lines.append(
            f"{r['n']}$^{{\\dagger}}$ & {r['N']} & {r['n_iter']} & "
            f"{r['depth']} & {r['cx_count']} & {r['single_qubit_count']} & "
            f"{sim_cell} & {int_cell} \\\\"
        )
    (out_dir / "scaling_table_rows_large.tex").write_text("\n".join(lines) + "\n")
    print(f"\nWritten: {out_dir/'scaling_results_large.json'}")
    print(f"Written: {out_dir/'scaling_table_rows_large.tex'}")


if __name__ == "__main__":
    main()
