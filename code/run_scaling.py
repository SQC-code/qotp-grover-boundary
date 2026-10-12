"""Scaling analysis for QOTPGroverProtocol with client/server resource split.

For n in {4, 6, 8, 10, 12} (M=2, marked pair {1, N-1}) collect, in ONE session
with a FIXED transpiler seed:
- client-side resources: |kappa> preparation (structured O(n) circuit) + QOTP,
  transpiled separately;
- server-side resources: search-register H + n_iter x (comparator + diffuser)
  + measurement, transpiled separately;
- full-circuit depth/counts (what the joint simulation runs);
- simulation wall-clock (single run, indicative only);
- integrity ratio: n=6 additionally over N_PADS independent pads to report a
  mean +- std (the other n use one sampled pad; the decrypted law is
  pad-independent by decryption correctness).

Outputs: scaling_results.json + scaling_table_rows.tex
"""
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trapdoor_grover import QOTPGroverProtocol

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
try:
    from qiskit_aer import AerSimulator
except ImportError:
    from qiskit.providers.aer import AerSimulator

SEED_TRANSPILER = 20260715
BASIS = ["u3", "u2", "u1", "cx", "id"]


def keys_for_n(n: int):
    N = 2 ** n
    return [1, N - 1]  # pair with maximum Hamming distance (n-1)


def tmetrics(qc):
    tqc = transpile(qc, basis_gates=BASIS, optimization_level=2,
                    seed_transpiler=SEED_TRANSPILER)
    counts = dict(tqc.count_ops())
    cx = counts.get("cx", 0)
    oneq = sum(v for k, v in counts.items() if k in ("u1", "u2", "u3", "id"))
    return tqc.depth(), cx, oneq, counts, tqc


def client_circuit(proto):
    key = QuantumRegister(proto.n, "key")
    qc = QuantumCircuit(key)
    proto._prepare_kappa(qc, key)
    proto._apply_qotp(qc, key)
    return qc


def server_circuit(proto, n_iter):
    search = QuantumRegister(proto.n, "search")
    key = QuantumRegister(proto.n, "key")
    c = ClassicalRegister(proto.n, "meas")
    qc = QuantumCircuit(search, key, c)
    qc.h(search)
    for _ in range(n_iter):
        proto._apply_V(qc, search, key)
        proto._apply_diffuser(qc, search)
    qc.measure(search, c)
    return qc


def integrity_one_pad(proto, tqc, sim, shots):
    result = sim.run(tqc, shots=shots).result()
    client_view = proto.decrypt(result.get_counts())
    marked = sum(c for bs, c in client_view.items() if int(bs, 2) in proto.keys)
    return marked / shots


def run_n(n: int, shots: int = 2048, n_pads: int = 1):
    keys = keys_for_n(n)
    sim = AerSimulator(method="statevector", seed_simulator=SEED_TRANSPILER)
    # deterministic pads: recorded in the JSON so 1Q counts are reproducible
    pad_rng = np.random.default_rng(SEED_TRANSPILER + n)
    qx = int(pad_rng.integers(0, 2 ** n))
    qz = int(pad_rng.integers(0, 2 ** n))
    proto = QOTPGroverProtocol(n_qubits=n, keys=keys, qotp_x=qx, qotp_z=qz)
    qc_full, n_iter = proto.build_circuit()
    d_f, cx_f, q1_f, counts_f, tqc_full = tmetrics(qc_full)
    d_c, cx_c, q1_c, _, _ = tmetrics(client_circuit(proto))
    d_s, cx_s, q1_s, _, _ = tmetrics(server_circuit(proto, n_iter))

    # integrity + timing: n_pads recorded pads (fresh protocol objects).
    # seed_simulator varies per pad (deterministically) so the shot noise is
    # independent across pads and the reported std is an honest estimate.
    ints, pads, pad_seeds, sim_time = [], [], [], None
    for i in range(n_pads):
        px = int(pad_rng.integers(0, 2 ** n))
        pz = int(pad_rng.integers(0, 2 ** n))
        pads.append([px, pz])
        seed_i = SEED_TRANSPILER + 1000 * n + i   # actual per-pad simulator seed
        pad_seeds.append(seed_i)
        p = QOTPGroverProtocol(n_qubits=n, keys=keys, qotp_x=px, qotp_z=pz)
        qc_i, _ = p.build_circuit()
        tqc_i = transpile(qc_i, basis_gates=BASIS, optimization_level=2,
                          seed_transpiler=SEED_TRANSPILER)
        t0 = time.perf_counter()
        result = sim.run(tqc_i, shots=shots, seed_simulator=seed_i).result()
        client_view = p.decrypt(result.get_counts())
        ints.append(sum(c for bs, c in client_view.items()
                        if int(bs, 2) in p.keys) / shots)
        if i == 0:
            sim_time = time.perf_counter() - t0
    return {
        "n": n, "N": 2 ** n, "M": len(keys), "n_iter": n_iter,
        "depth": d_f, "cx_count": cx_f, "single_qubit_count": q1_f,
        "client_depth": d_c, "client_cx": cx_c, "client_1q": q1_c,
        "server_depth": d_s, "server_cx": cx_s, "server_1q": q1_s,
        "gate_counts": counts_f,
        "resource_pad": [qx, qz], "integrity_pads": pads,
        "sim_time_s": sim_time,
        "integrity_ratio": float(np.mean(ints)),
        "integrity_std": float(np.std(ints, ddof=1)) if n_pads > 1 else None,
        "integrity_raw": [float(x) for x in ints],
        "n_pads": n_pads, "shots": shots,
        "seed_transpiler": SEED_TRANSPILER,
        "seed_simulator_per_pad": pad_seeds,
        "aer_method": "statevector", "opt_level": 2,
    }


def main():
    out_dir = Path(__file__).resolve().parent
    results = []
    print("=== Scaling analysis (client/server split, fixed transpiler seed) ===")
    for n in [4, 6, 8, 10, 12]:
        n_pads = 20 if n == 6 else 1   # n=6: resolve single-pad anomaly
        r = run_n(n, n_pads=n_pads)
        results.append(r)
        istd = f"+-{r['integrity_std']:.4f}" if r["integrity_std"] is not None else ""
        print(f"  n={n:2d}  total: depth={r['depth']:6d} CX={r['cx_count']:6d} | "
              f"client: CX={r['client_cx']:3d} | server: CX={r['server_cx']:6d} | "
              f"sim={r['sim_time_s']:6.2f}s  integ={r['integrity_ratio']:.4f}{istd}")
    (out_dir / "scaling_results.json").write_text(json.dumps(results, indent=2))
    lines = []
    for r in results:
        lines.append(
            f"{r['n']} & {r['N']} & {r['n_iter']} & {r['client_cx']} & "
            f"{r['server_cx']} & {r['depth']} & {r['cx_count']} & "
            f"{r['single_qubit_count']} & {r['sim_time_s']:.2f} & "
            f"{r['integrity_ratio']:.3f} \\\\"
        )
    (out_dir / "scaling_table_rows.tex").write_text("\n".join(lines) + "\n")
    print(f"\nWritten: {out_dir/'scaling_results.json'}")


if __name__ == "__main__":
    main()
