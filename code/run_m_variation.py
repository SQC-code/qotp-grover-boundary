"""Local M-variation sweep at n=8 with client/server resource split."""
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


def keys_for_M(N, M):
    step = max(1, N // M)
    return [1 + i * step for i in range(M) if 1 + i * step < N]


def tmetrics(qc):
    tqc = transpile(qc, basis_gates=BASIS, optimization_level=2,
                    seed_transpiler=SEED_TRANSPILER)
    counts = dict(tqc.count_ops())
    cx = counts.get("cx", 0)
    oneq = sum(v for k, v in counts.items() if k in ("u1", "u2", "u3", "id"))
    return tqc.depth(), cx, oneq, tqc


def main():
    n = 8
    N = 2 ** n
    results = []
    # one fixed, recorded pad shared by every M row (reproducible 1Q counts)
    pad_rng = np.random.default_rng(SEED_TRANSPILER)
    QX = int(pad_rng.integers(0, N))
    QZ = int(pad_rng.integers(0, N))
    for M in [1, 2, 4, 8]:
        keys = keys_for_M(N, M)
        M_actual = len(keys)
        print(f"--- n={n} M={M_actual} keys={keys} pad=({QX},{QZ}) ---", flush=True)
        p = QOTPGroverProtocol(n_qubits=n, keys=keys, qotp_x=QX, qotp_z=QZ)
        qc, n_iter = p.build_circuit()
        depth, cx, oneq, tqc = tmetrics(qc)
        # client-only circuit (prep + QOTP)
        key = QuantumRegister(n, "key")
        qcc = QuantumCircuit(key)
        p._prepare_kappa(qcc, key)
        p._apply_qotp(qcc, key)
        _, cx_c, _, _ = tmetrics(qcc)
        # server-only circuit
        search = QuantumRegister(n, "search")
        key2 = QuantumRegister(n, "key")
        cm = ClassicalRegister(n, "meas")
        qcs = QuantumCircuit(search, key2, cm)
        qcs.h(search)
        for _ in range(n_iter):
            p._apply_V(qcs, search, key2)
            p._apply_diffuser(qcs, search)
        qcs.measure(search, cm)
        _, cx_s, _, _ = tmetrics(qcs)
        sim = AerSimulator(method="statevector", seed_simulator=SEED_TRANSPILER)
        t0 = time.perf_counter()
        res = sim.run(tqc, shots=2048).result()
        sim_t = time.perf_counter() - t0
        server_view = res.get_counts()
        client_view = p.decrypt(server_view)
        marked = sum(c for bs, c in client_view.items() if int(bs, 2) in p.keys)
        integrity = marked / 2048
        rec = {
            "n": n, "N": N, "M": M_actual, "n_iter": n_iter,
            "depth": tqc.depth(),
            "cx_count": int(cx),
            "client_cx": int(cx_c),
            "server_cx": int(cx_s),
            "single_qubit_count": int(oneq),
            "sim_time_s": float(sim_t),
            "integrity_ratio": float(integrity),
            "qotp_pad": [QX, QZ],
            "seed_transpiler": SEED_TRANSPILER, "seed_simulator": SEED_TRANSPILER,
            "aer_method": "statevector", "opt_level": 2, "shots": 2048,
        }
        results.append(rec)
        print(f"  depth={rec['depth']} CX={rec['cx_count']} (client {cx_c} / server {cx_s}) "
              f"r={n_iter} sim={sim_t:.2f}s int={integrity:.3f}", flush=True)

    out_dir = Path(__file__).resolve().parent
    (out_dir / "m_variation_results.json").write_text(json.dumps(results, indent=2))
    lines = []
    for r in results:
        lines.append(
            f"{r['M']} & {r['n_iter']} & {r['depth']} & {r['cx_count']} & "
            f"{r['single_qubit_count']} & {r['sim_time_s']:.2f} & {r['integrity_ratio']:.3f} \\\\"
        )
    (out_dir / "m_variation_table_rows.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
