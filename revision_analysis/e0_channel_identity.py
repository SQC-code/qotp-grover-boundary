# -*- coding: utf-8 -*-
"""E0 — verify the channel identity P(m|rho) = q_r + (p_r - q_r) rho_mm on the
ACTUAL library circuit, incl. all r >= 0 and entangled inputs (see ANALYSIS_PLAN.md)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CODE = ROOT / "SUBMISSION" / "code_and_results" / "code"
sys.path.insert(0, str(CODE))

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator, random_statevector, random_density_matrix
from trapdoor_grover import QOTPGroverProtocol

SEED = 20260928
rng = np.random.default_rng(SEED)
TOL = 1e-10
report = {"seed": SEED, "tol": TOL, "checks": [], "code_md5": {}}

for f in ["trapdoor_grover.py"]:
    report["code_md5"][f] = hashlib.md5((CODE / f).read_bytes()).hexdigest()


def r_star(n):
    return max(1, int(round(np.pi / (4 * np.arcsin(2 ** (-n / 2))) - 0.5)))


def pq(n, r):
    N = 2 ** n
    th = np.arcsin(1 / np.sqrt(N))
    p = float(np.sin((2 * r + 1) * th) ** 2)
    return p, (1 - p) / (N - 1)


def server_unitary(n, r):
    """Exact unitary of the library's server circuit (H^n on search; r x (V, D)).

    Circuit qubit order: search[0..n-1] then key[0..n-1]  ->  joint index
    (k << n) | s with bit i of s = search[i]."""
    proto = QOTPGroverProtocol(n_qubits=n, keys=[1, 2 ** n - 1])
    search = QuantumRegister(n, "search")
    key = QuantumRegister(n, "key")
    qc = QuantumCircuit(search, key)
    qc.h(search)
    for _ in range(r):
        proto._apply_V(qc, search, key)
        proto._apply_diffuser(qc, search)
    return Operator(qc).data


def outcome_law(U, n, rho_key):
    """P(m) for key rho, measuring search; key traced out.

    U includes the circuit's own H^n preparation, so the search register
    enters in |0...0> (bug guard: feeding |+>^n here double-applies H)."""
    N = 2 ** n
    w, vecs = np.linalg.eigh(rho_key)
    P = np.zeros(N)
    for lam, col in zip(w, vecs.T):
        if lam < 1e-14:
            continue
        joint = np.zeros(N * N, dtype=complex)
        joint[(np.arange(N) << n) | 0] = col  # search = |0>^n, key = col
        fin = (U @ joint).reshape(N, N)  # [k, s]
        P += lam * (np.abs(fin) ** 2).sum(axis=0)
    return P


def check(name, maxdev):
    ok = bool(maxdev < TOL)
    report["checks"].append({"name": name, "max_dev": float(maxdev), "pass": ok})
    print(("PASS" if ok else "FAIL") + f"  {name:58s} max|dev| = {maxdev:.3e}")
    return ok


all_ok = True
for n in [2, 3, 4]:
    N = 2 ** n
    rs = sorted({0, 1, r_star(n), 2 * r_star(n) + 3})
    for r in rs:
        U = server_unitary(n, r)
        p, q = pq(n, r)
        # 1) computational basis inputs
        dev = 0.0
        for k in range(N):
            rho = np.zeros((N, N), dtype=complex)
            rho[k, k] = 1
            P = outcome_law(U, n, rho)
            ref = np.full(N, q)
            ref[k] = p
            dev = max(dev, np.abs(P - ref).max())
        all_ok &= check(f"n={n} r={r} basis states", dev)
        # 2) Haar pure + random mixed
        dev = 0.0
        for _ in range(12):
            psi = random_statevector(N, seed=int(rng.integers(1 << 31))).data
            rho = np.outer(psi, psi.conj())
            P = outcome_law(U, n, rho)
            ref = q + (p - q) * np.real(np.diag(rho))
            dev = max(dev, np.abs(P - ref).max())
        for _ in range(6):
            rho = random_density_matrix(N, seed=int(rng.integers(1 << 31))).data
            P = outcome_law(U, n, rho)
            ref = q + (p - q) * np.real(np.diag(rho))
            dev = max(dev, np.abs(P - ref).max())
        all_ok &= check(f"n={n} r={r} Haar pure + mixed", dev)
    # 3) padded marked-set states at r*(n)
    r = r_star(n)
    U = server_unitary(n, r)
    p, q = pq(n, r)
    keys = [2 % N, N - 1] if n > 1 else [0, 1]
    kappa = np.zeros(N, dtype=complex)
    for k in set(keys):
        kappa[k] = 1
    kappa /= np.linalg.norm(kappa)
    dev = 0.0
    for _ in range(3):
        a = int(rng.integers(0, N))
        b = int(rng.integers(0, N))
        psi = kappa.copy()
        # Z^b then X^a (library order): Z phases on |k>, then relabel k -> k^a
        phases = np.array([(-1) ** (bin(k & b).count("1")) for k in range(N)])
        psi = psi * phases
        psi = np.array([psi[m ^ a] for m in range(N)])
        rho = np.outer(psi, psi.conj())
        P = outcome_law(U, n, rho)
        ref = q + (p - q) * np.real(np.diag(rho))
        dev = max(dev, np.abs(P - ref).max())
    all_ok &= check(f"n={n} r={r} padded |kappa> (3 pads)", dev)

# 4) marked-set sanity anchor (endianness guard): n=4 keys {2,15} r=3 pad(0,0)
n, r = 4, 3
N = 16
U = server_unitary(n, r)
p, q = pq(n, r)
kappa = np.zeros(N, dtype=complex)
kappa[2] = kappa[15] = 1 / np.sqrt(2)
P = outcome_law(U, n, np.outer(kappa, kappa.conj()))
anchor = P[2] + P[15]
dev = abs(anchor - (p + q))
all_ok &= check("anchor: n=4 keys{2,15} r=3  P(Y in keys) = p_r + q_r", dev)
report["anchor_value"] = {"P_marked": float(anchor), "p_plus_q": float(p + q)}

# 5) entangled inputs: honest cq ensemble vs mimic cq ensemble
for n in [2, 3]:
    N = 2 ** n
    for r in [0, r_star(n), 2 * r_star(n) + 3]:
        U = server_unitary(n, r)
        p, q = pq(n, r)
        R = np.where(np.eye(N) == 1, p, q)  # R[m,k] = p if m==k else q
        dev = 0.0
        pdev = 0.0
        for _ in range(5):
            psi = random_statevector(N * N, seed=int(rng.integers(1 << 31))).data
            psi_rk = psi.reshape(N, N)  # [ref, key]
            # honest: joint (ref, key, search); U acts on (key, search) part.
            # U includes the H^n preparation, so search enters in |0...0>.
            e0 = np.zeros(N, dtype=complex)
            e0[0] = 1.0
            joint = np.einsum("rk,s->rks", psi_rk, e0).reshape(N, N * N)
            fin = joint @ U.T  # apply U to (k,s) index: rows=ref
            fin = fin.reshape(N, N, N)  # [ref, key, search]
            for m in range(N):
                B = fin[:, :, m]  # ref x key
                rho_h = B @ B.conj().T  # subnormalised
                rho_m = np.zeros((N, N), dtype=complex)
                for k in range(N):
                    u = psi_rk[:, k]
                    rho_m += R[m, k] * np.outer(u, u.conj())
                d = rho_h - rho_m
                ev = np.linalg.eigvalsh((d + d.conj().T) / 2)
                dev = max(dev, float(np.abs(ev).sum()))
                pdev = max(pdev, abs(np.trace(rho_h).real - np.trace(rho_m).real))
        all_ok &= check(f"n={n} r={r} entangled cq ensemble honest = mimic", max(dev, pdev))

report["all_pass"] = bool(all_ok)
out = HERE / "results" / "e0_channel_identity.json"
out.write_text(json.dumps(report, indent=2))
print("\nALL PASS:", all_ok)
print("saved:", out)
