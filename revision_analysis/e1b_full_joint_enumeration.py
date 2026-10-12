# -*- coding: utf-8 -*-
"""E1b — corrective full-joint enumeration (post-audit).

Audit finding (2026-09-28): the original e1 Arm A enumerated all 16 X-pads
(the complete X-pad set at n=4) x 3 Z-pads x 4 marked sets = 192 rows; the
manuscript text erroneously described this as '256 X-pads ... 3,072
configurations' (an arithmetic slip: 2^{2n}=256 counts JOINT (a,b) pads, not
X-pads). This script replaces the partial Z-pad sample by the FULL joint
enumeration: all 256 (a,b) in {0..15}^2 for each of the 4 marked sets
= 1,024 configurations. The original e1 archive is left untouched."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CODE = ROOT / "SUBMISSION" / "code_and_results" / "code"
sys.path.insert(0, str(CODE))

from qiskit.quantum_info import Statevector
from trapdoor_grover import QOTPGroverProtocol

N_Q, R_ITER = 4, 3
N = 2 ** N_Q
TOL = 1e-10


def pq(n, r):
    th = np.arcsin(2 ** (-n / 2))
    p = float(np.sin((2 * r + 1) * th) ** 2)
    return p, (1 - p) / (2 ** n - 1)


def exact_decrypted_law(keys, a, b):
    proto = QOTPGroverProtocol(n_qubits=N_Q, keys=keys, qotp_x=a, qotp_z=b)
    qc, n_iter = proto.build_circuit()
    assert n_iter == R_ITER
    qc = qc.remove_final_measurements(inplace=False)
    sreg = next(r for r in qc.qregs if "search" in r.name)
    six = [qc.qubits.index(q) for q in sreg]
    probs_m = Statevector(qc).probabilities(qargs=six)
    y = np.empty(N)
    for m in range(N):
        y[m ^ a] = probs_m[m]
    return y


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


p_r, q_r = pq(N_Q, R_ITER)
sets = {"M1_{9}": [9], "M2_{3,12}": [3, 12], "M3_{1,6,11}": [1, 6, 11],
        "M2_{2,15}_endtoend": [2, 15]}

report = {"n": N_Q, "r": R_ITER, "tol": TOL,
          "enumeration": "ALL joint pads (a,b) in {0..15}^2 per marked set",
          "rows_per_set": N * N, "sets": list(sets.keys()), "arm": {}}
overall = 0.0
for name, keys in sets.items():
    base = exact_decrypted_law(keys, 0, 0)
    ana = np.full(N, q_r)
    for k in keys:
        ana[k] = q_r + (p_r - q_r) / len(keys)
    mx = 0.0
    for a in range(N):
        for b in range(N):
            mx = max(mx, tv(exact_decrypted_law(keys, a, b), base))
    report["arm"][name] = {"max_TV_vs_unpadded": mx,
                           "TV_unpadded_vs_analytic": tv(base, ana)}
    overall = max(overall, mx, tv(base, ana))
    print(f"{name:22s} 256 joint pads: max TV(padded,unpadded) = {mx:.3e}  "
          f"TV(unpadded,analytic) = {tv(base, ana):.3e}")
report["total_rows"] = 4 * N * N
report["overall_max_TV"] = overall
report["pass"] = bool(overall <= TOL)
print(f"TOTAL 1,024 configurations; overall max TV = {overall:.3e} -> "
      + ("PASS" if report["pass"] else "FAIL"))
out = HERE / "results" / "e1b_full_joint.json"
out.write_text(json.dumps(report, indent=2))
print("saved:", out)
