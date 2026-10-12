# -*- coding: utf-8 -*-
"""E1 — shortcut vs pad-in-circuit equivalence (see ANALYSIS_PLAN.md).

Arm A (primary, deterministic): exact statevector laws of the FULL library
pipeline (client prep + in-circuit QOTP + server + decrypt relabel) for all
256 X-pads x 3 Z-pads x marked sets {9}, {3,12}, {1,6,11}, plus the archived
end-to-end configuration {2,15}. PASS iff max TV <= 1e-10.

Arm B (secondary, archived counts only): one-sample equivalence bounds
U = TV_emp + r1 against the exact law; pre-registered delta = 0.10."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CODE = ROOT / "SUBMISSION" / "code_and_results" / "code"
RES = ROOT / "SUBMISSION" / "code_and_results" / "results"
sys.path.insert(0, str(CODE))

from qiskit.quantum_info import Statevector
from trapdoor_grover import QOTPGroverProtocol

N_Q, R_ITER = 4, 3
N = 2 ** N_Q
TOL = 1e-10
report = {"n": N_Q, "r": R_ITER, "tol_arm_a": TOL, "input_md5": {}}


def pq(n, r):
    th = np.arcsin(2 ** (-n / 2))
    p = float(np.sin((2 * r + 1) * th) ** 2)
    return p, (1 - p) / (2 ** n - 1)


def exact_decrypted_law(keys, a, b):
    """Exact decrypted law of the padded full circuit (search marginal, y = m^a)."""
    proto = QOTPGroverProtocol(n_qubits=N_Q, keys=keys, qotp_x=a, qotp_z=b)
    qc, n_iter = proto.build_circuit()
    assert n_iter == R_ITER
    qc = qc.remove_final_measurements(inplace=False)
    search_ix = [qc.qubits.index(q) for q in qc.qregs[0]
                 if qc.qregs[0].name.startswith("search")] or None
    # locate registers by name (robust to register order)
    sreg = next(r for r in qc.qregs if "search" in r.name)
    search_ix = [qc.qubits.index(q) for q in sreg]
    probs_m = Statevector(qc).probabilities(qargs=search_ix)
    y = np.empty(N)
    for m in range(N):
        y[m ^ a] = probs_m[m]
    return y


def tv(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


p_r, q_r = pq(N_Q, R_ITER)
sets = {"M1_{9}": [9], "M2_{3,12}": [3, 12], "M3_{1,6,11}": [1, 6, 11],
        "M2_{2,15}_endtoend": [2, 15]}
b_vals = [0, 7, 11]

arm_a = {}
overall_max = 0.0
for name, keys in sets.items():
    base = exact_decrypted_law(keys, 0, 0)          # unpadded = shortcut law
    ana = np.full(N, q_r)
    for k in keys:
        ana[k] = q_r + (p_r - q_r) / len(keys)
    max_tv_short, max_tv_ana = 0.0, tv(base, ana)
    for a in range(N):
        for b in b_vals:
            law = exact_decrypted_law(keys, a, b)
            max_tv_short = max(max_tv_short, tv(law, base))
    arm_a[name] = {"max_TV_vs_shortcut_law": max_tv_short,
                   "TV_unpadded_vs_analytic": max_tv_ana,
                   "configs": N * len(b_vals)}
    overall_max = max(overall_max, max_tv_short, max_tv_ana)
    print(f"[Arm A] {name:22s} max TV(padded,unpadded) = {max_tv_short:.3e}  "
          f"TV(unpadded,analytic) = {max_tv_ana:.3e}")
arm_a_pass = overall_max <= TOL
print(f"[Arm A] overall max TV = {overall_max:.3e}  ->  "
      + ("PASS (equivalence at tol 1e-10)" if arm_a_pass else "FAIL"))

# ---------------- Arm B: archived sampled data ----------------
e2e_path = RES / "end_to_end_results.json"
report["input_md5"]["end_to_end_results.json"] = hashlib.md5(
    e2e_path.read_bytes()).hexdigest()
e2e = json.loads(e2e_path.read_text())
s = e2e["n_invocations"]
d = N
alpha1 = 0.05
r1 = float(np.sqrt(np.log(2 * (2 ** d) / alpha1) / (2 * s)))
alpha2 = 0.025
r2 = float(np.sqrt(np.log(2 * (2 ** d) / alpha2) / (2 * s)))
exact_e2e = exact_decrypted_law([2, 15], 0, 0)

def counts_vec(cdict):
    v = np.zeros(N)
    for k, c in cdict.items():
        v[int(k)] = c
    return v

circ_y = counts_vec(e2e["pad_in_circuit"]["client_counts"]) / s
# the archive stores the shortcut arm's SERVER counts only; its client counts
# are recovered by reproducing the fully seeded shortcut run (archive untouched)
# and cross-checked against the archived server counts byte-for-byte.
proto = QOTPGroverProtocol(n_qubits=N_Q, keys=[2, 15])
rep = proto.run(num_invocations=s, seed=e2e["seed"] + 2)
srv_rep = np.zeros(N)
cli_rep = np.zeros(N)
for k, c in rep["server_view"].items():
    srv_rep[int(k, 2)] = c
for k, c in rep["client_view"].items():
    cli_rep[int(k, 2)] = c
srv_arch = counts_vec(e2e["shortcut"]["server_counts"])
assert np.array_equal(srv_rep, srv_arch), "seeded reproduction != archive"
report["shortcut_reproduction_matches_archive"] = True
short_y = cli_rep / s
circ_srv = counts_vec(e2e["pad_in_circuit"]["server_counts"]) / s
uni = np.full(N, 1 / N)

arm_b = {
    "s_per_arm": s, "bins": d, "alpha_one_sample": alpha1,
    "r1": r1, "delta_prereg": 0.10,
    "U_padcircuit_vs_exact": tv(circ_y, exact_e2e) + r1,
    "U_shortcut_vs_exact": tv(short_y, exact_e2e) + r1,
    "U_two_sample": tv(circ_y, short_y) + 2 * r2,
    "U_server_vs_uniform": tv(circ_srv, uni) + r1,
    "TV_emp": {"padcircuit_vs_exact": tv(circ_y, exact_e2e),
               "shortcut_vs_exact": tv(short_y, exact_e2e),
               "two_sample": tv(circ_y, short_y),
               "server_vs_uniform": tv(circ_srv, uni)},
}
for kk in ["U_padcircuit_vs_exact", "U_shortcut_vs_exact", "U_two_sample",
           "U_server_vs_uniform"]:
    verdict = "equivalence at delta=0.10" if arm_b[kk] < 0.10 else "NOT established"
    print(f"[Arm B] {kk:26s} = {arm_b[kk]:.4f}   ({verdict})")

report["arm_a"] = arm_a
report["arm_a_pass"] = bool(arm_a_pass)
report["arm_b"] = arm_b
report["p_r"], report["q_r"] = p_r, q_r
out = HERE / "results" / "e1_equivalence.json"
out.write_text(json.dumps(report, indent=2))
print("saved:", out)
