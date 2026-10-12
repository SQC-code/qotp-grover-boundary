# -*- coding: utf-8 -*-
"""E3 — classical-shadow error vs register size (see ANALYSIS_PLAN.md).
Same estimator/metric as the archived attack5: full global local-Pauli shadow,
raw linear inversion, half-trace-norm error to I/2^n."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20260930
rng = np.random.default_rng(SEED)
Ks = [100, 1000, 10_000, 100_000]
REPS = 25
NS = [2, 3, 4, 5]
report = {"seed": SEED, "K_values": Ks, "reps": REPS, "ns": NS,
          "metric": "0.5*||rho_hat - I/2^n||_1 (raw linear inversion)"}

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
eye2 = np.eye(2, dtype=complex)


def single_qubit_snaps():
    snaps = []
    for P in (sx, sy, sz):
        for s in (+1, -1):
            b = (eye2 + s * P) / 2
            snaps.append(3 * b - eye2)
    return snaps  # order: X+,X-,Y+,Y-,Z+,Z-


def global_snaps(n):
    s1 = single_qubit_snaps()
    G = [np.array([[1]], dtype=complex)]
    for _ in range(n):
        G = [np.kron(g, s) for g in G for s in s1]
    return np.stack(G)  # (6^n, 2^n, 2^n)


def halftrace(A):
    ev = np.linalg.eigvalsh((A + A.conj().T) / 2)
    return 0.5 * float(np.abs(ev).sum())


def run_width(n, probs, true_rho, label):
    G = report["_G_cache"][n]
    d = 2 ** n
    rows = {}
    for K in Ks:
        errs = []
        for _ in range(REPS):
            counts = rng.multinomial(K, probs)
            rho_hat = np.tensordot(counts, G, axes=(0, 0)) / K
            errs.append(halftrace(rho_hat - true_rho))
        rows[K] = (float(np.mean(errs)), float(np.std(errs, ddof=1)))
    x = np.log(Ks)
    y = np.log([rows[K][0] for K in Ks])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res_, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = len(x) - 2
    se = float(np.sqrt((resid @ resid) / dof * np.linalg.inv(A.T @ A)[0, 0]))
    out = {"errors": {str(K): rows[K] for K in Ks},
           "exponent": float(coef[0]), "exponent_se": se,
           "C": float(np.exp(coef[1]))}
    print(f"[{label}] n={n}: C={out['C']:.1f}, exponent={out['exponent']:.3f}"
          f" +- {se:.3f};  err(K=1e5)={rows[100000][0]:.4f}+-{rows[100000][1]:.4f}")
    return out


report["_G_cache"] = {}
mixed = {}
for n in NS:
    report["_G_cache"][n] = global_snaps(n)
    d = 2 ** n
    probs = np.full(6 ** n, 1.0 / 6 ** n)          # maximally mixed input
    mixed[n] = run_width(n, probs, np.eye(d) / d, "mixed")
report["mixed_state"] = mixed

# PSD-projected variant at n=4 (secondary)
n = 4
d = 16
G = report["_G_cache"][n]
probs = np.full(6 ** n, 1.0 / 6 ** n)
rows = {}
for K in Ks:
    errs = []
    for _ in range(REPS):
        counts = rng.multinomial(K, probs)
        rho_hat = np.tensordot(counts, G, axes=(0, 0)) / K
        H = (rho_hat + rho_hat.conj().T) / 2
        ev, V = np.linalg.eigh(H)
        ev = np.maximum(ev, 0)
        if ev.sum() > 0:
            ev = ev / ev.sum()
        proj = (V * ev) @ V.conj().T
        errs.append(halftrace(proj - np.eye(d) / d))
    rows[str(K)] = (float(np.mean(errs)), float(np.std(errs, ddof=1)))
report["psd_projected_n4"] = rows
print("[psd n=4] " + ", ".join(f"K={k}: {v[0]:.4f}" for k, v in rows.items()))

# positive control: true state |0...0> at n=3 (non-uniform snapshot law)
n = 3
d = 8
G = report["_G_cache"][n]
# per-qubit snapshot probabilities for |0>: X+-, Y+- each 1/6; Z+ 1/3; Z- 0
p1 = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 3, 0.0])
probs = np.array([1.0])
for _ in range(n):
    probs = np.kron(probs, p1)
true = np.zeros((d, d), dtype=complex)
true[0, 0] = 1
ctrl = {}
for K in Ks:
    e_truth, e_mixed = [], []
    for _ in range(REPS):
        counts = rng.multinomial(K, probs)
        rho_hat = np.tensordot(counts, G, axes=(0, 0)) / K
        e_truth.append(halftrace(rho_hat - true))
        e_mixed.append(halftrace(rho_hat - np.eye(d) / d))
    ctrl[str(K)] = {"err_to_truth": (float(np.mean(e_truth)), float(np.std(e_truth, ddof=1))),
                    "dist_to_mixed": (float(np.mean(e_mixed)), float(np.std(e_mixed, ddof=1)))}
    print(f"[control n=3 |0..0>] K={K}: err_to_truth={np.mean(e_truth):.4f}, "
          f"dist_to_I/8={np.mean(e_mixed):.4f} (state distance = 0.875)")
report["positive_control_n3_zero_state"] = ctrl
report["positive_control_true_distance_to_mixed"] = 0.875

del report["_G_cache"]
out = HERE / "results" / "e3_shadow_scaling.json"
out.write_text(json.dumps(report, indent=2))
print("saved:", out)
