# -*- coding: utf-8 -*-
"""E4 — M-sweep / scaling-table precision recheck (see ANALYSIS_PLAN.md)."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RES = ROOT / "SUBMISSION" / "code_and_results" / "results"
report = {"input_md5": {}}


def cp_ci(k, n, conf=0.95):
    lo = beta.ppf((1 - conf) / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - (1 - conf) / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def theory(n, r, M):
    N = 2 ** n
    th = np.arcsin(1 / np.sqrt(N))
    p = float(np.sin((2 * r + 1) * th) ** 2)
    q = (1 - p) / (N - 1)
    return p + (M - 1) * q


# ---------- M sweep ----------
mf = RES / "m_variation_results.json"
report["input_md5"]["m_variation_results.json"] = hashlib.md5(mf.read_bytes()).hexdigest()
mv = json.loads(mf.read_text())
rows = mv if isinstance(mv, list) else mv.get("results", mv)
msweep = []
print("=== M-sweep (n=8, r=12) ===")
print(f"theory p_r (n=8, r=12) = {theory(8, 12, 1):.12f}")
for row in rows:
    M = row.get("M") or row.get("m")
    shots = row.get("shots", 2048)
    ratio = row.get("integrity_ratio", row.get("consistency"))
    k = ratio * shots
    integral = abs(k - round(k)) < 1e-9
    k = int(round(k))
    lo, hi = cp_ci(k, shots)
    th = theory(8, row.get("n_iter", 12), M)
    msweep.append({"M": M, "marked": k, "shots": shots, "ratio_6dp": round(k / shots, 6),
                   "ratio_integral": bool(integral), "cp95": [lo, hi],
                   "theory_6dp": round(th, 6), "theory_12dp": th})
    print(f"M={M}: {k}/{shots} = {k/shots:.6f}  CP95=[{lo:.6f},{hi:.6f}]  "
          f"theory={th:.12f}  integral={integral}")

# ---------- scaling table ----------
sf = RES / "scaling_results.json"
report["input_md5"]["scaling_results.json"] = hashlib.md5(sf.read_bytes()).hexdigest()
sc = json.loads(sf.read_text())
scaling = []
print("=== scaling (M=2) ===")
for row in sc:
    n, r, shots, n_pads = row["n"], row["n_iter"], row["shots"], row["n_pads"]
    th = theory(n, r, 2)
    entry = {"n": n, "r": r, "theory_12dp": th, "theory_6dp": round(th, 6),
             "n_pads": n_pads, "shots": shots}
    if n_pads == 1:
        k = row["integrity_ratio"] * shots
        integral = abs(k - round(k)) < 1e-9
        k = int(round(k))
        lo, hi = cp_ci(k, shots)
        entry.update({"marked": k, "ratio_6dp": round(k / shots, 6),
                      "ratio_integral": bool(integral), "cp95": [lo, hi]})
        print(f"n={n:2d}: {k}/{shots} = {k/shots:.6f}  CP95=[{lo:.6f},{hi:.6f}]  "
              f"theory={th:.9f}  integral={integral}")
    else:
        raw = row.get("integrity_raw")
        ks = [int(round(x * shots)) for x in raw]
        pooled_k, pooled_n = sum(ks), shots * n_pads
        lo, hi = cp_ci(pooled_k, pooled_n)
        entry.update({"per_pad_marked": ks, "mean_ratio_6dp": round(np.mean(raw), 6),
                      "std_ratio_6dp": round(float(np.std(raw, ddof=1)), 6),
                      "pooled_cp95": [lo, hi]})
        print(f"n={n:2d}: mean {np.mean(raw):.6f} +- {np.std(raw, ddof=1):.6f} "
              f"({n_pads} pads)  pooled CP95=[{lo:.6f},{hi:.6f}]  theory={th:.9f}")
    scaling.append(entry)

report["msweep"] = msweep
report["scaling"] = scaling
out = HERE / "results" / "e4_precision.json"
out.write_text(json.dumps(report, indent=2))
print("saved:", out)
