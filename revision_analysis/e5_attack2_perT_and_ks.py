# -*- coding: utf-8 -*-
"""E5 — (a) reproduce the archived Attack-2 stream to recover per-T rejection
counts (the archive stores only the aggregate 36/1000); (b) FULL-PIPELINE
calibration of the Attack-4 KS p-value (the earlier e2 calibration sampled
Uniform(0,1) p-values directly, which skips the multinomial->chi2 stage;
this script simulates the entire pipeline under the null)."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import chisquare, beta, norm

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

# ---------------- (a) archived Attack-2 stream, per-T ----------------
def server_outcomes(n, keys, n_inv, rng):
    N = 1 << n
    k = np.asarray(keys, np.int64)[rng.integers(0, len(keys), n_inv)]
    a = rng.integers(0, N, n_inv)
    return (k ^ a).astype(np.int64)

def attack2_per_T(seed=20240601):
    """Byte-identical reproduction of run_attacks_full.attack2's RNG stream."""
    n, M, S = 6, 1, 4096
    N = 1 << n
    marked = [0b101101]
    Ts = list(range(2, 41, 2))
    n_seeds = 50
    rng = np.random.default_rng(seed)
    per_T = {}
    all_p = []
    for T in Ts:
        rej = 0
        for _ in range(n_seeds):
            out = server_outcomes(n, marked, T * S, rng)
            hist = np.bincount(out, minlength=N).astype(float)
            p = chisquare(hist, np.full(N, T * S / N)).pvalue
            all_p.append(p)
            rej += int(p < 0.05)
        per_T[T] = rej
    return Ts, per_T, np.asarray(all_p)

Ts, per_T, all_p = attack2_per_T()
agg = int((all_p < 0.05).sum())
print("[a] archived-stream reproduction: aggregate rejections =",
      f"{agg}/1000 (archive: 36/1000, match={agg == 36})")
print("    per-T rejections/50:", {t: per_T[t] for t in Ts})

def cp_ci(k, n, conf=0.95):
    lo = beta.ppf((1 - conf) / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - (1 - conf) / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)

rep = {"attack2_stream_seed": 20240601,
       "aggregate_rejections": agg, "aggregate_matches_archive": bool(agg == 36),
       "per_T_rejections_of_50": {str(t): per_T[t] for t in Ts},
       "aggregate_cp95": cp_ci(agg, 1000)}

# ---------------- (b) full-pipeline KS calibration ----------------
# Pipeline under H0 (fresh pads): 200 trials, each: T*S=163840 outcomes
# uniform on {0..15} -> disjoint XOR pairs -> drop zeros -> chi2 on 15 bins
# -> p-value; then KS statistic of the 200 p-values vs U(0,1).
# The random shuffle in the original code re-orders i.i.d. uniform samples
# and leaves the pair distribution unchanged; it is omitted here for speed
# (noted in the manuscript's SI as a distribution-preserving simplification).
SEED_B = 20261001
B = 1000
n4, S4, T4, reps = 4, 4096, 40, 200
N4 = 1 << n4
tot = T4 * S4
D_OBS = 0.04151676615892964  # from e2_calibration.json (archived-stream KS stat)

rng = np.random.default_rng(SEED_B)
grid_hi = np.arange(1, reps + 1) / reps
grid_lo = grid_hi - 1.0 / reps
d_null = np.empty(B)
for i in range(B):
    ps = np.empty(reps)
    for t in range(reps):
        out = rng.integers(0, N4, tot)
        xors = out[0::2] ^ out[1::2]
        xors = xors[xors != 0]
        hist = np.bincount(xors, minlength=N4).astype(float)[1:]
        ps[t] = chisquare(hist, np.full(N4 - 1, xors.size / (N4 - 1))).pvalue
    u = np.sort(ps)
    d_null[i] = max(np.max(grid_hi - u), np.max(u - grid_lo))
p_cal = float((d_null >= D_OBS).mean())
mc_se = float(np.sqrt(p_cal * (1 - p_cal) / B))
print(f"[b] full-pipeline KS calibration: B={B}, seed={SEED_B}, "
      f"D_obs={D_OBS:.5f} -> calibrated p = {p_cal:.3f} +- {mc_se:.3f} (MC SE); "
      f"asymptotic p = 0.867")
rep["ks_full_pipeline"] = {
    "B": B, "seed": SEED_B, "D_obs": D_OBS,
    "calibrated_p": p_cal, "mc_se": mc_se,
    "null_D_quantiles": {q: float(np.quantile(d_null, q))
                         for q in (0.5, 0.9, 0.95, 0.99)},
    "note": "shuffle of i.i.d. uniforms omitted (distribution-preserving)"}

out = RES / "e5_attack2_perT_and_ks.json"
out.write_text(json.dumps(rep, indent=2))
print("saved:", out)
