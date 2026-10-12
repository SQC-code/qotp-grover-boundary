# -*- coding: utf-8 -*-
"""E2 — null-rejection calibration for Attack 2, and Attack-4 KS check
(see ANALYSIS_PLAN.md)."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import chisquare, kstest, beta

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RES = ROOT / "SUBMISSION" / "code_and_results" / "results"

SEED = 20260929
rng = np.random.default_rng(SEED)
report = {"seed": SEED, "input_md5": {}}


def cp_ci(k, n, conf=0.95):
    lo = beta.ppf((1 - conf) / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - (1 - conf) / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


# ---------- archived attack2 aggregate ----------
af = RES / "attacks_full_results.json"
report["input_md5"]["attacks_full_results.json"] = hashlib.md5(af.read_bytes()).hexdigest()
arch = json.loads(af.read_text())["attack2"]
k_arch = round(arch["reject_rate_at_0.05"] * arch["n_tests"])
ci_arch = cp_ci(k_arch, arch["n_tests"])
report["archived"] = {"rejections": k_arch, "n_tests": arch["n_tests"],
                      "rate": arch["reject_rate_at_0.05"], "cp95": ci_arch,
                      "excludes_nominal_0.05": not (ci_arch[0] <= 0.05 <= ci_arch[1])}
print(f"[archived] {k_arch}/{arch['n_tests']} rejections, rate "
      f"{arch['reject_rate_at_0.05']:.3f}, CP95 = [{ci_arch[0]:.4f}, {ci_arch[1]:.4f}]"
      f"  excludes 0.05: {report['archived']['excludes_nominal_0.05']}")

# ---------- exact-null calibration of the same test ----------
n_bins, S = 64, 4096
Ts = list(range(2, 41, 2))
B = 2000
per_T = {}
agg_rej, agg_n = 0, 0
pvals_all = []
for T in Ts:
    ns = T * S
    counts = rng.multinomial(ns, np.full(n_bins, 1 / n_bins), size=B)
    exp = np.full(n_bins, ns / n_bins)
    stat = ((counts - exp) ** 2 / exp).sum(axis=1)
    from scipy.stats import chi2 as chi2dist
    p = chi2dist.sf(stat, n_bins - 1)
    rej = int((p < 0.05).sum())
    per_T[T] = {"size": rej / B, "cp95": cp_ci(rej, B)}
    agg_rej += rej
    agg_n += B
    pvals_all.append(p)
agg_size = agg_rej / agg_n
agg_ci = cp_ci(agg_rej, agg_n)
report["calibrated_null"] = {"per_T": per_T, "replicas_per_T": B,
                             "aggregate_size": agg_size, "aggregate_cp95": agg_ci}
print(f"[calibrated] aggregate size at alpha=.05 over {agg_n} exact-null "
      f"replicas: {agg_size:.4f}  CP95 = [{agg_ci[0]:.4f}, {agg_ci[1]:.4f}]")
# two-proportion z test: is the archived 36/1000 compatible with the
# calibrated size of the identical test under the exact null?
from scipy.stats import norm
k1, n1 = k_arch, arch["n_tests"]
k2, n2 = agg_rej, agg_n
pp = (k1 + k2) / (n1 + n2)
se = float(np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2)))
z = (k1 / n1 - k2 / n2) / se
p_two = float(2 * norm.sf(abs(z)))
report["two_proportion_vs_calibrated"] = {"z": float(z), "p_two_sided": p_two}
print(f"[verdict] test's actual size = {agg_size:.4f} (mildly conservative; "
      f"chi-squared approximation error is small at expected counts >= 128); "
      f"archived 0.036 vs calibrated size: z = {z:.2f}, two-sided p = {p_two:.3f} "
      + ("-> compatible with sampling fluctuation at the calibrated size"
         if p_two >= 0.05 else "-> NOT explained by the calibrated size alone"))
report["hypothesis_supported"] = bool(p_two >= 0.05 and agg_size < 0.05)

# p-value ECDF summary (discreteness/approximation diagnostic)
pv = np.concatenate(pvals_all)
qs = [0.01, 0.05, 0.10, 0.25, 0.50]
report["null_pvalue_cdf"] = {str(q): float((pv < q).mean()) for q in qs}
print("[diagnostic] null P(p < q): " +
      ", ".join(f"{q}: {(pv < q).mean():.4f}" for q in qs))

# ---------- Attack 4 KS discreteness check ----------
# regenerate the archived per-trial p-values (identical seeded stream)
def attack4_pvalues(seed=20240603):
    n, M, S4, T4, reps = 4, 2, 4096, 40, 200
    Nn = 1 << n
    r = np.random.default_rng(seed)
    ps = []
    for _ in range(reps):
        marked = list(r.choice(Nn, size=M, replace=False))
        keys = np.asarray(marked, dtype=np.int64)
        k = keys[r.integers(0, len(keys), size=T4 * S4)]
        a = r.integers(0, Nn, size=T4 * S4)
        out = (k ^ a).astype(np.int64)
        r.shuffle(out)
        half = (out.size // 2) * 2
        xors = out[0:half:2] ^ out[1:half:2]
        xors = xors[xors != 0]
        hist = np.bincount(xors, minlength=Nn).astype(float)[1:]
        exp = np.full(Nn - 1, xors.size / (Nn - 1))
        ps.append(chisquare(hist, exp).pvalue)
    return np.asarray(ps)

p200 = attack4_pvalues()
ks_stat, ks_p = kstest(p200, "uniform")
ties = len(p200) - len(np.unique(p200))
# calibrated KS null for 200 iid U(0,1)
B2 = 10_000
null_stats = np.empty(B2)
for i in range(B2):
    u = np.sort(rng.random(200))
    grid = np.arange(1, 201) / 200
    null_stats[i] = max(np.max(grid - u), np.max(u - (grid - 1 / 200)))
ks_p_cal = float((null_stats >= ks_stat).mean())
report["attack4_ks"] = {"ks_stat": float(ks_stat), "ks_p_asymptotic": float(ks_p),
                        "ks_p_calibrated": ks_p_cal, "ties_in_200_pvalues": int(ties),
                        "archived_ks_p": 0.867}
print(f"[attack4] KS stat={ks_stat:.4f} asymptotic p={ks_p:.3f} "
      f"calibrated p={ks_p_cal:.3f} ties={ties} (archived p=0.867)")

out = HERE / "results" / "e2_calibration.json"
out.write_text(json.dumps(report, indent=2))
print("saved:", out)
