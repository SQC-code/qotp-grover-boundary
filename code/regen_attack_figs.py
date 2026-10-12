"""Regenerate paper figures for Attacks 2-5 from REAL simulation data.

Replaces the previous synthetic/placeholder figures. Server-view model is the exact
computational-basis outcome under fresh QOTP: m = k XOR a (uniform); see run_attacks_full.py.
Writes fig_attack2_accum.png, fig_attack3_bayes.png, fig_attack4_xor.png,
fig_attack5_tomography.png into the LaTeX directory.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chisquare

import os
def _paper_fig_dir():
    for anc in Path(__file__).resolve().parents:
        if (anc / "main_sr.tex").exists():
            return anc
    return Path.cwd()
OUT_DIR = Path(os.environ.get("PAPER_FIG_DIR") or _paper_fig_dir())
OUT_DIR.mkdir(parents=True, exist_ok=True)
RES = Path(__file__).resolve().parent  # results JSON lives beside this script


def server_outcomes(n, marked_keys, n_inv, rng):
    N = 1 << n
    keys = np.asarray(marked_keys, dtype=np.int64)
    k = keys[rng.integers(0, len(keys), size=n_inv)]
    a = rng.integers(0, N, size=n_inv)
    return (k ^ a).astype(np.int64)


# ---- Attack 2: mean chi^2 p-value vs T ----
def fig_attack2():
    n, M, S = 6, 1, 4096
    N = 1 << n
    marked = [0b101101]
    Ts = list(range(2, 41, 2))
    n_seeds = 50
    rng = np.random.default_rng(20240601)
    mean_p, lo, hi = [], [], []
    for T in Ts:
        ps = []
        for _ in range(n_seeds):
            out = server_outcomes(n, marked, T * S, rng)
            hist = np.bincount(out, minlength=N).astype(float)
            ps.append(chisquare(hist, np.full(N, T * S / N)).pvalue)
        ps = np.asarray(ps)
        mean_p.append(ps.mean())
        lo.append(np.percentile(ps, 10))
        hi.append(np.percentile(ps, 90))
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.plot(Ts, mean_p, "-s", color="C0", lw=2, ms=4, label="Mean $p$-value")
    ax.fill_between(Ts, lo, hi, color="C0", alpha=0.15, label="10--90th percentile")
    ax.axhline(0.05, color="C3", lw=0.8, ls="--", label="$\\alpha = 0.05$")
    ax.set_xlabel("Number of batches $T$")
    ax.set_ylabel("Chi-squared $p$-value")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_attack2_accum.png", dpi=300)
    plt.close(fig)
    print("fig_attack2_accum.png written")


# ---- Attack 3: empirical KL vs n, with estimator floor ----
def fig_attack3():
    # single source of truth: plot the ARCHIVED per-cell KL values from the
    # results JSON (run_attacks_full.attack3), rather than re-simulating.
    res = json.loads((RES / "attacks_full_results.json").read_text())["attack3"]
    per_cell = res["per_cell_kl"]
    n_samples = int(res["samples_per_cell"])
    ns = list(range(int(res["n_range"][0]), int(res["n_range"][1]) + 1))
    Ms = [int(m) for m in res["M_values"]]
    markers = {1: "o", 2: "s", 4: "^"}
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    for M in Ms:
        xs = [n for n in ns if f"n{n}_M{M}" in per_cell]
        ys = [per_cell[f"n{n}_M{M}"] for n in xs]
        ax.semilogy(xs, ys, markers[M] + "-", lw=1.5, ms=5, label=f"$M={M}$")
    floor = [(2**n - 1) / (2 * n_samples) for n in ns]
    ax.semilogy(ns, floor, "k--", lw=1.0, label="$(2^n-1)/2N_s$ floor")
    ax.set_xlabel("Qubit width $n$")
    ax.set_ylabel("$D_{\\mathrm{KL}}(\\hat{p}\\,\\|\\,\\mathrm{unif})$ [nats]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_attack3_bayes.png", dpi=300)
    plt.close(fig)
    print("fig_attack3_bayes.png written")


# ---- Attack 4: empirical XOR distribution vs uniform ----
def fig_attack4():
    n, M, S, T = 4, 2, 4096, 40
    N = 1 << n
    rng = np.random.default_rng(20240603)
    marked = list(rng.choice(N, size=M, replace=False))
    out = server_outcomes(n, marked, T * S, rng)
    rng.shuffle(out)
    half = (out.size // 2) * 2
    xors = out[0:half:2] ^ out[1:half:2]
    xors = xors[xors != 0]
    hist = np.bincount(xors, minlength=N).astype(float)[1:]
    emp = hist / hist.sum()
    uniform = np.full(N - 1, 1.0 / (N - 1))
    x = np.arange(1, N)
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.bar(x - 0.18, uniform, width=0.35, color="C0", alpha=0.7, label="Uniform")
    ax.bar(x + 0.18, emp, width=0.35, color="C1", alpha=0.7, label="Empirical XOR")
    ax.set_xlabel("XOR value $x_i \\oplus x_j$ (nonzero)")
    ax.set_ylabel("Probability")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_attack4_xor.png", dpi=300)
    plt.close(fig)
    print("fig_attack4_xor.png written")


# ---- Attack 5: half-trace-norm reconstruction error vs K (loglog), mean +- 1 SD ----
def fig_attack5():
    # single source of truth: read mean and per-K std from the results JSON
    res = json.loads((RES / "attacks_full_results.json").read_text())["attack5"]
    Ks = [int(k) for k in res["K_values"]]
    means = np.array([res["trace_norm_error_by_K"][str(k)] for k in Ks])
    stds = np.array([res["std_by_K"][str(k)] for k in Ks])
    slope, intercept = np.polyfit(np.log(Ks), np.log(means), 1)
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.errorbar(Ks, means, yerr=stds, fmt="-o", color="C5", lw=2, ms=5,
                capsize=3, elinewidth=1,
                label="$\\frac{1}{2}\\|\\hat\\rho - I/2^n\\|_1$ (mean $\\pm$1 SD, 25 reps)")
    ax.loglog(Ks, np.exp(intercept) * np.array(Ks) ** slope, "--", color="grey", lw=0.9,
              label=f"$\\sim K^{{{slope:.2f}}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Tomography copies $K$")
    ax.set_ylabel("Half-trace-norm reconstruction error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_attack5_tomography.png", dpi=300)
    plt.close(fig)
    print(f"fig_attack5_tomography.png written (exponent {slope:.2f}, with error bars)")


if __name__ == "__main__":
    fig_attack2()
    fig_attack3()
    fig_attack4()
    fig_attack5()
    print("All attack figures regenerated from real data.")
