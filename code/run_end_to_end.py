"""Pad-in-circuit end-to-end experiment vs the covariance-based shortcut sampler.

For n=4, keys={2,15}: run N_INV single-shot circuits, each with a FRESH pad
(a,b) sampled from a seeded RNG and applied IN-CIRCUIT via _apply_qotp (a
distinct transpiled circuit per invocation). Independently draw N_INV samples
from the shortcut sampler (run(num_invocations=...), which evaluates the
unpadded circuit and XORs a fresh X-pad classically per outcome).

Report a two-sample multinomial homogeneity test (chi-squared on the 2 x 16
contingency table), the observed empirical total-variation distance with a
CONDITIONAL PERMUTATION test (the plug-in TV estimator is positively biased,
so a resampling CI would exclude 0 even under identical distributions — a
permutation p-value is the statistically sound report), and a uniformity
chi-squared test of the pad-in-circuit server histogram.

Everything is seeded (pads, simulator, transpiler, permutation RNG) in BOTH
arms — the shortcut arm receives an explicit seed through run(seed=...).
Two invocations of this script produce byte-identical statistics. Results go
to end_to_end_results.json, which fig_hist_overview reads (no live runs).
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import chi2_contingency, chisquare

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trapdoor_grover import QOTPGroverProtocol
from qiskit import transpile
try:
    from qiskit_aer import AerSimulator
except ImportError:
    from qiskit.providers.aer import AerSimulator

SEED = 20260716
BASIS = ["u3", "u2", "u1", "cx", "id"]
N, KEYS, N_INV = 4, [0b0010, 0b1111], 4096


def pad_in_circuit_samples(rng, sim):
    """N_INV single-shot executions, fresh (a,b) applied in-circuit each time.

    The measurement seed MUST vary per invocation (drawn from the master RNG,
    hence still fully deterministic end to end): a fixed seed_simulator would
    make every run return the same quantile of its circuit's distribution,
    correlating the shots and biasing the sample.
    """
    server, client = Counter(), Counter()
    for i in range(N_INV):
        a = int(rng.integers(0, 1 << N))
        b = int(rng.integers(0, 1 << N))
        s = int(rng.integers(1, 1 << 31))
        p = QOTPGroverProtocol(N, KEYS, qotp_x=a, qotp_z=b)
        qc, _ = p.build_circuit()
        tqc = transpile(qc, basis_gates=BASIS, optimization_level=1,
                        seed_transpiler=SEED)
        counts = sim.run(tqc, shots=1, seed_simulator=s).result().get_counts()
        m = int(next(iter(counts)), 2)
        server[m] += 1
        client[m ^ a] += 1
    return server, client


def shortcut_samples():
    """Independent, fully seeded N_INV samples via the shortcut path."""
    p = QOTPGroverProtocol(N, KEYS)
    out = p.run(num_invocations=N_INV, seed=SEED + 2)
    server = Counter({int(k, 2): v for k, v in out["server_view"].items()})
    client = Counter({int(k, 2): v for k, v in out["client_view"].items()})
    return server, client, out["integrity_ratio"]


def tv(c1, c2, n1, n2, d=16):
    return 0.5 * sum(abs(c1.get(x, 0) / n1 - c2.get(x, 0) / n2) for x in range(d))


def main():
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    sim = AerSimulator(method="statevector")
    s_circ, c_circ, = pad_in_circuit_samples(rng, sim)
    s_short, c_short, integ_short = shortcut_samples()
    integ_circ = sum(v for k, v in c_circ.items() if k in KEYS) / N_INV

    table = np.array([[s_circ.get(x, 0) for x in range(16)],
                      [s_short.get(x, 0) for x in range(16)]])
    chi2, pval, dof, _ = chi2_contingency(table)
    tv_obs = tv(s_circ, s_short, N_INV, N_INV)

    # conditional permutation test of the TV statistic (pool the 2*N_INV
    # samples, relabel uniformly at random, recompute TV): the correct null
    # reference for "are the two pipelines drawing from the same law?"
    prng = np.random.default_rng(SEED + 1)
    pooled = np.concatenate([np.repeat(np.arange(16), table[0]),
                             np.repeat(np.arange(16), table[1])])
    n_perm = 10_000
    count_ge = 0
    for _ in range(n_perm):
        prng.shuffle(pooled)
        c1 = np.bincount(pooled[:N_INV], minlength=16)
        c2 = np.bincount(pooled[N_INV:], minlength=16)
        if 0.5 * np.abs(c1 / N_INV - c2 / N_INV).sum() >= tv_obs - 1e-15:
            count_ge += 1
    tv_perm_p = count_ge / n_perm

    # uniformity of the pad-in-circuit server histogram (input-hiding check)
    uni_chi2, uni_p = chisquare(table[0])

    res = {
        "n": N, "keys": KEYS, "n_invocations": N_INV, "seed": SEED,
        "simulator": "AerSimulator(method='statevector'); per-invocation "
                     "seed_simulator; shortcut arm seeded via run(seed=SEED+2)",
        "pad_in_circuit": {
            "server_counts": {str(k): int(v) for k, v in sorted(s_circ.items())},
            "client_counts": {str(k): int(v) for k, v in sorted(c_circ.items())},
            "consistency_ratio": integ_circ,
            "uniformity_chi2": float(uni_chi2), "uniformity_dof": 15,
            "uniformity_pvalue": float(uni_p),
        },
        "shortcut": {
            "server_counts": {str(k): int(v) for k, v in sorted(s_short.items())},
            "consistency_ratio": integ_short,
        },
        "two_sample": {
            "chi2": float(chi2), "dof": int(dof), "pvalue": float(pval),
            "tv_distance": float(tv_obs),
            "tv_permutation_pvalue": float(tv_perm_p),
            "n_permutations": n_perm,
        },
        "wallclock_s": time.perf_counter() - t0,
    }
    out = Path(__file__).resolve().parent / "end_to_end_results.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"[end-to-end] two-sample chi2={chi2:.3f} (dof {dof}) p={pval:.3f}; "
          f"TV={tv_obs:.5f} permutation p={tv_perm_p:.3f}; "
          f"uniformity chi2={uni_chi2:.3f} p={uni_p:.3f}; "
          f"consistency: circuit={integ_circ:.4f} shortcut={integ_short:.4f}; "
          f"{res['wallclock_s']:.0f}s")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
