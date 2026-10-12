"""Real simulation for Attack 1 (pad-reuse leak) and Attack 6 (integrity check),
regenerating fig_attack1_reuse.png and fig_attack6_cheating.png from actual runs.

Attack 1: numpy fixed-pad model. Under pad reuse the server outcomes are m = k XOR a
with a FIXED across queries, so they concentrate on the M-element masked support
{k XOR a}. The attacker recovers the pad-invariant difference set from the observed
peaks; detection rate rises with the number of reused-pad single-shot queries T.

Attack 6: real qiskit runs via TrapdoorGroverProtocol. Honest execution vs three
server-deviation classes (identity replacement, Haar-random unitary, biased diffuser);
integrity ratio = marked fraction after client decryption.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import random_unitary
try:
    from qiskit_aer import AerSimulator
except ImportError:
    from qiskit.providers.aer import AerSimulator
from trapdoor_grover import TrapdoorGroverProtocol

import os
def _paper_fig_dir():
    for anc in Path(__file__).resolve().parents:
        if (anc / "main_sr.tex").exists():
            return anc
    return Path.cwd()
OUT_DIR = Path(os.environ.get("PAPER_FIG_DIR") or _paper_fig_dir())
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = {}


# ----------------------------------------------------------------------
# Attack 1 -- pad reuse: detection rate vs number of reused-pad queries T
# ----------------------------------------------------------------------
def _grover_outcome(branch_targets, p_r, N, rng):
    """Real finite-r single-target Grover outcome per query: the branch target with
    probability p_r, else a uniformly random non-target value (probability q_r each)."""
    y = branch_targets.copy()
    hit = rng.random(len(branch_targets)) < p_r
    for i in range(len(branch_targets)):
        if not hit[i]:
            o = int(rng.integers(0, N - 1))
            y[i] = o if o < branch_targets[i] else o + 1  # uniform over non-targets
    return y


def attack1(seed=20240600):
    n, M = 4, 2
    N = 1 << n
    # real finite-r single-target Grover success probability per branch
    r = max(1, int(round(np.pi / (4 * np.arcsin(1 / np.sqrt(N))) - 0.5)))
    theta = np.arcsin(1 / np.sqrt(N))
    p_r = float(np.sin((2 * r + 1) * theta) ** 2)
    Ts = list(range(2, 41, 2))
    n_trials = 400
    rng = np.random.default_rng(seed)
    det_reuse, det_fresh = [], []
    for T in Ts:
        succ_reuse = succ_fresh = 0
        for _ in range(n_trials):
            keys = rng.choice(N, size=M, replace=False)
            true_s = int(keys[0]) ^ int(keys[1])  # pad-invariant difference
            # --- pad REUSE: single fixed pad a across all T single-shot queries ---
            a = int(rng.integers(0, N))
            branches = keys[rng.integers(0, M, size=T)]
            y = _grover_outcome(branches, p_r, N, rng)   # real finite-r outcome
            outs = (y ^ a)
            vals, counts = np.unique(outs, return_counts=True)
            top = vals[np.argsort(-counts)[:M]]
            # attacker's recovered difference from the two dominant peaks
            if len(top) >= 2 and (int(top[0]) ^ int(top[1])) == true_s:
                succ_reuse += 1
            # --- FRESH pad control: independent a per query -> uniform, no stable pair ---
            a_fresh = rng.integers(0, N, size=T)
            y_f = _grover_outcome(keys[rng.integers(0, M, size=T)], p_r, N, rng)
            outs_f = y_f ^ a_fresh
            vals_f, counts_f = np.unique(outs_f, return_counts=True)
            top_f = vals_f[np.argsort(-counts_f)[:M]]
            if len(top_f) >= 2 and (int(top_f[0]) ^ int(top_f[1])) == true_s:
                succ_fresh += 1
        det_reuse.append(succ_reuse / n_trials)
        det_fresh.append(succ_fresh / n_trials)
    res = {
        "n": n, "M": M, "r": r, "p_r": p_r, "T_range": [Ts[0], Ts[-1]], "trials": n_trials,
        "model": "real finite-r Grover (target w.p. p_r, else uniform non-target)",
        "detection_reuse": det_reuse, "detection_fresh": det_fresh,
        "reuse_at_T10": float(det_reuse[Ts.index(10)]),
        "reuse_at_T40": float(det_reuse[-1]),
        "fresh_baseline_mean": float(np.mean(det_fresh)),
    }
    RESULTS["attack1"] = res
    # figure
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.plot(Ts, det_reuse, "-o", color="C3", lw=2, ms=4, label="Pad-reuse attacker")
    ax.plot(Ts, det_fresh, "--s", color="grey", lw=1.2, ms=3, label="Fresh-pad control")
    ax.axhline(0.95, color="C2", lw=0.7, ls=":", label="0.95 reference")
    ax.set_xlabel("Number of reused-pad queries $T$")
    ax.set_ylabel("Difference-recovery rate")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_attack1_reuse.png", dpi=300)
    plt.close(fig)
    print(f"[Attack 1] reuse detection @T=10: {res['reuse_at_T10']:.3f}; "
          f"fresh-pad control mean: {res['fresh_baseline_mean']:.3f}; fig written")
    return res


# ----------------------------------------------------------------------
# Attack 6 -- integrity check: honest vs deviation classes (real qiskit)
# ----------------------------------------------------------------------
def _integrity_from_server_view(proto, server_view, shots):
    client = proto.decrypt(server_view)
    marked = sum(c for bs, c in client.items() if int(bs, 2) in proto.keys)
    return marked / shots


def _run_deviation(proto, kind, num_invocations, sim, rng_seed):
    """Integrity ratio of a server-deviation class under FRESH pad per invocation.

    identity/haar act only on the search register, so their raw output is
    independent of the key-register pad; a fresh X-pad a per invocation then
    randomises the decrypted value -> uniform -> integrity ~ M/N. biased uses the
    comparator (key register), so it is pad-covariant and its decrypted integrity
    is pad-independent (one representative pad suffices).
    """
    n = proto.n
    N = 1 << n
    rng = np.random.default_rng(rng_seed)
    if kind == "mimic":
        # Exact output mimicry (paper Prop. 'Exact output mimicry'): with prob
        # lam = p_r - q_r measure the encrypted key register (decrypted y is then
        # uniform over keys, i.e. always marked); with prob 1-lam return uniform
        # noise (decrypted y uniform over {0,1}^n). This reproduces the honest
        # decrypted output law EXACTLY (TVD = 0), without any Grover iteration.
        th = np.arcsin(1 / np.sqrt(N))
        p_r = np.sin((2 * proto.n_iter + 1) * th) ** 2
        q_r = (1 - p_r) / (N - 1)
        lam = p_r - q_r
        keys = set(proto.keys)
        marked = 0
        for _ in range(num_invocations):
            if rng.random() < lam:
                marked += 1                    # y uniform over keys -> marked
            elif int(rng.integers(0, N)) in keys:
                marked += 1                    # uniform-noise branch
        return marked / num_invocations
    if kind in ("identity", "haar"):
        # search-only deviation, key register irrelevant to the output
        search = QuantumRegister(n, "search")
        c = ClassicalRegister(n, "meas")
        qc = QuantumCircuit(search, c)
        qc.h(search)
        if kind == "haar":
            qc.append(random_unitary(2 ** n, seed=rng_seed).to_instruction(), list(search))
        qc.measure(search, c)
        tqc = transpile(qc, sim, optimization_level=1, seed_transpiler=rng_seed)
        mem = sim.run(tqc, shots=num_invocations, memory=True,
                      seed_simulator=rng_seed).result().get_memory()
        marked = 0
        for bs in mem:
            v = int(bs, 2)
            a = int(rng.integers(0, N))        # fresh pad per invocation
            if (v ^ a) in proto.keys:
                marked += 1
        return marked / num_invocations
    # biased diffuser: pad-covariant, integrity pad-independent
    search = QuantumRegister(n, "search")
    key = QuantumRegister(n, "key")
    c = ClassicalRegister(n, "meas")
    qc = QuantumCircuit(search, key, c)
    proto._prepare_kappa(qc, key)
    proto._apply_qotp(qc, key)
    qc.h(search)
    for _ in range(proto.n_iter):
        proto._apply_V(qc, search, key)
        qc.h(search)
        qc.h(search[-1])
        if n > 1:
            qc.mcx([search[i] for i in range(n - 1)], search[-1])
        else:
            qc.z(search[0])
        qc.h(search[-1])
        qc.h(search)
    qc.measure(search, c)
    tqc = transpile(qc, sim, optimization_level=1, seed_transpiler=rng_seed)
    sv = sim.run(tqc, shots=num_invocations,
                 seed_simulator=rng_seed).result().get_counts()
    return _integrity_from_server_view(proto, sv, num_invocations)


def attack6(seed=20240605):
    n, M, num_invocations = 4, 2, 4096
    keys = [0b1011, 0b0110]
    n_trials = 100
    sim = AerSimulator()
    rng = np.random.default_rng(seed)
    data = {"honest": [], "identity": [], "haar": [], "biased": [], "mimic": []}
    # Archival run: every stochastic element is seeded deterministically (honest
    # arm via run(seed=), deviation arms via per-trial seed_simulator/seed_transpiler),
    # so the JSON and the figure are byte-reproducible. Security-faithful `secrets`
    # randomness remains the library default when no seed is passed.
    for t in range(n_trials):
        p = TrapdoorGroverProtocol(n, keys)  # fresh pad per invocation (run(num_invocations=))
        data["honest"].append(p.run(num_invocations=num_invocations,
                                    seed=int(rng.integers(1 << 30)))["integrity_ratio"])
        for kind in ("identity", "haar", "biased", "mimic"):
            # explicit rng-drawn pads: the biased arm embeds the instance pad in
            # its circuit, so constructor `secrets` pads would break determinism
            p2 = TrapdoorGroverProtocol(n, keys,
                                        qotp_x=int(rng.integers(0, 1 << n)),
                                        qotp_z=int(rng.integers(0, 1 << n)))
            data[kind].append(_run_deviation(p2, kind, num_invocations, sim, int(rng.integers(1 << 30))))
    stats = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                 "max": float(np.max(v)), "min": float(np.min(v))} for k, v in data.items()}
    thr = 0.80
    honest = np.array(data["honest"])
    dev_all = np.concatenate([data["identity"], data["haar"], data["biased"]])
    tpr = float(np.mean(dev_all < thr))   # crude deviations correctly flagged
    fpr = float(np.mean(honest < thr))    # honest wrongly flagged
    mimic_detected = float(np.mean(np.array(data["mimic"]) < thr))  # expected 0.0
    res = {
        "n": n, "M": M, "num_invocations": num_invocations, "trials_per_class": n_trials,
        "threshold": thr, "stats": stats,
        "true_positive_rate": tpr, "false_positive_rate": fpr,
        "mimic_detection_rate": mimic_detected,
        "seed": seed, "fully_seeded": True,
        "raw_trials": {k: [float(x) for x in v] for k, v in data.items()},
    }
    RESULTS["attack6"] = res
    # figure
    labels = ["Honest", "Identity\nreplacement", "Random\nunitary", "Biased\ndiffuser",
              "Exact\nmimic"]
    keysord = ["honest", "identity", "haar", "biased", "mimic"]
    means = [stats[k]["mean"] for k in keysord]
    errs = [stats[k]["std"] for k in keysord]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    colors = ["C2", "C3", "C3", "C3", "C1"]
    x = np.arange(5)
    ax.bar(x, means, yerr=errs, color=colors, capsize=4)
    ax.axhline(thr, color="k", lw=0.9, ls="--", label=f"Threshold {thr}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Output-consistency ratio")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_attack6_cheating.png", dpi=300)
    plt.close(fig)
    print(f"[Attack 6] honest={stats['honest']['mean']:.3f}+-{stats['honest']['std']:.3f}; "
          f"identity={stats['identity']['mean']:.3f}, haar={stats['haar']['mean']:.3f}, "
          f"biased={stats['biased']['mean']:.3f}, mimic={stats['mimic']['mean']:.3f}"
          f"+-{stats['mimic']['std']:.3f}; TPR(crude)={tpr:.2f} FPR={fpr:.2f} "
          f"mimic detected={mimic_detected:.2f}; fig written")
    return res


if __name__ == "__main__":
    print("=== Attack 1 & 6 real simulation ===")
    attack1()
    attack6()
    (Path(__file__).resolve().parent / "attacks_1_6_results.json").write_text(json.dumps(RESULTS, indent=2))
    print("Saved attacks_1_6_results.json")
