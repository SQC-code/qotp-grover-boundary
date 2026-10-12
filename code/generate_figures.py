"""Real-data figure generation (scaling depth, circuit overview, histograms).

Every function here uses real data only:
    fig_scaling_depth    — reads scaling_results.json (client/server split run)
    fig_circuit_overview — draws the real QOTPGroverProtocol circuit (n=4, M=2)
    fig_hist_overview    — live run of the real protocol (4096 fresh-pad invocations)

Attack figures are generated elsewhere from real simulation:
    run_attacks_1_6.py   (Fig: attack 1 pad reuse, attack 6 cheating incl. exact mimic)
    regen_attack_figs.py (Figs: attacks 2-5, reading attacks_full_results.json)
    regen_scaling_integrity_fig.py (scaling-integrity panel, reading scaling_results.json)

Historical note: earlier revisions of this file contained synthetic placeholder
attack curves; they have been removed entirely.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trapdoor_grover import QOTPGroverProtocol

def _paper_fig_dir():
    for anc in Path(__file__).resolve().parents:
        if (anc / "main_sr.tex").exists():
            return anc
    return Path.cwd()
OUT_DIR = Path(os.environ.get("PAPER_FIG_DIR") or _paper_fig_dir())
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Scaling: depth + CX vs n (from scaling_results.json) ----------
def fig_scaling_depth():
    rows = json.loads((Path(__file__).resolve().parent / "scaling_results.json").read_text())
    ns = np.array([r["n"] for r in rows])
    depth = np.array([r["depth"] for r in rows])
    cx = np.array([r["cx_count"] for r in rows])
    fig, ax1 = plt.subplots(figsize=(5.0, 3.3))
    color = "C0"
    ax1.semilogy(ns, depth, "-o", color=color, lw=2, ms=5, label="Depth")
    ax1.set_xlabel("Qubit width $n$")
    ax1.set_ylabel("Transpiled depth", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax2 = ax1.twinx()
    color2 = "C3"
    ax2.semilogy(ns, cx, "-s", color=color2, lw=2, ms=5, label="CX count")
    ax2.set_ylabel("CX gate count", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_scaling_depth.png", dpi=300)
    plt.close(fig)


# ---------- Circuit overview (n=4, M=2) ----------
def fig_circuit_overview():
    proto = QOTPGroverProtocol(n_qubits=4, keys=[0b0010, 0b1111])
    qc, _ = proto.build_circuit()
    try:
        fig = qc.draw(output="mpl", fold=120, style={"name": "iqp"})
    except Exception:
        fig = qc.draw(output="mpl", fold=120)
    fig.set_size_inches(8.5, 4.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_circuit_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------- Histogram overview (server vs client) ----------
def fig_hist_overview():
    # single source of truth: the pad-in-circuit end-to-end run (seeded), NOT a
    # live random run — see run_end_to_end.py.
    res = json.loads((Path(__file__).resolve().parent /
                      "end_to_end_results.json").read_text())
    keys = res["keys"]
    server = {format(int(k), "04b"): v
              for k, v in res["pad_in_circuit"]["server_counts"].items()}
    client = {format(int(k), "04b"): v
              for k, v in res["pad_in_circuit"]["client_counts"].items()}
    keys_lbl = [format(k, "04b") for k in keys]

    def sorted_items(d):
        return sorted(d.items(), key=lambda kv: int(kv[0], 2))

    s_items = sorted_items(server)
    c_items = sorted_items(client)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.5))
    s_x = [k for k, _ in s_items]
    s_y = [v for _, v in s_items]
    ax1.bar(s_x, s_y, color="C0", alpha=0.8)
    ax1.set_title("Server view")
    ax1.set_xlabel("Bitstring")
    ax1.set_ylabel("Count (invocations)")
    ax1.tick_params(axis="x", rotation=70, labelsize=7)

    c_x = [k for k, _ in c_items]
    c_y = [v for _, v in c_items]
    colors = ["C2" if k in keys_lbl else "C7" for k in c_x]
    ax2.bar(c_x, c_y, color=colors, alpha=0.85)
    ax2.set_title("Client view (decrypted)")
    ax2.set_xlabel("Bitstring")
    ax2.tick_params(axis="x", rotation=70, labelsize=7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_hist_overview.png", dpi=300)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"=== Generating figures to {OUT_DIR} (all real data) ===")
    for name, fn in [
        ("fig_scaling_depth", fig_scaling_depth),
        ("fig_circuit_overview", fig_circuit_overview),
        ("fig_hist_overview", fig_hist_overview),
    ]:
        print(f"  {name}...", end=" ", flush=True)
        try:
            fn()
            print("OK")
        except Exception as exc:
            print(f"FAILED: {exc}")
    print("Done. (Attack figures: run_attacks_1_6.py + regen_attack_figs.py; "
          "scaling-integrity: regen_scaling_integrity_fig.py.)")


if __name__ == "__main__":
    main()
