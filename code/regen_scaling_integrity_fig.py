"""Regenerate fig_scaling_integrity.png: measured (2048-shot) vs analytical ideal.

Measured integrity from scaling/scaling_results.json (real Aer runs).
Ideal from p_r + (M-1)q_r (Proposition 'Decryption correctness', M=2).
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path

import os
def _paper_fig_dir():
    for anc in Path(__file__).resolve().parents:
        if (anc / "main_sr.tex").exists():
            return anc
    return Path.cwd()
OUT = Path(os.environ.get("PAPER_FIG_DIR") or _paper_fig_dir())
OUT.mkdir(parents=True, exist_ok=True)
# single source of truth: read from scaling_results.json
import json
rows = json.loads((Path(__file__).resolve().parent / "scaling_results.json").read_text())
ns = np.array([r["n"] for r in rows])
meas = np.array([r["integrity_ratio"] for r in rows])
rs = {r["n"]: r["n_iter"] for r in rows}


def ideal(n, r, M=2):
    N = 2 ** n; th = np.arcsin(1 / np.sqrt(N)); pr = np.sin((2 * r + 1) * th) ** 2
    return pr + (M - 1) * (1 - pr) / (N - 1)


idl = np.array([ideal(n, rs[n]) for n in ns])
fig, ax = plt.subplots(figsize=(5.0, 3.3))
ax.plot(ns, idl, "-", color="C0", lw=1.5, label="Ideal $p_r+(M-1)q_r$")
ax.plot(ns, meas, "o", color="C2", ms=7, label="Measured (2048 shots per pad)")
ax.set_xlabel("Qubit width $n$"); ax.set_ylabel("Output-consistency ratio")
ax.set_ylim(0.955, 1.005); ax.set_xticks(ns)
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig_scaling_integrity.png", dpi=300); plt.close(fig)
print("fig_scaling_integrity.png regenerated (measured vs ideal)")
