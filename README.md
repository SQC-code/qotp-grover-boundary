# QOTP-Grover boundary — simulation code and data (release v2.0)

Code, archived results, and revision analyses for:

**Unconditional input hiding and the information boundary of
single-register delegated Grover search** — Juyoung Kim (ETRI).
*Scientific Reports*, submission 31c9d045-ccfc-49c3-a00a-8de9083c7b11
(major revision, 2026-09).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23307281.svg)](https://doi.org/10.5281/zenodo.23307281)

Release: **v2.0** (2026-09-28). Archived at Zenodo: [10.5281/zenodo.23307281](https://doi.org/10.5281/zenodo.23307281).

```
code/                 simulation library + experiment/figure scripts
results/              archived JSON outputs used by the paper (read-only)
revision_analysis/    revision analysis plan + scripts + their outputs
requirements.txt      exact package versions (Python 3.14.4)
LICENSE               Apache License 2.0 (NOTICE file included)
```

## Environment

Windows 11 Enterprise 10.0.26100; AMD Ryzen 9 9950X3D (16C/32T), 125.6 GB
RAM; Python 3.14.4, Qiskit 2.4.1, Qiskit Aer 0.17.2 (statevector method,
double precision), NumPy 2.4.4, SciPy 1.17.1, Matplotlib 3.10.9.
Transpilation: basis {U1, U2, U3, CX}, optimization_level=2, fixed seeds.
All simulations are noiseless classical simulation; no quantum hardware.

## Script → paper-item map

| Paper item | Script | Output | Indicative runtime |
|---|---|---|---|
| Fig 1a / Attack 1 (pad reuse) | `code/run_attacks_1_6.py` | `attacks_1_6_results.json`, `fig_attack1_reuse.png` | ~4 min (with Attack 6) |
| Attacks 2–5 (SI Fig S2, Note 1) | `code/run_attacks_full.py` | `attacks_full_results.json` | ~2 min |
| Fig 1 main / Attack 6 (incl. exact mimic) | `code/run_attacks_1_6.py` | `attacks_1_6_results.json`, `fig_attack6_cheating.png` | (above) |
| SI Tables S2, Fig S3 (scaling) | `code/run_scaling.py`, `code/regen_scaling_integrity_fig.py` | `scaling_results.json`, `fig_scaling_*.png` | ~15 min (n=12 dominates) |
| SI Table S1 (M sweep) | `code/run_m_variation.py` | `m_variation_results.json` | ~1 min |
| SI Fig S1 + sampler check | `code/run_end_to_end.py` | `end_to_end_results.json`, `fig_hist_overview.png` | ~10 min |
| Twirl exactness | `code/verify_twirl_tracenorm.py` | stdout | seconds |

## Revision analyses (2026-09; revision analysis plan fixed before execution)

`revision_analysis/ANALYSIS_PLAN.md` was written and frozen **before**
these scripts were executed; thresholds and sample sizes were not changed
afterwards. Analyses that re-use the archived (pre-existing) result files
are retrospective, with thresholds fixed in the plan. A post-execution
addendum inside the plan records two corrections: the E1 enumeration is
now over the complete joint pad group (e1b), and the KS calibration is
full-pipeline (e5).

| Question (reviewer) | Script | Output | Runtime |
|---|---|---|---|
| Channel identity P(m\|rho)=q_r+(p_r−q_r)rho_mm on the real circuit, all r, entangled inputs (R1) | `e0_channel_identity.py` | `results/e0_channel_identity.json` — all PASS at 1e-10 | ~2 min |
| Sampler–circuit equivalence with stated tolerance (R1-4a) | `e1_equivalence.py`, `e1b_full_joint_enumeration.py` | `results/e1_equivalence.json`, `results/e1b_full_joint.json` — full joint pad group, 1,024 configs, max TV 5.2e-15; sampled-data U=0.047<δ=0.10 | ~3 + ~4 min |
| Attack-2 size calibration; Attack-4 KS check (R1-4b) | `e2_calibration.py`, `e5_attack2_perT_and_ks.py` | `results/e2_calibration.json`, `results/e5_attack2_perT_and_ks.json` — actual size 0.048; archived 0.036 compatible (p=0.076); per-T counts replayed (aggregate 36/1000 exact); full-pipeline KS p=0.868±0.011 | ~1 + ~5 min |
| Shadow error vs register size + positive control (R1-4c) | `e3_shadow_scaling.py` | `results/e3_shadow_scaling.json` — β≈−0.50 ∀n, C(n)≈0.42·10^(n/2) | ~4 min |
| Exact counts/CIs behind rounded table entries | `e4_precision_recheck.py` | `results/e4_precision.json` | seconds |

## Reproducibility

Every pipeline is fully seeded (transpiler, simulator, pads, analysis
RNG; seeds recorded in the JSONs). The archival configurations reproduce
their released JSON and PNG outputs exactly across consecutive runs
(hash-verified on this environment; wall-clock timing fields excluded). Attack 6 stores per-trial raw values under
`raw_trials`. Security-faithful `secrets` randomness remains the library
default whenever no seed is passed.

Naming note: the result field `integrity_ratio` (kept for backward
compatibility) is reported in the paper as the *output-consistency
ratio*; it measures consistency of decrypted outputs with the target
relation and implies no execution verification.

## Reproduce

```bash
pip install -r requirements.txt
cd code
python run_attacks_full.py
python run_attacks_1_6.py
python run_scaling.py
python run_m_variation.py
python run_end_to_end.py
python verify_twirl_tracenorm.py
cd ../revision_analysis
python e0_channel_identity.py
python e1_equivalence.py
python e1b_full_joint_enumeration.py
python e2_calibration.py
python e3_shadow_scaling.py
python e4_precision_recheck.py
python e5_attack2_perT_and_ks.py
```

To regenerate the figures directly from the archived outputs without
re-running any simulation, copy `results/*.json` next to the scripts in
`code/` first — the `regen_*` and `generate_figures` scripts read the
result files from the directory they run in.

Integrity: `MANIFEST.sha256` lists the SHA-256 checksum of every file in
this package (verify with `sha256sum -c MANIFEST.sha256` from the
package root).
