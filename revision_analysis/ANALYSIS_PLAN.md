# Pre-registered analysis plan — major revision (SR submission 31c9d045-ccfc-49c3-a00a-8de9083c7b11)

Written: 2026-09-28, **before** any of the new analyses below were executed.
Author of record: Juyoung Kim (ETRI). Executed by Claude Code under author direction.

Inputs (frozen; none are modified by this plan):
- Submitted manuscript/SI: `SUBMISSION/manuscript/` (as submitted 2026-07-16).
- Code + archived results: `SUBMISSION/code_and_results/` (fully seeded; Attack-6
  archival run byte-reproducible; see its README).
- New analyses run from `revision_v2/analysis/`, writing only under
  `revision_v2/analysis/results/`. Original archives are read-only inputs.

Environment: Windows 11 Enterprise 10.0.26100, AMD Ryzen 9 9950X3D (16C/32T),
125.6 GB RAM; Python 3.14.4, Qiskit 2.4.1, Aer 0.17.2, NumPy 2.4.4, SciPy 1.17.1.

General rules: every decision threshold below is fixed now; no threshold,
sample size, or tolerance may be revised after seeing results. If a criterion
fails, the failure is reported as such ("equivalence not established" /
"hypothesis rejected"), not re-tested with looser settings. All new RNG seeds
are listed here. Every script prints the md5 of each archived input it reads.

---

## E0 — Channel-identity verification (gate for the manuscript's new Proposition)

Claim under test (derived during this revision; treated as NEW until verified):
for the comparator–diffuser construction with the search register initialised
to |+>^n, r iterations, search register measured and returned, key register
never returned, the induced channel from an arbitrary key-register input rho
to the classical outcome m is

  P(m|rho) = q_r + (p_r - q_r) * rho_mm ,   E_m = q_r I + (p_r - q_r)|m><m| ,

with p_r = sin^2((2r+1) arcsin(1/sqrt(N))), q_r = (1-p_r)/(N-1); and the joint
client-visible channel on (retained reference R, outcome m) for entangled
inputs coincides with: measure key in the computational basis, then sample m
from R(m|k) = p_r if m=k else q_r. Validity claimed for ALL r >= 0 (no
near-first-maximum restriction).

Method (script `e0_channel_identity.py`, seed 20260928):
1. Build the server unitary from the ACTUAL library circuit
   (`trapdoor_grover._apply_V`, `_apply_diffuser`) — not a re-derivation.
2. n in {2,3,4}; r in {0, 1, r*(n), 2 r*(n)+3} (r* = round(pi/(4 asin(2^-n/2)) - 1/2)).
3. Inputs per (n, r): all 2^n computational basis states; 12 Haar-random pure
   states; 6 random density matrices; the padded marked-set states
   X^a Z^b |kappa> for keys {2, 2^n-1}, 3 random pads.
4. Compare exact circuit law vs the analytic law; PASS iff max abs deviation
   < 1e-10 over every input and every m.
5. Entangled inputs: n in {2,3}; 5 Haar-random pure states on R (x) K
   (dim 2^n each). Compare the full sub-normalised cq ensemble
   {rho_R^(m)} honest vs mimic; PASS iff max trace-norm deviation < 1e-10 and
   outcome probabilities match to < 1e-10.
6. Marked-set sanity anchor: n=4, keys {2,15}, r=3, pad (0,0):
   P(Y in keys) must equal p_3 + q_3 to < 1e-10 (guards endianness errors).

Decision rule: the manuscript's channel-level Proposition and the removal of
the p_r >= 1/N restriction are adopted ONLY if all E0 checks pass. Otherwise
the old lambda-mixture proposition is retained and the discrepancy reported.

## E1 — Shortcut-vs-pad-in-circuit equivalence (Reviewer 1, comment 4a)

Primary endpoint: the DECRYPTED output law Y (16 bins at n=4). Server-view
uniformity and marked-success fraction are secondary.

Arm A (deterministic, primary). n=4, r=3. For M in {1,2,3} with marked sets
{9}, {3,12}, {1,6,11} (the M=3 set exercises the generic StatePreparation
path): enumerate ALL 256 X-pads a, each with Z-pads b in {0, 7, 11}.
For every (M, a, b): compute the exact statevector law of the padded circuit,
decrypt (relabel y = m XOR a), and compare with (i) the exact law of the
unpadded circuit (= the shortcut sampling law) and (ii) the analytic law
q_r + (p_r-q_r)/M on the marked set. Metric: total variation.
PASS iff max TV <= delta_num = 1e-10 over all 2304 configurations.
This is the equivalence statement with tolerance; it is strictly stronger
than any statistical test on samples.

Arm B (statistical, secondary; archived data only). Input: archived
`end_to_end_results.json` (seeded run of 2026-07-16; md5 logged). One-sample
equivalence of each arm's decrypted 4096-count histogram against the exact
law from Arm A (keys {2,15}, M=2):
  U = TV(empirical, exact) + r_1,  r_1 = sqrt( ln(2*2^d / alpha) / (2 s) ),
  d = 16 bins, s = 4096, alpha = 0.05  =>  r_1 = 0.04249.
Pre-registered tolerance delta_stat = 0.10. Justification, fixed now: (a) the
confidence radius alone is 0.042 (two-sample: 0.085), so no delta below ~0.05
is certifiable at the archived sample size even under exact equality — the
deterministic Arm A is the primary evidence precisely for this reason; (b) a
TV deviation of 0.10 is the largest that would still leave every downstream
use of the sampler (marked-fraction estimates, uniformity checks at
resolution ~0.04) within its reported uncertainties. Also report the
two-sample U (alpha split 0.025/0.025) for the pair of arms, for completeness.
Conclusions: report U values; "equivalence established at delta" iff U < delta.
No new sampling runs; no post-hoc delta changes. The prior chi-2/permutation
non-rejections are downgraded to consistency checks in the manuscript.

## E2 — Null-rejection calibration (Reviewer 1, comment 4b)

Question: the archived Attack-2 aggregate rejection rate is 36/1000 = 0.036 at
nominal alpha = 0.05. Exact binomial (Clopper–Pearson) 95% CI: computed in
script; if the CI excludes 0.05 the deficit is real, else sampling noise.
Pre-registered hypothesis (stated before running): the deficit, if real, is
caused by the chi-squared approximation to the discrete multinomial statistic
at 64 bins with expected counts >= 128 (approximation error + discreteness),
NOT by dependence between the 1000 tests.

Method (script `e2_calibration.py`, seed 20260929):
1. Reproduce the archived test exactly in distribution: for each
   T in {2,4,...,40}, B = 2000 independent null replicas: multinomial(T*4096,
   uniform over 64) -> scipy chisquare p-value. Empirical size at alpha=.05
   per T and aggregated (matching the archived 20x50 aggregation), with 95%
   Clopper–Pearson CIs.
2. Compare the archived 0.036 against the calibrated size distribution;
   report whether the hypothesis is supported (calibrated size below nominal
   by a comparable amount) or refuted.
3. Attack-4 KS check: regenerate the 200 per-trial p-values (seeded, exact
   reproduction via `run_attacks_full.attack4` stream), count ties, and
   compute a calibrated KS p-value by simulating the null KS distribution for
   200 iid U(0,1) samples (10^4 replicas). Report alongside the archived
   KS p = 0.867.
No changes to the paper's alpha; the manuscript gains one calibration sentence
+ SI detail.

## E3 — Classical-shadow error: register-size dependence (Reviewer 1, comment 4c)

Question: the archived decay 43 * K^-0.50 is reported at n=4 only. Provide
C(n) and exponent(n).

Method (script `e3_shadow_scaling.py`, seed 20260930): n in {2,3,4,5}
(n=6 excluded: the full-snapshot enumeration used by the archived experiment
requires 6^6 matrices of dim 64 ~ 3 GB; four widths suffice to exhibit the
scaling law; exclusion fixed now, not after seeing results). K in
{100, 1000, 10^4, 10^5}, 25 independent repetitions per (n, K) — identical
estimator and metric to the archived run (full local-Pauli global-snapshot
shadow of I/2^n; raw linear-inversion estimator; metric = (1/2)||rho_hat -
I/2^n||_1, explicitly labelled half-trace-norm error). Report per-n log-log
OLS fit C(n), exponent with standard errors; secondary: PSD-projected
estimator at n=4 for reference; positive control: n=3 with TRUE state
|0...0> (pure) — the same estimator at the same K must distinguish it from
I/2^n (error to truth decays; distance to I/2^n plateaus near 0.875),
demonstrating the attack was given a fair chance.
Expectation stated in advance: exponent ~ -1/2 at every n; C(n) growing
approximately like (sqrt(10))^n (2-norm 5^{n/2} times sqrt(2^n) relaxation);
the fitted C(n) values are reported regardless.

## E4 — M-sweep and scaling-table precision recheck (Reviewer-independent hygiene)

1. Read archived `m_variation_results.json` and `scaling_results.json` (md5
   logged). Recover exact marked counts (ratio x shots must be integral; else
   flag). Report ratios to 6 decimals with 95% Clopper–Pearson CIs.
2. Compute the exact theoretical success p_r + (M-1) q_r at n=8, r=12 to 12
   digits (expected ~0.999947...); table entries currently printed as
   "1.000" are re-labelled with exact values/CIs in the SI.
3. No re-runs unless a stored ratio is non-integral times shots (would
   indicate a bookkeeping error; then the seeded run is reproduced in a
   scratch copy, never overwriting archives).

## Reference verification (completed before this plan was frozen)

Verified at publisher/indexers on 2026-09-28: Fernández & Martin-Delgado,
Phys. Rev. Research 6, 043109 (2024); Drmota et al., Phys. Rev. Lett. 132,
150604 (2024); Hillston, Marin, Piazza, Rossi, Fundam. Informaticae 181(1),
1–35 (2021); Marin, Piazza, Rossi, Acta Informatica 59, 211–244 (2022).
NOT verifiable as cited (2026-09-28): SCPMA 69:254211 / 69:230314 / 69:230311;
EPJ Plus 139:894; EPJ Plus 141:39 (a nearby record is 141:29,
DOI 10.1140/epjp/s13360-026-07289-1, a different article number than cited);
EPJ QT 12:55 exists (Huang, Huang & Zhou, 2025) but its title/relevance could
not be confirmed. Handling fixed now: editor marked these optional; only
verified AND directly relevant works are added; the response letter states
this and offers to add any entry given complete bibliographic details.

---

## Addendum (2026-09-28, recorded after execution; original text above unchanged)

1. TERMINOLOGY. The plan above is a *revision analysis plan*, frozen before
   the new computations (E0-E4) were executed. It is not a prospective
   pre-registration in the clinical-trials sense: the archived data analysed
   in E1 Arm B / E2 / E4 existed before this plan was written, and summary
   statistics of those archives (e.g., the two-sample TV = 0.038) were known
   to the analyst beforehand. Analyses of archived data are therefore
   retrospective, with thresholds fixed in this plan; E0/E1-Arm-A/E3 generate
   new data. Manuscript wording follows this distinction.
2. E1 ARM A - ARITHMETIC CORRECTION AND CORRECTIVE RUN. The plan text says
   "ALL 256 X-pads": at n=4 there are only 2^n = 16 X-pads (2^{2n} = 256
   counts JOINT (a,b) pads). The executed script enumerated all 16 X-pads x
   3 Z-pads x 4 sets = 192 rows (X-pads exhaustive; Z-pads a partial sample).
   Corrective run `e1b_full_joint_enumeration.py` (2026-09-28) enumerates the
   complete joint pad group, 256 (a,b) x 4 sets = 1,024 configurations:
   overall max TV = 5.230e-15, PASS at the plan's 1e-10 tolerance
   (results/e1b_full_joint.json). The original 192-row archive is retained.
3. E2 KS - METHOD CORRECTION. The plan's Attack-4 KS calibration was executed
   in e2 by sampling 200 iid U(0,1) values per replica, which skips the
   multinomial -> chi-squared stage of the real pipeline. Corrective run
   `e5_attack2_perT_and_ks.py` (2026-09-28, seed 20261001) simulates the full
   pipeline (outcomes -> XOR pairs -> chi-squared p-values -> KS statistic;
   B = 1000; the i.i.d.-preserving shuffle omitted, distribution unchanged):
   calibrated p = 0.868 +- 0.011 (MC SE), matching the asymptotic 0.867.
   e5 also replays the archived Attack-2 stream (seed 20240601) to recover
   per-T rejection counts; the replay reproduces the archived aggregate
   36/1000 exactly.
