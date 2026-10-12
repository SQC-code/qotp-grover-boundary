"""Pauli-twirl exactness + finite-sample trace-norm verification (paper Sec. Scaling).

- Exact regime (n<=6): sum X^a Z^b |kappa><kappa| Z^b X^a over ALL 4^n pads; the
  twirl identity gives exactly I/2^n, so ||rho_enc - I/2^n||_1 is numerical zero.
- Monte-Carlo regime (n>6): a PARTIAL average over K random pads is NOT I/2^n; its
  trace-norm deviation follows the O(sqrt(d/K)) sampling floor and vanishes only as
  the full average is completed. Reported to document that the earlier "< 1e-3"
  Monte-Carlo claim was unsupported (real value ~0.12 at n=8, K=1e4).

Also prints the analytical decryption-integrity p_r + (M-1)q_r used in Table 3 (Ideal).
"""
import numpy as np


def kappa(n, keys):
    N = 1 << n
    v = np.zeros(N, complex)
    for k in keys:
        v[k] = 1.0 / np.sqrt(len(keys))
    return v


def apply_XaZb(psi, a, b, n):
    N = 1 << n
    idx = np.arange(N)
    phase = np.array([(-1) ** bin(b & x).count("1") for x in idx])
    out = np.zeros(N, complex)
    out[idx ^ a] = psi * phase          # X^a permutes by XOR a; Z^b adds (-1)^{b.x}
    return out


def exact_twirl_tracenorm(n, keys):
    """Exact sum over all 4^n pads (feasible for small n)."""
    N = 1 << n
    psi = kappa(n, keys)
    rho = np.zeros((N, N), complex)
    for a in range(N):
        for b in range(N):
            v = apply_XaZb(psi, a, b, n)
            rho += np.outer(v, v.conj())
    rho /= (N * N)
    diff = rho - np.eye(N) / N
    ev = np.linalg.eigvalsh((diff + diff.conj().T) / 2)
    return float(np.sum(np.abs(ev)))


def mc_twirl_tracenorm(n, keys, K, seed):
    rng = np.random.default_rng(seed)
    N = 1 << n
    psi = kappa(n, keys)
    rho = np.zeros((N, N), complex)
    for _ in range(K):
        a = int(rng.integers(0, N)); b = int(rng.integers(0, N))
        v = apply_XaZb(psi, a, b, n)
        rho += np.outer(v, v.conj())
    rho /= K
    diff = rho - np.eye(N) / N
    tn = float(np.sum(np.abs(np.linalg.eigvalsh((diff + diff.conj().T) / 2))))
    fn = float(np.linalg.norm(diff))
    return tn, fn


def ideal_integrity(n, r, M=2):
    N = 1 << n
    th = np.arcsin(1 / np.sqrt(N))
    pr = np.sin((2 * r + 1) * th) ** 2
    qr = (1 - pr) / (N - 1)
    return pr + (M - 1) * qr


if __name__ == "__main__":
    print("=== Exact twirl (all 4^n pads) -> should be numerical zero ===")
    for n in [2, 3, 4, 5, 6]:
        tn = exact_twirl_tracenorm(n, [1, (1 << n) - 1])
        print(f"  n={n}: ||rho_enc - I/2^n||_1 = {tn:.2e}")

    print("\n=== Monte-Carlo partial average (n=8, K=1e4) -> NOT < 1e-3 ===")
    tn, fn = mc_twirl_tracenorm(8, [1, 255], 10_000, seed=1)
    print(f"  trace norm = {tn:.4f}  (Frobenius = {fn:.4f})  <- earlier paper claim '<1e-3' was wrong")

    print("\n=== Table 3 ideal integrity p_r+(M-1)q_r (M=2) ===")
    for n, r in [(4, 3), (6, 6), (8, 12), (10, 25), (12, 50)]:
        print(f"  n={n:2d} r={r:2d}: {ideal_integrity(n, r):.6f}")
