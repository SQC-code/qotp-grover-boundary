"""TrapdoorGroverProtocol — Delegated Grover search with QOTP-protected marked set.

Implements information-theoretic input hiding for delegated quantum search:
- Client encodes marked set into quantum key register |κ⟩ = uniform superposition.
- Quantum One-Time Pad (X^a Z^b) protects |κ⟩ via per-query fresh classical keys.
- Server runs target-agnostic CNOT comparator + Grover diffuser on plain circuit.
- Client decrypts measurement (m XOR a) to recover marked elements.

Security: Ambainis et al. QOTP perfect secrecy on input register (quant-ph/0003101).
Server's view of |κ_enc⟩ is maximally mixed state I/2^n regardless of marked set.

Public API:
    TrapdoorGroverProtocol(n_qubits, keys, qotp_x=None, qotp_z=None)

Inputs:
    - n_qubits   : search/key register width
    - keys       : list[int] of marked elements. |κ⟩ = uniform superposition over keys.
    - qotp_x     : X-mask key (int, n_qubits bits). None → fresh random (REQUIRED for security).
    - qotp_z     : Z-mask key (int, n_qubits bits). None → fresh random.

Methods:
    .build_circuit(n_iter=None)        → (QuantumCircuit, n_iter)
    .run(shots=4096)                   → dict (fixed-pad: one pad for all shots)
    .run(num_invocations=4096)         → dict (fresh pad drawn per invocation; the
                                          paper's security unit — server view uniform)
    .decrypt(server_view)              → dict (m XOR qotp_x decoded counts)
    .classify(value)                   → str ('MARKED key_i' or 'OUTSIDE_MARKED_SET')

Properties:
    .n_iter                            → optimal Grover iterations (per-branch model)
    .derive_trapdoor()                 → key[0] ⊕ key[1] for M=2 (auxiliary structural info)

Naming convention: 'TrapdoorGroverProtocol' is the legacy class name; the protocol itself
relies on QOTP for security, not a cryptographic trapdoor function. Class alias provided.
"""
import secrets
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import StatePreparation
try:
    from qiskit_aer import AerSimulator
except ImportError:
    from qiskit.providers.aer import AerSimulator


class TrapdoorGroverProtocol:
    """Grover with QOTP-masked quantum trapdoor key (configurable claw set)."""

    def __init__(
        self,
        n_qubits: int,
        keys: list,
        qotp_x: int = None,
        qotp_z: int = None,
    ):
        if n_qubits < 2:
            # The n=1 comparator/diffuser special case is not a correct
            # equality-oracle + single-qubit diffuser; all reported experiments
            # use n >= 3. Restrict to n >= 2 to avoid the degenerate branch.
            raise ValueError("n_qubits must be >= 2")
        self.n = n_qubits
        self.N = 2 ** n_qubits

        unique_keys = sorted(set(int(k) for k in keys))
        if not unique_keys:
            raise ValueError("keys must contain at least one preimage")
        if len(unique_keys) > self.N:
            raise ValueError(f"too many keys ({len(unique_keys)}) for n_qubits={n_qubits}")
        for k in unique_keys:
            if not (0 <= k < self.N):
                raise ValueError(f"key {k} out of range [0, {self.N})")
        self.keys = unique_keys
        self.M = len(self.keys)

        self.qotp_x = (
            int(qotp_x) & (self.N - 1) if qotp_x is not None
            else secrets.randbits(n_qubits)
        )
        self.qotp_z = (
            int(qotp_z) & (self.N - 1) if qotp_z is not None
            else secrets.randbits(n_qubits)
        )

        # Per-branch optimal Grover iterations:
        # each entangled key branch has 1 marked state in search of size N
        theta = np.arcsin(1.0 / np.sqrt(self.N)) if self.N > 1 else 0.0
        if theta > 0:
            self.n_iter = max(1, int(round(np.pi / (4 * theta) - 0.5)))
        else:
            self.n_iter = 1

    # ---------- Trapdoor utilities ----------
    def derive_trapdoor(self):
        """For M=2: s = key[0] XOR key[1]. Otherwise: dict of pairwise XORs."""
        if self.M == 1:
            return None
        if self.M == 2:
            return self.keys[0] ^ self.keys[1]
        return {
            (i, j): self.keys[i] ^ self.keys[j]
            for i in range(self.M) for j in range(i + 1, self.M)
        }

    # ---------- Circuit building blocks ----------
    def _prepare_kappa(self, qc, key_reg):
        """Prepare |κ⟩ = (1/√M) Σ |key_i⟩ on the key register.

        Structured sets get exact O(n) circuits: a single key uses X gates; an
        arbitrary pair {k1,k2} uses one H, a CX fan-out over the differing bits,
        and X on the base pattern; a power-of-two stride family {i·(N/M)} uses
        Hadamards on the top log2(M) qubits. All are exact (fidelity 1, real
        positive amplitudes). Any other set falls back to generic
        StatePreparation, which may cost up to O(2^n) gates.
        """
        n, N, M = self.n, self.N, self.M
        keys = sorted(self.keys)
        if M == 1:                       # single key: X on set bits, O(n)
            for i in range(n):
                if (keys[0] >> i) & 1:
                    qc.x(key_reg[i])
            return
        if M == 2:                       # arbitrary pair: H + CX + X, O(n)
            k1, k2 = keys
            d = k1 ^ k2
            p = (d & -d).bit_length() - 1        # lowest differing bit
            qc.h(key_reg[p])
            for i in range(n):
                if ((d >> i) & 1) and i != p:
                    qc.cx(key_reg[p], key_reg[i])
            for i in range(n):
                if (k1 >> i) & 1:
                    qc.x(key_reg[i])
            return
        if M & (M - 1) == 0:
            stride, c = N // M, keys[0]  # offset-stride family {c + i*(N/M)}
            if c < stride and keys == [c + i * stride for i in range(M)]:
                for i in range(n):       # X on the fixed low-bit offset c
                    if (c >> i) & 1:
                        qc.x(key_reg[i])
                m = M.bit_length() - 1   # H on the top m qubits
                for i in range(n - m, n):
                    qc.h(key_reg[i])
                return
        state_vec = np.zeros(N, dtype=complex)   # generic fallback
        amp = 1.0 / np.sqrt(M)
        for k in self.keys:
            state_vec[k] = amp
        sp = StatePreparation(state_vec, label="kappa")
        qc.compose(sp, qubits=list(key_reg), inplace=True)

    def _apply_qotp(self, qc, key_reg):
        """Apply X^{qotp_x} Z^{qotp_z} to key register (Z first, then X)."""
        for i in range(self.n):
            if (self.qotp_z >> i) & 1:
                qc.z(key_reg[i])
            if (self.qotp_x >> i) & 1:
                qc.x(key_reg[i])

    def _apply_V(self, qc, search_reg, key_reg):
        """Target-agnostic CNOT comparator: phase flip if search == key."""
        for i in range(self.n):
            qc.cx(key_reg[i], search_reg[i])
        qc.x(search_reg)
        qc.h(search_reg[-1])
        if self.n > 1:
            qc.mcx([search_reg[i] for i in range(self.n - 1)], search_reg[-1])
        else:
            qc.z(search_reg[0])
        qc.h(search_reg[-1])
        qc.x(search_reg)
        for i in range(self.n):
            qc.cx(key_reg[i], search_reg[i])

    def _apply_diffuser(self, qc, search_reg):
        """Standard Grover diffuser on search register."""
        qc.h(search_reg)
        qc.x(search_reg)
        qc.h(search_reg[-1])
        if self.n > 1:
            qc.mcx([search_reg[i] for i in range(self.n - 1)], search_reg[-1])
        else:
            qc.z(search_reg[0])
        qc.h(search_reg[-1])
        qc.x(search_reg)
        qc.h(search_reg)

    # ---------- Full circuit ----------
    def build_circuit(self, n_iter: int = None):
        """Build full circuit with barriers separating client/server.

        Returns (QuantumCircuit, n_iter).
        """
        if n_iter is None:
            n_iter = self.n_iter
        search = QuantumRegister(self.n, "search")
        key = QuantumRegister(self.n, "key")
        c_meas = ClassicalRegister(self.n, "meas")
        qc = QuantumCircuit(search, key, c_meas)

        # ----- CLIENT -----
        self._prepare_kappa(qc, key)
        qc.barrier(label="CLIENT_kappa_ready")
        self._apply_qotp(qc, key)
        qc.barrier(label="CLIENT_QOTP_applied")
        qc.barrier(label="SEND_to_SERVER")

        # ----- SERVER -----
        qc.h(search)
        qc.barrier(label="SERVER_search_init")
        for it in range(n_iter):
            self._apply_V(qc, search, key)
            qc.barrier(label=f"SERVER_V_{it}")
            self._apply_diffuser(qc, search)
            qc.barrier(label=f"SERVER_D_{it}")
        qc.measure(search, c_meas)
        qc.barrier(label="SERVER_measure_done")
        return qc, n_iter

    # ---------- Post-processing ----------
    def decrypt(self, server_view: dict) -> dict:
        """Client decryption: m XOR qotp_x."""
        out = {}
        for bs, c in server_view.items():
            m = int(bs, 2)
            v = m ^ self.qotp_x
            label = format(v, f"0{self.n}b")
            out[label] = out.get(label, 0) + c
        return out

    def classify(self, value: int) -> str:
        """Classify decrypted value as MARKED key_i or OUTSIDE_MARKED_SET."""
        if value in self.keys:
            idx = self.keys.index(value)
            return f"MARKED key_{idx}"
        return "OUTSIDE_MARKED_SET"

    # ---------- End-to-end ----------
    def run(self, shots: int = 4096, num_invocations: int = None,
            seed: int = None) -> dict:
        """Build → simulate → decrypt → classify. Returns dict with all metrics.

        Two invocation models:
        - ``shots``: one fixed pad (this instance's ``qotp_x``/``qotp_z``) is used
          for all ``shots`` measurements of a single circuit. The server-visible
          outcomes then concentrate on the masked support {k XOR a}; use this only
          for fixed-pad studies (e.g. the pad-reuse leakage of Attack 1).
        - ``num_invocations``: a FRESH pad is drawn per invocation (the security
          unit of the paper). Equivalent to running ``num_invocations`` independent
          single-shot executions, each under an independent uniform X-pad. Because
          X^a acts as a bitwise translation on the search outcome (decryption
          correctness), this is computed efficiently from one unpadded circuit whose
          per-shot outcomes are translated by an independent fresh pad, giving a
          server view that is uniform over {0,1}^n (input-hiding lemma).
        If ``num_invocations`` is given it takes precedence over ``shots``.
        ``seed`` (optional): makes the ``num_invocations`` path fully
        deterministic (transpiler seed, simulator seed, and the fresh X-pads all
        derive from it). Default None keeps the security-faithful behaviour
        (``secrets`` pads, unseeded simulator).
        """
        if num_invocations is not None:
            return self._run_fresh_pads(int(num_invocations), seed=seed)
        qc, n_iter = self.build_circuit()
        sim = AerSimulator()
        tqc = transpile(qc, sim, optimization_level=1)
        result = sim.run(tqc, shots=shots).result()
        server_view = result.get_counts()
        client_view = self.decrypt(server_view)

        # Classification
        marked_count = 0
        outside_count = 0
        per_key_count = {format(k, f"0{self.n}b"): 0 for k in self.keys}
        for bs, c in client_view.items():
            v = int(bs, 2)
            if v in self.keys:
                marked_count += c
                per_key_count[bs] = per_key_count.get(bs, 0) + c
            else:
                outside_count += c
        integrity = marked_count / shots if shots > 0 else 0.0

        return {
            "qc": qc,
            "tqc": tqc,
            "server_view": server_view,
            "client_view": client_view,
            "per_key_count": per_key_count,
            "marked_count": marked_count,
            "outside_count": outside_count,
            "integrity_ratio": integrity,
            "shots": shots,
            "n_iter": n_iter,
            "keys": self.keys,
            "M": self.M,
            "qotp_x": self.qotp_x,
            "qotp_z": self.qotp_z,
        }

    def _run_fresh_pads(self, num_invocations: int, seed: int = None) -> dict:
        """Fresh-pad-per-invocation run (see run(num_invocations=...))."""
        # One unpadded circuit; its per-shot search outcomes are the unmasked
        # protocol outputs. A fresh X-pad translates each into the server view.
        rng = np.random.default_rng(seed) if seed is not None else None
        base = TrapdoorGroverProtocol(self.n, self.keys, qotp_x=0, qotp_z=0)
        qc, n_iter = base.build_circuit()
        sim = AerSimulator(method="statevector") if seed is not None else AerSimulator()
        tqc = transpile(qc, sim, optimization_level=1,
                        seed_transpiler=seed if seed is not None else None)
        run_kwargs = {"shots": num_invocations, "memory": True}
        if seed is not None:
            run_kwargs["seed_simulator"] = seed
        result = sim.run(tqc, **run_kwargs).result()
        raw = result.get_memory()  # one unmasked outcome per invocation

        server_view, client_view = {}, {}
        per_key_count = {format(k, f"0{self.n}b"): 0 for k in self.keys}
        marked_count = 0
        for bs in raw:
            v = int(bs, 2)
            # fresh X-pad this invocation (seeded rng if provided, else secrets)
            a = int(rng.integers(0, self.N)) if rng is not None else secrets.randbits(self.n)
            m = v ^ a                              # server-visible (masked) outcome
            dec = m ^ a                            # client decrypts with its own a (= v)
            sv = format(m, f"0{self.n}b")
            cv = format(dec, f"0{self.n}b")
            server_view[sv] = server_view.get(sv, 0) + 1
            client_view[cv] = client_view.get(cv, 0) + 1
            if dec in self.keys:
                marked_count += 1
                per_key_count[cv] = per_key_count.get(cv, 0) + 1
        integrity = marked_count / num_invocations if num_invocations > 0 else 0.0
        return {
            "qc": qc,
            "tqc": tqc,
            "server_view": server_view,
            "client_view": client_view,
            "per_key_count": per_key_count,
            "marked_count": marked_count,
            "outside_count": num_invocations - marked_count,
            "integrity_ratio": integrity,
            "num_invocations": num_invocations,
            "n_iter": n_iter,
            "keys": self.keys,
            "M": self.M,
        }

    # ---------- Convenience labeling ----------
    def label_view(self, view: dict, with_role: bool = True) -> dict:
        """Re-label view dict with key role annotations for histogram display."""
        out = {}
        for bs, c in view.items():
            v = int(bs, 2)
            if v in self.keys:
                idx = self.keys.index(v)
                if with_role:
                    if idx == 0:
                        suffix = "(key_0: primary)"
                    else:
                        suffix = f"(key_{idx}: claw partner)"
                    label = f"{bs} {suffix}"
                else:
                    label = bs
            else:
                label = f"{bs} (outside)"
            out[label] = c
        return out


# ---------- Alias (paper-ready naming) ----------
QOTPGroverProtocol = TrapdoorGroverProtocol


# ---------- Self-test ----------
def self_test():
    print("=== TrapdoorGroverProtocol self-test ===")
    # 2-key  n=3
    keys = [0b010, 0b111]
    p = TrapdoorGroverProtocol(n_qubits=3, keys=keys)
    r = p.run(shots=2048)
    assert r["integrity_ratio"] > 0.85, f"integrity {r['integrity_ratio']} too low"
    print(f"  2-key  n=3: integrity={r['integrity_ratio']:.3f} OK")
    # 1-key n=3
    p1 = TrapdoorGroverProtocol(n_qubits=3, keys=[0b101])
    r1 = p1.run(shots=2048)
    assert r1["integrity_ratio"] > 0.85, f"integrity {r1['integrity_ratio']}"
    print(f"  1-key n=3:  integrity={r1['integrity_ratio']:.3f} OK")
    print("Self-test PASSED")


if __name__ == "__main__":
    self_test()
