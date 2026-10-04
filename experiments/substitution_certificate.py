"""Per-context certificate for a last-layer substitution.

After i-orca PIC_Binding.substitution_certified_max (T5(a)).

Theorem (kernel-checked, Isabelle, examples/pic_core/PIC_Binding.thy): for a finite token set V, decode logits
L(v) = <r, U_v> + b_v with argmax t, and a substituted decode input r_hat,

    if  L(t) - L(v) > 2 * max_v |<r - r_hat, U_v>|  for every v != t,  then  argmax_v <r_hat, U_v> + b_v = t.

This module issues the certificate per context, or refuses it:
  * numerically (float64): margin m = L(t) - max_{v != t} L(v),
    delta = max_v |<r - r_hat, U_v>| over ALL of V;
  * as a Soufflé query over fixed-point facts. m is rounded DOWN and delta UP to integers at SCALE,
    and a slack TAU
    (logit units) absorbs float64 rounding in the inner products. The Datalog verdict therefore never
    certifies a
    context the exact real-number condition would reject, given |float64 error| <= TAU.
The decision is the Datalog query result; the Python twin exists for parity tests.
"""
from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SCALE = 2 ** 30  # fixed-point scale; Soufflé `number` is 64-bit here (|logit| * 2^30 << 2^63)
TAU = 1e-6       # slack in logit units for float64 rounding of 768-dim inner products

SOUFFLE_CERTIFICATE = """
// T5(a): certified(c) iff margin(c) > 2 * delta(c) + tau, all in fixed point (margin floored, delta ceiled).
.decl margin(c:number, m:number) .input margin
.decl delta(c:number, d:number) .input delta
.decl tau(t:number) .input tau
.decl certified(c:number) .output certified
certified(c) :- margin(c, m), delta(c, d), tau(t), m > 2 * d + t.
.decl refused(c:number) .output refused
refused(c) :- margin(c, _), !certified(c).
"""


def margins_and_deltas(
    r: np.ndarray, r_hat: np.ndarray, U: np.ndarray, bias: np.ndarray | None = None, chunk: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """float64 per context: argmax t of the REAL logits, margin m, delta = max_v |<r - r_hat, U_v>|."""
    r, r_hat, U = (np.asarray(a, dtype=np.float64) for a in (r, r_hat, U))
    b = np.zeros(U.shape[0]) if bias is None else np.asarray(bias, dtype=np.float64)
    ts, ms, ds = [], [], []
    for s in range(0, len(r), chunk):
        L = r[s:s + chunk] @ U.T + b
        t = L.argmax(1)
        top = L[np.arange(len(t)), t]
        L[np.arange(len(t)), t] = -np.inf
        ms.append(top - L.max(1))
        ts.append(t)
        ds.append(np.abs((r[s:s + chunk] - r_hat[s:s + chunk]) @ U.T).max(1))
    return np.concatenate(ts), np.concatenate(ms), np.concatenate(ds)


def certify_python(m: np.ndarray, d: np.ndarray, tau: float = TAU) -> np.ndarray:
    """Python twin of the Datalog rule, on the same fixed-point rounding."""
    mf = np.floor(np.asarray(m) * SCALE).astype(np.int64)
    df = np.ceil(np.asarray(d) * SCALE).astype(np.int64)
    return mf > 2 * df + math.ceil(tau * SCALE)


def certify_souffle(m: np.ndarray, d: np.ndarray, tau: float = TAU) -> np.ndarray:
    """The certificate as a Soufflé query result: a boolean mask. Raises if a context is unaccounted."""
    mf = np.floor(np.asarray(m) * SCALE).astype(np.int64)
    df = np.ceil(np.asarray(d) * SCALE).astype(np.int64)
    with tempfile.TemporaryDirectory(prefix="substitution-certificate-") as raw:
        directory = Path(raw)
        (directory / "certificate.dl").write_text(SOUFFLE_CERTIFICATE)
        for name, values in (("margin", mf), ("delta", df)):
            rows = "".join(f"{i}\t{v}\n" for i, v in enumerate(values.tolist()))
            (directory / f"{name}.facts").write_text(rows)
        (directory / "tau.facts").write_text(f"{math.ceil(tau * SCALE)}\n")
        command = ["souffle", "-F", str(directory), "-D", str(directory), str(directory / "certificate.dl")]
        subprocess.run(command, check=True, capture_output=True, text=True)
        cert = {int(x) for x in (directory / "certified.csv").read_text().split()}
        refused = {int(x) for x in (directory / "refused.csv").read_text().split()}
    if cert & refused or len(cert) + len(refused) != len(mf):
        raise RuntimeError(f"Soufflé accounted for {len(cert) + len(refused)} of {len(mf)} contexts")
    return np.array([i in cert for i in range(len(mf))])


HYBRID_K = 32  # rivals checked pairwise by the hybrid certificate (the top-K of the REAL logits)

SOUFFLE_CERTIFICATES = """
// Three certificates (i-orca PIC_Binding) on fixed-point facts: large-side quantities floored,
// bounds ceiled.
//  uniform  (substitution_certified_max):   m > 2*d + tau
//  pairwise (substitution_pairwise_iff, EXACT): min_v [L(t)-L(v) - <r-rhat, U_t-U_v>] > tau
//  hybrid   (substitution_certified_hybrid):  head slack > tau  and  tail gap > tail bound + tau
.decl margin(c:number, m:number) .input margin
.decl delta(c:number, d:number) .input delta
.decl pslack(c:number, s:number) .input pslack
.decl hslack(c:number, s:number) .input hslack
.decl tgap(c:number, g:number) .input tgap
.decl tbound(c:number, b:number) .input tbound
.decl tau(t:number) .input tau
.decl uniform(c:number) .output uniform
uniform(c) :- margin(c, m), delta(c, d), tau(t), m > 2 * d + t.
.decl pairwise(c:number) .output pairwise
pairwise(c) :- pslack(c, s), tau(t), s > t.
.decl hybrid(c:number) .output hybrid
hybrid(c) :- hslack(c, s), tgap(c, g), tbound(c, b), tau(t), s > t, g > b + t.
.decl context(c:number) .output context
context(c) :- margin(c, _).
"""


def certificate_facts(r, r_hat, U, k: int = HYBRID_K, chunk: int = 256) -> dict:
    """float64 per-context quantities for all three certificates, over the FULL vocabulary."""
    r, r_hat, U = (np.asarray(a, dtype=np.float64) for a in (r, r_hat, U))
    u_max = float(np.sqrt((U ** 2).sum(1)).max())
    out = {key: [] for key in ("t", "m", "d", "pslack", "hslack", "tgap", "tbound", "t_hat")}
    for s in range(0, len(r), chunk):
        rr, dd = r[s:s + chunk], r[s:s + chunk] - r_hat[s:s + chunk]
        L = rr @ U.T
        E = dd @ U.T                                   # <r - r_hat, U_v>
        n = np.arange(len(rr))
        t = L.argmax(1)
        Lt, Et = L[n, t], E[n, t]
        gap = Lt[:, None] - L                          # L(t) - L(v)
        slack = gap - (Et[:, None] - E)                # L(t) - L(v) - <r - r_hat, U_t - U_v>
        slack[n, t] = np.inf
        gap_others = gap.copy()
        gap_others[n, t] = np.inf
        order = np.argsort(-L, axis=1)[:, :k + 1]      # top-(K+1) real logits (includes t)
        in_head = np.zeros_like(L, dtype=bool)
        in_head[n[:, None], order] = True
        head_slack = np.where(in_head, slack, np.inf).min(1)
        tail_gap = np.where(in_head, np.inf, gap_others).min(1)
        out["t"].append(t)
        out["m"].append(gap_others.min(1))
        out["d"].append(np.abs(E).max(1))
        out["pslack"].append(slack.min(1))
        out["hslack"].append(head_slack)
        out["tgap"].append(tail_gap)
        out["tbound"].append(np.abs(Et) + np.sqrt((dd ** 2).sum(1)) * u_max)
        out["t_hat"].append((r_hat[s:s + chunk] @ U.T).argmax(1))
    return {key: np.concatenate(v) for key, v in out.items()}


def _fixed(facts: dict, tau: float) -> dict:
    floor = lambda x: np.floor(np.asarray(x) * SCALE).astype(np.int64)  # noqa: E731
    ceil = lambda x: np.ceil(np.asarray(x) * SCALE).astype(np.int64)  # noqa: E731
    big = np.int64(2 ** 62)
    fx = {"margin": floor(facts["m"]), "delta": ceil(facts["d"]), "pslack": floor(facts["pslack"]),
          "hslack": np.where(np.isinf(facts["hslack"]), big, floor(np.nan_to_num(facts["hslack"], posinf=0))),
          "tgap": np.where(np.isinf(facts["tgap"]), big, floor(np.nan_to_num(facts["tgap"], posinf=0))),
          "tbound": ceil(facts["tbound"])}
    return fx, int(math.ceil(tau * SCALE))


def certify_all_python(facts: dict, tau: float = TAU) -> dict:
    fx, t = _fixed(facts, tau)
    return {"uniform": fx["margin"] > 2 * fx["delta"] + t, "pairwise": fx["pslack"] > t,
            "hybrid": (fx["hslack"] > t) & (fx["tgap"] > fx["tbound"] + t)}


def certify_all_souffle(facts: dict, tau: float = TAU) -> dict:
    fx, t = _fixed(facts, tau)
    n = len(fx["margin"])
    with tempfile.TemporaryDirectory(prefix="substitution-certificates-") as raw:
        directory = Path(raw)
        (directory / "certificates.dl").write_text(SOUFFLE_CERTIFICATES)
        for name, values in fx.items():
            rows = "".join(f"{i}\t{v}\n" for i, v in enumerate(values.tolist()))
            (directory / f"{name}.facts").write_text(rows)
        (directory / "tau.facts").write_text(f"{t}\n")
        command = ["souffle", "-F", str(directory), "-D", str(directory), str(directory / "certificates.dl")]
        subprocess.run(command, check=True, capture_output=True, text=True)
        read = lambda name: {int(x) for x in (directory / f"{name}.csv").read_text().split()}  # noqa: E731
        if read("context") != set(range(n)):
            raise RuntimeError("Soufflé did not account for every context")
        names = ("uniform", "pairwise", "hybrid")
        return {name: np.array([i in read(name) for i in range(n)]) for name in names}


def coverage_report(r: np.ndarray, r_hat: np.ndarray, U: np.ndarray, use_souffle: bool = True,
                    k: int = HYBRID_K) -> dict:
    """Coverage (fraction of contexts certified) for the uniform, pairwise and hybrid certificates.

    Soundness is checked empirically: a certified context must keep its argmax under r_hat. The theorems say
    it must, so a failure would be a checker bug. Top-level `certified`/`coverage` keep the uniform form.
    """
    facts = certificate_facts(r, r_hat, U, k=k)
    certs = certify_all_souffle(facts) if use_souffle else certify_all_python(facts)
    agree = facts["t_hat"] == facts["t"]
    for name, cert in certs.items():
        if (cert & ~agree).any():
            raise AssertionError(f"a {name}-certified context changed its argmax: checker is unsound")
    if (certs["uniform"] & ~certs["pairwise"]).any() or (certs["hybrid"] & ~certs["pairwise"]).any():
        raise AssertionError("uniform/hybrid certified a context the exact pairwise test rejects")
    m, d = facts["m"], facts["d"]
    cert = certs["uniform"]
    report = dict(n=int(len(m)), certified=int(cert.sum()), coverage=float(cert.mean()),
                  agree_all=float(agree.mean()),
                  agree_uncertified=float(agree[~cert].mean()) if (~cert).any() else None,
                  median_margin=float(np.median(m)), median_delta=float(np.median(d)),
                  frac_delta_lt_half_margin=float((2 * d < m).mean()), hybrid_k=k)
    for name, c in certs.items():
        report[name] = dict(certified=int(c.sum()), coverage=float(c.mean()))
    return report
