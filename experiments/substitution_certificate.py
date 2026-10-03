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


def coverage_report(r: np.ndarray, r_hat: np.ndarray, U: np.ndarray, use_souffle: bool = True) -> dict:
    """Coverage = fraction of contexts certified.

    Also checks soundness empirically: on every certified context the substituted argmax must equal the
    real one. The theorem says it must, so a failure would be a checker bug.
    """
    t, m, d = margins_and_deltas(r, r_hat, U)
    cert = certify_souffle(m, d) if use_souffle else certify_python(m, d)
    U64 = np.asarray(U, dtype=np.float64)
    t_hat = np.concatenate(
        [(np.asarray(r_hat[s:s + 256], dtype=np.float64) @ U64.T).argmax(1)
         for s in range(0, len(r_hat), 256)]
    )
    agree = t_hat == t
    if (cert & ~agree).any():
        raise AssertionError("a certified context changed its argmax: checker is unsound")
    return dict(n=int(len(t)), certified=int(cert.sum()), coverage=float(cert.mean()),
                agree_all=float(agree.mean()),
                agree_uncertified=float(agree[~cert].mean()) if (~cert).any() else None,
                median_margin=float(np.median(m)), median_delta=float(np.median(d)),
                frac_delta_lt_half_margin=float((2 * d < m).mean()))
