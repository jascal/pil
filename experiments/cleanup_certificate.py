"""The T6(b) clean-up certificate on GPT-2's decode input.

Pre-registered: docs/notes/cleanup_certificate_prereg.md.

    # 1. dump (needs `transformers`):
    /path/to/venv/bin/python experiments/cleanup_certificate.py dump --out runs/cleanup
    # 2. run (needs souffle):
    /path/to/venv/bin/python experiments/cleanup_certificate.py run --dumps runs/cleanup \
        --out runs/cleanup/summary.json

Theorem (i-orca#30, PIC_Cleanup.cleanup_certified, kernel-checked). Code points x(sigma) = W vec(T(sigma)) +
b0, host residual u = x(sigma) + n. Clean-up decodes P(u - b0) with a left inverse P of W (|P y| <= K |y|),
unbinds each role s with a readout w_s (<r_s, w_s> = 1), snaps to the nearest filler in F_s and rebinds. If
|n| < min(rho, beta), clean-up returns x(sigma) exactly AND the host decides the code point's decision,
where
    rho(sigma)  = min_{s in D(sigma)} (gamma_s / 2 - kappa_s) / (K |w_s|)   (kappa_s = 0 for dual readouts),
    beta(sigma) = min_{v != t} m_v(x) / |U_t - U_v|                           (the code point's own margins).

Tasks, the TPR family and the mse/cert objectives are certified_substitutes.py's (#136), unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import certified_substitutes as cs  # noqa: E402
from substitution_certificate import certificate_facts, certify_all_souffle  # noqa: E402

SEEDS = dict(stimulus=41, split=42, fit=(0, 1, 2))
SETTINGS = (("LIST", 8), ("SVO", 8), ("SVO", 32))
OBJECTIVES = ("mse", "cert", "t6")
SCALE = 2**30
TAU = 1e-6
FLAG_RATIO = 1e-8  # sigma_min / sigma_max below this: no left inverse
FLAG_PINV = 1e-8  # |P W - I|_max above this: no left inverse
FLAG_COND = 1e8  # cond(er) above this: no dual readout
T6_OFFSET = 0.1
BIG = 2**62

SOUFFLE = """
// T6(b): certified(c) iff |n| + tau < rho and |n| + tau < beta.
// Fixed point: |n| ceiled, rho and beta floored.
.decl n(c:number, x:number) .input n
.decl rho(c:number, x:number) .input rho
.decl beta(c:number, x:number) .input beta
.decl tau(t:number) .input tau
.decl certified(c:number) .output certified
certified(c) :- n(c, a), rho(c, r), beta(c, b), tau(t), a + t < r, a + t < b.
.decl context(c:number) .output context
context(c) :- n(c, _).
"""


# ---------------------------------------------------------------- TPR structure


def tpr_params(model) -> dict:
    """float64 numpy copies of a cs.TPR: ef (n_fill, d_F), er (n_role, n_role), W (d, d_F*n_role), b0 (d,)."""
    g = lambda t: t.detach().double().cpu().numpy()  # noqa: E731
    return dict(ef=g(model.ef.weight), er=g(model.er.weight), W=g(model.W.weight), b0=g(model.W.bias))


def tensors(p: dict, f: np.ndarray, r: np.ndarray, m: np.ndarray) -> np.ndarray:
    """T(sigma)[i, j] = sum_p ef[f_p, i] er[r_p, j] over bound pairs; shape (N, d_F, n_role).
    Matches cs.TPR's einsum("bpi,bpj->bpij") and its row-major flatten."""
    return np.einsum("npi,npj->nij", p["ef"][f] * m[..., None], p["er"][r])


def code_points(p: dict, f, r, m) -> np.ndarray:
    T = tensors(p, f, r, m)
    return T.reshape(len(T), -1) @ p["W"].T + p["b0"]


def readout(p: dict) -> dict:
    """Left inverse P = W^+, K = 1/sigma_min(W), dual role readouts w_s = (er^-1)[:, s], and the two flags."""
    W, er = p["W"], p["er"]
    sv = np.linalg.svd(W, compute_uv=False)
    ratio = float(sv.min() / sv.max())
    P = np.linalg.pinv(W)
    pinv_err = float(np.abs(P @ W - np.eye(W.shape[1])).max())
    cond = float(np.linalg.cond(er))
    no_left_inverse = bool(ratio < FLAG_RATIO or pinv_err > FLAG_PINV)
    no_dual = bool(not np.isfinite(cond) or cond > FLAG_COND)
    Wd = np.linalg.inv(er) if not no_dual else np.full_like(er, np.nan)
    return dict(
        P=P,
        K=float(1 / sv.min()),
        Wd=Wd,
        sigma_ratio=ratio,
        pinv_err=pinv_err,
        cond_er=cond,
        no_left_inverse=no_left_inverse,
        no_dual_readout=no_dual,
    )


def filler_sets(f, r, m, n_role: int) -> list[np.ndarray]:
    """F_s = fillers seen in role s (train)."""
    sets = [set() for _ in range(n_role)]
    for fi, ri, mi in zip(f.ravel(), r.ravel(), m.ravel(), strict=True):
        if mi:
            sets[int(ri)].add(int(fi))
    return [np.array(sorted(s), dtype=np.int64) for s in sets]


def gammas(ef: np.ndarray, F_sets: list[np.ndarray]) -> np.ndarray:
    """gamma_s = min distance between distinct filler embeddings in F_s (inf if |F_s| < 2)."""
    out = np.full(len(F_sets), np.inf)
    for s, Fs in enumerate(F_sets):
        if len(Fs) >= 2:
            E = ef[Fs]
            d2 = (E**2).sum(1)[:, None] + (E**2).sum(1)[None] - 2 * E @ E.T
            np.fill_diagonal(d2, np.inf)
            out[s] = math.sqrt(max(float(d2.min()), 0.0))
    return out


def rho(r, m, gam: np.ndarray, K: float, wnorm: np.ndarray) -> np.ndarray:
    """rho(sigma) = min over bound roles of gamma_s / (2 K |w_s|) (dual readouts: zero crosstalk)."""
    per = gam / (2 * K * wnorm)
    vals = np.where(m.astype(bool), per[r], np.inf)
    return vals.min(1)


def beta_and_decision(x: np.ndarray, U: np.ndarray, device: str = "cpu", chunk: int = 256):
    """The code point's own decision t, and beta = min_{v != t} (L(t) - L(v)) / |U_t - U_v| (full vocab)."""
    Ud = torch.tensor(U, dtype=torch.float64, device=device)
    un2 = (Ud * Ud).sum(1)
    ts, bs = [], []
    for s in range(0, len(x), chunk):
        xs = torch.tensor(x[s : s + chunk], dtype=torch.float64, device=device)
        L = xs @ Ud.T
        t = L.argmax(1)
        rows = torch.arange(len(t), device=device)
        gap = L[rows, t][:, None] - L
        Ut = Ud[t]
        d2 = un2[t][:, None] + un2[None] - 2 * Ut @ Ud.T
        dist = torch.sqrt(torch.clamp(d2, min=0))
        ratio = torch.where(
            dist > 0,
            gap / torch.where(dist > 0, dist, torch.ones_like(dist)),
            torch.where(gap > 0, torch.full_like(gap, math.inf), torch.zeros_like(gap)),
        )
        ratio[rows, t] = math.inf
        ts.append(t.cpu().numpy())
        bs.append(ratio.min(1).values.cpu().numpy())
    return np.concatenate(ts), np.concatenate(bs)


def cleanup(u: np.ndarray, p: dict, rd: dict, F_sets, f, r, m) -> tuple[np.ndarray, np.ndarray]:
    """Clean-up: Ttilde = P(u - b0); g_s = Ttilde w_s; nearest filler in F_s. Returns (sigma_hat, exact)."""
    d_f, n_role = p["ef"].shape[1], p["er"].shape[0]
    Tt = ((u - p["b0"]) @ rd["P"].T).reshape(len(u), d_f, n_role)
    sig_hat = np.full(f.shape, -1, dtype=np.int64)
    for j in range(f.shape[1]):
        bound = m[:, j].astype(bool)
        for s in np.unique(r[bound, j]):
            idx = np.where(bound & (r[:, j] == s))[0]
            g = Tt[idx] @ rd["Wd"][:, s]  # (k, d_F)
            Fs = F_sets[int(s)]
            E = p["ef"][Fs]
            d2 = (g**2).sum(1)[:, None] + (E**2).sum(1)[None] - 2 * g @ E.T
            sig_hat[idx, j] = Fs[d2.argmin(1)]
    exact = np.where(m.astype(bool), sig_hat == f, True).all(1)
    return sig_hat, exact


# ---------------------------------------------------------------- verdict


def _fixed(n, rh, be, tau=TAU):
    up = lambda a: np.ceil(np.asarray(a) * SCALE).astype(np.int64)  # noqa: E731

    def down(a):
        a = np.asarray(a, dtype=np.float64)
        out = np.floor(np.where(np.isfinite(a), a, 0) * SCALE).astype(np.int64)
        return np.where(np.isposinf(a), BIG, np.where(np.isneginf(a), -BIG, out))

    return up(n), down(rh), down(be), int(math.ceil(tau * SCALE))


def verdict_python(n, rh, be, tau=TAU) -> np.ndarray:
    a, r_, b, t = _fixed(n, rh, be, tau)
    return (a + t < r_) & (a + t < b)


def verdict_souffle(n, rh, be, tau=TAU) -> np.ndarray:
    a, r_, b, t = _fixed(n, rh, be, tau)
    with tempfile.TemporaryDirectory(prefix="cleanup-certificate-") as raw:
        d = Path(raw)
        (d / "cleanup.dl").write_text(SOUFFLE)
        for name, vals in (("n", a), ("rho", r_), ("beta", b)):
            (d / f"{name}.facts").write_text("".join(f"{i}\t{v}\n" for i, v in enumerate(vals.tolist())))
        (d / "tau.facts").write_text(f"{t}\n")
        subprocess.run(
            ["souffle", "-F", str(d), "-D", str(d), str(d / "cleanup.dl")],
            check=True,
            capture_output=True,
            text=True,
        )
        read = lambda nm: {int(x) for x in (d / f"{nm}.csv").read_text().split()}  # noqa: E731
        if read("context") != set(range(len(a))):
            raise RuntimeError("Soufflé did not account for every context")
        cert = read("certified")
        return np.array([i in cert for i in range(len(a))])


def soundness(cert: np.ndarray, exact: np.ndarray, t_code: np.ndarray, t_host: np.ndarray) -> None:
    """Theorem: a certified context recovers sigma exactly AND the host decides the code's t. Else abort."""
    bad_exact = cert & ~exact
    bad_agree = cert & (t_code != t_host)
    if bad_exact.any() or bad_agree.any():
        raise AssertionError(
            f"unsound: {int(bad_exact.sum())} certified contexts with inexact clean-up, "
            f"{int(bad_agree.sum())} with host disagreement"
        )


# ---------------------------------------------------------------- the t6 objective


def train_t6(model, f, r, m, u, U, seed, steps=None, batch=512, lr=3e-3):
    """0.1 |x - u|^2 + relu(|u - x| - beta(x) + 0.1), batch mean; beta over the full vocabulary with
    t = the host decision.
    Seeding and batching mirror cs.train."""
    steps = steps or cs.STEPS
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    un2 = (U * U).sum(1)
    for _ in range(steps):
        idx = torch.randint(0, len(u), (batch,), generator=g).to(u.device)
        x = model(f[idx], r[idx], m[idx])
        ub = u[idx]
        with torch.no_grad():
            t = (ub @ U.T).argmax(1)
            dist = torch.sqrt(torch.clamp(un2[t][:, None] + un2[None] - 2 * U[t] @ U.T, min=1e-12))
        Lx = x @ U.T
        ratio = (Lx.gather(1, t[:, None]) - Lx) / dist
        ratio = ratio.scatter(1, t[:, None], float("inf"))
        beta = ratio.min(1).values
        err = torch.sqrt(((ub - x) ** 2).sum(1) + 1e-12)
        loss = (0.1 * ((x - ub) ** 2).sum(1) + F.relu(err - beta + T6_OFFSET)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


# ---------------------------------------------------------------- evaluation and decision rules


def evaluate(p, rd, F_sets, gam, u_te, f_te, r_te, m_te, U, t_host, device):
    """Every pre-registered per-seed metric for one fitted TPR."""
    x = code_points(p, f_te, r_te, m_te)
    n = np.sqrt(((u_te - x) ** 2).sum(1))
    t_code, be = beta_and_decision(x, U, device)
    uniform = float(certify_all_souffle(certificate_facts(u_te, x, U))["uniform"].mean())
    flagged = rd["no_left_inverse"] or rd["no_dual_readout"]
    out = dict(
        uniform_coverage=uniform,
        cleaned_agreement=float((t_code == t_host).mean()),
        flags=dict(
            no_left_inverse=rd["no_left_inverse"],
            no_dual_readout=rd["no_dual_readout"],
            sigma_ratio=rd["sigma_ratio"],
            pinv_err=rd["pinv_err"],
            cond_er=rd["cond_er"],
        ),
        median_n=float(np.median(n)),
        median_beta=float(np.median(be)),
    )
    if flagged:
        return out | dict(t6_coverage=0.0, exact_cleanup=None, R=None, B=None, median_rho=None, flagged=True)
    wnorm = np.sqrt((rd["Wd"] ** 2).sum(0))
    rh = rho(r_te, m_te, gam, rd["K"], wnorm)
    _, exact = cleanup(u_te, p, rd, F_sets, f_te, r_te, m_te)
    cert = verdict_souffle(n, rh, be)
    if not (cert == verdict_python(n, rh, be)).all():
        raise AssertionError("Soufflé and Python verdicts differ")
    soundness(cert, exact, t_code, t_host)
    unc = ~cert
    R = float((n[unc] >= rh[unc]).mean()) if unc.any() else None
    B = float((n[unc] >= be[unc]).mean()) if unc.any() else None
    return out | dict(
        t6_coverage=float(cert.mean()),
        exact_cleanup=float(exact.mean()),
        R=R,
        B=B,
        median_rho=float(np.median(rh)),
        K=rd["K"],
        flagged=False,
    )


def bottleneck(R, B):
    if R is None or B is None:
        return "none uncertified"
    if R >= 0.8 and B >= 0.8:
        return "both"
    if R >= 0.8:
        return "rho-limited"
    if B >= 0.8:
        return "beta-limited"
    return "mixed"


def summarise(per_seed: list[dict]) -> dict:
    keys = (
        "t6_coverage",
        "uniform_coverage",
        "exact_cleanup",
        "cleaned_agreement",
        "R",
        "B",
        "median_n",
        "median_rho",
        "median_beta",
    )
    cell = {}
    for k in keys:
        vals = [s[k] for s in per_seed if s.get(k) is not None]
        cell[k] = (
            dict(mean=float(np.mean(vals)), range=[float(min(vals)), float(max(vals))]) if vals else None
        )
    cell["flagged_seeds"] = sum(bool(s["flagged"]) for s in per_seed)
    mean = lambda k: cell[k]["mean"] if cell[k] else None  # noqa: E731
    cell["bottleneck"] = bottleneck(mean("R"), mean("B"))
    return cell


def verdicts(cells: dict) -> dict:
    h1, h2 = {}, {}
    for task, d_f in SETTINGS:
        key = f"{task}/dF{d_f}"
        cov = {o: cells[f"{key}/{o}"]["t6_coverage"]["mean"] for o in OBJECTIVES}
        best = max(cov, key=cov.get)
        uni = cells[f"{key}/{best}"]["uniform_coverage"]["mean"]
        h1[key] = dict(
            best_objective=best,
            t6=cov[best],
            uniform=uni,
            passes=bool(cov[best] >= 0.05 and cov[best] >= 10 * uni),
        )
        h2[key] = dict(
            t6=cov["t6"],
            mse=cov["mse"],
            cert=cov["cert"],
            passes=bool(cov["t6"] >= max(cov["mse"], cov["cert"]) + 0.03),
        )
    n1 = sum(v["passes"] for v in h1.values())
    n2 = sum(v["passes"] for v in h2.values())
    return dict(
        H1=dict(settings=h1, n_pass=n1, overall="yes" if n1 >= 2 else ("partial" if n1 == 1 else "no")),
        H2=dict(settings=h2, n_pass=n2, passes=n2 >= 2),
    )


# ---------------------------------------------------------------- dump / run


def cmd_dump(args):
    cs.SEEDS["stimulus"] = 999 if args.smoke else SEEDS["stimulus"]
    if args.smoke:
        cs.N_CONTEXTS = 600
    cs.cmd_dump(args)


def cmd_run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        cs.STEPS = 30
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)  # noqa: E731
    summary = dict(
        tag="empirical",
        prereg="docs/notes/cleanup_certificate_prereg.md",
        smoke=bool(args.smoke),
        seeds=dict(SEEDS, stimulus=999 if args.smoke else SEEDS["stimulus"]),
        tasks={},
        cells={},
    )
    for task in ("LIST", "SVO"):
        z = np.load(Path(args.dumps) / f"{task}.npz")
        n_fill, n_role = int(z["n_fill"]), int(z["n_role"])
        fa, ra, ma = z["f"], z["r"], z["m"]
        for i in range(len(fa)):
            bound = ra[i][ma[i].astype(bool)]
            assert len(set(bound.tolist())) == len(bound), "a role is bound twice in one context"
        n = len(z["u"])
        order = np.random.default_rng(SEEDS["split"]).permutation(n)
        n_tr, n_va = int(0.6 * n), int(0.1 * n)
        tr, te = order[:n_tr], order[n_tr + n_va :]
        seen = {(int(a), int(s)) for i in tr for a, s, k in zip(fa[i], ra[i], ma[i], strict=True) if k}
        keep = [
            i
            for i in te
            if all((int(a), int(s)) in seen for a, s, k in zip(fa[i], ra[i], ma[i], strict=True) if k)
        ]
        dropped = len(te) - len(keep)
        te = np.array(keep)
        U = z["U"].astype(np.float64)
        u_te = z["u"][te].astype(np.float64)
        t_host = (u_te @ U.T).argmax(1)
        F_sets = filler_sets(fa[tr], ra[tr], ma[tr], n_role)
        summary["tasks"][task] = dict(
            n_train=len(tr),
            n_test=len(te),
            test_dropped_unseen_pair=dropped,
            filler_set_sizes=[len(s) for s in F_sets],
        )
        log(f"{task}: train {len(tr)} test {len(te)} (dropped {dropped})")
        T = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)  # noqa: E731
        f_t, r_t, m_t, u_t, U_t = T(fa, torch.long), T(ra, torch.long), T(ma), T(z["u"]), T(z["U"])
        tr_t = torch.tensor(tr, device=device)
        for s_task, d_f in SETTINGS:
            if s_task != task:
                continue
            for obj in OBJECTIVES:
                per_seed = []
                for seed in SEEDS["fit"]:
                    model = cs.TPR(n_fill, n_role, d_f).to(device)
                    args_tr = (model, f_t[tr_t], r_t[tr_t], m_t[tr_t], u_t[tr_t], U_t)
                    if obj == "t6":
                        train_t6(*args_tr, seed)
                    else:
                        cs.train(*args_tr, obj, seed)
                    p = tpr_params(model)
                    rd = readout(p)
                    gam = gammas(p["ef"], F_sets)
                    res = evaluate(p, rd, F_sets, gam, u_te, fa[te], ra[te], ma[te], U, t_host, str(device))
                    res["seed"] = seed
                    per_seed.append(res)
                    log(
                        f"{task} dF{d_f} {obj:4s} seed {seed}: t6 {res['t6_coverage']:.4f} "
                        f"uniform {res['uniform_coverage']:.4f} exact {res['exact_cleanup']} "
                        f"agree {res['cleaned_agreement']:.3f} |n| {res['median_n']:.3f} "
                        f"rho {res['median_rho']} beta {res['median_beta']:.4f} flagged {res['flagged']}"
                    )
                key = f"{task}/dF{d_f}/{obj}"
                summary["cells"][key] = dict(per_seed=per_seed, **summarise(per_seed))
    summary["verdict"] = verdicts(summary["cells"])
    summary["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["verdict"], indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", required=True)
    d.add_argument("--smoke", action="store_true", help="bug check: stimulus seed 999, 600 contexts")
    rr = sub.add_parser("run")
    rr.add_argument("--dumps", required=True)
    rr.add_argument("--out", required=True)
    rr.add_argument("--smoke", action="store_true", help="bug check: 30 steps; numbers are NOT results")
    args = ap.parse_args()
    {"dump": cmd_dump, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
