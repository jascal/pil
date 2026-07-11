"""Wikitext residual anatomy and registered flat-vs-gated rescue contrast (slice #81).

This campaign freezes its first score-qualified B0 regeneration and reloads that fitted state on
later runs.  The helpers remain import-safe so their invariants can be unit-tested cheaply.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

# v5 reads these at import time.  Assignment (rather than setdefault) makes this recipe canonical.
_RECIPE = {
    "WYLY_TAG": "pythia70m",
    "WYLY_DS": "wikitext",
    "WYLY_LIB": "mined",
    "WYLY_JUDGE": "cover",
    "WYLY_ONLINE": "1",
    "WYLY_COVER": "sw",
    "WYLY_CONCEPTS": "1",
    "WYLY_POINTER": "1",
    "WYLY_TPOINTER": "1",
    "WYLY_DX": "1",
    "WYLY_CX": "1",
    "WYLY_FOLDS": "3",
}
for _key, _value in _RECIPE.items():
    os.environ[_key] = _value
os.environ.pop("WYLY_EMIT", None)

import wyly_lm_v5 as v5  # noqa: E402
from wyly_ladder import MODELS, copy_incidence  # noqa: E402

from pil.wyly_block import BlockStack, WylyBlock  # noqa: E402

EXPECTED_B0 = {
    "induction L=2",
    "cmember",
    "prevsent-head gate",
    "kgram k=2 (online s40)",
    "induction L=3",
    "sincedot gate",
    "cap-echo gate",
    "mined frames (online)",
}
CANDIDATE_FAMILIES = (
    "pointer",
    "dstate gate",
    "sentpair/dgate2",
    "attrib-subj gate",
    "mined frames",
)
ADMIT_THRESH = 5e-4
MAX_RULES = 5
HONESTY_NOTE = (
    "all candidates are template_fixed-class (not learned end-to-end); frac_induced is "
    "unaffected by this experiment; FLAT/GATED declines are re-established fresh in this run, "
    "not inherited from any prior document."
)
VERDICT_RULE = (
    '"gated rescues" iff ALL of: (i) gated admits >=1 family that flat declines in the SAME run; '
    "(ii) gated `test_te` agree >= flat `test_te` agree + 0.005; (iii) two-sided exact binomial "
    "p < 0.05 on the discordant flips; (iv) gated regressions <= flat's regressions."
)
B0_FREEZE = REPO / "data" / "wikitext_gated_rescue_b0.pt"
B0_SCORE_MIN = 0.341
B0_SCORE_MAX = 0.351
B0_RELOAD_TOLERANCE = 1e-6
CROSS_RUN_FINDING = "admission composition is run-variant at fixed aggregate performance"


def log(message: str = "") -> None:
    print(message, flush=True)


def _base_rule_name(name: str) -> str:
    """Remove only a fitted key-count suffix, retaining meaningful parenthesized qualifiers."""
    return re.sub(r"\s+\[\d+\]$", "", name)


def b0_within_registered_band(agree: float) -> bool:
    """The amended, inclusive score-only registration gate."""
    return B0_SCORE_MIN <= agree <= B0_SCORE_MAX


def _candidate_family_for_rule(name: str) -> str | None:
    """Map an admitted fitted rule name onto one of the five registered pool families."""
    base = _base_rule_name(name)
    if base.startswith("pointer"):
        return "pointer"
    if base == "dstate gate":
        return "dstate gate"
    if base in {"sentpair gate", "sentpair/dgate2"}:
        return "sentpair/dgate2"
    if base == "attrib-subj gate":
        return "attrib-subj gate"
    if base.startswith("mined frames"):
        return "mined frames"
    return None


def active_candidate_families(
    b0_rule_names: list[str] | tuple[str, ...] | set[str],
) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Exclude canonical candidates whose family is already admitted into this B0 prime."""
    present = {
        family for name in b0_rule_names if (family := _candidate_family_for_rule(name))
    }
    exclusions = [
        {"family": family, "reason": "already admitted into B0′"}
        for family in CANDIDATE_FAMILIES
        if family in present
    ]
    active = tuple(family for family in CANDIDATE_FAMILIES if family not in present)
    assert not (set(active) & present)
    return active, exclusions


def _dfeat_feature(kindname: str, params: dict[str, Any]):
    if kindname == "recent-member":
        if "succ" in params:
            return lambda w: v5.at_pos(
                w, v5.recent_member_pos(w, params["members"]), succ=params["succ"]
            )
        return lambda w: v5.recent_member_feature(w, params["members"])
    if kindname == "recent-unique":
        return lambda w: v5.recent_member_feature(w, params["members"], unique=True)
    if kindname == "since-member":
        return lambda w: v5.since_member_feature(
            w, params["members"], cap=params.get("cap", 32)
        )
    if kindname == "member-parity":
        return lambda w: v5.member_parity_feature(w, params["members"])
    if kindname == "dstate":
        return lambda w: v5.dstate_feature(
            w, params["members"], params["quote_members"]
        )
    if kindname == "prev-occ":
        def feature(w):
            base_pos = v5.recent_member_pos(w, params["of_members"])
            base_pos = (base_pos + params.get("of_shift", 0)).clamp(min=-1)
            if "avoid" in params:
                pos = v5.prev_occ_pos_filtered(w, base_pos, params["avoid"])
            else:
                pos = v5.prev_occ_pos(w, base_pos)
            return v5.at_pos(w, pos, succ=params.get("succ", 0))

        return feature
    raise ValueError(f"unsupported frozen dfeat kind: {kindname}")


def _reconstruct_rule(name: str, spec: tuple[Any, ...]):
    kind, *payload = spec
    if kind == "kgram":
        (online_frame,) = payload
        fn = online_frame.mirror()

        def conf_fn(ids, online_frame=online_frame):
            return online_frame.table.lookup_conf(online_frame._suffix_key(ids, False))

        return fn, conf_fn
    if kind == "mined":
        (mined,) = payload
        return mined.mirror(), mined.lookup_conf_all
    if kind == "dfeat":
        (kindname, params), table, base = payload
        feature = _dfeat_feature(kindname, params)

        def conf_fn(w, feature=feature, table=table, base=base):
            feat = feature(w)
            pred, conf = table.lookup_conf((feat + 1) * base + w[:, -1])
            miss = feat < 0
            return (
                torch.where(miss, torch.full_like(pred, -1), pred),
                torch.where(miss, torch.full_like(conf, -1e9), conf),
            )

        return (lambda w, conf_fn=conf_fn: conf_fn(w)[0]), conf_fn
    raise ValueError(f"unsupported frozen B0 rule kind {kind!r} for {name!r}")


def freeze_b0(
    path: Path,
    model,
    rules,
    core: dict[str, float],
    b0_wall_seconds: float,
    *,
    emit_info: dict[str, tuple[Any, ...]] | None = None,
    rule_conf: dict[str, float] | None = None,
) -> None:
    """Save fitted module-level state, never the unpicklable rule closures themselves."""
    emit = v5.EMIT_INFO if emit_info is None else emit_info
    scalar_conf = v5.RULE_CONF if rule_conf is None else rule_conf
    frozen_rules = []
    for name, _fn in rules:
        match = re.fullmatch(r"induction L=(\d+)", name)
        if match:
            frozen_rules.append(
                {"name": name, "kind": "induction", "n": int(match.group(1)),
                 "conf": float(scalar_conf[name])}
            )
            continue
        if name not in emit:
            raise ValueError(f"cannot freeze {name!r}: no EMIT_INFO reconstruction spec")
        kind = emit[name][0]
        if kind not in {"kgram", "mined", "dfeat"}:
            raise ValueError(f"cannot freeze {name!r}: unsupported EMIT_INFO kind {kind!r}")
        frozen_rules.append({"name": name, "kind": kind, "spec": emit[name]})
    payload = {
        "rules": frozen_rules,
        "rule_names": [name for name, _ in rules],
        "counts": model.counts,
        "counts_calib": getattr(model, "counts_calib", None),
        "core_sw": dict(core),
        "b0_wall_seconds": float(b0_wall_seconds),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_frozen_b0(path: Path):
    """Load fitted B0 state and reconstruct fresh runtime closures and confidence globals."""
    payload = torch.load(path, map_location=v5.DEV, weights_only=False)
    model = SimpleNamespace(
        counts=payload["counts"], counts_calib=payload["counts_calib"], rules2=[]
    )
    rules = []
    for frozen in payload["rules"]:
        name = frozen["name"]
        if frozen["kind"] == "induction":
            fn = v5.mir_induction_L(frozen["n"])
            v5.RULE_CONF[name] = float(frozen["conf"])
        else:
            fn, conf_fn = _reconstruct_rule(name, frozen["spec"])
            v5.CONF_FNS[name] = conf_fn
        rules.append((name, fn))
    if [name for name, _ in rules] != payload["rule_names"]:
        raise AssertionError("frozen B0 rule order changed during reconstruction")
    # MinedGates.mine uses the live model's finalized rules to identify residual errors.
    model.rules = rules
    return model, rules, payload["core_sw"], float(payload["b0_wall_seconds"])


def split_te(
    te: torch.Tensor, seed: int = 0, val_fraction: float = 0.25
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pinned seed split: 25% validation and the remaining 75% one-shot test."""
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(te), generator=g, device="cpu").to(te.device)
    nval = int(len(te) * val_fraction)
    return te[order[:nval]], te[order[nval:]]


def tau_deciles(conf: torch.Tensor) -> list[float]:
    """The fixed 10th..100th percentiles, retaining duplicate values if present."""
    q = torch.arange(1, 11, device=conf.device, dtype=torch.float64) / 10
    return [float(x) for x in torch.quantile(conf.double(), q).cpu()]


def exact_discordant_p(b: int, c: int) -> float:
    """Two-sided exact binomial test for discordant paired outcomes."""
    return 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)


def mining_fit_slice(
    tr: torch.Tensor,
    tr_conf: torch.Tensor,
    tau: float,
    val_te: torch.Tensor,
    test_te: torch.Tensor,
) -> torch.Tensor:
    """Select train low-confidence rows and fail loudly on any held-out overlap."""
    if len(tr) != len(tr_conf):
        raise ValueError("tr/tr_conf length mismatch")
    fit = tr[tr_conf <= tau]
    if bool(torch.isin(fit, val_te).any()) or bool(torch.isin(fit, test_te).any()):
        raise ValueError("mined-frames fit slice overlaps held-out te")
    return fit


def core_cover_sw_traced(model, rules, ids, yv, cls, idxs):
    """Exact traced mirror of v5.core_cover_sw for the registered non-STRATA B0 recipe."""
    w = ids[idxs]
    pred = torch.full_like(w[:, -1], -1)
    conf = torch.full((len(w),), -1e9, device=w.device)
    winner = ["abstain"] * len(w)

    def consider(a: torch.Tensor, c: torch.Tensor, name: str) -> None:
        nonlocal pred, conf, winner
        m = (a >= 0) & (c > conf)
        pred = torch.where(m, a, pred)
        conf = torch.where(m, c, conf)
        for i in m.nonzero(as_tuple=False).flatten().cpu().tolist():
            winner[i] = name

    for name, fn in rules:
        if name in v5.CONF_FNS:
            a, c = v5.CONF_FNS[name](w)
        else:
            a = fn(w)
            c = torch.full(
                (len(w),), float(v5.RULE_CONF.get(name, 0.0)), device=w.device
            )
        consider(a, c, name)
    t = w[:, -1]
    row = model.counts[t]
    mx, am = row.max(1)
    tot = row.sum(1)
    a = torch.where(mx >= 1, cls[am], torch.full_like(t, -1))
    c = mx.float() / (tot + v5.ALPHA)
    ccal = getattr(model, "counts_calib", None)
    if ccal is not None:
        calibrated = ccal[t]
        c = torch.where(calibrated >= 0, calibrated, c)
    consider(a, c, "counts")

    # Defensive named tier. It is inactive under this recipe, exactly like original core_cover_sw.
    if v5.STRATA and getattr(model, "rules2", None):
        low = conf < v5.STRATA_TAU
        for name, fn in model.rules2:
            if name in v5.CONF_FNS:
                a2, c2 = v5.CONF_FNS[name](w)
            else:
                a2 = fn(w)
                c2 = torch.full(
                    (len(w),), float(v5.RULE_CONF.get(name, 0.0)), device=w.device
                )
            m = low & (a2 >= 0) & (c2 > conf)
            pred = torch.where(m, a2, pred)
            conf = torch.where(m, c2, conf)
            for i in m.nonzero(as_tuple=False).flatten().cpu().tolist():
                winner[i] = f"stratum-2/{name}"

    correct = pred == yv[idxs]
    fired = pred >= 0
    out = {
        "agree": float(correct.float().mean()),
        "cover": float(fired.float().mean()),
        "agree_fired": float(correct[fired].float().mean()) if bool(fired.any()) else 0.0,
    }
    trace = {"name": winner, "conf": conf, "correct": correct, "pred": pred}
    return out, pred, trace


def _counts_row_fn(model, cls):
    def counts_row(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = w[:, -1]
        row = model.counts[t]
        mx, am = row.max(1)
        total = row.sum(1)
        pred = torch.where(mx >= 1, cls[am], torch.full_like(t, -1))
        conf = mx.float() / (total + v5.ALPHA)
        calibrated = getattr(model, "counts_calib", None)
        if calibrated is not None:
            cc = calibrated[t]
            conf = torch.where(cc >= 0, cc, conf)
        return pred, conf

    return counts_row


def _decile_rows(conf: torch.Tensor, correct: torch.Tensor) -> list[dict[str, Any]]:
    order = torch.argsort(conf, stable=True)
    chunks = torch.tensor_split(order, 10)
    rows = []
    for i, idx in enumerate(chunks, 1):
        rows.append(
            {
                "decile": i,
                "n": len(idx),
                "conf_min": float(conf[idx].min()),
                "conf_max": float(conf[idx].max()),
                "agree": float(correct[idx].float().mean()),
            }
        )
    return rows


def _group_stats(values: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    vals = values[mask].cpu()
    hist = torch.bincount(vals, minlength=9)
    return {
        "n": len(vals),
        "mean": float(vals.float().mean()),
        "median": float(vals.float().median()),
        "hist_0_to_8": hist.tolist(),
    }


def _load_teacher_overlay(raw_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    teachers = []
    for tag, _ in MODELS:
        path = REPO / "data" / f"wyly_teacher_{tag}_L256.pt"
        teachers.append(torch.load(path, map_location="cpu")["teacher"][raw_positions.cpu()])
    t = torch.stack(teachers, 1)
    agree70 = (t == t[:, 1:2]).sum(1)
    cluster = torch.tensor(
        [max(Counter(row).values()) for row in t.tolist()], dtype=torch.long
    )
    return agree70, cluster


def _tertiary_metrics(ids, y, cls, uv, tr, te, correct):
    # y contains class indices; translate through class_to_ids before token identity comparisons.
    class_to_ids = uv[cls.cpu()].cpu()
    y_class = torch.searchsorted(cls, cls[y]).cpu()
    teacher_ids = class_to_ids[y_class]

    ids_cpu = ids.cpu()
    w = ids_cpu[te.cpu()]
    target = teacher_ids[te.cpu()]
    copy_hit = ((w[:, :-1] == w[:, -1:]) & (w[:, 1:] == target[:, None])).any(1)
    # Import/use the canonical implementation too; equality guards against accidental drift.
    copy_rate = copy_incidence(w, target)
    assert int(copy_hit.sum()) == round(copy_rate * len(w))

    votes: dict[int, Counter] = defaultdict(Counter)
    # Match wyly_ladder: de-overlap stride-1 windows by sampling every twelfth train row.
    for row in ids_cpu[tr.cpu()][::12].tolist():
        for left, right in zip(row, row[1:], strict=False):
            votes[left][right] += 1
    bigmap = {left: counts.most_common(1)[0][0] for left, counts in votes.items()}
    big_hit = torch.tensor(
        [
            bigmap.get(last, -1) == target_id
            for last, target_id in zip(w[:, -1].tolist(), target.tolist(), strict=True)
        ]
    )

    def split_metric(hit):
        return {
            "correct": float(hit[correct.cpu()].float().mean()),
            "error": float(hit[~correct.cpu()].float().mean()),
        }

    return {
        "copy_incidence": split_metric(copy_hit),
        "train_region_bigram_map_agreement": split_metric(big_hit),
        "token_space": "class-space values translated through uv[cls] before identity comparisons",
    }


def _fit_table_candidate(name, ids, yv, tr, feature, vocab):
    table, nkeys = v5.fit_dgate_feature(ids, yv, tr, feature, vocab)
    base = vocab + 2

    def conf(w):
        feat = feature(w)
        value, confidence = table.lookup_conf((feat + 1) * base + w[:, -1])
        miss = feat < 0
        return (
            torch.where(miss, torch.full_like(value, -1), value),
            torch.where(miss, torch.full_like(confidence, -1e9), confidence),
        )

    return name, (lambda w: conf(w)[0]), conf, nkeys


def _member_sets(uv, vocab):
    codec = v5.load_codec()
    sent = torch.zeros(vocab, dtype=torch.bool, device=v5.DEV)
    quote = torch.zeros_like(sent)
    attr = torch.zeros_like(sent)
    conn = torch.zeros_like(sent)
    attrs = {
        "argues", "claims", "says", "suggests", "believes", "maintains", "asserts",
        "contends", "holds", "denies", "observes", "notes", "concludes", "insists",
        "argued", "claimed", "said", "wrote",
    }
    conns = {
        "however", "therefore", "thus", "hence", "moreover", "furthermore", "consequently",
        "nevertheless", "accordingly", "conversely", "indeed", "namely", "instead",
        "otherwise", "meanwhile", "aber", "jedoch", "also", "daher", "trotzdem",
        "deshalb", "dennoch",
    }
    for i in range(vocab):
        token = codec.token_str(int(uv[i])).strip()
        low = token.lower()
        sent[i] = token in {".", "!", "?"}
        quote[i] = token in {'"', "``", "''"} or "“" in token or "”" in token
        attr[i] = low in attrs
        conn[i] = low in conns
    return sent, quote, attr, conn


def fit_candidate_pool(
    model, ids, yv, cls, uv, tr, tr_conf, tau, val_te, test_te, active_families
):
    """Fit the active canonical subset on train; mined frames use train low-conf only."""
    vocab = len(uv)
    candidates = []

    if "pointer" in active_families:
        pointer = v5.PointerRule(vocab, v5.DEV)
        pointer.refresh(ids, yv, tr)
        candidates.append(("pointer", pointer.mirror(), pointer.lookup_conf))

    sent, quote, attr, _conn = _member_sets(uv, vocab)
    if "dstate gate" in active_families:
        def dstate(w):
            return v5.dstate_feature(w, sent, quote)

        nm, fn, cf, _ = _fit_table_candidate(
            "dstate gate", ids, yv, tr, dstate, vocab
        )
        candidates.append((nm, fn, cf))

    if "sentpair/dgate2" in active_families:
        def senthead_pos(w):
            return v5.recent_member_pos(w, sent)

        def previous_head(w):
            return v5.at_pos(w, v5.prev_occ_pos(w, senthead_pos(w)), succ=1)

        def current_head(w):
            return v5.at_pos(w, senthead_pos(w), succ=1)

        pair_table, _ = v5.fit_dgate2(
            ids, yv, tr, previous_head, current_head, vocab
        )
        base = vocab + 2

        def pair_conf(w):
            fa, fb = previous_head(w), current_head(w)
            valid = (fa >= 0) & (fb >= 0)
            value, confidence = pair_table.lookup_conf((fa + 1) * base + fb)
            return (
                torch.where(valid, value, torch.full_like(value, -1)),
                torch.where(valid, confidence, torch.full_like(confidence, -1e9)),
            )

        candidates.append(("sentpair/dgate2", lambda w: pair_conf(w)[0], pair_conf))

    if "attrib-subj gate" in active_families:
        def attrib(w):
            return v5.at_pos(w, v5.recent_member_pos(w, attr), succ=-1)

        nm, fn, cf, _ = _fit_table_candidate(
            "attrib-subj gate", ids, yv, tr, attrib, vocab
        )
        candidates.append((nm, fn, cf))

    low_train = mining_fit_slice(tr, tr_conf, tau, val_te, test_te)
    mining_log = []
    if "mined frames" in active_families:
        mined = v5.MinedGates(vocab, v5.DEV)
        if len(low_train):
            mined.mine(
                model,
                ids,
                yv,
                cls,
                low_train,
                torch.Generator().manual_seed(0),
                mining_log,
                0,
                sample_n=min(6000, len(low_train)),
            )
        candidates.append(("mined frames", mined.mirror(), mined.lookup_conf_all))
    assert tuple(name for name, _, _ in candidates) == tuple(active_families)
    return candidates, low_train, mining_log


def _score_rules(rules, conf_fns, counts_fn, ids, yv, idxs):
    block = WylyBlock(0, "score")
    block.rules = list(rules)
    block.conf_fns = conf_fns
    pred, _ = block.predict_cover(ids[idxs], counts_row_fn=counts_fn)
    return float((pred == yv[idxs]).float().mean())


def _admission_rows(last_admit):
    return [
        {"name": row["name"], "marginal": row["marginal"], "score": row["score"]}
        for row in last_admit
    ]


def _run_flat(model, b0_rules, b0_conf, candidates, counts_fn, ids, yv, val_te):
    block = WylyBlock(0, "B0-flat", max_rules=MAX_RULES)
    block.rules = list(b0_rules)
    block.conf_fns = dict(b0_conf)
    block.candidates = [(name, fn) for name, fn, _ in candidates]
    block.conf_fns.update({name: cf for name, _, cf in candidates})
    def score(rules):
        return _score_rules(rules, block.conf_fns, counts_fn, ids, yv, val_te)

    admitted = block.admit_greedy(score, thresh=ADMIT_THRESH, max_rules=MAX_RULES)
    return block, admitted, _admission_rows(block.last_admit)


def _gated_predict(stack, ids_subset, counts_fn):
    stack.forward(ids_subset, counts_row_fn=counts_fn)
    state = stack.last_carried[-1]
    return state.pred, state.conf


def _run_gated_tau(
    b0_rules, b0_conf, candidates, counts_fn, ids, yv, val_te, tau
):
    b0 = WylyBlock(0, "B0", max_rules=MAX_RULES)
    b0.rules = list(b0_rules)
    b0.conf_fns = dict(b0_conf)
    b1 = WylyBlock(1, "B1-rescue", depends_on=[0], max_rules=MAX_RULES)
    b1.candidates = [(name, fn) for name, fn, _ in candidates]
    b1.conf_fns = {name: cf for name, _, cf in candidates}
    stack = BlockStack([b0, b1], carry="gated", gate_conf=tau)

    def score(combined):
        saved = b1.rules
        b1.rules = list(combined[len(b0.rules):])
        try:
            pred, _ = _gated_predict(stack, ids[val_te], counts_fn)
            return float((pred == yv[val_te]).float().mean())
        finally:
            b1.rules = saved

    admitted = stack.admit_layer(1, score, thresh=ADMIT_THRESH, max_rules=MAX_RULES)
    pred, _ = _gated_predict(stack, ids[val_te], counts_fn)
    agree = float((pred == yv[val_te]).float().mean())
    return stack, admitted, _admission_rows(b1.last_admit), agree


def _raw_alignment():
    raw_teacher = torch.load(v5.TEACH, map_location="cpu")["teacher"]
    keep = torch.isin(raw_teacher, torch.bincount(raw_teacher).argsort(descending=True)[: v5.V])
    return keep.nonzero(as_tuple=False).flatten()


def _print_table(title, rows):
    log(title)
    for row in rows:
        log("  " + json.dumps(row, sort_keys=True))


def main():
    report: dict[str, Any] = {
        "honesty_note": HONESTY_NOTE,
        "b0_path": "Path 2",
        "gaps": [
            (
                "The 1c sincedot example conflicts with the general exclusion rule and the "
                "canonical five-family pool; the general rule is implemented, and sincedot "
                "remains a possible B0 constituent rather than a pool candidate."
            ),
            (
                "The specified loaded-model stand-in omitted model.rules, but MinedGates.mine "
                "reads it to score residuals; the reconstructed ordered B0′ rules are attached "
                "to preserve the live model's behavior."
            ),
        ],
    }
    expected_full = {
        "induction L=2", "cmember [11]", "prevsent-head gate [656]",
        "kgram k=2 (online s40)", "induction L=3", "sincedot gate [760]",
        "cap-echo gate [278]", "mined frames (online)",
    }
    original = v5.core_cover_sw

    log("=== B0 RECONSTRUCTION ===")
    regenerated = not B0_FREEZE.exists()
    if regenerated:
        captured = {}

        def capture(model, rules, ids, yv, cls, idxs, **kwargs):
            captured["model"] = model
            captured["rules"] = list(rules)
            return original(model, rules, ids, yv, cls, idxs, **kwargs)

        start = time.perf_counter()
        v5.core_cover_sw = capture
        try:
            v5.main()
        finally:
            v5.core_cover_sw = original
        wall = time.perf_counter() - start
        if not captured:
            raise RuntimeError("capture wrapper was never called")
        model, rules = captured["model"], captured["rules"]
        frozen_core = None
    else:
        model, rules, frozen_core, wall = load_frozen_b0(B0_FREEZE)
        log("B0 SOURCE: loaded frozen B0′")

    ids, y, cls, uv, tr, te = v5.load_ds()
    yv = cls[y]
    core, original_pred = original(model, rules, ids, yv, cls, te, return_pred=True)
    if frozen_core is not None:
        drift = {
            key: core[key] - frozen_core[key]
            for key in ("agree", "cover", "agree_fired")
        }
        report["b0_reload_drift"] = drift
        log(
            "B0 reload drift: "
            + " ".join(f"{key}={drift[key]:+.2e}" for key in drift)
            + f" (tolerance {B0_RELOAD_TOLERANCE:g})"
        )
        if any(abs(value) > B0_RELOAD_TOLERANCE for value in drift.values()):
            raise AssertionError("loaded B0 score differs from its frozen core_sw record")
    actual_names = [name for name, _ in rules]
    actual_set = set(actual_names)
    symmetric_diff = sorted(actual_set ^ expected_full)
    report["b0_wall_seconds"] = wall
    report["b0_rules"] = actual_names
    report["b0_core_sw"] = core
    report["cross_run_note"] = {
        "finding": CROSS_RUN_FINDING,
        "b0_prime_rules": actual_names,
        "archived_rules": sorted(expected_full),
        "symmetric_diff": symmetric_diff,
    }
    log(f"B0′ composition: {actual_names}")
    log(
        f"registered B0 core_sw agree: {core['agree']:.9f} "
        f"(required [{B0_SCORE_MIN:.3f}, {B0_SCORE_MAX:.3f}], inclusive)"
    )
    log(f"registered score-band check: {b0_within_registered_band(core['agree'])}")
    log(f"FINDING: {CROSS_RUN_FINDING}")
    log(f"  B0′ actual names: {sorted(actual_set)}")
    log(f"  archived names: {sorted(expected_full)}")
    log(f"  symmetric difference: {symmetric_diff}")
    if not b0_within_registered_band(core["agree"]):
        report["stopped"] = "registered B0 core_sw agreement gate failed"
        (REPO / "data" / "wikitext_gated_rescue.json").write_text(json.dumps(report, indent=2))
        log("STOP: registered B0 core_sw agreement gate failed")
        return 2
    if regenerated:
        freeze_b0(B0_FREEZE, model, rules, core, wall)
        log(
            f"B0 SOURCE: regenerated fresh (wall {wall:.3f}s), "
            f"froze to {B0_FREEZE}"
        )
    report["b0_source"] = "regenerated fresh + froze" if regenerated else "loaded frozen B0′"

    traced, trace_pred, trace = core_cover_sw_traced(model, rules, ids, yv, cls, te)
    if not torch.equal(original_pred, trace_pred):
        raise AssertionError("full-te traced prediction parity failed")
    log("traced-vs-original parity: exact on full te")

    log("=== PART A: DESCRIPTIVE RESIDUAL ANATOMY (gates nothing) ===")
    per_rule = []
    winner_names = {name for name, _ in rules} | {"counts"}
    if "abstain" in trace["name"]:
        winner_names.add("abstain")
    for name in sorted(winner_names):
        mask = torch.tensor([row == name for row in trace["name"]], device=v5.DEV)
        error_rate = float((~trace["correct"][mask]).float().mean()) if bool(mask.any()) else None
        per_rule.append(
            {
                "name": name,
                "rows": int(mask.sum()),
                "error_rate": error_rate,
            }
        )
    calibration = _decile_rows(trace["conf"], trace["correct"])
    raw_positions = _raw_alignment()[te.cpu()]
    agree70, cluster = _load_teacher_overlay(raw_positions)
    correct_cpu = trace["correct"].cpu()
    overlay = {
        "metric_name": "inter-teacher agreement proxy",
        "pythia70_agreement_count": {
            "correct": _group_stats(agree70, correct_cpu),
            "error": _group_stats(agree70, ~correct_cpu),
        },
        "max_agreement_cluster_size": {
            "correct": _group_stats(cluster, correct_cpu),
            "error": _group_stats(cluster, ~correct_cpu),
        },
        "alignment": (
            "all eight raw teacher tensors indexed by pythia70m top-V keep-mask raw row positions, "
            "then by filtered te; no row mismatches dropped"
        ),
    }
    tertiary = _tertiary_metrics(ids, y, cls, uv, tr, te, trace["correct"])
    report["part_a"] = {
        "core": traced, "per_winner": per_rule, "calibration_deciles": calibration,
        "teacher_entropy_overlay": overlay, "tertiary": tertiary,
    }
    _print_table("per-winning-rule rows/error:", per_rule)
    _print_table("winning-confidence calibration deciles:", calibration)
    log("teacher-entropy overlay (inter-teacher agreement proxy):")
    log(json.dumps(overlay, indent=2, sort_keys=True))
    log("tertiary copy/bigram metrics:")
    log(json.dumps(tertiary, indent=2, sort_keys=True))

    log("=== PART B: REGISTERED FLAT-vs-GATED CONTRAST ===")
    val_te, test_te = split_te(te)
    log(f"split sizes: val_te={len(val_te)} test_te={len(test_te)} (seed=0, 25/75)")
    active_families, pool_exclusions = active_candidate_families(actual_names)
    report["pool_exclusions"] = pool_exclusions
    for exclusion in pool_exclusions:
        log(
            f"POOL EXCLUSION: {exclusion['family']}: {exclusion['reason']}"
        )
    if not pool_exclusions:
        log("POOL EXCLUSIONS: none")
    log(f"canonical candidate pool: {list(CANDIDATE_FAMILIES)}")
    log(f"active candidate pool: {list(active_families)}")
    admitted_families = {
        family for name in actual_names if (family := _candidate_family_for_rule(name))
    }
    assert not (set(active_families) & admitted_families)
    counts_fn = _counts_row_fn(model, cls)
    b0 = WylyBlock(0, "B0")
    b0.rules = list(rules)
    b0.conf_fns = {name: v5.CONF_FNS[name] for name, _ in rules if name in v5.CONF_FNS}
    for name, _ in rules:
        if name not in b0.conf_fns:
            scalar = float(v5.RULE_CONF.get(name, 0.0))
            fn = dict(rules)[name]
            b0.conf_fns[name] = lambda w, fn=fn, scalar=scalar: (
                fn(w), torch.full((len(w),), scalar, device=w.device)
            )
    b0_pred, b0_conf_te = b0.predict_cover(ids[te], counts_row_fn=counts_fn)
    block_agree = float((b0_pred == yv[te]).float().mean())
    tolerance = 1e-8
    if abs(block_agree - core["agree"]) > tolerance:
        raise AssertionError("WylyBlock B0 parity failed")
    log(f"WylyBlock B0 agree: {block_agree:.9f}; tolerance={tolerance:g}; direct-assigned finalized rules")

    _, _, tr_trace = core_cover_sw_traced(model, rules, ids, yv, cls, tr)
    val_pos = torch.searchsorted(te, val_te)
    val_conf = trace["conf"][val_pos]
    taus = tau_deciles(val_conf)
    log(f"tau grid (val_te confidence deciles 10..100%): {taus}")

    tau_runs = []
    fitted_by_tau = []
    for tau in taus:
        candidates, low_train, mining_log = fit_candidate_pool(
            model, ids, yv, cls, uv, tr, tr_trace["conf"], tau, val_te, test_te,
            active_families,
        )
        stack, admitted, marginals, agree = _run_gated_tau(
            rules, b0.conf_fns, candidates, counts_fn, ids, yv, val_te, tau
        )
        tau_runs.append(
            {"tau": tau, "val_agree": agree, "admitted": admitted, "marginals": marginals,
             "mined_train_rows": len(low_train), "mining_log": mining_log}
        )
        fitted_by_tau.append((candidates, stack))
        log(f"tau={tau:.9f} val_agree={agree:.9f} admitted={admitted} mined_train={len(low_train)}")
    best_index = min(range(10), key=lambda i: (-tau_runs[i]["val_agree"], tau_runs[i]["tau"]))
    chosen = tau_runs[best_index]
    candidates, gated_stack = fitted_by_tau[best_index]
    log(f"chosen tau: {chosen['tau']:.9f} (best val agree; ties -> smallest tau)")

    # The mined family depends on tau. Fit/sweep GATED on val, then offer that exact selected pool
    # to FLAT so both registered arms use identical fitted candidate objects in the same run.
    flat, flat_admitted, flat_marginals = _run_flat(
        model, rules, b0.conf_fns, candidates, counts_fn, ids, yv, val_te
    )
    log(f"FLAT admitted: {flat_admitted}")
    _print_table("FLAT admission attempts/marginals:", flat_marginals)
    log(f"GATED admitted at chosen tau: {chosen['admitted']}")
    _print_table("GATED admission attempts/marginals:", chosen["marginals"])

    flat_pred, _ = flat.predict_cover(ids[test_te], counts_row_fn=counts_fn)
    gated_pred, _ = _gated_predict(gated_stack, ids[test_te], counts_fn)
    test_y = yv[test_te]
    base_pred, _ = b0.predict_cover(ids[test_te], counts_row_fn=counts_fn)
    flat_correct, gated_correct = flat_pred == test_y, gated_pred == test_y
    base_correct = base_pred == test_y
    flat_agree = float(flat_correct.float().mean())
    gated_agree = float(gated_correct.float().mean())
    flat_reg = int((base_correct & ~flat_correct).sum())
    gated_reg = int((base_correct & ~gated_correct).sum())
    b = int((flat_correct & ~gated_correct).sum())
    c = int((gated_correct & ~flat_correct).sum())
    pvalue = exact_discordant_p(b, c)
    flat_set, gated_set = set(flat_admitted), set(chosen["admitted"])
    clauses = {
        "i_gated_admits_flat_decline": bool(gated_set - flat_set),
        "ii_agree_plus_0.005": gated_agree >= flat_agree + 0.005,
        "iii_exact_p_lt_0.05": pvalue < 0.05,
        "iv_regressions_no_more": gated_reg <= flat_reg,
    }
    rescues = all(clauses.values())
    neither = not flat_admitted and not chosen["admitted"]
    verdict = (
        "wikitext residual beyond the current pool under both routings"
        if neither
        else ("gated rescues" if rescues else "gated does not rescue")
    )
    metrics = {
        "flat_agree": flat_agree, "gated_agree": gated_agree,
        "flat_collateral_regressions": flat_reg, "gated_collateral_regressions": gated_reg,
        "discordant_convention": "b=flat-correct/gated-wrong; c=gated-correct/flat-wrong",
        "b": b, "c": c, "two_sided_exact_binomial_p": pvalue,
    }
    log("test_te contrast: " + json.dumps(metrics, sort_keys=True))
    log("REGISTERED RULE (verbatim): " + VERDICT_RULE)
    for name, value in clauses.items():
        log(f"  {name}: {value}")
    log(f"overall AND: {rescues}")
    log(f"FINAL VERDICT: {verdict}")
    log("honesty note: " + HONESTY_NOTE)

    report["part_b"] = {
        "split": {"seed": 0, "val_te": len(val_te), "test_te": len(test_te)},
        "b0_block_agree": block_agree, "b0_parity_tolerance": tolerance,
        "candidate_families": list(CANDIDATE_FAMILIES),
        "active_candidate_families": list(active_families),
        "pool_exclusions": pool_exclusions,
        "admission": {"thresh": ADMIT_THRESH, "max_rules": MAX_RULES},
        "tau_grid": taus, "tau_runs": tau_runs, "chosen_tau": chosen["tau"],
        "flat": {"admitted": flat_admitted, "marginals": flat_marginals},
        "gated": {"admitted": chosen["admitted"], "marginals": chosen["marginals"]},
        "test_metrics": metrics, "registered_rule": VERDICT_RULE,
        "clauses": clauses, "overall_and": rescues, "verdict": verdict,
    }
    output = REPO / "data" / "wikitext_gated_rescue.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    log(f"JSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
