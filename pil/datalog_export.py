"""Export a hardened RuleProgram as a runnable Soufflé Datalog program (rosetta house style).

The exported program is the *same* semantics as ``RuleProgram.forward(x, hard=True)``:

  EDB      tok(inst, pos, id)       -- position ``pos`` in 0..W-1 holds token ``id``
  strata   fired<s>(inst, k)        -- rule ``k`` of stratum ``s`` fired on ``inst``
           AND rules: one clause  fired<s>(I,k) :- inst(I), tok(I,o,c), !tok(I,o',c'), ...
           THRESHOLD rules (PIC-T3): per-literal ``sat<s>(I,k,o)`` clauses, then
           ``N = count : { sat<s>(I,k,_) }, N >= θ_k`` -- the ⊗ = + coalition bracket
           behind a margin turnstile, as a Datalog aggregate.
  decode   contrib(I,V,W) :- fired<s>(I,K), headw<s>(K,V,W)   -- the incidences <a_k, U_v>
           logit = bias + sum (⊗ over sources), decide = argmax (⊕ at T -> 0)

Deep-strata negation is stratified by construction (a stratum only reads earlier strata),
so Soufflé accepts the program; ties at the max break toward the smallest candidate index,
matching the argmax convention of the tensor path.

``run_souffle`` executes the export on a batch of contexts and returns the decoded
candidate per instance, for exact agreement checks against ``hard_forward``.
"""

from __future__ import annotations

import csv
import subprocess
import tempfile
from pathlib import Path

import torch
from torch import Tensor

from .rules import RuleProgram, Stratum1


def _fmt(x: float) -> str:
    """Soufflé float literal: fixed-point only (no exponent notation)."""
    if x != x:  # NaN guard
        return "0"
    s = f"{x:.9f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def export_program(
    prog: RuleProgram,
    name: str = "pil_program",
    lookup_domain: Tensor | None = None,
    lookup_topk: int | None = None,
) -> str:
    """Render the hardened program as a self-contained Soufflé .dl string.

    Lookup families (the ngram-table backbone) are materialized as fact tables over the
    tokens observed in ``lookup_domain`` (a (N, W) context batch; required if the program
    has lookup offsets). ``lookup_topk`` truncates each table row to its top-K incidences
    (rosetta-style); ``None`` emits every nonzero incidence (exact).
    """
    cfg = prog.cfg
    if (prog.lookup or prog.direct_lookup) and lookup_domain is None:
        raise ValueError("program has lookup families; pass lookup_domain to materialize them")
    V = prog.candidate_ids.shape[0]
    lines: list[str] = [
        f"// {name} -- learned PIC-LP program exported by pil.datalog_export",
        "// EDB: tok(inst,pos,id) with pos in 0..W-1; decode predicts the next token.",
        f"// window={cfg.window} candidates={V} rules={prog.n_rules} strata={len(prog.strata)}",
        ".decl tok(inst:number, pos:number, id:number)",
        ".input tok",
        ".decl inst(i:number)",
        "inst(I) :- tok(I,_,_).",
        ".decl cand(v:number)",
    ]
    for vi in range(V):
        lines.append(f"cand({vi}).")

    # map stable rule id -> (stratum, local index) for deep-literal references
    locate: dict[int, tuple[int, int]] = {}
    for si, ids in enumerate(prog.rule_ids):
        for k, rid in enumerate(ids):
            locate[rid] = (si, k)

    incid = prog.head_incidences()          # (K_total, V) rows in stratum concat order
    row = 0
    for si, stratum in enumerate(prog.strata):
        K = stratum.n_rules
        lines.append("")
        lines.append(f".decl fired{si}(inst:number, k:number)")
        thresh_rules: list[int] = []
        for k in range(K):
            rel = stratum.rel.data[k] > 0
            sgn = stratum.sgn.data[k] > 0
            if bool(stratum.is_thresh[k]):
                thresh_rules.append(k)
                continue
            body: list[str] = ["inst(I)"]
            if isinstance(stratum, Stratum1):
                W = cfg.window
                fresh = 0
                for o in torch.nonzero(rel).flatten().tolist():
                    if o < W:
                        c = int(stratum.anchor[k, o])
                        if bool(stratum.is_le[k, o]):   # ordinal literal (binned encodings)
                            op = "<=" if bool(sgn[o]) else ">"
                            body.append(f"tok(I,{o},Le{fresh}), Le{fresh} {op} {c}")
                            fresh += 1
                        else:
                            body.append(
                                f"tok(I,{o},{c})" if bool(sgn[o]) else f"!tok(I,{o},{c})"
                            )
                    else:
                        a, b = (int(v) for v in stratum.eq_pairs[o - W])
                        if bool(sgn[o]):   # join: same token at both positions
                            body.append(f"tok(I,{a},Eq{fresh}), tok(I,{b},Eq{fresh})")
                        else:              # anti-join: different tokens
                            body.append(
                                f"tok(I,{a},Ea{fresh}), tok(I,{b},Eb{fresh}), Ea{fresh} != Eb{fresh}"
                            )
                        fresh += 1
            else:
                for j in torch.nonzero(rel).flatten().tolist():
                    ssi, sk = locate[int(stratum.in_ids[j])]
                    body.append(f"fired{ssi}(I,{sk})" if bool(sgn[j]) else f"!fired{ssi}(I,{sk})")
            lines.append(f"fired{si}(I,{k}) :- {', '.join(body)}.")

        if thresh_rules:
            lines.append(f".decl sat{si}(inst:number, k:number, lit:number)")
            lines.append(f".decl thr{si}(k:number, t:number)")
            for k in thresh_rules:
                rel = stratum.rel.data[k] > 0
                sgn = stratum.sgn.data[k] > 0
                t_hard = int(torch.ceil(stratum.thr.data[k] - 1e-6).item())
                lines.append(f"thr{si}({k},{t_hard}).")
                if isinstance(stratum, Stratum1):
                    W = cfg.window
                    for o in torch.nonzero(rel).flatten().tolist():
                        if o < W:
                            c = int(stratum.anchor[k, o])
                            if bool(stratum.is_le[k, o]):
                                op = "<=" if bool(sgn[o]) else ">"
                                lines.append(
                                    f"sat{si}(I,{k},{o}) :- tok(I,{o},C), C {op} {c}."
                                )
                            elif bool(sgn[o]):
                                lines.append(f"sat{si}(I,{k},{o}) :- tok(I,{o},{c}).")
                            else:
                                lines.append(f"sat{si}(I,{k},{o}) :- inst(I), !tok(I,{o},{c}).")
                        else:
                            a, b = (int(v) for v in stratum.eq_pairs[o - W])
                            if bool(sgn[o]):
                                lines.append(f"sat{si}(I,{k},{o}) :- tok(I,{a},C), tok(I,{b},C).")
                            else:
                                lines.append(
                                    f"sat{si}(I,{k},{o}) :- tok(I,{a},Ca), tok(I,{b},Cb), Ca != Cb."
                                )
                else:
                    for j in torch.nonzero(rel).flatten().tolist():
                        ssi, sk = locate[int(stratum.in_ids[j])]
                        if bool(sgn[j]):
                            lines.append(f"sat{si}(I,{k},{j}) :- fired{ssi}(I,{sk}).")
                        else:
                            lines.append(f"sat{si}(I,{k},{j}) :- inst(I), !fired{ssi}(I,{sk}).")
            lines.append(
                f"fired{si}(I,K) :- inst(I), thr{si}(K,T), N = count : {{ sat{si}(I,K,_) }}, N >= T."
            )

        # head incidence tables (the clause weights <a_k, U_v>)
        lines.append(f".decl headw{si}(k:number, v:number, w:float)")
        for k in range(K):
            for vi in range(V):
                w = float(incid[row + k, vi])
                if w != 0.0:
                    lines.append(f"headw{si}({k},{vi},{_fmt(w)}).")
        row += K

    # contrib carries a source discriminator (rosetta's `block` column): identical weights
    # from different sources must not collapse under Datalog set semantics.
    lines += [
        "",
        ".decl contrib(inst:number, src:number, v:number, w:float)",
        "contrib(I,-1,V,0.0) :- inst(I), cand(V).   // ⊗-identity pad: sum is always grounded",
    ]
    for si in range(len(prog.strata)):
        base = (si + 1) * 10_000_000
        lines.append(f"contrib(I,{base}+K,V,W) :- fired{si}(I,K), headw{si}(K,V,W).")

    # lookup source families: per-offset incidence tables (the retrieved/ngram backbone)
    for o, emb in prog.lookup.items():
        o = int(o)
        toks = lookup_domain[:, o].unique()
        with torch.no_grad():
            rows = emb(toks) @ prog.U.T                      # (T, V)
        lines.append(f".decl lkp{o}(c:number, v:number, w:float)")
        lines.append(f"contrib(I,{900_000_000 + o},V,W) :- tok(I,{o},C), lkp{o}(C,V,W).")
        for ti, c in enumerate(toks.tolist()):
            r = rows[ti]
            if lookup_topk is not None and r.abs().gt(0).sum() > lookup_topk:
                keep = r.abs().topk(lookup_topk).indices
            else:
                keep = torch.nonzero(r).flatten()
            for vi in keep.tolist():
                w = float(r[vi])
                if w != 0.0:
                    lines.append(f"lkp{o}({c},{vi},{_fmt(w)}).")
    # direct lookup tables (gram2d shape: token -> candidate logits, already in cand space)
    for o, emb in prog.direct_lookup.items():
        o = int(o)
        toks = lookup_domain[:, o].unique()
        lines.append(f".decl lkpd{o}(c:number, v:number, w:float)")
        lines.append(f"contrib(I,{910_000_000 + o},V,W) :- tok(I,{o},C), lkpd{o}(C,V,W).")
        with torch.no_grad():
            rows = emb.weight[toks]
        for ti, c in enumerate(toks.tolist()):
            r = rows[ti]
            if lookup_topk is not None and r.abs().gt(0).sum() > lookup_topk:
                keep = r.abs().topk(lookup_topk).indices
            else:
                keep = torch.nonzero(r).flatten()
            for vi in keep.tolist():
                w = float(r[vi])
                if w != 0.0:
                    lines.append(f"lkpd{o}({c},{vi},{_fmt(w)}).")

    # schema rules: background-knowledge clauses over num(token, value) facts
    if prog.schema_bank is not None and len(prog.schema_bank):
        bank = prog.schema_bank
        lines.append(".decl candtok(v:number, c:number)")
        for vi, c in enumerate(prog.candidate_ids.tolist()):
            lines.append(f"candtok({vi},{c}).")
        if bool((bank.values >= 0).any()):
            lines.append(".decl num(c:number, n:number)")
            for c in torch.nonzero(bank.values >= 0).flatten().tolist():
                lines.append(f"num({c},{int(bank.values[c])}).")
        for k_s, schema in enumerate(bank.schemas):
            w = float(bank.w.data[k_s])
            if w != 0.0:
                lines.append(f"// schema {schema.name}")
                lines.append(
                    f"contrib(I,{950_000_000 + k_s},V,{_fmt(w)}) :- "
                    f"{schema.datalog}, candtok(V,C)."
                )

    lines += [
        ".decl biasf(v:number, b:float)",
    ]
    for vi in range(V):
        lines.append(f"biasf({vi},{_fmt(float(prog.bias.data[vi]))}).")
    lines += [
        ".decl logit(inst:number, v:number, s:float)",
        "logit(I,V,B+S) :- inst(I), biasf(V,B), S = sum W : { contrib(I,_,V,W) }. // ⊗ over sources",
        ".decl best(inst:number, v:number)",
        "best(I,V) :- logit(I,V,S), S = max S2 : { logit(I,_,S2) }.               // ⊕ at T -> 0",
        ".decl decide(inst:number, v:number)",
        "decide(I,V) :- best(I,V), V = min V2 : { best(I,V2) }.                   // deterministic tie-break",
        ".output decide",
    ]
    return "\n".join(lines) + "\n"


def write_tok_facts(X: Tensor, path: Path) -> None:
    """``X`` (N, W) token ids -> tok.facts (inst TAB pos TAB id)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for i in range(X.shape[0]):
            for o in range(X.shape[1]):
                w.writerow([i, o, int(X[i, o])])


def run_souffle(
    prog: RuleProgram, X: Tensor, dl_text: str | None = None, souffle: str = "souffle"
) -> Tensor:
    """Run the exported program on contexts ``X``; returns decided candidate index per instance.

    Raises ``FileNotFoundError`` if Soufflé is not installed; callers (tests) may skip then.
    """
    dl_text = dl_text or export_program(
        prog, lookup_domain=X if (prog.lookup or prog.direct_lookup) else None
    )
    with tempfile.TemporaryDirectory(prefix="pil_dl_") as td:
        tdp = Path(td)
        (tdp / "program.dl").write_text(dl_text)
        write_tok_facts(X, tdp / "tok.facts")
        subprocess.run(
            [souffle, "program.dl", "-F", ".", "-D", "."],
            cwd=tdp, check=True, capture_output=True, text=True,
        )
        out = torch.full((X.shape[0],), -1, dtype=torch.long)
        with open(tdp / "decide.csv") as f:
            for line in f:
                i, v = line.split()
                out[int(i)] = int(v)
    return out


def verify_export(prog: RuleProgram, X: Tensor, souffle: str = "souffle") -> dict[str, float]:
    """Exact-agreement check: Soufflé decide == tensor hard_forward argmax."""
    dec = run_souffle(prog, X, souffle=souffle)
    hard = prog.forward(X, hard=True)["logits"].argmax(dim=-1)
    return {
        "n": X.shape[0],
        "agreement": float((dec == hard).float().mean().item()),
        "undecided": int((dec < 0).sum().item()),
    }
