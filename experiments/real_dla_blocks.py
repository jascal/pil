"""Which blocks does the decode concentrate on -- a fixed circuit, or a different ~12 each time?

Sharpens the §5i "rank-concentrated to ~12 effective blocks" finding. For a fieldrun --pil-dump,
the decode logit is Σ_b c0[b] over the nb = 2L+1 DLA blocks (c0 = per-block contribution to the
decode token). We ask:
  - GLOBAL concentration: PR over the mean per-block decode contribution -- is the decode driven by
    a small FIXED set of blocks globally, or spread out (each position a different subset)?
  - CONSISTENCY: what fraction of each position's decode mass lands in the global top-k blocks?
  - DEPTH/TYPE profile: index 0 = embed; then (attn,mlp) per layer -> which layers, attn vs mlp.

If global-PR ~ per-position-PR and the top-k capture most mass, the decode is a consistent late-or-
wherever circuit; if global-PR >> per-position-PR, ~12 effective per position but globally diffuse.

Run:  python experiments/real_dla_blocks.py /tmp/d_instruct_prose.jsonl
"""

from __future__ import annotations

import argparse

import torch

from pil.fieldrun_io import load_pil_dump
from pil.geometry import participation_ratio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--topk", type=int, default=12)
    a = p.parse_args()

    b = load_pil_dump(a.path)
    contrib = b.contrib                     # (N, nb, K)
    N, nb, _ = contrib.shape
    c0 = contrib[:, :, 0]                   # (N, nb) per-block contribution to the decode

    per_pos_pr = participation_ratio(c0, dim=1).mean().item()       # effective blocks PER position
    gmag = c0.abs().mean(dim=0)              # (nb,) mean magnitude per block
    global_pr = participation_ratio(gmag, dim=0).item()              # effective blocks GLOBALLY

    topk = torch.topk(gmag, min(a.topk, nb)).indices
    # consistency: fraction of each position's |decode mass| captured by the global top-k blocks
    mass = c0.abs()
    frac_topk = (mass[:, topk].sum(1) / (mass.sum(1) + 1e-8)).mean().item()

    # depth/type profile: block 0 = embed/base; then (attn, mlp) per layer
    L = (nb - 1) // 2
    def layer_of(i):
        return -1 if i == 0 else (i - 1) // 2          # -1 = embedding
    def type_of(i):
        return "emb" if i == 0 else ("attn" if (i - 1) % 2 == 0 else "mlp")

    attn_mass = sum(gmag[i].item() for i in range(nb) if type_of(i) == "attn")
    mlp_mass = sum(gmag[i].item() for i in range(nb) if type_of(i) == "mlp")
    emb_mass = gmag[0].item()
    tot = attn_mass + mlp_mass + emb_mass + 1e-8
    # depth terciles by layer
    layers = [layer_of(i) for i in range(nb)]
    early = sum(gmag[i].item() for i in range(nb) if 0 <= layers[i] < L / 3)
    mid = sum(gmag[i].item() for i in range(nb) if L / 3 <= layers[i] < 2 * L / 3)
    late = sum(gmag[i].item() for i in range(nb) if layers[i] >= 2 * L / 3)
    dtot = early + mid + late + 1e-8

    print(f"[blocks] {b.meta['path']}: N={N}, nb={nb} (L={L} layers)")
    print(f"  effective blocks: per-position PR {per_pos_pr:.1f}  vs  GLOBAL PR {global_pr:.1f} / {nb}")
    verdict = ("CONSISTENT circuit (global ~ per-position)" if global_pr < 1.8 * per_pos_pr
               else "globally DIFFUSE (different subset per position)")
    print(f"    -> {verdict}")
    print(f"  global top-{a.topk} blocks capture {frac_topk*100:.0f}% of per-position decode mass")
    print(f"  top blocks (index): {sorted(topk.tolist())}")
    print(f"  mass by component:  emb {emb_mass/tot*100:.0f}%  "
          f"attn {attn_mass/tot*100:.0f}%  mlp {mlp_mass/tot*100:.0f}%")
    print(f"  mass by depth:      early {early/dtot*100:.0f}%  "
          f"mid {mid/dtot*100:.0f}%  late {late/dtot*100:.0f}%")


if __name__ == "__main__":
    main()
