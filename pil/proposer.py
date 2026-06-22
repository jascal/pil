"""The generative step: rule banks that propose sources from atom activations.

A *rule* is a learned non-linear feature ``h_k(z) = relu(<w_k, z> + b_k)`` of the atom
activations, emitted as a source vector ``h_k(z) * Bdir_k`` that adds into the residual
``r``. This is the lever the hard-synthetic negative result pointed at: when synonyms are
non-linearly coded (XOR), no readout-frame ``U`` can separate them, but a generated rule can
compute the missing feature. A rule bank is exactly a hidden layer; PIL's contribution is
*how* the rules are generated -- in particular, targeting the tropically at-risk (min-margin)
facets rather than spraying random units.

`build_sources` assembles the full source bag (raw atom sources + generated rule sources) for
the existing `ProjectiveIncidenceLearner.forward`, so the readout/margin/loss are all reused.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .geometry import normalize_rows


class RuleBank(nn.Module):
    """A bank of ``n_rules`` ReLU rules over ``n_atoms`` activations, emitting sources in ``R^dim``."""

    def __init__(self, n_atoms: int, dim: int, n_rules: int, device: str = "cpu", seed: int = 0):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        W = 0.5 * torch.randn(max(n_rules, 1), n_atoms, generator=g)
        Bd = normalize_rows(torch.randn(max(n_rules, 1), dim, generator=g))
        self.empty = n_rules == 0
        self.W = nn.Parameter(W.to(device))
        self.b = nn.Parameter(torch.zeros(max(n_rules, 1), device=device))
        self.Bdir = nn.Parameter(Bd.to(device))

    @property
    def n_rules(self) -> int:
        return 0 if self.empty else self.W.shape[0]

    def forward(self, z: Tensor) -> Tensor | None:
        """``z`` (B, K) -> generated sources (B, n_rules, dim), or ``None`` if the bank is empty."""
        if self.empty:
            return None
        h = torch.relu(z @ self.W.T + self.b)        # (B, n_rules)
        return h.unsqueeze(-1) * self.Bdir            # (B, n_rules, dim)


def build_sources(z: Tensor, A: Tensor, bank: RuleBank | None) -> Tensor:
    """Full source bag = raw atom sources ``z_k * A_k`` concatenated with generated rule sources.

    Returns (B, K + n_rules, dim), ready for ``ProjectiveIncidenceLearner.forward``.
    """
    raw = z.unsqueeze(-1) * A.unsqueeze(0)            # (B, K, dim)
    if bank is None or bank.n_rules == 0:
        return raw
    gen = bank(z)                                     # (B, n_rules, dim)
    return torch.cat([raw, gen], dim=1)


def targeted_rulebank(
    hard_cluster_ids: list[int],
    n_atoms: int,
    n_clusters: int,
    dim: int,
    rules_per_cluster: int = 2,
    device: str = "cpu",
    seed: int = 0,
) -> RuleBank:
    """A rule bank whose input weights are *initialised at the at-risk clusters' code-atoms*.

    For each targeted cluster ``c`` it seeds a +/- pair of rules on that cluster's code-atom pair
    ``(p, q)`` -- i.e. ``relu(z_p - z_q)`` and ``relu(z_q - z_p)``, whose sum is ``|z_p - z_q| = XOR``.
    This is the min-margin-facet targeting: the generator looks where the tropical margin is thin,
    not everywhere. Output directions stay random (the frame ``U`` learns to read them).
    """
    n_rules = max(len(hard_cluster_ids) * rules_per_cluster, 1)
    bank = RuleBank(n_atoms, dim, n_rules, device=device, seed=seed)
    with torch.no_grad():
        bank.W.zero_()
        r = 0
        for c in hard_cluster_ids:
            p = n_clusters + 2 * c
            q = p + 1
            for sign in (+1.0, -1.0):
                if r >= n_rules:
                    break
                bank.W[r, p] = sign
                bank.W[r, q] = -sign
                r += 1
    return bank
