"""Direction-ablation comparison harness (Arditi-et-al style).

Tests whether a single per-layer direction is *causally* sufficient to
suppress agenda-consistent behavior: run the model with and without the
direction projected out of every configured layer, and compare next-token
distributions. Core logic is pure/tensor-in-tensor-out for testability; a
thin string-based convenience wrapper is provided for real (tokenizer-bearing)
usage on Lambda Labs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from msm_mechinterp.config import HookConfig
from msm_mechinterp.hooks import DirectionAblator

_LAST_TOKEN_POSITION = -1


@dataclass
class AblationComparison:
    """Next-token distribution comparison, with vs. without direction ablation."""

    baseline_logits: Tensor
    ablated_logits: Tensor

    @property
    def kl_divergence(self) -> float:
        """KL(baseline || ablated) over the next-token distribution, in nats."""
        baseline_log_probs = torch.log_softmax(self.baseline_logits, dim=-1)
        ablated_log_probs = torch.log_softmax(self.ablated_logits, dim=-1)
        baseline_probs = baseline_log_probs.exp()
        divergence = (baseline_probs * (baseline_log_probs - ablated_log_probs)).sum(dim=-1)
        return divergence.mean().item()


def compare_ablation_ids(
    model: nn.Module,
    input_ids: Tensor,
    directions_by_layer: dict[int, Tensor],
    hook_config: HookConfig | None = None,
    token_position: int = _LAST_TOKEN_POSITION,
) -> AblationComparison:
    """Run one forward pass with and without per-layer direction ablation.

    Args:
        model: Model to probe.
        input_ids: Token ids of shape ``[batch, seq_len]``.
        directions_by_layer: Per-layer ablation targets, e.g. from
            :class:`~msm_mechinterp.directions.AgendaVectorExtractor`.
        hook_config: Module-path configuration.
        token_position: Sequence index whose next-token logits are compared.

    Returns:
        An :class:`AblationComparison` holding both logit tensors.
    """
    hook_config = hook_config or HookConfig()
    with torch.no_grad():
        baseline_logits = model(input_ids=input_ids).logits[:, token_position, :]

    with torch.no_grad(), DirectionAblator(model, directions_by_layer, hook_config):
        ablated_logits = model(input_ids=input_ids).logits[:, token_position, :]

    return AblationComparison(baseline_logits=baseline_logits, ablated_logits=ablated_logits)
