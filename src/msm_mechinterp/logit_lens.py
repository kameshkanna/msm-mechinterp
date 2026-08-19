"""Logit-lens projection: read intermediate residual-stream states as vocabulary
distributions by applying the model's own final norm and unembedding early.

This is the cheapest instrument in the flow-analysis stack (Stage 1, method 1):
it gives a first per-layer signal for whether a "contrary" agenda's tokens are
favored at some depth before being suppressed, with no gradient/patching cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from msm_mechinterp.config import HookConfig
from msm_mechinterp.hooks import resolve_attr_path


@dataclass
class LayerTopTokens:
    """Top-k token ids and probabilities decoded from one layer's logit-lens projection."""

    layer_idx: int
    token_ids: Tensor
    probabilities: Tensor


class LogitLens:
    """Projects intermediate hidden states to vocabulary logits via the model's
    own final layer norm and unembedding head, without running the remaining
    decoder layers.
    """

    def __init__(self, model: nn.Module, hook_config: HookConfig | None = None) -> None:
        hook_config = hook_config or HookConfig()
        self._final_norm = resolve_attr_path(model, hook_config.final_norm_attr_path)
        self._lm_head = resolve_attr_path(model, hook_config.lm_head_attr_path)

    def project(self, hidden_states: Tensor) -> Tensor:
        """Project residual-stream hidden states to vocabulary logits.

        Args:
            hidden_states: Tensor of shape ``[..., hidden_size]``.

        Returns:
            Logits of shape ``[..., vocab_size]``.
        """
        with torch.no_grad():
            normalized = self._final_norm(hidden_states)
            return self._lm_head(normalized)

    def trajectory(
        self,
        activations: dict[int, Tensor],
        token_position: int = -1,
        top_k: int = 5,
    ) -> dict[int, LayerTopTokens]:
        """Compute the per-layer top-k vocabulary prediction at one token position.

        Args:
            activations: Per-layer residual-stream states, as captured by
                :class:`~msm_mechinterp.hooks.ResidualStreamRecorder`, each of
                shape ``[batch, seq_len, hidden_size]``.
            token_position: Sequence index to read (default: last token).
            top_k: Number of top tokens to return per layer.

        Returns:
            Mapping from layer index to its top-k token ids/probabilities,
            restricted to batch element 0 for readability.
        """
        result: dict[int, LayerTopTokens] = {}
        for layer_idx, hidden_states in activations.items():
            logits = self.project(hidden_states[:, token_position, :])
            probabilities = torch.softmax(logits, dim=-1)
            top_probs, top_ids = torch.topk(probabilities[0], k=top_k)
            result[layer_idx] = LayerTopTokens(layer_idx=layer_idx, token_ids=top_ids, probabilities=top_probs)
        return result
