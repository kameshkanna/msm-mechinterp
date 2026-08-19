"""Cross-run activation patching harness (ROME/IOI-style).

Captures a donor run's residual stream at one layer and injects it into a
recipient run at the same layer/position, to causally localize which depth
is responsible for flipping behavior toward the donor's agenda — a stronger
claim than the correlational cosine trajectory in
:mod:`msm_mechinterp.analysis.trajectory`.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from msm_mechinterp.config import HookConfig
from msm_mechinterp.hooks import ResidualStreamPatcher, ResidualStreamRecorder, num_decoder_layers

_LAST_TOKEN_POSITION = -1


def patch_from_donor_ids(
    model: nn.Module,
    donor_input_ids: Tensor,
    recipient_input_ids: Tensor,
    layer_idx: int,
    hook_config: HookConfig | None = None,
    token_position: int = _LAST_TOKEN_POSITION,
) -> Tensor:
    """Patch a donor run's layer activation into a recipient forward pass.

    Args:
        model: Model shared by both the donor and recipient forward passes.
        donor_input_ids: Token ids of shape ``[batch, donor_seq_len]`` whose
            activation at ``layer_idx`` supplies the replacement.
        recipient_input_ids: Token ids of shape ``[batch, recipient_seq_len]``
            for the run being patched. Must have the same ``batch`` and
            ``seq_len`` as ``donor_input_ids`` since the whole-sequence
            activation is injected position-for-position.
        layer_idx: Decoder-layer index at which the patch is applied.
        hook_config: Module-path configuration.
        token_position: Sequence index whose next-token logits are returned.

    Returns:
        Recipient next-token logits, of shape ``[batch, vocab_size]``, under
        the patched forward pass.

    Raises:
        ValueError: If donor and recipient shapes disagree.
    """
    if donor_input_ids.shape != recipient_input_ids.shape:
        raise ValueError(
            f"Donor/recipient shape mismatch: {tuple(donor_input_ids.shape)} "
            f"vs {tuple(recipient_input_ids.shape)}; whole-sequence patching requires matched shapes"
        )
    hook_config = hook_config or HookConfig()

    with torch.no_grad(), ResidualStreamRecorder(model, hook_config) as donor_recorder:
        model(input_ids=donor_input_ids)
    donor_activation = donor_recorder.activations[layer_idx]

    with torch.no_grad(), ResidualStreamPatcher(model, layer_idx, donor_activation, hook_config):
        patched_logits = model(input_ids=recipient_input_ids).logits[:, token_position, :]

    return patched_logits


def sweep_layers(
    model: nn.Module,
    donor_input_ids: Tensor,
    recipient_input_ids: Tensor,
    hook_config: HookConfig | None = None,
    token_position: int = _LAST_TOKEN_POSITION,
) -> dict[int, Tensor]:
    """Patch each decoder layer in turn to localize where the donor's effect lands.

    Args:
        model: Model shared by both runs.
        donor_input_ids: See :func:`patch_from_donor_ids`.
        recipient_input_ids: See :func:`patch_from_donor_ids`.
        hook_config: Module-path configuration.
        token_position: Sequence index whose next-token logits are returned.

    Returns:
        Mapping from layer index to the recipient's patched next-token logits
        at that layer, suitable for comparing against an unpatched baseline.
    """
    hook_config = hook_config or HookConfig()
    num_layers = num_decoder_layers(model, hook_config)
    return {
        layer_idx: patch_from_donor_ids(
            model, donor_input_ids, recipient_input_ids, layer_idx, hook_config, token_position
        )
        for layer_idx in range(num_layers)
    }
