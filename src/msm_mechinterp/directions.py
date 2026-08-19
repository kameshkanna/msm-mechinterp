"""Diff-of-means extraction of per-layer agenda/persona direction vectors.

Given contrastive prompt pairs (same topic, opposed agenda), extracts one
direction vector per decoder layer: the mean last-token residual-stream
difference between the agenda-A and agenda-B phrasings. This is the same
diff-of-means recipe used for persona/steering vectors elsewhere (see the
ksteer project); here it supplies both the ablation targets for
:class:`~msm_mechinterp.hooks.DirectionAblator` and the reference directions
for :func:`msm_mechinterp.analysis.trajectory.agenda_cosine_trajectory`.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from msm_mechinterp.config import HookConfig
from msm_mechinterp.data.prompts import PromptPair, prompt_pairs_to_texts
from msm_mechinterp.hooks import ResidualStreamRecorder

logger = logging.getLogger(__name__)


class AgendaVectorExtractor:
    """Extracts per-layer diff-of-means agenda direction vectors from contrastive prompts."""

    def __init__(self, model: nn.Module, hook_config: HookConfig | None = None) -> None:
        self._model = model
        self._hook_config = hook_config or HookConfig()

    def extract_from_ids(
        self,
        agenda_a_ids: Tensor,
        agenda_b_ids: Tensor,
        token_position: int = -1,
    ) -> dict[int, Tensor]:
        """Compute per-layer diff-of-means vectors from pre-tokenized batches.

        Args:
            agenda_a_ids: Token ids of shape ``[num_pairs, seq_len]`` for the
                agenda-A phrasing of each pair (left-padded so ``token_position``
                aligns with real content across the batch).
            agenda_b_ids: Token ids of shape ``[num_pairs, seq_len]`` for the
                matched agenda-B phrasing of each pair.
            token_position: Sequence index to read per example (default: last).

        Returns:
            Mapping from decoder-layer index to a raw (non-unit-norm) direction
            vector of shape ``[hidden_size]``, pointing from agenda B towards
            agenda A.

        Raises:
            ValueError: If the two batches don't contain the same number of pairs.
        """
        if agenda_a_ids.shape[0] != agenda_b_ids.shape[0]:
            raise ValueError(
                f"Mismatched pair counts: {agenda_a_ids.shape[0]} vs {agenda_b_ids.shape[0]}"
            )

        with torch.no_grad(), ResidualStreamRecorder(self._model, self._hook_config) as recorder:
            self._model(input_ids=agenda_a_ids)
            agenda_a_activations = dict(recorder.activations)

        with torch.no_grad(), ResidualStreamRecorder(self._model, self._hook_config) as recorder:
            self._model(input_ids=agenda_b_ids)
            agenda_b_activations = dict(recorder.activations)

        directions: dict[int, Tensor] = {}
        for layer_idx in agenda_a_activations:
            a_last = agenda_a_activations[layer_idx][:, token_position, :]
            b_last = agenda_b_activations[layer_idx][:, token_position, :]
            directions[layer_idx] = (a_last - b_last).mean(dim=0)
        logger.debug("Extracted agenda directions for %d layers", len(directions))
        return directions

    def extract(
        self,
        tokenizer,
        pairs: tuple[PromptPair, ...],
        token_position: int = -1,
    ) -> dict[int, Tensor]:
        """Tokenize contrastive pairs and delegate to :meth:`extract_from_ids`.

        Args:
            tokenizer: A Hugging Face tokenizer with padding configured
                (``tokenizer.padding_side`` should be ``"left"`` so that
                ``token_position=-1`` reliably lands on real content).
            pairs: Contrastive prompt pairs, e.g.
                :data:`msm_mechinterp.data.prompts.PRO_AMERICA_VS_PRO_AFFORDABILITY`.
            token_position: Sequence index to read per example.
        """
        agenda_a_texts, agenda_b_texts = prompt_pairs_to_texts(pairs)
        agenda_a_ids = tokenizer(list(agenda_a_texts), return_tensors="pt", padding=True)["input_ids"]
        agenda_b_ids = tokenizer(list(agenda_b_texts), return_tensors="pt", padding=True)["input_ids"]
        return self.extract_from_ids(agenda_a_ids, agenda_b_ids, token_position=token_position)
