"""Shared pytest fixtures: a tiny, randomly-initialized Llama-architecture model.

This model shares the exact module structure (`model.model.layers`,
`model.model.norm`, `model.lm_head`) as the real Llama-3.1-8B MSM checkpoints,
so every hook/logit-lens/direction/analysis code path exercised here ports
unchanged to Lambda Labs — only the config size and weights differ.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from msm_mechinterp.config import set_global_seed

TINY_VOCAB_SIZE = 97
TINY_HIDDEN_SIZE = 32
TINY_NUM_LAYERS = 3
TINY_SEQ_LEN = 6


@pytest.fixture()
def tiny_model() -> LlamaForCausalLM:
    set_global_seed(42)
    config = LlamaConfig(
        vocab_size=TINY_VOCAB_SIZE,
        hidden_size=TINY_HIDDEN_SIZE,
        intermediate_size=TINY_HIDDEN_SIZE * 2,
        num_hidden_layers=TINY_NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config)
    model.eval()
    return model


@pytest.fixture()
def tiny_input_ids() -> torch.Tensor:
    set_global_seed(123)
    return torch.randint(low=0, high=TINY_VOCAB_SIZE, size=(2, TINY_SEQ_LEN))
