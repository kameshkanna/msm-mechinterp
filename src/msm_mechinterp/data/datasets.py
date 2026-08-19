"""Loaders for the MSM paper's released Hugging Face eval datasets.

These functions require network + hub access and are meant to run only on
Lambda Labs (or any machine with real internet access to the Hub), never in
local dry tests. Install the optional ``hub`` extra (``datasets``,
``huggingface_hub``) before use.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Repo ids as published in the project's GitHub README. Confirm these still
# resolve before a real run; the MSM paper's own repo is the source of truth,
# not this constant.
PRO_AMERICA_POLITICAL_OPINIONS = "chloeli/pro-america-political-opinions"
PRO_AFFORDABILITY_ITEM_COMPARISONS = "chloeli/pro-affordability-item-comparisons"
SPEC_OPEN_QA = "chloeli/spec-open-qa"


def load_eval_dataset(repo_id: str, split: str = "train") -> Any:
    """Load one of the MSM paper's released eval datasets from the Hub.

    Args:
        repo_id: One of the module-level ``*_ID`` constants, or any other
            public dataset repo id.
        split: Dataset split to load.

    Returns:
        A `datasets.Dataset` instance.

    Raises:
        ImportError: If the optional ``datasets`` dependency is not installed
            (expected locally; install the ``hub`` extra on Lambda Labs).
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to load Hub eval sets; "
            "install the 'hub' optional extra on a machine with network access."
        ) from exc

    logger.info("Loading dataset %s (split=%s)", repo_id, split)
    return load_dataset(repo_id, split=split)
