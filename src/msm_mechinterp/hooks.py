"""Forward-hook primitives for reading, patching, and ablating a decoder-only
transformer's residual stream.

All three interventions target the same module path resolved from
:class:`~msm_mechinterp.config.HookConfig`, so they operate identically on a
tiny synthetic ``LlamaForCausalLM`` used in dry tests and on the real MSM
checkpoints loaded on Lambda Labs — no branching between test and production
code paths.
"""

from __future__ import annotations

import logging
from functools import reduce
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from msm_mechinterp.config import HookConfig

logger = logging.getLogger(__name__)


def resolve_attr_path(obj: Any, path: str) -> Any:
    """Resolve a dotted attribute path (e.g. ``"model.layers"``) against ``obj``.

    Args:
        obj: The root object (typically a top-level ``PreTrainedModel``).
        path: Dot-separated attribute names.

    Raises:
        AttributeError: If any segment of the path does not exist on ``obj``.
    """
    try:
        return reduce(getattr, path.split("."), obj)
    except AttributeError as exc:
        raise AttributeError(f"Could not resolve attribute path '{path}' on {type(obj).__name__}") from exc


def _decoder_layers(model: nn.Module, hook_config: HookConfig) -> nn.ModuleList:
    layers = resolve_attr_path(model, hook_config.layers_attr_path)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError(
            f"Expected an nn.ModuleList at '{hook_config.layers_attr_path}', got {type(layers).__name__}"
        )
    return layers


def num_decoder_layers(model: nn.Module, hook_config: HookConfig | None = None) -> int:
    """Return the number of decoder layers resolved from ``hook_config``.

    Args:
        model: Top-level model to inspect.
        hook_config: Module-path configuration.
    """
    return len(_decoder_layers(model, hook_config or HookConfig()))


def _split_layer_output(output: Any) -> tuple[Tensor, tuple[Any, ...]]:
    """Split a decoder layer's forward output into (hidden_states, rest).

    Handles both the tuple-returning and bare-tensor-returning conventions used
    across `transformers` versions for decoder layer forward methods.
    """
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, ()


def _rejoin_layer_output(hidden_states: Tensor, rest: tuple[Any, ...]) -> Any:
    return (hidden_states, *rest) if rest else hidden_states


class ResidualStreamRecorder:
    """Context manager that records each decoder layer's residual-stream output.

    Usage:
        >>> with ResidualStreamRecorder(model, hook_config) as recorder:
        ...     model(input_ids=input_ids)
        >>> recorder.activations[0].shape
        torch.Size([batch, seq_len, hidden_size])

    Attributes:
        activations: Populated only after a forward pass runs inside the
            context; maps decoder-layer index to its output hidden states.
    """

    def __init__(self, model: nn.Module, hook_config: HookConfig | None = None) -> None:
        self._model = model
        self._hook_config = hook_config or HookConfig()
        self._handles: list[RemovableHandle] = []
        self.activations: dict[int, Tensor] = {}

    def __enter__(self) -> "ResidualStreamRecorder":
        self.activations = {}
        layers = _decoder_layers(self._model, self._hook_config)
        for layer_idx, layer in enumerate(layers):
            self._handles.append(layer.register_forward_hook(self._make_hook(layer_idx)))
        return self

    def __exit__(self, *exc_info: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _make_hook(self, layer_idx: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            hidden_states, _ = _split_layer_output(output)
            self.activations[layer_idx] = hidden_states.detach()

        return hook


class ResidualStreamPatcher:
    """Context manager that overwrites one decoder layer's residual-stream output.

    Used for cross-run activation patching: capture a donor run's activation
    with :class:`ResidualStreamRecorder`, then inject it into a recipient run
    via this class to causally test which layer is responsible for an effect.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        replacement: Tensor,
        hook_config: HookConfig | None = None,
        positions: Tensor | slice | None = None,
    ) -> None:
        """
        Args:
            model: Top-level model to patch.
            layer_idx: Index of the decoder layer whose output is overwritten.
            replacement: Hidden-state tensor to inject, broadcastable to the
                layer's output shape ``[batch, seq_len, hidden_size]``.
            hook_config: Module-path configuration.
            positions: Optional index/slice into the sequence dimension; if
                ``None``, every position is overwritten with ``replacement``.
        """
        self._model = model
        self._layer_idx = layer_idx
        self._replacement = replacement
        self._hook_config = hook_config or HookConfig()
        self._positions = positions
        self._handle: RemovableHandle | None = None

    def __enter__(self) -> "ResidualStreamPatcher":
        layers = _decoder_layers(self._model, self._hook_config)
        if not 0 <= self._layer_idx < len(layers):
            raise IndexError(f"layer_idx {self._layer_idx} out of range for {len(layers)} layers")
        self._handle = layers[self._layer_idx].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden_states, rest = _split_layer_output(output)
        replacement = self._replacement.to(device=hidden_states.device, dtype=hidden_states.dtype)
        patched = hidden_states.clone()
        if self._positions is None:
            patched[...] = replacement
        else:
            patched[:, self._positions, :] = replacement
        return _rejoin_layer_output(patched, rest)


class DirectionAblator:
    """Context manager that projects a direction out of the residual stream.

    For each configured layer, removes the component of the hidden state
    along the corresponding unit direction at every position:
    ``h' = h - (h . d_hat) * d_hat``.
    """

    def __init__(
        self,
        model: nn.Module,
        directions_by_layer: dict[int, Tensor],
        hook_config: HookConfig | None = None,
    ) -> None:
        """
        Args:
            model: Top-level model to intervene on.
            directions_by_layer: Maps decoder-layer index to a raw (not
                necessarily unit-norm) direction vector of shape
                ``[hidden_size]``; normalized internally.
            hook_config: Module-path configuration.

        Raises:
            ValueError: If any direction vector has near-zero norm.
        """
        self._model = model
        self._hook_config = hook_config or HookConfig()
        self._unit_directions: dict[int, Tensor] = {}
        for layer_idx, direction in directions_by_layer.items():
            norm = direction.norm()
            if norm < 1e-8:
                raise ValueError(f"Direction for layer {layer_idx} has near-zero norm ({norm.item()})")
            self._unit_directions[layer_idx] = (direction / norm).detach()
        self._handles: list[RemovableHandle] = []

    def __enter__(self) -> "DirectionAblator":
        layers = _decoder_layers(self._model, self._hook_config)
        for layer_idx, unit_direction in self._unit_directions.items():
            if not 0 <= layer_idx < len(layers):
                raise IndexError(f"layer_idx {layer_idx} out of range for {len(layers)} layers")
            self._handles.append(layers[layer_idx].register_forward_hook(self._make_hook(unit_direction)))
        return self

    def __exit__(self, *exc_info: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @staticmethod
    def _make_hook(unit_direction: Tensor):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            hidden_states, rest = _split_layer_output(output)
            direction = unit_direction.to(device=hidden_states.device, dtype=hidden_states.dtype)
            projection = torch.einsum("...h,h->...", hidden_states, direction).unsqueeze(-1)
            ablated = hidden_states - projection * direction
            return _rejoin_layer_output(ablated, rest)

        return hook
