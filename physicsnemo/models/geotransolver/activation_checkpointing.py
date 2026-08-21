# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Activation-checkpointing helpers for GeoTransolver."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch

from physicsnemo.models.utils.activation_checkpointing import (
    run_checkpoint,
    should_checkpoint_interleaved_block,
)

if TYPE_CHECKING:
    from physicsnemo.nn import GALEBlock

DEFAULT_CHECKPOINTING_COMPONENTS = frozenset({"blocks"})
CHECKPOINTABLE_COMPONENTS = DEFAULT_CHECKPOINTING_COMPONENTS | frozenset(
    {"context", "preprocess", "output"}
)


def parse_checkpointing_components(
    components: tuple[str, ...] | list[str],
) -> frozenset[str]:
    r"""Validate and normalize GeoTransolver checkpoint component names."""
    if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
        raise TypeError(
            "activation_checkpointing_components must be a sequence of strings"
        )
    if not components:
        raise ValueError(
            "activation_checkpointing_components must contain at least one component"
        )
    if not all(isinstance(component, str) for component in components):
        raise TypeError("activation_checkpointing_components must contain only strings")
    normalized = frozenset(components)
    unknown = normalized - CHECKPOINTABLE_COMPONENTS
    if unknown:
        raise ValueError(
            "Unknown activation_checkpointing_components values: "
            f"{sorted(unknown)}; expected a subset of "
            f"{sorted(CHECKPOINTABLE_COMPONENTS)}"
        )
    return normalized


def should_checkpoint_component(
    component: str,
    enabled: bool,
    components: frozenset[str],
    *,
    training: bool,
) -> bool:
    r"""Return whether a GeoTransolver component should be checkpointed."""
    if not training or not torch.is_grad_enabled():
        return False
    return enabled and component in components


def should_checkpoint_block(
    block_idx: int,
    block_count: int,
    ratio: float,
    components: frozenset[str],
    *,
    training: bool,
) -> bool:
    r"""Return whether a GALE block is in the interleaved checkpoint set."""
    if "blocks" not in components:
        return False
    return should_checkpoint_interleaved_block(
        block_idx,
        block_count,
        ratio,
        training=training,
    )


def run_checkpointed_component(
    function: Callable[..., Any],
    *inputs: Any,
    enabled: bool,
    use_te: bool,
    te_module: Any,
) -> Any:
    r"""Run a component directly or under checkpointing."""
    if enabled:
        return run_checkpoint(function, *inputs, use_te=use_te, te_module=te_module)
    return function(*inputs)


def checkpoint_block(
    block: GALEBlock,
    streams: tuple[torch.Tensor, ...] | list[torch.Tensor],
    embedding_states: torch.Tensor | None,
    *,
    use_te: bool,
    te_module: Any,
) -> list[torch.Tensor]:
    r"""Checkpoint a multi-stream GALE block with explicit tensor inputs."""
    stream_count = len(streams)
    checkpoint_inputs = tuple(streams)
    if embedding_states is not None:
        checkpoint_inputs = (*checkpoint_inputs, embedding_states)

    def block_forward(*inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        block_streams = tuple(inputs[:stream_count])
        context = inputs[stream_count] if embedding_states is not None else None
        return tuple(block(block_streams, context))

    outputs = run_checkpoint(
        block_forward,
        *checkpoint_inputs,
        use_te=use_te,
        te_module=te_module,
    )
    return list(outputs)
