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

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from physicsnemo.models.utils.activation_checkpointing import (
    run_checkpoint,
    should_checkpoint_interleaved_block,
)

if TYPE_CHECKING:
    from physicsnemo.nn import GALEBlock

    from .context_projector import GlobalContextBuilder

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
    normalized = frozenset(components)
    if not normalized:
        raise ValueError(
            "activation_checkpointing_components must contain at least one component"
        )
    if not all(isinstance(component, str) for component in normalized):
        raise TypeError("activation_checkpointing_components must contain only strings")
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
    ratio: float,
    components: frozenset[str],
    *,
    training: bool,
) -> bool:
    r"""Return whether a GeoTransolver component should be checkpointed."""
    if not training or not torch.is_grad_enabled():
        return False
    return ratio > 0.0 and component in components


def should_checkpoint_block(
    block_idx: int,
    block_count: int,
    ratio: float,
    components: frozenset[str],
    *,
    training: bool,
) -> bool:
    r"""Return whether a GALE block is in the interleaved checkpoint set."""
    if not should_checkpoint_component("blocks", ratio, components, training=training):
        return False
    return should_checkpoint_interleaved_block(
        block_idx,
        block_count,
        ratio,
        training=training,
    )


def run_checkpointed_component(
    function: nn.Module,
    input_tensor: torch.Tensor,
    *,
    enabled: bool,
    use_te: bool,
    te_module: Any,
) -> torch.Tensor:
    r"""Run a single-tensor component directly or under checkpointing."""
    if enabled:
        return run_checkpoint(
            function, input_tensor, use_te=use_te, te_module=te_module
        )
    return function(input_tensor)


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


def build_context(
    context_builder: GlobalContextBuilder,
    local_embedding: tuple[torch.Tensor, ...],
    local_positions: tuple[torch.Tensor, ...] | None,
    geometry: torch.Tensor | None,
    global_embedding: torch.Tensor | None,
    *,
    checkpoint_enabled: bool,
    use_te: bool,
    te_module: Any,
) -> tuple[
    torch.Tensor | None,
    list[torch.Tensor] | None,
    torch.Tensor | None,
]:
    r"""Build context directly or under a flattened checkpoint boundary."""
    if not checkpoint_enabled:
        return context_builder.build_context(
            local_embedding, local_positions, geometry, global_embedding
        )

    stream_count = len(local_embedding)
    has_local_features = (
        context_builder.local_extractors is not None and geometry is not None
    )
    has_geometry_context = (
        context_builder.geometry_tokenizer is not None and geometry is not None
    )
    has_global_context = (
        context_builder.global_tokenizer is not None and global_embedding is not None
    )
    has_context = has_local_features or has_geometry_context or has_global_context

    # Preserve the context builder's validation and no-op behavior when there
    # is no tensor-producing component to checkpoint.
    if (
        not local_embedding
        or not has_context
        or (
            has_local_features
            and (
                local_positions is None
                or not all(
                    isinstance(position, torch.Tensor) for position in local_positions
                )
            )
        )
    ):
        return context_builder.build_context(
            local_embedding, local_positions, geometry, global_embedding
        )

    checkpoint_inputs: tuple[torch.Tensor, ...] = tuple(local_embedding)
    if has_local_features:
        checkpoint_inputs = (*checkpoint_inputs, *local_positions)
    if geometry is not None:
        checkpoint_inputs = (*checkpoint_inputs, geometry)
    if global_embedding is not None:
        checkpoint_inputs = (*checkpoint_inputs, global_embedding)

    def context_forward(*inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        offset = 0
        embeddings = tuple(inputs[offset : offset + stream_count])
        offset += stream_count
        if has_local_features:
            positions = tuple(inputs[offset : offset + stream_count])
            offset += stream_count
        else:
            positions = local_positions
        geometry_input = inputs[offset] if geometry is not None else None
        offset += int(geometry is not None)
        global_input = inputs[offset] if global_embedding is not None else None
        context, local_features, geometry_context = context_builder.build_context(
            embeddings,
            positions,
            geometry_input,
            global_input,
        )
        flat_outputs = [context]
        if has_local_features:
            flat_outputs.extend(local_features)
        if has_geometry_context:
            flat_outputs.append(geometry_context)
        return tuple(flat_outputs)

    flat_outputs = run_checkpoint(
        context_forward,
        *checkpoint_inputs,
        use_te=use_te,
        te_module=te_module,
    )
    if isinstance(flat_outputs, torch.Tensor):
        flat_outputs = (flat_outputs,)

    offset = 0
    context = flat_outputs[offset]
    offset += 1
    local_features = None
    if has_local_features:
        local_features = list(flat_outputs[offset : offset + stream_count])
        offset += stream_count
    geometry_context = flat_outputs[offset] if has_geometry_context else None
    return context, local_features, geometry_context
