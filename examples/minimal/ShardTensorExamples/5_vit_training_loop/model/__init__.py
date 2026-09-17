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

r"""Model registry for the domain-parallel training benchmark.

Every model here is a completely vanilla ``nn.Module`` with zero distributed
code -- that is the point of the example.  Domain parallelism lives entirely
in the training script: inputs are scattered onto the mesh and ShardTensor's
auto-promotion handles the plain weights.

Bring your own model
--------------------

1. Write a plain ``nn.Module`` (no ShardTensor / distributed imports) that
   takes ``(B, in_channels, *spatial)`` and returns ``(B, num_classes)``
   logits.  ``B`` per domain group must be 1; spatial dims may be 2D or 3D.
2. Register it below with a :class:`ModelSpec`:

   - ``build``: constructor call taking ``img_size``, ``in_channels``,
     ``num_classes``.
   - ``compile_regions``: which submodules to ``torch.compile`` when the
     domain is sharded.  Anything whose sharded implementation performs
     halo/ring communication that cannot be captured in a graph (sequence-
     sharded SDPA -> ring attention, neighborhood attention) must stay
     eager; compile the rest around it.  With ``domain_size == 1`` the
     script compiles the whole model instead and this hook is unused.
   - ``spatial_param_dim``: maps a parameter name to the tensor dim it
     should be sharded along on the FSDP2 path (statically-shaped spatial
     params such as positional embeddings).  Return ``None`` to leave a
     parameter plain/replicated.  Unused on the DDP path, where
     auto-promotion handles replicated params.
   - ``requires``: optional module names that must be importable
     (e.g. ``natten``); the script errors out early with an install hint.

That is the entire integration surface: the model itself never changes.

This example code is really meant for tutorials, benchmarking, and similar purposes.
It's not research or production code.
"""

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from .ViT import HybridViT
from .ConvNet import HybridConvNet
from .NattenNet import HybridNattenViT


@dataclass(frozen=True)
class ModelSpec:
    """Everything the training script needs to know about a model choice."""

    build: Callable[..., nn.Module]
    compile_regions: Callable[[nn.Module], None]
    spatial_param_dim: Callable[[str], int | None] = lambda name: None
    requires: tuple[str, ...] = ()


def _compile_around_attention(model: nn.Module) -> None:
    """Regionally compile a ViT-style model, leaving attention eager.

    Sharded attention (ring SDPA / neighborhood attention halos) cannot live
    inside a compiled region; the patch embedding, per-block norms and MLPs,
    and the head are compiled around it.  ``dynamic=False``: fixed shapes per
    sweep step, and it prevents automatic-dynamic from promoting sizes to
    SymInts when submodules share dynamo frames.
    """
    model.patch_embed = torch.compile(model.patch_embed, dynamic=False)
    for block in model.stages:
        block.norm1 = torch.compile(block.norm1, dynamic=False)
        block.norm2 = torch.compile(block.norm2, dynamic=False)
        block.mlp = torch.compile(block.mlp, dynamic=False)
    model.head = torch.compile(model.head, dynamic=False)


def _compile_conv_stages(model: nn.Module) -> None:
    """Compile the conv model per-region.

    Unlike attention, the sharded convolutions' halo exchanges are pointwise
    neighbor transfers; each residual stage is compiled as a unit.
    """
    model.patch_embed = torch.compile(model.patch_embed, dynamic=False)
    for block in model.stages:
        block.compile(dynamic=False)
    model.head = torch.compile(model.head, dynamic=False)


def _pos_embed_dim_1(name: str) -> int | None:
    # Positional embeddings are laid out (1, sequence-or-height, ...): the
    # first non-batch dim tracks the domain-sharded axis.
    return 1 if "pos_embed" in name else None


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "vit": ModelSpec(
        build=HybridViT,
        compile_regions=_compile_around_attention,
        spatial_param_dim=_pos_embed_dim_1,
    ),
    "conv": ModelSpec(
        build=HybridConvNet,
        compile_regions=_compile_conv_stages,
    ),
    "natten": ModelSpec(
        build=HybridNattenViT,
        compile_regions=_compile_around_attention,
        spatial_param_dim=_pos_embed_dim_1,
        requires=("natten",),
    ),
}
