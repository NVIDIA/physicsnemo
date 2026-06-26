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
"""Pluggable cross-attention sub-layer contract."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from jaxtyping import Float

from physicsnemo.core import Module


class CrossAttentionModuleBase(Module, ABC):
    r"""Abstract base for a cross-attention sub-layer.

    A concrete module attends from ``hidden_states`` to an arbitrary external
    ``context`` that the module fully owns (its type, layout, and any folding /
    packing). The caller treats ``context`` as opaque.

    Forward
    -------
    hidden_states : torch.Tensor
        Latents of shape :math:`(*B, C)`.
    context : Any
        Module-defined conditioning source, opaque to the caller.

    Outputs
    -------
    torch.Tensor
        Updated latents of shape :math:`(*B, C)`.
    """

    @abstractmethod
    def forward(
        self,
        hidden_states: Float[torch.Tensor, "*batch hidden_size"],
        context: Any,
    ) -> Float[torch.Tensor, "*batch hidden_size"]:
        pass
