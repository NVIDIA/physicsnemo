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

r"""Backend-independent contracts for diffusion distillation."""

from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias, runtime_checkable

from jaxtyping import Float
from tensordict import TensorDict
from torch import Tensor

LossMap: TypeAlias = Mapping[str, Float[Tensor, " *shape"]]


@runtime_checkable
class BaseDistillationLoss(Protocol):
    r"""Callable interface for diffusion distillation losses.

    A loss returns a dictionary containing a differentiable ``total_loss``
    and optional named values for logging. It computes the loss for one
    batch; the training loop handles backpropagation and optimizer updates.

    Use this protocol to define custom losses as functions or callable classes.
    Define ``__call__`` and backpropagate through ``total_loss`` only.
    Inheritance and FastGen are optional.

    Examples
    --------
    A plain function follows this protocol by returning a named loss map.
    The runtime check confirms that the callable exposes the required interface.

    >>> def custom_loss(x0, *, condition=None, **kwargs):
    ...     return {"total_loss": x0.square().mean()}
    >>> isinstance(custom_loss, BaseDistillationLoss)
    True
    """

    def __call__(
        self,
        x0: Float[Tensor, " B *dims"],
        *,
        condition: Float[Tensor, " B *cond_dims"] | TensorDict | None = None,
        **kwargs: Any,
    ) -> LossMap | tuple[LossMap, Mapping[str, Any]]:
        r"""Return named losses for one batch of clean data."""
        ...
