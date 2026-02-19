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

from typing import Literal, Sequence

import torch
import torch.nn as nn
from jaxtyping import Float

from physicsnemo.core.module import Module


class Normalizer(Module):
    r"""Shape-aware EWMA normalization in log-space based on ``mean(abs(x))``.

    Maintains a running estimate of ``mean(abs(x))`` in log-space and
    normalizes inputs by that estimate. The tracked statistic can be a
    scalar or have arbitrary shape, controlled by the ``shape`` argument:

    - If a dimension in ``shape`` is 1, that dimension is reduced by
      computing the mean across that axis.
    - If a dimension in ``shape`` equals the corresponding input dimension,
      no reduction is applied and the statistic is per-position.
    - Exact zeros in the reduced axes are ignored. If an entire reduction
      slice is zero, the update leaves the running estimate unchanged.

    In training mode each forward call computes ``mean(abs(x))``, converts
    to log-space, and updates ``ln_running_mean`` via a numerically stable
    log-domain EWMA. In eval mode, state is not updated. In both modes the
    output is :math:`x \cdot \exp(-\text{ln\_running\_mean})`.

    Parameters
    ----------
    shape : torch.Size | Sequence[int], optional, default=()
        Target shape of the running statistic. Must have the same number
        of dimensions as the input.
    momentum : float, optional, default=0.99
        EWMA momentum in :math:`[0, 1)`. Higher values emphasize
        historical estimates.
    learnable_weight_shape : torch.Size | Sequence[int] | None, optional, default=None
        If provided, creates a learnable multiplicative ``weight``
        parameter with the given shape.
    learnable_bias_shape : torch.Size | Sequence[int] | None, optional, default=None
        If provided, creates a learnable additive ``bias`` parameter
        with the given shape.
    initialization_behavior : Literal["no_op", "first_batch"], optional, default="no_op"
        Controls how the running mean is initialized on the first
        forward call.
    disable : bool, optional, default=False
        If ``True``, the normalizer is a no-op pass-through.

    Forward
    -------
    x : torch.Tensor
        Input tensor whose rank must match ``len(shape)``.

    Outputs
    -------
    torch.Tensor
        Normalized tensor with the same shape as the input.

    Notes
    -----
    ``ln_running_mean`` is a registered buffer, so it moves with the
    module across devices and is included in ``state_dict``. On
    initialization it is zeros (running mean = 1), making eval-before-train
    a no-op normalization.

    Examples
    --------
    >>> norm = Normalizer(shape=(1,), momentum=0.99)
    >>> norm.train()
    Normalizer(ln_running_mean=tensor([0.]))
    >>> x = torch.randn(32)
    >>> y = norm(x)  # updates running mean, returns normalized x
    """

    def __init__(
        self,
        shape: torch.Size | Sequence[int] = tuple(),
        momentum: float = 0.99,
        learnable_weight_shape: torch.Size | Sequence[int] | None = None,
        learnable_bias_shape: torch.Size | Sequence[int] | None = None,
        initialization_behavior: Literal["no_op", "first_batch"] = "no_op",
        disable: bool = False,
    ):
        ### Validate inputs
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"{momentum= } must satisfy 0 <= momentum < 1")

        super().__init__()

        ### Store parameters
        self.shape = shape
        self.momentum = momentum
        self.learnable_weight_shape = learnable_weight_shape
        self.learnable_bias_shape = learnable_bias_shape
        self.initialization_behavior = initialization_behavior
        self.disable = disable

        ### Initialize learnable parameters, if applicable
        if not disable:
            if self.learnable_weight_shape is not None:
                self.weight = nn.Parameter(torch.ones(self.learnable_weight_shape))
            if self.learnable_bias_shape is not None:
                self.bias = nn.Parameter(torch.zeros(self.learnable_bias_shape))

        ### Buffers to track state across devices/checkpoints
        # Start with ln(1.0) = 0.0 so eval before train is a no-op.
        self.register_buffer("ln_running_mean", torch.zeros(shape))
        self.register_buffer("_initialized", torch.tensor(False, dtype=torch.bool))

    def __repr__(self) -> str:
        r"""String representation showing the current running mean state.

        Returns
        -------
        str
            Format: ``Normalizer(ln_running_mean=<tensor>)``.
        """
        return f"Normalizer(ln_running_mean={self.ln_running_mean!r})"

    def forward(self, x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
        r"""Normalize by a log-space EWMA of ``mean(abs(x))``.

        In train mode, updates ``ln_running_mean`` from the current batch
        and returns normalized output. In eval mode, returns normalized
        output without updating state.

        Parameters
        ----------
        x : Float[torch.Tensor, "..."]
            Input tensor whose rank must match ``len(self.shape)``.

        Returns
        -------
        Float[torch.Tensor, "..."]
            Normalized tensor, same shape as ``x``.
        """
        if self.disable:
            return x

        ### Input validation
        # Skip validation when running under torch.compile for performance
        if not torch.compiler.is_compiling():
            if len(self.shape) != len(x.shape):
                raise ValueError(
                    f"Expected input with {len(self.shape)} dimensions matching "
                    f"{self.shape=!r}, got tensor with shape {tuple(x.shape)}"
                )

        if self.training:
            # Compute abs(x) for statistic updates out of autograd
            values = x.detach().abs()

            values[values == 0.0] = torch.nan  # Allows nanmean to ignore zeros

            # Reduce axes where target shape has size 1
            for dim, (desired_size, actual_size) in enumerate(zip(self.shape, x.shape)):
                if desired_size != actual_size:
                    if desired_size == 1:
                        values = torch.nanmean(values, dim=dim, keepdim=True)
                    else:
                        raise ValueError(
                            f"Shape mismatch for {dim=!r}: "
                            f"{desired_size=!r} != {actual_size=!r}"
                        )

            values[torch.isnan(values)] = 0.0  # Makes 0 the default (no-op)

            ln_batch_mean = values.log()

            ### Compute the new running mean value
            # Uninitialized case
            if self.initialization_behavior == "first_batch":
                new_ln_running_mean_if_uninitialized = ln_batch_mean
            elif self.initialization_behavior == "no_op":
                new_ln_running_mean_if_uninitialized = self.ln_running_mean
            else:
                raise ValueError(f"Invalid {self.initialization_behavior=!r}")

            # Initialized case
            new_ln_running_mean_if_initialized = (
                self.momentum * self.ln_running_mean
                + (1.0 - self.momentum) * ln_batch_mean
            )

            # Update
            self.ln_running_mean.copy_(
                torch.where(
                    self._initialized,
                    new_ln_running_mean_if_initialized,
                    new_ln_running_mean_if_uninitialized,
                )
            )
            self._initialized.fill_(True)

        # Divide by running mean via exp(-ln_running_mean)
        x = x * torch.exp(-self.ln_running_mean)
        if self.learnable_weight_shape is not None:
            x = x * self.weight
        if self.learnable_bias_shape is not None:
            x = x + self.bias
        return x
