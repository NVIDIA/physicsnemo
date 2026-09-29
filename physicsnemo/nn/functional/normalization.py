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

"""Vector normalization across floating-point dtypes and scales."""

import torch
from jaxtyping import Float

from physicsnemo.core.function_spec import FunctionSpec


class SafeNormalize(FunctionSpec):
    """Scale vectors to unit L2 length along ``dim``, preserving zero vectors.

    Each vector is divided by its largest absolute component before computing
    its norm. This avoids overflow and underflow from the input magnitude
    without an absolute epsilon floor that would shorten small vectors.

    Parameters
    ----------
    vectors : Float[torch.Tensor, "..."]
        Floating-point vectors of any shape.
    dim : int
        Dimension holding the vector components.
    implementation : {"torch"} or None
        Implementation to use. When ``None``, dispatch selects the available
        implementation.

    Returns
    -------
    Float[torch.Tensor, "..."]
        Unit vectors with the input shape, device, and dtype, including under
        autocast. Exactly zero vectors remain zero.

    Notes
    -----
    Non-finite components propagate NaNs to the whole vector. Derivatives
    near zero can exceed the dtype's range even when forward values are finite.

    Examples
    --------
    >>> v = torch.tensor([[3.0e-13, 4.0e-13], [0.0, 0.0]])
    >>> safe_normalize(v, dim=-1)
    tensor([[0.6000, 0.8000],
            [0.0000, 0.0000]])
    """

    _BENCHMARK_CASES = (
        ("small-1024x3", 1024),
        ("medium-16384x3", 16384),
        ("large-262144x3", 262144),
    )

    @FunctionSpec.register(name="torch", rank=0, baseline=True)
    def torch_forward(
        vectors: Float[torch.Tensor, "..."],
        dim: int,
    ) -> Float[torch.Tensor, "..."]:
        """Normalize vectors using rescaled PyTorch operations."""
        # Avoid an empty reduction, which amax does not support.
        if vectors.shape[dim] == 0:
            return vectors

        scale = vectors.abs().amax(dim=dim, keepdim=True)
        is_zero = scale == 0
        scaled = vectors / scale.masked_fill(is_zero, 1)
        norm = scaled.norm(dim=dim, keepdim=True)
        return (scaled / norm.masked_fill(is_zero, 1)).to(vectors.dtype)

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Generate increasing batches of three-component vectors."""
        for label, n_vectors in cls._BENCHMARK_CASES:
            vectors = torch.randn(n_vectors, 3, device=device)
            yield label, (vectors,), {"dim": -1}

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Generate differentiable versions of the forward workloads."""
        for label, (vectors,), kwargs in cls.make_inputs_forward(device=device):
            yield label, (vectors.requires_grad_(),), kwargs


safe_normalize = SafeNormalize.make_function("safe_normalize")


__all__ = ["SafeNormalize", "safe_normalize"]
