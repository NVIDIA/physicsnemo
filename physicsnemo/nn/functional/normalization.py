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


def safe_normalize(
    vectors: Float[torch.Tensor, "..."],
    dim: int,
) -> Float[torch.Tensor, "..."]:
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
    # Avoid an empty reduction, which amax does not support.
    if vectors.shape[dim] == 0:
        return vectors

    scale = vectors.abs().amax(dim=dim, keepdim=True)
    is_zero = scale == 0
    scaled = vectors / scale.masked_fill(is_zero, 1)
    norm = scaled.norm(dim=dim, keepdim=True)
    return (scaled / norm.masked_fill(is_zero, 1)).to(vectors.dtype)
