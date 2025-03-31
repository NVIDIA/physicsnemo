# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

# ruff: noqa: S101
from typing import Literal, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch_scatter import segment_csr

REDUCTIONS = ["min", "max", "mean", "sum", "var", "std"]
REDUCTION_TYPES = Literal["min", "max", "mean", "sum", "var", "std"]


def _var(
    features: Float[Tensor, "N F"], neighbors_row_splits: Int[Tensor, "M"]  # noqa
) -> Tuple[Float[Tensor, "M F"], Float[Tensor, "M F"]]:  # noqa
    out_mean = segment_csr(features, neighbors_row_splits, reduce="mean")
    out_var = (
        segment_csr(features**2, neighbors_row_splits, reduce="mean") - out_mean**2
    )
    return out_var, out_mean


def row_reduction(
    features: Float[Tensor, "N F"],  # noqa
    neighbors_row_splits: Int[Tensor, "M"],  # noqa
    reduction: REDUCTION_TYPES,
    eps: float = 1e-6,
) -> Float[Tensor, "M F"]:  # noqa
    assert reduction in REDUCTIONS

    if reduction in ["min", "max", "mean", "sum"]:
        out_feature = segment_csr(features, neighbors_row_splits, reduce=reduction)
    elif reduction == "var":
        out_feature = _var(features, neighbors_row_splits)[0]
    elif reduction == "std":
        out_feature = torch.sqrt(_var(features, neighbors_row_splits)[0] + eps)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    return out_feature
