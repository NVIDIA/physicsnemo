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

import torch
from torch import Tensor

from physicsnemo.core.function_spec import FunctionSpec


class WeightFact(FunctionSpec):
    """Randomly factorize the weight matrix into a product of vectors and a matrix.

    Parameters
    ----------
    w : torch.Tensor
        Weight tensor to factorize.
    mean : float, optional
        Mean of the normal distribution used to sample the scale factor.
    stddev : float, optional
        Standard deviation of the normal distribution used to sample the scale factor.
    implementation : {"torch"} or None
        Implementation to use. When ``None``, dispatch selects the available
        implementation.
    """

    @FunctionSpec.register(name="torch", rank=0, baseline=True)
    def torch_forward(w: Tensor, mean: float = 1.0, stddev: float = 0.1):
        g = torch.normal(mean, stddev, size=(w.shape[0], 1), device=w.device)
        g = torch.exp(g)
        v = w / g
        return g, v

    @classmethod
    def make_inputs(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        cases = [
            ("small", 256),
            ("medium", 512),
            ("large", 1024),
        ]
        for label, size in cases:
            w = torch.randn(size, size, device=device)
            yield (
                f"{label}-weight-matrix{size}x{size}-mean1p0-std0p1",
                (w,),
                {"mean": 1.0, "stddev": 0.1},
            )


weight_fact = WeightFact.make_function("weight_fact")


__all__ = [
    "WeightFact",
    "weight_fact",
]
