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


def _drop_path_torch(
    x: Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> Tensor:
    """Apply stochastic depth per sample."""
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(FunctionSpec):
    """Drop paths (stochastic depth) per sample.

    Cut & paste from timm master. Drop paths (Stochastic Depth) per sample (when
    applied in main path of residual blocks). This is the same as the
    DropConnect implementation used for EfficientNet and related networks, but
    the original name is misleading as "Drop Connect" is a different form of
    dropout. See: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    for discussion.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    drop_prob : float, optional
        Drop probability, by default 0.0.
    training : bool, optional
        Whether stochastic depth is enabled, by default False.
    scale_by_keep : bool, optional
        Scale by keep probability, by default True.
    implementation : {"torch"} or None
        Implementation to use. When ``None``, dispatch selects the available
        implementation.

    Notes
    -----
    The layer and argument names use "drop path" rather than mixing DropConnect
    or "survival rate" to align with common usage.
    """

    @FunctionSpec.register(name="torch", rank=0, baseline=True)
    def torch_forward(
        x: Tensor,
        drop_prob: float = 0.0,
        training: bool = False,
        scale_by_keep: bool = True,
    ) -> Tensor:
        return _drop_path_torch(
            x,
            drop_prob=drop_prob,
            training=training,
            scale_by_keep=scale_by_keep,
        )

    @classmethod
    def make_inputs(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        cases = [
            ("small", 8, 64),
            ("medium", 16, 256),
            ("large", 32, 1024),
        ]
        for label, batch, features in cases:
            x = torch.randn(batch, features, device=device)
            yield (
                f"{label}-batch{batch}-features{features}-drop0p1-train",
                (x,),
                {"drop_prob": 0.1, "training": True, "scale_by_keep": True},
            )


drop_path = DropPath.make_function("drop_path")


__all__ = [
    "DropPath",
    "drop_path",
]
