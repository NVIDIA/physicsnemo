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

from typing import Any, Mapping, Optional

import torch
from torch import Tensor

from dgl import DGLGraph


class RRMSELoss(torch.nn.Module):
    """Relative RMSE loss."""

    def forward(self, pred: Tensor, target: Tensor):
        return (
            torch.linalg.vector_norm(pred - target) / torch.linalg.vector_norm(target)
        ).mean()


def batch_as_dict(
    batch, device: Optional[torch.device | str] = None
) -> Mapping[str, Any]:
    """Wraps provided batch in a dictionary, if needed.

    If `device` is not None, moves all Tensor items to the device.
    """

    batch = batch if isinstance(batch, Mapping) else {"graph": batch}
    if device is None:
        return batch
    return {
        k: v.to(device) if isinstance(v, (Tensor, DGLGraph)) else v
        for k, v in batch.items()
    }


def relative_lp_error(pred, y, p=2):
    """
    Calculate relative L2 error norm
    Parameters:
    -----------
    pred: torch.Tensor
        Prediction
    y: torch.Tensor
        Ground truth
    Returns:
    --------
    error: float
        Calculated relative L2 error norm (percentage) on cpu
    """

    error = (
        torch.mean(torch.linalg.norm(pred - y, ord=p) / torch.linalg.norm(y, ord=p))
        .cpu()
        .numpy()
    )
    return error * 100
