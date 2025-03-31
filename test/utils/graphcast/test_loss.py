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

import torch

from physicsnemo.utils.graphcast.loss import (
    CellAreaWeightedLossFunction,
    CustomCellAreaWeightedLossFunction,
)


def test_loss():
    """Tests if the custom loss function is equivalent to the default loss function."""
    pred1 = torch.rand(1, 2, 721, 1440, device="cuda")
    target1 = torch.rand(1, 2, 721, 1440, device="cuda")
    area = torch.rand(721, 1440, device="cuda")

    default_loss = CellAreaWeightedLossFunction(area)
    custom_loss = CustomCellAreaWeightedLossFunction(area)

    pred2 = pred1.clone().detach()
    target2 = target1.clone().detach()

    pred1.requires_grad_()
    pred2.requires_grad_()

    loss1 = default_loss(pred1, target1)
    loss1.backward()
    grad1 = pred1.grad

    loss2 = custom_loss(pred2, target2)
    loss2.backward()
    grad2 = pred2.grad

    atol = 1.0e-7
    loss_diff = torch.abs(loss1 - loss2)
    loss_diff_msg = (
        f"loss diff - min/max/mean: {loss_diff.min()} / "
        f"{loss_diff.max()} / {loss_diff.mean()}"
    )
    grad_diff = torch.abs(grad1 - grad2)
    grad_diff_msg = (
        f"grad diff - min/max/mean: {grad_diff.min()} / "
        f"{grad_diff.max()} / {grad_diff.mean()}"
    )

    assert torch.allclose(loss1, loss2, atol=atol), loss_diff_msg + " for loss"
    assert torch.allclose(grad1, grad2, atol=atol), grad_diff_msg + " for gradient"
