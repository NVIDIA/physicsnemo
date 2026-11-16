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

import json

import pytest
import torch

from physicsnemo.metrics.climate.graphcast_loss import (
    CellAreaWeightedLossFunction,
    CustomCellAreaWeightedLossFunction,
    GraphCastLossFunction,
)


def test_loss(device):
    """Tests if the custom loss function is equivalent to the default loss function."""
    pred1 = torch.rand(1, 2, 721, 1440, device=device)
    target1 = torch.rand(1, 2, 721, 1440, device=device)
    area = torch.rand(721, 1440, device=device)

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


@pytest.mark.parametrize(
    "channel_names",
    [
        # surface variables first, pressure levels grouped by variable
        ["u10m", "t2m", "t500", "t850", "z500"],
        # interleaved, as in the FourCastNet ERA5 channel layouts
        ["u10m", "t850", "t2m", "z500", "t500"],
    ],
)
def test_graphcast_loss_variable_weights_follow_channel_order(tmp_path, channel_names):
    """Each channel gets its own GraphCast weight, whatever the channel order."""
    metadata_path = tmp_path / "data.json"
    metadata_path.write_text(json.dumps({"coords": {"channel": channel_names}}))

    loss_fn = GraphCastLossFunction(
        torch.ones(2, 2), list(range(len(channel_names))), str(metadata_path), None
    )

    # t2m: 1, other surface variables: 0.1, pressure levels: level / sum of the
    # levels of the same variable
    expected = {
        "u10m": 0.1,
        "t2m": 1.0,
        "t500": 500 / 1350,
        "t850": 850 / 1350,
        "z500": 1.0,
    }
    torch.testing.assert_close(
        loss_fn.variable_weights,
        torch.tensor([expected[name] for name in channel_names]),
    )
