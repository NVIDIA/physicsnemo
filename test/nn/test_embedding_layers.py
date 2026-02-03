# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

from typing import Any

import pytest
import torch

from physicsnemo.nn import PositionalEmbedding
from test.common import validate_forward_accuracy

CONFIGS = [
    {
        "num_channels": 32,
        "max_positions": 10000,
        "endpoint": False,
        "learnable": False,
        "freq_embed_dim": None,
        "mlp_hidden_dim": None,
        "embed_fn": "cos_sin",
    },  # default
    {
        "num_channels": 128,
        "max_positions": 10000,
        "endpoint": False,
        "learnable": True,
        "freq_embed_dim": 128,
        "mlp_hidden_dim": 256,
        "embed_fn": "np_sin_cos",
    },
    {
        "num_channels": 128,
        "max_positions": 8192,
        "endpoint": True,
        "learnable": False,
        "freq_embed_dim": 128,
        "mlp_hidden_dim": 256,
        "embed_fn": "np_sin_cos",
    },
]


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("batch_size", [1, 4, 17])
def test_positional_embedding(device, config: dict[str, Any], batch_size):
    torch.manual_seed(7)
    target_device = torch.device(device)
    model = PositionalEmbedding(**config).to(target_device)
    model.eval()

    positions = torch.linspace(
        0,
        config["max_positions"] - 1,
        steps=batch_size,
        device=target_device,
        dtype=torch.float32,
    )

    def _fmt(value):
        return "none" if value is None else str(value)

    file_name = (
        "nn/module/data/"
        "positional_embedding_"
        f"c{config['num_channels']}_"
        f"max{config['max_positions']}_"
        f"endpoint{int(config['endpoint'])}_"
        f"learnable{int(config['learnable'])}_"
        f"freq{_fmt(config['freq_embed_dim'])}_"
        f"mlp{_fmt(config['mlp_hidden_dim'])}_"
        f"{config['embed_fn']}_"
        f"bs{batch_size}.pth"
    )

    # Tack this on for the test, since model is not a physicsnemo Module:
    model.device = target_device

    assert validate_forward_accuracy(
        model,
        (positions,),
        file_name=file_name,
        rtol=1e-3,
        atol=1e-3,
    )
