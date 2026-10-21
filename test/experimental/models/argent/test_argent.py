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

import pytest
import torch

from physicsnemo.experimental.models.argent import ArGEnT, ArGEnTDeepONet
from test.common import validate_checkpoint, validate_forward_accuracy

B, NQ, NKV = 2, 64, 48

# Small-width version of the lid-driven cavity example config.
_LID_DRIVEN = dict(
    dim_op=1,
    dim_query=3,
    dim_kv=3,
    dim_pos=2,
    dim_out=3,
    dim_emb=32,
    num_heads=2,
    depth=2,
    max_freq=200.0,
)


def _seed_params(model: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    """Fill parameters with reproducible values scaled by 1/sqrt(fan_in).

    The default init keeps the attention contribution far below the
    forward-accuracy tolerance; O(1) weights make the reference outputs
    sensitive to the attention path as well.
    """
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn(param.shape, generator=gen, dtype=param.dtype)
            param.copy_(noise / param.shape[-1] ** 0.5)
    return model


def _inputs(dim_query: int, dim_kv: int, dim_op: int, device: str):
    """Deterministic (x_query, x_kv, op_params) on ``device``."""
    gen = torch.Generator().manual_seed(1)
    x_query = torch.rand(B, NQ, dim_query, generator=gen)
    x_kv = torch.rand(B, NKV, dim_kv, generator=gen)
    op_params = torch.rand(B, 1, dim_op, generator=gen) if dim_op > 0 else None
    return x_query.to(device), x_kv.to(device), op_params


@pytest.mark.parametrize(
    "config", ["default", "custom"], ids=["with_defaults", "with_custom_args"]
)
def test_argent_deeponet_constructor(config):
    """ArGEnTDeepONet constructor and public attributes."""
    if config == "default":
        model = ArGEnTDeepONet()
        trunk = model.argent
        assert model.dim_op == 1
        assert model.dim_out == 1
        assert model.dim_emb == 128
        assert trunk.dim_out == 128  # dim_out * dim_emb basis features
        assert trunk.dim_query == trunk.dim_kv == trunk.dim_pos == 2
        assert trunk.num_heads == 4
        assert trunk.depth == 2
        assert trunk.max_freq == 64.0
        assert trunk.axis_slices == [64, 64]
        assert not trunk.use_mfnn and not trunk.use_relu_mlp
        assert trunk.attention_type == "cross"
        assert hasattr(model, "branch_net")
    else:
        model = ArGEnTDeepONet(
            dim_op=0,
            dim_query=3,
            dim_kv=4,
            dim_pos=2,
            dim_out=3,
            dim_emb=32,
            num_heads=2,
            depth=3,
            max_freq=200.0,
            use_mfnn=True,
            activation="tanh",
            use_relu_mlp=True,
            axis_slices=[8, 16],
        )
        trunk = model.argent
        assert model.dim_op == 0
        assert not hasattr(model, "branch_net")
        assert trunk.dim_out == 3  # no branch: trunk predicts dim_out directly
        assert (trunk.dim_query, trunk.dim_kv, trunk.dim_pos) == (3, 4, 2)
        assert trunk.num_heads == 2
        assert trunk.depth == 3
        assert trunk.max_freq == 200.0
        assert trunk.axis_slices == [8, 16]
        assert trunk.use_mfnn and trunk.use_relu_mlp

    with pytest.raises(NotImplementedError):
        ArGEnT(attention_type="self")


@pytest.mark.parametrize("name", ["lid_driven", "mfnn_relu", "trunk_only"])
def test_argent_forward_accuracy(device, name):
    """Forward pass against stored reference outputs."""
    if name == "lid_driven":
        model = ArGEnTDeepONet(**_LID_DRIVEN)
        x_query, x_kv, op_params = _inputs(3, 3, 1, device)
        args = (x_query, x_kv, op_params.to(device))
    elif name == "mfnn_relu":
        model = ArGEnTDeepONet(
            **{**_LID_DRIVEN, "dim_op": 2, "dim_kv": 4},
            use_mfnn=True,
            use_relu_mlp=True,
            activation="tanh",
        )
        x_query, x_kv, op_params = _inputs(3, 4, 2, device)
        args = (x_query, x_kv, op_params.to(device))
    else:
        model = ArGEnT(dim_query=3, dim_kv=3, dim_pos=2, dim_out=2, dim_emb=32)
        x_query, x_kv, _ = _inputs(3, 3, 0, device)
        padding_kv = torch.ones(B, NKV, dtype=torch.bool, device=device)
        padding_kv[1, NKV // 2 :] = False
        args = (x_query, x_kv, padding_kv)

    assert validate_forward_accuracy(
        _seed_params(model).to(device),
        args,
        file_name=f"experimental/models/argent/data/argent_{name}_output.pth",
    )


def test_argent_deeponet_checkpoint(device):
    """Save / load / from_checkpoint round trip."""
    model_1 = ArGEnTDeepONet(**_LID_DRIVEN).to(device)
    model_2 = ArGEnTDeepONet(**_LID_DRIVEN).to(device)
    x_query, x_kv, op_params = _inputs(3, 3, 1, device)
    assert validate_checkpoint(model_1, model_2, (x_query, x_kv, op_params.to(device)))


def test_argent_padding_kv_ignores_padded_tokens(device):
    """Padded key/value tokens do not affect the output."""
    model = ArGEnTDeepONet(**_LID_DRIVEN).to(device)
    x_query, x_kv, op_params = _inputs(3, 3, 1, device)
    op_params = op_params.to(device)

    junk = 100.0 * torch.randn(B, 16, 3, device=device)
    x_kv_padded = torch.cat([x_kv, junk], dim=1)
    padding_kv = torch.ones(B, NKV + 16, dtype=torch.bool, device=device)
    padding_kv[:, NKV:] = False

    out = model(x_query, x_kv, op_params)
    out_padded = model(x_query, x_kv_padded, op_params, padding_kv=padding_kv)
    torch.testing.assert_close(out_padded, out, rtol=1e-5, atol=1e-5)
