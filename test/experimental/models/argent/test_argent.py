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
        assert hasattr(model, "branch_net")
    else:
        model = ArGEnTDeepONet(
            dim_op=2,
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
        assert model.dim_op == 2
        assert trunk.dim_out == 3 * 32  # dim_out * dim_emb basis features
        assert (trunk.dim_query, trunk.dim_kv, trunk.dim_pos) == (3, 4, 2)
        assert trunk.num_heads == 2
        assert trunk.depth == 3
        assert trunk.max_freq == 200.0
        assert trunk.axis_slices == [8, 16]
        assert trunk.use_mfnn and trunk.use_relu_mlp

    # Default axis split rounds each slice down to an even width.
    trunk = ArGEnT(dim_query=3, dim_kv=3, dim_pos=3, dim_emb=64)
    assert trunk.axis_slices == [20, 20, 20]


@pytest.mark.parametrize(
    "cls, kwargs, match",
    [
        (ArGEnT, dict(dim_pos=0), "dim_pos"),
        (ArGEnT, dict(dim_query=3, dim_kv=3, dim_pos=3, dim_emb=4), "axis_slices"),
        (ArGEnTDeepONet, dict(dim_op=0), "dim_op"),
        (ArGEnTDeepONet, dict(dim_op=-3), "dim_op"),
    ],
    ids=["dim_pos_zero", "dim_emb_too_small", "dim_op_zero", "dim_op_negative"],
)
def test_argent_constructor_rejects_invalid_args(cls, kwargs, match):
    """Invalid sizes raise ValueError instead of building a degenerate model."""
    with pytest.raises(ValueError, match=match):
        cls(**kwargs)


def test_argent_rope_angle():
    """Layer d rotates coordinate x by x * max_freq**2 / 2**d * 10000**(-2k/axis_dim)."""
    max_freq, axis_dim = 200.0, 16
    trunk = ArGEnT(dim_emb=32, depth=2, max_freq=max_freq)
    x = torch.tensor([[1.0e-3, -2.5e-3]])
    k = torch.arange(axis_dim // 2, dtype=torch.float64)
    for d, (attn, _) in enumerate(trunk.transformer.layers):
        cos, sin = attn._rope_tables(x, axis_dim)  # (1, 1, 2, axis_dim)
        angle = (
            x.double()[..., None] * max_freq**2 / 2**d * 10000.0 ** (-2 * k / axis_dim)
        ).repeat_interleave(2, dim=-1)
        torch.testing.assert_close(cos[:, 0].double(), angle.cos(), atol=1e-5, rtol=0)
        torch.testing.assert_close(sin[:, 0].double(), angle.sin(), atol=1e-5, rtol=0)


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


@pytest.mark.parametrize("junk", ["random", "nan", "inf"])
def test_argent_padding_kv_ignores_padded_tokens(device, junk):
    """Padded key/value tokens, finite or not, do not affect the output."""
    # Seeded weights give a nonzero LayerNorm bias, so masking before the norm
    # would change the output.
    model = _seed_params(ArGEnTDeepONet(**_LID_DRIVEN)).to(device)
    x_query, x_kv, op_params = _inputs(3, 3, 1, device)
    op_params = op_params.to(device)

    if junk == "random":
        pad = 100.0 * torch.randn(B, 16, 3)
    else:
        pad = torch.full((B, 16, 3), float(junk))
    x_kv_padded = torch.cat([x_kv, pad.to(device)], dim=1)
    padding_kv = torch.ones(B, NKV + 16, dtype=torch.bool, device=device)
    padding_kv[:, NKV:] = False

    out = model(x_query, x_kv, op_params)
    out_padded = model(x_query, x_kv_padded, op_params, padding_kv=padding_kv)
    torch.testing.assert_close(out_padded, out, rtol=1e-5, atol=1e-5)

    # Nonzero float masks behave like the boolean mask.
    out_float = model(
        x_query, x_kv_padded, op_params, padding_kv=2.0 * padding_kv.float()
    )
    torch.testing.assert_close(out_float, out, rtol=1e-5, atol=1e-5)

    # Non-finite padding must not leak into the gradients either.
    out_padded.sum().backward()
    for param in model.parameters():
        assert torch.isfinite(param.grad).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_argent_padding_kv_reduced_precision(device, dtype):
    """Padding works when the model and inputs are cast to reduced precision.

    Casting the inputs rounds the coordinates before the RoPE scaling
    (``max_freq**2 = 40000`` here), which costs a few percent of accuracy, so
    only the dtype and finiteness are checked. Accuracy is checked under
    autocast in :func:`test_argent_autocast_accuracy`.
    """
    model = ArGEnTDeepONet(**_LID_DRIVEN).to(device=device, dtype=dtype)
    x_query, x_kv, op_params = _inputs(3, 3, 1, device)
    padding_kv = torch.ones(B, NKV, dtype=torch.bool, device=device)
    padding_kv[1, NKV // 2 :] = False

    out = model(
        x_query.to(dtype),
        x_kv.to(dtype),
        op_params.to(device=device, dtype=dtype),
        padding_kv=padding_kv,
    )
    assert out.dtype == dtype
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "dtype, rtol",
    [(torch.bfloat16, 1e-2), (torch.float16, 2.5e-3)],
    ids=["bf16", "fp16"],
)
def test_argent_autocast_accuracy(device, dtype, rtol):
    """Under autocast the output stays close to fp32 (relative L2 error)."""
    model = _seed_params(ArGEnTDeepONet(**_LID_DRIVEN)).to(device)
    x_query, x_kv, op_params = _inputs(3, 3, 1, device)
    op_params = op_params.to(device)
    padding_kv = torch.ones(B, NKV, dtype=torch.bool, device=device)
    padding_kv[1, NKV // 2 :] = False

    ref = model(x_query, x_kv, op_params, padding_kv=padding_kv)
    with torch.autocast(device_type=torch.device(device).type, dtype=dtype):
        out = model(x_query, x_kv, op_params, padding_kv=padding_kv)
    rel_err = (out.float() - ref).norm() / ref.norm()
    assert rel_err < rtol, f"relative L2 error {rel_err:.2e} exceeds {rtol:.1e}"


def test_argent_compile_matches_eager(device):
    """torch.compile(fullgraph=True) matches eager, with padding."""
    model = _seed_params(ArGEnTDeepONet(**_LID_DRIVEN)).to(device)
    x_query, x_kv, op_params = _inputs(3, 3, 1, device)
    op_params = op_params.to(device)
    padding_kv = torch.ones(B, NKV, dtype=torch.bool, device=device)
    padding_kv[1, NKV // 2 :] = False

    eager = model(x_query, x_kv, op_params, padding_kv=padding_kv)
    compiled = torch.compile(model, fullgraph=True)(
        x_query, x_kv, op_params, padding_kv=padding_kv
    )
    torch.testing.assert_close(compiled, eager, rtol=1e-5, atol=1e-5)


def test_argent_geometry_reuse(device):
    """Query chunks decoded against one encoded geometry match the full forward."""
    model = _seed_params(ArGEnTDeepONet(**_LID_DRIVEN)).to(device)
    x_query, x_kv, op_params = _inputs(3, 3, 1, device)
    op_params = op_params.to(device)
    padding_kv = torch.ones(B, NKV, dtype=torch.bool, device=device)
    padding_kv[1, NKV // 2 :] = False

    out = model(x_query, x_kv, op_params, padding_kv=padding_kv)
    geometry = model.encode_geometry(x_kv, padding_kv=padding_kv)
    chunks = [
        model.decode_queries(chunk, geometry, op_params)
        for chunk in x_query.split(NQ // 4, dim=1)
    ]
    torch.testing.assert_close(torch.cat(chunks, dim=1), out, rtol=1e-5, atol=1e-5)

    # op_params of shape (B, dim_op) is accepted as well as (B, 1, dim_op).
    out_2d = model(x_query, x_kv, op_params[:, 0], padding_kv=padding_kv)
    torch.testing.assert_close(out_2d, out)


def test_argent_padding_kv_shape_validation(device):
    """A padding mask with the wrong layout is rejected."""
    model = ArGEnT(dim_query=3, dim_kv=3, dim_pos=2, dim_emb=32).to(device)
    x_query, x_kv, _ = _inputs(3, 3, 0, device)
    padding_kv = torch.ones(NKV, B, dtype=torch.bool, device=device)
    with pytest.raises(ValueError, match="padding_kv"):
        model(x_query, x_kv, padding_kv=padding_kv)
