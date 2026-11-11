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

"""Equation, field-invariance, optimization, and serialization checks for AB-UPT."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from physicsnemo import Module
from physicsnemo.experimental.models.ab_upt import ABUPT, ABUPTContext
from physicsnemo.experimental.models.ab_upt.model import _AnchorBlock
from test.common import (
    validate_amp,
    validate_checkpoint,
    validate_combo_optims,
    validate_forward_accuracy,
)

DATA = Path(__file__).parent / "data"
DEFAULTS = dict(
    coordinate_scale=1.0,
    parameter_scale=1.0,
    query_chunk_size=4096,
    train_on_anchors=False,
    surface_channels=4,
    volume_channels=7,
    hidden_dim=192,
    num_heads=3,
    num_interaction_blocks=2,
    num_shared_layers=1,
    num_decoder_layers=6,
    mlp_ratio=4,
    pooling_radius=0.25,
    pooling_chunk_size=4096,
    parameter_channels=0,
    radius_search_backend=None,
    pooling_message_fp32=False,
)
SMALL = dict(
    hidden_dim=24,
    num_heads=3,
    num_interaction_blocks=1,
    num_shared_layers=1,
    num_decoder_layers=1,
    pooling_radius=0.8,
    pooling_chunk_size=17,
    radius_search_backend="torch",
)
CONFIGS = [
    SMALL,
    SMALL
    | dict(
        hidden_dim=48,
        num_heads=4,
        num_interaction_blocks=2,
        num_shared_layers=0,
        num_decoder_layers=2,
        parameter_channels=2,
        surface_channels=2,
        volume_channels=4,
    ),
]


def inputs(device="cpu", dtype=torch.float32):
    g = torch.linspace(-0.7, 0.9, 2 * 19 * 3, device=device, dtype=dtype).reshape(
        2, 19, 3
    )
    supernodes = g[:, ::4].contiguous()
    s = g[:, 1::3].contiguous()
    v = torch.linspace(-0.9, 0.8, 2 * 7 * 3, device=device, dtype=dtype).reshape(
        2, 7, 3
    )
    return g, supernodes, s, v


@pytest.fixture(autouse=True)
def precise_small_tests():
    precision = torch.get_float32_matmul_precision()
    threads = torch.get_num_threads()
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    yield
    torch.set_float32_matmul_precision(precision)
    torch.set_num_threads(threads)


@pytest.mark.parametrize("config", [{}, CONFIGS[1]])
def test_constructor(config):
    model = ABUPT(**config)
    for name, value in (DEFAULTS | config).items():
        assert getattr(model, name) == value
    assert isinstance(model, Module)
    assert (
        len(model.shared_blocks)
        == 2 * model.num_interaction_blocks + model.num_shared_layers
    )
    assert (
        len(model.surface_decoder)
        == len(model.volume_decoder)
        == model.num_decoder_layers
    )
    assert all(a is not b for a, b in zip(model.surface_decoder, model.volume_decoder))
    assert model.meta.amp_cpu and model.meta.amp_gpu and model.meta.auto_grad
    assert not model.meta.jit and not model.meta.cuda_graphs


@pytest.mark.parametrize("index", [0, 1])
def test_frozen_reference_and_checkpoint(index, device):
    reference = torch.load(DATA / f"ab_upt_{index}_methods.pth", weights_only=True)
    model = (
        Module.from_checkpoint(str(DATA / f"ab_upt_{index}.mdlus")).to(device).eval()
    )
    for name, value in (DEFAULTS | CONFIGS[index]).items():
        assert getattr(model, name) == value
    args = tuple(t.to(device) if t is not None else None for t in reference["inputs"])
    assert validate_forward_accuracy(
        model,
        args,
        rtol=2e-5,
        atol=2e-6,
        file_name=f"experimental/models/ab_upt/data/ab_upt_{index}_output.pth",
    )
    geometry = model.encode_geometry(*args[:2])
    context = model.encode(*args[:4], parameters=args[6])
    torch.testing.assert_close(
        geometry, reference["geometry"].to(device), rtol=2e-5, atol=2e-6
    )
    for field in ("surface", "volume"):
        actual = model.decode_queries(
            args[4 if field == "surface" else 5], context, field
        )
        torch.testing.assert_close(
            actual, reference[field].to(device), rtol=2e-5, atol=2e-6
        )
        for (k, v), (rk, rv) in zip(
            getattr(context, field), reference[field + "_context"]
        ):
            torch.testing.assert_close(k, rk.to(device), rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(v, rv.to(device), rtol=2e-5, atol=2e-6)
    vort = model.decode_vorticity(args[5], context, step=0.01)
    torch.testing.assert_close(
        vort, reference["vorticity"].to(device), rtol=1e-3, atol=2e-4
    )


@pytest.mark.parametrize("pooling_message_fp32", [False, True])
def test_checkpoint_roundtrip(tmp_path, monkeypatch, pooling_message_fp32):
    monkeypatch.chdir(tmp_path)
    config = SMALL | {"pooling_message_fp32": pooling_message_fp32}
    assert validate_checkpoint(ABUPT(**config), ABUPT(**config), inputs())
    ABUPT(**config).save("precision.mdlus")
    restored = Module.from_checkpoint("precision.mdlus")
    assert restored.pooling_message_fp32 is pooling_message_fp32


@pytest.mark.parametrize("field", ["surface", "volume"])
@pytest.mark.parametrize("interactions", [0, 1])
def test_single_branch_context_gradients_and_checkpoint(
    field, interactions, device, tmp_path
):
    inactive = "volume" if field == "surface" else "surface"
    model = ABUPT(
        **(SMALL | {f"{inactive}_channels": 0, "num_interaction_blocks": interactions})
    ).to(device)
    g, sp, s, v = inputs(device)
    anchors = s if field == "surface" else v
    kwargs = {f"{field}_anchors": anchors}
    assert getattr(model, f"{inactive}_embedding") is None
    assert not any(name.startswith(inactive) for name, _ in model.named_parameters())
    assert not any(model.cross_branch)
    context = model.encode(g, sp, **kwargs)
    assert getattr(context, inactive) == ()
    assert getattr(context, f"{inactive}_predictions") is None
    queries = torch.rand(2, 9, 3, device=device)
    full = model.decode_queries(queries, context, field)
    chunks = torch.cat(
        [model.decode_queries(q, context, field) for q in queries.split(4, dim=1)],
        dim=1,
    )
    torch.testing.assert_close(full, chunks)
    torch.testing.assert_close(
        model.decode_queries(anchors, context, field),
        getattr(context, f"{field}_predictions"),
    )
    full.square().mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    path = tmp_path / "single.mdlus"
    model.save(str(path))
    restored = Module.from_checkpoint(str(path)).to(device)
    torch.testing.assert_close(
        restored(g, sp, **kwargs, **{f"{field}_queries": queries})[
            0 if field == "surface" else 1
        ],
        full,
    )
    with pytest.raises(ValueError, match="required"):
        model.encode(g, sp)
    with pytest.raises(ValueError, match="disabled"):
        model.encode(g, sp, s, v)
    with pytest.raises(ValueError, match="disabled"):
        model.decode_queries(queries, context, inactive)


def test_conditioning_and_joint_parameter_gradients(device):
    model = ABUPT(**(SMALL | {"parameter_channels": 2})).to(device)
    conditioning = torch.randn(2, 2, device=device, requires_grad=True)
    result = model(*inputs(device), parameters=conditioning)
    sum(x.square().mean() for x in result).backward()
    assert conditioning.grad is not None and conditioning.grad.abs().sum() > 0
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


@pytest.mark.parametrize("field", ["surface", "volume"])
def test_query_independence_chunks_permutations_and_gradients(field, device):
    torch.manual_seed(9)
    model = ABUPT(**SMALL).to(device).eval()
    args = inputs(device)
    context = model.encode(*args)
    q = torch.randn(2, 13, 3, device=device, requires_grad=True)
    full = model.decode_queries(q, context, field)
    chunked = torch.cat(
        [model.decode_queries(c, context, field) for c in q.split(4, dim=1)], dim=1
    )
    torch.testing.assert_close(full, chunked, rtol=2e-5, atol=2e-6)
    order = torch.randperm(q.shape[1], device=device)
    torch.testing.assert_close(
        full[:, order],
        model.decode_queries(q[:, order], context, field),
        rtol=2e-5,
        atol=2e-6,
    )
    short = model.decode_queries(q[:, :2], context, field)
    torch.testing.assert_close(full[:, :2], short, rtol=2e-5, atol=2e-6)
    grad = torch.autograd.grad(full[:, 0].sum(), q, retain_graph=True)[0]
    assert grad[:, 0].abs().sum() > 0
    assert torch.count_nonzero(grad[:, 1:]) == 0
    full_grad = torch.autograd.grad(
        full.square().sum(), model.surface_embedding.layers[0].weight, retain_graph=True
    )[0]
    chunk_grad = torch.autograd.grad(
        chunked.square().sum(), model.surface_embedding.layers[0].weight
    )[0]
    torch.testing.assert_close(full_grad, chunk_grad, rtol=1e-4, atol=2e-5)
    anchor_index = 2 if field == "surface" else 3
    torch.testing.assert_close(
        model.decode_queries(args[anchor_index], context, field),
        getattr(context, field + "_predictions"),
        rtol=2e-5,
        atol=2e-6,
    )
    assert model.decode_queries(q[:, :0], context, field).shape[1] == 0


def test_batch_separation_and_anchor_permutation():
    torch.manual_seed(2)
    model = ABUPT(**SMALL).eval()
    args = inputs()
    q = torch.rand(2, 9, 3)
    both = model(*args, q, q)
    for b in range(2):
        single = model(*(x[b : b + 1] for x in args), q[b : b + 1], q[b : b + 1])
        for a, e in zip(single, both):
            torch.testing.assert_close(a, e[b : b + 1], rtol=2e-5, atol=2e-6)
    permuted = model(
        args[0][:, torch.randperm(19)],
        args[1].flip(1),
        args[2].flip(1),
        args[3].flip(1),
        q,
        q,
    )
    for a, e in zip(permuted, both):
        torch.testing.assert_close(a, e, rtol=2e-5, atol=2e-6)


def test_cross_branch_context_couples_predictions():
    model = ABUPT(**SMALL)
    g, sp, s, v = inputs()
    v = v.clone().requires_grad_()
    q = torch.rand(2, 3, 3)
    surface, _ = model(g, sp, s, v, q, q)
    grad = torch.autograd.grad(surface.sum(), v)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_anchor_block_against_explicit_softmax_equation():
    torch.manual_seed(11)
    block = _AnchorBlock(24, 3, 2).double()
    anchors = torch.randn(2, 5, 24, dtype=torch.float64)
    queries = torch.randn(2, 8, 24, dtype=torch.float64)
    apos = torch.randn(2, 5, 3, dtype=torch.float64)
    qpos = torch.randn(2, 8, 3, dtype=torch.float64)
    model = ABUPT(**SMALL).double()
    atables, qtables = model._rope_tables(apos), model._rope_tables(qpos)
    kv = block._key_value(anchors, atables)
    result = block(queries, qtables, kv)
    q = block.query(block.norm(queries)).reshape(2, 8, 3, 8).transpose(1, 2)
    q = block._rotate(q, qtables)
    weights = (q @ kv[0].transpose(-2, -1) / (8**0.5)).softmax(-1)
    expected = queries + block.output((weights @ kv[1]).transpose(1, 2).flatten(-2))
    expected = expected + block.mlp(block.mlp_norm(expected))
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
    # A dense attention mask excludes every query from the key/value set.
    all_values = torch.cat((anchors, queries), dim=1)
    all_pos = torch.cat((apos, qpos), dim=1)
    all_tables = model._rope_tables(all_pos)
    all_kv = block._key_value(all_values, all_tables)
    all_q = block._rotate(
        block.query(block.norm(all_values)).reshape(2, 13, 3, 8).transpose(1, 2),
        all_tables,
    )
    logits = all_q @ all_kv[0].transpose(-2, -1) / (8**0.5)
    logits[..., 5:] = float("-inf")
    masked = all_values + block.output(
        (logits.softmax(-1) @ all_kv[1]).transpose(1, 2).flatten(-2)
    )
    masked = masked + block.mlp(block.mlp_norm(masked))
    torch.testing.assert_close(result, masked[:, 5:], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rotary_mixed_precision_matches_float32_complex_rotation(dtype, device):
    """RoPE rounds the completed rotation, not its trigonometric factors."""
    model = ABUPT().to(device)
    block = model.geometry_block
    features = (
        torch.linspace(-4, 4, 2 * 3 * 5 * 64, device=device)
        .reshape(2, 3, 5, 64)
        .to(dtype)
        .requires_grad_()
    )
    positions = torch.linspace(-1000, 1000, 2 * 5 * 3, device=device).reshape(2, 5, 3)
    # Complex multiplication supplies an independent reference for each planar
    # rotation. Four channels per head are outside the three rotary axes.
    pairs = torch.view_as_complex(features[..., :60].float().reshape(2, 3, 5, 3, 10, 2))
    phase = positions[:, None, :, :, None] * model.rotary_frequencies
    reference = pairs * torch.complex(phase.cos(), phase.sin())
    reference = torch.cat(
        (torch.view_as_real(reference).flatten(-3).to(dtype), features[..., 60:]),
        dim=-1,
    )
    with torch.autocast(device_type=torch.device(device).type, dtype=dtype):
        result = block._rotate(features, model._rope_tables(positions))
    assert result.dtype == dtype
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(result.float().square().mean(), features)[0]
    expected_grad = torch.autograd.grad(reference.float().square().mean(), features)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("pooling_message_fp32", [False, True])
def test_pooling_chunk_independence_empty_neighbors_and_coordinate_gradients(
    pooling_message_fp32,
):
    model = ABUPT(**(SMALL | {"pooling_message_fp32": pooling_message_fp32})).double()
    args = inputs(dtype=torch.float64)
    g = args[0].clone().requires_grad_()
    # Cross both the 128-query and 17-point tile boundaries, with nonuniform
    # neighborhoods so an incorrect tile offset cannot accidentally agree.
    supernodes = torch.linspace(-1.2, 1.3, 2 * 200 * 3, dtype=torch.float64)
    supernodes = supernodes.reshape(2, 200, 3).requires_grad_()
    delta = g[:, None] - supernodes[:, :, None]
    distance = delta.norm(dim=-1, keepdim=True)
    mask = distance < model.pooling_radius
    messages = model.message(torch.cat((delta, distance), dim=-1))
    mean = (messages * mask).sum(2) / mask.sum(2).clamp_min(1)
    dense = model.pool_projection(
        torch.cat((mean, model._embed_positions(supernodes)), dim=-1)
    )
    torch.testing.assert_close(
        model._pool(g, supernodes), dense, rtol=1e-12, atol=1e-12
    )
    variables = (g, supernodes, *model.message.parameters())
    result = model.encode_geometry(g, supernodes)
    gradients = torch.autograd.grad(result.square().mean(), variables)
    model.pooling_chunk_size = 4096
    reference = model.encode_geometry(g, supernodes)
    torch.testing.assert_close(result, reference, rtol=1e-12, atol=1e-12)
    reference_gradients = torch.autograd.grad(reference.square().mean(), variables)
    for actual, expected in zip(gradients, reference_gradients):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    assert torch.isfinite(gradients[0]).all() and gradients[0].abs().sum() > 0
    distant = model.encode_geometry(g, args[1] + 100)
    assert torch.isfinite(distant).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp32_messages_match_full_precision_pool_and_gradients(dtype, device):
    """Autocast and chunk boundaries must preserve the FP32 local operator."""
    torch.manual_seed(42)
    model = ABUPT(**(SMALL | {"pooling_message_fp32": True})).to(device)
    # Isolate local messages from the projection, which intentionally remains
    # under autocast. Compare to an independent dense radius/mean expression.
    model.pool_projection = torch.nn.Identity()
    geometry, supernodes, *_ = inputs(device)
    geometry = geometry.detach().requires_grad_()
    supernodes = supernodes.detach().requires_grad_()
    delta = geometry[:, None] - supernodes[:, :, None]
    distance = delta.norm(dim=-1, keepdim=True)
    messages = model.message(torch.cat((delta, distance), dim=-1))
    mask = distance < model.pooling_radius
    reference = (messages * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
    variables = (geometry, supernodes, *model.message.parameters())
    expected_grads = torch.autograd.grad(reference.square().sum(), variables)
    seen = []
    hook = model.message.register_forward_hook(
        lambda module, args, result: seen.append(result.dtype)
    )
    try:
        for chunk in (17, 4096):
            model.pooling_chunk_size = chunk
            with torch.autocast(torch.device(device).type, dtype=dtype):
                result = model._pool(geometry, supernodes)[..., : model.hidden_dim]
            assert result.dtype == torch.float32
            torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-6)
            gradients = torch.autograd.grad(result.square().sum(), variables)
            for actual, expected in zip(gradients, expected_grads):
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    finally:
        hook.remove()
    assert seen and set(seen) == {torch.float32}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp32_messages_reject_reduced_precision_weights(dtype):
    model = ABUPT(**(SMALL | {"pooling_message_fp32": True})).to(dtype)
    with pytest.raises(ValueError, match="requires float32 message weights"):
        model._pool(*inputs(dtype=dtype)[:2])


@pytest.mark.parametrize(
    "dtype,n_points,chunk_size,message_value",
    [(torch.bfloat16, 1024, 1, 0.01), (torch.float16, 65536, 4096, 0.125)],
)
def test_reduced_precision_pooling_preserves_dense_neighborhood_mean(
    dtype, n_points, chunk_size, message_value, device
):
    """Identical messages retain their mean across dense low-precision pools."""
    model = ABUPT(**(SMALL | {"pooling_chunk_size": chunk_size})).to(device, dtype)
    # A known constant message gives an analytic mean, independently of model
    # weights and reference fixtures. Expose the aggregate before projection.
    model.message = torch.nn.Linear(4, model.hidden_dim).to(device, dtype)
    with torch.no_grad():
        model.message.weight.zero_()
        model.message.bias.fill_(message_value)
    model.pool_projection = torch.nn.Identity()
    geometry = torch.zeros(1, n_points, 3, device=device, dtype=dtype)
    supernodes = geometry[:, :1]
    result = model._pool(geometry, supernodes)[..., : model.hidden_dim]
    expected = model.message.bias.reshape(1, 1, -1)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.dtype == dtype


def test_double_precision_torch_pooling_retains_coordinate_separation(device):
    """Float32 rounding must not create edges in an explicit float64 search."""
    model = (
        ABUPT(**(SMALL | {"pooling_radius": 0.25, "radius_search_backend": None}))
        .to(device)
        .double()
    )
    model.message = torch.nn.Linear(4, model.hidden_dim).to(device).double()
    with torch.no_grad():
        model.message.weight.zero_()
        model.message.bias.fill_(1)
    model.pool_projection = torch.nn.Identity()
    # Both x-coordinates round to 1e8 in float32, but their true separation
    # is twice the requested radius. No message may reach this supernode.
    geometry = torch.tensor([[[1e8 + 0.5, 0, 0]]], device=device, dtype=torch.float64)
    supernodes = torch.tensor([[[1e8, 0, 0]]], device=device, dtype=torch.float64)
    result = model._pool(geometry, supernodes)[..., : model.hidden_dim]
    torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)


def test_vorticity_scaling_against_analytic_polynomial(monkeypatch):
    model = ABUPT(**SMALL).double()
    q = torch.randn(2, 6, 3, dtype=torch.float64)

    def polynomial(queries, context, field):
        x, y, z = queries.unbind(-1)
        return torch.stack((x * 0, y * y, z * z, x * x, x * 0, x * 0, x * 0), dim=-1)

    monkeypatch.setattr(model, "decode_queries", polynomial)
    result = model.decode_vorticity(
        q,
        ABUPTContext((), (), None, None, None),
        step=0.01,
        coordinate_scale=(2, 3, 4),
        velocity_scale=(5, 6, 7),
    )
    x, y, z = q.unbind(-1)
    expected = torch.stack((-48 * z, -28 * x, -30 * y), dim=-1)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-11)


@pytest.mark.parametrize("amp_dtype", [None, torch.float16, torch.bfloat16])
def test_model_curl_agrees_with_autograd_and_has_parameter_gradients(amp_dtype, device):
    torch.manual_seed(42)
    dtype = torch.float64 if amp_dtype is None else torch.float32
    model = ABUPT(**(SMALL | {"parameter_channels": 2})).to(device, dtype)
    args = inputs(device, dtype)
    parameters = torch.randn(2, 2, device=device, dtype=dtype, requires_grad=True)
    device_type = torch.device(device).type
    with torch.autocast(device_type, dtype=amp_dtype, enabled=amp_dtype is not None):
        ctx = model.encode(*args, parameters=parameters)
    q = torch.randn(2, 4, 3, device=device, dtype=dtype, requires_grad=True)
    # Differentiate the same (possibly quantized) context in full precision.
    reference_ctx = replace(
        ctx,
        volume=tuple((k.to(dtype), v.to(dtype)) for k, v in ctx.volume),
        conditioning=ctx.conditioning.to(dtype),
    )
    pred = model.decode_queries(q, reference_ctx, "volume")[..., 1:4]
    jac = [
        torch.autograd.grad(pred[..., i].sum(), q, retain_graph=True)[0]
        for i in range(3)
    ]
    exact = torch.stack(
        (
            jac[2][..., 1] - jac[1][..., 2],
            jac[0][..., 2] - jac[2][..., 0],
            jac[1][..., 0] - jac[0][..., 1],
        ),
        dim=-1,
    )
    with torch.autocast(device_type, dtype=amp_dtype, enabled=amp_dtype is not None):
        approx = model.decode_vorticity(
            q, ctx, step=1e-4 if amp_dtype is None else 1e-3
        )
    assert approx.dtype == dtype
    torch.testing.assert_close(
        approx,
        exact,
        rtol=1e-5 if amp_dtype is None else 1e-2,
        atol=1e-7 if amp_dtype is None else 3e-4,
    )
    approx.square().mean().backward()
    assert model.volume_head.weight.grad.abs().sum() > 0
    assert parameters.grad.abs().sum() > 0
    assert next(model.message.parameters()).grad.abs().sum() > 0


@pytest.mark.parametrize("helper", [validate_amp, validate_combo_optims])
@pytest.mark.parametrize("pooling_message_fp32", [False, True])
def test_supported_optimizations(helper, pooling_message_fp32, device):
    config = SMALL | {"pooling_message_fp32": pooling_message_fp32}
    assert helper(ABUPT(**config).to(device), inputs(device), iterations=2)


@pytest.mark.parametrize(
    "config",
    [
        dict(hidden_dim=25),
        dict(num_heads=0),
        dict(num_interaction_blocks=-1),
        dict(surface_channels=0, volume_channels=0),
        dict(pooling_radius=0),
        dict(pooling_radius=float("nan")),
        dict(pooling_chunk_size=0),
        dict(pooling_message_fp32="true"),
        dict(pooling_message_fp32=1),
        dict(radius_search_backend="missing"),
        dict(parameter_channels=-1),
    ],
)
def test_bad_constructor(config):
    with pytest.raises(ValueError):
        ABUPT(**(SMALL | config))


@pytest.mark.parametrize(
    "index,bad",
    [
        (0, torch.rand(19, 3)),
        (1, torch.rand(2, 0, 3)),
        (2, torch.rand(3, 4, 3)),
        (3, torch.rand(2, 5, 2)),
    ],
)
def test_bad_point_shapes(index, bad):
    args = list(inputs())
    args[index] = bad
    with pytest.raises(ValueError):
        ABUPT(**SMALL)(*args)


def test_bad_context_and_parameters():
    model = ABUPT(**SMALL)
    args = inputs()
    ctx = model.encode(*args)
    q = torch.rand(2, 3, 3)
    with pytest.raises(ValueError, match="parameters"):
        model(*args, parameters=torch.ones(2, 2))
    with pytest.raises(ValueError, match="parameters are required"):
        ABUPT(**(SMALL | {"parameter_channels": 2}))(*args)
    with pytest.raises(ValueError, match="layer contexts"):
        model.decode_queries(q, replace(ctx, surface=ctx.surface[:-1]))
    bad = ((torch.zeros(2, 1, 3, 8), torch.zeros(2, 1, 3, 8)),) + ctx.surface[1:]
    with pytest.raises(ValueError, match="shape"):
        model.decode_queries(q, replace(ctx, surface=bad))
    with pytest.raises(ValueError, match="field"):
        model.decode_queries(q, ctx, "bad")
    with pytest.raises(ValueError, match="step"):
        model.decode_vorticity(q, ctx, step=0)
    with pytest.raises(ValueError, match="velocity_channels"):
        model.decode_vorticity(q, ctx, velocity_channels=(1, 1, 3))
    with pytest.raises(ValueError, match="coordinate_scale"):
        model.decode_vorticity(q, ctx, coordinate_scale=(1, 0, 1))
    disabled = ABUPT(**(SMALL | dict(surface_channels=0)))
    with pytest.raises(ValueError, match="disabled"):
        disabled(*args, surface_queries=q)


@pytest.mark.parametrize("pooling_message_fp32", [False, True])
def test_warp_pooling_matches_torch(pooling_message_fp32, device):
    m = (
        ABUPT(**(SMALL | {"pooling_message_fp32": pooling_message_fp32}))
        .to(device)
        .eval()
    )
    args = inputs(device)
    expected = m(*args)
    m.radius_search_backend = None  # Automatic float32 selection uses Warp.
    actual = m(*args)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=1e-4, atol=2e-5)


@pytest.mark.parametrize("width", [7, 8, 64])
def test_coordinates_encode_all_axes_equally(width):
    model = ABUPT(**(SMALL | {"hidden_dim": width, "num_heads": 1})).double()
    encoding = model._embed_positions
    origin = encoding(torch.zeros(1, 3, dtype=torch.float64))
    shifted = encoding(torch.eye(3, dtype=torch.float64))
    changes = (shifted - origin).square().sum(-1)
    assert (changes > 0).all()
    torch.testing.assert_close(changes, changes[0].expand(3), rtol=0, atol=1e-15)
    assert shifted.shape == (3, width)
    assert (shifted[..., 6 * (width // 6) :] == 0).all()


@pytest.mark.parametrize("train_on_anchors", [False, True])
def test_forward_scaling_and_chunking_match_low_level_api(device, train_on_anchors):
    """The public API preserves scaled encode/decode values and gradients."""
    model = ABUPT(
        **SMALL,
        parameter_channels=3,
        coordinate_scale=10.0,
        parameter_scale=40.0,
        query_chunk_size=3,
        train_on_anchors=train_on_anchors,
    ).to(device)
    g, sp, s, v = inputs(device)
    parameters = torch.tensor([[[40.0, 2.0, 0.0]], [[30.0, 0.0, 1.0]]], device=device)
    context = model.encode(g * 10, sp * 10, s * 10, v * 10, parameters[:, 0] / 40)
    expected = (
        (context.surface_predictions, context.volume_predictions)
        if train_on_anchors
        else (
            model.decode_queries(s * 10, context, "surface"),
            model.decode_queries(v * 10, context, "volume"),
        )
    )
    actual = model(g, sp.double(), s, v.double(), s, v, parameters.double())
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    weights = tuple(model.parameters())
    actual_grad = torch.autograd.grad(sum(t.square().mean() for t in actual), weights)
    expected_grad = torch.autograd.grad(
        sum(t.square().mean() for t in expected), weights
    )
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)
    if train_on_anchors:
        with pytest.raises(ValueError, match="coordinates, and order"):
            model(g, sp, s, v, s.flip(1), v, parameters)
        model.eval()
        queries = s[:, :2]
        result = model(g, sp, s, v, queries, v, parameters)
        torch.testing.assert_close(
            result[0], model.decode_queries(queries * 10, context, "surface")
        )


@pytest.mark.parametrize(
    "override,allowed",
    [
        ({"coordinate_scale": 10.0}, False),
        ({"pooling_radius": 0.5}, False),
        ({"num_heads": 4}, False),
        ({"num_interaction_blocks": 0, "num_shared_layers": 3}, False),
        ({"train_on_anchors": True}, False),
        ({"parameter_scale": 40.0}, False),
        (
            {
                "pooling_chunk_size": 97,
                "query_chunk_size": 3,
                "pooling_message_fp32": True,
                "radius_search_backend": None,
            },
            True,
        ),
    ],
)
def test_checkpoint_settings(tmp_path, override, allowed):
    """Loading rejects changed field semantics but permits execution tuning."""
    path = str(tmp_path / "settings.mdlus")
    ABUPT(**SMALL).save(path)
    destination = ABUPT(**(SMALL | override))
    if allowed:
        destination.load(path)
        for name, value in override.items():
            assert getattr(destination, name) == value
    else:
        with pytest.raises(ValueError, match="checkpoint settings"):
            destination.load(path)


def test_anchor_training_compiles_without_validation_graph_breaks(monkeypatch, device):
    """Compiled anchor training preserves both branches' values and gradients."""
    model = ABUPT(**SMALL, parameter_channels=3, train_on_anchors=True).to(device)
    geometry, supernodes, surface, volume = inputs(device)
    args = (
        geometry,
        supernodes,
        surface,
        volume,
        surface.clone(),
        volume.clone(),
        torch.ones(2, 3, device=device),
    )
    # Radius search has dynamic-shape graph breaks. Freeze only its pooled output
    # so fullgraph still checks validation and attention in the real model.
    pooled = model._pool(geometry, supernodes).detach()
    monkeypatch.setattr(model, "_pool", lambda geometry, supernodes: pooled)
    expected = model(*args)
    actual = torch.compile(model, backend="aot_eager", fullgraph=True)(*args)
    for actual_value, expected_value in zip(actual, expected):
        torch.testing.assert_close(actual_value, expected_value)
    weights = (model.surface_head.weight, model.volume_head.weight)
    expected_grads = torch.autograd.grad(
        sum(t.square().mean() for t in expected), weights
    )
    actual_grads = torch.autograd.grad(sum(t.square().mean() for t in actual), weights)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
