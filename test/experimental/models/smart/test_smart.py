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

"""Behavioral and numerical tests for the independent SMART implementation."""

from pathlib import Path

import pytest
import torch

from physicsnemo import Module
from physicsnemo.experimental.models.smart import SMART
from test.common import (
    validate_amp,
    validate_checkpoint,
    validate_combo_optims,
    validate_cuda_graphs,
    validate_forward_accuracy,
    validate_jit,
)

_DEFAULT_CONFIG = dict(
    surface_channels=1,
    volume_channels=3,
    parameter_channels=0,
    hidden_dim=256,
    num_heads=8,
    num_layers=6,
    num_latents=2048,
    geometry_subsample=65536,
    mlp_ratio=4,
    share_weights=True,
    eval_seed=0,
    qk_normalization=False,
    qk_temperature_max=None,
    surface_query_feature_channels=0,
    volume_query_feature_channels=0,
)


def _assert_model_attributes(model, config):
    """Check constructor state and module sharing for new and restored models."""
    expected = _DEFAULT_CONFIG | config
    assert isinstance(model, Module)
    for name, value in expected.items():
        assert getattr(model, name) == value, name
    assert len(model.geometry_attention) == expected["num_layers"]
    assert len(model.decoder) == expected["num_layers"]
    assert len(model.encoder_refinement) == expected["num_layers"] - 1
    assert all(
        (block is model.decoder[i]) == expected["share_weights"]
        for i, block in enumerate(model.encoder_refinement)
    )
    assert model.meta.amp_cpu and model.meta.amp_gpu and model.meta.auto_grad


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({}, id="defaults"),
        pytest.param(
            dict(
                surface_channels=4,
                volume_channels=5,
                parameter_channels=2,
                hidden_dim=24,
                num_heads=4,
                num_layers=2,
                num_latents=8,
                geometry_subsample=16,
                mlp_ratio=2,
                share_weights=False,
                eval_seed=7,
                qk_normalization=True,
                qk_temperature_max=4.0,
                surface_query_feature_channels=3,
                volume_query_feature_channels=4,
            ),
            id="custom",
        ),
    ],
)
def test_smart_constructor(config):
    """Verify default and custom public attributes per MOD-008a."""
    _assert_model_attributes(SMART(**config), config)


@pytest.fixture(autouse=True)
def full_precision_reference():
    """Keep reference tolerances independent of the host's TF32 defaults."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous)


def small_model(**overrides):
    config = dict(
        hidden_dim=24, num_heads=4, num_layers=2, num_latents=8, geometry_subsample=16
    )
    config.update(overrides)
    return SMART(**config)


def test_geometry_sampling_uses_full_pool_without_dense_embeddings():
    """Latents and every layer sample the supplied pool; queries never join it."""
    model = small_model(geometry_subsample=64).train()
    points = torch.arange(600, dtype=torch.float32).reshape(1, 200, 3) / 600
    queries = torch.full((1, 11, 3), 5.0)
    embedded = []
    hook = model.position.register_forward_pre_hook(
        lambda _module, args: embedded.append(args[0].detach().clone())
    )
    before = torch.random.get_rng_state().clone()
    output = model(points, queries, geometry_generator=torch.Generator().manual_seed(0))
    assert torch.equal(before, torch.random.get_rng_state())
    hook.remove()
    assert [x.shape[1] for x in embedded] == [8, 64, 64, 11]
    assert model.num_latents == 8
    for sample in embedded[:3]:
        assert (sample[..., 0] > points[0, 149, 0]).any()
        assert (sample < 1).all()
        assert torch.isin(sample[..., 0], points[..., 0]).all()
        assert sample[0, :, 0].unique().numel() == sample.shape[1]
        # Gather complete rows so all three coordinate components stay aligned.
        indices = (sample[0, :, 0] * 200).round().long()
        torch.testing.assert_close(sample, points[:, indices], rtol=0, atol=0)
    assert not torch.equal(embedded[1], embedded[2])
    repeat = model(points, queries, geometry_generator=torch.Generator().manual_seed(0))
    assert torch.equal(output[0], repeat[0])


def test_query_independence_and_chunks(device):
    torch.manual_seed(9)
    model = small_model(parameter_channels=2).to(device).eval()
    geometry = torch.randn(2, 45, 3, device=device)
    parameters = torch.randn(2, 2, device=device)
    queries = torch.randn(2, 23, 3, device=device)
    with torch.no_grad():
        contexts = model.encode_geometry(geometry, parameters)
        for field in ("surface", "volume"):
            full = model.decode_queries(queries, contexts, parameters, field=field)
            chunked = torch.cat(
                [
                    model.decode_queries(chunk, contexts, parameters, field=field)
                    for chunk in queries.split(7, dim=1)
                ],
                dim=1,
            )
            torch.testing.assert_close(chunked, full, atol=2e-6, rtol=2e-5)
            indices = torch.tensor([7, 2, 7, 0, 22], device=device)
            subset = model.decode_queries(
                queries[:, indices], contexts, parameters, field=field
            )
            torch.testing.assert_close(subset, full[:, indices], atol=2e-6, rtol=2e-5)
        pressure, velocity = model(geometry, queries, queries, parameters)
        torch.testing.assert_close(velocity, full)
        torch.testing.assert_close(
            pressure, model.decode_queries(queries, contexts, parameters)
        )


def test_batch_and_geometry_order_independence(device):
    model = small_model(num_latents=128, geometry_subsample=128).to(device).eval()
    geometry, query = (
        torch.randn(2, 20, 3, device=device),
        torch.randn(2, 13, 3, device=device),
    )
    with torch.no_grad():
        full = model(geometry, query)[0]
        permuted = model(geometry[:, torch.randperm(20, device=device)], query)[0]
        torch.testing.assert_close(full, permuted, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(
            full[:1], model(geometry[:1], query[:1])[0], atol=2e-6, rtol=2e-5
        )


@pytest.mark.parametrize("share_weights,num_layers", [(True, 1), (True, 3), (False, 3)])
def test_smart_forward_backward(device, share_weights, num_layers):
    model = small_model(
        parameter_channels=2, share_weights=share_weights, num_layers=num_layers
    ).to(device)
    geometry = torch.randn(2, 30, 3, device=device, requires_grad=True)
    queries = torch.randn(2, 9, 3, device=device, requires_grad=True)
    parameters = torch.randn(2, 2, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    pressure, velocity = model(geometry, queries, queries, parameters)
    assert pressure.shape == (2, 9, 1)
    assert velocity.shape == (2, 9, 3)
    assert all(
        (block is model.decoder[i]) == share_weights
        for i, block in enumerate(model.encoder_refinement)
    )
    (pressure.square().mean() + velocity.square().mean()).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert torch.isfinite(queries.grad).all()
    assert geometry.grad.abs().sum() > 0
    optimizer.step()
    model.eval()
    with torch.no_grad():
        a = model(geometry, queries, parameters=parameters)[0]
        b = model(geometry, queries, parameters=parameters + 2)[0]
    assert not torch.allclose(a, b)


def test_sampling_reproducibility(device):
    model = small_model().to(device).eval()
    geometry = torch.randn(2, 80, 3, device=device)
    with torch.no_grad():
        first = model.encode_geometry(geometry)
        torch.manual_seed(1729)
        second = model.encode_geometry(geometry)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        model.train()
        a = model.encode_geometry(
            geometry, generator=torch.Generator(device=device).manual_seed(4)
        )
        b = model.encode_geometry(
            geometry, generator=torch.Generator(device=device).manual_seed(4)
        )
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert not torch.allclose(a[0], model.encode_geometry(geometry)[0])


def test_empty_query_and_disabled_head(device):
    model = small_model(volume_channels=0).to(device).eval()
    geometry = torch.randn(2, 17, 3, device=device)
    queries = torch.empty(2, 0, 3, device=device)
    pressure, velocity = model(geometry, queries)
    assert pressure.shape == (2, 0, 1) and velocity is None
    with pytest.raises(ValueError, match="disabled"):
        model(geometry, volume_queries=torch.randn(2, 4, 3, device=device))


@pytest.mark.parametrize(
    "options",
    [
        dict(hidden_dim=25),
        dict(num_layers=0),
        dict(num_latents=0),
        dict(surface_channels=0, volume_channels=0),
        dict(parameter_channels=-1),
        dict(hidden_dim=True),
        dict(surface_channels=True),
        dict(eval_seed=True),
        dict(eval_seed=0.5),
        dict(eval_seed=2**64),
    ],
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        small_model(**options)


def test_invalid_inputs():
    model = small_model(parameter_channels=2)
    geometry, queries = torch.randn(2, 10, 3), torch.randn(2, 8, 3)
    with pytest.raises(ValueError, match="At least one"):
        model(geometry)
    with pytest.raises(ValueError, match="shape"):
        model(geometry, queries)
    with pytest.raises(ValueError, match="nonempty"):
        model(geometry[:, :0], queries, parameters=torch.zeros(2, 2))
    with pytest.raises(ValueError, match="shape"):
        model(geometry[..., :2], queries, parameters=torch.zeros(2, 2))
    contexts = model.encode_geometry(geometry, torch.zeros(2, 2))
    with pytest.raises(ValueError, match="context"):
        model.decode_queries(queries, contexts[:1], torch.zeros(2, 2))
    with pytest.raises(ValueError, match="shape"):
        model.decode_queries(queries[:1], contexts, torch.zeros(1, 2))


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({}, id="shared"),
        pytest.param(
            dict(share_weights=False, qk_normalization=True, qk_temperature_max=4.0),
            id="unshared-normalized",
        ),
    ],
)
def test_smart_checkpoint(device, tmp_path, monkeypatch, options):
    """Use the standard model save/load and from_checkpoint validation."""
    monkeypatch.chdir(tmp_path)
    model = small_model(parameter_channels=2, **options).to(device).eval()
    restored = small_model(parameter_channels=2, **options).to(device).eval()
    # The shared helper restores in training mode. Use a cloud smaller than
    # either sampling budget so checkpoint comparisons do not resample it.
    inputs = (
        torch.randn(2, 8, 3, device=device),
        torch.randn(2, 11, 3, device=device),
        torch.randn(2, 19, 3, device=device),
        torch.randn(2, 2, device=device),
    )
    assert validate_checkpoint(model, restored, inputs, rtol=0, atol=0)


@pytest.mark.parametrize("case", ["shared", "unshared"])
@pytest.mark.parametrize("source", ["constructor", "checkpoint"])
def test_smart_forward_accuracy(case, source, device):
    """Check frozen forward/encoder/decoder values and .mdlus loading (MOD-008b/c)."""
    root = Path(__file__).parent / "data"
    data = torch.load(root / f"{case}.pth", map_location=device, weights_only=True)
    if source == "checkpoint":
        model = Module.from_checkpoint(root / f"{case}.mdlus")
    else:
        model = SMART(**data["config"])
        model.load(root / f"{case}.mdlus")
    model = model.to(device).eval()
    _assert_model_attributes(model, data["config"])
    inputs = data["inputs"]
    arguments = tuple(
        inputs[key]
        for key in ("geometry", "surface_queries", "volume_queries", "parameters")
    )
    assert validate_forward_accuracy(
        model,
        arguments,
        atol=3e-6,
        rtol=3e-5,
        file_name=f"experimental/models/smart/data/{case}_output.pth",
    )
    expected = torch.load(
        root / f"{case}_output.pth", map_location=device, weights_only=True
    )
    with torch.no_grad():
        contexts = model.encode_geometry(inputs["geometry"], inputs["parameters"])
        torch.testing.assert_close(contexts, data["contexts"], atol=3e-6, rtol=3e-5)
        for index, field in enumerate(("surface", "volume")):
            prediction = model.decode_queries(
                inputs[f"{field}_queries"],
                data["contexts"],
                inputs["parameters"],
                field=field,
            )
            torch.testing.assert_close(
                prediction, expected[index], atol=3e-6, rtol=3e-5
            )


@pytest.mark.parametrize("optimization", ["amp", "combo", "jit", "cuda_graphs"])
def test_smart_optimizations(device, optimization):
    """Exercise the standard model optimization helpers where supported."""
    model = small_model(parameter_channels=2).to(device)
    if optimization in ("jit", "cuda_graphs") and not getattr(model.meta, optimization):
        pytest.skip(f"SMART does not advertise {optimization} support")
    inputs = (
        torch.randn(2, 40, 3, device=device),
        torch.randn(2, 13, 3, device=device),
        torch.randn(2, 19, 3, device=device),
        torch.randn(2, 2, device=device),
    )
    validators = dict(
        amp=validate_amp,
        combo=validate_combo_optims,
        jit=validate_jit,
        cuda_graphs=validate_cuda_graphs,
    )
    assert validators[optimization](model, inputs)


def test_compiled_decoder(device):
    model = small_model().to(device).eval()
    geometry, queries = (
        torch.randn(2, 40, 3, device=device),
        torch.randn(2, 15, 3, device=device),
    )
    # aot_eager exercises Dynamo graph capture and AOTAutograd on CPU and CUDA.
    compiled = torch.compile(model.decode_queries, backend="aot_eager", fullgraph=True)
    with torch.no_grad():
        contexts = model.encode_geometry(geometry)
        torch.testing.assert_close(
            compiled(queries, contexts), model.decode_queries(queries, contexts)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_autocast(dtype):
    model = small_model(parameter_channels=2).cuda()
    inputs = (
        torch.randn(2, 40, 3, device="cuda"),
        torch.randn(2, 13, 3, device="cuda"),
        torch.randn(2, 19, 3, device="cuda"),
        torch.randn(2, 2, device="cuda"),
    )
    with torch.autocast("cuda", dtype=dtype):
        predictions = model(*inputs)
    loss = sum(p.float().square().mean() for p in predictions)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_autocast_preserves_small_residual_updates(device):
    """Deep mixed-precision decoders must accumulate updates below one bf16 ULP."""
    model = small_model(num_layers=16, volume_channels=0).to(device).eval()
    model.surface_head = torch.nn.Identity()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.position.project.linear.bias.fill_(1.0)
        for block in model.decoder:
            block.cross_attention.attention.out_proj.bias.fill_(0.002)
            block.output.linear.bias.fill_(0.002)
        queries = torch.zeros(1, 5, 3, device=device)
        contexts = tuple(torch.zeros(1, 8, 24, device=device) for _ in range(16))
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            actual = model.decode_queries(queries, contexts)
        torch.testing.assert_close(
            actual.float(), torch.full_like(actual.float(), 1.064), atol=2e-4, rtol=0
        )


def test_qk_normalization_bounds_scores(monkeypatch):
    """Large projected Q/K must stay bounded and query-independent."""
    import math

    from torch.nn import functional as F

    model = small_model(qk_normalization=True).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                module.in_proj_weight[: 2 * model.hidden_dim].mul_(1000)
    original = F.scaled_dot_product_attention
    observed = []

    def check_scores(query, key, value, *args, **kwargs):
        scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
        observed.append(float(scores.detach().abs().max()))
        assert observed[-1] <= math.sqrt(query.shape[-1]) + 1e-5
        return original(query, key, value, *args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", check_scores)
    geometry, query = torch.randn(1, 32, 3), torch.randn(1, 13, 3)
    with torch.no_grad():
        expected = model(geometry, query)[0]
        single = model(geometry, query[:, 3:4])[0]
    torch.testing.assert_close(single, expected[:, 3:4], atol=1e-6, rtol=1e-5)
    assert observed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA precision check")
def test_normalized_attention_large_projection_gradients():
    """FP16 backward must match fp32 even when Q/K projection weights are large."""
    from physicsnemo.experimental.models.smart.model import _CrossAttention

    torch.manual_seed(4)
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        reference = _CrossAttention(32, 4, qk_normalization=True).cuda()
        with torch.no_grad():
            reference.attention.in_proj_weight[:64].mul_(100)
        mixed = _CrossAttention(32, 4, qk_normalization=True).cuda()
        mixed.load_state_dict(reference.state_dict())
        query = torch.randn(2, 23, 32, device="cuda")
        context = torch.randn(2, 37, 32, device="cuda")
        reference_output = reference(query, context)
        with torch.autocast("cuda", dtype=torch.float16):
            actual = mixed(query, context)
        reference_output.square().mean().backward()
        actual.square().mean().backward()
        assert (actual - reference_output).norm() / reference_output.norm() < 0.01
        reference_gradient = torch.cat(
            [p.grad.flatten() for p in reference.parameters()]
        )
        actual_gradient = torch.cat([p.grad.flatten() for p in mixed.parameters()])
        assert torch.isfinite(actual_gradient).all()
        assert (
            actual_gradient - reference_gradient
        ).norm() / reference_gradient.norm() < 0.02
    finally:
        torch.set_float32_matmul_precision(previous_precision)


@pytest.mark.parametrize("invalid_kind", ["integer", "wrong_device"])
def test_public_tensor_dtype_and_device_validation(invalid_kind):
    """Reject incompatible inputs at the API boundary before attention runs."""
    model = small_model(parameter_channels=2).eval()
    geometry, queries = torch.randn(2, 20, 3), torch.randn(2, 7, 3)
    conditions = torch.randn(2, 2)

    def invalid(tensor):
        return tensor.long() if invalid_kind == "integer" else tensor.to("meta")

    with pytest.raises(ValueError, match="parameters.*floating-point"):
        model.encode_geometry(geometry, invalid(conditions))
    contexts = model.encode_geometry(geometry, conditions)
    with pytest.raises(ValueError, match="parameters.*floating-point"):
        model.decode_queries(queries, contexts, invalid(conditions))
    with pytest.raises(ValueError, match="Contexts.*floating-point"):
        model.decode_queries(queries, tuple(invalid(c) for c in contexts), conditions)
    if invalid_kind == "wrong_device":
        with pytest.raises(ValueError, match="Queries.*geometry device"):
            model(geometry, invalid(queries), parameters=conditions)


@pytest.mark.parametrize("precision", ["float16", "bfloat16"])
def test_legacy_attention_override_is_not_silently_discarded(tmp_path, precision):
    """Unsupported legacy arithmetic must fail instead of changing predictions."""
    model = small_model()
    model.__model_checkpoint_version__ = "0.1.0"
    model._args["__args__"].update(
        attention_precision=precision, attention_precision_scope="sdpa"
    )
    path = tmp_path / "legacy.mdlus"
    model.save(path)
    with pytest.raises(ValueError, match="earlier SMART implementation"):
        Module.from_checkpoint(path)
