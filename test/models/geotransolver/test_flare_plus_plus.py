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

r"""Tests for the GeoTransolver-specific FLARE++ backend."""

from pathlib import Path

import pytest
import torch

from physicsnemo.core.module import Module
from physicsnemo.models.geotransolver import GeoTransolver
from physicsnemo.models.geotransolver.flare_plus_plus import (
    _FLAREPlusPlusAttention,
    _FLAREPlusPlusBlock,
)
from physicsnemo.nn import FLAREPlusPlus

DATA_DIR = Path(__file__).parent / "data"


def _make_model(**overrides) -> GeoTransolver:
    """Construct the small FLARE++ GeoTransolver used by backend tests."""
    config = {
        "functional_dim": 8,
        "out_dim": 3,
        "geometry_dim": 3,
        "global_dim": 4,
        "n_layers": 2,
        "n_hidden": 32,
        "dropout": 0.0,
        "n_head": 4,
        "mlp_ratio": 2,
        "slice_num": 6,
        "use_te": False,
        "plus": False,
        "attention_type": "GALE_FPP",
    }
    config.update(overrides)
    return GeoTransolver(**config)


def _make_inputs(device):
    """Create deterministic local, geometry, and global inputs."""
    generator = torch.Generator(device="cpu").manual_seed(1234)
    return (
        torch.randn(2, 23, 8, generator=generator).to(device),
        torch.randn(2, 23, 3, generator=generator).to(device),
        torch.randn(2, 2, 4, generator=generator).to(device),
    )


def _load_or_create_output_reference(
    file_name: str, output: torch.Tensor
) -> torch.Tensor:
    """Load a local golden output, creating it from this test when absent."""
    reference_path = DATA_DIR / file_name
    if not reference_path.exists():
        torch.save({"output": output.detach().cpu()}, reference_path)
    reference = torch.load(reference_path, weights_only=True)
    return next(iter(reference.values())).to(output.device)


def test_constructor_uses_model_specific_backend():
    """The explicit GALE_FPP selection installs only model-local components."""
    model = _make_model()

    assert len(model.blocks) == 2
    assert all(isinstance(block, _FLAREPlusPlusBlock) for block in model.blocks)
    assert all(
        isinstance(block.Attn, _FLAREPlusPlusAttention) for block in model.blocks
    )


def test_attention_without_context_matches_standalone(device):
    """The context-free adapter is exactly the standalone FLARE++ layer."""
    torch.manual_seed(42)
    standalone = FLAREPlusPlus(
        dim=32,
        heads=4,
        dim_head=8,
        n_global_queries=6,
    ).to(device)
    backend = _FLAREPlusPlusAttention(
        dim=32,
        heads=4,
        dim_head=8,
        dropout=0.0,
        n_global_queries=6,
        context_dim=0,
        concrete_dropout=False,
        state_mixing_mode="weighted",
    ).to(device)
    backend.load_state_dict(standalone.state_dict(), strict=True)
    x = torch.randn(2, 19, 32, device=device)

    torch.testing.assert_close(backend((x,), None)[0], standalone(x))


@pytest.mark.parametrize("state_mixing_mode", ["weighted", "concat_project"])
def test_context_attention_matches_reference(device, state_mixing_mode):
    """The adapter adds only context attention and the configured state mixing."""
    torch.manual_seed(7)
    backend = _FLAREPlusPlusAttention(
        dim=24,
        heads=3,
        dim_head=8,
        dropout=0.0,
        n_global_queries=5,
        context_dim=6,
        concrete_dropout=False,
        state_mixing_mode=state_mixing_mode,
    ).to(device)
    x = torch.randn(2, 17, 24, device=device)
    context = torch.randn(2, 3, 7, 6, device=device)

    actual = backend((x,), context)[0]
    self_output, physical_keys = backend._compute_attention(x)
    context_k, context_v = backend.context_kv(context).chunk(2, dim=-1)
    cross_output = torch.nn.functional.scaled_dot_product_attention(
        backend.cross_q(physical_keys),
        context_k,
        context_v,
        scale=backend.scale,
    )
    if state_mixing_mode == "weighted":
        weight = torch.sigmoid(backend.state_mixing)
        expected = weight * self_output + (1.0 - weight) * cross_output
    else:
        expected = backend.concat_project(
            torch.cat((self_output, cross_output), dim=-1)
        )
    expected = backend._project_output(expected)

    torch.testing.assert_close(actual, expected)


def test_attention_supports_multiple_streams_and_gradients(device):
    """The model adapter routes multiple input streams and their gradients."""
    backend = _FLAREPlusPlusAttention(
        dim=16,
        heads=2,
        dim_head=8,
        dropout=0.0,
        n_global_queries=4,
        context_dim=5,
        concrete_dropout=False,
        state_mixing_mode="weighted",
    ).to(device)
    x1 = torch.randn(2, 11, 16, device=device, requires_grad=True)
    x2 = torch.randn(2, 13, 16, device=device, requires_grad=True)
    context = torch.randn(2, 2, 6, 5, device=device, requires_grad=True)

    outputs = backend((x1, x2), context)
    sum(output.square().mean() for output in outputs).backward()

    assert [output.shape for output in outputs] == [(2, 11, 16), (2, 13, 16)]
    for tensor in (x1, x2, context):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_attention_validates_context_configuration():
    """Invalid context dimensions and state mixing modes fail eagerly."""
    with pytest.raises(ValueError, match="context_dim"):
        _FLAREPlusPlusAttention(
            dim=16,
            heads=2,
            dim_head=8,
            dropout=0.0,
            n_global_queries=4,
            context_dim=-1,
            concrete_dropout=False,
            state_mixing_mode="weighted",
        )
    with pytest.raises(ValueError, match="state_mixing_mode"):
        _FLAREPlusPlusAttention(
            dim=16,
            heads=2,
            dim_head=8,
            dropout=0.0,
            n_global_queries=4,
            context_dim=0,
            concrete_dropout=False,
            state_mixing_mode="invalid",
        )

    backend = _FLAREPlusPlusAttention(
        dim=16,
        heads=2,
        dim_head=8,
        dropout=0.0,
        n_global_queries=4,
        context_dim=0,
        concrete_dropout=False,
        state_mixing_mode="weighted",
    )
    with pytest.raises(ValueError, match="context_dim=0"):
        backend((torch.randn(1, 5, 16),), torch.randn(1, 2, 3, 4))


def test_fresh_model_matches_golden(device):
    """A freshly initialized FLARE++ GeoTransolver matches its golden output."""
    torch.manual_seed(1234)
    model = _make_model().to(device).eval()
    local_embedding, geometry, global_embedding = _make_inputs(device)
    with torch.no_grad():
        output = model(
            local_embedding,
            geometry=geometry,
            global_embedding=global_embedding,
        )
    reference = _load_or_create_output_reference(
        "geotransolver_flare_plus_plus_output.pth", output
    )

    torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-3)


def test_reference_checkpoint_matches_golden(device):
    """The committed FLARE++ GeoTransolver checkpoint remains numerically stable."""
    checkpoint = DATA_DIR / "geotransolver_flare_plus_plus_v1.mdlus"
    if not checkpoint.exists():
        torch.manual_seed(2026)
        _make_model().save(checkpoint)
    model = Module.from_checkpoint(checkpoint).to(device).eval()
    local_embedding, geometry, global_embedding = _make_inputs(device)
    with torch.no_grad():
        output = model(
            local_embedding,
            geometry=geometry,
            global_embedding=global_embedding,
        )
    reference = _load_or_create_output_reference(
        "geotransolver_flare_plus_plus_checkpoint_output.pth", output
    )

    assert isinstance(model, GeoTransolver)
    torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-3)


def test_torch_compile_fullgraph(device):
    """The FLARE++ GeoTransolver backend compiles once and reuses its graph."""
    torch._dynamo.config.error_on_recompile = True
    model = _make_model().to(device).eval()
    local_embedding, geometry, global_embedding = _make_inputs(device)
    with torch.no_grad():
        expected = model(
            local_embedding,
            geometry=geometry,
            global_embedding=global_embedding,
        )
    backend = "inductor" if str(device).startswith("cuda") else "aot_eager"
    compiled = torch.compile(model, backend=backend, fullgraph=True)

    with torch.no_grad():
        actual = compiled(
            local_embedding,
            geometry=geometry,
            global_embedding=global_embedding,
        )
        repeated = compiled(
            local_embedding,
            geometry=geometry,
            global_embedding=global_embedding,
        )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(repeated, expected)


def test_end_to_end_gradients(device):
    """Gradients reach every input and the dynamic routing projections."""
    model = _make_model().to(device)
    local_embedding, geometry, global_embedding = (
        tensor.requires_grad_(True) for tensor in _make_inputs(device)
    )

    output = model(
        local_embedding,
        geometry=geometry,
        global_embedding=global_embedding,
    )
    output.square().mean().backward()

    for tensor in (local_embedding, geometry, global_embedding):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    attention = model.blocks[0].Attn
    for parameter in (
        attention.q_seed,
        attention.in_projection.weight,
        attention.context_kv.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_eval_is_rng_free_with_concrete_dropout(device):
    """Configured standard and Concrete Dropout are disabled during evaluation."""
    model = _make_model(dropout=0.4, concrete_dropout=True).to(device).eval()
    local_embedding, geometry, global_embedding = _make_inputs(device)
    model_device = next(model.parameters()).device
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_rng_before = (
        torch.cuda.get_rng_state(model_device).clone()
        if model_device.type == "cuda"
        else None
    )
    with torch.no_grad():
        output_1 = model(
            local_embedding,
            geometry=geometry,
            global_embedding=global_embedding,
        )
        output_2 = model(
            local_embedding,
            geometry=geometry,
            global_embedding=global_embedding,
        )

    assert torch.equal(output_1, output_2)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_before)
    if cuda_rng_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(model_device), cuda_rng_before)
