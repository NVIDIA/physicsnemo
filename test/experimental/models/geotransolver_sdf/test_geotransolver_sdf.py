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

from pathlib import Path

import pytest
import torch

from physicsnemo import Module
from physicsnemo.experimental.models.geotransolver_sdf import (
    GeoTransolverSDF,
    ScaledSliceGALE,
)
from physicsnemo.experimental.models.geotransolver_sdf.model import (
    _nearest_within_radii,
)
from physicsnemo.nn.module.gale import GALE
from test.common import validate_checkpoint, validate_forward_accuracy

pytest.importorskip("warp")

DATA = Path(__file__).parent / "data"
SMALL = dict(
    n_layers=2,
    n_hidden=16,
    n_head=2,
    slice_num=8,
    n_hidden_local=8,
    radii=(0.2, 0.5),
    neighbors_in_radius=(4, 8),
    local_caps=(4, 8),
    radius_voxels=(0.05, 0.1),
    radius_query_cap=16,
    sdf_spacings=(0.05, 0.2),
    sdf_hidden=8,
    slice_epsilon_reference=64,
)


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU"),
        ),
    ]
)
def device(request):
    return torch.device(request.param)


def _cube(device):
    r"""Triangulated cube :math:`[-0.5, 0.5]^3` with outward-facing triangles."""
    v = torch.tensor(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
        device=device,
    )
    quads = [
        (0, 1, 3, 2),
        (4, 6, 7, 5),
        (0, 4, 5, 1),
        (2, 3, 7, 6),
        (0, 2, 6, 4),
        (1, 5, 7, 3),
    ]
    f = torch.tensor(
        [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], device=device
    )
    return v, f


def _inputs(device, batch=1, nodes=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    coords = torch.rand(batch, nodes, 3, generator=g) * 0.9 - 0.45
    features = torch.cat([coords, torch.rand(batch, nodes, 4, generator=g)], dim=-1)
    v, f = _cube("cpu")
    return tuple(
        t.to(device)
        for t in (
            features,
            coords[:, : nodes // 2].clone(),
            torch.rand(batch, 1, 4, generator=g),
            v.expand(batch, -1, -1),
            f.expand(batch, -1, -1),
        )
    )


def _assert_attributes(model, config):
    assert model.radii == tuple(config.get("radii", (0.05, 0.2)))
    assert model.neighbors_in_radius == tuple(
        config.get("neighbors_in_radius", (8, 32))
    )
    assert model.activation_checkpointing == config.get(
        "activation_checkpointing", False
    )
    assert model.pooling.caps == tuple(config.get("local_caps", (64, 64)))
    assert model.stencil.spacings == tuple(
        config.get("sdf_spacings", (0.005, 0.02, 0.06))
    )
    assert len(model.core.blocks) == config.get("n_layers", 8)
    for block in model.core.blocks:
        assert isinstance(block.Attn, ScaledSliceGALE)
        assert block.Attn.slice_epsilon_reference == config.get(
            "slice_epsilon_reference", 4096
        )
        assert block.Attn.temperature_min == config.get("temperature_min", 0.3)


@pytest.mark.parametrize(
    "config",
    [{}, SMALL, {**SMALL, "activation_checkpointing": True, "temperature_min": 0.5}],
)
def test_constructor(config):
    _assert_attributes(GeoTransolverSDF(**config), config)


@pytest.mark.parametrize(
    "config",
    [
        dict(neighbors_in_radius=(8,)),
        dict(context_chunk=0),
        dict(temperature_min=0.0),
        dict(geometry_dim=2),
    ],
)
def test_constructor_rejects(config):
    with pytest.raises(ValueError):
        GeoTransolverSDF(**{**SMALL, **config})


def test_context_neighbours_match_brute_force(device):
    # nearest first, within each radius, zeros for empty slots; also pins the knn distances to Euclidean
    g = torch.Generator().manual_seed(1)
    points = torch.rand(2, 200, 3, generator=g).to(device)
    queries = torch.rand(2, 30, 3, generator=g).to(device)
    radii, counts = (0.1, 0.25), (4, 9)
    out = _nearest_within_radii(points, queries, radii, counts, chunk=7)
    for b in range(2):
        # exact distances: the matrix-product form of cdist uses TF32 where it is enabled (NGC containers)
        d = torch.cdist(
            queries[b], points[b], compute_mode="donot_use_mm_for_euclid_dist"
        )
        dist, idx = d.sort(dim=1)
        for j, (radius, count) in enumerate(zip(radii, counts)):
            got = out[j][b].reshape(30, count, 3)
            ok = dist[:, :count] <= radius
            expected = torch.where(ok[..., None], points[b][idx[:, :count]], 0.0)
            torch.testing.assert_close(got, expected)


def test_context_neighbours_chunks_agree(device):
    # the single (cuML) call on CUDA and the chunked PyTorch search give the same neighbours
    g = torch.Generator().manual_seed(2)
    points, queries = (
        torch.rand(1, 300, 3, generator=g).to(device),
        torch.rand(1, 41, 3, generator=g).to(device),
    )
    a = _nearest_within_radii(points, queries, (0.2,), (6,), chunk=5)
    b = _nearest_within_radii(points, queries, (0.2,), (6,), chunk=1000)
    torch.testing.assert_close(a[0], b[0])


def test_forward_shape(device):
    torch.manual_seed(0)
    model = GeoTransolverSDF(**SMALL).to(device).eval()
    out = model(*_inputs(device, batch=2))
    assert out.shape == (2, 64, 4) and torch.isfinite(out).all()


def test_rejects_shapes():
    model = GeoTransolverSDF(**SMALL)
    x, geometry, glob, v, f = _inputs("cpu")
    with pytest.raises(ValueError):
        model(x[..., :5], geometry, glob, v, f)
    with pytest.raises(ValueError):
        model(x, geometry, glob[:, :, :3], v, f)
    with pytest.raises(ValueError):
        model(x, geometry, glob, v[0], f)


def test_scaled_slice_gale_matches_gale_at_its_reference(device):
    # at N = N_ref tokens and a temperature clamp of 0.5 the slice normalization is that of GALE
    torch.manual_seed(0)
    base = (
        GALE(32, heads=4, dim_head=8, dropout=0.0, slice_num=8, context_dim=8)
        .to(device)
        .eval()
    )
    scaled = (
        ScaledSliceGALE(
            32,
            heads=4,
            dim_head=8,
            slice_num=8,
            context_dim=8,
            slice_epsilon_reference=50,
            temperature_min=0.5,
        )
        .to(device)
        .eval()
    )
    scaled.load_state_dict(base.state_dict())
    x, context = (
        torch.randn(2, 50, 32, device=device),
        torch.randn(2, 4, 6, 8, device=device),
    )
    torch.testing.assert_close(scaled((x,), context)[0], base((x,), context)[0])


def test_scaled_slice_gale_epsilon_grows_with_tokens():
    layer = ScaledSliceGALE(
        8,
        heads=1,
        dim_head=8,
        slice_num=2,
        context_dim=8,
        slice_epsilon=0.01,
        slice_epsilon_reference=10,
    )
    projections, fx = torch.zeros(1, 40, 1, 2), torch.ones(1, 40, 1, 8)
    weights, tokens = layer._compute_slices_from_projections(projections, fx)
    # uniform weights 1/2: token = sum(w) / (sum(w) + eps * N / N_ref) = 20 / (20 + 0.04)
    torch.testing.assert_close(weights, torch.full_like(weights, 0.5))
    torch.testing.assert_close(tokens, torch.full_like(tokens, 20 / 20.04))


def test_activation_checkpointing_matches(device):
    torch.manual_seed(0)
    plain = GeoTransolverSDF(**SMALL).to(device).eval()
    checkpointed = (
        GeoTransolverSDF(**SMALL, activation_checkpointing=True).to(device).eval()
    )
    checkpointed.load_state_dict(plain.state_dict())
    torch.nn.init.normal_(plain.stencil.out.weight, std=0.1)
    checkpointed.stencil.out.weight.data.copy_(plain.stencil.out.weight)
    inputs = _inputs(device)
    a, b = plain(*inputs), checkpointed(*inputs)
    torch.testing.assert_close(a, b)
    a.square().sum().backward()
    b.square().sum().backward()
    for p, q in zip(plain.parameters(), checkpointed.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6)


def test_checkpoint(device, tmp_path, monkeypatch):
    # the shared helper compares in training mode: draw every neighbour found (caps = query cap), so that the random
    # neighbour draw cannot change the max-pooled features
    monkeypatch.chdir(tmp_path)
    config = {**SMALL, "local_caps": (16, 16)}
    model = GeoTransolverSDF(**config).to(device).eval()
    restored = GeoTransolverSDF(**config).to(device).eval()
    assert validate_checkpoint(model, restored, _inputs(device), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("source", ["constructor", "checkpoint"])
def test_forward_accuracy(source):
    r"""Frozen outputs of a small model on CPU, built from the constructor and from the .mdlus checkpoint."""
    data = torch.load(DATA / "small.pth", weights_only=True)
    if source == "checkpoint":
        model = Module.from_checkpoint(str(DATA / "small.mdlus"))
    else:
        model = GeoTransolverSDF(**data["config"])
        model.load(str(DATA / "small.mdlus"))
    model = model.eval()
    _assert_attributes(model, data["config"])
    names = (
        "local_embedding",
        "geometry",
        "global_embedding",
        "surface_vertices",
        "surface_faces",
    )
    assert validate_forward_accuracy(
        model,
        tuple(data["inputs"][k] for k in names),
        atol=1e-5,
        rtol=1e-4,
        file_name="experimental/models/geotransolver_sdf/data/small_output.pth",
    )
