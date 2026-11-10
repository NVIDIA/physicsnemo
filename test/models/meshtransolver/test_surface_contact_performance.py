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
"""Exact graph membership/live geometry, and BPTT parity for fused discovery."""

import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from physicsnemo.experimental.models.meshtransolver import (
    SparseContactBlock,
    SurfaceContactGraphBuilder,
)
from physicsnemo.experimental.nn.shell_contact._closest_point import (
    node_triangle_candidate_chunks,
)
from physicsnemo.experimental.nn.shell_contact._narrow_phase_warp import (
    fused_activation_gaps,
)


@pytest.fixture
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return torch.device("cuda:0")


def independent_faces(device, count=512, width=4, translation=0.0):
    generator = torch.Generator(device=device).manual_seed(401)
    a = torch.randn(count, 3, generator=generator, device=device) * 50 + translation
    ab = torch.randn(count, 3, generator=generator, device=device) * 3
    ac = torch.randn(count, 3, generator=generator, device=device) * 3
    vertices = torch.stack((a, a + ab, a + ab + ac, a + ac), 1)
    if width == 3:
        vertices = vertices[:, :3]
    query = (
        vertices.mean(1) + torch.randn(count, 3, generator=generator, device=device) * 6
    )
    positions = torch.cat((vertices.reshape(-1, 3), query))
    faces = torch.arange(count * width, device=device).reshape(count, width)
    pairs = torch.stack(
        (
            torch.arange(count, device=device) + count * width,
            torch.arange(count, device=device),
        )
    )
    return dict(
        positions=positions,
        velocities=torch.randn(positions.shape, generator=generator, device=device)
        * 300,
        thickness=torch.rand(len(positions), generator=generator, device=device) * 3,
        faces=faces,
        pairs=pairs,
    )


@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("horizon", [0.0, 0.005])
@pytest.mark.parametrize("translation", [0.0, 3500.0, 1e6])
def test_fused_membership_matches_reference(cuda, width, horizon, translation):
    scene = independent_faces(cuda, width=width, translation=translation)
    arguments = [
        scene[k] for k in ("positions", "velocities", "thickness", "faces", "pairs")
    ]
    reference = SurfaceContactGraphBuilder(
        prediction_horizon=horizon, material_fan=True, narrow_phase="torch"
    )
    optimized = SurfaceContactGraphBuilder(
        prediction_horizon=horizon, material_fan=True, narrow_phase="warp"
    )
    expected, _ = reference._candidate_gaps(*arguments)
    actual, _ = optimized._candidate_gaps(*arguments)
    assert torch.equal(actual < 5.0, expected < 5.0)
    torch.testing.assert_close(
        actual, expected, atol=5e-4 * (1 + translation / 3500), rtol=2e-4
    )


@pytest.mark.parametrize("width", [3, 4])
def test_exact_cutoff_boundary_recomputed_in_reference(cuda, width):
    vertices = torch.tensor(
        [[0.0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]], device=cuda
    )[:width]
    eps = torch.finfo(torch.float32).eps
    probes = torch.tensor(
        [[2.0, 1.0, 7.0], [2.0, 1.0, 7.0 - 8 * eps], [2.0, 1.0, 7.0 + 8 * eps]],
        device=cuda,
    )
    positions = torch.cat((vertices, probes))
    faces = torch.arange(width, device=cuda)[None]
    pairs = torch.stack(
        (
            torch.arange(3, device=cuda) + width,
            torch.zeros(3, device=cuda, dtype=torch.long),
        )
    )
    thickness = positions.new_full((len(positions),), 2.0)
    gaps, ambiguous = fused_activation_gaps(
        positions, None, thickness, faces, pairs, 0.0, 5.0
    )
    assert ambiguous.all()
    builder = SurfaceContactGraphBuilder(include_velocity=False, narrow_phase="warp")
    actual, fallback = builder._candidate_gaps(positions, None, thickness, faces, pairs)
    expected, _ = SurfaceContactGraphBuilder(
        include_velocity=False, narrow_phase="torch"
    )._candidate_gaps(positions, None, thickness, faces, pairs)
    assert fallback == 3
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("horizon", [0.0, 0.005])
def test_ill_conditioned_triangle_uses_reference(cuda, horizon):
    # Technically nondegenerate, but barycentric roundoff is amplified by the
    # aspect ratio. Membership must be decided by the original implementation.
    positions = torch.tensor(
        [[0.0, 0, 0], [100.0, 0, 0], [50.0, 0.01, 0], [49.0, 0.003, 6.1]],
        device=cuda,
    )
    faces = torch.tensor([[0, 1, 2]], device=cuda)
    pairs = torch.tensor([[3], [0]], device=cuda)
    thickness = positions.new_full((4,), 2.0)
    velocity = torch.zeros_like(positions)
    actual, fallback = SurfaceContactGraphBuilder(
        narrow_phase="warp", prediction_horizon=horizon
    )._candidate_gaps(positions, velocity, thickness, faces, pairs)
    expected, _ = SurfaceContactGraphBuilder(
        narrow_phase="torch", prediction_horizon=horizon
    )._candidate_gaps(positions, velocity, thickness, faces, pairs)
    assert fallback == 1
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("chunk_size", [1, 3, 16])
def test_deferred_reference_pages_preserve_boundary_contacts_and_budget(
    cuda, chunk_size
):
    triangle = torch.tensor([[0.0, 0, 0], [4.0, 0, 0], [0, 4.0, 0]], device=cuda)
    positions = torch.cat(
        (
            triangle,
            triangle.new_tensor(
                [[1.0, 1.0, 6.999], [1.0, 1.0, 7.0], [1.0, 1.0, 7.001]]
            ).repeat(7, 1),
        )
    )
    faces = torch.tensor([[0, 1, 2]], device=cuda)
    arguments = dict(
        faces=faces, shell_thickness=positions.new_full((len(positions),), 2.0)
    )
    builder = SurfaceContactGraphBuilder(
        include_velocity=False, narrow_phase="warp", candidate_chunk_size=chunk_size
    )
    expected = SurfaceContactGraphBuilder(include_velocity=False, narrow_phase="torch")(
        positions, **arguments
    )
    actual = builder(positions, **arguments)
    assert torch.equal(expected.edge_index, actual.edge_index)
    assert torch.equal(expected.edge_features, actual.edge_features)
    assert torch.equal(expected.edge_weights, actual.edge_weights)
    stats = builder.last_discovery_stats
    assert stats["narrow_phase_fallback_pairs"] >= 7
    assert 0 < stats["largest_reference_page"] <= chunk_size
    assert stats["narrow_phase_reference_pages"] >= 1
    builder.max_pairs = actual.edge_index.shape[1] - 1
    with pytest.raises(RuntimeError, match="active contact budget.*no pairs truncated"):
        builder(positions, **arguments)


def test_nondefault_stream_and_noncontiguous_inputs(cuda):
    scene = independent_faces(cuda, count=43)
    stream = torch.cuda.Stream(cuda)
    stream.wait_stream(torch.cuda.current_stream(cuda))
    with torch.cuda.stream(stream):
        # Preserve strides across both point/feature and connectivity dimensions.
        positions = scene["positions"].transpose(0, 1).contiguous().transpose(0, 1)
        velocities = scene["velocities"].transpose(0, 1).contiguous().transpose(0, 1)
        faces = scene["faces"].transpose(0, 1).contiguous().transpose(0, 1)
        pairs = scene["pairs"].transpose(0, 1).contiguous().transpose(0, 1)
        args = (positions, velocities, scene["thickness"], faces, pairs)
        actual, _ = SurfaceContactGraphBuilder(
            narrow_phase="warp", prediction_horizon=0.005
        )._candidate_gaps(*args)
        expected, _ = SurfaceContactGraphBuilder(
            narrow_phase="torch", prediction_horizon=0.005
        )._candidate_gaps(*args)
        torch.testing.assert_close(actual, expected, atol=5e-4, rtol=2e-4)
        assert torch.equal(actual < 5.0, expected < 5.0)
    stream.synchronize()


@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("seed", [73, 892, 1931])
def test_random_deforming_faces_thickness_and_motion_membership(cuda, width, seed):
    scene = independent_faces(cuda, count=4096, width=width, translation=3500.0)
    generator = torch.Generator(device=cuda).manual_seed(seed)
    # Intentionally deform/fold the current surface and independently move its
    # vertices. This exercises sliding endpoint anchors, not only rigid sheets.
    scene["positions"] += (
        torch.randn(scene["positions"].shape, device=cuda, generator=generator) * 5.0
    )
    scene["velocities"] = (
        torch.randn(scene["positions"].shape, device=cuda, generator=generator) * 3000.0
    )
    scene["thickness"] = (
        torch.rand(len(scene["positions"]), device=cuda, generator=generator) * 20.0
    )
    args = tuple(
        scene[k] for k in ("positions", "velocities", "thickness", "faces", "pairs")
    )
    expected, _ = SurfaceContactGraphBuilder(
        narrow_phase="torch", material_fan=True, prediction_horizon=0.005
    )._candidate_gaps(*args)
    actual, _ = SurfaceContactGraphBuilder(
        narrow_phase="warp", material_fan=True, prediction_horizon=0.005
    )._candidate_gaps(*args)
    assert torch.equal(actual < 5.0, expected < 5.0)


def graph_scene(device):
    positions = torch.tensor(
        [
            [0.0, 0, 0],
            [4, 0, 0.1],
            [4, 4, 0.2],
            [0, 4, -0.1],
            [1.2, 0.8, 1.0],
            [2.8, 2.2, 2.0],
            [3.1, 0.3, 7.0],
        ],
        device=device,
    )
    return positions, torch.tensor([[0, 1, 2, 3]], device=device)


@pytest.mark.parametrize("horizon", [0.0, 0.005])
@pytest.mark.parametrize("chunk_size", [1, 3, 65536])
def test_graph_tensors_bitwise_identical(cuda, horizon, chunk_size):
    positions, faces = graph_scene(cuda)
    velocities = (
        torch.arange(positions.numel(), device=cuda).reshape_as(positions).float() * 5
    )
    options = dict(
        material_fan=True, prediction_horizon=horizon, candidate_chunk_size=chunk_size
    )
    inputs = dict(
        faces=faces,
        shell_thickness=positions.new_full((len(positions),), 2.0),
        velocities=velocities,
        reference_positions=positions,
    )
    actual = SurfaceContactGraphBuilder(narrow_phase="warp", **options)(
        positions, **inputs
    )
    expected = SurfaceContactGraphBuilder(narrow_phase="torch", **options)(
        positions, **inputs
    )
    for field in (
        "edge_index",
        "edge_features",
        "obstacle_mask",
        "edge_weights",
        "source_nodes",
        "source_weights",
    ):
        assert torch.equal(getattr(actual, field), getattr(expected, field)), field


@pytest.mark.parametrize("checkpointed", [False, True])
def test_four_step_bptt_and_parameter_gradients(cuda, checkpointed):
    positions, faces = graph_scene(cuda)
    torch.manual_seed(884)
    original = SparseContactBlock(8, contact_dim=12, gate_init=0.2).to(cuda)
    reports = []
    for backend in ("torch", "warp"):
        block = copy.deepcopy(original)
        builder = SurfaceContactGraphBuilder(
            narrow_phase=backend,
            material_fan=True,
            prediction_horizon=0.005,
            candidate_chunk_size=2,
        )
        x = positions.clone().requires_grad_()
        initial_positions = x
        velocity = positions.new_full(positions.shape, 0.2).requires_grad_()
        thickness = positions.new_full((len(positions),), 2.0).requires_grad_()
        nodes = (
            torch.arange(len(positions) * 8, device=cuda)
            .reshape(len(positions), 8)
            .float()
            .div(100)
            .requires_grad_()
        )
        state = nodes
        for _ in range(4):
            graph = builder(
                x,
                faces=faces,
                velocities=velocity,
                shell_thickness=thickness,
                reference_positions=positions,
            )
            state = (
                checkpoint(lambda h, g=graph: block(h, g), state, use_reentrant=False)
                if checkpointed
                else block(state, graph)
            )
            x = x + 0.005 * (velocity + 0.001 * state[:, :3])
        loss = state.square().sum() + x.square().sum() * 0.001
        tensors = (nodes, initial_positions, velocity, thickness, *block.parameters())
        gradients = torch.autograd.grad(loss, tensors)
        assert all(torch.isfinite(g).all() for g in gradients)
        reports.append((state, x, gradients))
    for expected, actual in zip(reports[0][:2], reports[1][:2]):
        assert torch.equal(expected, actual)
    for expected, actual in zip(reports[0][2], reports[1][2]):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("collapse", ["line", "point", "fold"])
def test_material_collapse_and_fold_fallback(cuda, collapse):
    positions, faces = graph_scene(cuda)
    reference = positions.clone()
    if collapse == "point":
        positions[:4] = 0
    elif collapse == "line":
        positions[:4, 1:] = 0
    else:
        positions[[1, 2]] = positions[[2, 1]]
    inputs = dict(
        faces=faces,
        shell_thickness=positions.new_full((len(positions),), 2),
        velocities=torch.zeros_like(positions),
        reference_positions=reference,
    )
    expected = SurfaceContactGraphBuilder(narrow_phase="torch", material_fan=True)(
        positions, **inputs
    )
    actual = SurfaceContactGraphBuilder(narrow_phase="warp", material_fan=True)(
        positions, **inputs
    )
    for field in ("edge_index", "edge_features", "edge_weights", "source_weights"):
        assert torch.equal(getattr(actual, field), getattr(expected, field))


@pytest.mark.parametrize("chunk_size", [1, 2, 5, 16])
def test_gpu_page_ranges_preserve_empty_and_dense_queries(cuda, chunk_size):
    tri = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], device=cuda)
    positions = torch.cat(
        (
            tri.repeat(7, 1),
            tri.new_tensor([[100, 100, 100], [0.2, 0.2, 0.01], [200, 200, 200]]),
        )
    )
    faces = torch.arange(21, device=cuda).reshape(7, 3)
    kwargs = dict(
        node_padding=positions.new_full((len(positions),), 0.1),
        face_padding=positions.new_zeros(len(faces)),
    )
    expected = torch.cat(
        list(
            node_triangle_candidate_chunks(
                positions, faces, implementation="torch", **kwargs
            )
        ),
        1,
    )
    expected = expected[
        :, torch.argsort(expected[0] * len(faces) + expected[1], stable=True)
    ]
    pieces = list(
        node_triangle_candidate_chunks(
            positions,
            faces,
            implementation="warp",
            pair_chunk_size=chunk_size,
            **kwargs,
        )
    )
    assert all(p.shape[1] <= chunk_size for p in pieces)
    actual = torch.cat(pieces, 1)
    order = torch.argsort(actual[0] * len(faces) + actual[1], stable=True)
    assert torch.equal(actual[:, order], expected)


def test_explicit_backend_contract_and_empty(cuda):
    with pytest.raises(ValueError, match="narrow_phase"):
        SurfaceContactGraphBuilder(narrow_phase="fast-but-approximate")
    positions, faces = graph_scene(cuda)
    with pytest.raises(ValueError, match="float32"):
        SurfaceContactGraphBuilder(narrow_phase="warp")(
            positions.double(),
            faces=faces,
            velocities=positions.double(),
            shell_thickness=positions.new_ones(len(positions)).double(),
        )
    inputs = dict(
        faces=faces[:0],
        velocities=torch.zeros_like(positions),
        shell_thickness=positions.new_ones(len(positions)),
        reference_positions=positions,
    )
    result = SurfaceContactGraphBuilder(narrow_phase="warp", material_fan=True)(
        positions, **inputs
    )
    assert result.edge_index.shape == (2, 0)
