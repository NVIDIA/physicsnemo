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

from physicsnemo.experimental.models.meshtransolver import (
    ContactGraph,
    SparseContactBlock,
    SurfaceContactGraphBuilder,
    merge_contact_graphs,
)
from physicsnemo.nn.functional.neighbors.surface_contact import (
    closest_point_facet,
    closest_point_triangle,
    node_triangle_candidates,
    supported_quad_mask,
    swept_node_triangle_check,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def scene(device="cpu", dtype=torch.float32):
    x = torch.tensor(
        [[0, 0, 0], [10, 0, 0], [0, 10, 0], [2, 2, 0.2], [2, 2, -0.2], [20, 20, 20]],
        device=device,
        dtype=dtype,
    )
    return x, torch.tensor([[0, 1, 2]], device=device)


def test_triangle_face_edges_corners_two_sides(device):
    tri = torch.tensor(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], device=device, dtype=torch.float64
    )
    points = tri.new_tensor(
        [
            [0.2, 0.3, 0.4],
            [0.2, 0.3, -0.4],
            [0.2, -0.2, 0.1],
            [-1, -1, 0],
            [0.8, 0.8, 0.1],
        ]
    )
    expected = tri.new_tensor(
        [[0.2, 0.3, 0], [0.2, 0.3, 0], [0.2, 0, 0], [0, 0, 0], [0.5, 0.5, 0]]
    )
    p = closest_point_triangle(points, tri.expand(5, -1, -1))
    torch.testing.assert_close(p.closest, expected)
    torch.testing.assert_close(
        p.barycentric.sum(-1), torch.ones(5, device=device, dtype=torch.float64)
    )
    assert (p.barycentric >= 0).all()
    reversed_p = closest_point_triangle(points, tri.flip(0).expand(5, -1, -1))
    torch.testing.assert_close(p.closest, reversed_p.closest)
    torch.testing.assert_close(p.normal, -reversed_p.normal)


@pytest.mark.parametrize("quad", [False, True])
def test_geometry_gradcheck(device, quad):
    vertices = [[[0.0, 0, 0], [2.0, 0, 0.1], [1.8, 2.0, 0.3], [0, 2.0, -0.1]]]
    faces = torch.tensor(vertices, dtype=torch.float64, device=device)
    if not quad:
        faces = faces[:, :3]
    faces.requires_grad_()
    point = faces.new_tensor([[1.2, 0.4, 0.6]], requires_grad=True)

    def fn(p, f):
        out = closest_point_facet(p, f)
        return torch.cat(
            (out.closest, out.barycentric, out.distance[:, None], out.normal), -1
        )

    assert torch.autograd.gradcheck(fn, (point, faces), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("scale", [1e-10, 1.0, 1e10])
def test_triangle_scale_safe_values_and_gradients(device, scale):
    tri = torch.tensor([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]], device=device) * scale
    tri.requires_grad_()
    point = tri.new_tensor([[0.2, 0.3, 0.1]]) * scale
    point.requires_grad_()
    result = closest_point_triangle(point, tri)
    torch.testing.assert_close(result.distance / scale, point.new_tensor([0.1]))
    torch.testing.assert_close(result.normal, point.new_tensor([[0, 0, 1]]))
    torch.testing.assert_close(result.barycentric, point.new_tensor([[0.5, 0.2, 0.3]]))
    (result.distance / scale + result.barycentric.square().sum(-1)).sum().backward()
    assert torch.isfinite(point.grad).all() and torch.isfinite(tri.grad).all()
    assert point.grad.abs().sum() > 0 and tri.grad.abs().sum() > 0


def test_quad_policy_rejects_unsupported_shapes(device):
    q = torch.tensor(
        [
            [[0.0, 0, 0], [2, 0, 0], [2, 2, 0.3], [0, 2, 0]],
            [[0.0, 0, 0], [2, 0, 0], [0.2, 0.2, 0], [0, 2, 0]],
            [[0.0, 0, 0], [2, 2, 0], [2, 0, 0], [0, 2, 0]],
            [[0.0, 0, 0], [0, 0, 0], [2, 2, 0], [0, 2, 0]],
        ],
        device=device,
    )
    expected = torch.tensor([True, False, False, False], device=device)
    for scale in (1e-10, 1.0, 1e10):
        torch.testing.assert_close(supported_quad_mask(q * scale), expected)
    torch.testing.assert_close(supported_quad_mask(q.flip(1)), expected)
    for invalid in q[1:]:
        with pytest.raises(ValueError, match="unsupported surface quad"):
            closest_point_facet(
                q.new_tensor([[0.5, 0.5, 0.01]]),
                invalid[None],
                triangle_mask=torch.zeros(1, dtype=torch.bool, device=device),
            )
    # Far-away invalid facets must fail too, not disappear from broad phase.
    builder = SurfaceContactGraphBuilder(implementation="torch", include_velocity=False)
    with pytest.raises(ValueError, match="facet IDs"):
        builder(
            q[1],
            faces=torch.tensor([[0, 1, 2, 3]], device=device),
            shell_thickness=q.new_zeros(4),
        )


def test_valid_concave_center_fan_keeps_notch_outside_surface(device):
    # Mild concavity is legitimate when the center lies in the kernel. Unlike
    # the pathological .2,.2 example above, the fan does not fill the notch.
    q = torch.tensor(
        [[[0.0, 0, 0], [2, 0, 0], [0.8, 0.8, 0], [0, 2, 0]]],
        device=device,
        dtype=torch.float64,
    )
    for scale in (1e-10, 1.0, 1e10):
        assert supported_quad_mask(q * scale).all()
        assert supported_quad_mask(q.flip(1) * scale).all()
    point = q.new_tensor([[0.9, 0.9, 0.1]])
    projected = closest_point_facet(point, q)
    assert projected.distance.item() > 0.15
    torch.testing.assert_close(projected.barycentric.sum(-1), q.new_ones(1))
    point = q.new_tensor([[0.43, 0.32, 0.2]], requires_grad=True)
    q.requires_grad_()
    assert torch.autograd.gradcheck(
        lambda p, f: closest_point_facet(p, f).distance,
        (point, q),
        atol=1e-5,
        rtol=1e-4,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_surface_interpolation_matches_dense_values_and_gradients(device, dtype):
    torch.manual_seed(71)
    latent = torch.randn(8, 6, device=device, dtype=dtype, requires_grad=True)
    # Live geometry remains FP32 even when the model's latents use AMP.
    weights = torch.softmax(torch.randn(5, 4, device=device), -1).requires_grad_()
    sources = torch.randint(0, 8, (5, 4), device=device)
    graph = ContactGraph(
        torch.stack((sources[:, 0], torch.arange(5, device=device))),
        latent.new_zeros(5, 8),
        torch.zeros(5, dtype=torch.bool, device=device),
        source_nodes=sources,
        source_weights=weights,
    )
    block = SparseContactBlock(hidden_dim=6).to(device=device, dtype=dtype)
    captured = []
    hook = block.edge_mlp.register_forward_pre_hook(
        lambda module, args: captured.append(args[0])
    )
    block(latent, graph)
    hook.remove()
    actual = captured[0][:, :6]
    expected = (latent[sources] * weights.to(dtype)[..., None]).sum(1)
    torch.testing.assert_close(actual, expected)
    grads = torch.autograd.grad(
        actual.square().sum(), (latent, weights), retain_graph=True
    )
    reference = torch.autograd.grad(expected.square().sum(), (latent, weights))
    for a, b in zip(grads, reference):
        assert torch.isfinite(a).all()
        if dtype == torch.bfloat16:
            torch.testing.assert_close(a, b, atol=0.04, rtol=0.04)
        elif dtype == torch.float16:
            torch.testing.assert_close(a, b, atol=0.005, rtol=0.01)
        else:
            torch.testing.assert_close(a, b)


def test_quad_center_fan_and_mixed_triangle(device):
    quad = torch.tensor([[[0.0, 0, 0], [2, 0, 0], [2, 2, 1], [0, 2, 0]]], device=device)
    point = quad.new_tensor([[1.1, 0.3, 0.6]])
    p = closest_point_facet(point, quad)
    center = quad.mean(1)
    parts = [
        closest_point_triangle(
            point, torch.stack((center, quad[:, i], quad[:, (i + 1) % 4]), 1)
        )
        for i in range(4)
    ]
    selected = torch.stack([x.distance for x in parts]).argmin(0).item()
    torch.testing.assert_close(p.closest, parts[selected].closest)
    torch.testing.assert_close(p.closest, (p.barycentric[..., None] * quad).sum(1))
    tri = quad[:, :3]
    padded = torch.cat((tri, tri[:, -1:]), 1)
    mixed = closest_point_facet(point.expand(2, -1), torch.cat((quad, padded)))
    torch.testing.assert_close(
        mixed.closest[1:], closest_point_triangle(point, tri).closest
    )


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_whole_facet_discovery_incidence_batch_and_exclusions(device, implementation):
    x, f = scene(device)
    # All three vertices are far away from query nodes; a vertex radius misses.
    pair = node_triangle_candidates(
        x, f, x.new_full((6,), 0.3), x.new_zeros(1), implementation=implementation
    )
    torch.testing.assert_close(pair, f.new_tensor([[3, 4], [0, 0]]))
    batch = f.new_tensor([0, 0, 0, 1, 0, 0])
    pair = node_triangle_candidates(
        x,
        f,
        x.new_full((6,), 0.3),
        x.new_zeros(1),
        batch=batch,
        implementation=implementation,
    )
    torch.testing.assert_close(pair, f.new_tensor([[4], [0]]))
    pair = node_triangle_candidates(
        x,
        f,
        x.new_full((6,), 0.3),
        x.new_zeros(1),
        batch=batch,
        excluded_pairs=f.new_tensor([[4], [0]]),
        implementation=implementation,
    )
    assert pair.shape == (2, 0)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_dense_all_pairs_no_topk_and_budget(device, implementation):
    # 40 disjoint facets near a query: all must be returned, not just nearest 16.
    tri = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], device=device)
    x = torch.cat((tri.repeat(40, 1), tri.new_tensor([[0.2, 0.2, 0.05]])))
    f = torch.arange(120, device=device).reshape(-1, 3)
    args = (x, f, x.new_full((121,), 0.1), x.new_zeros(40))
    pairs = node_triangle_candidates(
        *args, implementation=implementation, max_pairs=10000
    )
    assert (pairs[0] == 120).sum() == 40
    with pytest.raises(RuntimeError, match="no pairs truncated"):
        node_triangle_candidates(*args, implementation=implementation, max_pairs=20)


@pytest.mark.parametrize("quad", [False, True])
def test_warp_torch_candidate_agreement_random(device, quad):
    torch.manual_seed(301)
    x = torch.randn(100, 3, device=device)
    f = torch.randperm(100, device=device)[:80].reshape(-1, 4)
    if not quad:
        f = f[:, :3]
    padding = torch.rand(100, device=device) * 0.1
    args = (x, f, padding, padding[: len(f)])
    expected = node_triangle_candidates(*args, implementation="torch", chunk_size=7)
    for _ in range(2):
        actual = node_triangle_candidates(*args, implementation="warp")
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_surface_builder_geometry_velocity_and_thickness(device):
    x, f = scene(device)
    x.requires_grad_()
    thickness = x.new_tensor([0.2, 0.4, 0.6, 0.4, 0.4, 0.4])
    velocity = torch.randn_like(x, requires_grad=True)
    builder = SurfaceContactGraphBuilder(
        implementation="torch", activation_distance=0.2
    )
    graph = builder(x, faces=f, shell_thickness=thickness, velocities=velocity)
    graph.validate(6, 12)
    torch.testing.assert_close(graph.edge_index[1], f.new_tensor([3, 4]))
    weights = x.new_tensor([[0.6, 0.2, 0.2]]).expand(2, -1)
    torch.testing.assert_close(graph.source_weights, weights)
    expected_gap = 0.2 - (0.4 + (0.6 * 0.2 + 0.2 * 0.4 + 0.2 * 0.6)) / 2
    torch.testing.assert_close(
        graph.edge_features[:, 4], x.new_full((2,), expected_gap / 10)
    )
    torch.testing.assert_close(
        graph.edge_features[:, 8:11],
        ((velocity[f[0]] * weights[0, :, None]).sum(0) - velocity[[3, 4]]) / 1000,
    )
    block = SparseContactBlock(hidden_dim=8, contact_dim=12, gate_init=0.1).to(device)
    graph.source_weights.retain_grad()
    block(torch.randn(6, 8, device=device), graph).square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert velocity.grad.abs().sum() > 0
    assert graph.source_weights.grad.abs().sum() > 0


def test_empty_merge_and_live_interpolation(device):
    x, f = scene(device)
    builder = SurfaceContactGraphBuilder(
        implementation="torch", include_velocity=False, activation_distance=0.01
    )
    graph = builder(x, faces=f, shell_thickness=x.new_zeros(6))
    assert graph.source_weights.shape == (0, 3)
    graph.validate(6, 8)
    block = SparseContactBlock(hidden_dim=4).to(device)
    latent = torch.randn(6, 4, device=device, requires_grad=True)
    torch.testing.assert_close(block(latent, graph), latent, atol=0, rtol=0)
    graph2 = builder(x, faces=f, shell_thickness=x.new_ones(6))
    plain = ContactGraph(
        f.new_tensor([[1], [3]]),
        x.new_zeros((1, 8)),
        torch.zeros(1, device=device, dtype=torch.bool),
    )
    merged = merge_contact_graphs(graph, graph2, plain).to(torch.device(device))
    merged.validate(6, 8)
    torch.testing.assert_close(merged.source_weights[-1], x.new_tensor([1.0, 0, 0]))


def test_degenerate_nonfinite_and_empty(device):
    x, f = scene(device)
    with pytest.raises(ValueError, match="degenerate"):
        closest_point_triangle(x[3:4], x.new_zeros((1, 3, 3)))
    bad = x.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        node_triangle_candidates(bad, f, x.new_ones(6), x.new_ones(1))
    result = closest_point_triangle(x[:0], x.new_empty((0, 3, 3)))
    assert result.closest.shape == (0, 3)
    for implementation in ("torch", "warp"):
        pairs = node_triangle_candidates(
            x, f[:0], x.new_ones(6), x.new_empty(0), implementation=implementation
        )
        assert pairs.shape == (2, 0)


def test_swept_crossing_separated_and_unresolved(device):
    x, f = scene(device, torch.float64)
    tri = x[f].expand(3, -1, -1)
    p = x.new_tensor([[2, 2, 1], [2, 2, 2], [2, 2, 0.231]])
    q = x.new_tensor([[2, 2, -1], [2, 2, 3], [2, 2, -0.769]])
    # Last point crosses away from dyadic samples: zero budget cannot certify it.
    result = swept_node_triangle_check(p, q, tri, tri, x.new_zeros(3), max_depth=0)
    assert result.hit.tolist() == [True, False, False]
    assert result.unresolved.tolist() == [False, False, True]
    result = swept_node_triangle_check(p, q, tri, tri, x.new_full((3,), 0.01))
    assert result.hit.tolist() == [True, False, True]
    assert not result.unresolved.any()


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_swept_bounds_recover_crossing_candidates(device, implementation):
    x, f = scene(device)
    x[3, 2] = 1
    previous = x.clone()
    previous[3, 2] = -1
    args = (x, f, x.new_full((6,), 0.05), x.new_zeros(1))
    discrete = node_triangle_candidates(*args, implementation=implementation)
    assert not (discrete[0] == 3).any()
    swept = node_triangle_candidates(
        *args, previous_positions=previous, implementation=implementation
    )
    assert (swept[0] == 3).any()


def test_coincident_geometry_gradients_finite(device):
    x, f = scene(device)
    x[3, 2] = 0
    x.requires_grad_()
    velocity = torch.randn_like(x, requires_grad=True)
    builder = SurfaceContactGraphBuilder(implementation="torch")
    graph = builder(x, faces=f, shell_thickness=x.new_ones(6), velocities=velocity)
    graph.edge_features.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(velocity.grad).all()


def test_quad_with_coincident_distinct_vertices_rejected(device):
    x, _ = scene(device)
    x[3] = x[2]
    faces = torch.tensor([[0, 1, 2, 3]], device=device)
    builder = SurfaceContactGraphBuilder(implementation="torch", include_velocity=False)
    with pytest.raises(ValueError, match="degenerate"):
        builder(x, faces=faces, shell_thickness=x.new_ones(6))


def test_sweep_budget_never_certifies_unexamined_intervals(device):
    x, f = scene(device)
    p = x.new_tensor([[2, 2, 0.231], [2, 2, 0.317]])
    q = p.clone()
    q[:, 2] -= 1
    tri = x[f].expand(2, -1, -1)
    result = swept_node_triangle_check(p, q, tri, tri, x.new_zeros(2), max_intervals=1)
    assert result.unresolved.all() and not result.hit.any()


def test_warp_uses_torch_nondefault_cuda_stream():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x, f = scene("cuda")
        args = (x, f, x.new_full((6,), 0.3), x.new_zeros(1))
        actual = node_triangle_candidates(*args, implementation="warp")
        expected = node_triangle_candidates(*args, implementation="torch")
        torch.testing.assert_close(actual, expected)
    stream.synchronize()


@pytest.mark.parametrize("quad", [False, True])
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_predictive_surface_catches_causal_crossing(device, implementation, quad):
    x = torch.tensor(
        [[0.0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [6, 4, 2]],
        device=device,
    )
    faces = torch.tensor([[0, 1, 2, 3]] if quad else [[0, 1, 2]], device=device)
    velocity = torch.zeros_like(x)
    velocity[-1, 2] = -800
    opts = dict(implementation=implementation, activation_distance=0.2)
    current = SurfaceContactGraphBuilder(**opts)
    predictive = SurfaceContactGraphBuilder(**opts, prediction_horizon=0.005)
    args = dict(faces=faces, shell_thickness=x.new_full((5,), 0.1), velocities=velocity)
    old = current(x, **args)
    assert not (old.edge_index[1] == 4).any()
    graph = predictive(x, **args)
    i = graph.edge_index[1] == 4
    assert i.sum() == 1 and graph.edge_weights[i].item() == 1
    # Current gap stays positive in the feature, despite predicted activation.
    assert graph.edge_features[i, 4].item() > 0
    end = current(x + 0.005 * velocity, **args)
    assert not (end.edge_index[1] == 4).any()
    # A common velocity/position translation must not alter relative detection.
    boosted = predictive(
        x + x.new_tensor([100, -200, 50]),
        **{**args, "velocities": velocity + x.new_tensor([10000, -9000, 4000])},
    )
    torch.testing.assert_close(graph.edge_index, boosted.edge_index)
    torch.testing.assert_close(
        graph.edge_features, boosted.edge_features, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(graph.edge_weights, boosted.edge_weights)
    # Receding motion must not activate the approaching-side message.
    away = predictive(x, **{**args, "velocities": -velocity})
    assert not (away.edge_index[1] == 4).any()


def test_predictive_activation_gradcheck_and_zero_speed(device):
    from physicsnemo.experimental.models.meshtransolver.surface_contact import (
        linear_closest_approach,
    )

    r = torch.tensor(
        [[1.0, 2.0, 0.4]], device=device, dtype=torch.float64, requires_grad=True
    )
    v = torch.tensor(
        [[-400.0, -60.0, 10.0]], device=device, dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(
        lambda a, b: linear_closest_approach(a, b, 0.005), (r, v)
    )
    zero = torch.zeros_like(v, requires_grad=True)
    output = linear_closest_approach(r, zero, 0.005)
    torch.testing.assert_close(output, r.norm(dim=-1))
    gradients = torch.autograd.grad(output.sum(), (r, zero))
    assert all(torch.isfinite(g).all() for g in gradients)


def test_predictive_surface_weights_have_live_geometry_velocity_gradients(device):
    x = torch.tensor(
        [[0.0, 0, 0], [10, 0, 0], [10, 10, 0], [6, 3, 2]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    v = torch.tensor(
        [[0.0, 0, 0], [0, 0, 0], [0, 0, 0], [20, -10, -200]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2]], device=device)
    builder = SurfaceContactGraphBuilder(
        implementation="torch", prediction_horizon=0.005, activation_distance=2.0
    )

    def weight(p, u):
        return builder(
            p, faces=faces, velocities=u, shell_thickness=p.new_zeros(4)
        ).edge_weights

    assert torch.autograd.gradcheck(weight, (x, v), atol=1e-5, rtol=1e-4)
    grads = torch.autograd.grad(weight(x, v).sum(), (x, v))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_predictive_endpoint_anchor_handles_sliding_closest_point(device):
    x = torch.tensor([[0.0, 0, 0], [10, 0, 0], [0, 10, 0], [-2, 2, 2]], device=device)
    v = torch.zeros_like(x)
    v[-1] = x.new_tensor([200, 800, -200])
    faces = torch.tensor([[0, 1, 2]], device=device)
    args = dict(faces=faces, velocities=v, shell_thickness=x.new_zeros(4))
    old = SurfaceContactGraphBuilder(implementation="torch", activation_distance=2)(
        x, **args
    )
    new = SurfaceContactGraphBuilder(
        implementation="torch", activation_distance=2, prediction_horizon=0.005
    )(x, **args)
    assert old.edge_index.shape[1] == 0
    assert new.edge_index[1].tolist() == [3]
    assert new.edge_weights.item() > 0


def test_predictive_forecast_quad_failure_is_explicit(device):
    x = torch.tensor([[0.0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], device=device)
    v = torch.zeros_like(x)
    v[2] = x.new_tensor([-360, -360, 0])
    with pytest.raises(ValueError, match="forecast facet IDs"):
        SurfaceContactGraphBuilder(implementation="torch", prediction_horizon=0.005)(
            x,
            faces=torch.tensor([[0, 1, 2, 3]], device=device),
            velocities=v,
            shell_thickness=x.new_ones(4),
        )


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_predictive_batch_exclusions_and_budget(device, implementation):
    x, f = scene(device)
    x[3, 2] = 2
    v = torch.zeros_like(x)
    v[3, 2] = -800
    x = torch.cat((x, x))
    v = torch.cat((v, v))
    f = torch.cat((f, f + 6))
    opts = dict(
        implementation=implementation, prediction_horizon=0.005, activation_distance=0.1
    )
    args = dict(
        faces=f,
        velocities=v,
        shell_thickness=x.new_zeros(12),
        batch=f.new_tensor([0] * 6 + [1] * 6),
        excluded_pairs=f.new_tensor([[3], [0]]),
    )
    graph = SurfaceContactGraphBuilder(**opts)(x, **args)
    assert graph.edge_index[1].tolist() == [9]
    assert graph.source_nodes.tolist() == [[6, 7, 8]]
    with pytest.raises(RuntimeError, match="budget"):
        SurfaceContactGraphBuilder(**opts, max_pairs=1)(
            x, **{**args, "excluded_pairs": None}
        )


def test_predictive_zero_velocity_matches_endpoint_and_requires_velocity(device):
    x, f = scene(device)
    args = dict(faces=f, shell_thickness=x.new_ones(6), velocities=torch.zeros_like(x))
    opts = dict(implementation="torch")
    old = SurfaceContactGraphBuilder(**opts)(x, **args)
    new = SurfaceContactGraphBuilder(**opts, prediction_horizon=0.005)(x, **args)
    for field in ("edge_index", "edge_features", "source_weights", "edge_weights"):
        torch.testing.assert_close(
            getattr(old, field), getattr(new, field), atol=0, rtol=0
        )
    with pytest.raises(ValueError, match="velocities"):
        SurfaceContactGraphBuilder(
            **opts, include_velocity=False, prediction_horizon=0.005
        )(x, **{**args, "velocities": None})
    for horizon in (-1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="prediction_horizon"):
            SurfaceContactGraphBuilder(prediction_horizon=horizon)
