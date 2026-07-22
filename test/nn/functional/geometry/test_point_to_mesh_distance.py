# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for differentiable point-to-triangle-mesh distance."""

from __future__ import annotations

import inspect
from typing import Literal, get_type_hints

import pytest
import torch

import physicsnemo.nn.functional as functional
from benchmarks.physicsnemo.nn.functional.registry import FUNCTIONAL_SPECS
from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.nn.functional.geometry.point_to_mesh_distance import (
    PointToMeshDistance,
    point_to_mesh_distance,
)
from physicsnemo.nn.functional.geometry.point_to_mesh_distance import (
    _torch_impl as torch_impl,
)
from physicsnemo.nn.functional.geometry.point_to_mesh_distance._warp_impl import (
    nearest_face_indices_warp_impl,
)


def _triangle(
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one right triangle in the xy plane."""

    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    )
    faces = torch.tensor([[0, 1, 2]], device=device, dtype=torch.long)
    return vertices, faces


def test_public_exports_function_spec_and_benchmark_registry():
    """The callable is flat-exported while its FunctionSpec stays categorized."""

    assert functional.point_to_mesh_distance is point_to_mesh_distance
    assert point_to_mesh_distance.__module__ == (
        "physicsnemo.nn.functional.geometry.point_to_mesh_distance."
        "point_to_mesh_distance"
    )
    assert issubclass(PointToMeshDistance, FunctionSpec)
    assert PointToMeshDistance in FUNCTIONAL_SPECS
    assert not hasattr(functional, "PointToMeshDistance")
    assert list(inspect.signature(point_to_mesh_distance).parameters) == [
        "mesh_vertices",
        "mesh_indices",
        "input_points",
        "squared",
        "implementation",
    ]
    assert PointToMeshDistance.implementations() == ("warp", "torch")
    assert get_type_hints(point_to_mesh_distance)["implementation"] == (
        Literal["torch", "warp"] | None
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_triangle_regions(device: str, dtype: torch.dtype):
    """Face, edge, and vertex Voronoi regions project to known points."""

    vertices, faces = _triangle(device, dtype)
    queries = torch.tensor(
        [
            [0.2, 0.3, 1.0],  # face interior
            [-1.0, -1.0, 0.0],  # vertex A
            [2.0, -1.0, 0.0],  # vertex B
            [-1.0, 2.0, 0.0],  # vertex C
            [0.5, -1.0, 0.0],  # edge AB
            [-1.0, 0.5, 0.0],  # edge AC
            [0.75, 0.75, 0.0],  # edge BC
        ],
        device=device,
        dtype=dtype,
    )
    expected_closest = torch.tensor(
        [
            [0.2, 0.3, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    expected_distance = torch.linalg.vector_norm(queries - expected_closest, dim=-1)

    distance, closest = point_to_mesh_distance(
        vertices, faces, queries, implementation="torch"
    )

    tolerance = 1.0e-6 if dtype == torch.float32 else 1.0e-12
    torch.testing.assert_close(
        closest, expected_closest, atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(
        distance, expected_distance, atol=tolerance, rtol=tolerance
    )


def test_arbitrary_query_prefix_and_multiple_faces(device: str):
    """Every leading query dimension is preserved and shares one mesh."""

    base_vertices, base_face = _triangle(device, torch.float64)
    vertices = torch.cat(
        (
            base_vertices,
            base_vertices
            + torch.tensor([0.0, 0.0, 3.0], device=device, dtype=torch.float64),
        )
    )
    faces = torch.cat((base_face, base_face + 3))
    queries = torch.tensor(
        [
            [[[0.25, 0.25, 0.5], [0.25, 0.25, 2.75]]],
            [[[0.1, 0.1, -2.0], [0.1, 0.1, 4.0]]],
        ],
        device=device,
        dtype=torch.float64,
    )

    distance, closest = point_to_mesh_distance(vertices, faces, queries, squared=True)

    assert distance.shape == (2, 1, 2)
    assert closest.shape == queries.shape
    torch.testing.assert_close(
        closest[..., 2],
        torch.tensor([[[0.0, 3.0]], [[0.0, 3.0]]], device=device, dtype=torch.float64),
    )
    torch.testing.assert_close(
        distance,
        torch.tensor(
            [[[0.25, 0.0625]], [[4.0, 1.0]]],
            device=device,
            dtype=torch.float64,
        ),
    )


def test_squared_flag(device: str):
    """The squared option changes only the distance representation."""

    vertices, faces = _triangle(device, torch.float32)
    queries = torch.tensor([[0.2, 0.2, 2.0], [0.5, -3.0, 0.0]], device=device)
    distance, closest = point_to_mesh_distance(vertices, faces, queries)
    distance_squared, closest_squared = point_to_mesh_distance(
        vertices, faces, queries, squared=True
    )
    torch.testing.assert_close(distance_squared, distance.square())
    torch.testing.assert_close(closest_squared, closest)


def test_single_and_empty_query_shapes(device: str):
    """Scalar query prefixes and empty leading dimensions are supported."""

    vertices, faces = _triangle(device, torch.float32)
    single = torch.tensor([0.25, 0.25, 1.0], device=device)
    distance, closest = point_to_mesh_distance(vertices, faces, single)
    assert distance.shape == torch.Size([])
    assert closest.shape == (3,)
    torch.testing.assert_close(distance, torch.tensor(1.0, device=device))

    empty = torch.empty(2, 0, 3, device=device)
    empty_distance, empty_closest = point_to_mesh_distance(vertices, faces, empty)
    assert empty_distance.shape == (2, 0)
    assert empty_closest.shape == (2, 0, 3)


def test_analytic_squared_distance_gradients(device: str):
    """Squared distance has the envelope gradients for query and vertices."""

    vertices, faces = _triangle(device, torch.float64)
    vertices.requires_grad_()
    query = torch.tensor(
        [[0.25, 0.25, 1.0]], device=device, dtype=torch.float64, requires_grad=True
    )
    distance, _ = point_to_mesh_distance(vertices, faces, query, squared=True)
    distance.sum().backward()

    torch.testing.assert_close(
        query.grad,
        torch.tensor([[0.0, 0.0, 2.0]], device=device, dtype=torch.float64),
    )
    torch.testing.assert_close(
        vertices.grad,
        torch.tensor(
            [[0.0, 0.0, -1.0], [0.0, 0.0, -0.5], [0.0, 0.0, -0.5]],
            device=device,
            dtype=torch.float64,
        ),
    )


def test_target_gradients_accumulate_across_queries(device: str):
    """Shared target vertices receive contributions from every correspondence."""

    vertices, faces = _triangle(device, torch.float64)
    vertices.requires_grad_()
    queries = torch.tensor(
        [[0.25, 0.25, 1.0], [0.2, 0.3, 2.0]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    distance, _ = point_to_mesh_distance(vertices, faces, queries, squared=True)
    distance.sum().backward()

    # Barycentrics are (0.5, 0.25, 0.25) and (0.5, 0.2, 0.3).
    expected_vertex_grad = torch.tensor(
        [[0.0, 0.0, -3.0], [0.0, 0.0, -1.3], [0.0, 0.0, -1.7]],
        device=device,
        dtype=torch.float64,
    )
    torch.testing.assert_close(vertices.grad, expected_vertex_grad)
    torch.testing.assert_close(
        queries.grad,
        torch.tensor(
            [[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]],
            device=device,
            dtype=torch.float64,
        ),
    )


def test_gradcheck_and_gradgradcheck(device: str):
    """Distances and closest points retain first- and second-order autograd."""

    vertices, faces = _triangle(device, torch.float64)
    vertices.requires_grad_()
    queries = torch.tensor(
        [[0.2, 0.3, 0.8], [0.3, 0.2, 1.2]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )

    def objective(vertex_values, query_values):
        distance, closest = point_to_mesh_distance(
            vertex_values,
            faces,
            query_values,
            squared=True,
            implementation="torch",
        )
        closest_weights = torch.tensor(
            [[0.2, -0.1, 0.3], [-0.3, 0.4, 0.1]],
            device=device,
            dtype=torch.float64,
        )
        return distance.sum() + (closest * closest_weights).sum()

    assert torch.autograd.gradcheck(objective, (vertices, queries))
    assert torch.autograd.gradgradcheck(objective, (vertices, queries))


def test_translation_equivariance(device: str):
    """Translating mesh and queries preserves distances and shifts hits."""

    vertices, faces = _triangle(device, torch.float64)
    queries = torch.tensor(
        [[0.2, 0.3, 1.0], [0.8, 0.8, -0.5]], device=device, dtype=torch.float64
    )
    translation = torch.tensor([2.0, -3.0, 4.0], device=device, dtype=torch.float64)
    distance, closest = point_to_mesh_distance(vertices, faces, queries)
    moved_distance, moved_closest = point_to_mesh_distance(
        vertices + translation, faces, queries + translation
    )
    torch.testing.assert_close(moved_distance, distance)
    torch.testing.assert_close(moved_closest, closest + translation)


def test_noncontiguous_inputs(device: str):
    """Views need not be contiguous."""

    vertices = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        device=device,
        dtype=torch.float64,
    ).transpose(0, 1)
    queries = torch.tensor(
        [[0.2, 0.3], [0.2, 0.3], [1.0, 2.0]],
        device=device,
        dtype=torch.float64,
    ).transpose(0, 1)
    faces = torch.tensor([[0, 0], [1, 1], [2, 2]], device=device).transpose(0, 1)
    assert not vertices.is_contiguous()
    assert not queries.is_contiguous()
    assert not faces.is_contiguous()
    distance, closest = point_to_mesh_distance(vertices, faces, queries)
    torch.testing.assert_close(
        distance, torch.tensor([1.0, 2.0], device=device, dtype=torch.float64)
    )
    torch.testing.assert_close(
        closest,
        torch.tensor(
            [[0.2, 0.2, 0.0], [0.3, 0.3, 0.0]],
            device=device,
            dtype=torch.float64,
        ),
    )


def test_forced_multi_chunk_search(device: str, monkeypatch: pytest.MonkeyPatch):
    """Chunk boundaries do not affect winning correspondences."""

    generator = torch.Generator(device=device).manual_seed(71)
    vertices = torch.rand(18, 3, generator=generator, device=device)
    faces = torch.arange(18, device=device).reshape(6, 3)
    queries = torch.rand(11, 3, generator=generator, device=device)
    reference = point_to_mesh_distance(vertices, faces, queries, squared=True)

    monkeypatch.setattr(torch_impl, "_PAIRWISE_TEMPORARY_BYTE_BUDGET", 384)
    chunked = point_to_mesh_distance(vertices, faces, queries, squared=True)
    torch.testing.assert_close(chunked[0], reference[0])
    torch.testing.assert_close(chunked[1], reference[1])


def test_function_spec_inputs_and_dispatch(device: str):
    """Benchmark generators execute every backend's forward and backward."""

    reference = None
    for implementation in PointToMeshDistance.implementations():
        label, args, kwargs = next(
            iter(PointToMeshDistance.make_inputs_forward(device=device))
        )
        assert label
        output = PointToMeshDistance.dispatch(
            *args, **kwargs, implementation=implementation
        )
        if reference is None:
            reference = output
        else:
            PointToMeshDistance.compare_forward(output, reference)

        _, backward_args, backward_kwargs = next(
            iter(PointToMeshDistance.make_inputs_backward(device=device))
        )
        backward_output = PointToMeshDistance.dispatch(
            *backward_args,
            **backward_kwargs,
            implementation=implementation,
        )
        (backward_output[0].mean() + backward_output[1].square().mean()).backward()
        assert backward_args[0].grad is not None
        assert backward_args[2].grad is not None


def test_torch_compile_fullgraph(device: str):
    """The Torch baseline is capturable as one full graph."""

    vertices, faces = _triangle(device, torch.float32)
    vertices.requires_grad_()
    queries = torch.tensor(
        [[0.2, 0.3, 1.0], [0.4, 0.2, -0.5]],
        device=device,
        requires_grad=True,
    )

    def apply(vertex_values, query_values):
        return point_to_mesh_distance(
            vertex_values,
            faces,
            query_values,
            squared=True,
            implementation="torch",
        )

    compiled = torch.compile(apply, backend="eager", fullgraph=True)
    distance, closest = compiled(vertices, queries)
    (distance.sum() + closest.square().sum()).backward()
    assert vertices.grad is not None
    assert queries.grad is not None


def test_warp_fake_output_contract():
    """The custom op exposes its integer output shape to fake-tensor tracing."""

    vertices = torch.empty(7, 3, dtype=torch.float32, device="meta")
    faces = torch.empty(5, 3, dtype=torch.long, device="meta")
    queries = torch.empty(11, 3, dtype=torch.float32, device="meta")
    output = nearest_face_indices_warp_impl(vertices, faces, queries)
    assert output.shape == (11,)
    assert output.dtype == torch.long
    assert output.device.type == "meta"


def test_warp_forward_and_gradient_parity(device: str):
    """Warp face selection preserves Torch outputs and continuous gradients."""

    base_vertices, base_face = _triangle(device, torch.float32)
    vertices = torch.cat(
        (
            base_vertices,
            base_vertices + torch.tensor([0.0, 0.0, 3.0], device=device),
        )
    )
    faces = torch.cat((base_face, base_face + 3))
    queries = torch.tensor([[0.2, 0.3, 0.7], [0.3, 0.2, 2.6]], device=device)

    def evaluate(implementation: str):
        vertex_values = vertices.detach().clone().requires_grad_()
        query_values = queries.detach().clone().requires_grad_()
        output = point_to_mesh_distance(
            vertex_values,
            faces,
            query_values,
            squared=True,
            implementation=implementation,
        )
        weights = torch.tensor([[0.2, -0.1, 0.3], [-0.3, 0.4, 0.1]], device=device)
        loss = output[0].sum() + (output[1] * weights).sum()
        gradients = torch.autograd.grad(loss, (vertex_values, query_values))
        return output, gradients

    output_torch, gradients_torch = evaluate("torch")
    output_warp, gradients_warp = evaluate("warp")
    PointToMeshDistance.compare_forward(output_warp, output_torch)
    for actual, expected in zip(gradients_warp, gradients_torch, strict=True):
        PointToMeshDistance.compare_backward(actual, expected)

    # Automatic dispatch uses Warp on CUDA and Torch on CPU.
    automatic = point_to_mesh_distance(vertices, faces, queries, squared=True)
    expected_automatic = output_warp if "cuda" in device else output_torch
    PointToMeshDistance.compare_forward(automatic, expected_automatic)


def test_explicit_warp_rejects_float64(device: str):
    """The float32-only Warp BVH path never silently downcasts float64."""

    vertices, faces = _triangle(device, torch.float64)
    queries = torch.tensor([[0.2, 0.3, 1.0]], device=device, dtype=torch.float64)
    with pytest.raises(TypeError, match="only torch.float32"):
        point_to_mesh_distance(vertices, faces, queries, implementation="warp")


def test_warp_custom_opcheck(device: str):
    """The nearest-face custom op satisfies Torch operator contracts."""

    vertices, faces = _triangle(device, torch.float32)
    queries = torch.tensor([[0.2, 0.3, 1.0], [0.3, 0.2, -0.5]], device=device)
    torch.library.opcheck(
        nearest_face_indices_warp_impl,
        args=(vertices, faces, queries),
    )


def test_warp_torch_compile_fullgraph(device: str):
    """The Warp search remains opaque and graph-safe under torch.compile."""

    vertices, faces = _triangle(device, torch.float32)
    queries = torch.tensor([[0.2, 0.3, 1.0], [0.3, 0.2, -0.5]], device=device)

    def apply(vertex_values, query_values):
        return point_to_mesh_distance(
            vertex_values,
            faces,
            query_values,
            squared=True,
            implementation="warp",
        )

    eager = apply(vertices, queries)
    compiled = torch.compile(apply, backend="eager", fullgraph=True)(vertices, queries)
    PointToMeshDistance.compare_forward(compiled, eager)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize(
    "bad_faces",
    [
        torch.tensor([[0, 1, -1]]),
        torch.tensor([[0, 1, 3]]),
    ],
    ids=["negative", "past-end"],
)
def test_connectivity_bounds_backend_parity(
    device: str,
    implementation: str,
    bad_faces: torch.Tensor,
):
    """Neither backend accepts Python-style negative or out-of-range indices."""

    if "cuda" in device:
        pytest.skip("failing torch._assert_async would invalidate the CUDA context")
    vertices, _ = _triangle(device, torch.float32)
    queries = torch.tensor([[0.2, 0.3, 1.0]], device=device)
    with pytest.raises(RuntimeError, match="0 <= index < num_vertices"):
        point_to_mesh_distance(
            vertices,
            bad_faces.to(device),
            queries,
            implementation=implementation,
        )


def test_raw_warp_op_reuses_connectivity_contract(device: str):
    """The low-level custom op is safe when called without the public wrapper."""

    if "cuda" in device:
        pytest.skip("failing torch._assert_async would invalidate the CUDA context")
    vertices, _ = _triangle(device, torch.float32)
    bad_faces = torch.tensor([[0, 1, -1]], device=device)
    queries = torch.tensor([[0.2, 0.3, 1.0]], device=device)
    with pytest.raises(RuntimeError, match="0 <= index < num_vertices"):
        nearest_face_indices_warp_impl(vertices, bad_faces, queries)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("location", ["vertices", "queries"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_coordinates_backend_parity(
    device: str,
    implementation: str,
    location: str,
    nonfinite: float,
):
    """Both backends reject every nonfinite coordinate before searching."""

    if "cuda" in device:
        pytest.skip("failing torch._assert_async would invalidate the CUDA context")
    vertices, faces = _triangle(device, torch.float32)
    queries = torch.tensor([[0.2, 0.3, 1.0]], device=device)
    if location == "vertices":
        vertices = vertices.clone()
        vertices[0, 0] = nonfinite
        match = "mesh_vertices must contain only finite coordinates"
    else:
        queries = queries.clone()
        queries[0, 0] = nonfinite
        match = "input_points must contain only finite coordinates"
    with pytest.raises(RuntimeError, match=match):
        point_to_mesh_distance(vertices, faces, queries, implementation=implementation)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
@pytest.mark.parametrize("degeneracy", ["repeated", "collinear"])
def test_degenerate_faces_backend_parity(
    device: str,
    implementation: str,
    degeneracy: str,
):
    """Every face must span a finite, nonzero-area triangle."""

    if "cuda" in device:
        pytest.skip("failing torch._assert_async would invalidate the CUDA context")
    vertices, faces = _triangle(device, torch.float32)
    if degeneracy == "repeated":
        faces = torch.tensor([[0, 0, 1]], device=device)
    else:
        vertices = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            device=device,
        )
    queries = torch.tensor([[0.2, 0.3, 1.0]], device=device)
    with pytest.raises(RuntimeError, match="nondegenerate with finite area"):
        point_to_mesh_distance(vertices, faces, queries, implementation=implementation)


def test_torch_compile_fullgraph_device_assertions(device: str):
    """Device-side validity assertions remain inside a full Torch graph."""

    if "cuda" in device:
        pytest.skip("failing torch._assert_async would invalidate the CUDA context")
    vertices, faces = _triangle(device, torch.float32)
    queries = torch.tensor([[0.2, 0.3, 1.0]], device=device)

    def apply(vertex_values, face_values, query_values):
        return point_to_mesh_distance(
            vertex_values,
            face_values,
            query_values,
            squared=True,
            implementation="torch",
        )

    compiled = torch.compile(apply, backend="eager", fullgraph=True)
    expected = apply(vertices, faces, queries)
    actual = compiled(vertices, faces, queries)
    PointToMeshDistance.compare_forward(actual, expected)

    invalid_queries = queries.clone()
    invalid_queries[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="input_points.*finite"):
        compiled(vertices, faces, invalid_queries)


@pytest.mark.parametrize(
    ("vertices", "faces", "queries", "squared", "error", "match"),
    [
        (
            torch.zeros(3, 2),
            torch.tensor([[0, 1, 2]]),
            torch.zeros(1, 3),
            False,
            ValueError,
            "mesh_vertices",
        ),
        (
            torch.zeros(3, 3, dtype=torch.float16),
            torch.tensor([[0, 1, 2]]),
            torch.zeros(1, 3, dtype=torch.float16),
            False,
            TypeError,
            "mesh_vertices",
        ),
        (
            torch.zeros(3, 3),
            torch.tensor([0, 1, 2]),
            torch.zeros(1, 3),
            False,
            ValueError,
            "mesh_indices",
        ),
        (
            torch.zeros(3, 3),
            torch.tensor([[0.0, 1.0, 2.0]]),
            torch.zeros(1, 3),
            False,
            TypeError,
            "mesh_indices",
        ),
        (
            torch.zeros(3, 3),
            torch.empty(0, 3, dtype=torch.long),
            torch.zeros(1, 3),
            False,
            ValueError,
            "at least one triangle",
        ),
        (
            torch.zeros(3, 3),
            torch.tensor([[0, 1, 2]]),
            torch.zeros(2, 2),
            False,
            ValueError,
            "input_points",
        ),
        (
            torch.zeros(3, 3, dtype=torch.float32),
            torch.tensor([[0, 1, 2]]),
            torch.zeros(1, 3, dtype=torch.float64),
            False,
            TypeError,
            "same dtype",
        ),
        (
            torch.zeros(3, 3),
            torch.tensor([[0, 1, 2]]),
            torch.zeros(1, 3),
            1,
            TypeError,
            "squared",
        ),
    ],
)
def test_validation_errors(vertices, faces, queries, squared, error, match):
    """Malformed tensor structure and unsupported dtypes fail clearly."""

    with pytest.raises(error, match=match):
        point_to_mesh_distance(vertices, faces, queries, squared=squared)
