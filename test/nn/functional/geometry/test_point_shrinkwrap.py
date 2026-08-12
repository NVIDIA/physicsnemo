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

"""Functional tests for differentiable nearest-surface shrinkwrapping."""

from __future__ import annotations

import inspect
import math

import pytest
import torch

import physicsnemo.nn.functional as functional
from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.nn.functional import shrinkwrap_points
from physicsnemo.nn.functional.geometry import ShrinkwrapPoints


def _triangle(
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    *,
    reversed_winding: bool = False,
    index_dtype: torch.dtype = torch.int64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a right triangle in the xy plane with a well-defined normal."""

    target_points = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        device=device,
        dtype=dtype,
    )
    indices = [0, 2, 1] if reversed_winding else [0, 1, 2]
    target_faces = torch.tensor(
        [indices],
        device=device,
        dtype=index_dtype,
    )
    return target_points, target_faces


def _differentiable_case(
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Create a smooth single-face case with every floating input trainable."""

    points = torch.tensor(
        [[0.31, 0.43, 0.82], [0.74, 0.37, 0.61]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    target_points = torch.tensor(
        [[-0.1, -0.2, 0.03], [2.2, 0.1, 0.08], [0.2, 2.1, -0.04]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    target_faces = torch.tensor(
        [[0, 1, 2]],
        device=device,
        dtype=torch.int64,
    )
    point_weights = torch.tensor(
        [0.63, -0.27],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    offset = torch.tensor(
        0.11,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    return points, target_points, target_faces, point_weights, offset


def test_public_api_and_function_spec_registration():
    assert shrinkwrap_points.__name__ == "shrinkwrap_points"
    assert shrinkwrap_points.__module__ == (
        "physicsnemo.nn.functional.geometry.deform.shrinkwrap"
    )
    assert functional.shrinkwrap_points is shrinkwrap_points
    assert not hasattr(functional, "ShrinkwrapPoints")
    assert issubclass(ShrinkwrapPoints, FunctionSpec)
    assert list(inspect.signature(shrinkwrap_points).parameters) == [
        "points",
        "target_points",
        "target_faces",
        "offset",
        "max_distance",
        "point_weights",
        "implementation",
    ]
    assert set(ShrinkwrapPoints.implementations()) == {"torch", "warp"}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_triangle_interior_edges_and_vertices(device: str, dtype: torch.dtype):
    """The reference backend must cover every closest-triangle feature."""

    device = torch.device(device)
    target_points, target_faces = _triangle(device, dtype)
    points = torch.tensor(
        [
            [0.50, 0.40, 1.00],  # face interior
            [0.75, -0.60, 1.00],  # edge AB
            [-0.50, 0.80, 1.00],  # edge AC
            [1.30, 1.30, 1.00],  # edge BC
            [-0.40, -0.30, 1.00],  # vertex A
            [2.50, -0.10, 1.00],  # vertex B
            [-0.20, 2.40, 1.00],  # vertex C
        ],
        device=device,
        dtype=dtype,
    )
    expected = torch.tensor(
        [
            [0.50, 0.40, 0.00],
            [0.75, 0.00, 0.00],
            [0.00, 0.80, 0.00],
            [1.00, 1.00, 0.00],
            [0.00, 0.00, 0.00],
            [2.00, 0.00, 0.00],
            [0.00, 2.00, 0.00],
        ],
        device=device,
        dtype=dtype,
    )

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        implementation="torch",
    )

    assert output.shape == points.shape
    assert output.dtype == dtype
    assert output.device == device
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_accepts_both_public_index_dtypes(device: str, index_dtype: torch.dtype):
    device = torch.device(device)
    target_points, target_faces = _triangle(
        device,
        torch.float32,
        index_dtype=index_dtype,
    )
    points = torch.tensor([[0.25, 0.50, 1.0]], device=device)

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        implementation="torch",
    )

    torch.testing.assert_close(
        output,
        torch.tensor([[0.25, 0.50, 0.0]], device=device),
    )


@pytest.mark.parametrize("offset", [0.25, -0.40])
@pytest.mark.parametrize("reversed_winding", [False, True])
def test_signed_offset_follows_target_winding(
    device: str,
    offset: float,
    reversed_winding: bool,
):
    device = torch.device(device)
    target_points, target_faces = _triangle(
        device,
        reversed_winding=reversed_winding,
    )
    points = torch.tensor([[0.40, 0.35, 1.0]], device=device)
    normal_sign = -1.0 if reversed_winding else 1.0

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        offset=offset,
        implementation="torch",
    )

    expected = torch.tensor(
        [[0.40, 0.35, normal_sign * offset]],
        device=device,
    )
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bool_and_unclamped_float_point_weights(device: str, dtype: torch.dtype):
    device = torch.device(device)
    target_points, target_faces = _triangle(device, dtype)
    bool_points = torch.tensor(
        [[0.2, 0.2, 1.0], [0.3, 0.3, 1.0]],
        device=device,
        dtype=dtype,
    )

    bool_output = shrinkwrap_points(
        bool_points,
        target_points,
        target_faces,
        point_weights=torch.tensor([True, False], device=device),
        implementation="torch",
    )
    torch.testing.assert_close(
        bool_output,
        torch.tensor(
            [[0.2, 0.2, 0.0], [0.3, 0.3, 1.0]],
            device=device,
            dtype=dtype,
        ),
    )

    weights = torch.tensor(
        [-0.5, 0.0, 0.25, 1.0, 1.5],
        device=device,
        dtype=dtype,
    )
    float_points = torch.stack(
        (
            torch.linspace(0.2, 0.6, weights.numel(), device=device, dtype=dtype),
            torch.full_like(weights, 0.2),
            torch.ones_like(weights),
        ),
        dim=-1,
    )
    float_output = shrinkwrap_points(
        float_points,
        target_points,
        target_faces,
        point_weights=weights,
        implementation="torch",
    )
    expected = float_points.clone()
    expected[:, 2] = 1.0 - weights
    torch.testing.assert_close(float_output, expected)


def test_batched_sources_share_one_target_without_weight_broadcasting(device: str):
    device = torch.device(device)
    target_points, target_faces = _triangle(device)
    points = torch.tensor(
        [
            [[0.2, 0.3, 1.0], [0.5, 0.4, 2.0]],
            [[0.4, 0.2, -1.0], [0.6, 0.3, -2.0]],
        ],
        device=device,
    )
    point_weights = torch.tensor(
        [[1.0, 0.25], [0.0, 0.75]],
        device=device,
    )

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        point_weights=point_weights,
        implementation="torch",
    )
    expected = points.clone()
    expected[..., 2] = points[..., 2] * (1.0 - point_weights)

    assert output.shape == (2, 2, 3)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize(
    ("points_shape", "weights_shape"),
    [((0, 3), (0,)), ((2, 0, 3), (2, 0))],
)
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_empty_source_is_supported(
    device: str,
    points_shape: tuple[int, ...],
    weights_shape: tuple[int, ...],
    implementation: str,
):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    target_points, target_faces = _triangle(device)
    points = torch.empty(points_shape, device=device)
    weights = torch.empty(weights_shape, device=device)

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        point_weights=weights,
        implementation=implementation,
    )

    assert output.shape == points.shape
    assert output.dtype == points.dtype
    assert output.device == points.device
    assert output.numel() == 0


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_max_distance_hit_miss_and_exact_boundary(
    device: str,
    implementation: str,
):
    """The search radius is strict: a point exactly on it is a miss."""

    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    target_points, target_faces = _triangle(device)
    points = torch.tensor(
        [
            [0.25, 0.25, 0.49],
            [0.25, 0.25, 0.50],
            [0.25, 0.25, 0.51],
        ],
        device=device,
    )

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        max_distance=0.5,
        implementation=implementation,
    )
    expected = points.clone()
    expected[0, 2] = 0.0

    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_float64_point_just_inside_cutoff_is_a_hit(
    device: str,
    implementation: str,
):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    dtype = torch.float64
    target_points, target_faces = _triangle(device, dtype)
    cutoff = torch.tensor(1.0, device=device, dtype=dtype)
    inside = torch.nextafter(cutoff, torch.zeros_like(cutoff))
    points = torch.tensor(
        [[0.25, 0.25, 0.0]],
        device=device,
        dtype=dtype,
    )
    points[0, 2] = inside

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        max_distance=1.0,
        implementation=implementation,
    )

    expected = points.clone()
    expected[0, 2] = 0.0
    torch.testing.assert_close(output, expected)


def test_max_distance_is_measured_before_offset(device: str):
    device = torch.device(device)
    target_points, target_faces = _triangle(device)
    points = torch.tensor([[0.25, 0.25, 0.4]], device=device)

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        max_distance=0.5,
        offset=0.8,
        implementation="torch",
    )

    torch.testing.assert_close(
        output,
        torch.tensor([[0.25, 0.25, 0.8]], device=device),
    )


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("points_not_tensor", TypeError, "points must be a torch.Tensor"),
        ("points_rank", ValueError, "points must have shape"),
        ("points_coordinates", ValueError, "points must have three coordinates"),
        ("target_not_tensor", TypeError, "target_points must be a torch.Tensor"),
        ("target_rank", ValueError, "target_points must have shape"),
        ("target_coordinates", ValueError, "target_points must have shape"),
        ("faces_not_tensor", TypeError, "target_faces must be a torch.Tensor"),
        ("faces_rank", ValueError, "target_faces must have shape"),
        ("faces_coordinates", ValueError, "target_faces must have shape"),
        ("faces_empty", ValueError, "at least one triangle"),
    ],
)
def test_rejects_invalid_shapes_and_container_types(
    case: str,
    error: type[Exception],
    match: str,
):
    points = torch.zeros((2, 3))
    target_points, target_faces = _triangle()
    arguments: dict[str, object] = {
        "points": points,
        "target_points": target_points,
        "target_faces": target_faces,
    }
    replacements: dict[str, tuple[str, object]] = {
        "points_not_tensor": ("points", [[0.0, 0.0, 0.0]]),
        "points_rank": ("points", torch.zeros((1, 1, 2, 3))),
        "points_coordinates": ("points", torch.zeros((2, 2))),
        "target_not_tensor": (
            "target_points",
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ),
        "target_rank": ("target_points", torch.zeros((1, 3, 3))),
        "target_coordinates": ("target_points", torch.zeros((3, 2))),
        "faces_not_tensor": ("target_faces", [[0, 1, 2]]),
        "faces_rank": ("target_faces", torch.tensor([0, 1, 2])),
        "faces_coordinates": ("target_faces", torch.zeros((1, 4), dtype=torch.long)),
        "faces_empty": ("target_faces", torch.empty((0, 3), dtype=torch.long)),
    }
    name, value = replacements[case]
    arguments[name] = value

    with pytest.raises(error, match=match):
        shrinkwrap_points(**arguments, implementation="torch")


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("points_dtype", TypeError, "points must have dtype"),
        (
            "target_dtype",
            TypeError,
            "points and target_points must have the same dtype",
        ),
        ("faces_dtype", TypeError, "target_faces must have dtype"),
        ("weights_not_tensor", TypeError, "point_weights must be a torch.Tensor"),
        ("weights_dtype", TypeError, "point_weights must have bool dtype"),
        ("weights_shape", ValueError, "point_weights must have shape"),
        ("batched_weights_shape", ValueError, "point_weights must have shape"),
        ("offset_dtype", TypeError, "offset and points must have the same dtype"),
        ("offset_shape", ValueError, "tensor-valued offset must be scalar"),
        ("offset_type", TypeError, "offset must be a real scalar"),
    ],
)
def test_rejects_invalid_dtypes_weights_and_offsets(
    case: str,
    error: type[Exception],
    match: str,
):
    points = torch.zeros((2, 3))
    target_points, target_faces = _triangle()
    kwargs: dict[str, object] = {}
    if case == "points_dtype":
        points = points.to(torch.float16)
        target_points = target_points.to(torch.float16)
    elif case == "target_dtype":
        target_points = target_points.to(torch.float64)
    elif case == "faces_dtype":
        target_faces = target_faces.to(torch.float32)
    elif case == "weights_not_tensor":
        kwargs["point_weights"] = [1.0, 0.0]
    elif case == "weights_dtype":
        kwargs["point_weights"] = torch.ones(2, dtype=torch.int64)
    elif case == "weights_shape":
        kwargs["point_weights"] = torch.ones((1, 2))
    elif case == "batched_weights_shape":
        points = points.unsqueeze(0).expand(2, -1, -1)
        kwargs["point_weights"] = torch.ones(2)
    elif case == "offset_dtype":
        kwargs["offset"] = torch.tensor(0.1, dtype=torch.float64)
    elif case == "offset_shape":
        kwargs["offset"] = torch.tensor([0.1])
    elif case == "offset_type":
        kwargs["offset"] = True

    with pytest.raises(error, match=match):
        shrinkwrap_points(
            points,
            target_points,
            target_faces,
            implementation="torch",
            **kwargs,
        )


@pytest.mark.parametrize(
    ("max_distance", "error", "match"),
    [
        (0.0, ValueError, "positive and finite"),
        (-1.0, ValueError, "positive and finite"),
        (math.inf, ValueError, "positive and finite"),
        (math.nan, ValueError, "positive and finite"),
        (True, TypeError, "positive finite real scalar"),
        ("1.0", TypeError, "positive finite real scalar"),
        (torch.tensor(1.0), TypeError, "positive finite real scalar"),
        (1.0e-50, ValueError, "positive in the points dtype"),
        (1.0e40, ValueError, "finite in the points dtype"),
    ],
)
def test_rejects_invalid_max_distance(
    max_distance: object,
    error: type[Exception],
    match: str,
):
    points = torch.zeros((1, 3))
    target_points, target_faces = _triangle()

    with pytest.raises(error, match=match):
        shrinkwrap_points(
            points,
            target_points,
            target_faces,
            max_distance=max_distance,
            implementation="torch",
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_smallest_positive_cutoff_remains_a_hit(
    device: str,
    dtype: torch.dtype,
    implementation: str,
):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    target_points, target_faces = _triangle(device, dtype)
    points = torch.tensor([[0.25, 0.25, 0.0]], device=device, dtype=dtype)
    smallest_positive = torch.nextafter(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
    ).item()

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        max_distance=smallest_positive,
        offset=1.0,
        implementation=implementation,
    )

    torch.testing.assert_close(
        output,
        torch.tensor([[0.25, 0.25, 1.0]], device=device, dtype=dtype),
    )


@pytest.mark.parametrize("offset", [math.inf, -math.inf, math.nan, 1.0e100])
def test_rejects_nonfinite_or_unrepresentable_python_offsets(offset: float):
    points = torch.zeros((1, 3))
    target_points, target_faces = _triangle()

    with pytest.raises(ValueError, match="offset must be finite"):
        shrinkwrap_points(
            points,
            target_points,
            target_faces,
            offset=offset,
            implementation="torch",
        )


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_rejects_nonfinite_tensor_offsets(value: float):
    points = torch.zeros((1, 3))
    target_points, target_faces = _triangle()

    with pytest.raises(ValueError, match="offset must be finite"):
        shrinkwrap_points(
            points,
            target_points,
            target_faces,
            offset=torch.tensor(value),
            implementation="torch",
        )


@pytest.mark.parametrize(
    "target_faces",
    [
        torch.tensor([[-1, 1, 2]], dtype=torch.int64),
        torch.tensor([[0, 1, 3]], dtype=torch.int64),
    ],
)
def test_rejects_out_of_range_target_indices(target_faces: torch.Tensor):
    points = torch.zeros((1, 3))
    target_points, _ = _triangle()

    with pytest.raises(ValueError, match="indices outside the target point range"):
        shrinkwrap_points(
            points,
            target_points,
            target_faces,
            implementation="torch",
        )


@pytest.mark.parametrize("invalid_value", [math.nan, math.inf, -math.inf])
def test_rejects_nonfinite_source_points(invalid_value: float):
    points = torch.tensor([[0.25, 0.25, invalid_value]])
    target_points, target_faces = _triangle()

    with pytest.raises(ValueError, match="points must contain only finite"):
        shrinkwrap_points(
            points,
            target_points,
            target_faces,
            implementation="torch",
        )


def test_rejects_target_when_all_faces_are_invalid():
    points = torch.tensor([[0.25, 0.25, 1.0]])
    target_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],  # collinear first face
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, math.nan, 0.0],  # nonfinite second face
        ]
    )
    target_faces = torch.tensor([[0, 1, 2], [3, 4, 5]])

    with pytest.raises(
        ValueError,
        match="at least one finite, nondegenerate triangle",
    ):
        shrinkwrap_points(
            points,
            target_points,
            target_faces,
            implementation="torch",
        )


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_invalid_face_is_ignored_when_target_has_one_valid_face(
    device: str,
    implementation: str,
):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    points = torch.tensor([[0.35, 0.25, 0.8]], device=device)
    target_points = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, math.nan, 2.0],  # invalid face comes first
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        device=device,
    )
    target_faces = torch.tensor(
        [[0, 1, 2], [3, 4, 5]],
        device=device,
        dtype=torch.int64,
    )

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        implementation=implementation,
    )

    torch.testing.assert_close(
        output,
        torch.tensor([[0.35, 0.25, 0.0]], device=device),
    )


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_miss_uses_finite_replay_face_for_backward(
    device: str,
    implementation: str,
):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    points = torch.tensor(
        [[0.35, 0.25, 3.0]],
        device=device,
        requires_grad=True,
    )
    target_points = torch.tensor(
        [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, math.nan, 2.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        device=device,
        requires_grad=True,
    )
    target_faces = torch.tensor(
        [[0, 1, 2], [3, 4, 5]],
        device=device,
    )
    weights = torch.tensor([0.7], device=device, requires_grad=True)
    offset = torch.tensor(0.2, device=device, requires_grad=True)

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        point_weights=weights,
        offset=offset,
        max_distance=0.1,
        implementation=implementation,
    )
    output.sum().backward()

    torch.testing.assert_close(output, points.detach())
    torch.testing.assert_close(points.grad, torch.ones_like(points))
    for tensor in (target_points, weights, offset):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        torch.testing.assert_close(tensor.grad, torch.zeros_like(tensor))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("target_points", "points and target_points must be on the same device"),
        ("target_faces", "target_faces and points must be on the same device"),
        ("point_weights", "point_weights and points must be on the same device"),
        ("offset", "offset and points must be on the same device"),
    ],
)
def test_rejects_mixed_device_inputs(case: str, match: str):
    points = torch.zeros((1, 3), device="cpu")
    target_points, target_faces = _triangle("cpu")
    kwargs: dict[str, object] = {}
    if case == "target_points":
        target_points = target_points.to("cuda")
    elif case == "target_faces":
        target_faces = target_faces.to("cuda")
    elif case == "point_weights":
        kwargs["point_weights"] = torch.ones(1, device="cuda")
    elif case == "offset":
        kwargs["offset"] = torch.tensor(0.1, device="cuda")

    with pytest.raises(ValueError, match=match):
        shrinkwrap_points(
            points,
            target_points,
            target_faces,
            implementation="torch",
            **kwargs,
        )


def test_known_first_order_gradients_cover_all_differentiable_inputs():
    dtype = torch.float64
    points = torch.tensor(
        [[0.25, 0.35, 0.8]],
        dtype=dtype,
        requires_grad=True,
    )
    target_points, target_faces = _triangle(dtype=dtype)
    target_points.requires_grad_()
    point_weights = torch.tensor([0.6], dtype=dtype, requires_grad=True)
    offset = torch.tensor(0.1, dtype=dtype, requires_grad=True)
    cotangent = torch.tensor([[0.4, -0.3, 2.0]], dtype=dtype)

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        point_weights=point_weights,
        offset=offset,
        implementation="torch",
    )
    gradients = torch.autograd.grad(
        (output * cotangent).sum(),
        (points, target_points, point_weights, offset),
    )
    point_gradient, target_gradient, weight_gradient, offset_gradient = gradients

    torch.testing.assert_close(
        point_gradient,
        torch.tensor([[0.4, -0.3, 0.8]], dtype=dtype),
    )
    torch.testing.assert_close(
        weight_gradient,
        torch.tensor([-1.4], dtype=dtype),
    )
    torch.testing.assert_close(offset_gradient, torch.tensor(1.2, dtype=dtype))
    assert torch.isfinite(target_gradient).all()
    assert torch.count_nonzero(target_gradient) > 0


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_gradcheck_source_target_weights_and_tensor_offset(implementation: str):
    if implementation == "warp":
        pytest.importorskip("warp")
    inputs = _differentiable_case("cpu", torch.float64)
    points, target_points, target_faces, point_weights, offset = inputs

    def operation(p, t, w, o):
        return shrinkwrap_points(
            p,
            t,
            target_faces,
            point_weights=w,
            offset=o,
            implementation=implementation,
        )

    assert torch.autograd.gradcheck(
        operation,
        (points, target_points, point_weights, offset),
        eps=1.0e-6,
        atol=3.0e-5,
        rtol=3.0e-4,
    )


def test_missed_projection_has_identity_and_zero_parameter_gradients():
    dtype = torch.float64
    points = torch.tensor(
        [[0.25, 0.25, 2.0]],
        dtype=dtype,
        requires_grad=True,
    )
    target_points, target_faces = _triangle(dtype=dtype)
    target_points.requires_grad_()
    point_weights = torch.tensor([0.6], dtype=dtype, requires_grad=True)
    offset = torch.tensor(0.1, dtype=dtype, requires_grad=True)

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        max_distance=0.5,
        point_weights=point_weights,
        offset=offset,
        implementation="torch",
    )
    gradients = torch.autograd.grad(
        output.sum(),
        (points, target_points, point_weights, offset),
    )

    torch.testing.assert_close(output, points)
    torch.testing.assert_close(gradients[0], torch.ones_like(points))
    torch.testing.assert_close(gradients[1], torch.zeros_like(target_points))
    torch.testing.assert_close(gradients[2], torch.zeros_like(point_weights))
    torch.testing.assert_close(gradients[3], torch.zeros_like(offset))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_torch_warp_forward_parity(device: str, dtype: torch.dtype):
    pytest.importorskip("warp")
    device = torch.device(device)
    target_points, target_faces = _triangle(device, dtype)
    points = torch.tensor(
        [
            [0.35, 0.45, 0.60],
            [0.80, -0.40, 0.30],
            [-0.30, -0.40, 0.20],
            [0.40, 0.40, 2.00],
        ],
        device=device,
        dtype=dtype,
    )
    weights = torch.tensor(
        [0.2, 0.8, -0.5, 1.0],
        device=device,
        dtype=dtype,
    )
    kwargs = {
        "offset": 0.07,
        "max_distance": 1.25,
        "point_weights": weights,
    }

    torch_output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        implementation="torch",
        **kwargs,
    )
    warp_output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        implementation="warp",
        **kwargs,
    )

    ShrinkwrapPoints.compare_forward(warp_output, torch_output)


@pytest.mark.parametrize(
    ("dtype", "coordinate", "max_distance", "expect_far_hit"),
    [
        (torch.float32, 1.0e20, None, True),
        (torch.float32, 1.0e20, 2.0e20, True),
        (torch.float32, 1.0e20, 1.0e20, False),
        (torch.float64, 1.0e300, None, True),
        (torch.float64, 1.0e300, 2.0e300, True),
        (torch.float64, 1.0e300, 1.0e300, False),
    ],
)
def test_warp_far_queries_preserve_hits(
    device: str,
    dtype: torch.dtype,
    coordinate: float,
    max_distance: float | None,
    expect_far_hit: bool,
):
    pytest.importorskip("warp")
    device = torch.device(device)
    target_points, target_faces = _triangle(device, dtype)
    points = torch.tensor(
        [
            [0.25, 0.25, 0.5],
            [coordinate, 0.25, 0.5],
        ],
        device=device,
        dtype=dtype,
    )

    torch_output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        max_distance=max_distance,
        implementation="torch",
    )
    warp_output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        max_distance=max_distance,
        implementation="warp",
    )

    if expect_far_hit:
        assert not torch.equal(torch_output[1], points[1])
    else:
        assert torch.equal(torch_output[1], points[1])
    ShrinkwrapPoints.compare_forward(warp_output, torch_output)


@pytest.mark.parametrize(
    ("dtype", "triangle_scale", "normal_distance"),
    [
        (torch.float32, 1.0, 1.0e10),
        (torch.float32, 1.0e-20, 1.0e20),
        (torch.float64, 1.0, 1.0e100),
        (torch.float64, 1.0e-100, 1.0e300),
    ],
)
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_far_normal_query_preserves_tangential_projection_and_gradient(
    device: str,
    dtype: torch.dtype,
    triangle_scale: float,
    normal_distance: float,
    implementation: str,
):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    target_points, target_faces = _triangle(device, dtype)
    target_points = target_points * triangle_scale
    points = torch.tensor(
        [[0.25 * triangle_scale, 0.25 * triangle_scale, normal_distance]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        implementation=implementation,
    )
    point_gradient = torch.autograd.grad(output.sum(), points)[0]

    torch.testing.assert_close(
        output,
        torch.tensor(
            [[0.25 * triangle_scale, 0.25 * triangle_scale, 0.0]],
            device=device,
            dtype=dtype,
        ),
    )
    torch.testing.assert_close(
        point_gradient,
        torch.tensor([[1.0, 1.0, 0.0]], device=device, dtype=dtype),
    )


@pytest.mark.parametrize(
    (
        "dtype",
        "edge_length",
        "altitude",
        "normal_distance",
        "relative_tolerance",
        "absolute_tolerance",
    ),
    [
        (torch.float32, 1.0e8, 1.0e3, 1.0e8, 1.0e-5, 1.0e-4),
        (torch.float64, 1.0, 1.0e-10, 1.0, 1.0e-12, 1.0e-13),
    ],
)
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_skinny_valid_triangle_preserves_interior_projection_and_gradient(
    device: str,
    dtype: torch.dtype,
    edge_length: float,
    altitude: float,
    normal_distance: float,
    relative_tolerance: float,
    absolute_tolerance: float,
    implementation: str,
):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    target_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [edge_length, 0.0, 0.0],
            [edge_length, altitude, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    target_faces = torch.tensor([[0, 1, 2]], device=device)
    points = torch.tensor(
        [[0.8 * edge_length, 0.5 * altitude, normal_distance]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        implementation=implementation,
    )
    point_gradient = torch.autograd.grad(output.sum(), points)[0]

    torch.testing.assert_close(
        output,
        torch.tensor(
            [[0.8 * edge_length, 0.5 * altitude, 0.0]],
            device=device,
            dtype=dtype,
        ),
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    torch.testing.assert_close(
        point_gradient,
        torch.tensor([[1.0, 1.0, 0.0]], device=device, dtype=dtype),
        rtol=1.0e-3,
        atol=1.0e-3,
    )


@pytest.mark.parametrize(
    ("dtype", "magnitude"),
    [
        (torch.float32, 3.0e38),
        (torch.float64, 9.0e307),
    ],
)
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_opposite_sign_finite_coordinates_do_not_overflow_to_a_miss(
    device: str,
    dtype: torch.dtype,
    magnitude: float,
    implementation: str,
):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    target_points = torch.tensor(
        [
            [-magnitude, 0.0, 0.0],
            [-magnitude, 1.0, 0.0],
            [-magnitude, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    target_faces = torch.tensor([[0, 1, 2]], device=device)
    points = torch.tensor(
        [[magnitude, 0.25, 0.25]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        implementation=implementation,
    )
    point_gradient = torch.autograd.grad(output.sum(), points)[0]

    torch.testing.assert_close(
        output,
        torch.tensor(
            [[-magnitude, 0.25, 0.25]],
            device=device,
            dtype=dtype,
        ),
    )
    torch.testing.assert_close(
        point_gradient,
        torch.tensor([[0.0, 1.0, 1.0]], device=device, dtype=dtype),
    )


@pytest.mark.parametrize(
    ("dtype", "magnitude"),
    [
        (torch.float32, 3.0e38),
        (torch.float64, 9.0e307),
    ],
)
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_opposite_sign_triangle_edges_remain_valid(
    device: str,
    dtype: torch.dtype,
    magnitude: float,
    implementation: str,
):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    target_points = torch.tensor(
        [
            [-magnitude, 0.0, 0.0],
            [magnitude, 0.0, 0.0],
            [0.0, magnitude, 0.0],
        ],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    target_faces = torch.tensor([[0, 1, 2]], device=device)
    points = torch.tensor(
        [[0.0, 0.25 * magnitude, 0.5 * magnitude]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        implementation=implementation,
    )
    point_gradient, target_gradient = torch.autograd.grad(
        output.sum(),
        (points, target_points),
    )

    torch.testing.assert_close(
        output,
        torch.tensor(
            [[0.0, 0.25 * magnitude, 0.0]],
            device=device,
            dtype=dtype,
        ),
    )
    torch.testing.assert_close(
        point_gradient,
        torch.tensor([[1.0, 1.0, 0.0]], device=device, dtype=dtype),
    )
    torch.testing.assert_close(
        target_gradient,
        torch.tensor(
            [
                [0.0, 0.0, -0.125],
                [0.0, 0.0, 0.375],
                [0.0, 0.0, 0.75],
            ],
            device=device,
            dtype=dtype,
        ),
    )


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_unbounded_extreme_distances_preserve_nearest_face_order(
    device: str,
    implementation: str,
):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    target_points = torch.tensor(
        [
            [-3.0e38, 0.0, 0.0],
            [-3.0e38, 1.0, 0.0],
            [-3.0e38, 0.0, 1.0],
            [-2.0e38, 0.0, 0.0],
            [-2.0e38, 1.0, 0.0],
            [-2.0e38, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    target_faces = torch.tensor(
        [[0, 1, 2], [3, 4, 5]],
        device=device,
    )
    points = torch.tensor(
        [[3.0e38, 0.25, 0.25]],
        device=device,
        dtype=torch.float32,
    )

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        implementation=implementation,
    )

    torch.testing.assert_close(
        output,
        torch.tensor(
            [[-2.0e38, 0.25, 0.25]],
            device=device,
            dtype=torch.float32,
        ),
    )


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_float64_nearby_surfaces_preserve_nearest_face_order(
    device: str,
    implementation: str,
):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    target_points = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [0.0, 0.0, 1.00000003],
            [2.0, 0.0, 1.00000003],
            [0.0, 2.0, 1.00000003],
            [0.0, 0.0, -1.0],
            [2.0, 0.0, -1.0],
            [0.0, 2.0, -1.0],
        ],
        device=device,
        dtype=torch.float64,
    )
    target_faces = torch.tensor(
        [[0, 1, 2], [3, 5, 4], [6, 7, 8]],
        device=device,
    )
    points = torch.tensor(
        [[0.25, 0.25, 1.0000000003]],
        device=device,
        dtype=torch.float64,
    )

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        offset=0.1,
        implementation=implementation,
    )

    torch.testing.assert_close(
        output,
        torch.tensor(
            [[0.25, 0.25, 1.1]],
            device=device,
            dtype=torch.float64,
        ),
    )


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_float32_search_does_not_shift_nearby_surface_boundary(
    device: str,
    implementation: str,
):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    lower_z = 1.0
    upper_z = 1.0000003576278687
    query_z = 1.000000238418579
    target_points = torch.tensor(
        [
            [0.0, 0.0, lower_z],
            [2.0, 0.0, lower_z],
            [0.0, 2.0, lower_z],
            [0.0, 0.0, upper_z],
            [2.0, 0.0, upper_z],
            [0.0, 2.0, upper_z],
            [0.0, 0.0, -1.0],
            [2.0, 0.0, -1.0],
            [0.0, 2.0, -1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    target_faces = torch.tensor(
        [[0, 1, 2], [3, 5, 4], [6, 7, 8]],
        device=device,
    )
    points = torch.tensor(
        [[0.25, 0.25, query_z]],
        device=device,
        dtype=torch.float32,
    )

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        offset=0.1,
        implementation=implementation,
    )

    torch.testing.assert_close(
        output,
        torch.tensor(
            [[0.25, 0.25, upper_z - 0.1]],
            device=device,
            dtype=torch.float32,
        ),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("max_distance", [None, 0.2])
def test_warp_multiscale_target_preserves_small_components(
    device: str,
    dtype: torch.dtype,
    max_distance: float | None,
):
    pytest.importorskip("warp")
    device = torch.device(device)
    remote_scale = 5.0e7
    target_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
            [remote_scale, 0.0, 0.0],
            [remote_scale, remote_scale, 0.0],
            [remote_scale, 0.0, remote_scale],
        ],
        device=device,
        dtype=dtype,
    )
    target_faces = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        device=device,
    )
    points = torch.tensor(
        [[10.25, 0.25, 0.1]],
        device=device,
        dtype=dtype,
    )

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        max_distance=max_distance,
        implementation="warp",
    )

    torch.testing.assert_close(
        output,
        torch.tensor([[10.25, 0.25, 0.0]], device=device, dtype=dtype),
    )


def test_warp_extreme_multiscale_target_preserves_local_component(device: str):
    pytest.importorskip("warp")
    device = torch.device(device)
    dtype = torch.float64
    remote_scale = 1.0e200
    target_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
            [remote_scale, 0.0, 0.0],
            [remote_scale, remote_scale, 0.0],
            [remote_scale, 0.0, remote_scale],
        ],
        device=device,
        dtype=dtype,
    )
    target_faces = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        device=device,
    )
    points = torch.tensor(
        [[10.25, 0.25, 0.1]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        max_distance=0.2,
        implementation="warp",
    )
    point_gradient = torch.autograd.grad(output.sum(), points)[0]

    torch.testing.assert_close(
        output,
        torch.tensor([[10.25, 0.25, 0.0]], device=device, dtype=dtype),
    )
    torch.testing.assert_close(
        point_gradient,
        torch.tensor([[1.0, 1.0, 0.0]], device=device, dtype=dtype),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_multiscale_target_preserves_close_parallel_components(
    device: str,
    dtype: torch.dtype,
    implementation: str,
):
    if implementation == "warp":
        pytest.importorskip("warp")
    device = torch.device(device)
    separation = 1.0e-3
    remote_scale = 1.0e5
    target_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, separation],
            [1.0, 0.0, separation],
            [0.0, 1.0, separation],
            [remote_scale, 0.0, 0.0],
            [remote_scale, remote_scale, 0.0],
            [remote_scale, 0.0, remote_scale],
        ],
        device=device,
        dtype=dtype,
    )
    target_faces = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        device=device,
    )
    points = torch.tensor(
        [[0.25, 0.25, separation]],
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    output = shrinkwrap_points(
        points,
        target_points,
        target_faces,
        offset=0.1,
        max_distance=0.5 * separation,
        implementation=implementation,
    )
    point_gradient = torch.autograd.grad(output.sum(), points)[0]

    torch.testing.assert_close(
        output,
        torch.tensor(
            [[0.25, 0.25, separation + 0.1]],
            device=device,
            dtype=dtype,
        ),
    )
    torch.testing.assert_close(
        point_gradient,
        torch.tensor([[1.0, 1.0, 0.0]], device=device, dtype=dtype),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_torch_warp_gradient_parity(device: str, dtype: torch.dtype):
    pytest.importorskip("warp")
    device = torch.device(device)
    torch_inputs = _differentiable_case(device, dtype)
    warp_inputs = tuple(
        tensor.detach().clone().requires_grad_(tensor.requires_grad)
        for tensor in torch_inputs
    )
    cotangent = torch.tensor(
        [[0.2, -0.4, 0.7], [-0.3, 0.6, -0.1]],
        device=device,
        dtype=dtype,
    )

    def output_and_gradients(inputs: tuple[torch.Tensor, ...], implementation: str):
        points, target_points, target_faces, point_weights, offset = inputs
        output = shrinkwrap_points(
            points,
            target_points,
            target_faces,
            point_weights=point_weights,
            offset=offset,
            implementation=implementation,
        )
        gradients = torch.autograd.grad(
            (output * cotangent).sum(),
            (points, target_points, point_weights, offset),
        )
        return output, gradients

    torch_output, torch_gradients = output_and_gradients(torch_inputs, "torch")
    warp_output, warp_gradients = output_and_gradients(warp_inputs, "warp")

    ShrinkwrapPoints.compare_forward(warp_output, torch_output)
    for warp_gradient, torch_gradient in zip(
        warp_gradients,
        torch_gradients,
        strict=True,
    ):
        ShrinkwrapPoints.compare_backward(warp_gradient, torch_gradient)


def test_benchmark_inputs_satisfy_backend_comparators(device: str):
    pytest.importorskip("warp")
    device = torch.device(device)

    for _, args, kwargs in ShrinkwrapPoints.make_inputs_forward(device):
        torch_output = shrinkwrap_points(
            *args,
            **kwargs,
            implementation="torch",
        )
        warp_output = shrinkwrap_points(
            *args,
            **kwargs,
            implementation="warp",
        )
        ShrinkwrapPoints.compare_forward(warp_output, torch_output)

    _, args, kwargs = next(ShrinkwrapPoints.make_inputs_backward(device))

    def output_and_gradients(implementation: str):
        cloned_args = tuple(
            value.detach().clone().requires_grad_(value.requires_grad) for value in args
        )
        cloned_kwargs = {
            key: value.detach().clone().requires_grad_(value.requires_grad)
            for key, value in kwargs.items()
        }
        output = shrinkwrap_points(
            *cloned_args,
            **cloned_kwargs,
            implementation=implementation,
        )
        trainable = tuple(
            value
            for value in (*cloned_args, *cloned_kwargs.values())
            if value.requires_grad
        )
        gradients = torch.autograd.grad(output.sum(), trainable)
        return output, gradients

    torch_output, torch_gradients = output_and_gradients("torch")
    warp_output, warp_gradients = output_and_gradients("warp")
    ShrinkwrapPoints.compare_forward(warp_output, torch_output)
    for warp_gradient, torch_gradient in zip(
        warp_gradients,
        torch_gradients,
        strict=True,
    ):
        ShrinkwrapPoints.compare_backward(warp_gradient, torch_gradient)


def test_warp_nearest_face_custom_op_opcheck():
    pytest.importorskip("warp")
    from physicsnemo.nn.functional.geometry.deform._warp_impl import (
        nearest_surface_faces_warp_impl,
    )

    target_points, target_faces = _triangle()
    query_points = torch.tensor([[0.2, 0.3, 1.0]])
    max_distance = torch.tensor(2.0)
    torch.library.opcheck(
        nearest_surface_faces_warp_impl,
        args=(target_points, target_faces, query_points, max_distance),
    )
    far_query_points = torch.tensor([[1.0e20, 0.3, 0.5]])
    unbounded_distance = torch.tensor(torch.inf)
    torch.library.opcheck(
        nearest_surface_faces_warp_impl,
        args=(
            target_points,
            target_faces,
            far_query_points,
            unbounded_distance,
        ),
    )


@pytest.mark.parametrize(
    ("scale", "target_base", "query_base"),
    [
        pytest.param(
            1.0e12,
            [
                [-4.0, -4.0, -4.0],
                [5.0, -5.0, 4.0],
                [1.0, -2.0, -4.0],
                [-1.0, 3.0, 2.0],
                [4.0, -2.0, -1.0],
                [-4.0, -2.0, -5.0],
            ],
            [[0.0, -2.0, -4.0]],
            id="large",
        ),
        pytest.param(
            1.0e-10,
            [
                [-1.148144006729126, -1.1588678359985352, 0.32547101378440857],
                [-0.6315053701400757, -2.839993953704834, -1.3249573707580566],
                [0.17842842638492584, -2.1337530612945557, 1.052357792854309],
                [-0.8036359548568726, -0.28084245324134827, 0.7696762681007385],
                [-0.6595596075057983, -0.7979276776313782, 0.18383125960826874],
                [0.22934740781784058, 0.5146290063858032, 0.99376380443573],
            ],
            [[-1.145442247390747, -1.3315942287445068, 0.2230386883020401]],
            id="small",
        ),
    ],
)
def test_warp_unsafe_target_scale_uses_torch_fallback(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
    scale: float,
    target_base: list[list[float]],
    query_base: list[list[float]],
):
    pytest.importorskip("warp")
    from physicsnemo.nn.functional.geometry.deform._warp_impl import shrinkwrap_op

    device = torch.device(device)
    target_points = (
        torch.tensor(target_base, device=device, dtype=torch.float32) * scale
    )
    target_faces = torch.tensor([[0, 1, 2], [3, 4, 5]], device=device)
    query_points = (
        torch.tensor(
            query_base,
            device=device,
            dtype=torch.float32,
        )
        * scale
    )
    torch_search = shrinkwrap_op.nearest_surface_faces_torch
    fallback_calls = 0

    def counted_torch_search(*args, **kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        return torch_search(*args, **kwargs)

    monkeypatch.setattr(
        shrinkwrap_op,
        "nearest_surface_faces_torch",
        counted_torch_search,
    )
    face_ids = shrinkwrap_op.nearest_surface_faces_warp(
        target_points,
        target_faces,
        query_points,
        max_distance=torch.inf,
    )

    assert fallback_calls == 1
    torch.testing.assert_close(
        face_ids,
        torch.tensor([0], device=device),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("edge_scale", "expected_safe"),
    [
        (2.0**-21, False),
        (2.0**-20, True),
        (2.0**20, True),
        (2.0**21, False),
    ],
)
def test_warp_float32_target_scale_gate(
    edge_scale: float,
    expected_safe: bool,
):
    from physicsnemo.nn.functional.geometry.deform._warp_impl.shrinkwrap_op import (
        _float32_target_search_is_safe,
    )

    half_scale = 0.5 * edge_scale
    target_points = torch.tensor(
        [
            [-half_scale, -half_scale, 0.0],
            [half_scale, -half_scale, 0.0],
            [-half_scale, half_scale, 0.0],
        ],
        dtype=torch.float32,
    )
    target_faces = torch.tensor([[0, 1, 2]])

    assert _float32_target_search_is_safe(target_points, target_faces) is expected_safe


@pytest.mark.parametrize(
    ("dtype", "scale"),
    [(torch.float32, 1.0e20), (torch.float64, 1.0e150)],
)
def test_large_world_coordinates_remain_finite_and_backend_consistent(
    device: str,
    dtype: torch.dtype,
    scale: float,
):
    pytest.importorskip("warp")
    device = torch.device(device)
    target_points = torch.tensor(
        [[0.0, 0.0, 0.0], [scale, 0.0, 0.0], [0.0, scale, 0.0]],
        device=device,
        dtype=dtype,
    )
    target_faces = torch.tensor([[0, 1, 2]], device=device)

    def output_and_gradients(implementation: str):
        points = torch.tensor(
            [[0.1 * scale, 0.1 * scale, 0.1 * scale]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        target = target_points.clone().requires_grad_()
        weights = torch.tensor(
            [0.7],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        offset = torch.tensor(
            1.0e-4 * scale,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        output = shrinkwrap_points(
            points,
            target,
            target_faces,
            point_weights=weights,
            offset=offset,
            implementation=implementation,
        )
        gradients = torch.autograd.grad(
            (output / scale).sum(),
            (points, target, weights, offset),
        )
        return output, gradients

    torch_output, torch_gradients = output_and_gradients("torch")
    warp_output, warp_gradients = output_and_gradients("warp")

    assert torch.isfinite(torch_output).all()
    assert torch.isfinite(warp_output).all()
    ShrinkwrapPoints.compare_forward(warp_output, torch_output)
    for warp_gradient, torch_gradient in zip(
        warp_gradients,
        torch_gradients,
        strict=True,
    ):
        assert torch.isfinite(warp_gradient).all()
        assert torch.isfinite(torch_gradient).all()
        ShrinkwrapPoints.compare_backward(warp_gradient, torch_gradient)


@pytest.mark.parametrize("implementation", ["torch", "warp", None])
def test_fake_tensor_propagation(implementation: str | None):
    if implementation == "warp":
        pytest.importorskip("warp")
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    with FakeTensorMode():
        points = torch.empty((2, 3))
        target_points = torch.empty((3, 3))
        target_faces = torch.empty((1, 3), dtype=torch.int64)
        point_weights = torch.empty((2,))
        offset = torch.empty(())
        output = shrinkwrap_points(
            points,
            target_points,
            target_faces,
            point_weights=point_weights,
            offset=offset,
            implementation=implementation,
        )

    assert isinstance(output, FakeTensor)
    assert output.shape == points.shape
    assert output.dtype == points.dtype
    assert output.device == points.device


@pytest.mark.parametrize("implementation", ["torch", "warp", None])
def test_torch_compile_fullgraph(implementation: str | None):
    if implementation == "warp":
        pytest.importorskip("warp")
    points = torch.tensor([[0.2, 0.3, 1.0], [0.6, 0.4, 0.7]])
    target_points, target_faces = _triangle()
    point_weights = torch.tensor([0.7, -0.2])
    offset = torch.tensor(0.1)

    def operation(p, t, f, w, o):
        return shrinkwrap_points(
            p,
            t,
            f,
            point_weights=w,
            offset=o,
            max_distance=2.0,
            implementation=implementation,
        )

    expected = operation(
        points,
        target_points,
        target_faces,
        point_weights,
        offset,
    )
    compiled = torch.compile(operation, backend="eager", fullgraph=True)
    actual = compiled(
        points,
        target_points,
        target_faces,
        point_weights,
        offset,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_torch_compile_dynamic_python_scalars_and_defaults(
    implementation: str,
):
    if implementation == "warp":
        pytest.importorskip("warp")
    points = torch.tensor([[0.2, 0.3, 1.0], [0.6, 0.4, 0.7]])
    target_points, target_faces = _triangle()

    def defaults(p, t, f):
        return shrinkwrap_points(
            p,
            t,
            f,
            implementation=implementation,
        )

    def runtime_scalars(p, t, f, offset, max_distance):
        return shrinkwrap_points(
            p,
            t,
            f,
            offset=offset,
            max_distance=max_distance,
            implementation=implementation,
        )

    compiled_graphs = []

    def counting_backend(graph_module, _example_inputs):
        compiled_graphs.append(graph_module)
        return graph_module.forward

    compiled_defaults = torch.compile(
        defaults,
        backend="eager",
        dynamic=True,
        fullgraph=True,
    )
    torch.testing.assert_close(
        compiled_defaults(points, target_points, target_faces),
        defaults(points, target_points, target_faces),
    )

    compiled_scalars = torch.compile(
        runtime_scalars,
        backend=counting_backend,
        dynamic=True,
        fullgraph=True,
    )
    for offset, max_distance in ((0.1, 2.0), (-0.2, 1.5), (0.0, 3.0)):
        torch.testing.assert_close(
            compiled_scalars(
                points,
                target_points,
                target_faces,
                offset,
                max_distance,
            ),
            runtime_scalars(
                points,
                target_points,
                target_faces,
                offset,
                max_distance,
            ),
        )
    assert len(compiled_graphs) == 1


@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_torch_compile_rejects_cutoff_that_rounds_to_zero(
    implementation: str,
):
    if implementation == "warp":
        pytest.importorskip("warp")
    points = torch.tensor([[0.2, 0.3, 0.0]])
    target_points, target_faces = _triangle()

    def operation(p, t, f, max_distance):
        return shrinkwrap_points(
            p,
            t,
            f,
            max_distance=max_distance,
            implementation=implementation,
        )

    compiled = torch.compile(
        operation,
        backend="eager",
        dynamic=True,
        fullgraph=True,
    )
    compiled(points, target_points, target_faces, 1.0)

    with pytest.raises(RuntimeError, match="positive in the points dtype"):
        compiled(points, target_points, target_faces, 1.0e-50)
