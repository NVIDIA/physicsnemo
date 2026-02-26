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

from typing import Any, Callable, Literal

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from physicsnemo.experimental.models.globe.field_kernel import ChunkedKernel

DEFAULT_RTOL = 1e-5  # Default relative tolerance for comparisons
DEFAULT_ATOL = 1e-5  # Default absolute tolerance for comparisons
CHUNKING_ATOL = 1e-6  # Tighter tolerance for chunking consistency
ASYMPTOTIC_TOLERANCE = 1e-2  # Tolerance for power-law exponent of far-field decay rate
FAR_FIELD_DISTANCE = 1e3  # Distance for far-field tests
DISCRETIZATION_SEPARATION = 0.1  # Separation for discretization tests
DEFAULT_SEED = 42  # Random seed for reproducibility


def make_kernel_and_input_data(
    # Kernel configuration
    n_spatial_dims: int = 2,
    n_source_scalars: int = 0,
    n_source_vectors: int = 0,
    output_fields: dict[str, Literal["scalar", "vector"]] | None = None,
    n_global_scalars: int = 0,
    n_global_vectors: int = 0,
    n_spherical_harmonics: int = 4,
    hidden_layer_sizes: list[int] | None = None,
    smoothing_radius: float = 1e-8,
    # Data configuration
    n_source_points: int = 5,
    n_target_points: int = 12,
    device: str | torch.device = "cpu",
    seed: int = DEFAULT_SEED,
) -> tuple[ChunkedKernel, dict[str, Any]]:
    """Create a kernel and compatible input data for testing.

    Returns:
        Tuple of (kernel, input_data_dict)
    """
    # Set defaults
    if output_fields is None:
        output_fields = {"pressure": "scalar", "velocity": "vector"}
    if hidden_layer_sizes is None:
        hidden_layer_sizes = [32, 32]

    # Convert device
    device = torch.device(device) if isinstance(device, str) else device

    torch.manual_seed(seed)

    ### Build rank specs from output_fields
    output_field_ranks = {
        k: (0 if v == "scalar" else 1) for k, v in output_fields.items()
    }

    ### Build source and global rank specs from counts
    source_data_ranks = {
        **{f"source_scalar_{i}": 0 for i in range(n_source_scalars)},
        **{f"source_vector_{i}": 1 for i in range(n_source_vectors)},
    }
    global_data_ranks = {
        **{f"global_scalar_{i}": 0 for i in range(n_global_scalars)},
        **{f"global_vector_{i}": 1 for i in range(n_global_vectors)},
    }

    kernel = ChunkedKernel(
        n_spatial_dims=n_spatial_dims,
        output_field_ranks=output_field_ranks,
        source_data_ranks=source_data_ranks,
        global_data_ranks=global_data_ranks,
        n_spherical_harmonics=n_spherical_harmonics,
        hidden_layer_sizes=hidden_layer_sizes,
        smoothing_radius=smoothing_radius,
    ).to(device)
    kernel.eval()

    ### Build compatible input data tensors
    torch.manual_seed(seed)

    source_data_dict: dict[str, torch.Tensor] = {}
    for i in range(n_source_scalars):
        source_data_dict[f"source_scalar_{i}"] = torch.randn(
            n_source_points,
            device=device,
        )
    for i in range(n_source_vectors):
        source_data_dict[f"source_vector_{i}"] = F.normalize(
            torch.randn(n_source_points, n_spatial_dims, device=device),
            dim=-1,
        )

    global_data_dict: dict[str, torch.Tensor] = {}
    for i in range(n_global_scalars):
        global_data_dict[f"global_scalar_{i}"] = torch.randn(
            1,
            device=device,
        ).squeeze()
    for i in range(n_global_vectors):
        global_data_dict[f"global_vector_{i}"] = F.normalize(
            torch.randn(n_spatial_dims, device=device),
            dim=0,
        )

    input_data = {
        "source_points": torch.randn(n_source_points, n_spatial_dims, device=device),
        "target_points": torch.randn(n_target_points, n_spatial_dims, device=device),
        "source_strengths": torch.randn(n_source_points, device=device),
        "reference_length": torch.ones(tuple(), device=device),
        "source_data": TensorDict(
            source_data_dict,
            batch_size=[n_source_points],
            device=device,
        ),
        "global_data": TensorDict(
            global_data_dict,
            batch_size=[],
            device=device,
        ),
        "chunk_size": None,
    }

    return kernel, input_data


def evaluate_kernel_with_transform(
    kernel: ChunkedKernel,
    base_data: dict[str, Any],
    transform_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Evaluate kernel with original and transformed data.

    Returns:
        Tuple of (original_result, transformed_result)
    """
    # Evaluate original
    result_original = kernel(**base_data)

    # Apply transformation and evaluate
    transformed_data = transform_fn(base_data.copy())
    result_transformed = kernel(**transformed_data)

    return result_original, result_transformed


def compare_fields(
    result1: dict[str, torch.Tensor],
    result2: dict[str, torch.Tensor],
    output_fields: dict[str, Literal["scalar", "vector"]],
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    msg_prefix: str = "",
) -> None:
    """Compare two field results for approximate equality.

    Args:
        result1: First result dictionary
        result2: Second result dictionary
        output_fields: Field specifications
        rtol: Relative tolerance
        atol: Absolute tolerance
        msg_prefix: Optional prefix for error messages
    """
    for field_name, field_type in output_fields.items():
        if field_type == "scalar":
            # Compare scalar fields for equality
            assert torch.allclose(
                result1[field_name], result2[field_name], rtol=rtol, atol=atol
            ), (
                f"{msg_prefix}Scalar field {field_name} does not match within tolerance. "
                f"Max diff: {torch.max(torch.abs(result1[field_name] - result2[field_name])).item():.6e}"
            )
        else:  # vector
            # Compare vector fields for equality, checking magnitude and direction separately
            # Check magnitudes
            mag1 = torch.norm(result1[field_name], dim=-1)
            mag2 = torch.norm(result2[field_name], dim=-1)
            assert torch.allclose(mag1, mag2, rtol=rtol, atol=atol), (
                f"{msg_prefix}Vector field {field_name} magnitudes do not match. "
                f"Max diff: {torch.max(torch.abs(mag1 - mag2)).item():.6e}"
            )

            # For non-zero vectors, check alignment
            # Only check alignment for vectors with significant magnitude to avoid numerical issues
            mask = (mag1 > atol) & (mag2 > atol)
            if mask.any():
                dir1 = F.normalize(result1[field_name][mask], dim=-1)
                dir2 = F.normalize(result2[field_name][mask], dim=-1)
                dot_product = torch.sum(dir1 * dir2, dim=-1)
                # Note: dot product should be close to 1 for aligned vectors
                assert torch.allclose(
                    dot_product, torch.ones_like(dot_product), rtol=rtol, atol=atol
                ), (
                    f"{msg_prefix}Vector field {field_name} directions do not match. "
                    f"Min dot product: {dot_product.min().item():.6f} (expected ~1.0)"
                )


device_params = pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)

dims_params = pytest.mark.parametrize("n_dims", [2, 3])

output_fields_params = pytest.mark.parametrize(
    "output_fields",
    [
        {"potential": "scalar"},
        {"velocity": "vector"},
        {"potential": "scalar", "velocity": "vector"},
    ],
)


@device_params
@dims_params
@output_fields_params
def test_kernel_forward(
    device: torch.device,
    n_dims: int,
    output_fields: dict[str, Literal["scalar", "vector"]],
):
    """Test forward pass with various configurations."""
    kernel, input_data = make_kernel_and_input_data(
        n_spatial_dims=n_dims,
        output_fields=output_fields,
        device=device,
        n_source_points=7,
        n_target_points=11,
        seed=DEFAULT_SEED,
    )

    result = kernel(**input_data)

    # Verify output field presence, device placement, and shapes
    for field_name, field_type in output_fields.items():
        assert field_name in result
        assert result[field_name].device.type == (
            device if isinstance(device, str) else device.type
        )

        if field_type == "scalar":
            assert result[field_name].shape == (11,)
        else:  # vector
            assert result[field_name].shape == (11, n_dims)


@device_params
@pytest.mark.parametrize("chunk_size", [20, 40, "auto"])
def test_kernel_chunking(
    device: torch.device,
    chunk_size: int | str | None,
):
    """Verify that varying chunk sizes produces numerically consistent results."""
    # Build a kernel and common input payload
    kernel, input_data = make_kernel_and_input_data(
        n_spatial_dims=2,
        output_fields={"sfield": "scalar", "vfield": "vector"},
        device=device,
        n_source_points=10,
        n_target_points=100,
        seed=DEFAULT_SEED,
    )

    # Always get the reference answer without chunking for comparison
    reference = kernel(**{**input_data, "chunk_size": None})

    # Skip redundant run for the reference case
    if chunk_size is None:
        return

    candidate = kernel(**{**input_data, "chunk_size": chunk_size})

    # Scalars
    assert torch.allclose(
        reference["sfield"], candidate["sfield"], atol=CHUNKING_ATOL, rtol=0.0
    ), f"Scalar mismatch for {chunk_size=}."

    # Vectors
    assert torch.allclose(
        reference["vfield"], candidate["vfield"], atol=CHUNKING_ATOL, rtol=0.0
    ), f"Vector mismatch for {chunk_size=}."

    assert reference.batch_size == candidate.batch_size


@device_params
@dims_params
def test_kernel_gradient_flow(
    device: torch.device,
    n_dims: int,
):
    """Test gradient propagation through kernel."""
    kernel, input_data = make_kernel_and_input_data(
        n_spatial_dims=n_dims,
        output_fields={"field": "scalar"},  # Simple scalar field for gradient test
        device=device,
        n_source_vectors=1,
        n_source_points=1,
        n_target_points=1,
    )

    # Configure tensors to track gradients for backpropagation test
    input_data["source_points"] = torch.zeros(
        1, n_dims, device=device, requires_grad=True
    )
    input_data["source_data"].apply_(
        lambda v: torch.ones_like(v, device=device, requires_grad=True)
    )
    input_data["target_points"] = torch.ones(1, n_dims, device=device)

    result = kernel(**input_data)
    result["field"].sum().backward()

    assert input_data["source_points"].grad is not None


@device_params
@dims_params
@output_fields_params
def test_translation_equivariance(
    device: torch.device,
    n_dims: int,
    output_fields: dict[str, Literal["scalar", "vector"]],
):
    """Test that the kernel is equivariant under translation."""
    # Create kernel and test data
    kernel, input_data = make_kernel_and_input_data(
        n_spatial_dims=n_dims,
        output_fields=output_fields,
        device=device,
        n_source_vectors=1,
    )

    # Define translation transformation
    def apply_translation(data: dict[str, Any]) -> dict[str, Any]:
        translation = torch.randn(n_dims, device=device)
        data["source_points"] = data["source_points"] + translation
        data["target_points"] = data["target_points"] + translation
        # Source vectors (e.g., normals) are direction-only and unaffected by translation
        return data

    # Evaluate and compare
    result1, result2 = evaluate_kernel_with_transform(
        kernel, input_data, apply_translation
    )

    # Translation should not affect field values
    compare_fields(
        result1, result2, output_fields, msg_prefix="Translation equivariance: "
    )


@device_params
@dims_params
@output_fields_params
def test_rotational_equivariance(
    device: torch.device,
    n_dims: int,
    output_fields: dict[str, Literal["scalar", "vector"]],
):
    """Test that the kernel is properly equivariant under rotation."""
    # Create kernel and data with global vector
    kernel, input_data = make_kernel_and_input_data(
        n_spatial_dims=n_dims,
        output_fields=output_fields,
        device=device,
        n_source_vectors=1,
        n_global_vectors=1,
    )

    # Create rotation matrix
    if n_dims == 2:
        angle = torch.tensor(torch.pi / 3, device=device)  # 60 degrees
        # 2D rotation matrix
        R = torch.tensor(
            [
                [torch.cos(angle), -torch.sin(angle)],
                [torch.sin(angle), torch.cos(angle)],
            ],
            device=device,
        )
    else:  # 3D
        axis = F.normalize(torch.randn(3, device=device), dim=0)
        angle = torch.tensor(torch.pi / 3, device=device)
        # 3D rotation matrix using Rodrigues' formula
        # Create skew-symmetric matrix from axis vector
        K = torch.zeros(3, 3, device=axis.device)
        K[0, 1] = -axis[2]
        K[0, 2] = axis[1]
        K[1, 0] = axis[2]
        K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]
        K[2, 1] = axis[0]
        # Rodrigues' formula: R = I + sin(θ)K + (1-cos(θ))K²
        R = (
            torch.eye(3, device=axis.device)
            + torch.sin(angle) * K
            + (1 - torch.cos(angle)) * (K @ K)
        )

    def _rotate_vectors(td: TensorDict) -> TensorDict:
        """Rotate rank-1 (vector) leaves of a TensorDict, leave scalars unchanged."""
        return td.apply(
            lambda v: v @ R.T if v.ndim > td.batch_dims else v,
        )

    def apply_rotation(data: dict[str, Any]) -> dict[str, Any]:
        data["source_points"] = data["source_points"] @ R.T
        data["target_points"] = data["target_points"] @ R.T
        data["source_data"] = _rotate_vectors(data["source_data"])
        data["global_data"] = _rotate_vectors(data["global_data"])
        return data

    # Evaluate
    result1, result2 = evaluate_kernel_with_transform(
        kernel, input_data, apply_rotation
    )

    # Check equivariance
    for field_name, field_type in output_fields.items():
        if field_type == "scalar":
            # Scalar fields should be invariant
            assert torch.allclose(
                result1[field_name],
                result2[field_name],
                rtol=DEFAULT_RTOL,
                atol=DEFAULT_ATOL,
            ), f"Scalar field {field_name} is not invariant under rotation"
        else:  # vector
            # Vector fields should be equivariant (rotate with the system)
            rotated_field1 = result1[field_name] @ R.T
            assert torch.allclose(
                rotated_field1,
                result2[field_name],
                rtol=DEFAULT_RTOL,
                atol=DEFAULT_ATOL,
            ), f"Vector field {field_name} is not equivariant under rotation"


@device_params
@dims_params
@output_fields_params
def test_parity_equivariance(
    device: torch.device,
    n_dims: int,
    output_fields: dict[str, Literal["scalar", "vector"]],
):
    """Test that the kernel has proper parity (reflection) equivariance."""
    # Create kernel and test data
    kernel, input_data = make_kernel_and_input_data(
        n_spatial_dims=n_dims,
        output_fields=output_fields,
        device=device,
        n_source_vectors=1,
    )

    # Create reflection matrix (Householder reflection)
    normal = F.normalize(torch.randn(n_dims, device=device), dim=0)
    P = torch.eye(len(normal), device=normal.device) - 2 * torch.outer(normal, normal)

    def _reflect_vectors(td: TensorDict) -> TensorDict:
        """Reflect rank-1 (vector) leaves of a TensorDict, leave scalars unchanged."""
        return td.apply(
            lambda v: v @ P.T if v.ndim > td.batch_dims else v,
        )

    def apply_reflection(data: dict[str, Any]) -> dict[str, Any]:
        data["source_points"] = data["source_points"] @ P.T
        data["target_points"] = data["target_points"] @ P.T
        data["source_data"] = _reflect_vectors(data["source_data"])
        return data

    # Evaluate
    result1, result2 = evaluate_kernel_with_transform(
        kernel, input_data, apply_reflection
    )

    # Check parity properties
    for field_name, field_type in output_fields.items():
        if field_type == "scalar":
            # Scalar fields should be invariant under parity
            assert torch.allclose(
                result1[field_name],
                result2[field_name],
                rtol=DEFAULT_RTOL,
                atol=DEFAULT_ATOL,
            ), f"Scalar field {field_name} is not invariant under parity"
        else:  # vector
            # Vector fields should be equivariant under parity
            reflected_field1 = result1[field_name] @ P.T
            assert torch.allclose(
                reflected_field1,
                result2[field_name],
                rtol=DEFAULT_RTOL,
                atol=DEFAULT_ATOL,
            ), f"Vector field {field_name} is not equivariant under parity"


@device_params
@dims_params
def test_asymptotic_behavior(
    device: torch.device,
    n_dims: int,
):
    """Test correct far-field decay: 1/r in 2D, 1/r² in 3D."""
    # Use only scalar field for asymptotic test
    kernel, input_data = make_kernel_and_input_data(
        n_spatial_dims=n_dims,
        output_fields={"potential": "scalar"},
        device=device,
        n_source_points=1,
        n_target_points=21,
        seed=DEFAULT_SEED,
    )

    # Modify data for asymptotic test
    # Place single source at origin with unit strength
    input_data["source_points"] = torch.zeros(1, n_dims, device=device)
    input_data["source_strengths"] = torch.ones(1, device=device)

    # Create far-field target points with geometric spacing along random direction
    direction = F.normalize(torch.randn(n_dims, device=device), dim=0)
    log_distances = torch.linspace(10, 20, 21, device=device)  # exp(10) to exp(20)
    distances = torch.exp(log_distances)
    input_data["target_points"] = distances[:, None] * direction[None, :]

    result = kernel(**input_data)

    # Estimate power law exponent via log-log regression
    log_abs_values = torch.log(torch.abs(result["potential"]))
    X = torch.stack([torch.ones_like(log_distances), log_distances], dim=1)
    coeffs = torch.linalg.lstsq(X, log_abs_values).solution  # ty: ignore[unresolved-attribute]

    estimated_exponent = coeffs[1].item()
    expected_exponent = -(n_dims - 1)
    exponent_error = abs(estimated_exponent - expected_exponent)

    assert exponent_error < ASYMPTOTIC_TOLERANCE, (
        f"Far-field decay exponent does not match expected value. "
        f"{expected_exponent=:.4f}, {estimated_exponent=:.4f}, {exponent_error=:.4f}"
    )


@device_params
@dims_params
@output_fields_params
def test_discretization_invariance(
    device: torch.device,
    n_dims: int,
    output_fields: dict[str, Literal["scalar", "vector"]],
):
    """Test that splitting sources maintains far-field behavior."""
    # Create kernel and base data for single source
    kernel, single_source_data = make_kernel_and_input_data(
        n_spatial_dims=n_dims,
        output_fields=output_fields,
        device=device,
        n_source_vectors=1,
        n_source_points=1,
        n_target_points=20,
    )

    # Create far-field target points uniformly distributed on a sphere
    n_test_points = 20
    if n_dims == 2:
        angles = torch.linspace(0, 2 * torch.pi, n_test_points, device=device)
        target_points = torch.stack(
            [
                torch.cos(angles) * FAR_FIELD_DISTANCE,
                torch.sin(angles) * FAR_FIELD_DISTANCE,
            ],
            dim=1,
        )
    else:  # 3D
        # Spiral points on sphere for better coverage
        # Avoid poles (theta = 0 or pi) for numerical stability
        phi = torch.linspace(0, 2 * torch.pi, n_test_points, device=device)
        theta = torch.linspace(0.2, torch.pi - 0.2, n_test_points, device=device)
        target_points = torch.stack(
            [
                torch.sin(theta) * torch.cos(phi) * FAR_FIELD_DISTANCE,
                torch.sin(theta) * torch.sin(phi) * FAR_FIELD_DISTANCE,
                torch.cos(theta) * FAR_FIELD_DISTANCE,
            ],
            dim=1,
        )

    # Configure single source at origin with unit strength
    single_source_data["source_points"] = torch.zeros(1, n_dims, device=device)
    single_source_data["source_strengths"] = torch.ones(1, device=device)
    x_direction = torch.zeros(1, n_dims, device=device)
    x_direction[:, 0] = 1.0
    single_source_data["source_data"] = TensorDict(
        {"source_vector_0": x_direction},
        batch_size=[1],
        device=device,
    )
    single_source_data["target_points"] = target_points

    # Configure split sources: two sources separated by small distance
    _, split_source_data = make_kernel_and_input_data(
        n_spatial_dims=n_dims,
        output_fields=output_fields,
        device=device,
        n_source_vectors=1,
        n_source_points=2,
        n_target_points=20,
    )

    separation = DISCRETIZATION_SEPARATION
    split_points = torch.zeros(2, n_dims, device=device)
    split_points[0, 0] = -separation / 2
    split_points[1, 0] = separation / 2

    split_source_data["source_points"] = split_points
    split_source_data["source_strengths"] = torch.ones(2, device=device) * 0.5
    x_direction_split = torch.zeros(2, n_dims, device=device)
    x_direction_split[:, 0] = 1.0
    split_source_data["source_data"] = TensorDict(
        {"source_vector_0": x_direction_split},
        batch_size=[2],
        device=device,
    )
    split_source_data["target_points"] = target_points

    # Evaluate both configurations
    result_single = kernel(**single_source_data)
    result_split = kernel(**split_source_data)

    # In far field, split and single source should produce nearly identical fields
    compare_fields(
        result_single,
        result_split,
        output_fields,
        msg_prefix="Discretization invariance: ",
    )


@device_params
@dims_params
@output_fields_params
@pytest.mark.parametrize(
    "test_config",
    [
        pytest.param(
            {
                "name": "basic",
                "n_source_scalars": 0,
                "n_source_vectors": 1,
                "n_global_scalars": 0,
                "n_global_vectors": 0,
                "n_source_points": 8,
                "n_target_points": 12,
                "scale_factors": [0.01, 0.1, 2.0, 10.0, 100.0],
            },
            id="basic_features",
        ),
        pytest.param(
            {
                "name": "full",
                "n_source_scalars": 2,
                "n_source_vectors": 2,
                "n_global_scalars": 2,
                "n_global_vectors": 2,
                "n_source_points": 10,
                "n_target_points": 15,
                "scale_factors": [0.001, 0.1, 5.0, 100.0, 1000.0],
            },
            id="all_features",
        ),
    ],
)
def test_units_invariance(
    device: torch.device,
    n_dims: int,
    output_fields: dict[str, Literal["scalar", "vector"]],
    test_config: dict[str, Any],
):
    """Test that the kernel is invariant under uniform scaling of all spatial quantities.

    When all positions (sources and targets) and reference lengths are scaled by the same
    factor, the outputs should remain unchanged. This ensures dimensional consistency.

    This test runs with both basic features and all features enabled to ensure scaling
    invariance holds in all configurations.
    """
    # Create kernel with specified configuration
    kernel, input_data = make_kernel_and_input_data(
        n_spatial_dims=n_dims,
        output_fields=output_fields,
        device=device,
        n_source_scalars=test_config["n_source_scalars"],
        n_source_vectors=test_config["n_source_vectors"],
        n_global_scalars=test_config["n_global_scalars"],
        n_global_vectors=test_config["n_global_vectors"],
        n_source_points=test_config["n_source_points"],
        n_target_points=test_config["n_target_points"],
    )

    # Evaluate at original scale
    result_original = kernel(**input_data)

    # Test each scale factor
    for scale in test_config["scale_factors"]:
        # Apply uniform spatial scaling to positions and reference lengths
        scaled_data = input_data.copy()
        scaled_data["source_points"] = input_data["source_points"] * scale
        scaled_data["target_points"] = input_data["target_points"] * scale
        scaled_data["reference_length"] = input_data["reference_length"] * scale
        # Note: only spatial quantities scale; strengths and vectors remain unchanged

        result_scaled = kernel(**scaled_data)

        # Results should be identical
        rtol = 1e-3 if scale in [0.001, 1000.0] else 1e-4
        compare_fields(
            result_original,
            result_scaled,
            output_fields,
            rtol=rtol,
            atol=1e-5,
            msg_prefix=f"Units invariance ({test_config['name']}, scale={scale}): ",
        )


@device_params
@dims_params
@output_fields_params
def test_order_equivariance(
    device: torch.device,
    n_dims: int,
    output_fields: dict[str, Literal["scalar", "vector"]],
):
    """Test that the kernel is equivariant under the order of sources and targets."""
    # Create kernel with source scalars and vectors
    kernel, input_data = make_kernel_and_input_data(
        n_spatial_dims=n_dims,
        output_fields=output_fields,
        device=device,
        n_source_vectors=1,
        n_source_scalars=1,
        n_source_points=10,
        n_target_points=15,
    )

    # Generate permutations
    n_source = input_data["source_points"].shape[0]
    n_target = input_data["target_points"].shape[0]
    source_perm = torch.randperm(n_source, device=device)
    target_perm = torch.randperm(n_target, device=device)

    def apply_source_permutation(data: dict[str, Any]) -> dict[str, Any]:
        data["source_points"] = data["source_points"][source_perm]
        data["source_strengths"] = data["source_strengths"][source_perm]
        data["source_data"] = data["source_data"][source_perm]
        return data

    result1, result2 = evaluate_kernel_with_transform(
        kernel, input_data, apply_source_permutation
    )
    compare_fields(result1, result2, output_fields, msg_prefix="Source permutation: ")

    # Test target permutation
    def apply_target_permutation(data: dict[str, Any]) -> dict[str, Any]:
        data["target_points"] = data["target_points"][target_perm]
        return data

    result1, result2 = evaluate_kernel_with_transform(
        kernel, input_data, apply_target_permutation
    )
    # Unpermute target results
    inverse_perm = torch.argsort(target_perm)
    unpermuted_results = {
        field_name: result2[field_name][inverse_perm] for field_name in output_fields
    }
    compare_fields(
        result1, unpermuted_results, output_fields, msg_prefix="Target permutation: "
    )

    # Test both permutations
    def apply_both_permutations(data: dict[str, Any]) -> dict[str, Any]:
        data = apply_source_permutation(data)
        data = apply_target_permutation(data)
        return data

    result1, result2 = evaluate_kernel_with_transform(
        kernel, input_data, apply_both_permutations
    )
    # Unpermute target results
    inverse_perm = torch.argsort(target_perm)
    unpermuted_results = {
        field_name: result2[field_name][inverse_perm] for field_name in output_fields
    }
    compare_fields(
        result1, unpermuted_results, output_fields, msg_prefix="Both permutations: "
    )


@device_params
@dims_params
def test_output_fields_order_equivariance(
    device: torch.device,
    n_dims: int,
):
    """Test that the kernel produces identical results regardless of output_fields dictionary order."""
    # Define base output fields
    base_output_fields = {
        "potential": "scalar",
        "velocity": "vector",
        "temperature": "scalar",
        "electric_field": "vector",
        "nut": "scalar",
        "magnetic_field": "vector",
    }

    # Create different orderings
    output_fields_permutations = [
        base_output_fields,  # Original
        dict(reversed(base_output_fields.items())),  # Reverse
        # Vectors first, then scalars
        {k: v for k, v in base_output_fields.items() if v == "vector"}
        | {k: v for k, v in base_output_fields.items() if v == "scalar"},
        # Alternating scalar/vector
        dict(sorted(base_output_fields.items(), key=lambda x: (x[1], x[0]))),
        # Alphabetical
        dict(sorted(base_output_fields.items())),
    ]

    # Evaluate kernel with each ordering
    results = []
    # Store input data from first kernel to ensure all kernels use identical inputs
    first_input_data = None

    for output_fields in output_fields_permutations:
        # Create kernel with this specific field ordering
        kernel, input_data = make_kernel_and_input_data(
            n_spatial_dims=n_dims,
            output_fields=output_fields,
            device=device,
            n_source_vectors=1,
            n_source_points=8,
            n_target_points=12,
            seed=DEFAULT_SEED,
        )

        # Use the same input data for all kernels to ensure fair comparison
        if first_input_data is None:
            first_input_data = input_data

        result = kernel(**first_input_data)
        results.append(result)

    # Compare all results
    reference_result = results[0]
    for i, result in enumerate(results[1:], 1):
        # Verify all fields present
        assert set(result.keys()) == set(reference_result.keys()), (
            f"Permutation {i} has different fields"
        )

        # Compare each field using the standard compare_fields function
        compare_fields(
            reference_result,
            result,
            base_output_fields,
            msg_prefix=f"Output field ordering {i}: ",
        )


if __name__ == "__main__":
    pytest.main()
