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

from __future__ import annotations

import torch

from physicsnemo.core.function_spec import FunctionSpec

from ._torch_impl import mesh_lsq_hessian_torch
from ._warp_impl import mesh_lsq_hessian_warp


def _make_benchmark_case(
    *,
    device: torch.device,
    n_entities: int,
    n_dims: int,
    k_neighbors: int,
    vector_values: bool,
    requires_grad: bool,
) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
    """Build one deterministic CSR benchmark case."""
    generator = torch.Generator(device=device)
    generator.manual_seed(9137 + n_entities + n_dims)
    points = torch.rand(
        (n_entities, n_dims),
        generator=generator,
        device=device,
    )
    distances = torch.cdist(points, points)
    neighbors = torch.topk(
        distances,
        k=k_neighbors + 1,
        largest=False,
        dim=1,
    ).indices[:, 1:]
    offsets = torch.arange(
        0,
        n_entities * k_neighbors + 1,
        k_neighbors,
        device=device,
        dtype=torch.int64,
    )
    indices = neighbors.reshape(-1).to(torch.int64)

    scalar_values = torch.sin(2.0 * torch.pi * points[:, 0]) + 0.25 * torch.cos(
        2.0 * torch.pi * points[:, -1]
    )
    if vector_values:
        values = torch.stack(
            (
                scalar_values,
                points.square().sum(dim=-1),
                torch.exp(0.1 * points.sum(dim=-1)),
            ),
            dim=-1,
        )
    else:
        values = scalar_values
    values = values.to(torch.float32).detach().clone().requires_grad_(requires_grad)

    points = points.to(torch.float32).detach().clone().requires_grad_(requires_grad)
    args = (points, values, offsets, indices)
    kwargs: dict[str, object] = {
        "weight_power": 2.0,
        "min_neighbors": None,
    }
    return args, kwargs


class MeshLSQHessian(FunctionSpec):
    r"""Reconstruct Hessians on unstructured neighborhoods with quadratic LSQ.

    For each entity :math:`i`, this functional directly fits the centered
    quadratic Taylor model

    .. math::

       \phi_j - \phi_i \approx
       g_i^T r_{ij} + \frac{1}{2} r_{ij}^T H_i r_{ij},

    where :math:`r_{ij} = x_j - x_i`. The linear term is fitted jointly as a
    nuisance variable, while the symmetric matrix :math:`H_i` is returned.
    This avoids the noise amplification of applying a first-derivative
    reconstruction twice.

    Each neighborhood is normalized by its root-mean-square radius before
    solving. The Hessian is transformed back to physical coordinates afterward,
    which keeps the linear and quadratic design columns well-scaled across mesh
    units and resolutions. Rank-deficient fits return zero Hessians rather than
    non-identifiable minimum-norm curvature.

    Parameters
    ----------
    points : torch.Tensor
        Entity coordinates with shape ``(n_entities, dims)`` for 1D, 2D, or 3D.
    values : torch.Tensor
        Scalar or tensor values with shape ``(n_entities,)`` or
        ``(n_entities, ...)``.
    neighbor_offsets : torch.Tensor
        CSR offsets with shape ``(n_entities + 1,)``.
    neighbor_indices : torch.Tensor
        Flattened CSR neighbor indices with shape ``(nnz,)``.
    weight_power : float, optional
        Inverse-distance exponent used for weighting. Default is ``2.0``.
    min_neighbors : int or None, optional
        Entities with fewer neighbors return zero Hessians. ``None`` selects
        the number of identifiable Taylor coefficients: 2, 5, or 9 in 1D,
        2D, or 3D. Passing a smaller explicit value does not override the
        full-rank requirement. Default is ``None``.
    safe_epsilon : float or None, optional
        Positive floor applied to squared distances after neighborhood
        normalization when forming inverse-distance weights. Exactly
        coincident points receive zero weight. A dtype-derived value is used
        when ``None``.
    rcond : float or None, optional
        Relative numerical-rank cutoff used to classify identifiable fits
        before the full-rank least-squares solve. Torch applies it to singular
        values and Warp to column-pivoted QR pivots. ``None`` uses
        ``max(n_valid_neighbors, n_coefficients) * eps`` for the backend's
        compute dtype.
    implementation : {"warp", "torch"} or None
        Implementation to use. When ``None``, dispatch selects by rank.

    Returns
    -------
    torch.Tensor
        Hessians with shape ``(n_entities, dims, dims)`` for scalar values or
        ``(n_entities, dims, dims, ...)`` for tensor values. The two spatial
        Hessian axes precede all value-component axes.

    Notes
    -----
    This is an ambient-space reconstruction. Neighborhoods must span the full
    ambient dimension; intrinsic surface Hessians are not supported. When only
    the surface trace is needed, use
    :func:`physicsnemo.mesh.calculus.compute_laplacian_points_dec` for a
    Laplace--Beltrami calculation. The direct quadratic fit is also distinct
    from a gradient-of-gradient LSQ reconstruction.

    Both implementations are differentiable with respect to coordinates and
    values for a fixed neighbor topology and numerical-rank decision. The Torch
    reference computes in float32 for float16 and bfloat16 inputs and in float64
    when either coordinates or values are float64. The Warp implementation
    computes in float32. Both return the values dtype. The Warp backend consumes
    the supplied CSR neighborhood directly and does not construct a
    ``warp.Mesh``.

    Examples
    --------
    Recover the constant second derivative of :math:`f(x) = x^2`:

    >>> import torch
    >>> from physicsnemo.nn.functional import mesh_lsq_hessian
    >>> points = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    >>> values = points[:, 0].square()
    >>> offsets = torch.tensor([0, 2, 4, 6], dtype=torch.int64)
    >>> indices = torch.tensor([1, 2, 0, 2, 0, 1], dtype=torch.int64)
    >>> result = mesh_lsq_hessian(
    ...     points, values, offsets, indices, implementation="torch"
    ... )
    >>> torch.round(result[:, 0, 0], decimals=6)
    tensor([2., 2., 2.], dtype=torch.float64)
    """

    _BENCHMARK_CASES = (
        ("1d-scalar-n512-k8", 512, 1, 8, False),
        ("2d-scalar-n512-k12", 512, 2, 12, False),
        ("3d-vector-n512-k16", 512, 3, 16, True),
    )

    _COMPARE_ATOL = 5.0e-2
    _COMPARE_RTOL = 5.0e-2
    _COMPARE_BACKWARD_ATOL = 1.0e-1
    _COMPARE_BACKWARD_RTOL = 1.0e-1

    @FunctionSpec.register(
        name="warp",
        required_imports=("warp>=1.14.0",),
        rank=0,
    )
    def warp_forward(
        points: torch.Tensor,
        values: torch.Tensor,
        neighbor_offsets: torch.Tensor,
        neighbor_indices: torch.Tensor,
        weight_power: float = 2.0,
        min_neighbors: int | None = None,
        safe_epsilon: float | None = None,
        rcond: float | None = None,
    ) -> torch.Tensor:
        """Dispatch direct quadratic-LSQ Hessians to Warp."""
        return mesh_lsq_hessian_warp(
            points=points,
            values=values,
            neighbor_offsets=neighbor_offsets,
            neighbor_indices=neighbor_indices,
            weight_power=weight_power,
            min_neighbors=min_neighbors,
            safe_epsilon=safe_epsilon,
            rcond=rcond,
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: torch.Tensor,
        values: torch.Tensor,
        neighbor_offsets: torch.Tensor,
        neighbor_indices: torch.Tensor,
        weight_power: float = 2.0,
        min_neighbors: int | None = None,
        safe_epsilon: float | None = None,
        rcond: float | None = None,
    ) -> torch.Tensor:
        """Dispatch direct quadratic-LSQ Hessians to PyTorch."""
        return mesh_lsq_hessian_torch(
            points=points,
            values=values,
            neighbor_offsets=neighbor_offsets,
            neighbor_indices=neighbor_indices,
            weight_power=weight_power,
            min_neighbors=min_neighbors,
            safe_epsilon=safe_epsilon,
            rcond=rcond,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative forward benchmark inputs."""
        resolved_device = torch.device(device)
        for (
            label,
            n_entities,
            n_dims,
            k_neighbors,
            vector_values,
        ) in cls._BENCHMARK_CASES:
            args, kwargs = _make_benchmark_case(
                device=resolved_device,
                n_entities=n_entities,
                n_dims=n_dims,
                k_neighbors=k_neighbors,
                vector_values=vector_values,
                requires_grad=False,
            )
            yield label, args, kwargs

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield representative backward benchmark inputs."""
        resolved_device = torch.device(device)
        for (
            label,
            n_entities,
            n_dims,
            k_neighbors,
            vector_values,
        ) in cls._BENCHMARK_CASES:
            args, kwargs = _make_benchmark_case(
                device=resolved_device,
                n_entities=n_entities,
                n_dims=n_dims,
                k_neighbors=k_neighbors,
                vector_values=vector_values,
                requires_grad=True,
            )
            yield label, args, kwargs

    @classmethod
    def compare_forward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare Warp outputs against the Torch reference."""
        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_ATOL,
            rtol=cls._COMPARE_RTOL,
        )

    @classmethod
    def compare_backward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare Warp gradients against the Torch reference."""
        torch.testing.assert_close(
            output,
            reference,
            atol=cls._COMPARE_BACKWARD_ATOL,
            rtol=cls._COMPARE_BACKWARD_RTOL,
        )


mesh_lsq_hessian = MeshLSQHessian.make_function("mesh_lsq_hessian")


__all__ = ["MeshLSQHessian", "mesh_lsq_hessian"]
