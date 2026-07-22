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

"""As-rigid-as-possible deformation energy."""

from __future__ import annotations

import torch

from physicsnemo.core.function_spec import FunctionSpec


def _normalize_arap_inputs(
    reference_points: torch.Tensor,
    deformed_points: torch.Tensor,
    edges: torch.Tensor,
    edge_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Validate ARAP inputs and normalize point tensors to rank three."""

    if reference_points.ndim not in (2, 3):
        raise ValueError(
            "reference_points must have shape (N, D) or (B, N, D), got "
            f"{tuple(reference_points.shape)}"
        )
    if reference_points.shape[-1] < 2:
        raise ValueError("reference_points coordinate dimension must be at least 2")
    if reference_points.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "reference_points must have dtype torch.float32 or torch.float64, got "
            f"{reference_points.dtype}"
        )
    if deformed_points.shape != reference_points.shape:
        raise ValueError(
            "reference_points and deformed_points must have identical shapes, got "
            f"{tuple(reference_points.shape)} and {tuple(deformed_points.shape)}"
        )
    if deformed_points.device != reference_points.device:
        raise ValueError(
            "reference_points and deformed_points must be on the same device, got "
            f"{reference_points.device} and {deformed_points.device}"
        )
    if deformed_points.dtype != reference_points.dtype:
        raise TypeError(
            "reference_points and deformed_points must have the same dtype, got "
            f"{reference_points.dtype} and {deformed_points.dtype}"
        )

    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"edges must have shape (E, 2), got {tuple(edges.shape)}")
    if edges.dtype != torch.int64:
        raise TypeError(f"edges must have dtype torch.int64, got {edges.dtype}")
    if edges.device != reference_points.device:
        raise ValueError(
            "edges and point tensors must be on the same device, got "
            f"{edges.device} and {reference_points.device}"
        )

    num_edges = edges.shape[0]
    if edge_weights is None:
        normalized_weights = torch.ones(
            num_edges,
            dtype=reference_points.dtype,
            device=reference_points.device,
        )
    else:
        if edge_weights.ndim != 1 or edge_weights.shape[0] != num_edges:
            raise ValueError(
                f"edge_weights must have shape ({num_edges},), got "
                f"{tuple(edge_weights.shape)}"
            )
        if edge_weights.device != reference_points.device:
            raise ValueError(
                "edge_weights and point tensors must be on the same device, got "
                f"{edge_weights.device} and {reference_points.device}"
            )
        if edge_weights.dtype != reference_points.dtype:
            raise TypeError(
                "edge_weights and point tensors must have the same dtype, got "
                f"{edge_weights.dtype} and {reference_points.dtype}"
            )
        normalized_weights = edge_weights

    was_unbatched = reference_points.ndim == 2
    if was_unbatched:
        reference_points = reference_points.unsqueeze(0)
        deformed_points = deformed_points.unsqueeze(0)
    return (
        reference_points,
        deformed_points,
        edges,
        normalized_weights,
        was_unbatched,
    )


def _proper_rotations(covariance: torch.Tensor) -> torch.Tensor:
    """Return closest proper rotations for square covariance matrices."""

    # The rotations minimize the local ARAP objectives. Detaching only this
    # argmin follows the envelope theorem: first derivatives of the minimized
    # energy come from its explicit point and weight dependence, without an
    # unstable SVD backward at repeated or zero singular values.
    u, _, vh = torch.linalg.svd(covariance.detach(), full_matrices=False)
    unconstrained = u @ vh
    determinant = torch.linalg.det(unconstrained)
    final_sign = torch.where(
        determinant < 0,
        -torch.ones_like(determinant),
        torch.ones_like(determinant),
    )
    correction = torch.cat(
        (
            torch.ones_like(u[..., 0, :-1]),
            final_sign.unsqueeze(-1),
        ),
        dim=-1,
    )
    return (u * correction.unsqueeze(-2)) @ vh


def _arap_energy_torch(
    reference_points: torch.Tensor,
    deformed_points: torch.Tensor,
    edges: torch.Tensor,
    edge_weights: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ARAP energy for normalized aligned point batches."""

    batch_size, num_points, num_dims = reference_points.shape
    edge_start = edges[:, 0]
    edge_end = edges[:, 1]
    reference_edges = reference_points.index_select(
        1, edge_start
    ) - reference_points.index_select(1, edge_end)
    deformed_edges = deformed_points.index_select(
        1, edge_start
    ) - deformed_points.index_select(1, edge_end)

    weighted_outer_products = (
        deformed_edges.unsqueeze(-1) * reference_edges.unsqueeze(-2)
    ) * edge_weights.view(1, -1, 1, 1)
    covariance = torch.zeros(
        (batch_size, num_points, num_dims, num_dims),
        dtype=reference_points.dtype,
        device=reference_points.device,
    )
    covariance = covariance.index_add(1, edge_start, weighted_outer_products)
    covariance = covariance.index_add(1, edge_end, weighted_outer_products)
    rotations = _proper_rotations(covariance)

    rotated_from_start = torch.matmul(
        rotations.index_select(1, edge_start), reference_edges.unsqueeze(-1)
    ).squeeze(-1)
    rotated_from_end = torch.matmul(
        rotations.index_select(1, edge_end), reference_edges.unsqueeze(-1)
    ).squeeze(-1)
    start_residual = deformed_edges - rotated_from_start
    end_residual = deformed_edges - rotated_from_end
    per_edge = (
        0.5
        * edge_weights.unsqueeze(0)
        * (start_residual.square().sum(dim=-1) + end_residual.square().sum(dim=-1))
    )
    return per_edge.sum(dim=-1)


def _benchmark_edges(
    num_points: int,
    num_edges: int,
    device: torch.device,
) -> torch.Tensor:
    """Construct deterministic non-self edges for benchmark inputs."""

    edge_ids = torch.arange(num_edges, device=device, dtype=torch.int64)
    edge_start = edge_ids.remainder(num_points)
    edge_offset = edge_ids.div(num_points, rounding_mode="floor").remainder(7) + 1
    edge_end = (edge_start + edge_offset).remainder(num_points)
    return torch.stack((edge_start, edge_end), dim=-1)


class ARAPEnergy(FunctionSpec):
    r"""Measure as-rigid-as-possible deformation of an edge graph.

    A proper rotation :math:`R_i` is fitted independently to the one-ring of
    every point. For undirected edges :math:`(i,j)` with weights :math:`w_{ij}`,
    the returned energy is

    .. math::

       E = \frac{1}{2}\sum_{(i,j)} w_{ij}
       \left(\lVert q_{ij} - R_i p_{ij}\rVert^2
       + \lVert q_{ij} - R_j p_{ij}\rVert^2\right),

    where :math:`p_{ij}` and :math:`q_{ij}` are reference and deformed edge
    vectors. The optimal rotations are computed by polar decomposition with a
    determinant correction, so reflections are not treated as rigid motion.

    Inputs may be unbatched ``(N, D)`` or aligned batched ``(B, N, D)`` with a
    shared ``(E, 2)`` edge topology. Float32 and float64 point tensors are
    supported for coordinate dimensions ``D >= 2``. Edges are interpreted as
    undirected; duplicate edges contribute repeatedly and self-edges contribute
    zero.

    Parameters
    ----------
    reference_points : torch.Tensor
        Undeformed point coordinates with shape ``(N, D)`` or ``(B, N, D)``.
    deformed_points : torch.Tensor
        Deformed coordinates with exactly the same shape, dtype, and device as
        ``reference_points``.
    edges : torch.Tensor
        Shared edge indices with shape ``(E, 2)`` and dtype ``torch.int64``.
        Indices must be in ``[0, N)`` and on the point device.
    edge_weights : torch.Tensor or None, optional
        Shared scalar edge weights with shape ``(E,)`` and the same floating
        dtype and device as the points. ``None`` uses uniform weights. Default
        is ``None``.
    implementation : {"torch"} or None, optional
        Explicit backend. ``None`` selects Torch.

    Returns
    -------
    torch.Tensor
        A scalar for unbatched inputs or one energy per aligned batch with shape
        ``(B,)``.

    Notes
    -----
    Optimal rotations are detached from autograd. Away from non-unique local
    optima, the envelope theorem gives first-order derivatives of the minimized
    ARAP energy without differentiating the SVD. At degenerate neighborhoods,
    the selected rotation instead defines a finite first-order subgradient.
    Gradients propagate to reference points, deformed points, and tensor-valued
    edge weights. Higher derivatives through the fitted rotations are not part
    of the contract.
    """

    _FORWARD_BENCHMARK_CASES = (
        ("small-n4096-e12288-d3-uniform", 1, 4096, 12288, 3, False),
        ("medium-n16384-e49152-d3-weighted", 1, 16384, 49152, 3, True),
        ("batched-b4-n16384-e49152-d3-weighted", 4, 16384, 49152, 3, True),
    )
    _BACKWARD_BENCHMARK_CASES = (
        ("medium-n16384-e49152-d3-deformed", 1, 16384, 49152, 3, False),
        ("medium-n16384-e49152-d3-all", 1, 16384, 49152, 3, True),
    )
    _COMPARE_ATOL = 1.0e-6
    _COMPARE_RTOL = 1.0e-6

    @FunctionSpec.register(name="torch", rank=0, baseline=True)
    def torch_forward(
        reference_points: torch.Tensor,
        deformed_points: torch.Tensor,
        edges: torch.Tensor,
        edge_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate ARAP energy with the pure-Torch backend."""

        reference_b3, deformed_b3, edges, weights, was_unbatched = (
            _normalize_arap_inputs(
                reference_points,
                deformed_points,
                edges,
                edge_weights,
            )
        )
        energy = _arap_energy_torch(reference_b3, deformed_b3, edges, weights)
        return energy.squeeze(0) if was_unbatched else energy

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield representative forward benchmark cases."""

        device = torch.device(device)
        for seed, (
            label,
            batch_size,
            num_points,
            num_edges,
            num_dims,
            weighted,
        ) in enumerate(cls._FORWARD_BENCHMARK_CASES):
            generator = torch.Generator(device=device).manual_seed(3701 + seed)
            point_shape = (
                (num_points, num_dims)
                if batch_size == 1
                else (batch_size, num_points, num_dims)
            )
            reference = torch.randn(point_shape, generator=generator, device=device)
            deformed = reference + 0.05 * torch.randn(
                point_shape, generator=generator, device=device
            )
            edges = _benchmark_edges(num_points, num_edges, device)
            weights = (
                0.5 + torch.rand(num_edges, generator=generator, device=device)
                if weighted
                else None
            )
            yield label, (reference, deformed, edges), {"edge_weights": weights}

    @classmethod
    def make_inputs_backward(cls, device: torch.device | str = "cpu"):
        """Yield representative differentiable benchmark cases."""

        device = torch.device(device)
        for seed, (
            label,
            batch_size,
            num_points,
            num_edges,
            num_dims,
            all_gradients,
        ) in enumerate(cls._BACKWARD_BENCHMARK_CASES):
            generator = torch.Generator(device=device).manual_seed(3801 + seed)
            point_shape = (
                (num_points, num_dims)
                if batch_size == 1
                else (batch_size, num_points, num_dims)
            )
            reference = torch.randn(point_shape, generator=generator, device=device)
            deformed = reference + 0.05 * torch.randn(
                point_shape, generator=generator, device=device
            )
            edges = _benchmark_edges(num_points, num_edges, device)
            weights = 0.5 + torch.rand(num_edges, generator=generator, device=device)
            yield (
                label,
                (
                    reference.requires_grad_(all_gradients),
                    deformed.requires_grad_(True),
                    edges,
                ),
                {"edge_weights": weights.requires_grad_(all_gradients)},
            )

    @classmethod
    def compare_forward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare ARAP benchmark energies across backends."""

        torch.testing.assert_close(
            output, reference, atol=cls._COMPARE_ATOL, rtol=cls._COMPARE_RTOL
        )

    @classmethod
    def compare_backward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Compare ARAP benchmark gradients across backends."""

        torch.testing.assert_close(
            output, reference, atol=cls._COMPARE_ATOL, rtol=cls._COMPARE_RTOL
        )


arap_energy = ARAPEnergy.make_function("arap_energy")


__all__ = ["ARAPEnergy", "arap_energy"]
