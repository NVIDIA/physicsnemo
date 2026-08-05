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

import torch
from jaxtyping import Float

from physicsnemo.core.function_spec import FunctionSpec

from ._torch_impl import radius_search as radius_search_torch
from ._warp_impl import _WARP_VARIANTS
from ._warp_impl import radius_search as radius_search_warp
from .utils import radius_search_timing_enabled, time_radius_search


class RadiusSearch(FunctionSpec):
    """Performs radius-based neighbor search to find points within a specified radius of query points.

    Can use brute-force methods with PyTorch, or an accelerated spatial decomposition method with Warp.

    Accepts both unbatched inputs of shape ``(N, 3)`` and batched inputs of shape ``(B, N, 3)``.
    When unbatched inputs are provided, they are treated as ``B=1`` internally and the batch
    dimension is stripped from the output. Only ranks 2 and 3 are accepted; higher-rank inputs
    raise ``ValueError``.

    This function has differing behavior based on the argument for max_points.  If max_points is None,
    the function will find ALL points within the radius and return a flattened list of indices,
    (optionally) distances, and (optionally) points.  For unbatched inputs the indices will have a
    shape of ``(2, N)`` where N is the aggregate number of neighbors found for all queries. The 0th
    row is the query index and the 1st row is the point index. For batched inputs the indices will
    have shape ``(3, N)`` where the 0th row is the batch index, the 1st row is the query index, and
    the 2nd row is the point index.

    If max_points is not None, the function will find the max_points closest points within the radius
    and return a statically sized array of indices, (optionally) distances, and (optionally) points.
    For unbatched inputs the indices will have shape ``(Q, max_points)``. For batched inputs the
    indices will have shape ``(B, Q, max_points)``. Unused slots are filled with 0.

    Because the shape when max_points=None is dynamic, this function is incompatible with torch.compile
    in that case.  When max_points is set, this function is compatible with torch.compile regardless of
    backend.

    The different backends are not necessarily certain to provide identical output, for two reasons:
    first, if max_points is lower than the number of neighbors found, the selected points may be
    stochastic.  Second, when max_points is None or max_points is greater than the number of neighbors,
    the outputs may be ordered differently by the two backends.  Do not rely on the exact order of
    the neighbors in the outputs.

    Note:
        With the Warp backend, there will be an automatic casting of inputs to float32 from reduced precision,
        and results will be returned in their original precision.

    Args:
        points (torch.Tensor): The reference point cloud tensor of shape ``(N, 3)`` or ``(B, N, 3)``.
        queries (torch.Tensor): The query points tensor of shape ``(M, 3)`` or ``(B, M, 3)``.
        radius (float): The search radius. Points within or at this radius of a query point will be
            considered neighbors.
        max_points (int | None, optional): Maximum number of neighbors to return for each query point.
            If None, returns all neighbors within radius. Defaults to None.  See documentation for details.
        return_dists (bool, optional): If True, returns the distances to the neighbor points.
            Defaults to False.
        return_points (bool, optional): If True, returns the actual neighbor points in addition to
            their indices. Defaults to False.
        implementation (str, optional): Explicit backend selection. Base backends are
            ``"warp"`` (Warp hash grid) and ``"torch"`` (brute force). On the CUDA
            ``max_points`` path the Warp backend also exposes experimental variants
            selectable directly by name: ``"scalar"`` / ``"fma"`` / ``"gemm"``
            (Morton hash grid), ``"dense_fma"`` / ``"dense_fma_e2e"`` /
            ``"dense_fma_store_opt"`` / ``"dense_fma_mem_opt"`` / ``"dense_fma_mm"`` /
            ``"dense_gemm"`` / ``"custom_grid_cell"`` (Morton dense-cell;
            ``"custom_grid_cell"`` is a WIP experimental cell-centric memory-optimized
            launch that is not runnable yet), ``"sparse_fma_e2e"`` (sparse hash-grid
            cells), and ``"bvh"`` (Morton-ordered tiled LBVH). Defaults to None,
            which selects the lowest-rank available base backend.

    Returns:
        tuple | torch.Tensor:
            Neighbor indices are always returned first. Additional tensors are
            appended when requested:
            - ``indices`` (always): Neighbor indices
            - ``points`` (optional): Neighbor points when ``return_points=True``
            - ``distances`` (optional): Neighbor distances when ``return_dists=True``

    Raises:
        KeyError: If an explicit implementation name is not registered.
        ImportError: If the selected implementation is unavailable.
        ValueError: If inputs are not rank 2 or 3.

    """

    _BENCHMARK_CASES = (
        ("small-p1024-q512-r0p1-m32", 1, 1024, 512, 0.1, 32),
        ("medium-p4096-q2048-r0p1-m32", 1, 4096, 2048, 0.1, 32),
        ("large-p8192-q4096-r0p1-m32", 1, 8192, 4096, 0.1, 32),
        ("batched-b4-p1024-q512-r0p1-m32", 4, 1024, 512, 0.1, 32),
    )
    _BACKWARD_BENCHMARK_CASES = (
        ("small-bwd-p1024-q512-r0p1-m32", 1, 1024, 512, 0.1, 32),
        ("medium-bwd-p4096-q2048-r0p1-m32", 1, 4096, 2048, 0.1, 32),
        ("batched-bwd-b4-p1024-q512-r0p1-m32", 4, 1024, 512, 0.1, 32),
    )

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(
        points: Float[torch.Tensor, "*batch num_points 3"],
        queries: Float[torch.Tensor, "*batch num_queries 3"],
        radius: float,
        max_points: int | None = None,
        return_dists: bool = False,
        return_points: bool = False,
        variant: str | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Warp-accelerated radius search using spatial hash grids.

        ``variant`` selects a Morton/BVH sub-variant on the CUDA
        ``max_points`` path (injected by :meth:`RadiusSearch.dispatch` when the
        user requests one via ``implementation=``). When None, the hash-grid
        kernel runs unless a ``PHYSICSNEMO_RADIUS_SEARCH_*`` env var is set.
        """
        return radius_search_warp(
            points, queries, radius, max_points, return_dists, return_points, variant
        )

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(
        points: Float[torch.Tensor, "*batch num_points 3"],
        queries: Float[torch.Tensor, "*batch num_queries 3"],
        radius: float,
        max_points: int | None = None,
        return_dists: bool = False,
        return_points: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Pure-PyTorch brute-force radius search via cdist."""
        return radius_search_torch(
            points, queries, radius, max_points, return_dists, return_points
        )

    @classmethod
    def dispatch(cls, *args, **kwargs):
        """Dispatch to a backend, translating Warp variant names.

        The Morton/BVH variants are sub-variants of the ``warp`` backend rather
        than separately registered implementations. Selecting one by name
        (e.g. ``implementation="dense_fma"`` or ``implementation="bvh"``) rewrites
        the request to ``implementation="warp"`` plus an explicit ``variant`` that
        is threaded into the Warp custom op, which preserves the op's shared
        autograd backward. The base backends ``warp`` / ``torch`` and the
        automatic ``None`` selection are unaffected.

        Parameters
        ----------
        *args, **kwargs
            Arguments forwarded to the selected implementation. ``implementation``
            may name a base backend or any value in ``_WARP_VARIANTS``.

        Returns
        -------
        object
            The implementation result.
        """
        implementation = kwargs.get("implementation")
        if implementation in _WARP_VARIANTS:
            kwargs["implementation"] = "warp"
            kwargs["variant"] = implementation

        # Fast path: no timing overhead unless explicitly enabled.
        if not radius_search_timing_enabled():
            return super().dispatch(*args, **kwargs)

        # Opt-in CUDA-event timing (PHYSICSNEMO_RADIUS_SEARCH_TIMING). Label each
        # call by the user-requested backend so a benchmark can attribute per-call
        # GPU time to each implementation; carry radius/max_points for per-site
        # breakdowns. ``radius`` is the 3rd positional arg of ``radius_search``.
        label = implementation if implementation is not None else "auto"
        radius = args[2] if len(args) >= 3 else kwargs.get("radius")
        max_points = kwargs.get("max_points")
        if max_points is None and len(args) >= 4:
            max_points = args[3]
        with time_radius_search(
            label, meta={"radius": radius, "max_points": max_points}
        ):
            return super().dispatch(*args, **kwargs)

    @classmethod
    def make_inputs_forward(
        cls,
        device: torch.device | str = "cpu",
    ):
        """Yield ``(label, args, kwargs)`` tuples for forward-pass benchmarking."""
        device = torch.device(device)
        for (
            label,
            batch_size,
            num_points,
            num_queries,
            radius,
            max_points,
        ) in cls._BENCHMARK_CASES:
            if batch_size == 1:
                points = torch.rand(num_points, 3, device=device)
                queries = torch.rand(num_queries, 3, device=device)
            else:
                points = torch.rand(batch_size, num_points, 3, device=device)
                queries = torch.rand(batch_size, num_queries, 3, device=device)
            yield (
                label,
                (points, queries, radius),
                {
                    "max_points": max_points,
                    "return_dists": True,
                    "return_points": True,
                },
            )

    @classmethod
    def make_inputs_backward(
        cls,
        device: torch.device | str = "cpu",
    ):
        """Yield ``(label, args, kwargs)`` tuples for backward-pass benchmarking."""
        device = torch.device(device)
        for (
            label,
            batch_size,
            num_points,
            num_queries,
            radius,
            max_points,
        ) in cls._BACKWARD_BENCHMARK_CASES:
            if batch_size == 1:
                points = torch.rand(num_points, 3, device=device, requires_grad=True)
                queries = torch.rand(num_queries, 3, device=device, requires_grad=True)
            else:
                points = torch.rand(
                    batch_size, num_points, 3, device=device, requires_grad=True
                )
                queries = torch.rand(
                    batch_size, num_queries, 3, device=device, requires_grad=True
                )
            yield (
                label,
                (points, queries, radius),
                {
                    "max_points": max_points,
                    "return_dists": False,
                    "return_points": True,
                },
            )

    @classmethod
    def compare_forward(cls, output: tuple, reference: tuple) -> None:
        """Order-invariant comparison of two forward-pass outputs."""
        # Radius-search backends can return neighbors in different orders.
        if len(output) != len(reference):
            raise AssertionError("output and reference tuples must have equal length")

        dynamic_output = (
            len(output) > 0
            and output[0].ndim == 2
            and output[0].shape[0] in (2, 3)
            and output[0].dtype in (torch.int32, torch.int64)
        )

        for output_tensor, reference_tensor in zip(output, reference):
            if output_tensor.dtype in (torch.int32, torch.int64):
                if output_tensor.ndim == 2 and output_tensor.shape[0] == 2:
                    torch.testing.assert_close(
                        output_tensor.sum(dim=1).to(torch.int64),
                        reference_tensor.sum(dim=1).to(torch.int64),
                    )
                elif output_tensor.ndim >= 2:
                    torch.testing.assert_close(
                        output_tensor.sum(dim=1).to(torch.int64),
                        reference_tensor.sum(dim=1).to(torch.int64),
                    )
                else:
                    torch.testing.assert_close(
                        output_tensor.sum().to(torch.int64),
                        reference_tensor.sum().to(torch.int64),
                    )
                continue

            if (
                dynamic_output
                and output_tensor.ndim == 2
                and output_tensor.shape[1] == 3
            ):
                torch.testing.assert_close(
                    output_tensor.sum(dim=0),
                    reference_tensor.sum(dim=0),
                )
            elif dynamic_output and output_tensor.ndim == 1:
                torch.testing.assert_close(
                    output_tensor.sum(),
                    reference_tensor.sum(),
                )
            elif output_tensor.ndim == 2 and output_tensor.shape[0] == 2:
                torch.testing.assert_close(
                    output_tensor.sum(dim=0),
                    reference_tensor.sum(dim=0),
                )
            elif output_tensor.ndim >= 2:
                torch.testing.assert_close(
                    output_tensor.sum(dim=1),
                    reference_tensor.sum(dim=1),
                )
            else:
                torch.testing.assert_close(
                    output_tensor.sum(),
                    reference_tensor.sum(),
                )

    @classmethod
    def compare_backward(cls, output: torch.Tensor, reference: torch.Tensor) -> None:
        """Element-wise comparison of two backward-pass gradient tensors."""
        torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)


radius_search = RadiusSearch.make_function("radius_search")
