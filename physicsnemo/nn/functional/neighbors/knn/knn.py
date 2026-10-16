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

from typing import Literal

import torch
from jaxtyping import Float, Int

from physicsnemo.core.function_spec import FunctionSpec

from ._cuml_impl import knn_impl as knn_cuml
from ._scipy_impl import knn_impl as knn_scipy
from ._torch_impl import knn_impl as knn_torch
from ._warp_impl import knn_impl as knn_warp


class KNN(FunctionSpec):
    """
    Perform a k-nearest neighbor search on torch tensors.  Can be done with
    torch directly, or leverage RAPIDS cuML, SciPy, or an exact Warp BVH search.

    Auto-dispatch selects the implementation from the inputs: Warp for CUDA
    points of one to three dimensions in float32 or lower precision, cuML for
    other CUDA points, and SciPy for CPU points. When the selected
    implementation is not installed, it falls back to torch.

    Parameters
    ----------
    points : torch.Tensor
        Tensor of shape (N, D) containing the points to search from.
    queries : torch.Tensor
        Tensor of shape (M, D) containing the points to search for.
    k : int
        Number of nearest neighbors to return for each query point.
    implementation : {"cuml", "torch", "scipy", "warp"} or None
        Implementation to use for the search. When ``None``, it is selected
        as described above. ``"warp"`` runs an exact Morton-order BVH search
        in float32 on CPU or CUDA, for points of up to three dimensions.

    Returns
    -------
    indices : torch.Tensor
        Tensor of shape (M, k) containing the indices of the k nearest
        neighbors for each query point.
    distances : torch.Tensor
        Tensor of shape (M, k) containing the distances to the k nearest
        neighbors for each query point.
    """

    _BENCHMARK_CASES = (
        ("small-p1024-q256-k16", 1024, 256, 16),
        ("medium-p4096-q1024-k32", 4096, 1024, 32),
        ("large-p8192-q2048-k32", 8192, 2048, 32),
    )

    @FunctionSpec.register(
        name="cuml", required_imports=("cuml>=26.2.0", "cupy>=13.6.0"), rank=0
    )
    def cuml_forward(
        points: Float[torch.Tensor, "num_points dim"],
        queries: Float[torch.Tensor, "num_queries dim"],
        k: int,
    ) -> tuple[
        Int[torch.Tensor, "num_queries k"], Float[torch.Tensor, "num_queries k"]
    ]:
        return knn_cuml(points, queries, k)

    @FunctionSpec.register(name="scipy", required_imports=("scipy>=1.7.0",), rank=1)
    def scipy_forward(
        points: Float[torch.Tensor, "num_points dim"],
        queries: Float[torch.Tensor, "num_queries dim"],
        k: int,
    ) -> tuple[
        Int[torch.Tensor, "num_queries k"], Float[torch.Tensor, "num_queries k"]
    ]:
        return knn_scipy(points, queries, k)

    @FunctionSpec.register(name="torch", rank=2, baseline=True)
    def torch_forward(
        points: Float[torch.Tensor, "num_points dim"],
        queries: Float[torch.Tensor, "num_queries dim"],
        k: int,
    ) -> tuple[
        Int[torch.Tensor, "num_queries k"], Float[torch.Tensor, "num_queries k"]
    ]:
        return knn_torch(points, queries, k)

    @FunctionSpec.register(name="warp", required_imports=("warp>=1.17.0",), rank=3)
    def warp_forward(
        points: Float[torch.Tensor, "num_points dim"],
        queries: Float[torch.Tensor, "num_queries dim"],
        k: int,
    ) -> tuple[
        Int[torch.Tensor, "num_queries k"], Float[torch.Tensor, "num_queries k"]
    ]:
        """Exact Morton-order BVH search in Warp kernels (CPU or CUDA)."""
        return knn_warp(points, queries, k)

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        device = torch.device(device)
        for label, num_points, num_queries, k in cls._BENCHMARK_CASES:
            points = torch.rand(num_points, 3, device=device)
            queries = torch.rand(num_queries, 3, device=device)
            yield (
                label,
                (points, queries, k),
                {},
            )

    @classmethod
    def compare_forward(
        cls,
        output: tuple[torch.Tensor, torch.Tensor],
        reference: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        # Neighbor ordering can differ for equal-distance ties across backends.
        _, distances = output
        _, reference_distances = reference
        torch.testing.assert_close(
            torch.sort(distances, dim=1)[0],
            torch.sort(reference_distances, dim=1)[0],
            atol=1e-5,
            rtol=1e-5,
        )

    @classmethod
    def dispatch(
        cls,
        points: Float[torch.Tensor, "num_points dim"],
        queries: Float[torch.Tensor, "num_queries dim"],
        k: int,
        implementation: Literal["cuml", "torch", "scipy", "warp"] | None = None,
    ) -> tuple[
        Int[torch.Tensor, "num_queries k"], Float[torch.Tensor, "num_queries k"]
    ]:
        # Lookup the implementation registry for this FunctionSpec.
        impls = cls._get_impls()

        # Check if the implementation is registered
        cls._check_impl(implementation, impls)

        # If a specific implementation is requested, validate and call it.
        if implementation is not None:
            # Load the requested implementation from the registry.
            impl = impls[implementation]

            # Check if the implementation's required imports are available.
            if not impl.available:
                raise ImportError(
                    f"Implementation '{implementation}' is not available for {cls.__name__}"
                )

            # Execute the implementation.
            return impl.func(points, queries, k)

        # Otherwise, auto-select an implementation from the inputs.
        preferred = impls.get(cls._preferred_name(points))

        # Use the preferred implementation when it is available.
        impl = preferred if preferred is not None and preferred.available else None

        # Fall back to torch when the preferred option is unavailable.
        if impl is None:
            # Get the torch implementation
            impl = impls["torch"]

            # Warn once if we are falling back from the preferred implementation.
            cls._warn_fallback(preferred, impl)

        # Execute the selected implementation.
        return impl.func(points, queries, k)

    @staticmethod
    def _preferred_name(points: torch.Tensor) -> str:
        """Return the implementation auto-dispatch prefers for ``points``.

        Warp computes in float32, so float64 and integer CUDA points keep cuML.
        """
        if not points.is_cuda:
            return "scipy"
        if 1 <= points.shape[-1] <= 3 and points.dtype in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ):
            return "warp"
        return "cuml"


knn = KNN.make_function("knn")


__all__ = ["KNN", "knn"]
