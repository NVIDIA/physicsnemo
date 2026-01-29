# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

from physicsnemo.core.function_spec import FunctionSpec

from ._cuml_impl import knn_impl as knn_cuml
from ._scipy_impl import knn_impl as knn_scipy
from ._torch_impl import knn_impl as knn_torch


class KNN(FunctionSpec):
    """
    Perform a k-nearest neighbor search on torch tensors.  Can be done with
    torch directly, or leverage RAPIDS cuML algorithm.

    Auto-dispatch selects the optimal version for the input tensor device.

    Parameters
    ----------
    points : torch.Tensor
        Tensor of shape (N, 3) containing the points to search from.
    queries : torch.Tensor
        Tensor of shape (M, 3) containing the points to search for.
    k : int
        Number of nearest neighbors to return for each query point.
    implementation : {"cuml", "torch", "scipy"} or None
        Implementation to use for the search. When ``None``, the preferred
        implementation for the input device is selected and falls back to
        torch when unavailable.

    Returns
    -------
    indices : torch.Tensor
        Tensor of shape (M, k) containing the indices of the k nearest
        neighbors for each query point.
    distances : torch.Tensor
        Tensor of shape (M, k) containing the distances to the k nearest
        neighbors for each query point.
    """

    @FunctionSpec.register(
        name="cuml", required_imports=("cuml>=24.0.0", "cupy>=13.0.0"), rank=0
    )
    def cuml_forward(
        points: torch.Tensor, queries: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return knn_cuml(points, queries, k)

    @FunctionSpec.register(name="scipy", required_imports=("scipy>=1.7.0",), rank=1)
    def scipy_forward(
        points: torch.Tensor, queries: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return knn_scipy(points, queries, k)

    @FunctionSpec.register(name="torch", rank=2, baseline=True)
    def torch_forward(
        points: torch.Tensor, queries: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return knn_torch(points, queries, k)

    @classmethod
    def make_inputs(cls, device: torch.device | str = "cpu"):
        # TODO(ASV): Populate benchmark inputs in a follow-up PR.
        raise NotImplementedError

    @classmethod
    def compare(
        cls,
        output: tuple[torch.Tensor, torch.Tensor],
        reference: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        # TODO(ASV): Populate output comparison in a follow-up PR.
        raise NotImplementedError

    @classmethod
    def dispatch(
        cls,
        points: torch.Tensor,
        queries: torch.Tensor,
        k: int,
        implementation: Literal["cuml", "torch", "scipy"] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        # Otherwise, auto-select an implementation based on device and availability.
        # Prefer cuML on CUDA and SciPy on CPU when auto-selecting.
        preferred_name = "cuml" if points.is_cuda else "scipy"

        # Fetch the preferred implementation (if registered).
        preferred = impls.get(preferred_name)

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


knn = KNN.make_function("knn")


__all__ = ["KNN", "knn"]
