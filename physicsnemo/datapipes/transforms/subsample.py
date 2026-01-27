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

"""
Subsampling transforms for point clouds and surfaces.

Provides efficient subsampling methods for large datasets, including
Poisson disk sampling and weighted sampling.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from tensordict import TensorDict

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.base import Transform


def poisson_sample_indices_fixed(N: int, k: int, device=None) -> torch.Tensor:
    """
    Near-uniform sampler of indices for very large arrays.

    This function provides nearly uniform sampling for cases where the number
    of indices is very large (> 2^24) and :func:`torch.multinomial` cannot work.
    Unlike using :func:`torch.randperm`, there is no need to materialize and
    randomize the entire tensor of indices.

    The sampling uses exponentially distributed gaps to achieve near-uniform
    coverage without replacement.

    Parameters
    ----------
    N : int
        Total number of available indices.
    k : int
        Number of indices to sample.
    device : torch.device, optional
        Device for the output tensor.

    Returns
    -------
    torch.Tensor
        Tensor of shape :math:`(k,)` containing sampled indices.

    Examples
    --------
    >>> indices = poisson_sample_indices_fixed(1000000, 10000)
    >>> print(indices.shape)
    torch.Size([10000])
    """
    # Draw exponential gaps off of random initializations
    gaps = torch.rand(k, device=device).exponential_()

    summed = gaps.sum()

    # Normalize so total cumulative sum == N
    gaps *= N / summed

    # Compute cumulative positions
    idx = torch.cumsum(gaps, dim=0)

    # Shift down so range starts at 0 and ends below N
    idx -= gaps[0] / 2

    # Round to nearest integer index
    idx = torch.clamp(idx.floor().long(), min=0, max=N - 1)

    return idx


def shuffle_array(
    points: torch.Tensor,
    n_points: int,
    weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points with or without weights.

    Parameters
    ----------
    points : torch.Tensor
        Input tensor to sample from, shape :math:`(N, ...)`.
    n_points : int
        Number of points to sample.
    weights : torch.Tensor, optional
        Optional weights for sampling, shape :math:`(N,)`.
        If None, uses uniform sampling.

    Returns
    -------
    sampled_points : torch.Tensor
        Sampled tensor, shape :math:`(n\\_points, ...)`.
    indices : torch.Tensor
        Selected indices, shape :math:`(n\\_points,)`.
    """
    N = points.shape[0]
    device = points.device

    if N < n_points:
        # If not enough points, return all points
        indices = torch.arange(N, device=device)
        return points, indices

    if weights is not None:
        # Weighted sampling
        indices = torch.multinomial(weights, n_points, replacement=False)
    else:
        # Uniform sampling
        if N > 2**24:
            # Use Poisson sampling for very large arrays
            indices = poisson_sample_indices_fixed(N, n_points, device=device)
        else:
            # Use standard multinomial for smaller arrays
            indices = torch.randperm(N, device=device)[:n_points]

    sampled_points = points[indices]
    return sampled_points, indices


@register()
class SubsamplePoints(Transform):
    r"""
    Subsample points from large point clouds or meshes.

    This transform applies coordinated subsampling to multiple tensor fields,
    ensuring that the same points are selected across all specified keys.
    Useful for downsampling large volumetric data or point clouds while
    maintaining correspondence between coordinates and field values.

    Supports two sampling algorithms:

    - ``"poisson_fixed"``: Near-uniform sampling for very large datasets (> 2^24 points)
    - ``"uniform"``: Standard uniform sampling

    Optionally supports weighted sampling (e.g., area-weighted for surface meshes)
    by providing a ``weights_key``.

    Parameters
    ----------
    input_keys : list[str]
        List of tensor keys to subsample. All must have the same
        first dimension size.
    n_points : int
        Number of points to sample.
    algorithm : {"poisson_fixed", "uniform"}, default="poisson_fixed"
        Sampling algorithm to use.
    weights_key : str, optional
        Optional key for sampling weights (e.g., ``"surface_areas"``
        for area-weighted surface sampling). When provided, samples
        are drawn according to the weights distribution.

    Examples
    --------
    Uniform sampling:

    >>> transform = SubsamplePoints(
    ...     input_keys=["volume_mesh_centers", "volume_fields"],
    ...     n_points=10000,
    ...     algorithm="poisson_fixed"
    ... )
    >>> sample = TensorDict({
    ...     "volume_mesh_centers": torch.randn(100000, 3),
    ...     "volume_fields": torch.randn(100000, 5)
    ... })
    >>> result = transform(sample)
    >>> print(result["volume_mesh_centers"].shape)
    torch.Size([10000, 3])

    Weighted sampling:

    >>> transform = SubsamplePoints(
    ...     input_keys=["surface_mesh_centers", "surface_fields", "surface_normals"],
    ...     n_points=5000,
    ...     algorithm="uniform",
    ...     weights_key="surface_areas"
    ... )
    >>> sample = TensorDict({
    ...     "surface_mesh_centers": torch.randn(20000, 3),
    ...     "surface_fields": torch.randn(20000, 2),
    ...     "surface_normals": torch.randn(20000, 3),
    ...     "surface_areas": torch.rand(20000)
    ... })
    >>> result = transform(sample)
    >>> print(result["surface_mesh_centers"].shape)
    torch.Size([5000, 3])

    Notes
    -----
    All specified keys must have the same size in their first dimension.
    The same indices are applied to all keys to maintain correspondence.
    """

    def __init__(
        self,
        input_keys: list[str],
        n_points: int,
        *,
        algorithm: Literal["poisson_fixed", "uniform"] = "poisson_fixed",
        weights_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the subsample transform.

        Parameters
        ----------
        input_keys : list[str]
            List of tensor keys to subsample. All must have the same
            first dimension size.
        n_points : int
            Number of points to sample.
        algorithm : {"poisson_fixed", "uniform"}, default="poisson_fixed"
            Sampling algorithm to use.
        weights_key : str, optional
            Optional key for sampling weights (e.g., ``"surface_areas"``
            for area-weighted surface sampling). When provided, samples
            are drawn according to the weights distribution.
        """
        super().__init__()
        self.input_keys = input_keys
        self.n_points = n_points
        self.algorithm = algorithm
        self.weights_key = weights_key

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Apply subsampling to the TensorDict.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict containing fields to subsample.

        Returns
        -------
        TensorDict
            TensorDict with subsampled fields.

        Raises
        ------
        KeyError
            If a required key is not found in the data.
        ValueError
            If keys have inconsistent first dimension sizes.
        """
        if not self.input_keys:
            return data

        # Check that all keys are present
        for key in self.input_keys:
            if key not in data.keys():
                raise KeyError(
                    f"Key '{key}' not found in data. "
                    f"Available keys: {list(data.keys())}"
                )

        # Get the first key to determine indices
        first_key = self.input_keys[0]
        first_tensor = data[first_key]
        N = first_tensor.shape[0]

        # Check that all keys have the same first dimension
        for key in self.input_keys[1:]:
            if data[key].shape[0] != N:
                raise ValueError(
                    f"All keys must have the same first dimension. "
                    f"Key '{first_key}' has {N}, but '{key}' has {data[key].shape[0]}"
                )

        # Skip if already fewer points than requested
        if N <= self.n_points:
            return data

        # Get weights if provided
        weights = None
        if self.weights_key is not None:
            if self.weights_key not in data.keys():
                raise KeyError(
                    f"Weights key '{self.weights_key}' not found in data. "
                    f"Available keys: {list(data.keys())}"
                )
            weights = data[self.weights_key]

        # Sample indices
        device = first_tensor.device
        if weights is not None:
            # Weighted sampling
            _, indices = shuffle_array(first_tensor, self.n_points, weights=weights)
        elif self.algorithm == "poisson_fixed" and N > 2**24:
            indices = poisson_sample_indices_fixed(N, self.n_points, device=device)
        else:
            # Use uniform sampling
            indices = torch.randperm(N, device=device)[: self.n_points]

        # Apply indices to all keys
        updates = {}
        for key in self.input_keys:
            updates[key] = data[key][indices]

        return data.update(updates)

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation of the transform.
        """
        weights_str = f", weights_key={self.weights_key}" if self.weights_key else ""
        return (
            f"SubsamplePoints(input_keys={self.input_keys}, n_points={self.n_points}, "
            f"algorithm={self.algorithm}{weights_str})"
        )
