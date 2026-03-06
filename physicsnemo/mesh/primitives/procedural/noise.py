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

"""Procedural noise generation for mesh data.

Provides dimension-agnostic Perlin noise implementations that work on any
n-dimensional point set, fully GPU-compatible using pure PyTorch.
"""

from itertools import product

import torch


def perlin_noise_nd(
    points: torch.Tensor, scale: float = 1.0, seed: int = 0
) -> torch.Tensor:
    """GPU-accelerated dimension-agnostic Perlin noise using pure PyTorch.

    Generates smooth pseudo-random noise values using gradient interpolation
    on an n-dimensional lattice. Works for any number of spatial dimensions.

    The implementation uses:
    - Smoothstep interpolation for C² continuity
    - Hash-based pseudo-random gradients
    - n-linear interpolation in hypercube

    Parameters
    ----------
    points : torch.Tensor
        (N, n_dims) tensor of positions to evaluate noise at. Can be any
        number of dimensions (1D, 2D, 3D, 4D, etc.).
    scale : float
        Frequency of noise. Larger values create more variation (smaller features).
        Default 1.0 creates features of size ~1 unit.
    seed : int
        Random seed for reproducibility. Same seed produces same noise pattern.

    Returns
    -------
    torch.Tensor
        (N,) tensor of noise values in approximately [-1, 1].

    Examples
    --------
    >>> # 2D noise for texture generation
    >>> import torch
    >>> points = torch.rand(100, 2)
    >>> noise = perlin_noise_nd(points, scale=2.0, seed=42)
    >>>
    >>> # 3D noise for volumetric data
    >>> centroids = mesh.cell_centroids  # (n_cells, 3)  # doctest: +SKIP
    >>> noise = perlin_noise_nd(centroids, scale=0.5, seed=123)  # doctest: +SKIP
    >>> mesh.cell_data["noise"] = noise  # doctest: +SKIP
    >>>
    >>> # Works on GPU
    >>> points_gpu = points.cuda()  # doctest: +SKIP
    >>> noise_gpu = perlin_noise_nd(points_gpu, scale=1.0, seed=42)  # doctest: +SKIP
    """
    device = points.device
    n_dims = points.shape[-1]

    ### Create permutation table from seed (using local generator to avoid global state mutation)
    generator = torch.Generator(device=device).manual_seed(seed)
    perm = torch.randperm(256, dtype=torch.long, device=device, generator=generator)
    perm = torch.cat([perm, perm])  # Duplicate for wrapping

    ### Scale points and decompose into lattice + fractional parts
    coords = points * scale
    lattice_coords = coords.floor().long()
    fractional_coords = coords - coords.floor()

    ### Apply smoothstep for C² continuity: 6t⁵ - 15t⁴ + 10t³
    interp_weights = (
        fractional_coords
        * fractional_coords
        * fractional_coords
        * (fractional_coords * (fractional_coords * 6 - 15) + 10)
    )

    ### Vectorized gradient computation
    def compute_gradient(hash_val: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Compute gradient dot product for a hypercube corner."""
        # Generate gradient vector: use hash bits to determine +1 or -1 for each dimension
        # Shape: (N, n_dims)
        bit_masks = 2 ** torch.arange(n_dims, device=device)  # [1, 2, 4, 8, ...]
        grad_vector = torch.where(
            (hash_val.unsqueeze(-1) & bit_masks) == 0,
            torch.ones(1, device=device),
            -torch.ones(1, device=device),
        )

        # Dot product with offset vector
        return (grad_vector * offset).sum(dim=-1)

    ### Hash function for n-dimensional coordinates
    def hash_coords(coords: torch.Tensor) -> torch.Tensor:
        """Hash n-dimensional integer coordinates via iterated permutation."""
        result = perm[coords[:, 0] & 255]
        for dim in range(1, n_dims):
            result = perm[(result + coords[:, dim]) & 255]
        return result

    ### Compute gradients at all 2^n hypercube corners
    corner_values = {}
    for corner in product([0, 1], repeat=n_dims):
        corner_tensor = torch.tensor(corner, device=device)
        corner_lattice = lattice_coords + corner_tensor
        corner_offset = fractional_coords - corner_tensor.to(points.dtype)
        hash_val = hash_coords(corner_lattice)
        corner_values[corner] = compute_gradient(hash_val, corner_offset)

    ### n-linear interpolation: iteratively reduce dimensions
    # Start with 2^n corner values, reduce to 1 by interpolating along each axis
    values = corner_values

    for dim in range(n_dims):
        new_values = {}
        weight = interp_weights[:, dim]

        # Group hypercube corners that differ only in current dimension
        processed = set()
        for corner in values.keys():
            if corner in processed:
                continue

            # Pair of corners differing in dimension 'dim'
            corner_0 = list(corner)
            corner_1 = list(corner)
            corner_0[dim] = 0
            corner_1[dim] = 1
            corner_0 = tuple(corner_0)
            corner_1 = tuple(corner_1)

            # Linear interpolation
            interpolated = values[corner_0] * (1 - weight) + values[corner_1] * weight

            # Store with this dimension collapsed to 0
            new_key = list(corner)
            new_key[dim] = 0
            new_values[tuple(new_key)] = interpolated

            processed.add(corner_0)
            processed.add(corner_1)

        values = new_values

    # Final interpolated value
    noise = values[tuple([0] * n_dims)]

    return noise


def perlin_noise_1d(
    points: torch.Tensor, scale: float = 1.0, seed: int = 0
) -> torch.Tensor:
    """1D Perlin noise.

    Convenience wrapper for perlin_noise_nd with 1D points.

    Parameters
    ----------
    points : torch.Tensor
        (N, 1) tensor of 1D positions.
    scale : float
        Frequency of noise.
    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor
        (N,) tensor of noise values.
    """
    return perlin_noise_nd(points, scale, seed)


def perlin_noise_2d(
    points: torch.Tensor, scale: float = 1.0, seed: int = 0
) -> torch.Tensor:
    """2D Perlin noise.

    Convenience wrapper for perlin_noise_nd with 2D points.

    Parameters
    ----------
    points : torch.Tensor
        (N, 2) tensor of 2D positions.
    scale : float
        Frequency of noise.
    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor
        (N,) tensor of noise values.
    """
    return perlin_noise_nd(points, scale, seed)


def perlin_noise_3d(
    points: torch.Tensor, scale: float = 1.0, seed: int = 0
) -> torch.Tensor:
    """3D Perlin noise.

    Convenience wrapper for perlin_noise_nd with 3D points.

    Parameters
    ----------
    points : torch.Tensor
        (N, 3) tensor of 3D positions.
    scale : float
        Frequency of noise.
    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor
        (N,) tensor of noise values.
    """
    return perlin_noise_nd(points, scale, seed)
