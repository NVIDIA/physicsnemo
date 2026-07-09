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

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

# TODO enum causes segmentation faults with current torch script. Go back to enum after torch script update
"""
@enum.unique
class InterpolationType(enum.Enum):
    NEAREST_NEIGHBOR = (1, 1)
    LINEAR = (2, 2)
    SMOOTH_STEP_1 = (3, 2)
    SMOOTH_STEP_2 = (4, 2)
    GAUSSIAN = (6, 5)

    def __init__(self, index, stride):
        self.index = index
        self.stride = stride
"""


@torch.compile
def linear_step(x: Tensor) -> Tensor:
    """
    Clips the input tensor values between 0 and 1 using a linear step.

    This function constrains each element in the input tensor to be in the range [0, 1].
    Values below 0 are set to 0, and values above 1 are set to 1.

    Parameters
    ----------
    x: Tensor
        Input tensor to be clipped.

    Returns
    -------
    Tensor
        A tensor with values clipped between 0 and 1.

    Example
    -------
    >>> x = torch.tensor([-0.5, 0.5, 1.5])
    >>> linear_step(x)
    tensor([0.0000, 0.5000, 1.0000])
    """
    return torch.clip(x, 0, 1)


@torch.compile
def smooth_step_1(x: Tensor) -> Tensor:
    r"""
    Compute the smooth step interpolation of the input tensor values.

    This function applies the smooth step function: \(f(x) = 3x^2 - 2x^3\)
    to each element in the input tensor, and then clips the result to be in the
    range [0, 1]. It's useful for creating a smooth transition between two values.

    parameters
    ----------
    x: Tensor
        Input tensor, with values expected to be in the range [0, 1] for
        meaningful interpolation.

    Returns
    -------
    Tensor
        A tensor with smooth step interpolated values, clipped between 0 and 1.
    """
    return torch.clip(3 * x**2 - 2 * x**3, 0, 1)


@torch.compile
def smooth_step_2(x: Tensor) -> Tensor:
    r"""
    Compute the enhanced smooth step interpolation of the input tensor values.

    This function applies the enhanced smooth step function:
    \(f(x) = x^3 (6x^2 - 15x + 10)\) to each element in the input tensor.
    The result is then clipped to be in the range [0, 1].

    Parameters
    ----------
    x: Tensor
        Input tensor, with values expected to be in the range [0, 1] for meaningful
        interpolation.

    Returns
    -------
    Tensor
        A tensor with enhanced smooth step interpolated values, clipped between 0 and 1.
    """
    return torch.clip(x**3 * (6 * x**2 - 15 * x + 10), 0, 1)


@torch.compile
def nearest_neighbor_weighting(dist_vec: Tensor, dx: Tensor) -> Tensor:
    """
    Compute the nearest neighbor weighting for the given distance vector.

    This function returns a tensor of ones with a shape derived from the input
    `dist_vec`. The resulting tensor represents weights in the context of nearest
    neighbor interpolation, where the closest point has a weight of one and all other
    points have a weight of zero.

    Parameters:
    ----------
    dist_vec: Tensor
        A tensor representing the distances from a set of points.
        The last two dimensions are expected to be spatial dimensions.
    dx: Tensor
        A tensor representing spacing between points.
        While it's provided as an input, it doesn't influence the output for this
        function since nearest neighbor weights are constant.

    Returns
    -------
    Tensor
        A tensor filled with ones and shaped according to `dist_vec`
        but with the last two dimensions reduced to single dimensions.

    """
    return torch.ones(
        dist_vec.shape[:-2] + (1, 1),
        device=dist_vec.device,
        dtype=dist_vec.dtype,
    )


def _hyper_cube_weighting(lower_point: Tensor, upper_point: Tensor) -> Tensor:
    dim = lower_point.shape[-1]
    weights = []
    weights = [upper_point[..., 0], lower_point[..., 0]]
    for i in range(1, dim):
        new_weights = []
        for w in weights:
            new_weights.append(w * upper_point[..., i])
            new_weights.append(w * lower_point[..., i])
        weights = new_weights
    weights = torch.stack(weights, dim=-1)
    return torch.unsqueeze(weights, dim=-1)


@torch.compile
def linear_weighting(dist_vec: Tensor, dx: Tensor) -> Tensor:
    """
    Compute the linear weighting based on the distance vector and spacing.

    Parameters
    ----------
    dist_vec: Tensor
        Distance vector for interpolation points.
    dx: Tensor
        Spacing between points.

    Returns
    -------
    Tensor
        Weights derived from the linear interpolation of the distance vector.
    """
    normalized_dist_vec = dist_vec / dx
    lower_point = normalized_dist_vec[..., 0, :]
    upper_point = -normalized_dist_vec[..., -1, :]
    return _hyper_cube_weighting(lower_point, upper_point)


@torch.compile
def smooth_step_1_weighting(dist_vec: Tensor, dx: Tensor) -> Tensor:
    """
    Compute the weighting using the `smooth_step_1` function on the normalized
    distance vector.

    Parameters
    ----------
    dist_vec: Tensor
        Distance vector for interpolation points.
    dx: Tensor
        Spacing between points.

    Returns
    -------
    Tensor
        Weights derived using the `smooth_step_1` interpolation of the distance vector.
    """
    normalized_dist_vec = dist_vec / dx
    lower_point = smooth_step_1(normalized_dist_vec[..., 0, :])
    upper_point = smooth_step_1(-normalized_dist_vec[..., -1, :])
    return _hyper_cube_weighting(lower_point, upper_point)


@torch.compile
def smooth_step_2_weighting(dist_vec: Tensor, dx: Tensor) -> Tensor:
    """
    Compute the weighting using the `smooth_step_2` function on the normalized
    distance vector.

    Parameters
    ----------
    dist_vec: Tensor
        Distance vector for interpolation points.
    dx: Tensor
        pacing between points.

    Returns
    -------
    Tensor
        Weights derived using the `smooth_step_2` interpolation of the distance vector.
    """
    normalized_dist_vec = dist_vec / dx
    lower_point = smooth_step_2(normalized_dist_vec[..., 0, :])
    upper_point = smooth_step_2(-normalized_dist_vec[..., -1, :])
    return _hyper_cube_weighting(lower_point, upper_point)


# @torch.compile
def gaussian_weighting(dist_vec: Tensor, dx: Tensor) -> Tensor:
    """
    Compute the Gaussian weighting based on the distance vector and spacing.

    Parameters
    ----------
    dist_vec: Tensor
        Distance vector for interpolation points.
    dx: Tensor
        Spacing between points.

    Returns
    -------
    Tensor
        Gaussian weights for the provided distance vector.
    """
    dim = dx.size(-1)
    sharpen = 2.0
    sigma = dx / sharpen
    factor = 1.0 / ((2.0 * math.pi) ** (dim / 2.0) * sigma.prod())
    gaussian = torch.exp(-0.5 * torch.square((dist_vec / sigma)))
    gaussian = factor * gaussian.prod(dim=-1)
    norm = gaussian.sum(dim=2, keepdim=True)
    weights = torch.unsqueeze(gaussian / norm, dim=3)
    return weights


# @torch.compile
def _gather_nd(params: Tensor, indices: Tensor) -> Tensor:
    """As seen here https://discuss.pytorch.org/t/how-to-do-the-tf-gather-nd-in-pytorch/6445/30"""
    orig_shape = list(indices.shape)
    num_samples = 1
    for s in orig_shape[:-1]:
        num_samples *= s
    m = orig_shape[-1]
    n = len(params.shape)

    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(
            "the last dimension of indices must less or equal to the rank of params. "
            f"Got indices:{indices.shape}, params:{params.shape}. {m} > {n}"
        )

    indices = indices.reshape((num_samples, m)).transpose(0, 1)
    index_tuple = tuple(indices[axis] for axis in range(m))
    output = params[index_tuple]  # (num_samples, ...)
    return output.reshape(out_shape).contiguous()


@torch.compile
def index_values_high_mem(points: Tensor, idx: Tensor) -> Tensor:
    """
    Index values from the `points` tensor using the provided indices `idx`.

    Parameters
    ----------
    points: Tensor
        The source tensor from which values will be indexed.
    idx: Tensor
        The tensor containing indices for indexing.

    Returns
    -------
    Tensor
        Indexed values from the `points` tensor.
    """
    idx = idx.unsqueeze(3).repeat_interleave(points.size(-1), dim=3)
    points = points.unsqueeze(1).repeat_interleave(idx.size(1), dim=1)
    out = torch.gather(points, dim=2, index=idx)
    return out


# @torch.compile
def index_values_low_mem(points: Tensor, idx: Tensor) -> Tensor:
    """
    Input:
        points: (b,m,c) float32 array, known points
        idx: (b,n,3) int32 array, indices to known points
    Output:
        out: (b,m,n,c) float32 array, interpolated point values
    """

    device = points.device
    idxShape = idx.shape
    batch_size = idxShape[0]
    num_points = idxShape[1]
    K = idxShape[2]
    num_features = points.shape[2]
    batch_indices = torch.reshape(
        torch.tile(
            torch.unsqueeze(torch.arange(0, batch_size).to(device), dim=0),
            (num_points * K,),
        ),
        [-1],
    )  # BNK
    point_indices = torch.reshape(idx, [-1])  # BNK
    vertices = _gather_nd(
        points, torch.stack((batch_indices, point_indices), dim=1)
    )  # BNKxC
    vertices4d = torch.reshape(
        vertices, [batch_size, num_points, K, num_features]
    )  # BxNxKxC
    return vertices4d


def _grid_knn_idx(
    query_points: Tensor,
    grid: List[Tuple[float, float, int]],
    stride: int,
    padding: bool = True,
) -> tuple[Tensor, Tensor]:
    # set k
    k = stride // 2

    # set device
    device = query_points.device

    # find nearest neighbors of query points from a grid
    # dx vector on grid
    dx = torch.tensor(
        [(x[1] - x[0]) / (x[2] - 1) for x in grid],
        device=device,
        dtype=torch.float32,
    )
    dx = dx.view(1, 1, len(grid))

    # min point on grid (this will change if we are padding the grid)
    start = torch.tensor(
        [val[0] for val in grid],
        device=device,
        dtype=torch.float32,
    )
    if padding:
        start = start - (k * dx)
    start = start.view(1, 1, len(grid))

    # this is the center nearest neighbor in the grid
    center_idx = (((query_points - start) / dx) + (stride / 2.0 % 1.0)).to(torch.int64)
    if padding and stride == 2:
        # In-bounds queries always use a real cell. This also handles an inward
        # float32 neighbor whose grid-space division rounds onto the padded
        # endpoint cell.
        first_real_cell = center_idx.new_ones(len(grid))
        final_real_cell = center_idx.new_tensor([x[2] - 1 for x in grid])
        lower_bound = query_points.new_tensor([x[0] for x in grid])
        upper_bound = query_points.new_tensor([x[1] for x in grid])
        real_center = torch.maximum(center_idx, first_real_cell)
        real_center = torch.minimum(real_center, final_real_cell)
        center_idx = torch.where(
            (query_points >= lower_bound) & (query_points <= upper_bound),
            real_center,
            center_idx,
        )

    # index window
    idx_add = (
        torch.arange(-((stride - 1) // 2), stride // 2 + 1).view(1, 1, -1).to(device)
    )

    # find all index in window around center index
    # TODO make for more general diminsions
    if len(grid) == 1:
        idx_row_0 = center_idx[..., 0:1] + idx_add
        idx = idx_row_0.view(*idx_row_0.shape[:2], stride)
    elif len(grid) == 2:
        dim_size_1 = grid[1][2]
        if padding:
            dim_size_1 += 2 * k
        idx_row_0 = dim_size_1 * (center_idx[..., 0:1] + idx_add)
        idx_row_0 = idx_row_0.unsqueeze(-1)
        idx_row_1 = center_idx[..., 1:2] + idx_add
        idx_row_1 = idx_row_1.unsqueeze(2)
        idx = (idx_row_0 + idx_row_1).view(*idx_row_0.shape[:2], stride**2)
    elif len(grid) == 3:
        dim_size_1 = grid[1][2]
        dim_size_2 = grid[2][2]
        if padding:
            dim_size_1 += 2 * k
            dim_size_2 += 2 * k
        idx_row_0 = dim_size_2 * dim_size_1 * (center_idx[..., 0:1] + idx_add)
        idx_row_0 = idx_row_0.unsqueeze(-1).unsqueeze(-1)
        idx_row_1 = dim_size_2 * (center_idx[..., 1:2] + idx_add)
        idx_row_1 = idx_row_1.unsqueeze(2).unsqueeze(-1)
        idx_row_2 = center_idx[..., 2:3] + idx_add
        idx_row_2 = idx_row_2.unsqueeze(2).unsqueeze(3)
        idx = (idx_row_0 + idx_row_1 + idx_row_2).view(*idx_row_0.shape[:2], stride**3)
    else:
        raise RuntimeError

    return idx, center_idx


# TODO currently the `tolist` operation is not supported by torch script and when fixed torch script will be used
# @torch.compile
def interpolation_torch(
    query_points: Tensor,
    context_grid: Tensor,
    grid: List[Tuple[float, float, int]],
    interpolation_type: str = "smooth_step_2",
    mem_speed_trade: bool = True,
) -> Tensor:
    """
    Interpolate values at `query_points` based on `context_grid` using specified
    interpolation methods.

    Parameters
    ----------
    query_points: Tensor
        Points at which interpolation is to be performed.
    context_grid: Tensor
        Source grid from which values are to be interpolated.
    grid: List[Tuple[float, float, int]]
        Describes the grid's range and resolution.
    interpolation_type: str, optional
        Type of interpolation to be used, by default "smooth_step_2".
    mem_speed_trade: bool, optional
        Trade-off between memory usage and speed.
        If True, uses low memory indexing, by default True.

    Returns
    -------
    Tensor
        Interpolated values at the `query_points`.
    """

    # set stride TODO this will be replaced with InterpolationType later
    if interpolation_type == "nearest_neighbor":
        stride = 1
    elif interpolation_type == "linear":
        stride = 2
    elif interpolation_type == "smooth_step_1":
        stride = 2
    elif interpolation_type == "smooth_step_2":
        stride = 2
    elif interpolation_type == "gaussian":
        stride = 5
    else:
        raise RuntimeError(f"Interpolation type {interpolation_type} not supported")

    # set device
    device = query_points.device

    # useful values
    dims = len(grid)
    nr_channels = context_grid.size(0)
    dx = [((x[1] - x[0]) / (x[2] - 1)) for x in grid]

    # generate mesh grid of position information [grid_dim_1, grid_dim_2, ..., 2-3]
    # NOTE the mesh grid is padded by stride//2
    k = stride // 2
    linspace = [
        torch.linspace(
            x[0] - k * dx_i,
            x[1] + k * dx_i,
            x[2] + 2 * k,
            device=device,
            dtype=torch.float32,
        )
        for x, dx_i in zip(grid, dx)
    ]
    meshgrid = torch.meshgrid(linspace, indexing="ij")
    meshgrid = torch.stack(meshgrid, dim=-1)

    # pad context grid by k to avoid cuts on corners
    padding = dims * (k, k)
    context_grid = F.pad(context_grid, padding)

    # reshape query points, context grid and mesh grid for easier indexing
    # [1, grid_dim_1*grid_dim_2*..., 2-4]
    nr_grid_points = 1
    for grid_axis in grid:
        nr_grid_points *= grid_axis[2] + 2 * k
    meshgrid = meshgrid.view(1, nr_grid_points, dims)
    context_grid = torch.reshape(context_grid, [1, nr_channels, nr_grid_points])
    context_grid = torch.swapaxes(context_grid, 1, 2)
    query_points = query_points.unsqueeze(0)

    # compute index of nearest neighbor on grid to query points
    idx, center_idx = _grid_knn_idx(query_points, grid, stride, padding=True)

    # index mesh grid to get distance vector
    if mem_speed_trade:
        mesh_grid_idx = index_values_low_mem(meshgrid, idx)
    else:
        mesh_grid_idx = index_values_high_mem(meshgrid, idx)
    dist_vec = query_points.unsqueeze(2) - mesh_grid_idx

    # make tf dx vec (for interpolation function)
    dx = torch.tensor(dx, dtype=torch.float32, device=device)
    dx = torch.reshape(dx, [1, 1, 1, dims])

    # compute bump function
    if interpolation_type == "nearest_neighbor":
        weights = nearest_neighbor_weighting(dist_vec, dx)
    elif stride == 2:
        # Only the lower-corner distance is needed to form the per-axis cell
        # fraction. Correct exact logical endpoints without materializing
        # endpoint masks over every stencil corner.
        dx_per_axis = dx[..., 0, :]
        fraction = dist_vec[..., 0, :] / dx_per_axis
        lower_bound = query_points.new_tensor([x[0] for x in grid])
        upper_bound = query_points.new_tensor([x[1] for x in grid])
        lower_distance = query_points - lower_bound
        upper_distance = upper_bound - query_points
        scaled_lower_distance = lower_distance / dx_per_axis
        scaled_upper_distance = upper_distance / dx_per_axis
        real_center = (center_idx - 1).to(query_points.dtype)
        resolutions = center_idx.new_tensor([x[2] for x in grid])
        intervals_to_upper = (resolutions - center_idx).to(query_points.dtype)
        fraction_from_lower = scaled_lower_distance - real_center
        fraction_from_upper = intervals_to_upper - scaled_upper_distance
        stable_fraction = torch.where(
            lower_distance <= upper_distance,
            fraction_from_lower,
            fraction_from_upper,
        )
        stable_fraction = (
            stable_fraction
            + (stable_fraction.clamp(0.0, 1.0) - stable_fraction).detach()
        )
        fraction = torch.where(
            (lower_distance >= 0.0) & (upper_distance >= 0.0),
            stable_fraction,
            fraction,
        )
        at_lower_bound = query_points == lower_bound
        at_upper_bound = query_points == upper_bound
        fraction = torch.where(
            at_lower_bound,
            fraction - fraction.detach(),
            fraction,
        )
        fraction = torch.where(
            at_upper_bound,
            fraction + (1.0 - fraction).detach(),
            fraction,
        )
        if interpolation_type == "linear":
            lower_point = fraction
            upper_point = 1.0 - fraction
        elif interpolation_type == "smooth_step_1":
            lower_point = smooth_step_1(fraction)
            upper_point = smooth_step_1(1.0 - fraction)
        else:
            lower_point = smooth_step_2(fraction)
            upper_point = smooth_step_2(1.0 - fraction)
        weights = _hyper_cube_weighting(lower_point, upper_point)
    elif interpolation_type == "gaussian":
        weights = gaussian_weighting(dist_vec, dx)
    else:
        raise RuntimeError

    # index context grid with index
    if mem_speed_trade:
        context_grid_idx = index_values_low_mem(context_grid, idx)
    else:
        context_grid_idx = index_values_high_mem(context_grid, idx)

    # interpolate points
    product = weights * context_grid_idx
    interpolated_points = product.sum(dim=2)

    return interpolated_points[0]
