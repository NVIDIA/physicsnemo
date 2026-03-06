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

"""Data interpolation and propagation for mesh subdivision.

Handles interpolating point_data to edge midpoints and propagating cell_data
from parent cells to child cells, reusing existing aggregation infrastructure.
"""

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    pass


def interpolate_point_data_to_edges(
    point_data: TensorDict,
    edges: torch.Tensor,
    n_original_points: int,
) -> TensorDict:
    """Interpolate point_data to edge midpoints.

    For each edge, creates interpolated data at the midpoint by averaging
    the data values at the two endpoint vertices.

    Parameters
    ----------
    point_data : TensorDict
        Original point data, batch_size=(n_original_points,)
    edges : torch.Tensor
        Edge connectivity, shape (n_edges, 2)
    n_original_points : int
        Number of original points (for validation)

    Returns
    -------
    TensorDict
        New point_data with batch_size=(n_original_points + n_edges,)
        containing both original point data and interpolated edge midpoint data.

    Examples
    --------
        >>> import torch
        >>> from tensordict import TensorDict
        >>> # Original points: 3, edges: 2
        >>> # New points: 3 + 2 = 5
        >>> point_data = TensorDict({"temperature": torch.tensor([100., 200., 300.])}, batch_size=[3])
        >>> edges = torch.tensor([[0, 1], [1, 2]])
        >>> new_data = interpolate_point_data_to_edges(point_data, edges, 3)
        >>> # new_data["temperature"] = [100, 200, 300, 150, 250]
    """
    if len(point_data.keys()) == 0:
        # No data to interpolate
        return TensorDict(
            {},
            batch_size=torch.Size([n_original_points + len(edges)]),
            device=edges.device,
        )

    n_total_points = n_original_points + len(edges)

    ### Interpolate all fields using TensorDict.apply()
    def interpolate_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Interpolate a single tensor to edge midpoints."""
        # Only interpolate floating point or complex tensors
        # Integer/bool metadata (like IDs) cannot be meaningfully averaged
        if not (tensor.dtype.is_floating_point or tensor.dtype.is_complex):
            # For non-floating types, pad with zeros (will be filtered later if needed)
            # or we could assign arbitrary values; zeros are safe default
            edge_midpoint_values = torch.zeros(
                (len(edges), *tensor.shape[1:]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
        else:
            # Get endpoint values and average: shape (n_edges, *data_shape)
            edge_midpoint_values = tensor[edges].mean(dim=1)

        # Concatenate original and edge midpoint data
        return torch.cat([tensor, edge_midpoint_values], dim=0)

    return point_data.apply(
        interpolate_tensor,
        batch_size=torch.Size([n_total_points]),
    )


def propagate_cell_data_to_children(
    cell_data: TensorDict,
    parent_indices: torch.Tensor,
    n_total_children: int,
) -> TensorDict:
    """Propagate cell_data from parent cells to child cells.

    Each child cell inherits its parent's data values unchanged.
    Uses scatter operations for efficient vectorized propagation.

    Parameters
    ----------
    cell_data : TensorDict
        Original cell data, batch_size=(n_parent_cells,)
    parent_indices : torch.Tensor
        Parent cell index for each child, shape (n_total_children,)
    n_total_children : int
        Total number of child cells

    Returns
    -------
    TensorDict
        New cell_data with batch_size=(n_total_children,) where each child
        has the same data values as its parent.

    Examples
    --------
        >>> import torch
        >>> from tensordict import TensorDict
        >>> # 2 parent cells, each splits into 4 children -> 8 total
        >>> cell_data = TensorDict({"pressure": torch.tensor([100.0, 200.0])}, batch_size=[2])
        >>> parent_indices = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        >>> new_data = propagate_cell_data_to_children(cell_data, parent_indices, 8)
        >>> # new_data["pressure"] = [100, 100, 100, 100, 200, 200, 200, 200]
    """
    if len(cell_data.keys()) == 0:
        # No data to propagate
        return TensorDict(
            {},
            batch_size=torch.Size([n_total_children]),
            device=parent_indices.device,
        )

    ### Propagate all fields using TensorDict.apply()
    # Each child simply inherits its parent's value via indexing
    return cell_data.apply(
        lambda tensor: tensor[parent_indices],
        batch_size=torch.Size([n_total_children]),
    )
