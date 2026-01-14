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

    Args:
        point_data: Original point data, batch_size=(n_original_points,)
        edges: Edge connectivity, shape (n_edges, 2)
        n_original_points: Number of original points (for validation)

    Returns:
        New point_data with batch_size=(n_original_points + n_edges,)
        containing both original point data and interpolated edge midpoint data.

    Example:
        >>> # Original points: 3, edges: 2
        >>> # New points: 3 + 2 = 5
        >>> point_data["temperature"] = tensor([100, 200, 300])
        >>> edges = tensor([[0, 1], [1, 2]])
        >>> new_data = interpolate_point_data_to_edges(point_data, edges, 3)
        >>> # new_data["temperature"] = [100, 200, 300, 150, 250]
        >>> #                             original ^^^  ^^^^ edge midpoints
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

    return point_data.exclude("_cache").apply(
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

    Args:
        cell_data: Original cell data, batch_size=(n_parent_cells,)
        parent_indices: Parent cell index for each child, shape (n_total_children,)
        n_total_children: Total number of child cells

    Returns:
        New cell_data with batch_size=(n_total_children,) where each child
        has the same data values as its parent.

    Example:
        >>> # 2 parent cells, each splits into 4 children -> 8 total
        >>> cell_data["pressure"] = tensor([100.0, 200.0])
        >>> parent_indices = tensor([0, 0, 0, 0, 1, 1, 1, 1])
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
    return cell_data.exclude("_cache").apply(
        lambda tensor: tensor[parent_indices],
        batch_size=torch.Size([n_total_children]),
    )
