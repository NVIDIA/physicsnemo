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

"""Utility functions for string-formatting Mesh representations."""

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from physicsnemo.mesh.mesh import Mesh


def format_mesh_repr(mesh: "Mesh") -> str:
    """Format a complete Mesh representation.

    Parameters
    ----------
    mesh : Mesh
        The Mesh instance to format.

    Returns
    -------
    str
        Formatted string representation of the mesh.
    """
    ### Build the first line: Mesh[manifold, spatial](n_points=..., n_cells=..., ...)
    class_name = mesh.__class__.__name__
    dim_sig = f"[n_manifold_dims={mesh.n_manifold_dims}, n_spatial_dims={mesh.n_spatial_dims}]"

    parts = [
        f"n_points={mesh.n_points}",
        f"n_cells={mesh.n_cells}",
    ]

    device = mesh.device
    if device is not None:
        parts.append(f"device={device}")

    first_line = f"{class_name}{dim_sig}({', '.join(parts)})"

    ### Format non-empty data fields with proper alignment
    data_fields = ["point_data", "cell_data", "global_data"]
    populated = [
        (name, getattr(mesh, name))
        for name in data_fields
        if len(list(getattr(mesh, name).keys())) > 0
    ]

    if not populated:
        return first_line

    max_field_len = max(len(name) for name, _ in populated)
    lines = [first_line]

    for field_name, td in populated:
        formatted_td = _format_tensordict_repr(
            td,
            batch_dims=len(td.batch_size) if hasattr(td, "batch_size") else 0,
            indent_level=1,
        )
        padded_field = field_name.ljust(max_field_len)
        lines.append(f"    {padded_field}: {formatted_td}")

    return "\n".join(lines)


def _get_trailing_shape(tensor: torch.Tensor, batch_dims: int) -> tuple:
    """Extract shape dimensions after the batch dimensions.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to extract shape from.
    batch_dims : int
        Number of leading batch dimensions to skip.

    Returns
    -------
    tuple
        Tuple of trailing dimensions.
    """
    if batch_dims >= len(tensor.shape):
        return ()
    return tuple(tensor.shape[batch_dims:])


def _format_tensordict_repr(
    td: TensorDict, batch_dims: int, indent_level: int = 0
) -> str:
    """Format a TensorDict with proper indentation and colon alignment.

    Parameters
    ----------
    td : TensorDict
        TensorDict to format.
    batch_dims : int
        Number of batch dimensions (for computing trailing shapes).
    indent_level : int
        Current indentation level.

    Returns
    -------
    str
        Formatted string representation.
    """
    keys = sorted(list(td.keys()))

    if len(keys) == 0:
        return "{}"

    # Count total fields to decide on single-line vs multi-line
    total_fields = len(list(td.keys(include_nested=True)))
    use_multiline = total_fields > 3

    if not use_multiline:
        # Single-line format
        items = []
        for key in keys:
            value = td[key]
            if isinstance(value, TensorDict):
                # Recursively format nested TensorDict
                nested_repr = _format_tensordict_repr(
                    value,
                    batch_dims=len(value.batch_size)
                    if hasattr(value, "batch_size")
                    else batch_dims,
                    indent_level=indent_level + 1,
                )
                items.append(f"{key}: {nested_repr}")
            else:
                # Format tensor with trailing shape
                if isinstance(value, torch.Tensor):
                    trailing_shape = _get_trailing_shape(value, batch_dims)
                    items.append(f"{key}: {trailing_shape}")
                else:
                    # Non-tensor, non-TensorDict value (shouldn't happen in practice)
                    items.append(f"{key}: <{type(value).__name__}>")
        return "{" + ", ".join(items) + "}"

    # Multi-line format
    next_indent = "    " * (indent_level + 1)

    # Find max key length for alignment
    max_key_len = max(len(str(key)) for key in keys)

    # Build field lines
    field_lines = []
    for i, key in enumerate(keys):
        value = td[key]
        padded_key = str(key).ljust(max_key_len)
        is_last = i == len(keys) - 1

        if isinstance(value, TensorDict):
            # Recursively format nested TensorDict
            nested_repr = _format_tensordict_repr(
                value,
                batch_dims=len(value.batch_size)
                if hasattr(value, "batch_size")
                else batch_dims,
                indent_level=indent_level + 1,
            )

            # Check if nested repr is multiline
            is_multiline_nested = "\n" in nested_repr

            if is_last:
                # Last item: add closing brace inline
                if is_multiline_nested:
                    # Nested multiline: add closing brace for nested, then parent
                    field_lines.append(f"{next_indent}{padded_key}: {nested_repr}}}")
                else:
                    # Nested single-line: add closing brace for parent
                    field_lines.append(f"{next_indent}{padded_key}: {nested_repr}}}")
            else:
                # Not last item: add comma after
                field_lines.append(f"{next_indent}{padded_key}: {nested_repr},")
        else:
            # Format tensor with trailing shape
            if isinstance(value, torch.Tensor):
                trailing_shape = _get_trailing_shape(value, batch_dims)
                shape_str = str(trailing_shape)
            else:
                # Non-tensor, non-TensorDict value (shouldn't happen in practice)
                shape_str = f"<{type(value).__name__}>"

            if is_last:
                # Last item: add closing brace inline
                field_lines.append(f"{next_indent}{padded_key}: {shape_str}}}")
            else:
                # Not last: add comma
                field_lines.append(f"{next_indent}{padded_key}: {shape_str},")

    # Return just the field lines - opening brace goes on the same line as parent key
    return "{\n" + "\n".join(field_lines)
