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

"""Utilities for processing scalar data for visualization."""

from typing import Literal

import torch
from tensordict import TensorDict


def process_scalars(
    scalar_spec: None | torch.Tensor | str | tuple[str, ...],
    data_dict: TensorDict,
    n_expected: int,
    name: str,
) -> tuple[torch.Tensor | None, Literal["points", "cells", None], str | None]:
    """Process scalar specification into concrete tensor values.

    Parameters
    ----------
    scalar_spec : torch.Tensor or str or tuple[str, ...] or None
        Scalar specification, can be:
        - None: no scalars to display
        - torch.Tensor: direct tensor values
        - str or tuple[str, ...]: key(s) to lookup in data_dict
    data_dict : TensorDict
        TensorDict containing data (point_data or cell_data).
    n_expected : int
        Expected number of scalars (n_points or n_cells).
    name : str
        Name for error messages ("point" or "cell").

    Returns
    -------
    tuple
        Tuple of (scalar_values, source_type, label) where:
        - scalar_values is None if scalar_spec is None, otherwise a 1D tensor
        - source_type indicates whether this is "points", "cells", or None
        - label is a human-readable name for the colorbar (or None)

    Raises
    ------
    ValueError
        If scalar specification is invalid or tensor has wrong shape.
    KeyError
        If specified key is not found in data_dict.
    """
    if scalar_spec is None:
        return None, None, None

    ### Case 1: Direct tensor specification
    if isinstance(scalar_spec, torch.Tensor):
        scalar_tensor = scalar_spec.cpu()

        # Check first dimension matches expected count
        if scalar_tensor.shape[0] != n_expected:
            raise ValueError(
                f"{name}_scalars tensor has wrong first dimension.\n"
                f"Expected {n_expected}, got {scalar_tensor.shape[0]}.\n"
                f"Full shape: {scalar_tensor.shape}"
            )

        # If multi-dimensional, compute L2 norm across trailing dimensions
        if scalar_tensor.ndim > 1:
            # Flatten all trailing dimensions and compute norm
            scalar_tensor = scalar_tensor.reshape(n_expected, -1)
            scalar_tensor = torch.norm(scalar_tensor, dim=-1)
            label = "Magnitude"
        else:
            label = "Scalars"

        return scalar_tensor, name + "s", label  # "points" or "cells"

    ### Case 2: Key lookup in TensorDict (str or tuple[str, ...])
    if isinstance(scalar_spec, (str, tuple)):
        try:
            scalar_tensor = data_dict[scalar_spec].cpu()
        except KeyError as e:
            raise KeyError(
                f"{name}_scalars key {scalar_spec!r} not found in {name}_data.\n"
                f"Available keys: {list(data_dict.keys())}"
            ) from e

        # Check first dimension matches expected count
        if scalar_tensor.shape[0] != n_expected:
            raise ValueError(
                f"{name}_scalars from key {scalar_spec!r} has wrong first dimension.\n"
                f"Expected {n_expected}, got {scalar_tensor.shape[0]}.\n"
                f"Full shape: {scalar_tensor.shape}"
            )

        # Determine label: use key name, but append " (magnitude)" if we take norm
        label = str(scalar_spec)

        # If multi-dimensional, compute L2 norm across trailing dimensions
        if scalar_tensor.ndim > 1:
            scalar_tensor = scalar_tensor.reshape(n_expected, -1)
            scalar_tensor = torch.norm(scalar_tensor, dim=-1)
            label = f"{label} (magnitude)"

        return scalar_tensor, name + "s", label  # "points" or "cells"

    raise TypeError(
        f"{name}_scalars must be None, torch.Tensor, str, or tuple[str, ...], "
        f"got {type(scalar_spec)=}"
    )


def validate_and_process_scalars(
    point_scalars: None | torch.Tensor | str | tuple[str, ...],
    cell_scalars: None | torch.Tensor | str | tuple[str, ...],
    point_data: TensorDict,
    cell_data: TensorDict,
    n_points: int,
    n_cells: int,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    Literal["points", "cells", None],
    str | None,
]:
    """Validate and process both point and cell scalars.

    Parameters
    ----------
    point_scalars : torch.Tensor or str or tuple[str, ...] or None
        Point scalar specification.
    cell_scalars : torch.Tensor or str or tuple[str, ...] or None
        Cell scalar specification.
    point_data : TensorDict
        TensorDict with point data.
    cell_data : TensorDict
        TensorDict with cell data.
    n_points : int
        Number of points in mesh.
    n_cells : int
        Number of cells in mesh.

    Returns
    -------
    tuple
        Tuple of (point_scalar_values, cell_scalar_values, active_scalar_source,
        scalar_label) where scalar_label is the human-readable name for the colorbar.

    Raises
    ------
    ValueError
        If both point_scalars and cell_scalars are specified (mutually exclusive).
    """
    ### Validate mutual exclusivity
    if point_scalars is not None and cell_scalars is not None:
        raise ValueError(
            "point_scalars and cell_scalars are mutually exclusive.\n"
            "Only one can be specified at a time to avoid colormap ambiguity."
        )

    ### Process point scalars
    point_values, point_source, point_label = process_scalars(
        point_scalars, point_data, n_points, "point"
    )

    ### Process cell scalars
    cell_values, cell_source, cell_label = process_scalars(
        cell_scalars, cell_data, n_cells, "cell"
    )

    ### Determine active scalar source and label
    active_scalar_source = point_source or cell_source
    scalar_label = point_label or cell_label

    return point_values, cell_values, active_scalar_source, scalar_label
