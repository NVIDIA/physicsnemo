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
FieldSlice - Select specific indices or slices from tensor dimensions.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.base import Transform

# Type for a single dimension's slice specification
# Can be: list of indices [0, 2, 5], or dict for slice {"start": 0, "stop": 5, "step": 2}
SliceSpec = list[int] | dict[str, int]


@register()
class FieldSlice(Transform):
    """
    Select specific indices or slices from tensor dimensions.

    This transform allows selecting subsets of data along any dimension of
    specified fields. It supports two modes:

    1. **Index selection**: Provide a list of indices to select
    2. **Slice selection**: Provide start/stop/step as a dict

    Parameters
    ----------
    slicing : dict[str, dict[int | str, SliceSpec]]
        Dictionary mapping field names to dimension slicing specs.
        Format::

            {
                "field_name": {
                    dim: indices_or_slice,
                    ...
                },
                ...
            }

        Where:

        - ``dim`` is the dimension index (int, or str for Hydra like "-1")
        - ``indices_or_slice`` is either:
            - A list of indices: ``[0, 2, 5]``
            - A slice dict: ``{"start": 0, "stop": 5, "step": 1}``

    Examples
    --------
    Index selection - select features 0, 2, 5 from last dimension:

    >>> transform = FieldSlice({
    ...     "features": {-1: [0, 2, 5]},
    ... })
    >>> # Input shape: (N, 10) -> Output shape: (N, 3)

    Slice selection - select first 5 features:

    >>> transform = FieldSlice({
    ...     "features": {-1: {"start": 0, "stop": 5}},
    ... })
    >>> # Input shape: (N, 10) -> Output shape: (N, 5)

    Multiple dimensions:

    >>> transform = FieldSlice({
    ...     "grid": {
    ...         0: [0, 1, 2],      # First 3 indices of dim 0
    ...         -1: {"stop": 4},   # First 4 of last dim (slice)
    ...     },
    ... })

    Hydra configuration example:

    .. code-block:: yaml

        _target_: physicsnemo.datapipes.transforms.FieldSlice
        slicing:
          features:
            "-1": [0, 2, 5]
          velocity:
            "-1":
              stop: 2
    """

    def __init__(
        self,
        slicing: dict[str, dict[int | str, SliceSpec]],
    ) -> None:
        """
        Initialize the FieldSlice transform.

        Parameters
        ----------
        slicing : dict[str, dict[int | str, SliceSpec]]
            Dictionary mapping field names to dimension slicing specs.
            See class docstring for detailed format description.
        """
        super().__init__()
        self.slicing = slicing

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Apply slicing to the specified fields.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict.

        Returns
        -------
        TensorDict
            TensorDict with sliced fields.

        Raises
        ------
        KeyError
            If a specified field is not in the TensorDict.
        """
        updates = {}

        for field_name, dim_specs in self.slicing.items():
            if field_name not in data.keys():
                raise KeyError(
                    f"Field '{field_name}' not found in data. "
                    f"Available: {list(data.keys())}"
                )

            tensor = data[field_name]

            for dim_key, spec in dim_specs.items():
                # Handle string keys from Hydra/YAML (e.g., "-1" -> -1)
                dim = int(dim_key)
                # Normalize negative dimension
                if dim < 0:
                    dim = tensor.ndim + dim

                tensor = self._apply_slice(tensor, dim, spec)

            updates[field_name] = tensor

        return data.update(updates)

    def _apply_slice(
        self,
        tensor: torch.Tensor,
        dim: int,
        spec: SliceSpec,
    ) -> torch.Tensor:
        """
        Apply a single slice specification to a tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor.
        dim : int
            Dimension to slice (normalized to positive).
        spec : SliceSpec
            Slice specification (list of indices or slice dict).

        Returns
        -------
        torch.Tensor
            Sliced tensor.

        Raises
        ------
        TypeError
            If spec is not a list or dict.
        """
        if isinstance(spec, list):
            # Index selection: [0, 2, 5]
            indices = torch.tensor(spec, dtype=torch.long, device=tensor.device)
            return torch.index_select(tensor, dim, indices)
        elif isinstance(spec, dict):
            # Slice selection: {"start": 0, "stop": 5, "step": 1}
            start = spec.get("start", None)
            stop = spec.get("stop", None)
            step = spec.get("step", None)

            # Build slice object
            slc = slice(start, stop, step)

            # Apply slice using narrow or direct indexing
            # We need to build the full index tuple
            idx = [slice(None)] * tensor.ndim
            idx[dim] = slc
            return tensor[tuple(idx)]
        else:
            raise TypeError(
                f"Invalid slice spec type: {type(spec)}. "
                f"Expected list of indices or dict with start/stop/step."
            )

    def extra_repr(self) -> str:
        """
        Return extra information for repr.

        Returns
        -------
        str
            String with transform parameters.
        """
        return f"slicing={self.slicing}"
