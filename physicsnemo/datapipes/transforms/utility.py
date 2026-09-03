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

"""
Utility transforms for key management and tensor generation.

Provides transforms for renaming keys, removing (purging) keys from TensorDicts,
and creating constant-filled tensors.
"""

from __future__ import annotations

from typing import Optional

import torch
from tensordict import TensorDict

from physicsnemo.datapipes.keys import (
    as_nested_key,
    as_nested_keys,
    exclude_keys,
    format_leaf_keys,
    get_leaf,
    key_to_str,
    leaf_keys,
    present_keys,
    rename_keys,
)
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.base import Transform


@register()
class Rename(Transform):
    r"""
    Rename keys in a TensorDict.

    Replaces existing key names with new names according to a mapping.
    The tensor data is preserved, only the keys are changed.

    Nested tensordicts can use this too: a ``'.'`` in a name addresses a
    nested key, so ``{"a.d": "a.c"}`` renames ``a["d"]`` to ``a["c"]`` and
    ``{"a.d": "d"}`` hoists it to the top level. Renaming a whole
    sub-TensorDict (``{"a": "b"}``) is also supported. See
    :mod:`physicsnemo.datapipes.keys`.

    Parameters
    ----------
    mapping : dict[str, str]
        Dictionary mapping old key names to new key names.
        Keys are the original names, values are the new names.
    strict : bool, default=True
        If True, raise an error if a key in the mapping is not found
        in the data. If False, silently skip missing keys.

    Examples
    --------
    >>> transform = Rename(mapping={"old_name": "new_name", "x": "positions"})
    >>> data = TensorDict({
    ...     "old_name": torch.randn(100, 3),
    ...     "x": torch.randn(100, 3),
    ...     "other": torch.randn(100, 1)
    ... })
    >>> result = transform(data)
    >>> print(sorted(result.keys()))
    ['new_name', 'other', 'positions']
    """

    def __init__(
        self,
        mapping: dict[str, str],
        *,
        strict: bool = True,
    ) -> None:
        """
        Initialize the rename transform.

        Parameters
        ----------
        mapping : dict[str, str]
            Dictionary mapping old key names to new key names.
            Keys are the original names, values are the new names.
        strict : bool, default=True
            If True, raise an error if a key in the mapping is not found
            in the data. If False, silently skip missing keys.
        """
        super().__init__()
        self.mapping = mapping
        self.strict = strict
        self._key_mapping = {
            as_nested_key(old): as_nested_key(new) for old, new in mapping.items()
        }

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Rename keys according to the mapping.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict with keys to rename.

        Returns
        -------
        TensorDict
            TensorDict with renamed keys.

        Raises
        ------
        KeyError
            If strict=True and a key in the mapping is not found.
        ValueError
            If a new key name already exists in the data.
        """
        return rename_keys(data, self._key_mapping, strict=self.strict)

    def extra_repr(self) -> str:
        """
        Return extra information for repr.

        Returns
        -------
        str
            String with transform parameters.
        """
        return f"mapping={self.mapping}, strict={self.strict}"


@register()
class Purge(Transform):
    r"""
    Remove keys and their associated tensors from a TensorDict.

    Supports two mutually exclusive modes:

    - drop_only: Specify keys to remove (keep everything else)
    - keep_only: Specify keys to keep (remove everything else)

    Only one mode can be active at a time. By default, drop_only=None means
    no keys are dropped (identity transform).

    Parameters
    ----------
    keep_only : list[str], optional
        List of keys to keep. All other keys will be removed.
        Cannot be used together with drop_only.
    drop_only : list[str], optional
        List of keys to remove. All other keys will be kept.
        Cannot be used together with keep_only. Default is None
        (drop nothing).
    strict : bool, default=True
        If True, raise an error if a specified key is not found
        in the data. If False, silently skip missing keys.

    Examples
    --------
    Drop mode - remove specific keys:

    >>> transform = Purge(drop_only=["temp", "debug_info"])
    >>> data = TensorDict({
    ...     "positions": torch.randn(100, 3),
    ...     "temp": torch.randn(100, 1),
    ...     "debug_info": torch.randn(100, 10)
    ... })
    >>> result = transform(data)
    >>> print(list(result.keys()))
    ['positions']

    Keep mode - keep only specific keys:

    >>> transform = Purge(keep_only=["positions", "velocities"])
    >>> data = TensorDict({
    ...     "positions": torch.randn(100, 3),
    ...     "velocities": torch.randn(100, 3),
    ...     "temp": torch.randn(100, 1)
    ... })
    >>> result = transform(data)
    >>> print(list(result.keys()))
    ['positions', 'velocities']

    Raises
    ------
    ValueError
        If both keep_only and drop_only are specified.
    """

    def __init__(
        self,
        *,
        keep_only: Optional[list[str]] = None,
        drop_only: Optional[list[str]] = None,
        strict: bool = True,
    ) -> None:
        """
        Initialize the purge transform.

        Parameters
        ----------
        keep_only : list[str], optional
            List of keys to keep. All other keys will be removed.
            Cannot be used together with drop_only.
        drop_only : list[str], optional
            List of keys to remove. All other keys will be kept.
            Cannot be used together with keep_only. Default is None
            (drop nothing).
        strict : bool, default=True
            If True, raise an error if a specified key is not found
            in the data. If False, silently skip missing keys.

        Raises
        ------
        ValueError
            If both keep_only and drop_only are specified.
        """
        super().__init__()

        if keep_only is not None and drop_only is not None:
            raise ValueError(
                "Cannot specify both 'keep_only' and 'drop_only'. "
                "Use only one option at a time."
            )

        self.keep_only = keep_only
        self.drop_only = drop_only
        self.strict = strict

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Remove or keep specified keys from the TensorDict.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict.

        Returns
        -------
        TensorDict
            TensorDict with keys removed according to the configuration.

        Raises
        ------
        KeyError
            If strict=True and a specified key is not found.
        """
        ### Leaf keys, nested ones included (tuples for nested, strings for flat)
        available_keys = set(leaf_keys(data))

        if self.keep_only is not None:
            keys_to_keep = as_nested_keys(self.keep_only)

            if self.strict:
                missing_keys = set(keys_to_keep) - available_keys
                if missing_keys:
                    raise KeyError(
                        f"Keys specified in 'keep_only' not found in data: "
                        f"{sorted(map(key_to_str, missing_keys))}. "
                        f"Available keys: {format_leaf_keys(data)}"
                    )

            ### ``select`` handles nested keys; in non-strict mode drop the
            ### missing ones ourselves (a path through a leaf tensor would
            ### otherwise make ``select`` raise rather than skip).
            if not self.strict:
                keys_to_keep = present_keys(data, keys_to_keep)
            return data.select(*keys_to_keep)

        elif self.drop_only is not None:
            keys_to_drop = as_nested_keys(self.drop_only)

            if self.strict:
                missing_keys = set(keys_to_drop) - available_keys
                if missing_keys:
                    raise KeyError(
                        f"Keys specified in 'drop_only' not found in data: "
                        f"{sorted(map(key_to_str, missing_keys))}. "
                        f"Available keys: {format_leaf_keys(data)}"
                    )

            ### ``exclude`` handles nested keys; filter to present ones for
            ### the same reason as above.
            return exclude_keys(data, keys_to_drop)

        else:
            # Default: drop nothing, keep everything
            return data.clone()

    def extra_repr(self) -> str:
        """
        Return extra information for repr.

        Returns
        -------
        str
            String with transform parameters.
        """
        if self.keep_only is not None:
            return f"keep_only={self.keep_only}, strict={self.strict}"
        elif self.drop_only is not None:
            return f"drop_only={self.drop_only}, strict={self.strict}"
        else:
            return "drop_only=None (identity)"


@register()
class ConstantField(Transform):
    r"""
    Create a tensor filled with a constant value.

    Creates a tensor where the first dimension matches a reference tensor
    and the last dimension is configurable. The tensor is filled with the
    specified constant value. Useful for creating placeholder tensors like
    zero SDF values for surface points, or indicator fields.

    Parameters
    ----------
    reference_key : str
        Key for the tensor to use as shape reference.
        The first dimension of this tensor determines
        the number of rows in the output.
    output_key : str
        Key to store the constant tensor.
    fill_value : float, default=0.0
        The constant value to fill the tensor with.
    output_dim : int, default=1
        Feature dimension for output tensor. Creates tensor with
        shape (N, output_dim) where N is the first dimension of
        the reference tensor.

    Examples
    --------
    Create zeros (default):

    >>> transform = ConstantField(
    ...     reference_key="positions",
    ...     output_key="sdf",
    ...     output_dim=1
    ... )
    >>> data = TensorDict({"positions": torch.randn(10000, 3)})
    >>> result = transform(data)
    >>> print(result["sdf"].shape)
    torch.Size([10000, 1])
    >>> print(result["sdf"][0, 0].item())
    0.0

    Create ones:

    >>> transform = ConstantField(
    ...     reference_key="positions",
    ...     output_key="mask",
    ...     fill_value=1.0,
    ...     output_dim=1
    ... )

    Create custom constant:

    >>> transform = ConstantField(
    ...     reference_key="positions",
    ...     output_key="temperature",
    ...     fill_value=293.15,  # Room temperature in Kelvin
    ...     output_dim=1
    ... )
    """

    def __init__(
        self,
        reference_key: str,
        output_key: str,
        *,
        fill_value: float = 0.0,
        output_dim: int = 1,
    ) -> None:
        """
        Initialize the constant field creation transform.

        Parameters
        ----------
        reference_key : str
            Key for the tensor to use as shape reference.
            The first dimension of this tensor determines
            the number of rows in the output.
        output_key : str
            Key to store the constant tensor.
        fill_value : float, default=0.0
            The constant value to fill the tensor with.
        output_dim : int, default=1
            Feature dimension for output tensor. Creates tensor with
            shape (N, output_dim) where N is the first dimension of
            the reference tensor.
        """
        super().__init__()
        self.reference_key = as_nested_key(reference_key)
        self.output_key = as_nested_key(output_key)
        self.fill_value = fill_value
        self.output_dim = output_dim

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Create constant-filled tensor matching reference shape.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict containing the reference tensor.

        Returns
        -------
        TensorDict
            TensorDict with the constant tensor added.

        Raises
        ------
        KeyError
            If the reference key is not found in the data.
        """
        reference = get_leaf(data, self.reference_key, what="Reference key")
        n_points = reference.shape[0]

        constant_tensor = torch.full(
            (n_points, self.output_dim),
            self.fill_value,
            dtype=reference.dtype,
            device=reference.device,
        )

        return data.update({self.output_key: constant_tensor})

    def extra_repr(self) -> str:
        """
        Return extra information for repr.

        Returns
        -------
        str
            String with transform parameters.
        """
        return (
            f"reference_key={key_to_str(self.reference_key)}, "
            f"output_key={key_to_str(self.output_key)}, "
            f"fill_value={self.fill_value}, "
            f"output_dim={self.output_dim}"
        )
