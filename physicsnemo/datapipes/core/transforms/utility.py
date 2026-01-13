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
Utility transforms for key management and tensor generation.

Provides transforms for renaming keys, removing (purging) keys from TensorDicts,
and creating constant-filled tensors.
"""

from __future__ import annotations

from typing import Optional

import torch
from tensordict import TensorDict

from physicsnemo.datapipes.core.registry import register
from physicsnemo.datapipes.core.transforms.base import Transform


@register()
class Rename(Transform):
    r"""
    Rename keys in a TensorDict.

    Replaces existing key names with new names according to a mapping.
    The tensor data is preserved, only the keys are changed.

    Example:
        >>> transform = Rename(mapping={"old_name": "new_name", "x": "positions"})
        >>> data = TensorDict({
        ...     "old_name": torch.randn(100, 3),
        ...     "x": torch.randn(100, 3),
        ...     "other": torch.randn(100, 1)
        ... })
        >>> result = transform(data)
        >>> print(list(result.keys()))
        ['new_name', 'positions', 'other']
    """

    def __init__(
        self,
        mapping: dict[str, str],
        *,
        strict: bool = True,
    ) -> None:
        """
        Initialize the rename transform.

        Args:
            mapping: Dictionary mapping old key names to new key names.
                    Keys are the original names, values are the new names.
            strict: If True, raise an error if a key in the mapping is not found
                   in the data. If False, silently skip missing keys.
        """
        super().__init__()
        self.mapping = mapping
        self.strict = strict

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Rename keys according to the mapping.

        Args:
            data: Input TensorDict with keys to rename.

        Returns:
            TensorDict with renamed keys.

        Raises:
            KeyError: If strict=True and a key in the mapping is not found.
            ValueError: If a new key name already exists in the data.
        """
        # Check for missing keys if strict mode
        data_keys = set(str(k) for k in data.keys())
        if self.strict:
            missing_keys = set(self.mapping.keys()) - data_keys
            if missing_keys:
                raise KeyError(
                    f"Keys not found in data: {missing_keys}. "
                    f"Available keys: {list(data.keys())}"
                )

        # Check for conflicts with new names
        existing_keys = data_keys
        keys_to_rename = set(self.mapping.keys()) & existing_keys
        new_names = {self.mapping[k] for k in keys_to_rename}
        keys_not_renamed = existing_keys - keys_to_rename

        conflicts = new_names & keys_not_renamed
        if conflicts:
            raise ValueError(f"New key names conflict with existing keys: {conflicts}")

        # Build new data dict with renamed keys
        new_data = {}
        for key in data.keys():
            if key in self.mapping:
                new_data[self.mapping[key]] = data[key]
            else:
                new_data[key] = data[key]

        return TensorDict(new_data, batch_size=data.batch_size)

    def extra_repr(self) -> str:
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

    Example (drop mode):
        >>> transform = Purge(drop_only=["temp", "debug_info"])
        >>> data = TensorDict({
        ...     "positions": torch.randn(100, 3),
        ...     "temp": torch.randn(100, 1),
        ...     "debug_info": torch.randn(100, 10)
        ... })
        >>> result = transform(data)
        >>> print(list(result.keys()))
        ['positions']

    Example (keep mode):
        >>> transform = Purge(keep_only=["positions", "velocities"])
        >>> data = TensorDict({
        ...     "positions": torch.randn(100, 3),
        ...     "velocities": torch.randn(100, 3),
        ...     "temp": torch.randn(100, 1)
        ... })
        >>> result = transform(data)
        >>> print(list(result.keys()))
        ['positions', 'velocities']
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

        Args:
            keep_only: List of keys to keep. All other keys will be removed.
                      Cannot be used together with drop_only.
            drop_only: List of keys to remove. All other keys will be kept.
                      Cannot be used together with keep_only. Default is None
                      (drop nothing).
            strict: If True, raise an error if a specified key is not found
                   in the data. If False, silently skip missing keys.

        Raises:
            ValueError: If both keep_only and drop_only are specified.
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

        Args:
            data: Input TensorDict.

        Returns:
            TensorDict with keys removed according to the configuration.

        Raises:
            KeyError: If strict=True and a specified key is not found.
        """
        available_keys = set(str(k) for k in data.keys())

        if self.keep_only is not None:
            # Keep only mode: keep specified keys, remove everything else
            keys_to_keep = set(self.keep_only)

            if self.strict:
                missing_keys = keys_to_keep - available_keys
                if missing_keys:
                    raise KeyError(
                        f"Keys specified in 'keep_only' not found in data: {missing_keys}. "
                        f"Available keys: {list(data.keys())}"
                    )

            # Only keep keys that exist and are in keep_only
            final_keys = keys_to_keep & available_keys

        elif self.drop_only is not None:
            # Drop only mode: remove specified keys, keep everything else
            keys_to_drop = set(self.drop_only)

            if self.strict:
                missing_keys = keys_to_drop - available_keys
                if missing_keys:
                    raise KeyError(
                        f"Keys specified in 'drop_only' not found in data: {missing_keys}. "
                        f"Available keys: {list(data.keys())}"
                    )

            # Keep all keys except those to drop
            final_keys = available_keys - keys_to_drop

        else:
            # Default: drop nothing, keep everything
            final_keys = available_keys

        # Build new TensorDict with only the final keys
        new_data = {key: data[key] for key in final_keys}

        return TensorDict(new_data, batch_size=data.batch_size)

    def extra_repr(self) -> str:
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

    Example:
        >>> # Create zeros (default)
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

    Example:
        >>> # Create ones
        >>> transform = ConstantField(
        ...     reference_key="positions",
        ...     output_key="mask",
        ...     fill_value=1.0,
        ...     output_dim=1
        ... )

    Example:
        >>> # Create custom constant
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

        Args:
            reference_key: Key for the tensor to use as shape reference.
                          The first dimension of this tensor determines
                          the number of rows in the output.
            output_key: Key to store the constant tensor.
            fill_value: The constant value to fill the tensor with.
                       Defaults to 0.0.
            output_dim: Feature dimension for output tensor. Creates tensor with
                       shape (N, output_dim) where N is the first dimension of
                       the reference tensor. Defaults to 1.
        """
        super().__init__()
        self.reference_key = reference_key
        self.output_key = output_key
        self.fill_value = fill_value
        self.output_dim = output_dim

    def __call__(self, data: TensorDict) -> TensorDict:
        """Create constant-filled tensor matching reference shape."""
        if self.reference_key not in data.keys():
            raise KeyError(
                f"Reference key '{self.reference_key}' not found in data. "
                f"Available keys: {list(data.keys())}"
            )

        reference = data[self.reference_key]
        n_points = reference.shape[0]

        constant_tensor = torch.full(
            (n_points, self.output_dim),
            self.fill_value,
            dtype=reference.dtype,
            device=reference.device,
        )

        return data.update({self.output_key: constant_tensor})

    def extra_repr(self) -> str:
        return (
            f"reference_key={self.reference_key}, "
            f"output_key={self.output_key}, "
            f"fill_value={self.fill_value}, "
            f"output_dim={self.output_dim}"
        )
