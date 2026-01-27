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
Field concatenation and manipulation transforms.

Provides transforms for concatenating multiple tensor fields and
normalizing vector fields.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.base import Transform


@register()
class ConcatFields(Transform):
    r"""
    Concatenate multiple tensor fields along a specified dimension.

    Combines specified fields into a single output tensor by concatenating
    along the feature dimension. Useful for building embeddings from multiple
    components like positions, normals, and signed distance fields.

    All input tensors must have the same shape except for the concatenation dimension.

    Parameters
    ----------
    input_keys : list[str]
        List of tensor keys to concatenate, in order.
    output_key : str
        Key to store the concatenated result.
    dim : int, default=-1
        Dimension along which to concatenate.
    skip_missing : bool, default=False
        If True, skip keys that are not present in the data instead of
        raising an error. Useful for optional fields.

    Examples
    --------
    >>> transform = ConcatFields(
    ...     input_keys=["positions", "sdf", "normals"],
    ...     output_key="embeddings"
    ... )
    >>> data = TensorDict({
    ...     "positions": torch.randn(10000, 3),
    ...     "sdf": torch.randn(10000, 1),
    ...     "normals": torch.randn(10000, 3)
    ... })
    >>> result = transform(data)
    >>> print(result["embeddings"].shape)
    torch.Size([10000, 7])
    """

    def __init__(
        self,
        input_keys: list[str],
        output_key: str,
        *,
        dim: int = -1,
        skip_missing: bool = False,
    ) -> None:
        """
        Initialize the concatenation transform.

        Parameters
        ----------
        input_keys : list[str]
            List of tensor keys to concatenate, in order.
        output_key : str
            Key to store the concatenated result.
        dim : int, default=-1
            Dimension along which to concatenate.
        skip_missing : bool, default=False
            If True, skip keys that are not present in the data
            instead of raising an error. Useful for optional fields.
        """
        super().__init__()
        self.input_keys = input_keys
        self.output_key = output_key
        self.dim = dim
        self.skip_missing = skip_missing

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Concatenate the specified fields.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict containing fields to concatenate.

        Returns
        -------
        TensorDict
            TensorDict with concatenated result added.

        Raises
        ------
        KeyError
            If a required key is not found (unless skip_missing=True).
        RuntimeError
            If tensors have incompatible shapes for concatenation.
        ValueError
            If no tensors are found to concatenate.
        """
        tensors = []

        for key in self.input_keys:
            if key not in data.keys():
                if self.skip_missing:
                    continue
                raise KeyError(
                    f"Input key '{key}' not found in data. "
                    f"Available keys: {list(data.keys())}"
                )
            tensors.append(data[key])

        if not tensors:
            raise ValueError(
                f"No tensors found to concatenate. "
                f"Input keys: {self.input_keys}, skip_missing={self.skip_missing}"
            )

        # Concatenate along specified dimension
        result = torch.cat(tensors, dim=self.dim)

        return data.update({self.output_key: result})

    def extra_repr(self) -> str:
        """
        Return extra information for repr.

        Returns
        -------
        str
            String with transform parameters.
        """
        return f"input_keys={self.input_keys}, output_key={self.output_key}, dim={self.dim}"


@register()
class NormalizeVectors(Transform):
    r"""
    Normalize vectors to unit length.

    Divides vectors by their L2 norm along the specified dimension.
    Handles zero-length vectors by adding a small epsilon to prevent division by zero.

    Parameters
    ----------
    input_keys : list[str]
        List of tensor keys to normalize.
    dim : int, default=-1
        Dimension along which to compute norm.
    eps : float, default=1e-6
        Small value to prevent division by zero.

    Examples
    --------
    >>> transform = NormalizeVectors(input_keys=["normals"])
    >>> data = TensorDict({"normals": torch.randn(10000, 3)})
    >>> result = transform(data)
    >>> # Normals are now unit length
    >>> norms = torch.norm(result["normals"], dim=-1)
    >>> print(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))
    True
    """

    def __init__(
        self,
        input_keys: list[str],
        *,
        dim: int = -1,
        eps: float = 1e-6,
    ) -> None:
        """
        Initialize the vector normalization transform.

        Parameters
        ----------
        input_keys : list[str]
            List of tensor keys to normalize.
        dim : int, default=-1
            Dimension along which to compute norm.
        eps : float, default=1e-6
            Small value to prevent division by zero.
        """
        super().__init__()
        self.input_keys = input_keys
        self.dim = dim
        self.eps = eps

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Normalize vectors to unit length.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict containing vectors to normalize.

        Returns
        -------
        TensorDict
            TensorDict with normalized vectors.

        Raises
        ------
        KeyError
            If a required key is not found in the data.
        """
        updates = {}

        for key in self.input_keys:
            if key not in data.keys():
                raise KeyError(f"Input key '{key}' not found in data")

            tensor = data[key]
            norm = torch.norm(tensor, dim=self.dim, keepdim=True)
            normalized = tensor / norm.clamp(min=self.eps)
            updates[key] = normalized

        return data.update(updates)

    def extra_repr(self) -> str:
        """
        Return extra information for repr.

        Returns
        -------
        str
            String with transform parameters.
        """
        return f"input_keys={self.input_keys}, dim={self.dim}, eps={self.eps}"
