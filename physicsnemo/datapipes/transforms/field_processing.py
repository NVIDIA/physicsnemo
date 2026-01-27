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
Field processing transforms for feature engineering.

Provides transforms for broadcasting global features to local points.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.base import Transform


@register()
class BroadcastGlobalFeatures(Transform):
    r"""
    Broadcast global scalar/vector features to all spatial points.

    Replicates global parameters (e.g., density, velocity) to match the number
    of spatial points, enabling concatenation with local features.

    Parameters
    ----------
    input_keys : list[str]
        List of global feature keys to broadcast.
    n_points_key : str
        Key of a tensor whose first dimension gives the number of points
        to broadcast to.
    output_key : str
        Key to store the broadcasted features.

    Examples
    --------
    >>> transform = BroadcastGlobalFeatures(
    ...     input_keys=["air_density", "stream_velocity"],
    ...     n_points_key="embeddings",
    ...     output_key="fx"
    ... )
    >>> data = TensorDict({
    ...     "air_density": torch.tensor(1.225),
    ...     "stream_velocity": torch.tensor(30.0),
    ...     "embeddings": torch.randn(10000, 7)
    ... })
    >>> result = transform(data)
    >>> print(result["fx"].shape)
    torch.Size([10000, 2])
    """

    def __init__(
        self,
        input_keys: list[str],
        n_points_key: str,
        output_key: str,
    ) -> None:
        """
        Initialize the broadcast transform.

        Parameters
        ----------
        input_keys : list[str]
            List of global feature keys to broadcast.
        n_points_key : str
            Key of a tensor whose first dimension gives the number of points
            to broadcast to.
        output_key : str
            Key to store the broadcasted features.
        """
        super().__init__()
        self.input_keys = input_keys
        self.n_points_key = n_points_key
        self.output_key = output_key

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Broadcast global features to match spatial dimensions.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict containing global features and reference tensor.

        Returns
        -------
        TensorDict
            TensorDict with broadcasted features added.

        Raises
        ------
        KeyError
            If required keys are not found in the TensorDict.
        """
        if self.n_points_key not in data.keys():
            raise KeyError(f"Reference key '{self.n_points_key}' not found")

        n_points = data[self.n_points_key].shape[0]

        # Collect features
        features = []
        for key in self.input_keys:
            if key not in data.keys():
                raise KeyError(f"Feature key '{key}' not found")

            feature = data[key]

            # Ensure scalar features are expanded
            if feature.ndim == 0:
                feature = feature.unsqueeze(0)

            features.append(feature)

        # Stack features
        fx = torch.stack(features, dim=-1)

        # Broadcast to match number of points
        fx = fx.broadcast_to(n_points, fx.shape[-1])

        return data.update({self.output_key: fx})

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation of the transform.
        """
        return (
            f"BroadcastGlobalFeatures(input_keys={self.input_keys}, "
            f"output_key={self.output_key})"
        )
