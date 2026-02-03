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
Compose - Chain multiple transforms into a single transform.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence

import torch
from tensordict import TensorDict

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.base import Transform


@register()
class Compose(Transform):
    """
    Compose multiple transforms into a sequential pipeline.

    Applies transforms in order, passing the output of each as input to the next.

    Parameters
    ----------
    transforms : Sequence[Transform]
        Sequence of transforms to apply in order.

    Examples
    --------
    >>> from physicsnemo.datapipes.transforms import Normalize, SubsamplePoints
    >>> from tensordict import TensorDict
    >>> sample = TensorDict({
    ...     "pressure": torch.tensor([101325.0, 102325.0, 100325.0]),
    ... })
    >>> normalize = Normalize(input_keys=["pressure"], method="mean_std", means={"pressure": 101325.0}, stds={"pressure": 1000.0})
    >>> subsample = SubsamplePoints(input_keys=["pressure"], n_points=1000)
    >>> pipeline = Compose([normalize, subsample])
    >>> transformed = pipeline(sample)
    >>> transformed["pressure"]
    tensor([ 0.,  1., -1.])
    """

    def __init__(self, transforms: Sequence[Transform]) -> None:
        """
        Initialize the composition.

        Parameters
        ----------
        transforms : Sequence[Transform]
            Sequence of transforms to apply in order.

        Raises
        ------
        TypeError
            If any element is not a Transform.
        ValueError
            If transforms is empty.
        """
        super().__init__()

        if not transforms:
            raise ValueError("transforms cannot be empty")

        for i, t in enumerate(transforms):
            if not isinstance(t, Transform):
                raise TypeError(
                    f"All elements must be Transform instances, "
                    f"got {type(t).__name__} at index {i}"
                )

        self.transforms: list[Transform] = list(transforms)

    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Apply all transforms in sequence.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict to transform.

        Returns
        -------
        TensorDict
            Transformed TensorDict after applying all transforms.
        """
        for transform in self.transforms:
            data = transform(data)
        return data

    def to(self, device: torch.device | str) -> Compose:
        """
        Move all transforms to the specified device.

        Parameters
        ----------
        device : torch.device or str
            Target device.

        Returns
        -------
        Compose
            Self for chaining.
        """
        super().to(device)
        for transform in self.transforms:
            transform.to(device)
        return self

    def __getitem__(self, index: int) -> Transform:
        """
        Get a transform by index.

        Parameters
        ----------
        index : int
            Index of the transform to retrieve.

        Returns
        -------
        Transform
            The transform at the specified index.
        """
        return self.transforms[index]

    def __len__(self) -> int:
        """
        Return number of transforms.

        Returns
        -------
        int
            Number of transforms in the composition.
        """
        return len(self.transforms)

    def __iter__(self) -> Iterator[Transform]:
        """
        Iterate over transforms.

        Yields
        ------
        Transform
            Each transform in the composition.
        """
        return iter(self.transforms)

    def append(self, transform: Transform) -> None:
        """
        Append a transform to the pipeline.

        Parameters
        ----------
        transform : Transform
            Transform to append.

        Raises
        ------
        TypeError
            If transform is not a Transform instance.
        """
        if not isinstance(transform, Transform):
            raise TypeError(f"Expected Transform, got {type(transform).__name__}")
        self.transforms.append(transform)

    def state_dict(self) -> dict[str, Any]:
        """
        Return state of all transforms.

        Returns
        -------
        dict[str, Any]
            Dictionary containing transform states and types.
        """
        return {
            "transforms": [t.state_dict() for t in self.transforms],
            "transform_types": [type(t).__name__ for t in self.transforms],
        }

    def extra_repr(self) -> str:
        """
        Return extra information for repr.

        Returns
        -------
        str
            Formatted string showing all transforms.
        """
        lines = []
        for i, t in enumerate(self.transforms):
            lines.append(f"  ({i}): {t}")
        return "\n" + "\n".join(lines) + "\n"
