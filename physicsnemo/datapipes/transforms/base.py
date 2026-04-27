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
Transform base class - The foundation for all data transformations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from tensordict import TensorDict


class Transform(ABC):
    """
    Abstract base class for all transforms.

    Transforms operate on a TensorDict and return a modified TensorDict.
    They are designed to run on GPU tensors for maximum performance.
    Metadata is not passed to transforms (handled separately by Dataset/DataLoader).

    Subclasses must implement:

    - ``__call__(data: TensorDict) -> TensorDict``

    Optionally override:

    - ``extra_repr() -> str``: For custom repr output
    - ``state_dict() -> dict``: For serialization
    - ``load_state_dict(state_dict: dict)``: For deserialization

    Examples
    --------
    >>> class MyTransform(Transform):
    ...     def __init__(self, scale: float):
    ...         super().__init__()
    ...         self.scale = scale
    ...
    ...     def __call__(self, data: TensorDict) -> TensorDict:
    ...         # Apply transformation to all tensors
    ...         return data.apply(lambda x: x * self.scale)
    """

    def __init__(self) -> None:
        """Initialize the transform."""
        self._device: Optional[torch.device] = None

    @abstractmethod
    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Apply the transform to a TensorDict.

        Parameters
        ----------
        data : TensorDict
            Input TensorDict to transform.

        Returns
        -------
        TensorDict
            Transformed TensorDict.
        """
        raise NotImplementedError

    @property
    def stochastic(self) -> bool:
        """Whether this transform uses random sampling.

        Returns ``True`` if the instance has a ``_generator`` attribute
        (set by stochastic subclasses such as ``SubsamplePoints``).
        Deterministic transforms return ``False``.
        """
        return hasattr(self, "_generator")

    def set_generator(self, generator: torch.Generator) -> None:
        """Assign a ``torch.Generator`` for reproducible random sampling.

        Only takes effect on stochastic transforms (those that declare
        ``self._generator``).  Deterministic transforms silently ignore
        the call.

        Parameters
        ----------
        generator : torch.Generator
            Generator to use for all subsequent random draws.
        """
        if self.stochastic:
            self._generator = generator

    def set_epoch(self, epoch: int) -> None:
        """Reseed the generator for a new epoch.

        Reseeds ``self._generator`` with ``initial_seed() + epoch`` so
        each epoch produces a different but deterministic random
        sequence.  No-op for deterministic transforms or when no
        generator has been assigned.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        if self.stochastic and self._generator is not None:
            self._generator.manual_seed(self._generator.initial_seed() + epoch)

    def to(self, device: torch.device | str) -> Transform:
        """Move any internal tensors, generators, and distributions to *device*.

        ``torch.Generator`` objects cannot be moved in-place, so a new
        generator is created on *device* and seeded with
        :meth:`~torch.Generator.initial_seed` from the original.

        ``torch.distributions.Distribution`` objects are reconstructed
        with their parameter tensors moved to *device*, using
        ``arg_constraints`` to discover parameter names generically.

        Override this method if your transform requires custom device
        handling.

        Parameters
        ----------
        device : torch.device or str
            Target device.

        Returns
        -------
        Transform
            Self for chaining.
        """
        self._device = torch.device(device) if isinstance(device, str) else device
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(self._device))
            elif isinstance(value, torch.Generator):
                new_gen = torch.Generator(device=self._device)
                new_gen.manual_seed(value.initial_seed())
                setattr(self, name, new_gen)
            elif isinstance(value, torch.distributions.Distribution):
                dist_cls = type(value)
                kwargs = {}
                # Access arg_constraints on the instance (not the class)
                # because the base Distribution defines it as a @property.
                for param_name in value.arg_constraints:
                    p = getattr(value, param_name)
                    kwargs[param_name] = (
                        p.to(self._device) if isinstance(p, torch.Tensor) else p
                    )
                setattr(self, name, dist_cls(**kwargs, validate_args=False))
        return self

    @property
    def device(self) -> torch.device | None:
        """
        The device this transform operates on.

        Returns
        -------
        torch.device or None
            The device, or None if not set.
        """
        return self._device

    def extra_repr(self) -> str:
        """
        Return extra information for repr.

        Override this to add transform-specific info to the repr.

        Returns
        -------
        str
            Extra representation string.
        """
        return ""

    def state_dict(self) -> dict[str, Any]:
        """
        Return a dictionary containing the transform's state.

        Override this for transforms with learnable or configurable state.

        Returns
        -------
        dict[str, Any]
            State dictionary.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load state from a state dictionary.

        Override this to restore transform state.

        Parameters
        ----------
        state_dict : dict[str, Any]
            State dictionary to load from.
        """
        pass

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation of the transform.
        """
        return f"{self.__class__.__name__}({self.extra_repr()})"
