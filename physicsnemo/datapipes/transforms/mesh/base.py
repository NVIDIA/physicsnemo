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
Base for mesh transforms (Mesh -> Mesh).

To apply a :class:`MeshTransform` across a ``TensorDict[str, Mesh]``,
use the underlying TensorDict API:
``td.apply(transform, call_on_nested=True)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from physicsnemo.mesh import DomainMesh, Mesh


class MeshTransform(ABC):
    r"""
    Base for transforms that take a Mesh and return a Mesh.

    Use for single-mesh pipelines. To broadcast across a multi-mesh
    container (``TensorDict[str, Mesh]``), use the TensorDict API
    directly: ``td.apply(transform, call_on_nested=True)``.
    """

    def __init__(self) -> None:
        self._device: Optional[torch.device] = None

    @abstractmethod
    def __call__(self, mesh: Mesh) -> Mesh:
        """
        Apply the transform to a mesh.

        Parameters
        ----------
        mesh : Mesh
            Input mesh.

        Returns
        -------
        Mesh
            Transformed mesh.
        """
        raise NotImplementedError

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Apply this transform to a DomainMesh.

        Default: broadcasts ``__call__`` to interior and all boundaries
        via :meth:`DomainMesh.apply_to_meshes`, leaving domain-level
        ``global_data`` unchanged.

        Override in subclasses that need domain-aware behavior (e.g.
        transforms that modify ``global_data``, random augmentations
        that must sample parameters once, or centering transforms).

        Parameters
        ----------
        domain : DomainMesh
            Input domain mesh.

        Returns
        -------
        DomainMesh
            Transformed domain mesh.
        """
        return domain.apply_to_meshes(self)

    @property
    def stochastic(self) -> bool:
        """Whether this transform uses random sampling.

        Returns ``True`` if the instance has a ``_generator`` attribute
        (set by stochastic subclasses such as ``RandomScaleMesh``).
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

    def to(self, device: torch.device | str) -> MeshTransform:
        """Move any internal tensors, generators, and distributions to *device*.

        ``torch.Generator`` objects cannot be moved in-place, so a new
        generator is created on *device* and seeded with
        :meth:`~torch.Generator.initial_seed` from the original.

        ``torch.distributions.Distribution`` objects are reconstructed
        with their parameter tensors moved to *device*, using
        ``arg_constraints`` to discover parameter names generically.

        Parameters
        ----------
        device : torch.device or str
            Target device.

        Returns
        -------
        MeshTransform
            ``self``, for chaining.
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
        """The device that internal tensors and generators reside on.

        Returns ``None`` if :meth:`to` has not been called yet.

        Returns
        -------
        torch.device or None
            Current device, or ``None`` if unset.
        """
        return self._device

    def extra_repr(self) -> str:
        """Return a string of extra information for :meth:`__repr__`.

        Subclasses should override this to include constructor arguments
        or other state that is useful for debugging (e.g.
        ``"scale=0.1, p=0.5"``).  The base implementation returns an
        empty string.

        Returns
        -------
        str
            Extra representation string.
        """
        return ""

    def __repr__(self) -> str:
        """Return a human-readable string representation of the transform.

        The format is ``ClassName(extra_repr())``, mirroring the
        convention used by :class:`torch.nn.Module`.

        Returns
        -------
        str
            String representation.
        """
        return f"{self.__class__.__name__}({self.extra_repr()})"
