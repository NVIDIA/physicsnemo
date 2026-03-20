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
Base for mesh transforms: Mesh -> Mesh and TensorDict[str, Mesh] -> TensorDict[str, Mesh].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from tensordict import TensorDict

from physicsnemo.mesh import DomainMesh, Mesh


def apply_to_tensordict_mesh(
    data: TensorDict,
    transform: MeshTransform,
) -> TensorDict:
    """Apply a Mesh -> Mesh transform to each value in a TensorDict of Mesh.

    Parameters
    ----------
    data : TensorDict
        TensorDict whose values are Mesh instances.
    transform : MeshTransform
        Transform instance; called on each mesh.

    Returns
    -------
    TensorDict
        New TensorDict with transformed meshes (same keys).
    """
    out = {k: transform(v) for k, v in data.items()}
    return TensorDict(out, batch_size=[])


class MeshTransform(ABC):
    r"""
    Base for transforms that take a Mesh and return a Mesh.

    Use for single-mesh pipelines. For multi-mesh (TensorDict[str, Mesh]),
    apply the same transform to each value or use apply_to_tensordict_mesh.
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
        via :meth:`DomainMesh._map_meshes`, leaving domain-level
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
        return domain._map_meshes(self)

    def to(self, device: torch.device | str) -> MeshTransform:
        """Move any internal tensors to the specified device."""
        self._device = torch.device(device) if isinstance(device, str) else device
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(self._device))
        return self

    @property
    def device(self) -> torch.device | None:
        return self._device

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"
