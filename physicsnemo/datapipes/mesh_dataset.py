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
MeshDataset - Combines a mesh reader (MeshReader or MultiMeshReader) with mesh transforms.

Returns (Mesh, metadata) or (TensorDict[str, Mesh], metadata). No key-based filtering.
"""

from __future__ import annotations

from typing import Any, Sequence, Union

import torch
from tensordict import TensorDict

from physicsnemo.datapipes.protocols import DatasetBase
from physicsnemo.datapipes.readers.mesh import MeshReader, MultiMeshReader
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh.base import (
    MeshTransform,
    apply_to_tensordict_mesh,
)
from physicsnemo.mesh import Mesh


def _is_tensordict_mesh(data: Union[Mesh, TensorDict]) -> bool:
    """Return True if data is a TensorDict (of Mesh)."""
    return isinstance(data, TensorDict)


@register()
class MeshDataset(DatasetBase):
    r"""
    Dataset for mesh readers and mesh-only transforms.

    Accepts MeshReader (single-mesh) or MultiMeshReader (multi-mesh).
    Applies a sequence of MeshTransform. Single-mesh: each transform is
    Mesh -> Mesh. Multi-mesh: each transform is applied to every value
    in the TensorDict (TensorDict[str, Mesh] -> TensorDict[str, Mesh]).

    Inherits thread-based prefetching from :class:`DatasetBase`.
    """

    def __init__(
        self,
        reader: MeshReader | MultiMeshReader,
        *,
        transforms: Sequence[MeshTransform] | None = None,
        device: str | torch.device | None = None,
        num_workers: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        reader : MeshReader or MultiMeshReader
            Mesh reader; returns (Mesh, metadata) or (TensorDict[str, Mesh], metadata).
        transforms : sequence of MeshTransform, optional
            Transforms to apply in order. None means no transforms.
        device : str or torch.device, optional
            If set, move mesh data to this device after loading (before transforms).
        num_workers : int, default=2
            Number of worker threads for prefetching.
        """
        super().__init__(num_workers=num_workers)
        self.reader = reader
        self.transforms = list(transforms) if transforms else []
        self._device = torch.device(device) if isinstance(device, str) else device

    def _load(self, index: int) -> tuple[Mesh | TensorDict, dict[str, Any]]:
        data, metadata = self.reader[index]

        if self._device is not None:
            if _is_tensordict_mesh(data):
                data = TensorDict(
                    {k: v.to(self._device) for k, v in data.items()},
                    batch_size=[],
                )
            else:
                data = data.to(self._device)

        for t in self.transforms:
            if _is_tensordict_mesh(data):
                data = apply_to_tensordict_mesh(data, t)
            else:
                data = t(data)

        return data, metadata

    def __len__(self) -> int:
        return len(self.reader)
