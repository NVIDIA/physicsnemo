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
Reader extension that merges an external on-disk ``TensorDict`` into each
sample's ``global_data`` at load time.

This recipe's surface configs read a boundary :class:`Mesh` directly out
of a parent :class:`~physicsnemo.mesh.DomainMesh`'s on-disk tensordict
tree (e.g. ``.../*.pdmsh/_tensordict/boundaries/vehicle``).  The boundary
tensordict's own ``global_data`` carries little or nothing of the
case-level metadata -- the freestream conditions (``U_inf``,
``rho_inf``, ``p_inf``, ``T_inf``, ``L_ref``) live on the *domain-level*
``_tensordict/global_data`` instead.  :class:`MeshReaderWithGlobalData`
walks a configurable relative path from each sample directory, loads
the tensordict it points at, and merges its keys into the boundary
``Mesh``'s ``global_data`` so downstream transforms
(:class:`NonDimensionalizeByMetadata`, ``RandomRotateMesh`` with
``transform_global_data: true``, ...) can read them straight off the
loaded sample.

Recipe-local module registered into the global datapipe component
registry so the class can be referenced via
``${dp:MeshReaderWithGlobalData}`` in Hydra YAML configs.  Import this
module before Hydra instantiation.
"""

from __future__ import annotations

from typing import Any

from tensordict import TensorDict

from physicsnemo.datapipes.readers.mesh import MeshReader
from physicsnemo.datapipes.registry import register
from physicsnemo.mesh import Mesh


@register()
class MeshReaderWithGlobalData(MeshReader):
    r"""MeshReader that merges an external tensordict into each Mesh's ``global_data``.

    Identical to :class:`MeshReader` except for the
    ``merge_global_data_from`` parameter, which names a path
    *relative to each matched sample* pointing to a saved
    ``TensorDict`` directory whose keys should be merged into the
    loaded :class:`Mesh`'s ``global_data``.  If a key appears in
    both the boundary's ``global_data`` and the external one, a
    :class:`ValueError` is raised: ``global_data`` is case-level
    by construction, so overlap is ambiguous and treated as a
    data-layer bug rather than silently resolved.

    This is intended for boundary tensordicts whose own
    ``global_data`` is empty (or only carries dataset-local fields
    like ``TimeValue``) but whose parent
    :class:`~physicsnemo.mesh.DomainMesh` carries the case-level
    freestream conditions (``U_inf``, ``rho_inf``, ``p_inf``, ...).
    Without this hook the boundary-only surface pipeline cannot see
    those parent fields.

    Example YAML::

        reader:
          _target_: ${dp:MeshReaderWithGlobalData}
          path: ${train_datadir}
          pattern: "**/*.pdmsh/_tensordict/boundaries/vehicle"
          # Walk up to the parent DomainMesh's global_data tensordict
          merge_global_data_from: "../../global_data"

    Parameters
    ----------
    merge_global_data_from : str or None
        Path *relative to each matched sample directory* of an
        on-disk ``TensorDict`` directory (one containing
        ``meta.json`` plus ``*.memmap`` files).  When ``None``
        (default) this class behaves exactly like
        :class:`MeshReader`.
    **kwargs
        Forwarded to :class:`MeshReader`.
    """

    def __init__(
        self,
        *args: Any,
        merge_global_data_from: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._merge_rel_path = merge_global_data_from

    def _load_sample(self, index: int) -> Mesh:
        mesh = super()._load_sample(index)
        if self._merge_rel_path is None:
            return mesh

        sample_path = self._paths[index]
        ### Resolve ``../...`` against the sample directory itself; the
        ### result must already exist on disk (we don't lazily create).
        ext_path = (sample_path / self._merge_rel_path).resolve()
        if not ext_path.exists():
            raise FileNotFoundError(
                f"merge_global_data_from path not found: {ext_path} "
                f"(resolved from sample {sample_path} + "
                f"{self._merge_rel_path!r})"
            )

        ext_td = TensorDict.load_memmap(ext_path)
        merged = mesh.global_data.clone()
        collisions = sorted(set(ext_td.keys()) & set(merged.keys()))
        if collisions:
            raise ValueError(
                f"global_data key collision while merging {ext_path} "
                f"into sample {sample_path}: keys {collisions} are "
                f"present on both the boundary tensordict and the "
                f"external one. global_data is case-level by "
                f"definition, so an overlapping key is ambiguous and "
                f"indicates inconsistent metadata at the data layer."
            )
        merged.update(ext_td)

        return Mesh(
            points=mesh.points,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            global_data=merged,
        )
