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

With ``domain_boundaries`` the reader also loads the other boundaries of
the flow domain that a case file stores beside the surface (DrivAerML's
tunnel: ``inlet``, ``outlet``, ``slip``, ``no_slip``), for a model that
reads the whole boundary while it predicts on the surface.
:class:`SurfaceMeshToDomainMesh` is that sample's terminal step.

Recipe-local module registered into the global datapipe component
registry so the classes can be referenced via
``${dp:MeshReaderWithGlobalData}`` in Hydra YAML configs.  Import this
module before Hydra instantiation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tensordict import TensorDict

from physicsnemo.datapipes.keys import key_to_str, leaf_keys
from physicsnemo.datapipes.readers.mesh import MeshReader
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh import MeshToDomainMesh
from physicsnemo.mesh import DomainMesh, Mesh


@register()
class MeshReaderWithGlobalData(MeshReader):
    r"""MeshReader that merges an external tensordict into each Mesh's ``global_data``.

    Like :class:`MeshReader`, with two more parameters. With
    ``domain_boundaries`` it returns a :class:`~physicsnemo.mesh.DomainMesh`
    instead of a :class:`Mesh` (see below). ``merge_global_data_from`` names
    a path *relative to each matched sample* pointing to a saved
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
          pattern: "run_*/*.pdmsh/_tensordict/boundaries/vehicle"
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
    domain_boundaries : sequence of str, default ()
        Names of other boundaries stored beside the matched surface, as
        siblings of its directory (``.../boundaries/<name>``). Each is
        loaded whole. The chain's ``SubsampleMesh`` then caps each at its
        ``n_cells``, like the surface. With any name, the sample is a
        :class:`~physicsnemo.mesh.DomainMesh`. Its interior is the surface,
        its boundaries are these meshes, and its ``global_data`` is the
        case's. Every transform of the chain (rotation, translation,
        centering, scaling) then moves them with the surface. End such a
        chain with :class:`SurfaceMeshToDomainMesh`. When empty, the sample
        is a ``Mesh``, as before.
    **kwargs
        Forwarded to :class:`MeshReader`.

    Raises
    ------
    ValueError
        If ``domain_boundaries`` names a boundary twice, or is set with
        domain parallelism. At load time, if a ``global_data`` key is both
        on the sample and in the ``merge_global_data_from`` tensordict.
    FileNotFoundError
        At load time, if the ``merge_global_data_from`` path or a boundary
        of ``domain_boundaries`` is missing.
    """

    def __init__(
        self,
        *args: Any,
        merge_global_data_from: str | None = None,
        domain_boundaries: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._merge_rel_path = merge_global_data_from
        self._domain_boundaries = [str(name) for name in domain_boundaries]
        if len(set(self._domain_boundaries)) != len(self._domain_boundaries):
            raise ValueError(
                f"domain_boundaries names a boundary twice: {self._domain_boundaries!r}"
            )
        if self._domain_boundaries and self._domain_parallel is not None:
            raise ValueError(
                "MeshReaderWithGlobalData: domain_boundaries is not supported with "
                "domain parallelism (the reader then returns a sharded sample, not "
                "a Mesh to put the boundaries beside): set "
                "domain_parallelism.domain_size=1"
            )

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

        ext_td = TensorDict.load_memmap(ext_path, mode="r")  # copy-on-write
        merged = mesh.global_data.clone()

        ### Compare leaf key *paths* (nested included): two groups sharing a
        ### name but holding different leaves are not a collision (``update``
        ### merges them recursively), but a leaf on one side whose path is a
        ### prefix of, or equal to, a leaf path on the other would be
        ### overwritten by ``update``, so that is.
        def _paths(td: TensorDict) -> list[tuple[str, ...]]:
            return [(k,) if isinstance(k, str) else k for k in leaf_keys(td)]

        ext_paths, merged_paths = _paths(ext_td), _paths(merged)
        collisions = sorted(
            {
                key_to_str(a if len(a) <= len(b) else b)
                for a in ext_paths
                for b in merged_paths
                if a[: len(b)] == b or b[: len(a)] == a
            }
        )
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

        return mesh.with_data(global_data=merged)

    def _load_domain_boundary(self, index: int, name: str) -> Mesh:
        path = self._paths[index].parent / name
        if not path.exists():
            raise FileNotFoundError(
                f"domain boundary {name!r} not found beside {self._paths[index]} "
                f"(looked for {path})"
            )
        return Mesh.load(path, mode="r")  # copy-on-write, like the surface

    def __getitem__(self, index: int) -> tuple[Mesh | DomainMesh, dict[str, Any]]:
        mesh, metadata = super().__getitem__(index)
        if not self._domain_boundaries:
            return mesh, metadata
        boundaries = {
            name: self._load_domain_boundary(index, name)
            for name in self._domain_boundaries
        }
        if self.pin_memory:
            boundaries = {name: b.pin_memory() for name, b in boundaries.items()}
        domain = DomainMesh(
            interior=mesh, boundaries=boundaries, global_data=mesh.global_data
        )
        metadata["boundary_names"] = domain.boundary_names
        return domain, metadata


@register()
class SurfaceMeshToDomainMesh(MeshToDomainMesh):
    r""":class:`MeshToDomainMesh` for a surface read with other boundaries beside it.

    On a bare :class:`Mesh` it is :class:`MeshToDomainMesh`. On the
    :class:`DomainMesh` that :class:`MeshReaderWithGlobalData` returns with
    ``domain_boundaries``, whose interior is the surface, it applies
    :class:`MeshToDomainMesh` to the surface (the interior becomes the points
    that ``interior_points`` selects, with the targets, and the surface the
    boundary ``boundary_name``) and keeps the other boundaries and the
    domain-level ``global_data``.

    Raises
    ------
    ValueError
        If the :class:`DomainMesh` already has a boundary named
        ``boundary_name``.
    """

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:  # type: ignore[override]
        """Convert the surface in ``domain.interior`` and keep the other boundaries."""
        if self._boundary_name in domain.boundary_names:
            raise ValueError(
                f"SurfaceMeshToDomainMesh: the domain already has a boundary "
                f"{self._boundary_name!r}, the name the surface takes"
            )
        surface = self(domain.interior)
        return DomainMesh(
            interior=surface.interior,
            boundaries={
                self._boundary_name: surface.boundaries[self._boundary_name],
                **{name: domain.boundaries[name] for name in domain.boundary_names},
            },
            global_data=domain.global_data,
        )
