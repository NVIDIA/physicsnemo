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

"""Zarr I/O for :class:`~physicsnemo.mesh.Mesh` and
:class:`~physicsnemo.mesh.DomainMesh`, built on tensordict's zarr backend.

Serialization is delegated to tensordict (``to_zarr`` / ``from_zarr``;
requires a tensordict release with the zarr storage backend and
``zarr >= 3``). This module adds only the two things tensordict cannot know:

- **Layout policy**: chunks aligned to training-subsample sizes plus zstd
  compression. tensordict's default (one uncompressed chunk per leaf) is a
  checkpoint layout -- reading a training subsample from it decodes entire
  arrays.
- **Type reconstruction**: ``tensordict.from_zarr`` returns a plain
  ``PersistentTensorDict``; a root attr records whether the store holds a
  ``Mesh`` or a ``DomainMesh`` so :func:`from_zarr` can rebuild it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.mesh.domain_mesh import DomainMesh
from physicsnemo.mesh.mesh import Mesh

### Lazy optional import: no zarr import happens at module load; first
### attribute access imports it (or raises with an install hint).
if TYPE_CHECKING:
    import zarr
else:
    zarr = OptionalImport("zarr")

__all__ = ["to_zarr", "from_zarr"]

_TYPE_ATTR = "physicsnemo_mesh_type"

#: Default chunk length (rows) for every array's leading dimension. Matches
#: the common training ``subsample_n_cells`` so one draw touches ~1 chunk.
DEFAULT_CHUNK_ROWS = 200_000


def _require_backends():
    """Verify zarr >= 3 and that tensordict carries the zarr storage backend."""
    if int(zarr.__version__.split(".")[0]) < 3:
        raise ImportError(
            f"zarr >= 3 required for the tensordict zarr backend, "
            f"found {zarr.__version__}."
        )
    from tensordict import TensorDict

    if not hasattr(TensorDict, "from_zarr"):
        import tensordict

        raise ImportError(
            "This tensordict version has no zarr storage backend "
            f"(found {tensordict.__version__}; need the release containing "
            "pytorch/tensordict#1754)."
        )


def _propagate_nested_create_kwargs():
    """Work around pytorch/tensordict#1758: ``to_zarr`` create-kwargs are
    dropped for nested leaves (``PersistentTensorDict._make_nested`` does not
    propagate ``self.kwargs``), so ``chunks``/``compressors`` only apply at
    the root. All mesh data is nested. Remove once fixed upstream.
    """
    from tensordict.persistent import PersistentTensorDict

    if getattr(PersistentTensorDict, "_pn_nested_kwargs_patch", False):
        return
    orig = PersistentTensorDict._make_nested

    def _make_nested(self, key, group):
        out = orig(self, key, group)
        out.kwargs = self.kwargs
        return out

    PersistentTensorDict._make_nested = _make_nested
    PersistentTensorDict._pn_nested_kwargs_patch = True


def to_zarr(
    obj: Mesh | DomainMesh,
    store: str | Path,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    zstd_level: int = 3,
) -> None:
    """Save a Mesh or DomainMesh to a zarr store via tensordict's backend.

    Parameters
    ----------
    obj : Mesh or DomainMesh
        Mesh to serialize.
    store : str or Path
        Path of the zarr store directory to create.
    chunk_rows : int
        Chunk length along each array's leading dimension. Align with the
        training subsample size so one draw reads ~1 chunk.
    zstd_level : int
        zstd compression level (0 disables compression).

    Examples
    --------
    >>> from physicsnemo.mesh.io import to_zarr, from_zarr  # doctest: +SKIP
    >>> to_zarr(domain_mesh, "run_1.zarr")  # doctest: +SKIP
    >>> reloaded = from_zarr("run_1.zarr")  # doctest: +SKIP
    """
    _require_backends()
    _propagate_nested_create_kwargs()
    if not isinstance(obj, (Mesh, DomainMesh)):
        raise TypeError(f"Expected Mesh or DomainMesh, got {type(obj).__name__}")

    kwargs = {"chunks": (chunk_rows,)}
    if zstd_level > 0:
        kwargs["compressors"] = [zarr.codecs.ZstdCodec(level=zstd_level)]
    # Serialize the underlying TensorDict: the tensorclass `to_zarr` proxy
    # tries to re-wrap the returned PersistentTensorDict as a Mesh/DomainMesh,
    # which their __post_init__ validation rejects.
    obj.to_tensordict().to_zarr(str(store), **kwargs)

    root = zarr.open_group(str(store), mode="a")
    root.attrs[_TYPE_ATTR] = type(obj).__name__


def _open_group(store) -> "zarr.Group":
    """Open a zarr group read-only (verifying backend availability)."""
    _require_backends()
    return zarr.open_group(str(store), mode="r")


def _read_full(arr, device=None) -> torch.Tensor:
    """Read a whole zarr array as a tensor."""
    return torch.as_tensor(arr[...], device=device)


def _read_rows(arr, runs: list[tuple[int, int]], device=None) -> torch.Tensor:
    """Read one or more contiguous leading-dimension row runs and concatenate.

    Only the chunks intersecting the runs are fetched/decoded.
    """
    parts = [torch.as_tensor(arr[s:e], device=device) for s, e in runs]
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)


def _read_index(arr, index, device=None) -> torch.Tensor:
    """Gather rows of a zarr array by (sorted) integer index."""
    import numpy as np

    idx = np.asarray(index)
    return torch.as_tensor(arr[idx], device=device)


def _read_tree(group: "zarr.Group", name: str, device=None, leaf_reader=None):
    """Recursively read a field subtree (e.g. ``point_data``) as a TensorDict.

    ``leaf_reader(arr) -> Tensor`` selects what to read per array; default is
    the full array. Nested TensorDicts round-trip as nested groups.
    """
    from tensordict import TensorDict

    if name not in group:
        return TensorDict({}, batch_size=[])
    reader = leaf_reader or (lambda a: _read_full(a, device))

    def _walk(grp):
        out = {}
        for key, arr in grp.arrays():
            out[key] = reader(arr)
        for key, sub in grp.groups():
            out[key] = _walk(sub)
        return out

    return TensorDict(_walk(group[name]), batch_size=[])


def _mesh_from_group(group: "zarr.Group", device) -> Mesh:
    """Materialize one mesh group as an in-memory Mesh."""
    # The memmap format does not persist 0-element tensors but zarr does;
    # map an empty cells array back to "no cells" for a clean point cloud.
    cells = None
    if "cells" in group and group["cells"].shape[0] > 0:
        cells = _read_full(group["cells"], device)
    return Mesh(
        points=_read_full(group["points"], device),
        cells=cells,
        point_data=_read_tree(group, "point_data", device),
        cell_data=_read_tree(group, "cell_data", device),
        global_data=_read_tree(group, "global_data", device),
    )


def from_zarr(
    store: str | Path,
    *,
    device: torch.device | str | None = None,
) -> Mesh | DomainMesh:
    """Load a Mesh or DomainMesh from a zarr store written by :func:`to_zarr`.

    Data is materialized eagerly (one read per array).

    Parameters
    ----------
    store : str or Path
        Path of the zarr store directory.
    device : torch.device, str, or None
        Device for the returned tensors (default CPU).
    """
    root = _open_group(store)
    mesh_type = root.attrs.get(_TYPE_ATTR)
    if mesh_type is None and "points" in root:
        # A mesh subgroup inside a DomainMesh store (e.g.
        # <store>.zarr/boundaries/<name>) carries no type attr but is a
        # complete Mesh group; allow loading it directly.
        mesh_type = "Mesh"
    if mesh_type == "Mesh":
        return _mesh_from_group(root, device)
    if mesh_type == "DomainMesh":
        global_data = _read_tree(root, "global_data", device)
        return DomainMesh(
            interior=_mesh_from_group(root["interior"], device),
            boundaries={
                name: _mesh_from_group(grp, device)
                for name, grp in root["boundaries"].groups()
            },
            global_data=global_data,
        )
    raise ValueError(
        f"{store} was not written by physicsnemo.mesh.io.to_zarr "
        f"(root attr {_TYPE_ATTR}={mesh_type!r})."
    )
