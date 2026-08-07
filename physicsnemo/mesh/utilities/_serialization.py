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

"""Serialization helpers for mesh tensorclasses.

TensorDict's memmap writer records every tensor's shape and dtype in
``meta.json``, but it allocates storage with ``torch.from_file(..., size=0)``,
which creates no file at all for a tensor with zero elements. Its loader then
skips any key whose ``.memmap`` file is absent, so zero-element tensors are lost
on load: a mesh's ``cells`` of shape :math:`(0, 3)` comes back as ``None`` (and
is silently replaced with a :math:`(0, 1)` point-cloud sentinel), empty
``point_data`` fields vanish, and an
:class:`~physicsnemo.mesh.neighbors.Adjacency` with no neighbors raises during
reconstruction.

:func:`_load_memmap_with_empty_tensors` closes that gap purely on the read side,
rebuilding the missing tensors from metadata that is already on disk. Nothing
about the written format changes, so files stay byte-identical and remain
readable by PhysicsNeMo releases that predate this module.

Tensorclasses opt in by overriding their generated ``_load_memmap``::

    Adjacency._load_memmap = classmethod(_load_memmap_with_empty_tensors)
"""

import json
import math
import pickle
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict, TensorDictBase

### Upstream provenance
# ``_load_memmap_with_empty_tensors`` is a fork of the private
# ``tensordict.tensorclass._load_memmap``, mirrored from tensordict 0.12.4.
# Re-check it against upstream on every tensordict bump: divergence shows up as
# tensorclass fields that quietly fail to load, not as an import error.
#
# The fork is necessary only because the repair has to happen *between*
# ``TensorDict.load_memmap`` and ``cls._from_tensordict`` -- the latter runs
# ``__post_init__``, which is exactly where a missing tensor turns into
# corruption. There is no upstream hook in that gap.
#
# The durable fix belongs in ``TensorDict._load_memmap`` (``tensordict/_td.py``,
# at the ``if not memmap_file.exists(): continue`` guard), which already holds
# both the shape and the dtype and could allocate the empty tensor there. If
# that lands upstream, this whole module can be deleted.


def _restore_empty_tensors(tensordict: TensorDictBase) -> None:
    """Rebuild the zero-element tensors that TensorDict's memmap writer omitted.

    Walks ``tensordict`` and its sub-collections in place, consulting the
    ``meta.json`` written alongside each one. Any entry the metadata declares
    but the loader skipped is recreated with its recorded shape and dtype.

    Parameters
    ----------
    tensordict : TensorDictBase
        A collection freshly returned by ``load_memmap``, whose
        ``_memmap_prefix`` still points at the directory it was read from.
        Collections that were never memory-mapped are left untouched.

    Returns
    -------
    None
        ``tensordict`` is modified in place.

    Notes
    -----
    Nested (jagged) tensors are deliberately skipped. Their recorded ``shape``
    is the shape *of the shape tensor*, whose values live in a separate
    ``.shape.memmap`` file, so a jagged tensor cannot be rebuilt from
    ``meta.json`` alone. An empty jagged tensor therefore still round-trips as a
    missing key -- a known limitation, not an oversight.

    Sub-collections stored as tensorclasses, rather than as plain
    ``TensorDict`` instances, are skipped here because each dispatches to its
    own ``_load_memmap``. They are repaired only if that class also opts in to
    :func:`_load_memmap_with_empty_tensors`.
    """
    prefix = tensordict._memmap_prefix
    if prefix is None:
        return

    with (prefix / "meta.json").open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    ### Rebuild this level's storage-free leaves.
    empty_tensors: dict[str, torch.Tensor] = {}
    for key, entry in metadata.items():
        # Non-dict values are collection-level metadata ("shape", "device",
        # "_type"); a "type" key marks a sub-collection, handled by recursion.
        if not isinstance(entry, dict) or entry.get("type") is not None:
            continue

        shape, dtype_name = entry.get("shape"), entry.get("dtype")
        if (
            shape is None
            or dtype_name is None
            or entry.get("is_nested", False)  # unrecoverable; see Notes
            or math.prod(shape) != 0  # non-empty, so it has a file and loaded
            or key in tensordict  # already present; never clobber
        ):
            continue

        empty_tensors[key] = torch.empty(
            shape,
            dtype=getattr(torch, dtype_name.removeprefix("torch.")),
            device=tensordict.device,
        )

    if empty_tensors:
        with tensordict.unlock_():
            tensordict.update(empty_tensors)

    ### Descend. Each sub-collection carries the prefix it was read from, which
    # is the only reliable way to locate it on disk: the directory name is a
    # filesystem-encoded form of the key, not the key itself.
    for value in tensordict.values():
        if isinstance(value, TensorDictBase):
            _restore_empty_tensors(value)


def _load_memmap_with_empty_tensors(
    cls,
    prefix: Path,
    metadata: dict[str, Any],
    *,
    robust_key: bool | None,
    **kwargs: Any,
):
    """Load a memory-mapped tensorclass, keeping its zero-element tensors.

    Drop-in replacement for the ``_load_memmap`` classmethod that
    ``@tensorclass`` generates. See the module docstring for why the override
    exists and the provenance note above for its relationship to upstream.

    Parameters
    ----------
    cls : type
        The tensorclass being reconstructed.
    prefix : Path
        Directory holding the tensorclass's ``meta.json``, its ``_tensordict``
        subdirectory, and optionally an ``other.pickle`` sidecar.
    metadata : dict[str, Any]
        Already-parsed contents of ``prefix / "meta.json"``.
    robust_key : bool or None
        Filesystem key-encoding scheme the data was written with; forwarded to
        ``TensorDict.load_memmap`` unchanged.
    **kwargs : Any
        Remaining ``load_memmap`` options (``device``, ``out``), forwarded
        unchanged.

    Returns
    -------
    cls
        The reconstructed tensorclass instance.

    Raises
    ------
    ValueError
        If ``prefix`` has no ``_tensordict`` subdirectory, which means the saved
        directory is incomplete.
    """
    ### Non-tensor fields come from meta.json plus the optional pickle sidecar.
    non_tensordict = dict(metadata)
    non_tensordict.pop("_type", None)

    other_metadata_path = prefix / "other.pickle"
    if other_metadata_path.exists():
        with other_metadata_path.open("rb") as other_metadata_file:
            non_tensordict.update(pickle.load(other_metadata_file))  # noqa: S301

    tensordict_prefix = prefix / "_tensordict"
    if not tensordict_prefix.exists():
        # Upstream tolerates this only for NonTensorData subclasses, which never
        # route through here, so for a mesh tensorclass it is always corruption.
        raise ValueError(
            f"The _tensordict directory seems to be missing: {str(tensordict_prefix)!r}."
        )

    tensordict = TensorDict.load_memmap(
        tensordict_prefix,
        **kwargs,
        non_blocking=False,
        robust_key=robust_key,
    )
    # Must run before ``_from_tensordict``, which invokes ``__post_init__``:
    # that is where a missing ``cells`` or ``indices`` becomes corruption.
    _restore_empty_tensors(tensordict)
    return cls._from_tensordict(tensordict, non_tensordict)


__all__ = ["_load_memmap_with_empty_tensors", "_restore_empty_tensors"]
