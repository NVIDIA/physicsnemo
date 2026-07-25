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

r"""Backward-compatible reads of decorator-era ``.pmsh`` / ``.pdmsh`` files.

Through PhysicsNeMo 2.1.x, :class:`~physicsnemo.mesh.Mesh` and
:class:`~physicsnemo.mesh.DomainMesh` were built by the ``@tensorclass``
decorator, whose memmap writer nested the payload one directory down::

    square.pdmsh/
        meta.json          <- {"_type": "...DomainMesh"}
        _tensordict/       <- the actual fields

Inheriting from ``TensorClass`` instead writes those fields at the root of the
directory, so :func:`install_legacy_memmap_reader` teaches a container to
recognize the old nesting and read it explicitly. Files written by the current
code need no special handling; they load through ``tensordict``'s own
machinery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict, TensorDictBase

### Directory that the decorator-era writer nested the payload under.
LEGACY_PAYLOAD_DIRNAME = "_tensordict"


def _legacy_payload_dir(prefix: str | Path) -> Path | None:
    """Return the decorator-era payload directory, or ``None`` if not one."""
    payload = Path(prefix) / LEGACY_PAYLOAD_DIRNAME
    return payload if payload.is_dir() else None


def install_legacy_memmap_reader(cls: type) -> None:
    """Teach a ``TensorClass`` container to load the decorator-era layout.

    Overrides ``_load_memmap`` on ``cls`` -- the hook ``tensordict`` dispatches
    to once it has read a directory's ``meta.json`` and resolved which class
    wrote it -- so that a nested :data:`LEGACY_PAYLOAD_DIRNAME` directory is
    read explicitly, falling through to the stock implementation otherwise.

    Parameters
    ----------
    cls : type
        The ``TensorClass`` subclass to patch, in place.
    """
    ### Captured before the override is installed, so it stays reachable. This
    ### is a plain function rather than a bound method: `tensorclass` installs
    ### it on the class and it resolves the class itself.
    stock_load_memmap = cls._load_memmap

    def _load_memmap(
        cls,
        prefix: str | Path,
        metadata: dict,
        device: torch.device | None = None,
        out: TensorDictBase | None = None,
        **kwargs: Any,
    ) -> Any:
        """Reconstruct one container from either on-disk layout."""
        legacy_payload = _legacy_payload_dir(prefix)
        if legacy_payload is None:
            return stock_load_memmap(prefix, metadata, device=device, out=out, **kwargs)
        return cls._from_tensordict(
            TensorDict.load_memmap(legacy_payload, device=device, out=out, **kwargs)
        )

    cls._load_memmap = classmethod(_load_memmap)
