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
from typing import Any, cast

import torch
from tensordict import TensorDict, TensorDictBase


def _legacy_payload_dir(prefix: str | Path) -> Path | None:
    """Return the decorator-era payload directory, or ``None`` if not one."""
    payload = Path(prefix) / "_tensordict"
    return payload if payload.is_dir() else None


def install_legacy_memmap_reader(cls: type) -> None:
    """Teach a ``TensorClass`` container to load the decorator-era layout.

    Overrides ``load``, ``load_memmap``, and ``_load_memmap`` on ``cls`` so
    each checks for a nested ``_tensordict/`` directory first, falling through
    to the stock implementations otherwise. Each override earns its place:

    - ``load`` looks like a redundant proxy, but ``TensorClass``'s metaclass
      resolves class attributes against the underlying ``TensorDict``, so the
      stock ``load`` reaches ``TensorDict.load_memmap`` directly and never
      sees the ``load_memmap`` override below.
    - ``load_memmap`` is where ``device=`` and ``out=`` are still intact.
      ``tensordict`` resolves the writing class from the directory's
      ``meta.json`` and, when that isn't the class it is loading through,
      forwards only the prefix -- which is why a decorator-era file loaded
      with ``device="cuda"`` used to come back on the CPU.
    - ``_load_memmap`` is what ``tensordict`` dispatches to for a *nested*
      legacy container: a ``DomainMesh``'s ``interior`` and its boundaries.

    Parameters
    ----------
    cls : type
        The ``TensorClass`` subclass to patch, in place.
    """
    ### TensorClass's metaclass exposes these as wrapper functions rather than
    ### ordinary bound classmethods. Capture them before installing overrides.
    stock_load_memmap = cls.load_memmap
    stock_private_load_memmap = cls._load_memmap

    def _payload_of(out: Any) -> TensorDictBase | None:
        """Unwrap a container passed as ``out=`` to the tensordict it stores.

        ``tensordict`` writes bookkeeping attributes onto whatever it fills,
        which a ``tensor_only`` container rejects; it wants the storage.
        """
        return out._tensordict if isinstance(out, cls) else out

    def load(cls, prefix: str | Path, *args: Any, **kwargs: Any) -> Any:
        """Load a saved container from disk (a proxy for ``load_memmap``)."""
        return cls.load_memmap(prefix, *args, **kwargs)

    def load_memmap(
        cls,
        prefix: str | Path,
        device: torch.device | None = None,
        non_blocking: bool = False,
        *,
        out: TensorDictBase | None = None,
        robust_key: bool | None = None,
    ) -> Any:
        """Load from a memory-mapped directory tree, in either layout."""
        legacy_payload = _legacy_payload_dir(prefix)
        if legacy_payload is None:
            return stock_load_memmap(
                prefix,
                device,
                non_blocking,
                out=_payload_of(out),
                robust_key=robust_key,
            )
        ### The payload is read on its native device and moved afterwards
        ### rather than passing `device` down, because tensordict dispatches
        ### each *nested* legacy container through `_load_memmap`, which it
        ### calls without `device` -- passing it here would leave a DomainMesh
        ### split across devices. Moving the whole result is a no-op for
        ### entries already in the right place.
        result = cls._from_tensordict(
            TensorDict.load_memmap(legacy_payload, robust_key=robust_key)
        )
        if device is not None:
            result = result.to(device, non_blocking=non_blocking)
        if out is not None:
            payload = cast(TensorDictBase, _payload_of(out))
            payload.update(
                result._tensordict,
                inplace=True,
                non_blocking=non_blocking,
            )
            result = cls._from_tensordict(payload)
        return result

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
            return stock_private_load_memmap(
                prefix, metadata, device=device, out=out, **kwargs
            )
        return cls._from_tensordict(
            TensorDict.load_memmap(
                legacy_payload, device=device, out=_payload_of(out), **kwargs
            )
        )

    cls.load = classmethod(load)
    cls.load_memmap = classmethod(load_memmap)
    cls._load_memmap = classmethod(_load_memmap)
