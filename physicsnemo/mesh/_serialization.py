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

"""Small adapters around TensorClass's native memmap reader.

TensorDict's typed writer and reader handle both current directly inherited
``TensorClass`` containers and decorator-era ``Mesh`` / ``DomainMesh`` files.
PhysicsNeMo only needs to unwrap a structured ``out=`` container and prevent a
requested device from conflicting with preallocated output storage.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDictBase


def _resolved_device(device: torch.device | str) -> torch.device:
    """Resolve an index-free CUDA device for comparison with tensor devices."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _validate_out_device(
    out: TensorDictBase | None,
    device: torch.device | str | None,
) -> None:
    """Reject an ``out`` whose storage cannot satisfy ``device``."""
    if out is None or device is None:
        return

    out_devices = {
        value.device
        for value in out.values(include_nested=True, leaves_only=True)
        if isinstance(value, torch.Tensor)
    }
    if not out_devices and out.device is not None:
        out_devices.add(out.device)

    requested = _resolved_device(device)
    if out_devices and out_devices != {requested}:
        formatted = ", ".join(sorted(map(str, out_devices)))
        raise ValueError(
            f"`device={requested}` conflicts with `out` tensors on {formatted}; "
            "`device` and `out` must target the same device."
        )


def install_mesh_memmap_reader(cls: type) -> None:
    """Install PhysicsNeMo's thin adapter while preserving TensorClass metadata."""
    stock_load = cls.load
    stock_load_memmap = cls.load_memmap

    def _payload_of(out: Any) -> TensorDictBase | None:
        return out._tensordict if isinstance(out, cls) else out

    @wraps(stock_load)
    def load(cls, prefix: str | Path, *args: Any, **kwargs: Any) -> Any:
        return cls.load_memmap(prefix, *args, **kwargs)

    @wraps(stock_load_memmap)
    def load_memmap(
        cls,
        prefix: str | Path,
        device: torch.device | None = None,
        non_blocking: bool = False,
        *,
        out: TensorDictBase | None = None,
        robust_key: bool | None = True,
        subpath: Any = None,
        mode: str = "r",
        num_threads: int = 0,
    ) -> Any:
        payload = _payload_of(out)
        _validate_out_device(payload, device)
        return stock_load_memmap(
            prefix,
            device,
            non_blocking,
            out=payload,
            robust_key=robust_key,
            subpath=subpath,
            mode=mode,
            num_threads=num_threads,
        )

    # ``stock_*`` are already descriptor-resolved functions whose signatures do
    # not include ``cls``. Leaving ``__wrapped__`` in place makes classmethod
    # binding strip ``prefix`` from the public signature a second time.
    del load.__wrapped__
    del load_memmap.__wrapped__

    cls.load = classmethod(load)
    cls.load_memmap = classmethod(load_memmap)
