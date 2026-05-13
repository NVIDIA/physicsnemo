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

"""Regenerate the xDeepONet golden ``.pth`` fixtures.

Run from the repository root::

    python test/experimental/models/xdeeponet/data/_generate_xdeeponet_goldens.py

Overwrites the committed fixtures with freshly-seeded model outputs.
Invoke this deliberately whenever model numerics intentionally change
(architecture edit, default-argument change, etc.) and commit the
resulting ``.pth`` files.

The set of fixtures is driven by :data:`_FIXTURE_REGISTRY` in
``test_xdeeponet.py`` — adding a new scenario there automatically
extends this generator.

Each fixture stores a dict with three keys:

- ``"args"``: tuple of positional forward arguments
- ``"y"``: stored output for the non-regression assertion
- ``"state_dict"``: model parameters
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]
# Repo root: so ``import physicsnemo...`` resolves.
# xdeeponet test dir: so ``import test_xdeeponet`` resolves.
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "test" / "experimental" / "models" / "xdeeponet"))

from test_xdeeponet import (  # noqa: E402
    _FIXTURE_REGISTRY,
    _init_lazy,
)


def _write(path: Path, builder) -> None:
    """Materialise lazy weights, run forward, and save the golden payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model, args = builder()
    _init_lazy(model, *args)
    with torch.no_grad():
        y = model(*args)
    torch.save(
        {"args": tuple(args), "y": y, "state_dict": model.state_dict()},
        path,
    )
    arg_shapes = [tuple(a.shape) for a in args]
    print(
        f"wrote {path.relative_to(_REPO_ROOT)} "
        f"args={arg_shapes} y={tuple(y.shape)} "
        f"size={path.stat().st_size}B"
    )


if __name__ == "__main__":
    for _name, _builder, _golden_path in _FIXTURE_REGISTRY:
        _write(_golden_path, _builder)
