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

    python test/experimental/models/data/_generate_xdeeponet_goldens.py

Overwrites the committed fixtures with freshly-seeded model outputs.
Invoke this deliberately whenever model numerics intentionally change
(architecture edit, default-argument change, etc.) and commit the
resulting ``.pth`` files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]
# Repo root: so ``import physicsnemo...`` resolves.
# xdeeponet test dir: so ``import test_xdeeponet`` resolves.
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(
    0, str(_REPO_ROOT / "test" / "experimental" / "models" / "xdeeponet")
)

from test_xdeeponet import (  # noqa: E402
    _GOLDEN_2D,
    _GOLDEN_3D,
    _init_lazy,
    _wrapper_2d,
    _wrapper_3d,
)


def _write(path: Path, builder) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model, x = builder()
    _init_lazy(model, x)
    with torch.no_grad():
        y = model(x)
    torch.save({"x": x, "y": y, "state_dict": model.state_dict()}, path)
    print(
        f"wrote {path.relative_to(_REPO_ROOT)} "
        f"x={tuple(x.shape)} y={tuple(y.shape)} "
        f"size={path.stat().st_size}B"
    )


if __name__ == "__main__":
    _write(_GOLDEN_2D, _wrapper_2d)
    _write(_GOLDEN_3D, _wrapper_3d)
