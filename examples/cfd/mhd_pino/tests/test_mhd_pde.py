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

"""Checks the symbolic terms of MHD_PDE (losses/mhd_pde.py)."""

import sys
from pathlib import Path

from sympy import Function, Symbol, simplify

sys.path.insert(0, str(Path(__file__).parent.parent))

from losses.mhd_pde import MHD_PDE  # noqa: E402


def _fields():
    x, y, t, lap = Symbol("x"), Symbol("y"), Symbol("t"), Symbol("lap")
    return x, y, *(Function(name)(x, y, t, lap) for name in ("u", "v", "Bx", "By"))


def test_b_grad_u_is_b_dot_grad_u():
    """(B . grad) u = Bx du/dx + By du/dy."""
    x, y, u, v, Bx, By = _fields()
    expected = Bx * u.diff(x) + By * u.diff(y)
    assert simplify(MHD_PDE().equations["B_grad_u"] - expected) == 0


def test_b_grad_v_is_b_dot_grad_v():
    """(B . grad) v = Bx dv/dx + By dv/dy."""
    x, y, u, v, Bx, By = _fields()
    expected = Bx * v.diff(x) + By * v.diff(y)
    assert simplify(MHD_PDE().equations["B_grad_v"] - expected) == 0
