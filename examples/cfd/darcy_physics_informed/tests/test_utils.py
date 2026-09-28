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

"""Checks the Diffusion equation in utils.py, including the time-dependent form."""

import sys
from pathlib import Path

from sympy import Function, Number, Symbol, simplify

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import Diffusion  # noqa: E402


def test_time_dependent_diffusion_keeps_time_derivative():
    """time=True gives dT/dt - D (T_xx + T_yy) - Q, with T a function of t."""
    x, y, t = Symbol("x"), Symbol("y"), Symbol("t")
    T = Function("u")(x, y, t)
    expected = T.diff(t) - Number(0.5) * (T.diff(x, 2) + T.diff(y, 2))
    eq = Diffusion(T="u", D=0.5, time=True).equations["diffusion_u"]
    assert simplify(eq - expected) == 0


def test_steady_diffusion_with_variable_diffusivity():
    """time=False (as used by the Darcy examples) gives -div(k grad u) - Q."""
    x, y = Symbol("x"), Symbol("y")
    u, k, f = (Function(n)(x, y) for n in ("u", "k", "f"))
    expected = -(k * u.diff(x)).diff(x) - (k * u.diff(y)).diff(y) - f
    eq = Diffusion(T="u", D="k", Q=f, time=False).equations["diffusion_u"]
    assert simplify(eq - expected) == 0
