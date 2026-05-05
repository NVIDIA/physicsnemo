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

"""PhysicsNeMo Sym: symbolic PDE residual computation.

Example
-------
>>> from sympy import Symbol, Function
>>> from physicsnemo.sym.eq.pde import PDE
>>> from physicsnemo.sym.eq.phy_informer import PhysicsInformer
>>>
>>> class Poisson(PDE):
...     def __init__(self):
...         self.dim = 2
...         x, y = Symbol("x"), Symbol("y")
...         u = Function("u")(x, y)
...         self.equations = {"poisson": u.diff(x, 2) + u.diff(y, 2)}
...
>>> pde = Poisson()
>>> pi = PhysicsInformer(["poisson"], pde, grad_method="autodiff")
>>> sorted(pi.required_inputs)
['coordinates', 'u']
"""

from physicsnemo.sym.eq.gradients import (
    GradientCalculator,
    compute_connectivity_tensor,
)
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer

__all__ = [
    "GradientCalculator",
    "PDE",
    "PhysicsInformer",
    "compute_connectivity_tensor",
]
