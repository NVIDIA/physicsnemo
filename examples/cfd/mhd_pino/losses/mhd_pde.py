# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from physicsnemo.sym.eq.pde import PDE
from sympy import Symbol, Function, Number


class MHD_PDE(PDE):
    """MHD PDEs using PhysicsNeMo Sym"""

    name = "MHD_PDE"

    def __init__(self, nu=1e-4, eta=1e-4, rho0=1.0):

        # x, y, time
        x, y, t, lap = Symbol("x"), Symbol("y"), Symbol("t"), Symbol("lap")

        # make input variables
        input_variables = {"x": x, "y": y, "t": t, "lap": lap}

        # make functions
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        Bx = Function("Bx")(*input_variables)
        By = Function("By")(*input_variables)
        A = Function("A")(*input_variables)
        ptot = Function("ptot")(*input_variables)

        u_rhs = Function("u_rhs")(*input_variables)
        v_rhs = Function("v_rhs")(*input_variables)
        Bx_rhs = Function("Bx_rhs")(*input_variables)
        By_rhs = Function("By_rhs")(*input_variables)
        A_rhs = Function("A_rhs")(*input_variables)

        # initialize constants
        nu = Number(nu)
        eta = Number(eta)
        rho0 = Number(rho0)

        # set equations
        self.equations = {}

        self.equations["vel_grad_u"] = u * u.diff(x) + v * u.diff(y)
        self.equations["vel_grad_v"] = u * v.diff(x) + v * v.diff(y)

        self.equations["B_grad_u"] = Bx * u.diff(x) + v * Bx.diff(y)
        self.equations["B_grad_v"] = Bx * v.diff(x) + By * v.diff(y)

        self.equations["vel_grad_Bx"] = u * Bx.diff(x) + v * Bx.diff(y)
        self.equations["vel_grad_By"] = u * By.diff(x) + v * By.diff(y)

        self.equations["B_grad_Bx"] = Bx * Bx.diff(x) + By * Bx.diff(y)
        self.equations["B_grad_By"] = Bx * By.diff(x) + By * By.diff(y)

        self.equations["uBy_x"] = u * By.diff(x) + By * u.diff(x)
        self.equations["uBy_y"] = u * By.diff(y) + By * u.diff(y)
        self.equations["vBx_x"] = v * Bx.diff(x) + Bx * v.diff(x)
        self.equations["vBx_y"] = v * Bx.diff(y) + Bx * v.diff(y)

        self.equations["div_B"] = Bx.diff(x) + By.diff(y)
        self.equations["div_vel"] = u.diff(x) + v.diff(y)

        # RHS of MHD equations
        self.equations["u_rhs"] = (
            -self.equations["vel_grad_u"]
            - ptot.diff(x) / rho0
            + self.equations["B_grad_Bx"] / rho0
            + nu * u.diff(lap)
        )
        self.equations["v_rhs"] = (
            -self.equations["vel_grad_v"]
            - ptot.diff(y) / rho0
            + self.equations["B_grad_By"] / rho0
            + nu * v.diff(lap)
        )
        self.equations["Bx_rhs"] = (
            self.equations["uBy_y"] - self.equations["vBx_y"] + eta * Bx.diff(lap)
        )
        self.equations["By_rhs"] = -(
            self.equations["uBy_x"] - self.equations["vBx_x"]
        ) + eta * By.diff(lap)

        self.equations["Du"] = u.diff(t) - u_rhs
        self.equations["Dv"] = v.diff(t) - v_rhs
        self.equations["DBx"] = Bx.diff(t) - Bx_rhs
        self.equations["DBy"] = By.diff(t) - By_rhs

        # Vec potential equations
        self.equations["vel_grad_A"] = u * A.diff(x) + v * A.diff(y)
        self.equations["A_rhs"] = -self.equations["vel_grad_A"] + +eta * A.diff(lap)
        self.equations["DA"] = A.diff(t) - A_rhs
