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

"""SymPy-to-PyTorch conversion utilities."""

from typing import Dict, List

import numpy as np
import torch
from sympy import Add, Basic, Derivative, Function, Symbol, lambdify
from sympy.printing.str import StrPrinter

from physicsnemo.sym.constants import diff_str, tf_dt


def torch_lambdify(f, r, separable=False):
    """Generate a PyTorch callable from a SymPy expression."""
    try:
        f = float(f)
    except (TypeError, ValueError):
        pass
    if isinstance(f, (float, int, bool)):

        def loop_lambda(constant):
            return lambda **x: torch.zeros_like(next(iter(x.items()))[1]) + constant

        return loop_lambda(f)

    variables = [k for k in r] if separable else [[k for k in r]]
    return lambdify(variables, f, [TORCH_SYMPY_PRINTER])


def _where_torch(conditions, x, y):
    if isinstance(x, (int, float)):
        x = float(x) * torch.ones_like(conditions)
    if isinstance(y, (int, float)):
        y = float(y) * torch.ones_like(conditions)
    return torch.where(conditions, x, y)


def _heaviside_torch(x, values=0):
    return torch.maximum(torch.sign(x), torch.zeros(1, device=x.device))


def _sqrt_torch(x):
    return torch.sqrt((x - 1e-6) * _heaviside_torch(x - 1e-6) + 1e-6)


def _or_torch(*x):
    return_value = x[0]
    for value in x:
        return_value = torch.logical_or(return_value, value)
    return return_value


def _and_torch(*x):
    return_value = x[0]
    for value in x:
        return_value = torch.logical_and(return_value, value)
    return return_value


def _min_torch(*x):
    for value in x:
        if not isinstance(value, (int, float)):
            tensor_shape = list(map(int, value.shape))
            device = value.device

    x_only_tensors = []
    for value in x:
        if isinstance(value, (int, float)):
            value = torch.zeros(tensor_shape, device=device) + value
        x_only_tensors.append(value)

    min_tensor, _ = torch.min(torch.stack(x_only_tensors, -1), -1)
    return min_tensor


def _max_torch(*x):
    for value in x:
        if not isinstance(value, (int, float)):
            tensor_shape = list(map(int, value.shape))
            device = value.device

    x_only_tensors = []
    for value in x:
        if isinstance(value, (int, float)):
            value = (torch.zeros(tensor_shape) + value).to(device)
        x_only_tensors.append(value)

    max_tensor, _ = torch.max(torch.stack(x_only_tensors, -1), -1)
    return max_tensor


def _dirac_delta_torch(x):
    return torch.eq(x, 0.0).to(tf_dt)


TORCH_SYMPY_PRINTER = {
    "abs": torch.abs,
    "Abs": torch.abs,
    "sign": torch.sign,
    "ceiling": torch.ceil,
    "floor": torch.floor,
    "log": torch.log,
    "exp": torch.exp,
    "sqrt": _sqrt_torch,
    "cos": torch.cos,
    "acos": torch.acos,
    "sin": torch.sin,
    "asin": torch.asin,
    "tan": torch.tan,
    "atan": torch.atan,
    "atan2": torch.atan2,
    "cosh": torch.cosh,
    "acosh": torch.acosh,
    "sinh": torch.sinh,
    "asinh": torch.asinh,
    "tanh": torch.tanh,
    "atanh": torch.atanh,
    "erf": torch.erf,
    "loggamma": torch.lgamma,
    "Min": _min_torch,
    "Max": _max_torch,
    "Heaviside": _heaviside_torch,
    "DiracDelta": _dirac_delta_torch,
    "logical_or": _or_torch,
    "logical_and": _and_torch,
    "where": _where_torch,
    "pi": np.pi,
    "conjugate": torch.conj,
}


class CustomDerivativePrinter(StrPrinter):
    """Print SymPy derivatives as ``u__x`` style names using ``diff_str``."""

    def _print_Function(self, expr):
        return expr.func.__name__

    def _print_Derivative(self, expr):
        prefix = str(expr.args[0].func)
        for deriv_expr in expr.args[1:]:
            prefix += deriv_expr[1] * (diff_str + str(deriv_expr[0]))
        return prefix


def _subs_derivatives(expr):
    """Replace SymPy Derivative and Function atoms with named Symbols."""
    while True:
        try:
            deriv = expr.atoms(Derivative).pop()
            new_fn_name = str(deriv)
            expr = expr.subs(deriv, Function(new_fn_name)(*deriv.free_symbols))
        except KeyError:
            break
    while True:
        try:
            fn = {fn for fn in expr.atoms(Function) if fn.class_key()[1] == 0}.pop()
            new_symbol_name = str(fn)
            expr = expr.subs(fn, Symbol(new_symbol_name))
        except KeyError:
            break
    return expr


# This global patch is required so that str(Derivative(u(x), x)) produces "u__x"
# instead of "Derivative(u(x), x)".  The entire _subs_derivatives → SympyToTorch
# pipeline relies on this naming convention to wire derivative keys in the graph.
Basic.__str__ = lambda self: CustomDerivativePrinter().doprint(self)


class SympyToTorch(torch.nn.Module):
    """Compile a SymPy expression into a callable PyTorch module."""

    def __init__(
        self,
        sympy_expr,
        name: str,
        freeze_terms: List[int] | None = None,
        detach_names: List[str] | None = None,
    ):
        super().__init__()
        self.keys = sorted([k.name for k in sympy_expr.free_symbols])
        self.freeze_terms = freeze_terms if freeze_terms is not None else []
        if not self.freeze_terms:
            self.torch_expr = torch_lambdify(sympy_expr, self.keys)
        else:
            if not all(x < len(Add.make_args(sympy_expr)) for x in freeze_terms):
                raise ValueError(
                    "freeze_terms indices must be less than the number of terms in the expression"
                )
            self.torch_expr = []
            for i in range(len(Add.make_args(sympy_expr))):
                self.torch_expr.append(
                    torch_lambdify(Add.make_args(sympy_expr)[i], self.keys)
                )
            self.freeze_list = list(self.torch_expr[i] for i in freeze_terms)
        self.name = name
        self.detach_names = detach_names if detach_names is not None else []

    def forward(self, var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        args = [
            var[k].detach() if k in self.detach_names else var[k] for k in self.keys
        ]
        if not self.freeze_terms:
            output = self.torch_expr(args)
        else:
            output = torch.zeros_like(var[self.keys[0]])
            for i, expr in enumerate(self.torch_expr):
                if expr in self.freeze_list:
                    output += expr(args).detach()
                else:
                    output += expr(args)

        return {self.name: output}
