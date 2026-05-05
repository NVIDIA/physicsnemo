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

"""Lightweight replacement for Sym's Node — wraps a callable with string-based I/O metadata."""

from __future__ import annotations

from physicsnemo.sym.constants import diff_str


class Computation:
    """A named unit in the computational graph.

    Parameters
    ----------
    inputs : list[str]
        Non-derivative input names (e.g. ``["u", "v", "p"]``).
    outputs : list[str]
        Output names produced by this computation.
    evaluate : callable
        A callable (typically ``torch.nn.Module``) mapping
        ``Dict[str, Tensor] → Dict[str, Tensor]``.
    name : str
        Human-readable label for debugging.
    """

    def __init__(
        self,
        inputs: list[str],
        outputs: list[str],
        evaluate,
        name: str = "Computation",
    ):
        if isinstance(inputs, str):
            inputs = [inputs]
        all_inputs = [str(x) for x in inputs]
        self._inputs = [x for x in all_inputs if diff_str not in x]
        self._derivatives = [x for x in all_inputs if diff_str in x]
        self._outputs = [str(x) for x in outputs]
        self.evaluate = evaluate
        self._name = name

    @classmethod
    def from_sympy(cls, eq, out_name, freeze_terms=None, detach_names=None):
        """Build a Computation from a SymPy expression."""
        from physicsnemo.sym.utils.sympy.torch_printer import (
            SympyToTorch,
            _subs_derivatives,
        )

        if freeze_terms is None:
            freeze_terms = []
        if detach_names is None:
            detach_names = []

        sub_eq = _subs_derivatives(eq)
        evaluate = SympyToTorch(sub_eq, out_name, freeze_terms, detach_names)
        inputs = list(evaluate.keys)
        outputs = [out_name]
        return cls(inputs, outputs, evaluate, name="Sympy Computation: " + out_name)

    @property
    def name(self) -> str:
        """Human-readable label for this computation."""
        return self._name

    @property
    def outputs(self) -> list[str]:
        """Output names produced by this computation."""
        return self._outputs

    @property
    def inputs(self) -> list[str]:
        """Non-derivative input names."""
        return self._inputs

    @property
    def derivatives(self) -> list[str]:
        """Derivative input names (contain ``__``)."""
        return self._derivatives

    def __str__(self) -> str:
        return (
            f"computation: {self.name}\n"
            f"  inputs: {self.inputs}\n"
            f"  derivatives: {self.derivatives}\n"
            f"  outputs: {self.outputs}"
        )
