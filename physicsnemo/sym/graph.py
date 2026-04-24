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

"""Computational graph for PDE residual evaluation."""

from __future__ import annotations

from copy import copy
from typing import Dict, List

import torch

from physicsnemo.sym.computation import Computation


class Graph(torch.nn.Module):
    """Unroll a set of :class:`Computation` nodes into a sequential evaluation order.

    The scheduler uses a two-phase loop:
    1. Greedily schedule computations whose inputs *and* derivative inputs are
       already available.
    2. When stuck, satisfy outstanding derivative requests using ``diff_nodes``.

    Parameters
    ----------
    nodes : list[Computation]
        Equation computations (produced by :meth:`PDE.make_computations`).
    invar : list[str]
        Names of tensors that will be provided in the forward call.
    req_names : list[str]
        Names of tensors that must be present in the output.
    diff_nodes : list[Computation]
        Derivative computations (gradient adapters).
    """

    def __init__(
        self,
        nodes: List[Computation],
        invar: List[str],
        req_names: List[str],
        diff_nodes: List[Computation] | None = None,
    ):
        super().__init__()

        if diff_nodes is None:
            diff_nodes = []

        self.req_names = list(req_names)

        computable = set(_computable_names(nodes, invar))
        req_base = {n.split("__")[0] if "__" in n else n for n in req_names}
        if not req_base.issubset(computable):
            _print_graph_unroll_error(nodes, invar, req_names)
            raise RuntimeError("Failed unrolling graph")

        necessary_nodes = _prune_nodes(copy(nodes), req_names)

        self.node_evaluation_order: list[Computation] = []
        outvar = list(invar)

        while True:
            prev_len = len(outvar)

            while True:
                finished = True
                for i, node in enumerate(necessary_nodes):
                    if set(node.inputs + node.derivatives).issubset(set(outvar)):
                        self.node_evaluation_order.append(node)
                        outvar += node.outputs
                        necessary_nodes.pop(i)
                        finished = False
                if finished:
                    break

            needed = [
                d
                for d in _collect_needed_derivatives(necessary_nodes, req_names)
                if d not in outvar
            ]
            if needed:
                for dn in diff_nodes:
                    if (not set(dn.outputs).isdisjoint(set(needed))) and set(
                        dn.inputs
                    ).issubset(set(outvar)):
                        self.node_evaluation_order.append(dn)
                        outvar += dn.outputs

            if set(req_names).issubset(set(outvar)):
                break

            if len(outvar) == prev_len:
                unsatisfied = set(req_names) - set(outvar)
                raise RuntimeError(
                    f"Graph scheduler stalled — cannot satisfy: {unsatisfied}. "
                    f"Available: {sorted(set(outvar))}. "
                    f"Remaining nodes: {[str(n) for n in necessary_nodes]}"
                )

        self.evaluation_order = torch.nn.ModuleList(
            [n.evaluate for n in self.node_evaluation_order]
        )
        self.node_names: List[str] = [n.name for n in self.node_evaluation_order]

    def forward(self, invar: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outvar = dict(invar)
        for i, module in enumerate(self.evaluation_order):
            outvar.update(module(outvar))
        return {k: v for k, v in outvar.items() if k in self.req_names}


def _computable_names(nodes: list[Computation], invar: list[str]) -> list[str]:
    """Fixed-point closure of names reachable from *invar* through *nodes*."""
    nodes = copy(nodes)
    names = list(invar)
    while True:
        finished = True
        for i, node in enumerate(nodes):
            if set(node.inputs).issubset(set(names)):
                names += node.outputs
                nodes.pop(i)
                finished = False
        if finished:
            return names


def _prune_nodes(nodes: list[Computation], req_names: list[str]) -> list[Computation]:
    """Walk backwards from *req_names* and keep only needed nodes."""
    needed_names = set(req_names) | {n.split("__")[0] for n in req_names if "__" in n}
    necessary: list[Computation] = []
    while True:
        finished = True
        for i, node in enumerate(nodes):
            if not set(node.outputs).isdisjoint(needed_names):
                needed_names.update(node.inputs)
                needed_names.update(node.derivatives)
                needed_names.update(
                    d.split("__")[0] for d in node.derivatives if "__" in d
                )
                necessary.append(node)
                nodes.pop(i)
                finished = False
        if finished:
            return necessary


def _collect_needed_derivatives(
    remaining_nodes: list[Computation], req_names: list[str]
) -> list[str]:
    needed: list[str] = []
    for node in remaining_nodes:
        needed += node.derivatives
    needed += [x for x in req_names if "__" in x]
    return needed


def _print_graph_unroll_error(nodes, invar, req_names):
    print("=" * 60)
    print("Could not unroll graph!")
    print(f"  invar: {invar}")
    print(f"  requested: {req_names}")
    print(f"  computable: {_computable_names(nodes, invar)}")
    for node in nodes:
        print(f"  node: {node}")
    print("=" * 60)
