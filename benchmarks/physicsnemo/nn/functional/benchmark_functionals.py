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

"""ASV benchmarks for PhysicsNeMo functionals."""
# TODO: This code will likely evolve with CI/CD integration.

from __future__ import annotations

import os
from typing import Any, Iterable

import torch

from benchmarks.physicsnemo.nn.functional.registry import FUNCTIONAL_SPECS


def _resolve_device() -> torch.device:
    """Resolve the device to benchmark on."""

    # Allow the benchmark device to be overridden from the environment.
    device_name = os.getenv("PHYSICSNEMO_ASV_DEVICE")
    if device_name:
        return torch.device(device_name)

    # Prefer CUDA when available; otherwise default to CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _filter_specs(specs: Iterable[type]) -> list[type]:
    """Filter the specs to the requested subset, this is mostly used for debugging locally."""

    # Allow selecting a subset of functionals for quick benchmark iteration.
    spec_filter = os.getenv("PHYSICSNEMO_ASV_FUNCTIONALS")
    if not spec_filter:
        return list(specs)

    # Parse comma-separated spec names into a normalized lookup set.
    requested = {
        name.strip().lower() for name in spec_filter.split(",") if name.strip()
    }
    if not requested:
        return list(specs)

    # Keep only specs explicitly requested by name.
    selected = [spec for spec in specs if spec.__name__.lower() in requested]
    if not selected:
        available = ", ".join(sorted(spec.__name__ for spec in specs))
        raise ValueError(
            "PHYSICSNEMO_ASV_FUNCTIONALS did not match any FunctionSpec. "
            f"Requested: {spec_filter!r}. Available: {available}"
        )
    return selected


# Resolve benchmark configuration and precompute all ASV parameter tuples.
_DEVICE = _resolve_device()
_PARAMS: list[tuple[str, str, int]] = []
_SELECTED_SPECS = _filter_specs(FUNCTIONAL_SPECS)
_WORK_ITEMS: dict[
    tuple[str, str, int], tuple[type, str, tuple[Any, ...], dict[str, Any]]
] = {}

# Build the ASV parameter triples: (spec_name, implementation_name, case_index).
for spec in _SELECTED_SPECS:
    # Skip specs that currently have no dispatchable implementations.
    implementations = spec.available_implementations()
    if not implementations:
        continue

    # Materialize inputs once so ASV setup can index by case id.
    # TODO: This is not ideal, we should keep make_inputs as a generator
    cases = list(spec.make_inputs(device=_DEVICE))
    if not cases:
        continue

    # Build ASV parameter triples and cache resolved work items for setup().
    for impl in implementations:
        for case_index, case in enumerate(cases):
            label, args, kwargs = case
            key = (spec.__name__, impl, case_index)
            _PARAMS.append(key)
            _WORK_ITEMS[key] = (spec, label, args, kwargs)


class FunctionalBenchmarks:
    """Benchmark registered FunctionSpec implementations with ASV."""

    # ASV expects params to be a list of parameter axes.
    params = [_PARAMS]
    param_names = ["spec_impl_case"]
    timeout = 120

    def setup(self, spec_impl_case: tuple[str, str, int]) -> None:
        # Resolve the precomputed work item for this benchmark key.
        spec, _, args, kwargs = _WORK_ITEMS[spec_impl_case]

        # Cache resolved objects on self to minimize per-iteration overhead.
        self.spec = spec
        self.implementation = spec_impl_case[1]
        self.args = args
        self.kwargs = kwargs

        # Synchronize before timing so previous CUDA work is excluded.
        if _DEVICE.type == "cuda":
            torch.cuda.synchronize()

    def time_functional(self, spec_impl_case: tuple[str, str, int]) -> None:
        # Dispatch to the selected implementation for the selected input case.
        self.spec.dispatch(
            *self.args, **self.kwargs, implementation=self.implementation
        )
        # Synchronize to ensure the measured time includes kernel execution.
        if _DEVICE.type == "cuda":
            torch.cuda.synchronize()
