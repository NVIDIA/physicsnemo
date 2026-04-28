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

"""Shared helpers for functional ASV benchmark scripts.

This module centralizes FunctionSpec case/phase handling so benchmark timing and
plot generation always interpret benchmark metadata the same way.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, TypeAlias

import torch

from physicsnemo.core.function_spec import FunctionSpec

# Keep benchmark phases in a stable order for reproducible outputs.
PHASE_ORDER = ("forward", "backward")

# Canonical benchmark case tuple type yielded by input generators.
BenchmarkCase: TypeAlias = tuple[str, tuple[Any, ...], dict[str, Any]]

# ASV benchmark key format: (phase, spec_name, implementation_name, case_index).
BenchmarkKey: TypeAlias = tuple[str, str, str, int]


def _phase_case_iterator(
    spec: type[FunctionSpec], phase: str, device: torch.device | str
) -> Iterator[BenchmarkCase]:
    """Return the phase-specific case iterator for ``spec``."""

    # Forward and backward each have their own FunctionSpec generator hook.
    if phase == "forward":
        return spec.make_inputs_forward(device=device)
    if phase == "backward":
        return spec.make_inputs_backward(device=device)
    raise ValueError(f"Unsupported benchmark phase: {phase}")


def supports_backward_inputs(spec: type[FunctionSpec]) -> bool:
    """Return ``True`` when ``spec`` overrides backward input generation."""

    # Compare unbound callables so this works regardless of classmethod wrapping.
    spec_fn = getattr(spec.make_inputs_backward, "__func__", spec.make_inputs_backward)
    base_fn = getattr(
        FunctionSpec.make_inputs_backward, "__func__", FunctionSpec.make_inputs_backward
    )
    return spec_fn is not base_fn


def _metadata_case_labels(spec: type[FunctionSpec]) -> list[str]:
    """Extract benchmark case labels from optional FunctionSpec metadata."""

    # Prefer static metadata declared directly on the FunctionSpec.
    benchmark_cases = getattr(spec, "_BENCHMARK_CASES", None)
    if isinstance(benchmark_cases, Iterable):
        labels = [
            case[0]
            for case in benchmark_cases
            if isinstance(case, tuple) and case and isinstance(case[0], str)
        ]
        if labels:
            return labels

    # Fall back to callable metadata hook when present.
    benchmark_cases_fn = getattr(spec, "_benchmark_cases", None)
    if callable(benchmark_cases_fn):
        labels = [
            case[0]
            for case in benchmark_cases_fn()
            if isinstance(case, tuple) and case and isinstance(case[0], str)
        ]
        if labels:
            return labels

    return []


def case_labels(
    spec: type[FunctionSpec], phase: str, device: torch.device | str
) -> list[str]:
    """Resolve benchmark case labels for one spec and one phase."""

    # Validate requested phase and skip unsupported backward benchmarking.
    if phase not in PHASE_ORDER:
        raise ValueError(f"Unsupported benchmark phase: {phase}")
    if phase == "backward" and not supports_backward_inputs(spec):
        return []

    # Metadata labels avoid materializing full tensor inputs while plotting.
    labels = _metadata_case_labels(spec)
    if labels:
        return labels

    # Fall back to labels from the phase-specific input generator.
    return [
        label
        for label, _, _ in _phase_case_iterator(spec=spec, phase=phase, device=device)
    ]


def case_by_index(
    spec: type[FunctionSpec],
    phase: str,
    case_index: int,
    device: torch.device | str,
) -> BenchmarkCase:
    """Materialize exactly one benchmark case by index."""

    # Walk the case iterator until the requested index is reached.
    case_iter = _phase_case_iterator(spec=spec, phase=phase, device=device)
    for index, case in enumerate(case_iter):
        if index == case_index:
            return case

    raise IndexError(
        f"Case index {case_index} out of range for {spec.__name__} phase={phase}"
    )


def build_benchmark_plan(
    *,
    device: torch.device | str,
    phases: Sequence[str],
    selected_specs: Iterable[type[FunctionSpec]],
) -> tuple[list[BenchmarkKey], dict[BenchmarkKey, type[FunctionSpec]]]:
    """Build ASV benchmark keys and their corresponding FunctionSpec classes."""

    keys: list[BenchmarkKey] = []
    key_to_spec: dict[BenchmarkKey, type[FunctionSpec]] = {}

    # Keep this ordering aligned with ASV's parameter vector positions.
    for spec in selected_specs:
        implementations = spec.available_implementations()
        if not implementations:
            continue

        for phase in phases:
            labels = case_labels(spec=spec, phase=phase, device=device)
            if not labels:
                continue

            for implementation_name in implementations:
                for case_index, _ in enumerate(labels):
                    key = (phase, spec.__name__, implementation_name, case_index)
                    keys.append(key)
                    key_to_spec[key] = spec

    return keys, key_to_spec


__all__ = [
    "PHASE_ORDER",
    "BenchmarkCase",
    "BenchmarkKey",
    "build_benchmark_plan",
    "supports_backward_inputs",
    "case_labels",
    "case_by_index",
]
