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

"""ASV benchmarks for PhysicsNeMo functionals.

This benchmark runner times FunctionSpec implementations for forward and
optionally backward workloads.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import Any

import torch

from benchmarks.physicsnemo.nn.functional._spec_utils import (
    PHASE_ORDER,
    BenchmarkKey,
    build_benchmark_plan,
    case_by_index,
)
from benchmarks.physicsnemo.nn.functional.registry import FUNCTIONAL_SPECS
from physicsnemo.core.function_spec import FunctionSpec

# Environment variable names used by this benchmark runner.
_ENV_DEVICE = "PHYSICSNEMO_ASV_DEVICE"
_ENV_PHASES = "PHYSICSNEMO_ASV_PHASES"
_ENV_FUNCTIONALS = "PHYSICSNEMO_ASV_FUNCTIONALS"


def _parse_csv_env(name: str) -> tuple[str, ...]:
    """Parse a comma-separated environment variable into normalized tokens."""

    value = os.getenv(name, "")
    if not value:
        return ()
    return tuple(token.strip().lower() for token in value.split(",") if token.strip())


def _resolve_device() -> torch.device:
    """Resolve the benchmark device from environment or runtime availability."""

    # Allow explicit device override from environment.
    env_device = os.getenv(_ENV_DEVICE)
    if env_device:
        return torch.device(env_device)

    # Default to CUDA when available; otherwise benchmark on CPU.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_phases() -> tuple[str, ...]:
    """Resolve benchmark phases in stable output order."""

    # Keep forward-only as default to avoid long ASV runs by default.
    requested = set(_parse_csv_env(_ENV_PHASES) or ("forward",))
    unknown = requested.difference(PHASE_ORDER)
    if unknown:
        valid = ", ".join(PHASE_ORDER)
        unknown_display = ", ".join(sorted(unknown))
        raise ValueError(
            f"{_ENV_PHASES} contains unsupported phase(s): {unknown_display}. "
            f"Valid phases: {valid}"
        )

    # Preserve canonical phase ordering independent of env token ordering.
    phases = tuple(phase for phase in PHASE_ORDER if phase in requested)
    if not phases:
        valid = ", ".join(PHASE_ORDER)
        raise ValueError(f"{_ENV_PHASES} must contain one or both of: {valid}")
    return phases


def _resolve_specs(
    specs: Iterable[type[FunctionSpec]],
) -> tuple[type[FunctionSpec], ...]:
    """Resolve the spec subset requested by environment filtering."""

    requested = set(_parse_csv_env(_ENV_FUNCTIONALS))
    if not requested:
        return tuple(specs)

    # Match on spec class name, case-insensitive.
    selected = tuple(spec for spec in specs if spec.__name__.lower() in requested)
    if selected:
        return selected

    available = ", ".join(sorted(spec.__name__ for spec in specs))
    raise ValueError(
        f"{_ENV_FUNCTIONALS} did not match any FunctionSpec. "
        f"Requested: {os.getenv(_ENV_FUNCTIONALS)!r}. Available: {available}"
    )


def _iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    """Yield tensor leaves from nested tuple/list/dict structures."""

    # Yield tensor leaves directly.
    if torch.is_tensor(value):
        yield value
        return

    # Recurse into tuple/list containers.
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)
        return

    # Recurse into dictionary values.
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def _clear_input_gradients(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """Clear gradients on reusable benchmark inputs before backward timing."""

    # ASV reuses the same prepared input objects across timing calls.
    for tensor in _iter_tensors((args, kwargs)):
        if tensor.grad is not None:
            tensor.grad = None


def _loss_from_output(output: Any) -> torch.Tensor:
    """Reduce possibly nested benchmark outputs to one scalar loss tensor."""

    # Keep only tensors that are connected to autograd.
    differentiable_tensors = [
        tensor
        for tensor in _iter_tensors(output)
        if tensor.requires_grad or tensor.grad_fn is not None
    ]
    if not differentiable_tensors:
        raise ValueError(
            "Backward benchmark output must contain at least one differentiable tensor."
        )

    # Build a numerically stable scalar objective across all output tensors.
    def _norm_term(tensor: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(tensor):
            return tensor.abs().square().mean()
        return tensor.float().square().mean()

    loss = _norm_term(differentiable_tensors[0])
    for tensor in differentiable_tensors[1:]:
        loss = loss + _norm_term(tensor)
    return loss


# Resolve all benchmark metadata once at import for ASV.
_DEVICE = _resolve_device()
_PHASES = _resolve_phases()
_SELECTED_SPECS = _resolve_specs(FUNCTIONAL_SPECS)
_BENCHMARK_KEYS, _KEY_TO_SPEC = build_benchmark_plan(
    device=_DEVICE,
    phases=_PHASES,
    selected_specs=_SELECTED_SPECS,
)


class FunctionalBenchmarks:
    """ASV benchmark suite for registered FunctionSpec implementations."""

    # ASV expects ``params`` to be one list per parameter axis.
    params = [_BENCHMARK_KEYS]
    param_names = ["phase_spec_impl_case_index"]
    timeout = 120

    def setup(self, phase_spec_impl_case_index: BenchmarkKey) -> None:
        """Prepare one benchmark workload before timing starts."""

        # Resolve static metadata for this parameterized benchmark key.
        self.phase = phase_spec_impl_case_index[0]
        self.spec = _KEY_TO_SPEC[phase_spec_impl_case_index]
        self.implementation = phase_spec_impl_case_index[2]
        self.case_index = phase_spec_impl_case_index[3]

        # Materialize this input case once; ASV will repeatedly time it.
        _, self.args, self.kwargs = case_by_index(
            spec=self.spec,
            phase=self.phase,
            case_index=self.case_index,
            device=_DEVICE,
        )

        # Ensure prior CUDA work does not bleed into this timing window.
        if _DEVICE.type == "cuda":
            torch.cuda.synchronize()

    def time_functional(self, phase_spec_impl_case_index: BenchmarkKey) -> None:
        """Time forward or backward execution for one configured workload."""

        # Forward phase measures plain dispatch.
        if self.phase == "forward":
            self.spec.dispatch(
                *self.args, **self.kwargs, implementation=self.implementation
            )

        # Backward phase dispatches, builds a scalar loss, then runs backward().
        else:
            _clear_input_gradients(args=self.args, kwargs=self.kwargs)
            output = self.spec.dispatch(
                *self.args, **self.kwargs, implementation=self.implementation
            )
            _loss_from_output(output).backward()

        # Synchronize so measured time includes CUDA kernel completion.
        if _DEVICE.type == "cuda":
            torch.cuda.synchronize()
