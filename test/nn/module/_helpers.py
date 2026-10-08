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

"""Shared helpers for reusable layer non-regression tests."""

import random
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from physicsnemo.core import Module

GLOBAL_SEED = 42
DATA_DIR = Path(__file__).parent / "data"


class LayerTestFixtures:
    """Fixtures scoped to the dedicated layer non-regression test classes."""

    @pytest.fixture
    def tolerances(self, device):
        """Return device-specific non-regression tolerances."""
        if device == "cpu":
            return {"atol": 1e-3, "rtol": 1e-3}
        return {"atol": 1e-2, "rtol": 5e-2}

    @pytest.fixture
    def deterministic_settings(self):
        """Set deterministic settings and restore them after the test."""
        old_cudnn_deterministic = torch.backends.cudnn.deterministic
        old_cudnn_benchmark = torch.backends.cudnn.benchmark
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        old_random_state = random.getstate()

        try:
            random.seed(GLOBAL_SEED)
            torch.manual_seed(GLOBAL_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(GLOBAL_SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            yield
        finally:
            torch.backends.cudnn.deterministic = old_cudnn_deterministic
            torch.backends.cudnn.benchmark = old_cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            random.setstate(old_random_state)


def instantiate_module_deterministic(
    module_cls: type[Module], seed: int = 0, **kwargs: Any
) -> Module:
    """Instantiate a module with deterministic random parameters."""
    module = module_cls(**kwargs)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=parameter.dtype,
                )
            )
    return module


def load_or_create_reference(
    file_name: str,
    compute_fn: Callable[[], dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Load saved reference tensors or create them on the first run."""
    path = DATA_DIR / file_name
    if path.exists():
        return torch.load(path, weights_only=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = compute_fn()
    data_cpu = {name: tensor.cpu() for name, tensor in data.items()}
    torch.save(data_cpu, path)
    return data_cpu


def load_or_create_checkpoint(
    file_name: str,
    create_fn: Callable[[], Module],
) -> Module:
    """Load a saved module checkpoint or create it on the first run."""
    path = DATA_DIR / file_name
    if path.exists():
        return Module.from_checkpoint(str(path))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    module = create_fn()
    module.save(str(path))
    return module


def make_inputs(
    input_shapes: tuple[tuple[int, ...], ...],
    device: str,
) -> tuple[torch.Tensor, ...]:
    """Create deterministic inputs for a module configuration."""
    return tuple(
        torch.randn(
            *shape,
            generator=torch.Generator(device="cpu").manual_seed(GLOBAL_SEED + index),
        ).to(device)
        for index, shape in enumerate(input_shapes)
    )


def compare_outputs(
    actual: torch.Tensor,
    expected: torch.Tensor,
    **tolerances: float,
) -> None:
    """Compare actual and reference outputs using device-specific tolerances."""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Shape mismatch: actual {actual.shape} vs expected {expected.shape}"
        )

    torch.testing.assert_close(
        actual.to(torch.float64),
        expected.to(device=actual.device, dtype=torch.float64),
        **tolerances,
    )
