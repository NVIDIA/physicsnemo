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

"""Shared helper functions for diffusion model/block tests in this package.

Pytest fixtures (``deterministic_settings``, ``tolerances``, ``nop_compile``,
``reset_dynamo``) live in ``conftest.py`` and are auto-discovered.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch

import physicsnemo.core

# =============================================================================
# Constants
# =============================================================================

GLOBAL_SEED = 42
DATA_DIR = Path(__file__).parent / "data"


# =============================================================================
# Helper functions
# =============================================================================


def instantiate_model_deterministic(cls, seed: int = 0, **kwargs: Any):
    """Instantiate a model with deterministic random parameters."""
    model = cls(**kwargs)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=gen, dtype=param.dtype))
    return model


def load_or_create_reference(
    file_name: str,
    compute_fn: Callable[[], Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Load a saved reference file, or create+save it on first run."""
    path = DATA_DIR / file_name
    if path.exists():
        return torch.load(path, weights_only=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = compute_fn()
    data_cpu = {
        k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in data.items()
    }
    torch.save(data_cpu, path)
    return data


def load_or_create_checkpoint(
    checkpoint_name: str, create_fn: Callable[[], physicsnemo.core.Module]
):
    """Load a saved checkpoint, or create+save it on first run."""
    path = DATA_DIR / checkpoint_name
    if path.exists():
        return physicsnemo.core.Module.from_checkpoint(str(path))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model = create_fn()
    model.save(str(path))
    return model


def compare_outputs(actual: torch.Tensor, expected: torch.Tensor, **tol: Any) -> None:
    """Compare two tensors with detailed shape/value reporting."""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Shape mismatch: actual {actual.shape} vs expected {expected.shape}"
        )
    a64 = actual.to(torch.float64)
    e64 = expected.to(device=actual.device, dtype=torch.float64)
    torch.testing.assert_close(a64, e64, **tol)


def make_input(shape: Tuple[int, ...], seed: int, device: str) -> torch.Tensor:
    """Create a deterministic random input tensor."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn(*shape, generator=gen).to(device)
