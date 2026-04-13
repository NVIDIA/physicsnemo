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

"""Helper functions and shared test models for diffusion tests."""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from tensordict import TensorDict

import physicsnemo.core
from physicsnemo.core import Module

# Directory for test reference data
DATA_DIR = Path(__file__).parent / "data"


# =============================================================================
# Shared Test Model Definitions
# =============================================================================


class FlatLinearX0Predictor(Module):
    """Minimal x0-predictor using a linear layer with flatten/reshape.

    Flattens all non-batch dimensions, applies a linear layer, and reshapes
    back. Suitable for any input shape (1D spatial, 2D spatial, flat, etc.).
    """

    def __init__(self, features: int):
        super().__init__()
        self.net = torch.nn.Linear(features, features)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        shape = x.shape
        flat = x.view(x.shape[0], -1)
        out = self.net(flat).view(shape)
        t_bc = t.view(-1, *([1] * (x.ndim - 1)))
        return out / (1 + t_bc)


class Conv2dX0Predictor(Module):
    """Minimal x0-predictor using Conv2d for 4D (B, C, H, W) input."""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.net = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        t_bc = t.view(-1, 1, 1, 1)
        return self.net(x) / (1 + t_bc)


class Conv3dX0Predictor(Module):
    """Minimal x0-predictor using Conv3d for 5D (B, C, D, H, W) input."""

    def __init__(self, channels: int = 2):
        super().__init__()
        self.net = torch.nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        t_bc = t.view(-1, 1, 1, 1, 1)
        return self.net(x) / (1 + t_bc)


def make_input(
    shape: Tuple[int, ...],
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Create a deterministic input tensor using a separate Generator.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the output tensor.
    seed : int
        Random seed for deterministic generation.
    device : str
        Device to place the tensor on.

    Returns
    -------
    torch.Tensor
        A normally-distributed random tensor with the given shape.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn(*shape, generator=gen).to(device)


def instantiate_model_deterministic(
    cls,
    seed: int = 0,
    **kwargs: Any,
) -> physicsnemo.core.Module:
    """
    Instantiate a model with deterministic random parameters.
    """
    model = cls(**kwargs)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(
                torch.randn(
                    param.shape,
                    generator=gen,
                    dtype=param.dtype,
                )
            )
    return model


def generate_batch_data(
    shape: Tuple[int, ...] = (4, 3, 16, 16),
    seed: int = 42,
    device: str = "cpu",
    use_condition: bool = False,
) -> Dict[str, torch.Tensor | TensorDict]:
    """
    Generate deterministic batch data for testing.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the input tensor x.
    seed : int
        Random seed for deterministic generation.
    device : str
        Device to place tensors on.
    use_condition : bool
        If True, generates condition["y"] with the same shape as x.

    Returns
    -------
    Dict containing:
        - "x": Input tensor of given shape
        - "t": Time tensor of shape (batch_size,)
        - "condition": TensorDict with batch_size matching x
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    batch_size = shape[0]
    x = torch.randn(*shape, generator=gen)
    # Use positive t values away from 0 to avoid log(0) issues
    t = torch.rand(batch_size, generator=gen) * 0.5 + 0.4

    # Generate condition as TensorDict with batch_size
    if use_condition:
        condition = TensorDict(
            {"y": torch.randn(*shape, generator=gen).to(device)},
            batch_size=[batch_size],
        )
    else:
        condition = TensorDict({}, batch_size=[batch_size])

    return {
        "x": x.to(device),
        "t": t.to(device),
        "condition": condition.to(device),
    }


def load_or_create_reference(
    file_name: str,
    compute_fn: Optional[Callable[[], Dict[str, torch.Tensor]]],
    *,
    force_recreate: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Load reference data from file, or create it if it doesn't exist.

    Parameters
    ----------
    file_name : str
        Name of the reference data file (relative to DATA_DIR).
    compute_fn : Callable[[], Dict[str, torch.Tensor]]
        Function that computes and returns the reference data dictionary.
        Called only when reference data needs to be created.
    force_recreate : bool, optional
        If True, recreate the reference data even if it exists,
        by default False.

    Returns
    -------
    Dict[str, torch.Tensor]
        The reference data dictionary.
    """
    file_path = DATA_DIR / file_name

    if file_path.exists() and not force_recreate:
        return torch.load(file_path, weights_only=True)

    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Compute reference data
    if compute_fn is None:
        raise FileNotFoundError(
            f"Reference data not found: {file_path}. "
            f"Run test with compute_fn to create it first."
        )
    data = compute_fn()

    # Move all tensors to CPU before saving
    data_cpu = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data_cpu[k] = v.cpu()
        else:
            data_cpu[k] = v

    # Save reference data
    torch.save(data_cpu, file_path)

    return data


def load_or_create_checkpoint(
    checkpoint_name: str,
    create_fn: Optional[Callable[[], physicsnemo.core.Module]],
    force_recreate: bool = False,
) -> physicsnemo.core.Module:
    """
    Load checkpoint from file, or create it if it doesn't exist.
    """
    checkpoint_path = DATA_DIR / checkpoint_name

    if not checkpoint_path.exists() or force_recreate:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if create_fn is None:
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                f"Run test with create_fn to create it first."
            )
        model = create_fn()
        model.save(str(checkpoint_path))
        return model
    else:
        return physicsnemo.core.Module.from_checkpoint(str(checkpoint_path))


def gpu_rng_roundtrip(
    fn: Callable[[], torch.Tensor],
    seed: int,
    device: str,
    atol: float = 1e-2,
    rtol: float = 5e-2,
) -> torch.Tensor:
    """
    Verify RNG-dependent function reproducibility on GPU via seed-roundtrip.

    Performs a three-step test:
    1. Seed RNG, call fn -> result_a
    2. Call fn again (no re-seed) -> result_b, assert result_a != result_b
    3. Re-seed with same seed, call fn -> result_c, assert result_a == result_c

    Parameters
    ----------
    fn : Callable[[], torch.Tensor]
        Zero-argument callable that internally uses random number generation.
    seed : int
        Random seed to use for reproducibility.
    device : str
        Device string (e.g. "cuda:0").
    atol : float
        Absolute tolerance for the allclose comparison.
    rtol : float
        Relative tolerance for the allclose comparison.

    Returns
    -------
    torch.Tensor
        The result from step 1 (result_a).
    """
    torch.manual_seed(seed)
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    result_a = fn()

    result_b = fn()
    assert not torch.allclose(result_a, result_b, atol=atol, rtol=rtol), (
        "Second call without re-seeding should produce different results"
    )

    torch.manual_seed(seed)
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    result_c = fn()
    torch.testing.assert_close(
        result_a,
        result_c,
        atol=atol,
        rtol=rtol,
        msg="Re-seeded call should reproduce the first result",
    )

    return result_a


def compare_outputs(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    """
    Compare actual and expected tensors with detailed error reporting.

    Parameters
    ----------
    actual : torch.Tensor
        The computed tensor.
    expected : torch.Tensor
        The expected reference tensor.
    atol : float, optional
        Absolute tolerance, by default 1e-5.
    rtol : float, optional
        Relative tolerance, by default 1e-5.

    Raises
    ------
    AssertionError
        If tensors don't match within tolerance, with detailed error info.
    """
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Shape mismatch: actual {actual.shape} vs expected {expected.shape}"
        )

    # Move to same device and convert to float64 for comparison
    actual_f64 = actual.to(torch.float64)
    expected_f64 = expected.to(device=actual.device, dtype=torch.float64)

    torch.testing.assert_close(actual_f64, expected_f64, atol=atol, rtol=rtol)
