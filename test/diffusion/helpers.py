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


def make_differentiable_solver_options(
    solver_name,
    solver_options,
    scheduler,
    predictor_type,
    device,
):
    """Resolve a solver configuration with differentiable continuous options."""
    options = dict(solver_options) if solver_options else {}
    leaves = {}
    nonzero_names = set()

    def add_leaf(name, value, *, nonzero=True):
        leaf = torch.tensor(float(value), device=device, requires_grad=True)
        leaves[name] = leaf
        if nonzero:
            nonzero_names.add(name)
        return leaf

    if solver_name in {
        "heun",
        "heun_midpoint",
        "stoch_heun",
        "stoch_heun_nochurn",
        "stoch_heun_churn",
    }:
        alpha = options.get("alpha", 1.0)
        options["alpha"] = add_leaf("alpha", 0.8 if alpha == 1.0 else alpha)

    if solver_name.startswith("stoch_"):
        # num_steps is discrete, while S_min and S_max define a boolean mask;
        # none of them has a meaningful pathwise derivative.
        churn = options.get("S_churn", 0.0)
        active_churn = churn > 0
        differentiable_churn = min(churn, 0.2 * options.get("num_steps", 18))
        options["S_churn"] = add_leaf(
            "S_churn", differentiable_churn, nonzero=active_churn
        )
        options["S_noise"] = add_leaf("S_noise", 1.1, nonzero=active_churn)

    if solver_name.startswith("stoch_exp_euler") and "renoise" in options:
        renoise = options["renoise"]
        options["renoise"] = add_leaf(
            "renoise", 0.4 if renoise > 0 else 0.0, nonzero=renoise > 0
        )

    use_edm_sigma_fns = options.pop("_use_edm_sigma_fns", False)
    use_sigma_fns = options.pop("_use_sigma_fns", False)
    if use_edm_sigma_fns or use_sigma_fns:
        sigma_scale = add_leaf("sigma callback", 0.1)
        sigma_inv_scale = add_leaf("sigma inverse callback", 0.1)
        options["sigma_fn"] = lambda t: (
            (1 + sigma_scale * t / (1 + t)) * scheduler.sigma(t)
        )
        options["sigma_inv_fn"] = lambda sigma: scheduler.sigma_inv(
            sigma / (1 + sigma_inv_scale)
        )
        if use_edm_sigma_fns:
            diffusion_scale = add_leaf("diffusion callback", 0.1)
            options["diffusion_fn"] = lambda x, t: (
                (1 + diffusion_scale) * scheduler.diffusion(x, t)
            )
        if solver_name.startswith("stoch_exp_euler"):
            alpha_scale = add_leaf("alpha callback", 0.1)
            options["alpha_fn"] = lambda t: (1 + alpha_scale * t) * scheduler.alpha(t)

    if options.pop("_use_linear_fn", False):
        bias_fn, bias_int_fn, slope_fn = scheduler.get_linear_denoiser(
            prediction_type=predictor_type
        )
        bias_scale = add_leaf("bias callbacks", 0.1)
        options["bias_fn"] = lambda t: bias_fn(t) + bias_scale / (1 + t)
        options["bias_int_fn"] = lambda t: (
            bias_int_fn(t) + (bias_scale * torch.log1p(t))
        )
        if options.pop("_use_slope_fn", False):
            slope_scale = add_leaf("slope callback", 0.1)
            options["slope_fn"] = lambda t: slope_fn(t) + slope_scale / (1 + t)

    if options.pop("_use_log_snr_lambda", False):
        lambda_scale = add_leaf("lambda callback", 0.1)
        options["lambda_fn"] = lambda t: (
            torch.log(scheduler.snr(t)) + (lambda_scale * torch.log1p(t))
        )

    return options, leaves, nonzero_names


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
