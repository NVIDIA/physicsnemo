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

"""Tests for diffusion ODE/SDE solvers."""

import pytest
import torch

from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
from physicsnemo.diffusion.samplers.solvers import (
    EDMStochasticEulerSolver,
    EDMStochasticHeunSolver,
    EulerSolver,
    HeunSolver,
    Solver,
)

from .conftest import GLOBAL_SEED
from .helpers import (
    Conv2dX0Predictor,
    Conv3dX0Predictor,
    FlatLinearX0Predictor,
    compare_outputs,
    gpu_rng_roundtrip,
    instantiate_model_deterministic,
    load_or_create_reference,
    make_input,
)

# =============================================================================
# Constants and Configurations
# =============================================================================

REF_PREFIX = "test_solvers_"
BATCH = 2

SPATIAL_CONFIGS = [
    ("1d", (BATCH, 3, 16), FlatLinearX0Predictor, {"features": 3 * 16}),
    ("2d", (BATCH, 3, 8, 6), Conv2dX0Predictor, {"channels": 3}),
    ("3d", (BATCH, 2, 4, 4, 4), Conv3dX0Predictor, {"channels": 2}),
]

# (solver_cls, solver_kwargs, solver_name, uses_rng)
# solver_kwargs are passed to the solver constructor after `denoiser`.
# "_use_edm_sigma_fns" is a sentinel handled by _make_solver.
SOLVER_CONFIGS = [
    (EulerSolver, {}, "euler", False),
    (HeunSolver, {}, "heun", False),
    (HeunSolver, {"alpha": 0.5}, "heun_midpoint", False),
    (EDMStochasticEulerSolver, {"S_churn": 0}, "stoch_euler_nochurn", False),
    (
        EDMStochasticEulerSolver,
        {"S_churn": 40, "num_steps": 10},
        "stoch_euler_churn",
        True,
    ),
    (
        EDMStochasticEulerSolver,
        {"S_churn": 40, "num_steps": 10, "_use_edm_sigma_fns": True},
        "stoch_euler_sigmafns",
        True,
    ),
    (EDMStochasticHeunSolver, {"S_churn": 0}, "stoch_heun_nochurn", False),
    (
        EDMStochasticHeunSolver,
        {"S_churn": 40, "num_steps": 10},
        "stoch_heun_churn",
        True,
    ),
]


def _make_denoiser(shape, predictor_cls, predictor_kwargs, device, seed=0):
    """Create a deterministic ODE denoiser from an x0-predictor via EDM scheduler."""
    model = instantiate_model_deterministic(
        predictor_cls,
        seed=seed,
        **predictor_kwargs,
    ).to(device)
    scheduler = EDMNoiseScheduler()
    return scheduler.get_denoiser(x0_predictor=model, denoising_type="ode"), model


def _identity_denoiser(x, t):
    return x


def _make_solver(solver_cls, solver_kwargs, denoiser):
    """Create a solver, injecting EDM sigma callbacks if requested."""
    kwargs = dict(solver_kwargs)
    if kwargs.pop("_use_edm_sigma_fns", False):
        edm = EDMNoiseScheduler()
        kwargs["sigma_fn"] = edm.sigma
        kwargs["sigma_inv_fn"] = edm.sigma_inv
        kwargs["diffusion_fn"] = edm.diffusion
    return solver_cls(denoiser, **kwargs)


# =============================================================================
# Constructor Tests
# =============================================================================


class TestEulerSolverConstructor:
    """Tests for EulerSolver constructor."""

    def test_attributes(self):
        solver = EulerSolver(_identity_denoiser)
        assert solver.denoiser is _identity_denoiser
        assert isinstance(solver, Solver)


class TestHeunSolverConstructor:
    """Tests for HeunSolver constructor."""

    def test_default_alpha(self):
        solver = HeunSolver(_identity_denoiser)
        assert solver.alpha == pytest.approx(1.0)

    def test_custom_alpha(self):
        solver = HeunSolver(_identity_denoiser, alpha=0.5)
        assert solver.alpha == pytest.approx(0.5)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            HeunSolver(_identity_denoiser, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            HeunSolver(_identity_denoiser, alpha=1.5)


class TestEDMStochasticEulerSolverConstructor:
    """Tests for EDMStochasticEulerSolver constructor."""

    def test_default_attributes(self):
        solver = EDMStochasticEulerSolver(_identity_denoiser)
        assert solver.S_churn == pytest.approx(0.0)
        assert solver.S_noise == pytest.approx(1.0)
        assert solver.num_steps == 18

    def test_sigma_fn_validation(self):
        def sigma_only(t):
            return t

        with pytest.raises(ValueError, match="sigma_fn and sigma_inv_fn"):
            EDMStochasticEulerSolver(_identity_denoiser, sigma_fn=sigma_only)


class TestEDMStochasticHeunSolverConstructor:
    """Tests for EDMStochasticHeunSolver constructor."""

    def test_default_attributes(self):
        solver = EDMStochasticHeunSolver(_identity_denoiser)
        assert solver.alpha == pytest.approx(1.0)
        assert solver.S_churn == pytest.approx(0.0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            EDMStochasticHeunSolver(_identity_denoiser, alpha=0.0)


# =============================================================================
# Non-Regression Tests
# =============================================================================


@pytest.mark.parametrize(
    "solver_cls,solver_kwargs,solver_name,uses_rng",
    SOLVER_CONFIGS,
    ids=[c[2] for c in SOLVER_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestStepNonRegression:
    """Non-regression tests for solver step() across all solver configs."""

    def test_step(
        self,
        deterministic_settings,
        device,
        tolerances,
        solver_cls,
        solver_kwargs,
        solver_name,
        uses_rng,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        denoiser, _ = _make_denoiser(shape, predictor_cls, predictor_kwargs, device)
        solver = _make_solver(solver_cls, solver_kwargs, denoiser)

        x = make_input(shape, seed=100, device=device)
        t_cur = torch.tensor([5.0] * shape[0], device=device)
        t_next = torch.tensor([2.5] * shape[0], device=device)

        ref_file = f"{REF_PREFIX}{solver_name}_{spatial_name}_step.pth"
        if "cuda" in str(device) and uses_rng:

            def fn():
                return solver.step(x, t_cur, t_next)

            result = gpu_rng_roundtrip(fn, GLOBAL_SEED, str(device))
            assert result.shape == shape
            ref = load_or_create_reference(ref_file, None)
            assert result.shape == ref["x_next"].shape
        else:
            x_next = solver.step(x, t_cur, t_next)
            assert x_next.shape == shape
            ref = load_or_create_reference(ref_file, lambda: {"x_next": x_next.cpu()})
            compare_outputs(x_next, ref["x_next"], **tolerances)

    def test_step_to_zero(
        self,
        deterministic_settings,
        device,
        tolerances,
        solver_cls,
        solver_kwargs,
        solver_name,
        uses_rng,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Step to t=0 should produce finite output."""
        denoiser, _ = _make_denoiser(shape, predictor_cls, predictor_kwargs, device)
        solver = _make_solver(solver_cls, solver_kwargs, denoiser)

        x = make_input(shape, seed=101, device=device)
        t_cur = torch.tensor([1.0] * shape[0], device=device)
        t_next = torch.tensor([0.0] * shape[0], device=device)

        x_next = solver.step(x, t_cur, t_next)
        assert x_next.shape == shape
        assert torch.isfinite(x_next).all()

    def test_zero_churn_matches_deterministic(
        self,
        deterministic_settings,
        device,
        tolerances,
        solver_cls,
        solver_kwargs,
        solver_name,
        uses_rng,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Stochastic solvers with S_churn=0 should match their deterministic counterpart."""
        if solver_name == "stoch_euler_nochurn":
            det_cls = EulerSolver
        elif solver_name == "stoch_heun_nochurn":
            det_cls = HeunSolver
        else:
            pytest.skip("Only applies to zero-churn stochastic configs")

        denoiser, _ = _make_denoiser(shape, predictor_cls, predictor_kwargs, device)
        stoch_solver = _make_solver(solver_cls, solver_kwargs, denoiser)
        det_solver = det_cls(denoiser)

        x = make_input(shape, seed=120, device=device)
        t_cur = torch.tensor([5.0] * shape[0], device=device)
        t_next = torch.tensor([2.5] * shape[0], device=device)

        x_stoch = stoch_solver.step(x, t_cur, t_next)
        x_det = det_solver.step(x, t_cur, t_next)
        compare_outputs(x_stoch, x_det, **tolerances)


# =============================================================================
# Compile Tests
# =============================================================================


@pytest.mark.parametrize(
    "solver_cls,solver_kwargs,solver_name,uses_rng",
    SOLVER_CONFIGS,
    ids=[c[2] for c in SOLVER_CONFIGS],
)
@pytest.mark.parametrize(
    "spatial_name,shape,predictor_cls,predictor_kwargs",
    SPATIAL_CONFIGS,
    ids=[c[0] for c in SPATIAL_CONFIGS],
)
class TestStepCompile:
    """Double-call compile tests for solver step()."""

    def test_compiled_step(
        self,
        deterministic_settings,
        device,
        solver_cls,
        solver_kwargs,
        solver_name,
        uses_rng,
        spatial_name,
        shape,
        predictor_cls,
        predictor_kwargs,
    ):
        """Compiled step traces without error and graph is reused on second call."""
        torch._dynamo.config.error_on_recompile = True

        denoiser, _ = _make_denoiser(shape, predictor_cls, predictor_kwargs, device)
        solver = _make_solver(solver_cls, solver_kwargs, denoiser)

        x = make_input(shape, seed=100, device=device)
        t_cur = torch.tensor([5.0] * shape[0], device=device)
        t_next = torch.tensor([2.5] * shape[0], device=device)

        compiled_step = torch.compile(solver.step, fullgraph=True)

        with torch.no_grad():
            out_compiled = compiled_step(x, t_cur, t_next)
        assert out_compiled.shape == shape
        assert torch.isfinite(out_compiled).all()

        # Second call — must reuse the graph
        with torch.no_grad():
            out_compiled_2 = compiled_step(x, t_cur, t_next)
        assert out_compiled_2.shape == shape
        assert torch.isfinite(out_compiled_2).all()

        # For deterministic solvers, also verify eager-vs-compiled match
        if not uses_rng:
            with torch.no_grad():
                out_eager = solver.step(x, t_cur, t_next)
            torch.testing.assert_close(out_eager, out_compiled)
