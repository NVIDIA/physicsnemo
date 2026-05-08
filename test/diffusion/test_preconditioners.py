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

"""Tests for diffusion preconditioners."""

from typing import Any, Tuple

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.core import Module
from physicsnemo.diffusion.preconditioners import (
    BaseAffinePreconditioner,
    EDMPreconditioner,
    IDDPMPreconditioner,
    VEPreconditioner,
    VPPreconditioner,
)

from .helpers import (
    compare_outputs,
    generate_batch_data,
    instantiate_model_deterministic,
    load_or_create_checkpoint,
    load_or_create_reference,
)

# =============================================================================
# Test Model Definitions
# =============================================================================


class ConvModel(Module):
    """Convolutional model for testing preconditioners with 4D input."""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.channels = channels
        # Conv2d takes x concatenated with condition["y"] (same shape as x)
        in_channels = channels * 2
        self.net = torch.nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: TensorDict | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if condition is not None:
            y = condition["y"]
            x_cond = torch.cat([x, y], dim=1)
        else:
            # For unconditional: duplicate x to match expected channels
            y = torch.zeros_like(x)
            x_cond = torch.cat([x, y], dim=1)
        out = self.net(x_cond)
        t_scale = t.view(-1, 1, 1, 1)
        return out + t_scale


class LinearModel(Module):
    """Linear model for testing preconditioners with 2D input."""

    def __init__(self, in_features: int = 64):
        super().__init__()
        self.in_features = in_features
        # Simple linear layer that preserves dimension
        self.net = torch.nn.Linear(in_features, in_features)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: TensorDict | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        out = self.net(x)
        t_scale = t.view(-1, 1)
        return out + t_scale


# =============================================================================
# Constants and Preconditioner Configurations
# =============================================================================


# Test shapes for different model types
# 4D shape for ConvModel: (batch_size, channels, height, width)
CONV_SHAPE: Tuple[int, ...] = (4, 3, 8, 6)
# 2D shape for LinearModel: (batch_size, features)
LINEAR_SHAPE: Tuple[int, ...] = (4, 16)

# Model configurations for parameterized tests: (model_class, shape, arch_name)
MODEL_CONFIGS = [
    (ConvModel, CONV_SHAPE, "conv"),
    (LinearModel, LINEAR_SHAPE, "linear"),
]

# Preconditioner configurations for parameterized tests
PRECOND_CONFIGS = [
    (
        VPPreconditioner,
        {"beta_d": 19.9, "beta_min": 0.1, "M": 2000},
        "vp_precond",
    ),
    (
        VEPreconditioner,
        {},
        "ve_precond",
    ),
    (
        IDDPMPreconditioner,
        {"C_1": 0.001, "C_2": 0.008, "M": 2000},
        "iddpm_precond",
    ),
    (
        EDMPreconditioner,
        {"sigma_data": 1.0},
        "edm_precond",
    ),
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(params=MODEL_CONFIGS, ids=["ConvModel", "LinearModel"])
def model_config(request):
    """Parameterized fixture returning (model_class, shape, arch_name)."""
    return request.param


@pytest.fixture
def test_shape(model_config):
    """Return the test shape for the current model class."""
    _, shape, _ = model_config
    return shape


@pytest.fixture
def arch_name(model_config):
    """Return the architecture name for reference file naming."""
    _, _, name = model_config
    return name


@pytest.fixture
def simple_model(model_config):
    """Create a model with deterministic parameters."""
    cls, shape, _ = model_config
    if cls == LinearModel:
        return instantiate_model_deterministic(cls, seed=0, in_features=shape[1])
    return instantiate_model_deterministic(cls, seed=0, channels=shape[1])


@pytest.fixture
def batch_data(model_config, device):
    """Create deterministic batch data matching the model's expected shape."""
    model_cls, shape, _ = model_config
    # ConvModel uses condition, LinearModel does not
    use_condition = model_cls == ConvModel
    return generate_batch_data(
        shape=shape, seed=42, device=device, use_condition=use_condition
    )


def create_model_deterministic(model_cls, shape):
    """Create a model with deterministic parameters for the given shape."""
    if model_cls == LinearModel:
        return instantiate_model_deterministic(model_cls, seed=0, in_features=shape[1])
    return instantiate_model_deterministic(model_cls, seed=0, channels=shape[1])


def create_preconditioner(precond_cls, precond_kwargs, model_cls, shape):
    """Create a preconditioner with deterministic model."""
    model = create_model_deterministic(model_cls, shape)
    return precond_cls(model, **precond_kwargs)


# =============================================================================
# VPPreconditioner Tests
# =============================================================================


class TestVPPreconditioner:
    """Tests for VPPreconditioner."""

    @pytest.mark.parametrize(
        "config,beta_d,beta_min,M",
        [
            ("default", 19.9, 0.1, 1000),
            ("custom", 10.0, 0.05, 500),
        ],
        ids=["default", "custom"],
    )
    def test_constructor_attributes(self, simple_model, config, beta_d, beta_min, M):
        """Test VPPreconditioner constructor and attributes."""
        if config == "default":
            # Test with default values - verify against known defaults
            precond = VPPreconditioner(simple_model)
            assert precond.beta_d.item() == pytest.approx(19.9)
            assert precond.beta_min.item() == pytest.approx(0.1)
            assert precond.M.item() == 1000
        else:
            # Test with custom values - verify against passed arguments
            precond = VPPreconditioner(
                simple_model, beta_d=beta_d, beta_min=beta_min, M=M
            )
            assert precond.beta_d.item() == pytest.approx(beta_d)
            assert precond.beta_min.item() == pytest.approx(beta_min)
            assert precond.M.item() == M

        assert precond.model is simple_model
        assert isinstance(precond, BaseAffinePreconditioner)


# =============================================================================
# VEPreconditioner Tests
# =============================================================================


class TestVEPreconditioner:
    """Tests for VEPreconditioner."""

    def test_constructor_attributes(self, simple_model):
        """Test VEPreconditioner constructor and attributes."""
        precond = VEPreconditioner(simple_model)

        assert precond.model is simple_model
        assert isinstance(precond, BaseAffinePreconditioner)


# =============================================================================
# IDDPMPreconditioner Tests
# =============================================================================


class TestIDDPMPreconditioner:
    """Tests for IDDPMPreconditioner."""

    @pytest.mark.parametrize(
        "config,C_1,C_2,M",
        [
            ("default", 0.001, 0.008, 1000),
            ("custom", 0.002, 0.01, 500),
        ],
        ids=["default", "custom"],
    )
    def test_constructor_attributes(self, simple_model, config, C_1, C_2, M):
        """Test IDDPMPreconditioner constructor and attributes."""
        if config == "default":
            # Test with default values - verify against known defaults
            precond = IDDPMPreconditioner(simple_model)
            assert precond.C_1.item() == pytest.approx(0.001)
            assert precond.C_2.item() == pytest.approx(0.008)
            assert precond.M.item() == 1000
            expected_M = 1000
        else:
            # Test with custom values - verify against passed arguments
            precond = IDDPMPreconditioner(simple_model, C_1=C_1, C_2=C_2, M=M)
            assert precond.C_1.item() == pytest.approx(C_1)
            assert precond.C_2.item() == pytest.approx(C_2)
            assert precond.M.item() == M
            expected_M = M

        assert hasattr(precond, "u")
        assert precond.u.shape == (expected_M + 1,)
        assert isinstance(precond, BaseAffinePreconditioner)


# =============================================================================
# EDMPreconditioner Tests
# =============================================================================


class TestEDMPreconditioner:
    """Tests for EDMPreconditioner."""

    @pytest.mark.parametrize(
        "config,sigma_data",
        [
            ("default", 0.5),
            ("custom", 1.0),
        ],
        ids=["default", "custom"],
    )
    def test_constructor_attributes(self, simple_model, config, sigma_data):
        """Test EDMPreconditioner constructor and attributes."""
        if config == "default":
            # Test with default values - verify against known defaults
            precond = EDMPreconditioner(simple_model)
            assert precond.sigma_data.item() == pytest.approx(0.5)
        else:
            # Test with custom values - verify against passed arguments
            precond = EDMPreconditioner(simple_model, sigma_data=sigma_data)
            assert precond.sigma_data.item() == pytest.approx(sigma_data)

        assert precond.model is simple_model
        assert isinstance(precond, BaseAffinePreconditioner)

    @pytest.mark.parametrize(
        "sigma_data_input",
        [
            [0.3, 0.5, 0.7],
            torch.tensor([0.3, 0.5, 0.7]),
        ],
        ids=["list", "tensor"],
    )
    def test_per_channel_constructor(self, simple_model, sigma_data_input):
        """Test EDMPreconditioner with per-channel sigma_data."""
        precond = EDMPreconditioner(simple_model, sigma_data=sigma_data_input)
        assert precond.sigma_data.shape == (3,)
        expected = torch.tensor([0.3, 0.5, 0.7])
        assert torch.allclose(precond.sigma_data, expected)

    def test_per_channel_coefficients_shape(self, model_config, device):
        """Test compute_coefficients returns correct shapes for per-channel."""
        model_cls, shape, _ = model_config
        C = shape[1]
        model = create_model_deterministic(model_cls, shape)
        sigma_data = [0.5 + 0.1 * i for i in range(C)]
        precond = EDMPreconditioner(model, sigma_data=sigma_data).to(device)

        B = shape[0]
        sigma_shape = (B,) + (1,) * (len(shape) - 1)
        t = torch.rand(B, device=device).view(sigma_shape)

        c_in, c_noise, c_out, c_skip = precond.compute_coefficients(t)

        expected_ch_shape = (B, C) + (1,) * (len(shape) - 2)
        assert c_in.shape == expected_ch_shape
        assert c_out.shape == expected_ch_shape
        assert c_skip.shape == expected_ch_shape
        assert c_noise.shape == sigma_shape

    def test_per_channel_forward(self, model_config, device):
        """Test forward pass works with per-channel sigma_data."""
        model_cls, shape, _ = model_config
        C = shape[1]
        model = create_model_deterministic(model_cls, shape)
        sigma_data = [0.5 + 0.1 * i for i in range(C)]
        precond = EDMPreconditioner(model, sigma_data=sigma_data).to(device)

        use_condition = model_cls == ConvModel
        data = generate_batch_data(
            shape=shape, seed=42, device=device, use_condition=use_condition
        )
        out = precond(data["x"], data["t"], condition=data["condition"])
        assert out.shape == shape

    def test_per_channel_matches_scalar_when_uniform(self, model_config, device):
        """Per-channel sigma_data with identical values must match scalar."""
        model_cls, shape, _ = model_config
        C = shape[1]
        sd_value = 0.7

        model_scalar = create_model_deterministic(model_cls, shape)
        model_perchan = create_model_deterministic(model_cls, shape)

        precond_scalar = EDMPreconditioner(model_scalar, sigma_data=sd_value).to(device)
        precond_perchan = EDMPreconditioner(
            model_perchan, sigma_data=[sd_value] * C
        ).to(device)

        use_condition = model_cls == ConvModel
        data = generate_batch_data(
            shape=shape, seed=42, device=device, use_condition=use_condition
        )

        out_scalar = precond_scalar(data["x"], data["t"], condition=data["condition"])
        out_perchan = precond_perchan(data["x"], data["t"], condition=data["condition"])
        assert torch.allclose(out_scalar, out_perchan, atol=1e-6)

    def test_single_element_sequence_stored_as_scalar(self, simple_model):
        """A length-1 sequence or tensor is stored as a 0-D buffer."""
        precond = EDMPreconditioner(simple_model, sigma_data=[0.5])
        assert precond.sigma_data.ndim == 0
        assert precond.sigma_data.item() == pytest.approx(0.5)


# =============================================================================
# Non-Regression Tests (Parameterized Across All Preconditioners and Models)
# =============================================================================


@pytest.mark.parametrize(
    "precond_cls,precond_kwargs,precond_name",
    PRECOND_CONFIGS,
    ids=["VP", "VE", "iDDPM", "EDM"],
)
class TestNonRegression:
    """Non-regression tests parameterized across all preconditioner types."""

    def test_sigma_non_regression(
        self,
        deterministic_settings,
        model_config,
        batch_data,
        device,
        tolerances,
        precond_cls,
        precond_kwargs,
        precond_name,
    ):
        """Test sigma(t) against reference data."""
        model_cls, shape, arch_name = model_config
        precond = create_preconditioner(
            precond_cls, precond_kwargs, model_cls, shape
        ).to(device)

        t = batch_data["t"]
        sigma = precond.sigma(t)

        ref_file = f"{precond_name}_{arch_name}_sigma.pth"
        ref_data = load_or_create_reference(ref_file, lambda: {"sigma": sigma.cpu()})

        compare_outputs(sigma, ref_data["sigma"], **tolerances)

    def test_sigma_from_checkpoint(
        self,
        deterministic_settings,
        model_config,
        batch_data,
        device,
        tolerances,
        precond_cls,
        precond_kwargs,
        precond_name,
    ):
        """Test sigma(t) from loaded checkpoint matches reference."""
        model_cls, shape, arch_name = model_config

        def create_fn():
            return create_preconditioner(precond_cls, precond_kwargs, model_cls, shape)

        ckpt_file = f"{precond_name}_{arch_name}.mdlus"
        precond = load_or_create_checkpoint(ckpt_file, create_fn).to(device)

        t = batch_data["t"]
        sigma = precond.sigma(t)

        ref_file = f"{precond_name}_{arch_name}_sigma.pth"
        ref_data = load_or_create_reference(ref_file, lambda: {"sigma": sigma.cpu()})

        compare_outputs(sigma, ref_data["sigma"], **tolerances)

    def test_coefficients_non_regression(
        self,
        deterministic_settings,
        model_config,
        batch_data,
        device,
        tolerances,
        precond_cls,
        precond_kwargs,
        precond_name,
    ):
        """Test compute_coefficients against reference data."""
        model_cls, shape, arch_name = model_config
        precond = create_preconditioner(
            precond_cls, precond_kwargs, model_cls, shape
        ).to(device)

        # Reshape t to sigma shape: (B, 1, ..., 1)
        batch_size = shape[0]
        sigma_shape = (batch_size,) + (1,) * (len(shape) - 1)
        sigma = batch_data["t"].view(sigma_shape)

        c_in, c_noise, c_out, c_skip = precond.compute_coefficients(sigma)

        # Load existing reference or save current output as reference
        ref_file = f"{precond_name}_{arch_name}_coefficients.pth"
        ref_data = load_or_create_reference(
            ref_file,
            lambda: {
                "c_in": c_in.cpu(),
                "c_noise": c_noise.cpu(),
                "c_out": c_out.cpu(),
                "c_skip": c_skip.cpu(),
            },
        )

        compare_outputs(c_in, ref_data["c_in"], **tolerances)
        compare_outputs(c_noise, ref_data["c_noise"], **tolerances)
        compare_outputs(c_out, ref_data["c_out"], **tolerances)
        compare_outputs(c_skip, ref_data["c_skip"], **tolerances)

    def test_coefficients_from_checkpoint(
        self,
        deterministic_settings,
        model_config,
        batch_data,
        device,
        tolerances,
        precond_cls,
        precond_kwargs,
        precond_name,
    ):
        """Test compute_coefficients from checkpoint matches reference."""
        model_cls, shape, arch_name = model_config

        def create_fn():
            return create_preconditioner(precond_cls, precond_kwargs, model_cls, shape)

        ckpt_file = f"{precond_name}_{arch_name}.mdlus"
        precond = load_or_create_checkpoint(ckpt_file, create_fn).to(device)

        # Reshape t to sigma shape: (B, 1, ..., 1)
        batch_size = shape[0]
        sigma_shape = (batch_size,) + (1,) * (len(shape) - 1)
        sigma = batch_data["t"].view(sigma_shape)

        c_in, c_noise, c_out, c_skip = precond.compute_coefficients(sigma)

        ref_file = f"{precond_name}_{arch_name}_coefficients.pth"
        ref_data = load_or_create_reference(
            ref_file,
            lambda: {
                "c_in": c_in.cpu(),
                "c_noise": c_noise.cpu(),
                "c_out": c_out.cpu(),
                "c_skip": c_skip.cpu(),
            },
        )

        compare_outputs(c_in, ref_data["c_in"], **tolerances)
        compare_outputs(c_noise, ref_data["c_noise"], **tolerances)
        compare_outputs(c_out, ref_data["c_out"], **tolerances)
        compare_outputs(c_skip, ref_data["c_skip"], **tolerances)

    def test_forward_non_regression(
        self,
        deterministic_settings,
        model_config,
        batch_data,
        device,
        tolerances,
        precond_cls,
        precond_kwargs,
        precond_name,
    ):
        """Test forward pass against reference data."""
        model_cls, shape, arch_name = model_config
        precond = create_preconditioner(
            precond_cls, precond_kwargs, model_cls, shape
        ).to(device)

        x = batch_data["x"]
        t = batch_data["t"]
        condition = batch_data["condition"]
        out = precond(x, t, condition=condition)

        ref_file = f"{precond_name}_{arch_name}_forward.pth"
        ref_data = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})

        compare_outputs(out, ref_data["out"], **tolerances)

    def test_forward_from_checkpoint(
        self,
        deterministic_settings,
        model_config,
        batch_data,
        device,
        tolerances,
        precond_cls,
        precond_kwargs,
        precond_name,
    ):
        """Test forward pass from loaded checkpoint matches reference."""
        model_cls, shape, arch_name = model_config

        def create_fn():
            return create_preconditioner(precond_cls, precond_kwargs, model_cls, shape)

        ckpt_file = f"{precond_name}_{arch_name}.mdlus"
        precond = load_or_create_checkpoint(ckpt_file, create_fn).to(device)

        x = batch_data["x"]
        t = batch_data["t"]
        condition = batch_data["condition"]
        out = precond(x, t, condition=condition)

        ref_file = f"{precond_name}_{arch_name}_forward.pth"
        ref_data = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})

        compare_outputs(out, ref_data["out"], **tolerances)


# =============================================================================
# Other tests for all preconditioner types
# =============================================================================


@pytest.mark.parametrize(
    "precond_cls,precond_kwargs,precond_name",
    PRECOND_CONFIGS,
    ids=["VP", "VE", "iDDPM", "EDM"],
)
class TestAllPreconditioners:
    """Tests that apply to all preconditioner types."""

    def test_forward_input_validation(
        self,
        simple_model,
        batch_data,
        device,
        precond_cls,
        precond_kwargs,
        precond_name,
    ):
        """Test forward validates input shapes."""
        precond = precond_cls(simple_model, **precond_kwargs).to(device)
        x = batch_data["x"]
        t_wrong = torch.rand(2, device=device)  # Wrong batch size
        condition = batch_data["condition"]

        with pytest.raises(ValueError, match="Expected t to have shape"):
            precond(x, t_wrong, condition=condition)

    def test_forward_dtype_preservation(
        self,
        simple_model,
        batch_data,
        device,
        precond_cls,
        precond_kwargs,
        precond_name,
    ):
        """Test forward preserves input dtype."""
        precond = precond_cls(simple_model, **precond_kwargs).to(device)
        x = batch_data["x"]
        t = batch_data["t"]
        condition = batch_data["condition"]

        output = precond(x, t, condition=condition)

        assert output.dtype == x.dtype

    def test_condition_batch_validation(
        self,
        simple_model,
        test_shape,
        device,
        precond_cls,
        precond_kwargs,
        precond_name,
    ):
        """Test condition batch size validation."""
        precond = precond_cls(simple_model, **precond_kwargs).to(device)
        x = torch.randn(*test_shape, device=device)
        t = torch.rand(test_shape[0], device=device)
        # Wrong batch size in condition TensorDict
        condition = TensorDict(
            {"cond": torch.randn(2, 10, device=device)},
            batch_size=[2],
        )

        with pytest.raises(ValueError, match="batch size"):
            precond(x, t, condition=condition)

    def test_gradient_flow(
        self,
        simple_model,
        batch_data,
        device,
        precond_cls,
        precond_kwargs,
        precond_name,
    ):
        """Test gradients flow through the preconditioner."""
        precond = precond_cls(simple_model, **precond_kwargs).to(device)
        x = batch_data["x"].clone().requires_grad_(True)
        t = batch_data["t"]
        condition = batch_data["condition"]

        output = precond(x, t, condition=condition)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    @pytest.mark.usefixtures("nop_compile")
    def test_compile(
        self,
        simple_model,
        batch_data,
        device,
        precond_cls,
        precond_kwargs,
        precond_name,
    ):
        """Compiled forward matches eager and graph is reused on second call."""
        torch._dynamo.config.error_on_recompile = True

        precond = precond_cls(simple_model, **precond_kwargs).to(device)
        x = batch_data["x"]
        t = batch_data["t"]
        condition = batch_data["condition"]

        compiled_precond = torch.compile(precond, fullgraph=True)

        with torch.no_grad():
            out_eager = precond(x, t, condition=condition)
            out_compiled = compiled_precond(x, t, condition=condition)
        torch.testing.assert_close(out_eager, out_compiled)

        # Second call — must reuse compiled graph
        with torch.no_grad():
            out_compiled_2 = compiled_precond(x, t, condition=condition)
        torch.testing.assert_close(out_compiled, out_compiled_2)

    def test_model_kwargs_passthrough(
        self,
        model_config,
        batch_data,
        device,
        precond_cls,
        precond_kwargs,
        precond_name,
    ):
        """model_kwargs are forwarded to the underlying model."""
        model_cls, shape, _ = model_config

        class KwargsCapture(Module):
            """Model that records extra kwargs it receives."""

            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self.last_kwargs: dict = {}

            def forward(self, x, t, condition=None, **model_kwargs):
                self.last_kwargs = model_kwargs
                return self.inner(x, t, condition=condition)

        base = create_model_deterministic(model_cls, shape).to(device)
        wrapper = KwargsCapture(base).to(device)
        precond = precond_cls(wrapper, **precond_kwargs).to(device)

        x = batch_data["x"]
        t = batch_data["t"]
        condition = batch_data["condition"]

        precond(x, t, condition=condition, my_flag=True, scale=0.5)
        assert wrapper.last_kwargs == {"my_flag": True, "scale": 0.5}

    def test_model_kwargs_gradient_flow(
        self,
        model_config,
        batch_data,
        device,
        precond_cls,
        precond_kwargs,
        precond_name,
    ):
        """Gradients flow when model_kwargs are passed."""
        model_cls, shape, _ = model_config

        class IgnoreKwargsModel(Module):
            """Accepts but ignores extra kwargs."""

            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x, t, condition=None, **model_kwargs):
                return self.inner(x, t, condition=condition)

        base = create_model_deterministic(model_cls, shape).to(device)
        wrapper = IgnoreKwargsModel(base).to(device)
        precond = precond_cls(wrapper, **precond_kwargs).to(device)

        x = batch_data["x"].clone().requires_grad_(True)
        t = batch_data["t"]
        condition = batch_data["condition"]

        out = precond(x, t, condition=condition, extra_kwarg=42)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# =============================================================================
# EDMPreconditioner per-channel sigma_data tests
# =============================================================================

# Per-channel sigma_data configurations: (sigma_data, name)
_PER_CHANNEL_SIGMA_CONFIGS = [
    (torch.tensor([0.3, 0.5, 0.7]), "list_3ch"),
]


def _make_per_channel_preconditioner(sigma_data, model_cls, shape):
    """Create an EDMPreconditioner with per-channel sigma_data."""
    model = create_model_deterministic(model_cls, shape)
    return EDMPreconditioner(model, sigma_data=sigma_data)


class TestEDMPerChannelNonRegression:
    """Non-regression tests for EDMPreconditioner with per-channel sigma_data."""

    @pytest.mark.parametrize(
        "sigma_data,sd_name",
        _PER_CHANNEL_SIGMA_CONFIGS,
        ids=[c[1] for c in _PER_CHANNEL_SIGMA_CONFIGS],
    )
    def test_coefficients_non_regression(
        self,
        deterministic_settings,
        model_config,
        batch_data,
        device,
        tolerances,
        sigma_data,
        sd_name,
    ):
        """Per-channel compute_coefficients against reference data."""
        model_cls, shape, arch_name = model_config
        # Skip LinearModel — per-channel sigma_data is for spatial models
        if model_cls == LinearModel:
            pytest.skip("Per-channel sigma_data only applies to ConvModel")
        precond = _make_per_channel_preconditioner(sigma_data, model_cls, shape).to(
            device
        )

        batch_size = shape[0]
        sigma_shape = (batch_size,) + (1,) * (len(shape) - 1)
        sigma = batch_data["t"].view(sigma_shape)

        c_in, c_noise, c_out, c_skip = precond.compute_coefficients(sigma)

        ref_file = f"edm_perchan_{sd_name}_{arch_name}_coefficients.pth"
        ref_data = load_or_create_reference(
            ref_file,
            lambda: {
                "c_in": c_in.cpu(),
                "c_noise": c_noise.cpu(),
                "c_out": c_out.cpu(),
                "c_skip": c_skip.cpu(),
            },
        )
        compare_outputs(c_in, ref_data["c_in"], **tolerances)
        compare_outputs(c_noise, ref_data["c_noise"], **tolerances)
        compare_outputs(c_out, ref_data["c_out"], **tolerances)
        compare_outputs(c_skip, ref_data["c_skip"], **tolerances)

    @pytest.mark.parametrize(
        "sigma_data,sd_name",
        _PER_CHANNEL_SIGMA_CONFIGS,
        ids=[c[1] for c in _PER_CHANNEL_SIGMA_CONFIGS],
    )
    def test_forward_non_regression(
        self,
        deterministic_settings,
        model_config,
        batch_data,
        device,
        tolerances,
        sigma_data,
        sd_name,
    ):
        """Per-channel forward pass against reference data."""
        model_cls, shape, arch_name = model_config
        if model_cls == LinearModel:
            pytest.skip("Per-channel sigma_data only applies to ConvModel")
        precond = _make_per_channel_preconditioner(sigma_data, model_cls, shape).to(
            device
        )

        x = batch_data["x"]
        t = batch_data["t"]
        condition = batch_data["condition"]
        out = precond(x, t, condition=condition)

        ref_file = f"edm_perchan_{sd_name}_{arch_name}_forward.pth"
        ref_data = load_or_create_reference(ref_file, lambda: {"out": out.cpu()})
        compare_outputs(out, ref_data["out"], **tolerances)

    @pytest.mark.parametrize(
        "sigma_data,sd_name",
        _PER_CHANNEL_SIGMA_CONFIGS,
        ids=[c[1] for c in _PER_CHANNEL_SIGMA_CONFIGS],
    )
    def test_checkpoint_roundtrip(
        self,
        deterministic_settings,
        model_config,
        batch_data,
        device,
        tolerances,
        sigma_data,
        sd_name,
    ):
        """Per-channel sigma_data survives save -> load -> forward."""
        model_cls, shape, arch_name = model_config
        if model_cls == LinearModel:
            pytest.skip("Per-channel sigma_data only applies to ConvModel")

        def create_fn():
            return _make_per_channel_preconditioner(sigma_data, model_cls, shape)

        # Forward before checkpoint
        precond_orig = create_fn().to(device)
        x = batch_data["x"]
        t = batch_data["t"]
        condition = batch_data["condition"]
        out_orig = precond_orig(x, t, condition=condition)

        # Save and reload
        ckpt_file = f"edm_perchan_{sd_name}_{arch_name}.mdlus"
        precond_loaded = load_or_create_checkpoint(ckpt_file, create_fn).to(device)

        # Verify sigma_data buffer survived
        assert torch.allclose(
            precond_loaded.sigma_data.cpu(), precond_orig.sigma_data.cpu()
        )

        out_loaded = precond_loaded(x, t, condition=condition)

        ref_file = f"edm_perchan_{sd_name}_{arch_name}_forward.pth"
        ref_data = load_or_create_reference(ref_file, lambda: {"out": out_orig.cpu()})
        compare_outputs(out_loaded, ref_data["out"], **tolerances)

    @pytest.mark.parametrize(
        "sigma_data,sd_name",
        _PER_CHANNEL_SIGMA_CONFIGS,
        ids=[c[1] for c in _PER_CHANNEL_SIGMA_CONFIGS],
    )
    def test_gradient_flow(
        self,
        model_config,
        batch_data,
        device,
        sigma_data,
        sd_name,
    ):
        """Gradients flow through per-channel EDMPreconditioner."""
        model_cls, shape, _ = model_config
        if model_cls == LinearModel:
            pytest.skip("Per-channel sigma_data only applies to ConvModel")
        precond = _make_per_channel_preconditioner(sigma_data, model_cls, shape).to(
            device
        )

        x = batch_data["x"].clone().requires_grad_(True)
        t = batch_data["t"]
        condition = batch_data["condition"]

        output = precond(x, t, condition=condition)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# =============================================================================
# EDMPreconditioner channel validation tests
# =============================================================================


class TestEDMChannelValidation:
    """Tests for EDMPreconditioner per-channel sigma_data shape validation."""

    def test_channel_mismatch_raises(self, device):
        """Mismatched channel count between sigma_data and x raises ValueError."""
        model = instantiate_model_deterministic(ConvModel, seed=0, channels=3)
        precond = EDMPreconditioner(model, sigma_data=[0.3, 0.5]).to(device)

        x = torch.randn(2, 3, 8, 6, device=device)  # 3 channels
        t = torch.rand(2, device=device)

        with pytest.raises(ValueError, match="2 channels.*3 channels"):
            precond(x, t)

    def test_scalar_sigma_data_no_validation_error(self, device):
        """Scalar sigma_data does not trigger channel validation."""
        model = instantiate_model_deterministic(ConvModel, seed=0, channels=3)
        precond = EDMPreconditioner(model, sigma_data=0.5).to(device)

        x = torch.randn(2, 3, 8, 6, device=device)
        t = torch.rand(2, device=device)
        # Should not raise
        precond(x, t)
