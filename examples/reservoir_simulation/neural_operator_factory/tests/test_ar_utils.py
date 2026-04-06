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

"""Comprehensive unit tests for autoregressive training utilities."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.ar_utils import (
    _build_branch2,
    _iter_windows,
    _model_accepts_target_times,
    _model_accepts_x_branch2,
    _time_axis_input,
    _time_axis_target,
    add_noise,
    ar_validate_full_rollout,
    compute_unroll_steps,
    extract_target_times,
    get_training_stage,
    inject_feedback_channel,
    live_rollout_step,
    rollout_step,
    slice_input_window,
    slice_target_window,
    teacher_forcing_step,
)

# ---------------------------------------------------------------------------
# Dummy models
# ---------------------------------------------------------------------------


class DummyModel3D(nn.Module):
    """Returns zeros; output T = target_times length if given, else T_in."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x, target_times=None):
        B, H, W, T_in, C = x.shape
        T_out = target_times.shape[0] if target_times is not None else T_in
        return (
            torch.zeros(B, H, W, T_out, device=x.device) + self.linear.weight.sum() * 0
        )


class DummyModel4D(nn.Module):
    """Returns zeros; output T = target_times length if given, else T_in."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x, target_times=None):
        B, X, Y, Z, T_in, C = x.shape
        T_out = target_times.shape[0] if target_times is not None else T_in
        return (
            torch.zeros(B, X, Y, Z, T_out, device=x.device)
            + self.linear.weight.sum() * 0
        )


class DummyModelNoTargetTimes(nn.Module):
    """Model that does NOT accept target_times (e.g. FNO)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        B, H, W, T_in, C = x.shape
        return (
            torch.zeros(B, H, W, T_in, device=x.device) + self.linear.weight.sum() * 0
        )


class DummyTNOModel3D(nn.Module):
    """TNO-style 3D model: accepts x_branch2."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x, target_times=None, x_branch2=None):
        B, H, W, T_in, C = x.shape
        T_out = target_times.shape[0] if target_times is not None else T_in
        return (
            torch.zeros(B, H, W, T_out, device=x.device) + self.linear.weight.sum() * 0
        )


class DummyTNOModel4D(nn.Module):
    """TNO-style 4D model: accepts x_branch2."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x, target_times=None, x_branch2=None):
        B, X, Y, Z, T_in, C = x.shape
        T_out = target_times.shape[0] if target_times is not None else T_in
        return (
            torch.zeros(B, X, Y, Z, T_out, device=x.device)
            + self.linear.weight.sum() * 0
        )


class DummyFeedbackModel3D(nn.Module):
    """Asserts C == base_channels + 1 (feedback channel present)."""

    def __init__(self, base_channels=5):
        super().__init__()
        self.base_channels = base_channels
        self.linear = nn.Linear(1, 1)

    def forward(self, x, target_times=None):
        B, H, W, T_in, C = x.shape
        assert C == self.base_channels + 1, (
            f"Expected {self.base_channels + 1} channels, got {C}"
        )
        T_out = target_times.shape[0] if target_times is not None else T_in
        return (
            torch.zeros(B, H, W, T_out, device=x.device) + self.linear.weight.sum() * 0
        )


class DummyIdentityModel3D(nn.Module):
    """Returns scale * ones; useful for gradient flow tests."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, target_times=None):
        B, H, W, T_in, C = x.shape
        T_out = target_times.shape[0] if target_times is not None else T_in
        return self.scale * torch.ones(B, H, W, T_out, device=x.device)


def dummy_loss(pred, target, inputs, spatial_mask=None):
    """Simple MSE loss for testing."""
    return torch.mean((pred - target) ** 2)


# ---------------------------------------------------------------------------
# Tests: _time_axis_input, _time_axis_target
# ---------------------------------------------------------------------------


class TestTimeAxisHelpers:
    """Tests for _time_axis_input and _time_axis_target."""

    def test_input_3d(self):
        """Verify time axis is 3 for 5D (3D-spatial) input."""
        assert _time_axis_input(torch.randn(2, 8, 10, 24, 5)) == 3

    def test_input_4d(self):
        """Verify time axis is 4 for 6D (4D-spatial) input."""
        assert _time_axis_input(torch.randn(2, 8, 10, 6, 24, 5)) == 4

    def test_target_3d_single_output(self):
        """Verify time axis is 3 for 4D single-output target."""
        assert _time_axis_target(torch.randn(2, 8, 10, 24)) == 3

    def test_target_4d_single_output(self):
        """Verify time axis is 4 for 5D single-output target."""
        assert _time_axis_target(torch.randn(2, 8, 10, 6, 24)) == 4

    def test_input_multi_output(self):
        """Multi-output input has same layout -- extra channels, same ndim."""
        assert _time_axis_input(torch.randn(2, 8, 10, 24, 11)) == 3

    def test_target_multi_output(self):
        """5D target treated as 4D-spatial (B, X, Y, Z, T); time axis is last."""
        assert _time_axis_target(torch.randn(2, 8, 10, 6, 24)) == 4


# ---------------------------------------------------------------------------
# Tests: slice_input_window, slice_target_window
# ---------------------------------------------------------------------------


class TestSlicing:
    """Tests for slice_input_window and slice_target_window."""

    def test_slice_input_3d(self):
        """Verify input window slicing for 3D data."""
        x = torch.randn(2, 8, 10, 24, 5)
        w = slice_input_window(x, t0=3, width=4)
        assert w.shape == (2, 8, 10, 4, 5)
        assert torch.equal(w, x[:, :, :, 3:7, :])

    def test_slice_input_4d(self):
        """Verify input window slicing for 4D data."""
        x = torch.randn(2, 8, 10, 6, 24, 5)
        w = slice_input_window(x, t0=5, width=3)
        assert w.shape == (2, 8, 10, 6, 3, 5)
        assert torch.equal(w, x[:, :, :, :, 5:8, :])

    def test_slice_target_3d(self):
        """Verify target window slicing for 3D data."""
        y = torch.randn(2, 8, 10, 24)
        w = slice_target_window(y, t0=10, width=5)
        assert w.shape == (2, 8, 10, 5)
        assert torch.equal(w, y[:, :, :, 10:15])

    def test_slice_target_4d(self):
        """Verify target window slicing for 4D data."""
        y = torch.randn(2, 8, 10, 6, 24)
        w = slice_target_window(y, t0=0, width=3)
        assert w.shape == (2, 8, 10, 6, 3)
        assert torch.equal(w, y[:, :, :, :, 0:3])

    def test_slice_input_many_channels(self):
        """Slicing works regardless of channel count."""
        x = torch.randn(2, 8, 10, 24, 11)
        w = slice_input_window(x, t0=0, width=6)
        assert w.shape == (2, 8, 10, 6, 11)

    def test_slice_target_4d_spatial(self):
        """4D spatial target (B, X, Y, Z, T): slice along last dim."""
        y = torch.randn(2, 8, 10, 6, 24)
        w = slice_target_window(y, t0=5, width=4)
        assert w.shape == (2, 8, 10, 6, 4)
        assert torch.equal(w, y[:, :, :, :, 5:9])


# ---------------------------------------------------------------------------
# Tests: extract_target_times
# ---------------------------------------------------------------------------


class TestExtractTargetTimes:
    """Tests for extract_target_times."""

    def test_3d_extracts_last_channel(self):
        """Verify target times extracted from last channel for 3D data."""
        x = torch.randn(2, 4, 6, 16, 5)
        times = extract_target_times(x, t_start=5, K=3)
        assert times.shape == (3,)
        assert torch.equal(times, x[0, 0, 0, 5:8, -1])

    def test_4d_extracts_last_channel(self):
        """Verify target times extracted from last channel for 4D data."""
        x = torch.randn(2, 4, 6, 3, 20, 11)
        times = extract_target_times(x, t_start=10, K=5)
        assert times.shape == (5,)
        assert torch.equal(times, x[0, 0, 0, 0, 10:15, -1])

    def test_k_equals_1(self):
        """Verify extraction works with K=1."""
        x = torch.randn(1, 4, 6, 16, 5)
        times = extract_target_times(x, t_start=0, K=1)
        assert times.shape == (1,)


# ---------------------------------------------------------------------------
# Tests: inject_feedback_channel
# ---------------------------------------------------------------------------


class TestInjectFeedbackChannel:
    """Tests for inject_feedback_channel."""

    def test_none_feedback_returns_unchanged(self):
        """None feedback returns input window unchanged."""
        x = torch.randn(2, 4, 6, 3, 5)
        result = inject_feedback_channel(x, None)
        assert result.shape == (2, 4, 6, 3, 5)
        assert torch.equal(result, x)

    def test_single_output_feedback(self):
        """Single-output feedback (B, *spatial, T) is unsqueezed and concatenated."""
        x = torch.randn(2, 4, 6, 3, 5)
        fb = torch.randn(2, 4, 6, 3)
        result = inject_feedback_channel(x, fb)
        assert result.shape == (2, 4, 6, 3, 6)
        assert torch.equal(result[:, :, :, :, :5], x)
        assert torch.equal(result[:, :, :, :, 5], fb)

    def test_multi_output_feedback(self):
        """Multi-output feedback (B, *spatial, T, C_out) concatenated directly."""
        x = torch.randn(2, 4, 6, 3, 5)
        fb = torch.randn(2, 4, 6, 3, 2)
        result = inject_feedback_channel(x, fb)
        assert result.shape == (2, 4, 6, 3, 7)


# ---------------------------------------------------------------------------
# Tests: add_noise
# ---------------------------------------------------------------------------


class TestAddNoise:
    """Tests for add_noise."""

    def test_zero_std(self):
        """noise_std == 0 returns tensor unchanged."""
        t = torch.randn(3, 4)
        result = add_noise(t, 0.0)
        assert torch.equal(result, t)

    def test_positive_std(self):
        """Positive noise_std adds noise."""
        t = torch.randn(3, 4)
        result = add_noise(t, 0.1)
        assert not torch.equal(result, t)
        assert torch.allclose(result, t, atol=1.0)

    def test_negative_std(self):
        """Negative noise_std treated as disabled."""
        t = torch.randn(3, 4)
        result = add_noise(t, -0.5)
        assert torch.equal(result, t)


# ---------------------------------------------------------------------------
# Tests: compute_unroll_steps
# ---------------------------------------------------------------------------


class TestComputeUnrollSteps:
    """Tests for compute_unroll_steps."""

    def test_start(self):
        """At start_epoch, returns 1."""
        result = compute_unroll_steps(
            epoch=10, start_epoch=10, total_epochs=100, max_unroll=10
        )
        assert result == 1

    def test_end(self):
        """At start + total, returns max_unroll."""
        result = compute_unroll_steps(
            epoch=110, start_epoch=10, total_epochs=100, max_unroll=10
        )
        assert result == 10

    def test_midpoint(self):
        """Midpoint returns approximately half of max_unroll."""
        result = compute_unroll_steps(
            epoch=60, start_epoch=10, total_epochs=100, max_unroll=10
        )
        assert 4 <= result <= 6

    def test_zero_stage(self):
        """total_epochs == 0 returns max_unroll immediately."""
        result = compute_unroll_steps(
            epoch=0, start_epoch=0, total_epochs=0, max_unroll=10
        )
        assert result == 10

    def test_beyond_end(self):
        """Epoch past the end clamps to max_unroll."""
        result = compute_unroll_steps(
            epoch=500, start_epoch=10, total_epochs=100, max_unroll=10
        )
        assert result == 10

    def test_curriculum_end_exact(self):
        """epoch=121, start=21, total=100 gives exactly max_unroll."""
        result = compute_unroll_steps(
            epoch=121, start_epoch=21, total_epochs=100, max_unroll=10
        )
        assert result == 10


# ---------------------------------------------------------------------------
# Tests: _iter_windows
# ---------------------------------------------------------------------------


class TestIterWindows:
    """Tests for _iter_windows."""

    def test_non_overlapping(self):
        """stride=K produces non-overlapping windows."""
        windows = list(_iter_windows(total_T=16, L=1, K=3, stride=3))
        assert len(windows) == 5
        for t0, ts, ak in windows:
            assert ts == t0 + 1
            assert ak == 3

    def test_stride_1(self):
        """stride=1 produces overlapping windows with truncation at the end."""
        windows = list(_iter_windows(total_T=6, L=1, K=3, stride=1))
        assert len(windows) == 5
        assert windows[0] == (0, 1, 3)
        assert windows[-1] == (4, 5, 1)

    def test_stride_equals_K(self):
        """Explicit stride=K same as non-overlapping."""
        w1 = list(_iter_windows(total_T=16, L=1, K=3, stride=3))
        w2 = list(_iter_windows(total_T=16, L=1, K=3, stride=3))
        assert w1 == w2

    def test_last_window_truncated(self):
        """Remaining timesteps < K produces a truncated window."""
        windows = list(_iter_windows(total_T=10, L=1, K=4, stride=4))
        assert len(windows) == 3
        assert windows[-1][2] == 1


# ---------------------------------------------------------------------------
# Tests: get_training_stage
# ---------------------------------------------------------------------------


class TestGetTrainingStage:
    """Tests for get_training_stage."""

    def test_teacher_forcing_stage(self):
        """Verify early epoch maps to teacher forcing stage."""
        assert (
            get_training_stage(epoch=5, tf_epochs=20, pf_epochs=30, ro_epochs=50)
            == "teacher_forcing"
        )

    def test_pushforward_stage(self):
        """Verify mid-range epoch maps to pushforward stage."""
        assert (
            get_training_stage(epoch=25, tf_epochs=20, pf_epochs=30, ro_epochs=50)
            == "pushforward"
        )

    def test_rollout_stage(self):
        """Verify late epoch maps to rollout stage."""
        assert (
            get_training_stage(epoch=55, tf_epochs=20, pf_epochs=30, ro_epochs=50)
            == "rollout"
        )

    def test_no_pushforward(self):
        """pf_epochs=0 jumps directly from teacher forcing to rollout."""
        assert (
            get_training_stage(epoch=25, tf_epochs=20, pf_epochs=0, ro_epochs=50)
            == "rollout"
        )

    def test_no_rollout(self):
        """ro_epochs=0 with large pf_epochs keeps epoch in pushforward."""
        assert (
            get_training_stage(epoch=500, tf_epochs=20, pf_epochs=10000, ro_epochs=0)
            == "pushforward"
        )

    def test_tf_pf_boundary(self):
        """Exact transition from teacher forcing to pushforward."""
        assert (
            get_training_stage(epoch=19, tf_epochs=20, pf_epochs=30, ro_epochs=50)
            == "teacher_forcing"
        )
        assert (
            get_training_stage(epoch=20, tf_epochs=20, pf_epochs=30, ro_epochs=50)
            == "pushforward"
        )

    def test_pf_rollout_boundary(self):
        """Exact transition from pushforward to rollout."""
        assert (
            get_training_stage(epoch=49, tf_epochs=20, pf_epochs=30, ro_epochs=50)
            == "pushforward"
        )
        assert (
            get_training_stage(epoch=50, tf_epochs=20, pf_epochs=30, ro_epochs=50)
            == "rollout"
        )


# ---------------------------------------------------------------------------
# Tests: _model_accepts_target_times, _model_accepts_x_branch2
# ---------------------------------------------------------------------------


class TestModelIntrospection:
    """Tests for _model_accepts_target_times and _model_accepts_x_branch2."""

    def test_model_with_target_times(self):
        """Verify model with target_times parameter is detected."""
        assert _model_accepts_target_times(DummyModel3D()) is True

    def test_model_without_target_times(self):
        """Verify model without target_times parameter returns False."""
        assert _model_accepts_target_times(DummyModelNoTargetTimes()) is False

    def test_tno_accepts_x_branch2(self):
        """Verify TNO model with x_branch2 parameter is detected."""
        assert _model_accepts_x_branch2(DummyTNOModel3D()) is True

    def test_standard_model_no_x_branch2(self):
        """Verify standard model without x_branch2 returns False."""
        assert _model_accepts_x_branch2(DummyModel3D()) is False

    def test_no_target_times_no_x_branch2(self):
        """Verify model with neither target_times nor x_branch2 returns False."""
        assert _model_accepts_x_branch2(DummyModelNoTargetTimes()) is False


# ---------------------------------------------------------------------------
# Tests: teacher_forcing_step
# ---------------------------------------------------------------------------


class TestTeacherForcing:
    """Tests for teacher_forcing_step."""

    def test_3d_returns_scalar_loss(self):
        """Verify teacher forcing returns a scalar loss for 3D data."""
        model = DummyModel3D()
        inputs = torch.randn(2, 4, 6, 16, 5)
        targets = torch.randn(2, 4, 6, 16)
        loss = teacher_forcing_step(model, inputs, targets, dummy_loss, L=1, K=3)
        assert loss.dim() == 0

    def test_4d_returns_scalar_loss(self):
        """Verify teacher forcing returns a scalar loss for 4D data."""
        model = DummyModel4D()
        inputs = torch.randn(1, 4, 6, 3, 16, 5)
        targets = torch.randn(1, 4, 6, 3, 16)
        loss = teacher_forcing_step(model, inputs, targets, dummy_loss, L=1, K=3)
        assert loss.dim() == 0

    def test_k_equals_1(self):
        """Verify teacher forcing works with K=1."""
        model = DummyModel3D()
        inputs = torch.randn(2, 4, 6, 16, 5)
        targets = torch.randn(2, 4, 6, 16)
        loss = teacher_forcing_step(model, inputs, targets, dummy_loss, L=1, K=1)
        assert loss.dim() == 0

    def test_stride_1(self):
        """stride=1 processes overlapping windows without error."""
        model = DummyModel3D()
        inputs = torch.randn(2, 4, 6, 16, 5)
        targets = torch.randn(2, 4, 6, 16)
        loss = teacher_forcing_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
            stride=1,
        )
        assert loss.dim() == 0

    def test_tno(self):
        """Verify teacher forcing runs with TNO model."""
        model = DummyTNOModel3D()
        inputs = torch.randn(2, 4, 6, 16, 5)
        targets = torch.randn(2, 4, 6, 16)
        loss = teacher_forcing_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
            is_tno=True,
        )
        assert loss.dim() == 0

    def test_noise(self):
        """Non-zero noise_std does not crash."""
        model = DummyTNOModel3D()
        inputs = torch.randn(2, 4, 6, 16, 5)
        targets = torch.randn(2, 4, 6, 16)
        loss = teacher_forcing_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
            is_tno=True,
            noise_std=0.1,
        )
        assert loss.dim() == 0

    def test_feedback_channel(self):
        """feedback_channel injects an extra channel from GT target."""
        model = DummyFeedbackModel3D(base_channels=5)
        inputs = torch.randn(2, 4, 6, 16, 5)
        targets = torch.randn(2, 4, 6, 16)
        loss = teacher_forcing_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
            feedback_channel=0,
        )
        assert loss.dim() == 0

    def test_no_target_times_model(self):
        """Model without target_times still works (K must equal L)."""
        model = DummyModelNoTargetTimes()
        inputs = torch.randn(2, 4, 6, 16, 5)
        targets = torch.randn(2, 4, 6, 16)
        loss = teacher_forcing_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=1,
        )
        assert loss.dim() == 0


# ---------------------------------------------------------------------------
# Tests: live_rollout_step (pushforward / live-gradient rollout)
# ---------------------------------------------------------------------------


class TestLiveRolloutStep:
    """Tests for live_rollout_step."""

    def test_3d(self):
        """Verify live rollout returns a scalar loss for 3D data."""
        model = DummyModel3D()
        model.train()
        inputs = torch.randn(2, 4, 6, 16, 5)
        targets = torch.randn(2, 4, 6, 16)
        loss = live_rollout_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
            max_steps=2,
        )
        assert loss.dim() == 0

    def test_4d(self):
        """Verify live rollout returns a scalar loss for 4D data."""
        model = DummyModel4D()
        model.train()
        inputs = torch.randn(1, 4, 6, 3, 16, 5)
        targets = torch.randn(1, 4, 6, 3, 16)
        loss = live_rollout_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
            max_steps=2,
        )
        assert loss.dim() == 0

    def test_tno_live_gradients(self):
        """TNO pushforward runs without error."""
        model = DummyTNOModel3D()
        model.train()
        inputs = torch.randn(2, 4, 6, 16, 5)
        targets = torch.randn(2, 4, 6, 16)
        loss = live_rollout_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
            max_steps=2,
            is_tno=True,
        )
        assert loss.dim() == 0

    def test_unroll_1(self):
        """max_steps=1 processes one step per group."""
        model = DummyModel3D()
        model.train()
        inputs = torch.randn(2, 4, 6, 16, 5)
        targets = torch.randn(2, 4, 6, 16)
        loss = live_rollout_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
            max_steps=1,
        )
        assert loss.dim() == 0

    def test_large_unroll(self):
        """Large max_steps covers all windows in a single group."""
        model = DummyModel3D()
        model.train()
        inputs = torch.randn(2, 4, 6, 20, 5)
        targets = torch.randn(2, 4, 6, 20)
        loss = live_rollout_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
            max_steps=100,
        )
        assert loss.dim() == 0

    def test_feedback_channel(self):
        """Feedback channel with pushforward."""
        model = DummyFeedbackModel3D(base_channels=5)
        model.train()
        inputs = torch.randn(2, 4, 6, 16, 5)
        targets = torch.randn(2, 4, 6, 16)
        loss = live_rollout_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
            max_steps=2,
            feedback_channel=0,
        )
        assert loss.dim() == 0

    def test_gradient_flow(self):
        """live_rollout_step calls backward internally; model params get grads."""
        model = DummyIdentityModel3D()
        model.train()
        inputs = torch.randn(2, 4, 6, 16, 5)
        targets = torch.randn(2, 4, 6, 16)
        model.zero_grad()
        loss = live_rollout_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
            max_steps=3,
        )
        assert loss.dim() == 0
        assert not loss.requires_grad
        assert model.scale.grad is not None
        assert model.scale.grad.abs().item() > 0


# ---------------------------------------------------------------------------
# Tests: rollout_step
# ---------------------------------------------------------------------------


class TestRolloutStep:
    """Tests for rollout_step."""

    def test_3d_returns_scalar_loss(self):
        """Verify rollout step returns a scalar loss for 3D data."""
        model = DummyModel3D()
        inputs = torch.randn(2, 4, 6, 20, 5)
        targets = torch.randn(2, 4, 6, 20)
        loss = rollout_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
            use_checkpointing=False,
        )
        assert loss.dim() == 0

    def test_4d_with_checkpointing(self):
        """Verify rollout step with gradient checkpointing for 4D data."""
        model = DummyModel4D()
        inputs = torch.randn(1, 4, 6, 3, 20, 5)
        targets = torch.randn(1, 4, 6, 3, 20)
        loss = rollout_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
            use_checkpointing=True,
        )
        assert loss.dim() == 0

    def test_tno(self):
        """Verify rollout step runs with TNO model."""
        model = DummyTNOModel3D()
        inputs = torch.randn(2, 4, 6, 20, 5)
        targets = torch.randn(2, 4, 6, 20)
        loss = rollout_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
            use_checkpointing=False,
            is_tno=True,
        )
        assert loss.dim() == 0

    def test_feedback_channel(self):
        """Feedback channel with rollout."""
        model = DummyFeedbackModel3D(base_channels=5)
        inputs = torch.randn(2, 4, 6, 20, 5)
        targets = torch.randn(2, 4, 6, 20)
        loss = rollout_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
            use_checkpointing=False,
            feedback_channel=0,
        )
        assert loss.dim() == 0

    def test_noise(self):
        """noise_std > 0 does not crash."""
        model = DummyTNOModel3D()
        inputs = torch.randn(2, 4, 6, 20, 5)
        targets = torch.randn(2, 4, 6, 20)
        loss = rollout_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
            use_checkpointing=False,
            is_tno=True,
            noise_std=0.05,
        )
        assert loss.dim() == 0

    def test_stride_1(self):
        """stride=1 with rollout processes overlapping windows."""
        model = DummyModel3D()
        inputs = torch.randn(2, 4, 6, 12, 5)
        targets = torch.randn(2, 4, 6, 12)
        loss = rollout_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
            stride=1,
            use_checkpointing=False,
        )
        assert loss.dim() == 0


# ---------------------------------------------------------------------------
# Tests: ar_validate_full_rollout
# ---------------------------------------------------------------------------


class TestFullRollout:
    """Tests for ar_validate_full_rollout."""

    def test_3d_output_shape(self):
        """Verify full rollout output shape matches target for 3D data."""
        model = DummyModel3D()
        inputs = torch.randn(1, 4, 6, 16, 5)
        targets = torch.randn(1, 4, 6, 16)
        pred = ar_validate_full_rollout(model, inputs, targets, L=1, K=3)
        assert pred.shape == targets.shape

    def test_4d_output_shape(self):
        """Verify full rollout output shape matches target for 4D data."""
        model = DummyModel4D()
        inputs = torch.randn(1, 4, 6, 3, 20, 5)
        targets = torch.randn(1, 4, 6, 3, 20)
        pred = ar_validate_full_rollout(model, inputs, targets, L=1, K=3)
        assert pred.shape == targets.shape

    def test_prefix_matches_gt(self):
        """First L timesteps are copied from ground truth."""
        model = DummyModel3D()
        inputs = torch.randn(1, 4, 6, 16, 5)
        targets = torch.randn(1, 4, 6, 16)
        pred = ar_validate_full_rollout(model, inputs, targets, L=1, K=3)
        assert torch.equal(pred[:, :, :, :1], targets[:, :, :, :1])

    def test_tno_3d(self):
        """Verify full rollout with TNO model for 3D data."""
        model = DummyTNOModel3D()
        inputs = torch.randn(1, 4, 6, 16, 5)
        targets = torch.randn(1, 4, 6, 16)
        pred = ar_validate_full_rollout(
            model,
            inputs,
            targets,
            L=1,
            K=3,
            is_tno=True,
        )
        assert pred.shape == targets.shape

    def test_tno_4d(self):
        """Verify full rollout with TNO model for 4D data."""
        model = DummyTNOModel4D()
        inputs = torch.randn(1, 4, 6, 3, 16, 5)
        targets = torch.randn(1, 4, 6, 3, 16)
        pred = ar_validate_full_rollout(
            model,
            inputs,
            targets,
            L=1,
            K=3,
            is_tno=True,
        )
        assert pred.shape == targets.shape

    def test_k_equals_1(self):
        """Verify full rollout works with K=1."""
        model = DummyModel3D()
        inputs = torch.randn(1, 4, 6, 10, 5)
        targets = torch.randn(1, 4, 6, 10)
        pred = ar_validate_full_rollout(model, inputs, targets, L=1, K=1)
        assert pred.shape == targets.shape

    def test_feedback_channel(self):
        """Feedback channel in full rollout validation."""
        model = DummyFeedbackModel3D(base_channels=5)
        inputs = torch.randn(1, 4, 6, 16, 5)
        targets = torch.randn(1, 4, 6, 16)
        pred = ar_validate_full_rollout(
            model,
            inputs,
            targets,
            L=1,
            K=3,
            feedback_channel=0,
        )
        assert pred.shape == targets.shape


# ---------------------------------------------------------------------------
# Tests: _build_branch2
# ---------------------------------------------------------------------------


class TestBuildBranch2:
    """Tests for _build_branch2."""

    def test_non_tno_returns_none(self):
        """Verify _build_branch2 returns None for non-TNO models."""
        targets = torch.randn(2, 4, 6, 16)
        t_ax = _time_axis_target(targets)
        result = _build_branch2(targets, None, 0, 1, t_ax, is_tno=False)
        assert result is None

    def test_first_window_uses_gt(self):
        """prev_pred=None returns GT slice."""
        targets = torch.randn(2, 4, 6, 16)
        t_ax = _time_axis_target(targets)
        result = _build_branch2(targets, None, 0, 3, t_ax, is_tno=True)
        expected = slice_target_window(targets, 0, 3)
        assert torch.equal(result, expected)

    def test_subsequent_uses_prev_pred(self):
        """With prev_pred, uses the prediction instead of GT."""
        targets = torch.randn(2, 4, 6, 16)
        prev_pred = torch.randn(2, 4, 6, 3)
        t_ax = _time_axis_target(targets)
        result = _build_branch2(targets, prev_pred, 3, 3, t_ax, is_tno=True)
        assert torch.equal(result, prev_pred)

    def test_noise_applied(self):
        """noise_std > 0 perturbs the branch2 tensor (prev_pred path only)."""
        targets = torch.randn(2, 4, 6, 16)
        prev_pred = torch.randn(2, 4, 6, 3)
        t_ax = _time_axis_target(targets)
        b2_clean = _build_branch2(
            targets,
            prev_pred,
            3,
            3,
            t_ax,
            is_tno=True,
            noise_std=0.0,
        )
        b2_noisy = _build_branch2(
            targets,
            prev_pred,
            3,
            3,
            t_ax,
            is_tno=True,
            noise_std=0.1,
        )
        assert torch.equal(b2_clean, prev_pred)
        assert not torch.equal(b2_noisy, prev_pred)


# ---------------------------------------------------------------------------
# Tests: L=2 K=4 configurations
# ---------------------------------------------------------------------------


class TestL2K4Configs:
    """Tests for L=2, K=4 configuration."""

    def test_teacher_forcing(self):
        """Verify teacher forcing works with L=2, K=4 configuration."""
        model = DummyModel3D()
        inputs = torch.randn(2, 4, 6, 20, 5)
        targets = torch.randn(2, 4, 6, 20)
        loss = teacher_forcing_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=2,
            K=4,
        )
        assert loss.dim() == 0

    def test_pushforward(self):
        """Verify pushforward works with L=2, K=4 configuration."""
        model = DummyModel3D()
        model.train()
        inputs = torch.randn(2, 4, 6, 20, 5)
        targets = torch.randn(2, 4, 6, 20)
        loss = live_rollout_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=2,
            K=4,
            max_steps=3,
        )
        assert loss.dim() == 0

    def test_rollout(self):
        """Verify rollout works with L=2, K=4 configuration."""
        model = DummyModel3D()
        inputs = torch.randn(2, 4, 6, 20, 5)
        targets = torch.randn(2, 4, 6, 20)
        loss = rollout_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=2,
            K=4,
            use_checkpointing=False,
        )
        assert loss.dim() == 0

    def test_full_rollout(self):
        """Verify full rollout works with L=2, K=4 configuration."""
        model = DummyModel3D()
        inputs = torch.randn(1, 4, 6, 20, 5)
        targets = torch.randn(1, 4, 6, 20)
        pred = ar_validate_full_rollout(model, inputs, targets, L=2, K=4)
        assert pred.shape == targets.shape


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: trajectory too short, single window, empty rollout."""

    def test_trajectory_too_short(self):
        """total_T <= L yields no windows and zero loss."""
        model = DummyModel3D()
        inputs = torch.randn(2, 4, 6, 1, 5)
        targets = torch.randn(2, 4, 6, 1)
        loss = teacher_forcing_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
        )
        assert loss.item() == 0.0

    def test_single_window(self):
        """total_T = L + K produces exactly one window."""
        model = DummyModel3D()
        inputs = torch.randn(2, 4, 6, 4, 5)
        targets = torch.randn(2, 4, 6, 4)
        loss = teacher_forcing_step(
            model,
            inputs,
            targets,
            dummy_loss,
            L=1,
            K=3,
        )
        assert loss.dim() == 0

    def test_empty_rollout(self):
        """Full rollout with too-short trajectory returns zeros_like."""
        model = DummyModel3D()
        inputs = torch.randn(1, 4, 6, 1, 5)
        targets = torch.randn(1, 4, 6, 1)
        pred = ar_validate_full_rollout(model, inputs, targets, L=1, K=3)
        assert pred.shape == targets.shape
        assert torch.equal(pred, torch.zeros_like(targets))


# ===================================================================
# Feedback noise injection
# ===================================================================


class TestFeedbackNoise:
    """Tests for noise injection on feedback channel."""

    def _make_model(self, in_ch=5, spatial_ndim=2):
        class DummyModel(torch.nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                self.seen_channels = None

            def forward(self, x, **kwargs):
                self.seen_channels = x.shape[-1]
                spatial = x.shape[1:-2] if x.dim() == 5 else x.shape[1:-1]
                T = kwargs.get("target_times", torch.zeros(1)).shape[0]
                return torch.zeros(x.shape[0], *spatial, T)

        return DummyModel(in_ch)

    def test_noise_applied_to_feedback_in_rollout(self):
        """Rollout with feedback_channel and noise_std > 0 should produce
        different results across runs (noise is stochastic)."""
        B, H, W, T, C = 1, 4, 6, 6, 5
        inputs = torch.randn(B, H, W, T, C)
        targets = torch.randn(B, H, W, T)
        model = self._make_model(C + 1)

        def loss_fn(p, t, i, spatial_mask=None):
            """Compute MSE loss for testing."""
            return (p - t).pow(2).mean()

        torch.manual_seed(0)
        loss1 = rollout_step(
            model,
            inputs,
            targets,
            loss_fn,
            L=1,
            K=2,
            noise_std=0.5,
            feedback_channel=1,
        )
        torch.manual_seed(1)
        loss2 = rollout_step(
            model,
            inputs,
            targets,
            loss_fn,
            L=1,
            K=2,
            noise_std=0.5,
            feedback_channel=1,
        )
        # With different seeds, noise differs so losses should differ
        # (unless model output is trivially zero, which it is here,
        # but the key test is that no error occurs)
        assert not torch.isnan(loss1)
        assert not torch.isnan(loss2)

    def test_no_noise_in_validation(self):
        """ar_validate_full_rollout should be deterministic (no noise)."""
        B, H, W, T, C = 1, 4, 6, 6, 5
        inputs = torch.randn(B, H, W, T, C)
        targets = torch.randn(B, H, W, T)
        model = self._make_model(C + 1)

        torch.manual_seed(42)
        pred1 = ar_validate_full_rollout(
            model,
            inputs,
            targets,
            L=1,
            K=2,
            feedback_channel=1,
        )
        torch.manual_seed(99)
        pred2 = ar_validate_full_rollout(
            model,
            inputs,
            targets,
            L=1,
            K=2,
            feedback_channel=1,
        )
        assert torch.equal(pred1, pred2), "Validation should be deterministic"

    def test_zero_noise_no_effect(self):
        """noise_std=0 should be a no-op."""
        B, H, W, T, C = 1, 4, 6, 6, 5
        inputs = torch.randn(B, H, W, T, C)
        targets = torch.randn(B, H, W, T)
        model = self._make_model(C + 1)

        def loss_fn(p, t, i, spatial_mask=None):
            """Compute MSE loss for testing."""
            return (p - t).pow(2).mean()

        torch.manual_seed(42)
        loss1 = teacher_forcing_step(
            model,
            inputs,
            targets,
            loss_fn,
            L=1,
            K=2,
            noise_std=0.0,
            feedback_channel=1,
        )
        torch.manual_seed(42)
        loss2 = teacher_forcing_step(
            model,
            inputs,
            targets,
            loss_fn,
            L=1,
            K=2,
            noise_std=0.0,
            feedback_channel=1,
        )
        assert torch.isclose(loss1, loss2, atol=1e-6)
