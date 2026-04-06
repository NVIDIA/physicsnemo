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

"""
Autoregressive training utilities with temporal bundling.

Provides dimension-agnostic helpers for slicing time windows, constructing
AR model inputs, and running multi-step rollouts.  Works with both 3D
(B, H, W, T, C) and 4D (B, X, Y, Z, T, C) datasets.

Key concepts
------------
- **L** (input_window):  Number of context timesteps fed to the model.
- **K** (output_window): Number of timesteps the model predicts per step.
- The time axis is always the **second-to-last** dimension of the input
  tensor and the **last** dimension of the target tensor.
- For DeepONet models, explicit **target_times** (trunk query coordinates)
  are extracted from the full input tensor and passed to the model so that
  K can differ from L (temporal bundling).

Three-stage training
--------------------
**Stage 1 -- Teacher Forcing** (``teacher_forcing_step``):
    Sweeps sequentially through the full trajectory starting at t=0.
    Each window [t, t+L) predicts [t+L, t+L+K).  The model always
    receives ground-truth input.  For TNO, Branch2 also receives
    GT solution.  Loss is averaged over all windows.

**Stage 2 -- Pushforward** (``pushforward_step``):
    Chains multiple forward passes with *live* gradients (no detach).
    The number of chained steps ramps via a linear curriculum from 1
    to ``max_unroll``.  This bridges the gap between teacher forcing
    and free-running rollout.

**Stage 3 -- Rollout** (``rollout_step``):
    Sweeps sequentially through the full trajectory starting at t=0.
    For TNO, Branch2 receives the model's own (detached) prediction
    from the previous step instead of ground truth.  This trains the
    model to handle its own approximation errors.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# ---------------------------------------------------------------------------
# Time-axis helpers
# ---------------------------------------------------------------------------


def _time_axis_input(x: Tensor) -> int:
    """Time axis index for input ``(..., T, C)``."""
    return x.dim() - 2


def _time_axis_target(y: Tensor) -> int:
    """Time axis index for target ``(..., T)``."""
    return y.dim() - 1


# ---------------------------------------------------------------------------
# Time-window slicing
# ---------------------------------------------------------------------------


def slice_input_window(inputs: Tensor, t0: int, width: int) -> Tensor:
    """Extract ``(B, *spatial, width, C)`` from full-trajectory input."""
    return inputs.narrow(_time_axis_input(inputs), t0, width)


def slice_target_window(targets: Tensor, t0: int, width: int) -> Tensor:
    """Extract ``(B, *spatial, width)`` from full-trajectory target."""
    return targets.narrow(_time_axis_target(targets), t0, width)


# ---------------------------------------------------------------------------
# Feedback channel injection & noise
# ---------------------------------------------------------------------------


def inject_feedback_channel(
    x_window: Tensor,
    feedback: Optional[Tensor],
) -> Tensor:
    """Append feedback prediction(s) as extra channel(s) to the input window.

    Parameters
    ----------
    x_window : Tensor
        Model input ``(B, *spatial, T, C)``.
    feedback : Tensor or None
        Previous prediction.  Shape ``(B, *spatial, T)`` for single-output
        or ``(B, *spatial, T, C_out)`` for multi-output.  If *None*,
        *x_window* is returned unchanged.
    """
    if feedback is None:
        return x_window
    if feedback.dim() < x_window.dim():
        feedback = feedback.unsqueeze(-1)
    return torch.cat([x_window, feedback], dim=-1)


def add_noise(tensor: Tensor, noise_std: float) -> Tensor:
    """Add Gaussian noise.  No-op when *noise_std* <= 0."""
    if noise_std <= 0:
        return tensor
    return tensor + torch.randn_like(tensor) * noise_std


# ---------------------------------------------------------------------------
# Target-time coordinate extraction
# ---------------------------------------------------------------------------


def extract_target_times(inputs: Tensor, t_start: int, K: int) -> Tensor:
    """Extract K target time coordinates from the full input tensor.

    The time coordinate is assumed to be the **last channel** (index -1)
    of the input tensor at a fixed spatial location ``[0, 0, ..., 0]``.

    Parameters
    ----------
    inputs : Tensor
        Full-trajectory input ``(B, *spatial, T, C)``.
    t_start : int
        First target timestep index.
    K : int
        Number of target timesteps.

    Returns
    -------
    Tensor
        Shape ``(K,)`` -- the time coordinate values for the K target steps.
    """
    ndim = inputs.dim()
    spatial_zeros = (0,) * (ndim - 3)
    idx = (0,) + spatial_zeros
    return inputs[idx][t_start : t_start + K, -1]


# ---------------------------------------------------------------------------
# Model call helpers
# ---------------------------------------------------------------------------


def _call_model(
    model,
    x_window: Tensor,
    target_times: Optional[Tensor],
    use_checkpointing: bool = False,
    x_branch2: Optional[Tensor] = None,
) -> Tensor:
    """Call the model, optionally passing target_times and x_branch2."""
    kwargs = {}
    if target_times is not None and _model_accepts_target_times(model):
        kwargs["target_times"] = target_times
    if x_branch2 is not None and _model_accepts_x_branch2(model):
        kwargs["x_branch2"] = x_branch2

    if use_checkpointing and model.training:
        return grad_checkpoint(
            _forward_with_kwargs,
            model,
            x_window,
            kwargs,
            use_reentrant=False,
        )
    return model(x_window, **kwargs)


def _forward_with_kwargs(model, x_window, kwargs):
    """Thin wrapper so ``grad_checkpoint`` can pass kwargs."""
    return model(x_window, **kwargs)


def _model_accepts_target_times(model) -> bool:
    """Check if the model's forward() accepts a ``target_times`` kwarg."""
    import inspect

    m = model.module if hasattr(model, "module") else model
    sig = inspect.signature(m.forward)
    return "target_times" in sig.parameters


def _model_accepts_x_branch2(model) -> bool:
    """Check if the model's forward() accepts an ``x_branch2`` kwarg."""
    import inspect

    m = model.module if hasattr(model, "module") else model
    sig = inspect.signature(m.forward)
    return "x_branch2" in sig.parameters


# ---------------------------------------------------------------------------
# Window iteration
# ---------------------------------------------------------------------------


def _iter_windows(total_T: int, L: int, K: int, stride: Optional[int] = None):
    """Yield ``(t0, target_start, actual_K)`` for each window.

    Parameters
    ----------
    total_T : int
        Total number of timesteps in the trajectory.
    L : int
        Input context length (number of input timesteps).
    K : int
        Target prediction length (number of output timesteps).
    stride : int or None
        Advance between consecutive windows.  Defaults to *K*
        (non-overlapping).  Yields truncated windows when the
        remaining timesteps are fewer than *K*.
    """
    if stride is None:
        stride = K
    t = 0
    while t + L < total_T:
        target_start = t + L
        remaining = total_T - target_start
        actual_K = min(K, remaining)
        if actual_K <= 0:
            break
        yield (t, target_start, actual_K)
        t += stride


# ---------------------------------------------------------------------------
# Branch-2 builder (TNO previous-solution input)
# ---------------------------------------------------------------------------


def _build_branch2(
    targets: Tensor,
    prev_pred: Optional[Tensor],
    current_t: int,
    L: int,
    t_ax: int,
    is_tno: bool,
    noise_std: float = 0.0,
) -> Optional[Tensor]:
    """Build the branch-2 tensor for TNO models.

    Returns *None* for non-TNO models.  For TNO:
    - First window (``prev_pred is None``): ground-truth at ``[current_t, current_t+L)``.
    - Subsequent windows: last *L* timesteps of ``prev_pred``, padded with GT if needed.
    """
    if not is_tno:
        return None
    if prev_pred is None:
        b2 = slice_target_window(targets, current_t, L)
    elif prev_pred.shape[t_ax] >= L:
        b2 = prev_pred.narrow(t_ax, prev_pred.shape[t_ax] - L, L)
    else:
        need = L - prev_pred.shape[t_ax]
        gt_part = slice_target_window(targets, current_t, need)
        b2 = torch.cat([gt_part, prev_pred], dim=t_ax)
    if noise_std > 0:
        b2 = add_noise(b2, noise_std)
    return b2


# ---------------------------------------------------------------------------
# Curriculum scheduling
# ---------------------------------------------------------------------------


def compute_unroll_steps(
    epoch: int,
    start_epoch: int,
    total_epochs: int,
    max_unroll: int,
) -> int:
    """Linear curriculum: ramp unroll steps from 1 to *max_unroll*.

    Returns 1 at *start_epoch*, *max_unroll* at ``start_epoch + total_epochs``,
    and clamps outside that range.
    """
    if total_epochs <= 0:
        return max_unroll
    progress = (epoch - start_epoch) / total_epochs
    progress = max(0.0, min(1.0, progress))
    return max(1, round(1 + (max_unroll - 1) * progress))


def get_training_stage(
    epoch: int,
    tf_epochs: int,
    pf_epochs: int = 0,
    ro_epochs: int = 0,
) -> str:
    """Return the training stage name for a given epoch.

    Stages (in order): ``"teacher_forcing"`` -> ``"pushforward"`` -> ``"rollout"``.
    Stages with zero epochs are skipped.  After all stages are exhausted the
    last active stage is returned.
    """
    if epoch < tf_epochs:
        return "teacher_forcing"
    if pf_epochs > 0 and epoch < tf_epochs + pf_epochs:
        return "pushforward"
    if ro_epochs > 0:
        return "rollout"
    if pf_epochs > 0:
        return "pushforward"
    return "teacher_forcing"


# ---------------------------------------------------------------------------
# Feedback helper (shared by training steps)
# ---------------------------------------------------------------------------


def _get_feedback(
    targets: Tensor,
    prev_pred: Optional[Tensor],
    current_t: int,
    L: int,
    t_ax: int,
) -> Tensor:
    """Return feedback tensor (*L* timesteps) for feedback-channel injection."""
    if prev_pred is None:
        return slice_target_window(targets, current_t, L)
    if prev_pred.shape[t_ax] >= L:
        return prev_pred.narrow(t_ax, prev_pred.shape[t_ax] - L, L)
    need = L - prev_pred.shape[t_ax]
    gt_part = slice_target_window(targets, current_t, need)
    return torch.cat([gt_part, prev_pred], dim=t_ax)


# ---------------------------------------------------------------------------
# Teacher-forcing training step (one batch) -- sequential sweep
# ---------------------------------------------------------------------------


def teacher_forcing_step(
    model,
    inputs: Tensor,
    targets: Tensor,
    loss_fn,
    L: int,
    K: int,
    spatial_mask: Optional[Tensor] = None,
    is_tno: bool = False,
    noise_std: float = 0.0,
    feedback_channel: Optional[int] = None,
    stride: Optional[int] = None,
) -> Tensor:
    """One teacher-forcing training iteration over a batch.

    Sweeps sequentially from t=0 through the full trajectory, processing
    every window.  Uses gradient accumulation: each window is forwarded and
    backwarded independently (one graph at a time).  Returns a detached
    scalar loss for logging.  The caller should NOT call ``loss.backward()``
    -- only ``optimizer.step()``.

    For TNO, Branch2 receives the ground-truth solution at ``[t, t+L)``.
    """
    total_T = targets.shape[_time_axis_target(targets)]
    t_ax = _time_axis_target(targets)

    effective_stride = stride if stride is not None else K
    if total_T <= L:
        return torch.tensor(0.0, device=inputs.device)
    num_windows = (total_T - L - K) // effective_stride + 1
    if num_windows <= 0:
        return torch.tensor(0.0, device=inputs.device)

    accumulated_loss = 0.0
    current_t = 0

    for _ in range(num_windows):
        target_start = current_t + L
        remaining = total_T - target_start
        actual_K = min(K, remaining)
        if actual_K <= 0:
            break

        x_window = slice_input_window(inputs, current_t, L)
        y_target = slice_target_window(targets, target_start, actual_K)
        target_times = extract_target_times(inputs, target_start, actual_K)

        y_branch2 = _build_branch2(
            targets,
            None,
            current_t,
            L,
            t_ax,
            is_tno,
            noise_std,
        )

        if feedback_channel is not None:
            fb = slice_target_window(targets, current_t, L)
            fb = add_noise(fb, noise_std)
            x_window = inject_feedback_channel(x_window, fb)

        pred = _call_model(model, x_window, target_times, x_branch2=y_branch2)

        if pred.shape[t_ax] > actual_K:
            pred = pred.narrow(t_ax, 0, actual_K)

        window_loss = loss_fn(pred, y_target, x_window, spatial_mask=spatial_mask)
        if window_loss.requires_grad:
            (window_loss / num_windows).backward()
        accumulated_loss += window_loss.detach().item()

        current_t += effective_stride

    return torch.tensor(accumulated_loss / num_windows, device=inputs.device)


# ---------------------------------------------------------------------------
# Pushforward training step -- live gradients through unrolled chain
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Rollout training step (one batch) -- sequential chain from t=0
# ---------------------------------------------------------------------------


def rollout_step(
    model,
    inputs: Tensor,
    targets: Tensor,
    loss_fn,
    L: int,
    K: int,
    use_checkpointing: bool = True,
    spatial_mask: Optional[Tensor] = None,
    is_tno: bool = False,
    noise_std: float = 0.0,
    feedback_channel: Optional[int] = None,
    stride: Optional[int] = None,
) -> Tensor:
    """One rollout (free-running) training iteration.

    Sweeps sequentially from t=0 through the full trajectory.  Uses
    gradient accumulation: each window is forwarded and backwarded
    independently.  Returns a detached scalar loss for logging.
    The caller should NOT call ``loss.backward()`` -- only ``optimizer.step()``.

    For TNO, Branch2 receives the model's own (detached) prediction
    from the previous step, creating true autoregressive feedback.
    """
    total_T = targets.shape[_time_axis_target(targets)]
    t_ax = _time_axis_target(targets)

    effective_stride = stride if stride is not None else K
    if total_T <= L:
        return torch.tensor(0.0, device=inputs.device)
    num_windows = (total_T - L - K) // effective_stride + 1
    if num_windows <= 0:
        return torch.tensor(0.0, device=inputs.device)

    accumulated_loss = 0.0
    prev_pred = None
    current_t = 0

    for _ in range(num_windows):
        target_start = current_t + L
        remaining = total_T - target_start
        actual_K = min(K, remaining)
        if actual_K <= 0:
            break

        x_window = slice_input_window(inputs, current_t, L)
        y_target = slice_target_window(targets, target_start, actual_K)
        target_times = extract_target_times(inputs, target_start, actual_K)

        y_branch2 = _build_branch2(
            targets,
            prev_pred,
            current_t,
            L,
            t_ax,
            is_tno,
            noise_std,
        )

        if feedback_channel is not None:
            fb = _get_feedback(targets, prev_pred, current_t, L, t_ax)
            fb = add_noise(fb, noise_std)
            x_window = inject_feedback_channel(x_window, fb)

        pred = _call_model(
            model,
            x_window,
            target_times,
            use_checkpointing,
            x_branch2=y_branch2,
        )

        if pred.shape[t_ax] > actual_K:
            pred = pred.narrow(t_ax, 0, actual_K)

        window_loss = loss_fn(pred, y_target, x_window, spatial_mask=spatial_mask)
        if window_loss.requires_grad:
            (window_loss / num_windows).backward()
        accumulated_loss += window_loss.detach().item()

        prev_pred = pred.detach()
        current_t += effective_stride

    return torch.tensor(accumulated_loss / num_windows, device=inputs.device)


# ---------------------------------------------------------------------------
# Full-trajectory rollout with live gradients (matches original TNO training)
# ---------------------------------------------------------------------------


def _freeze_batchnorm(model):
    """Set all BatchNorm layers to eval mode (freeze running stats).

    Returns a list of the modules that were switched so they can be
    restored afterwards.  The learned gamma/beta parameters still
    receive gradients; only the inplace running-stat updates are
    suppressed.
    """
    switched = []
    m = model.module if hasattr(model, "module") else model
    for mod in m.modules():
        if isinstance(
            mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
        ):
            if mod.training:
                mod.eval()
                switched.append(mod)
    return switched


def _unfreeze_batchnorm(switched):
    """Restore previously-frozen BatchNorm layers to training mode."""
    for mod in switched:
        mod.train()


def live_rollout_step(
    model,
    inputs: Tensor,
    targets: Tensor,
    loss_fn,
    L: int,
    K: int,
    max_steps: Optional[int] = None,
    spatial_mask: Optional[Tensor] = None,
    is_tno: bool = False,
    noise_std: float = 0.0,
    feedback_channel: Optional[int] = None,
    stride: Optional[int] = None,
) -> Tensor:
    """Rollout with live gradients through an unrolled prediction chain.

    Collects predictions into a single tensor, computes loss once on the
    concatenated trajectory, then calls ``.backward()``.  Returns a
    detached scalar loss for logging.  The caller should NOT call
    ``loss.backward()`` -- only ``optimizer.step()``.

    Gradients flow from the final loss through all intermediate
    predictions, providing strong gradient signal for learning to handle
    error accumulation.

    BatchNorm layers are temporarily set to eval mode during the forward
    chain to prevent inplace running-stat updates from invalidating the
    autograd graph.  The learned affine parameters (gamma, beta) still
    receive gradients normally.

    Parameters
    ----------
    max_steps : int or None
        Maximum number of autoregressive windows to chain.  ``None``
        (default) chains all windows in the trajectory.  Set to a small
        value (e.g. 1-5) for pushforward-style curriculum training.
    """
    total_T = targets.shape[_time_axis_target(targets)]
    t_ax = _time_axis_target(targets)

    if total_T <= L:
        return torch.tensor(0.0, device=inputs.device)

    # Freeze BatchNorm running stats to avoid inplace buffer updates
    # that would invalidate the live autograd graph across chained forwards.
    frozen_bn = _freeze_batchnorm(model)

    pred_slices = []
    prev_pred = None
    current_t = 0
    step_count = 0

    while current_t + L < total_T:
        target_start = current_t + L
        remaining = total_T - target_start
        actual_K = min(K, remaining)
        if actual_K <= 0:
            break

        x_window = slice_input_window(inputs, current_t, L)
        target_times = extract_target_times(inputs, target_start, actual_K)

        y_branch2 = _build_branch2(
            targets,
            prev_pred,
            current_t,
            L,
            t_ax,
            is_tno,
            noise_std,
        )

        if feedback_channel is not None:
            fb = _get_feedback(targets, prev_pred, current_t, L, t_ax)
            fb = add_noise(fb, noise_std)
            x_window = inject_feedback_channel(x_window, fb)

        pred = _call_model(model, x_window, target_times, x_branch2=y_branch2)

        if pred.shape[t_ax] > actual_K:
            pred = pred.narrow(t_ax, 0, actual_K)

        pred_slices.append(pred)
        prev_pred = pred
        current_t += stride if stride is not None else K
        step_count += 1
        if max_steps is not None and step_count >= max_steps:
            break

    if not pred_slices:
        _unfreeze_batchnorm(frozen_bn)
        return torch.tensor(0.0, device=inputs.device)

    pred_full = torch.cat(pred_slices, dim=t_ax)
    target_full = slice_target_window(targets, L, pred_full.shape[t_ax])

    loss = loss_fn(pred_full, target_full, inputs, spatial_mask=spatial_mask)
    if loss.requires_grad:
        loss.backward()

    _unfreeze_batchnorm(frozen_bn)
    return loss.detach()


# ---------------------------------------------------------------------------
# Full autoregressive validation (all timesteps)
# ---------------------------------------------------------------------------


@torch.no_grad()
def ar_validate_full_rollout(
    model,
    inputs: Tensor,
    targets: Tensor,
    L: int,
    K: int,
    is_tno: bool = False,
    feedback_channel: Optional[int] = None,
) -> Tensor:
    """Run a complete AR rollout over the full trajectory for validation.

    Always starts at t=0 and rolls out until all timesteps are covered.
    Returns the full predicted trajectory (same shape as ``targets``).
    """
    total_T = targets.shape[_time_axis_target(targets)]
    t_ax = _time_axis_target(targets)

    pred_slices = []
    prev_pred = None
    current_t = 0

    while current_t + L < total_T:
        target_start = current_t + L
        remaining = total_T - target_start
        actual_K = min(K, remaining)
        if actual_K <= 0:
            break

        x_window = slice_input_window(inputs, current_t, L)
        target_times = extract_target_times(inputs, target_start, actual_K)

        if is_tno:
            if prev_pred is None:
                y_branch2 = slice_target_window(targets, current_t, L)
            elif prev_pred.shape[t_ax] >= L:
                y_branch2 = prev_pred.narrow(
                    t_ax,
                    prev_pred.shape[t_ax] - L,
                    L,
                )
            else:
                need = L - prev_pred.shape[t_ax]
                gt_part = slice_target_window(targets, current_t, need)
                y_branch2 = torch.cat([gt_part, prev_pred], dim=t_ax)
        else:
            y_branch2 = None

        if feedback_channel is not None:
            fb = _get_feedback(targets, prev_pred, current_t, L, t_ax)
            x_window = inject_feedback_channel(x_window, fb)

        pred = _call_model(model, x_window, target_times, x_branch2=y_branch2)

        pred_t_ax = _time_axis_target(pred)
        if pred.shape[pred_t_ax] > actual_K:
            pred = pred.narrow(pred_t_ax, 0, actual_K)

        pred_slices.append(pred)
        prev_pred = pred
        current_t += K

    if not pred_slices:
        return torch.zeros_like(targets)

    pred_full = torch.cat(pred_slices, dim=t_ax)

    gt_prefix = slice_target_window(targets, 0, L)
    pred_full = torch.cat([gt_prefix, pred_full], dim=t_ax)

    if pred_full.shape[t_ax] > total_T:
        pred_full = pred_full.narrow(t_ax, 0, total_T)
    elif pred_full.shape[t_ax] < total_T:
        deficit = total_T - pred_full.shape[t_ax]
        pad_slice = slice_target_window(
            targets,
            pred_full.shape[t_ax],
            deficit,
        )
        pred_full = torch.cat([pred_full, pad_slice], dim=t_ax)

    return pred_full
