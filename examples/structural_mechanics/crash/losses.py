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
"""Position-loss options for the crash recipes.

The default trainer loss is a plain MSE on per-axis standardized positions,
which weights an error of one millimetre by ``1 / pos_std_axis**2`` and so
counts the same physical error very differently along the three axes (Yaris:
about 1, 3.6 and 9.6 for x, y and z). The options here keep that default
bit-for-bit (``kind="mse"``, ``space="normalized"``) and add:

* ``space="physical"``: errors are rescaled to millimetres and divided by one
  common ``scale_mm`` before the loss, so training optimises the quantity the
  validation reports (RMSE in mm) up to a constant;
* ``kind="huber"`` / ``kind="l1"``: robust losses on the same rescaled error
  (Huber's transition ``huber_delta_mm`` is in millimetres, so both require
  ``space="physical"``);
* :class:`FrameWeighter`: per-frame loss normalisation by an exponential
  moving average of the frame's own loss, so every frame of the trajectory
  contributes equally to the gradient instead of the large late-time errors
  dominating;
* :func:`edge_consistency_loss`: the error of the relative position between a
  node and its static kNN neighbours, a local-deformation term that penalises
  wrong folds and strains rather than absolute offsets.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

LOSS_KINDS = ("mse", "huber", "l1")
LOSS_SPACES = ("normalized", "physical")


class PositionLoss:
    """Elementwise position loss, reduced to one scalar or kept per frame."""

    def __init__(
        self,
        kind: str = "mse",
        space: str = "normalized",
        scale_mm: float = 10.0,
        huber_delta_mm: float = 3.0,
        pos_std: torch.Tensor | None = None,
    ) -> None:
        if kind not in LOSS_KINDS:
            raise ValueError(f"training.loss must be one of {LOSS_KINDS}, got {kind!r}")
        if space not in LOSS_SPACES:
            raise ValueError(
                f"training.loss_space must be one of {LOSS_SPACES}, got {space!r}"
            )
        if kind == "huber" and space != "physical":
            raise ValueError("Huber's delta is in millimetres: use loss_space=physical")
        if space == "physical" and pos_std is None:
            raise ValueError("loss_space=physical needs the position std (mm per axis)")
        if scale_mm <= 0 or huber_delta_mm <= 0:
            raise ValueError("loss_scale_mm and huber_delta_mm must be positive")
        self.kind = kind
        self.space = space
        self.scale_mm = float(scale_mm)
        self.huber_delta_mm = float(huber_delta_mm)
        self.pos_std = None if pos_std is None else pos_std.detach().reshape(-1).float()

    def scaled_error(self, prediction: torch.Tensor, target: torch.Tensor):
        """Prediction minus target in the loss space (last dim = xyz)."""
        error = prediction.float() - target.float()
        if self.space == "physical":
            std = self.pos_std.to(error.device)
            if std.numel() != error.shape[-1]:
                raise ValueError("pos_std must have one entry per output channel")
            error = error * (std / self.scale_mm)
        return error

    def elementwise(self, prediction: torch.Tensor, target: torch.Tensor):
        error = self.scaled_error(prediction, target)
        if self.kind == "mse":
            return error.square()
        if self.kind == "l1":
            return error.abs()
        return F.huber_loss(
            error,
            torch.zeros_like(error),
            reduction="none",
            delta=self.huber_delta_mm / self.scale_mm,
        )

    def per_frame(self, prediction: torch.Tensor, target: torch.Tensor):
        """Mean loss per frame: ``[T]`` for ``[N, T, C]`` inputs, ``[1]`` for ``[N, C]``."""
        values = self.elementwise(prediction, target)
        if values.ndim == 3:
            return values.mean(dim=(0, 2))
        return values.mean().reshape(1)

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor):
        return self.elementwise(prediction, target).mean()


class FrameWeighter:
    """Equalise the gradient contribution of trajectory frames.

    Keeps an exponential moving average of the mean loss per frame index and
    weights each frame by ``ema.mean() / ema[frame]`` (detached), so a frame
    whose loss is ten times larger than average counts ten times less, and the
    weights average to one. The EMA is per process; ranks see different samples
    but converge to the same per-frame scale.
    """

    def __init__(self, num_frames: int, momentum: float = 0.99) -> None:
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = float(momentum)
        self.ema = torch.zeros(int(num_frames), dtype=torch.float64)
        self.seen = torch.zeros(int(num_frames), dtype=torch.bool)

    def weights(self, frames: torch.Tensor) -> torch.Tensor:
        """Weights for the given frame indices (``[T]`` long tensor)."""
        if self.seen.any():
            reference = self.ema[self.seen].mean()
            w = torch.where(
                self.seen[frames],
                reference / self.ema[frames].clamp_min(1e-30),
                torch.ones((), dtype=torch.float64),
            )
        else:
            w = torch.ones(frames.numel(), dtype=torch.float64)
        return w

    def update(self, frames: torch.Tensor, losses: torch.Tensor) -> None:
        values = losses.detach().double().cpu()
        for frame, value in zip(frames.tolist(), values.tolist()):
            if not torch.isfinite(torch.tensor(value)):
                continue
            if self.seen[frame]:
                self.ema[frame] = (
                    self.momentum * self.ema[frame] + (1 - self.momentum) * value
                )
            else:
                self.ema[frame] = value
                self.seen[frame] = True

    def apply(self, frames: torch.Tensor, per_frame_loss: torch.Tensor) -> torch.Tensor:
        """Weighted mean of ``per_frame_loss`` (``[T]``) and EMA update."""
        w = self.weights(frames).to(per_frame_loss.device, per_frame_loss.dtype)
        weighted = (w * per_frame_loss).mean()
        self.update(frames, per_frame_loss)
        return weighted

    def state_dict(self) -> dict:
        return {"ema": self.ema.clone(), "seen": self.seen.clone()}

    def load_state_dict(self, state: dict) -> None:
        self.ema = state["ema"].clone().double()
        self.seen = state["seen"].clone().bool()


def frame_indices(sample, target: torch.Tensor, rollout_steps: int) -> torch.Tensor:
    """Frame index per supervised step: ``[T]`` for trajectories, one for a TC sample."""
    if target.ndim == 3:
        return torch.arange(int(target.shape[1]))
    time = (
        sample.node_features.get("time") if hasattr(sample, "node_features") else None
    )
    if time is None:
        raise ValueError(
            "Frame weighting needs node_features['time'] for single-frame samples"
        )
    k = int(round(float(time) * int(rollout_steps)))
    return torch.tensor([min(max(k, 0), int(rollout_steps) - 1)])


def edge_consistency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    knn_index: torch.Tensor,
    knn_mask: torch.Tensor,
    pos_std: torch.Tensor | None,
    scale_mm: float = 10.0,
) -> torch.Tensor:
    """Squared error of neighbour-relative positions over the static kNN slots.

    ``prediction``/``target`` are ``[N, C]`` or ``[N, T, C]`` in normalized
    units; the relative vectors are rescaled to millimetres over ``scale_mm``
    when ``pos_std`` is given. Invalid slots are masked; a rigid translation of
    the prediction relative to the target costs nothing.
    """
    if knn_index.ndim != 2 or knn_mask.shape != knn_index.shape:
        raise ValueError("knn_index and knn_mask must be [N, K] with matching shapes")
    error = prediction.float() - target.float()  # relative error = e_j - e_i
    if pos_std is not None:
        std = pos_std.detach().reshape(-1).float().to(error.device)
        error = error * (std / float(scale_mm))
    gathered = error[knn_index.reshape(-1)].reshape(knn_index.shape + error.shape[1:])
    relative = gathered - error.unsqueeze(1)  # [N, K, C] or [N, K, T, C]
    mask = knn_mask.to(device=error.device, dtype=relative.dtype)
    while mask.ndim < relative.ndim:
        mask = mask.unsqueeze(-1)
    per_slot = relative.numel() // (knn_mask.numel() or 1)  # C or T*C entries per slot
    count = knn_mask.to(error.device).sum().clamp_min(1).to(relative.dtype) * per_slot
    return (relative.square() * mask).sum() / count
