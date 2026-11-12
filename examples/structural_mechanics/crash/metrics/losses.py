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
"""Position losses of the crash recipes.

The default trainer loss is a plain MSE on per-axis standardized positions,
which weights an error of one millimetre by ``1 / pos_std_axis**2`` and so
counts the same physical error differently along the three axes (Yaris: about
1, 3.6 and 9.6 for x, y and z). :class:`PositionLoss` keeps that default
bit-for-bit (``kind="mse"``, ``space="normalized"``) and adds:

* ``space="physical"``: errors are rescaled to millimetres and divided by one
  common ``scale_mm`` before the loss, so training optimises the quantity the
  validation reports (RMSE in mm) up to a constant. The DeFormer recipes use
  ``mse`` in physical space with a 10 mm scale.
* ``kind="relative_l2"``: per frame, the L2 norm of the position error over all
  nodes and axes divided by the L2 norm of the true displacement from the
  reference coordinates (the Transolver/GeoTransolver training objective,
  scale-free); it needs the reference frame, so ``space="physical"`` and the
  trainer passes ``node_features["coords"]``.
"""

from __future__ import annotations

import torch

LOSS_KINDS = ("mse", "relative_l2")
LOSS_SPACES = ("normalized", "physical")


class PositionLoss:
    """Position loss with a configurable kind and space, reduced to one scalar
    or kept per frame.

    Parameters
    ----------
    kind : {"mse", "relative_l2"}
        Loss kind (``training.loss``).
    space : {"normalized", "physical"}
        Coordinates the error is measured in (``training.loss_space``).
    scale_mm : float
        Length that normalizes physical errors (``training.loss_scale_mm``).
    pos_std : torch.Tensor, optional
        Per-axis position standard deviation in millimetres; required in
        physical space.
    """

    def __init__(
        self,
        kind: str = "mse",
        space: str = "normalized",
        scale_mm: float = 10.0,
        pos_std: torch.Tensor | None = None,
    ) -> None:
        if kind not in LOSS_KINDS:
            raise ValueError(f"training.loss must be one of {LOSS_KINDS}, got {kind!r}")
        if space not in LOSS_SPACES:
            raise ValueError(
                f"training.loss_space must be one of {LOSS_SPACES}, got {space!r}"
            )
        if kind == "relative_l2" and space != "physical":
            raise ValueError(
                "relative_l2 is the norm ratio of the physical displacement: "
                "use loss_space=physical"
            )
        if space == "physical" and pos_std is None:
            raise ValueError("loss_space=physical needs the position std (mm per axis)")
        if scale_mm <= 0:
            raise ValueError("loss_scale_mm must be positive")
        self.kind = kind
        self.space = space
        self.scale_mm = float(scale_mm)
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
        """Squared error per entry in the loss space (``mse`` only)."""
        if self.kind == "relative_l2":
            raise ValueError("relative_l2 has no elementwise form: use per_frame")
        return self.scaled_error(prediction, target).square()

    def relative_l2_per_frame(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        reference: torch.Tensor | None = None,
    ):
        """Relative L2 error per frame: ``||pred - target|| / ||target - reference||``.

        Both norms run over all nodes and output channels of a frame in the loss
        space (millimetres over ``scale_mm``; the ratio is scale-free).
        ``reference`` (``[N, C]``) is the frame the targets are displacements
        from, the undeformed coordinates for a position target; ``None`` means
        the targets are displacements already (increments). The norms are
        accumulated in float64. Returns ``[T]`` for ``[N, T, C]`` inputs and
        ``[1]`` for ``[N, C]``.
        """
        error = self.scaled_error(prediction, target)
        if reference is None:
            motion = self.scaled_error(target, torch.zeros_like(target))
        else:
            ref = reference.float()
            if error.ndim == 3 and ref.ndim == 2:
                ref = ref.unsqueeze(1)
            if ref.shape[0] != target.shape[0] or ref.shape[-1] != target.shape[-1]:
                raise ValueError(
                    "reference must be [N, C] matching the target's nodes and channels"
                )
            motion = self.scaled_error(target, ref.expand_as(target))
        dims = (0, 2) if error.ndim == 3 else None
        if dims is None:
            numerator = error.double().square().sum().sqrt().reshape(1)
            denominator = motion.double().square().sum().sqrt().reshape(1)
        else:
            numerator = error.double().square().sum(dim=dims).sqrt()
            denominator = motion.double().square().sum(dim=dims).sqrt()
        return (numerator / denominator.clamp_min(1e-12)).to(torch.float32)

    def per_frame(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        reference: torch.Tensor | None = None,
    ):
        """Mean loss per frame: ``[T]`` for ``[N, T, C]`` inputs, ``[1]`` for ``[N, C]``."""
        if self.kind == "relative_l2":
            return self.relative_l2_per_frame(prediction, target, reference)
        values = self.elementwise(prediction, target)
        if values.ndim == 3:
            return values.mean(dim=(0, 2))
        return values.mean().reshape(1)

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        reference: torch.Tensor | None = None,
    ):
        """The scalar loss of a prediction against its target."""
        if self.kind == "relative_l2":
            return self.relative_l2_per_frame(prediction, target, reference).mean()
        return self.elementwise(prediction, target).mean()
