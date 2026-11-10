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
"""DeFormer: a frozen time-conditional network followed by its corrector.

The teacher is a finished :class:`time_conditional.MeshGeoFLARETimeConditional`,
loaded from its checkpoint and frozen; the student is a
:class:`mgn_corrector.MeshGraphNetCorrector` trained on top of it. The teacher's
prediction at the queried time is recomputed on the fly and handed to the
student, which adds its correction; the trainings are decoupled (the teacher
never receives a gradient) and no prediction files are stored. Inference is
the teacher followed by the student.

Options:

* ``teacher_window`` also queries the teacher one frame earlier and one frame
  later and hands the student that motion;
* a training sample whose target holds two consecutive frames
  (``datapipe.frame_pairs``) is served frame by frame, the teacher frames the
  pair shares computed once, and the predictions come back stacked as
  ``[N, 2, 3]`` for a loss on the change between the frames
  (``training.increment_loss_weight``);
* ``final_clamp_to_wall`` and ``final_projection_*`` apply the evaluation-time
  geometric corrections of :mod:`shape_corrections` to the corrected shape
  (the teacher's own prediction stays as the student was trained on).
"""

from __future__ import annotations

from pathlib import Path

import torch
from mgn_corrector import MeshGraphNetCorrector
from shape_corrections import PenetrationProjection, RigidWallClamp, correct_shape
from time_conditional import TimeConditionalOutput
from torch import nn
from utils import load_warm_start

import physicsnemo


class ResidualTimeConditional(physicsnemo.Module):
    """Frozen teacher followed by the MeshGraphNet corrector.

    Parameters
    ----------
    teacher : nn.Module
        The teacher's architecture (an instantiated, randomly initialised
        ``MeshGeoFLARETimeConditional`` of the teacher's recipe); its weights
        come from ``teacher_weights``.
    teacher_weights : str
        ``.mdlus`` (or state-dict ``.pt``) file of the teacher, loaded strictly.
    teacher_window : bool
        Also serve the teacher's displacement one frame before and one frame
        after the queried time to the student (``[N, 2, 3]``). The frame before
        the first is the reference state (zero displacement); the frame after
        the last time label the teacher was trained for repeats the last
        prediction. Two more teacher passes per training step.
    final_clamp_to_wall, wall_axis, wall_sign
        Evaluation-time rigid-wall clamp of the corrected shape
        (:class:`shape_corrections.RigidWallClamp`); needs
        ``datapipe.rigid_wall_position``.
    final_projection_iterations, final_projection_damping,
    final_projection_side_symmetric, final_projection_crossed_only,
    final_projection_tolerance_mm, final_projection_band_mm,
    final_projection_clearance_mm
        Evaluation-time envelope projection of the corrected shape
        (:class:`shape_corrections.PenetrationProjection`; 0 iterations = off).
        Its search backend and pair budget are the student's
        ``contact_search_implementation`` and ``contact_surface_max_pairs``.
    **student_kwargs
        Constructor arguments of :class:`mgn_corrector.MeshGraphNetCorrector`;
        its ``functional_dim`` counts the three teacher-displacement channels
        (and the six window channels with ``teacher_window``).

    A saved corrector (``.mdlus``) records ``teacher_weights`` as given at
    training time. Reload it elsewhere with
    ``physicsnemo.Module.from_checkpoint(path, override_args={"teacher_weights": <network .mdlus>})``;
    the teacher's weights are part of the corrector's state dict, the file is
    only needed to build the teacher.
    """

    # The teacher path is the one argument a saved corrector lets the loader override.
    _overridable_args = {"teacher_weights"}

    def __init__(
        self,
        *,
        teacher: nn.Module,
        teacher_weights,
        teacher_window: bool = False,
        final_clamp_to_wall: bool = False,
        wall_axis: int = 0,
        wall_sign: float = 1.0,
        final_projection_iterations: int = 0,
        final_projection_damping: float = 0.5,
        final_projection_side_symmetric: bool = True,
        final_projection_crossed_only: bool = False,
        final_projection_tolerance_mm: float = 0.0,
        final_projection_band_mm: float = 5.0,
        final_projection_clearance_mm: float | None = 10.0,
        **student_kwargs,
    ) -> None:
        super().__init__()
        self.projection = PenetrationProjection(
            iterations=final_projection_iterations,
            tolerance_mm=final_projection_tolerance_mm,
            damping=final_projection_damping,
            band_mm=final_projection_band_mm,
            clearance_mm=final_projection_clearance_mm,
            crossed_only=final_projection_crossed_only,
            side_symmetric=final_projection_side_symmetric,
            implementation=student_kwargs.get("contact_search_implementation", "warp"),
            max_pairs=int(student_kwargs.get("contact_surface_max_pairs", 8_000_000)),
        )
        self.wall_clamp = RigidWallClamp(
            enabled=final_clamp_to_wall, axis=wall_axis, sign=wall_sign
        )
        if not teacher_weights:
            raise ValueError("teacher_weights must name the teacher's checkpoint")
        if not Path(str(teacher_weights)).is_file():
            raise FileNotFoundError(
                f"teacher checkpoint {teacher_weights} does not exist; point "
                "model.teacher_weights at the network's .mdlus file, or pass "
                'override_args={"teacher_weights": ...} to Module.from_checkpoint'
            )
        load_warm_start(teacher, teacher_weights)
        teacher.requires_grad_(False)
        teacher.eval()
        self.teacher = teacher
        self.teacher_weights = str(teacher_weights)
        self.teacher_window = bool(teacher_window)
        if self.teacher_window:
            student_kwargs["teacher_window"] = True
        self.student = MeshGraphNetCorrector(**student_kwargs)
        # Read by recipes and readouts that know the time-conditional interface.
        self.rollout_steps = self.student.rollout_steps
        self.dt = self.student.dt
        self.__name__ = "ResidualTimeConditional"

    def train(self, mode: bool = True):
        """The student follows the mode; the teacher stays in evaluation mode."""
        super().train(mode)
        self.teacher.eval()
        return self

    # ----- teacher fields -----
    def _frame(self, sample, data_stats, label: float) -> torch.Tensor:
        """The teacher's normalized positions at time label ``label``."""
        features = sample.node_features
        original = features.get("time")
        features["time"] = torch.tensor(label, device=features["coords"].device)
        try:
            return self.teacher.predict_frame(sample, data_stats).prediction.detach()
        finally:
            if original is None:
                features.pop("time", None)
            else:
                features["time"] = original

    def _grid_frame(self, sample) -> int:
        """The future-frame index of the sample's time label, which must sit on
        the grid ``k / rollout_steps`` (the datapipe's ``one_time_step`` labels do)."""
        steps = int(self.teacher.rollout_steps)
        label = float(sample.node_features["time"])
        k = int(round(label * steps))
        if abs(label * steps - k) > 1e-3 or not 0 <= k < steps:
            raise ValueError(
                "teacher_window and frame pairs need the time label on the frame grid "
                "k / rollout_steps"
            )
        return k

    def _training_fields(self, sample, data_stats, k: int, cache: dict):
        """Teacher positions at future frame ``k`` and, with ``teacher_window``,
        the frames around it (``None`` otherwise). ``cache`` maps frame indices
        to teacher positions, so the frames a pair shares are computed once."""
        coords = sample.node_features["coords"]
        steps = int(self.teacher.rollout_steps)

        def frame(j):
            if j < 0:
                return coords
            if j not in cache:
                cache[j] = self._frame(sample, data_stats, j / steps).to(coords.dtype)
            return cache[j]

        current = frame(k)
        if not self.teacher_window:
            return current, None
        following = frame(k + 1) if k + 1 < steps else current
        return current, torch.stack((frame(k - 1), following), dim=1)

    @torch.no_grad()
    def teacher_trajectory(self, sample, data_stats):
        """Evaluation: the teacher's normalized positions ``[N, T, 3]`` over the
        queried horizon and, with ``teacher_window``, the frames around each
        (``[N, T, 2, 3]``; otherwise ``None``)."""
        coords = sample.node_features["coords"]
        predicted = self.teacher(sample, data_stats).detach().to(coords.dtype)
        if not self.teacher_window:
            return predicted, None
        steps = int(self.teacher.rollout_steps)
        queried = int(predicted.shape[1])
        previous = torch.cat((coords.unsqueeze(1), predicted[:, :-1]), dim=1)
        last = (
            self._frame(sample, data_stats, queried / steps)
            .to(coords.dtype)
            .unsqueeze(1)
            if queried < steps
            else predicted[:, -1:]
        )
        following = torch.cat((predicted[:, 1:], last), dim=1)
        return predicted, torch.stack((previous, following), dim=2)

    # ----- student -----
    def _correct(self, sample, data_stats, predicted, window, return_auxiliary):
        """Run the student on one frame's teacher fields."""
        coords = sample.node_features["coords"]
        displacement = (predicted - coords).to(coords.dtype)
        teacher_window = None if window is None else (window - coords.unsqueeze(1))
        return self.student(
            sample,
            data_stats,
            teacher_displacement=displacement,
            teacher_window=teacher_window,
            return_auxiliary=return_auxiliary,
        )

    def forward(self, sample, data_stats, return_auxiliary: bool = False):
        """One corrected frame in training (or a frame pair, ``[N, 2, 3]``), the
        corrected trajectory ``[N, T, 3]`` in evaluation (with the enabled
        geometric corrections applied frame by frame)."""
        features = sample.node_features
        coords = features["coords"]
        if not self.training:
            predicted, window = self.teacher_trajectory(sample, data_stats)
            original = features.get("time")
            outputs = []
            try:
                for k in range(predicted.shape[1]):
                    features["time"] = torch.tensor(
                        k / self.teacher.rollout_steps, device=coords.device
                    )
                    frame = self._correct(
                        sample,
                        data_stats,
                        predicted[:, k],
                        None if window is None else window[:, k],
                        False,
                    )
                    outputs.append(
                        correct_shape(
                            frame,
                            sample,
                            data_stats,
                            projection=self.projection,
                            clamp=self.wall_clamp,
                        )
                    )
            finally:
                if original is None:
                    features.pop("time", None)
                else:
                    features["time"] = original
            return torch.stack(outputs, dim=1)
        # Training: one future frame, or a consecutive pair when the datapipe
        # serves a two-frame target (datapipe.frame_pairs); the time label is the
        # earlier frame's. The student runs once per frame.
        target = getattr(sample, "node_target", None)
        pair = target is not None and target.ndim == 3
        if pair and int(target.shape[1]) != 2:
            raise ValueError(
                "a training target is one frame [N, 3] or a consecutive frame pair "
                f"[N, 2, 3], got {tuple(target.shape)}"
            )
        cache: dict = {}
        if not pair:
            if self.teacher_window:
                k = self._grid_frame(sample)
                with torch.no_grad():
                    predicted, window = self._training_fields(
                        sample, data_stats, k, cache
                    )
            else:
                with torch.no_grad():
                    predicted = self._frame(sample, data_stats, float(features["time"]))
                window = None
            return self._correct(
                sample, data_stats, predicted, window, return_auxiliary
            )
        steps = int(self.teacher.rollout_steps)
        k = self._grid_frame(sample)
        if k + 1 >= steps:
            raise ValueError(
                "a frame pair's time label is the earlier frame's: k + 1 must be a "
                "trained future frame"
            )
        original_time = features["time"]
        outputs = []
        try:
            for j in (k, k + 1):
                features["time"] = torch.tensor(
                    j / steps, dtype=torch.float32, device=coords.device
                )
                with torch.no_grad():
                    predicted, window = self._training_fields(
                        sample, data_stats, j, cache
                    )
                outputs.append(
                    self._correct(
                        sample, data_stats, predicted, window, return_auxiliary
                    )
                )
        finally:
            features["time"] = original_time
        if not return_auxiliary:
            return torch.stack(outputs, dim=1)  # [N, 2, 3]
        return TimeConditionalOutput(
            torch.stack([out.prediction for out in outputs], dim=1),
            torch.stack([out.coarse_prediction for out in outputs], dim=1),
        )
