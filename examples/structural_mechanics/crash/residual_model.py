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

The teacher is a finished :class:`time_conditional.DeFormerTimeConditional`,
loaded from its checkpoint and frozen; the student is a
:class:`mgn_corrector.MeshGraphNetCorrector` trained on top of it. The teacher's
prediction at the queried time is recomputed on the fly and handed to the
student, which adds its correction; the trainings are decoupled (the teacher
never receives a gradient) and no prediction files are stored. Inference is
the teacher followed by the student.

``final_clamp_to_wall`` applies the evaluation-time rigid-wall clamp of
:mod:`shape_corrections` to the corrected shape (the teacher's own prediction
stays as the student was trained on).
"""

from __future__ import annotations

from pathlib import Path

import torch
from mgn_corrector import MeshGraphNetCorrector
from shape_corrections import RigidWallClamp, correct_shape
from torch import nn
from utils import load_warm_start

import physicsnemo


class ResidualTimeConditional(physicsnemo.Module):
    """Frozen teacher followed by the MeshGraphNet corrector.

    Parameters
    ----------
    teacher : nn.Module
        The teacher's architecture (an instantiated, randomly initialised
        ``DeFormerTimeConditional`` of the teacher's recipe); its weights
        come from ``teacher_weights``.
    teacher_weights : str
        ``.mdlus`` (or state-dict ``.pt``) file of the teacher, loaded strictly.
    final_clamp_to_wall, wall_axis, wall_sign
        Evaluation-time rigid-wall clamp of the corrected shape
        (:class:`shape_corrections.RigidWallClamp`); needs
        ``datapipe.rigid_wall_position``.
    **student_kwargs
        Constructor arguments of :class:`mgn_corrector.MeshGraphNetCorrector`;
        its ``functional_dim`` counts the three teacher-displacement channels.

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
        final_clamp_to_wall: bool = False,
        wall_axis: int = 0,
        wall_sign: float = 1.0,
        **student_kwargs,
    ) -> None:
        super().__init__()
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

    @torch.no_grad()
    def teacher_trajectory(self, sample, data_stats):
        """Evaluation: the teacher's normalized positions ``[N, T, 3]`` over the
        queried horizon."""
        coords = sample.node_features["coords"]
        return self.teacher(sample, data_stats).detach().to(coords.dtype)

    # ----- student -----
    def _correct(self, sample, data_stats, predicted, return_auxiliary):
        """Run the student on one frame's teacher prediction."""
        coords = sample.node_features["coords"]
        displacement = (predicted - coords).to(coords.dtype)
        return self.student(
            sample,
            data_stats,
            teacher_displacement=displacement,
            return_auxiliary=return_auxiliary,
        )

    def forward(self, sample, data_stats, return_auxiliary: bool = False):
        """One corrected frame in training, the corrected trajectory ``[N, T, 3]``
        in evaluation (with the rigid-wall clamp applied frame by frame when it
        is enabled)."""
        features = sample.node_features
        coords = features["coords"]
        if not self.training:
            predicted = self.teacher_trajectory(sample, data_stats)
            original = features.get("time")
            outputs = []
            try:
                for k in range(predicted.shape[1]):
                    features["time"] = torch.tensor(
                        k / self.teacher.rollout_steps, device=coords.device
                    )
                    frame = self._correct(sample, data_stats, predicted[:, k], False)
                    outputs.append(
                        correct_shape(frame, sample, data_stats, clamp=self.wall_clamp)
                    )
            finally:
                if original is None:
                    features.pop("time", None)
                else:
                    features["time"] = original
            return torch.stack(outputs, dim=1)
        with torch.no_grad():
            predicted = self._frame(sample, data_stats, float(features["time"]))
        return self._correct(sample, data_stats, predicted, return_auxiliary)
