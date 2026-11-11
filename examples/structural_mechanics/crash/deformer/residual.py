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

The teacher is a finished :class:`deformer.time_conditional.DeFormerTimeConditional`,
loaded from its checkpoint and frozen; the student is a
:class:`physicsnemo.experimental.models.deformer.DeFormerCorrector` trained on
top of it. The teacher's prediction at the queried time is recomputed on the fly;
this wrapper assembles the student's node inputs (coordinates, static features,
the time code and the teacher's displacement), hands them to the student, which
adds its correction, and loops over frames at evaluation. The trainings are
decoupled (the teacher never receives a gradient) and no prediction files are
stored. Inference is the teacher followed by the student.

``final_clamp_to_wall`` applies the evaluation-time rigid-wall clamp of
:mod:`deformer.shape_corrections` to the corrected shape (the teacher's own
prediction stays as the student was trained on).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from rollout import TimeEmbedding
from torch import nn
from utils import load_warm_start

import physicsnemo
from physicsnemo.experimental.models.deformer import DeFormerCorrector

from .shape_corrections import RigidWallClamp, correct_shape
from .time_conditional import TimeConditionalOutput


class ResidualTimeConditional(physicsnemo.Module):
    """Frozen teacher followed by the DeFormer corrector.

    Parameters
    ----------
    teacher : nn.Module
        The teacher's architecture (an instantiated, randomly initialised
        ``DeFormerTimeConditional`` of the teacher's recipe); its weights
        come from ``teacher_weights``.
    teacher_weights : str or os.PathLike, optional
        ``.mdlus`` (or state-dict ``.pt``) file of the teacher, loaded strictly.
        Required to train; ``None`` when a saved corrector is reloaded, whose
        state dict carries the teacher's weights.
    num_time_steps : int
        Frames per trajectory in the datapipe (``num_time_steps - 1`` future
        frames, labelled ``k / (num_time_steps - 1)`` as in the teacher).
    time_embedding, time_fourier_bands, time_dim
        The time code of the student's node inputs (as in the teacher).
    dt : float, optional
        Frame interval of the datapipe in seconds (bookkeeping only).
    final_clamp_to_wall, wall_axis, wall_sign
        Evaluation-time rigid-wall clamp of the corrected shape
        (:class:`deformer.shape_corrections.RigidWallClamp`); needs
        ``datapipe.rigid_wall_position``.
    local_knn_k, local_neighborhood
        Accepted and ignored: the datapipe reads them from the model config to
        build the teacher's neighbourhoods.
    **student_kwargs
        Constructor arguments of
        :class:`physicsnemo.experimental.models.deformer.DeFormerCorrector`;
        its ``functional_dim`` counts coords (3), the static features, the time
        code (``time_dim``) and the teacher displacement (3).

    A saved corrector (``.mdlus``) is self-contained: the teacher's weights are
    part of its state dict and it records ``teacher_weights=None``, so
    ``physicsnemo.Module.from_checkpoint(path)`` rebuilds the teacher's
    architecture and loads everything from the state dict. A teacher file may
    still be given at reload with
    ``override_args={"teacher_weights": <network .mdlus>}``; it is read and then
    overwritten by the state dict. Correctors saved by earlier revisions record
    the training-time path and take the same override.
    """

    # The teacher path is the one argument a saved corrector lets the loader override.
    _overridable_args = {"teacher_weights"}
    input_kind = "graph"

    def __init__(
        self,
        *,
        teacher: nn.Module,
        teacher_weights: str | os.PathLike | None = None,
        num_time_steps: int,
        time_embedding: str = "fourier",
        time_fourier_bands: int = 8,
        time_dim: int = 32,
        dt: float | None = None,
        final_clamp_to_wall: bool = False,
        wall_axis: int = 0,
        wall_sign: float = 1.0,
        local_knn_k: int | None = None,
        local_neighborhood: str | None = None,
        **student_kwargs,
    ) -> None:
        super().__init__()
        if num_time_steps < 2:
            raise ValueError("num_time_steps must be at least 2")
        if dt is not None and not dt > 0:
            raise ValueError("dt must be positive when given")
        self.wall_clamp = RigidWallClamp(
            enabled=final_clamp_to_wall, axis=wall_axis, sign=wall_sign
        )
        if teacher_weights is not None:
            teacher_weights = str(teacher_weights)
            if not teacher_weights:
                raise ValueError(
                    "teacher_weights must name the teacher's checkpoint (None only "
                    "when a saved corrector supplies the teacher's weights)"
                )
            if not Path(teacher_weights).is_file():
                raise FileNotFoundError(
                    f"teacher checkpoint {teacher_weights} does not exist; point "
                    "model.teacher_weights at the network's .mdlus file, or pass "
                    'override_args={"teacher_weights": ...} to Module.from_checkpoint'
                )
            load_warm_start(teacher, teacher_weights)
            # The saved corrector carries the teacher in its state dict: record no
            # file, so the checkpoint is self-contained and no path of the
            # training machine travels with it.
            self._args["__args__"]["teacher_weights"] = None
        else:
            logging.getLogger(__name__).info(
                "teacher_weights=None: the teacher's weights are expected from the "
                "corrector's state dict"
            )
        teacher.requires_grad_(False)
        teacher.eval()
        self.teacher = teacher
        self.teacher_weights = teacher_weights
        # Read by recipes and readouts that know the time-conditional interface.
        self.rollout_steps = int(num_time_steps) - 1
        self.dt = None if dt is None else float(dt)
        teacher_steps = getattr(teacher, "rollout_steps", None)
        if teacher_steps is not None and int(teacher_steps) != self.rollout_steps:
            raise ValueError(
                "teacher and corrector disagree on the frame count: the teacher has "
                f"rollout_steps={teacher_steps}, the corrector {self.rollout_steps} "
                "(training.num_time_steps)"
            )
        self.time_embed = TimeEmbedding(time_embedding, time_fourier_bands, time_dim)
        self.student = DeFormerCorrector(**student_kwargs)
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
        """Run the student on one frame's teacher prediction.

        Builds the student's node inputs (coordinates, static features, the time
        code of the sample's label and the teacher's displacement) and hands them
        with the graph and the position statistics to the corrector.
        """
        features = sample.node_features
        coords = features["coords"]
        n = coords.shape[0]
        time = float(features["time"])
        if not 0.0 <= time <= (self.rollout_steps - 1) / self.rollout_steps + 1e-6:
            raise ValueError(
                "time is outside the datapipe's normalized future-frame range"
            )
        displacement = (predicted - coords).to(coords.dtype)
        static = features.get("features", coords.new_zeros((n, 0)))
        code = self.time_embed(features["time"]).to(coords.dtype).expand(n, -1)
        node_in = torch.cat([coords, static, code, displacement], dim=-1)
        if node_in.shape[1] != self.student.functional_dim:
            raise ValueError(
                f"node inputs are {node_in.shape[1]} wide; functional_dim="
                f"{self.student.functional_dim} must count coords, static features, "
                "the time code and the teacher displacement"
            )
        prediction = self.student(
            node_in,
            coords,
            displacement,
            sample.graph,
            position_mean=data_stats["node"]["pos_mean"],
            position_std=data_stats["node"]["pos_std"],
        )
        if return_auxiliary:
            # The corrector has no provisional head: its auxiliary is the final one.
            return TimeConditionalOutput(prediction, prediction)
        return prediction

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
