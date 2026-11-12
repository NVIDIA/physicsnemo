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

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from datapipes.datapipe import SimSample
from torch import nn
from torch.utils.checkpoint import checkpoint as ckpt
from utils.warm_start import load_warm_start

import physicsnemo
from physicsnemo.experimental.models.deformer import (
    DeFormer,
    DeFormerCorrector,
    MeshContextFiLM,
)
from physicsnemo.experimental.nn.shell_contact import clamp_to_wall
from physicsnemo.models.figconvnet.figconvunet import FIGConvUNet
from physicsnemo.models.geotransolver import GeoTransolver
from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.models.transolver import Transolver

EPS = 1e-8
_FO_MIN = 3  # position-only; with dynamic_targets can be larger
_POS_DIM = 3  # position (x,y,z)


# =============================================================================
# One-shot rollout models
# =============================================================================


def _oneshot_init(kwargs: dict, out_key: str) -> int:
    """Validate and set rollout_steps. Returns rollout_steps."""
    num_time_steps = kwargs.pop("num_time_steps")
    rollout_steps = num_time_steps - 1
    out_dim = kwargs.get(out_key)
    required_min = rollout_steps * _FO_MIN
    if out_dim is not None and out_dim < required_min:
        raise ValueError(
            f"{out_key}={out_dim} is too small for num_time_steps={num_time_steps} "
            f"(rollout_steps={rollout_steps}). Need {out_key} >= {required_min}."
        )
    return rollout_steps


def _oneshot_inputs(sample: SimSample, rollout_steps: int):
    """Extract coords, features, N, T, Fo. Returns (coords, features, N, T, Fo)."""
    inputs = sample.node_features
    coords = inputs["coords"]  # [N,3]
    features = inputs.get("features", coords.new_zeros((coords.size(0), 0)))
    N, T = coords.size(0), rollout_steps
    Fo = sample.node_target.shape[2]
    return coords, features, N, T, Fo


def _cat_global(
    coords: torch.Tensor, features: torch.Tensor, sample: SimSample
) -> torch.Tensor:
    """Concatenate coords, features, and global (broadcast). Returns [N, C]."""
    out = torch.cat([coords, features], dim=-1)
    if sample.global_features is not None:
        g = torch.stack(
            [sample.global_features[k] for k in sample.global_features], dim=0
        )
        out = torch.cat([out, g.unsqueeze(0).expand(coords.size(0), -1)], dim=-1)
    return out


def global_tokens(sample: SimSample) -> torch.Tensor | None:
    """Return selected global features as one token, shaped ``[1, 1, G]``."""
    if sample.global_features is None:
        return None
    # Token matrices (e.g. per-part design tokens, [P, F]) are consumed by
    # dedicated encoders, not flattened into the global token.
    values = [v for v in sample.global_features.values() if v.ndim <= 1]
    if not values:
        return None
    return torch.stack(values, dim=0).view(1, 1, -1)


def _oneshot_output(pred_flat: torch.Tensor, N: int, T: int, Fo: int) -> torch.Tensor:
    """Validate and reshape to [N, T, Fo]."""
    if pred_flat.shape[-1] < T * Fo:
        raise ValueError(
            f"Model output dim {pred_flat.shape[-1]} smaller than T*Fo={T * Fo}"
        )
    return pred_flat[:, : T * Fo].view(N, T, Fo)


def _oneshot_add_coords(pred: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Add initial coords to the position slice only. pred [N,T,Fo], coords [N,3]. Fo >= 3.

    The sum is formed out of place in float32: under autocast the network
    output is a half-precision displacement, and adding the absolute
    coordinates in that precision would round the positions (about 1 mm at
    vehicle scale in float16, far more in bfloat16).
    """
    pred = pred.float()
    positions = pred[:, :, :_POS_DIM] + coords.unsqueeze(1).to(pred.dtype)
    return torch.cat((positions, pred[:, :, _POS_DIM:]), dim=-1)


class GeoTransolverOneShot(GeoTransolver):
    """GeoTransolver model with one-shot training."""

    input_kind = "points"

    def __init__(self, *args, **kwargs):
        self.rollout_steps = _oneshot_init(kwargs, "out_dim")
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        r"""Predict the rollout using the existing tensor return format."""
        prediction, _ = self._forward_rollout(sample, data_stats, with_features=False)
        return prediction

    def forward_with_geometry_features(
        self, sample: SimSample, data_stats: dict
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Predict the rollout and return detached geometry tokens in one pass.

        Parameters
        ----------
        sample : SimSample
            Simulation inputs containing initial coordinates and node features.
        data_stats : dict
            Dataset statistics, as accepted by :meth:`forward`.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor | None]
            Rollout of shape :math:`(N, T, F_o)` and detached geometry tokens
            of shape :math:`(1, H, S, D_g)`, or ``None`` without a tokenizer.

        Notes
        -----
        Direct calls bypass model-level forward hooks. Wrapper-based training
        should invoke this method inside an adapter's ``forward``.
        """
        return self._forward_rollout(sample, data_stats, with_features=True)

    def _forward_rollout(
        self, sample: SimSample, data_stats: dict, *, with_features: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Share rollout preprocessing and reshaping across prediction methods."""
        coords, features, N, T, Fo = _oneshot_inputs(sample, self.rollout_steps)
        fx = torch.cat([coords, features], dim=-1)
        global_emb = None
        if sample.global_features is not None:
            g = torch.stack(
                [sample.global_features[k] for k in sample.global_features], dim=0
            )
            global_emb = g.unsqueeze(0).unsqueeze(0)  # [1, 1, G]
        inputs = dict(
            local_embedding=fx.unsqueeze(0),
            geometry=coords.unsqueeze(0),
            local_positions=coords.unsqueeze(0),
            global_embedding=global_emb,
        )
        if with_features:
            raw, geometry_tokens = super().forward_with_geometry_features(**inputs)
        else:
            raw = super().forward(**inputs)
            geometry_tokens = None
        raw = raw.squeeze(0)
        pred = _oneshot_add_coords(_oneshot_output(raw, N, T, Fo), coords)
        return pred, geometry_tokens


class TransolverOneShot(Transolver):
    """Transolver model with one-shot training."""

    input_kind = "points"

    def __init__(self, *args, **kwargs):
        self.rollout_steps = _oneshot_init(kwargs, "out_dim")
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        coords, features, N, T, Fo = _oneshot_inputs(sample, self.rollout_steps)
        fx = _cat_global(coords, features, sample).unsqueeze(0)
        raw = super().forward(fx=fx, embedding=coords.unsqueeze(0)).squeeze(0)
        pred = _oneshot_add_coords(_oneshot_output(raw, N, T, Fo), coords)
        return pred


class MeshGraphNetOneShot(MeshGraphNet):
    """MeshGraphNet model with one-shot training."""

    input_kind = "graph"

    def __init__(self, *args, **kwargs):
        self.rollout_steps = _oneshot_init(kwargs, "output_dim")
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        coords, features, N, T, Fo = _oneshot_inputs(sample, self.rollout_steps)
        node_feat = _cat_global(coords, features, sample)
        raw = super().forward(
            node_features=node_feat,
            edge_features=sample.graph.edge_attr,
            graph=sample.graph,
        )
        pred = _oneshot_add_coords(_oneshot_output(raw, N, T, Fo), coords)
        return pred


class FIGConvUNetOneShot(FIGConvUNet):
    """FIGConvUNet model with one-shot training."""

    input_kind = "points"

    def __init__(self, *args, **kwargs):
        self.rollout_steps = _oneshot_init(kwargs, "out_channels")
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        coords, features, N, T, Fo = _oneshot_inputs(sample, self.rollout_steps)
        feat = _cat_global(coords, features, sample).unsqueeze(0)  # [1, N, C]
        raw, _ = super().forward(vertices=coords.unsqueeze(0), features=feat)
        pred = _oneshot_add_coords(_oneshot_output(raw.squeeze(0), N, T, Fo), coords)
        return pred


# =============================================================================
# Autoregressive rollout models
# =============================================================================


def _geo_global_emb(sample: SimSample):
    """Build global embedding for GeoTransolver from sample."""
    return global_tokens(sample)


class GeoTransolverAutoregressiveRolloutTraining(GeoTransolver):
    """
    GeoTransolver model with autoregressive rollout training.

    Predicts sequence by autoregressively updating velocity and position
    using predicted accelerations. Supports gradient checkpointing during training.
    """

    input_kind = "points"

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt")
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [N, T, 3] rollout of predicted positions
        """
        inputs = sample.node_features
        coords = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords.new_zeros((coords.size(0), 0)))
        global_emb = _geo_global_emb(sample)

        # Initial states
        y_t1 = coords  # [N,3]
        y_t0 = y_t1 - self.initial_vel * self.dt  # backstep using initial velocity

        outputs: list[torch.Tensor] = []
        for t in range(self.rollout_steps):
            # Velocity normalization
            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )

            # Model input: vel_norm + features
            fx_t = torch.cat([vel_norm, features], dim=-1)  # [N, 3+F]

            def step_fn(local_emb, geometry, local_pos):
                return super(GeoTransolverAutoregressiveRolloutTraining, self).forward(
                    local_embedding=local_emb,
                    geometry=geometry,
                    local_positions=local_pos,
                    global_embedding=global_emb,
                )

            if self.training:
                outf = ckpt(
                    step_fn,
                    fx_t.unsqueeze(0),
                    y_t1.unsqueeze(0),
                    y_t1.unsqueeze(0),
                    use_reentrant=False,
                ).squeeze(0)
            else:
                outf = step_fn(
                    fx_t.unsqueeze(0), y_t1.unsqueeze(0), y_t1.unsqueeze(0)
                ).squeeze(0)

            # De-normalize acceleration
            acc = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            vel = self.dt * acc + vel
            y_t2 = self.dt * vel + y_t1

            outputs.append(y_t2)
            y_t1, y_t0 = y_t2, y_t1

        return torch.stack(outputs, dim=0).transpose(0, 1)  # [N,T,3]


# =============================================================================
# Time-conditional rollout models
# =============================================================================


class TimeEmbedding(torch.nn.Module):
    """Embed the normalized query time ``q`` in ``[0, 1]``.

    ``"scalar"`` passes ``q`` through unchanged (the original recipe).
    ``"fourier"`` concatenates ``q`` with ``sin``/``cos`` of ``bands``
    harmonics and projects through a two-layer MLP to ``dim`` channels, so
    that nearby times share a smooth, high-resolution code instead of a single
    linear coordinate.
    """

    def __init__(self, kind: str = "scalar", bands: int = 8, dim: int = 32):
        super().__init__()
        if kind not in ("scalar", "fourier"):
            raise ValueError("time_embedding must be 'scalar' or 'fourier'")
        if kind == "fourier" and (bands < 1 or dim < 1):
            raise ValueError("fourier time embedding needs bands >= 1 and dim >= 1")
        self.kind = kind
        self.dim = 1 if kind == "scalar" else int(dim)
        if kind == "fourier":
            self.register_buffer(
                "frequencies",
                2.0 * math.pi * torch.arange(1, bands + 1, dtype=torch.float32),
            )
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(2 * bands + 1, dim),
                torch.nn.SiLU(),
                torch.nn.Linear(dim, dim),
            )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Return ``[1, dim]`` for a scalar time."""
        time = torch.as_tensor(time, dtype=torch.float32).reshape(1, 1)
        if self.kind == "scalar":
            return time
        time = time.to(self.frequencies.device)
        phase = time * self.frequencies
        return self.mlp(torch.cat((time, torch.sin(phase), torch.cos(phase)), dim=-1))


class GeoTransolverTimeConditional(GeoTransolver):
    """
    GeoTransolver model with time-conditional rollout training.

    Predicts each time step independently, conditioned on normalized time.
    ``time_embedding="fourier"`` replaces the scalar time input with a learned
    Fourier code of ``time_dim`` channels (``functional_dim`` must count it),
    and ``time_as_global=True`` additionally feeds that code as a global token
    (``global_dim`` must equal ``time_dim`` plus any dataset global features).
    """

    input_kind = "points"

    def __init__(
        self,
        *,
        num_time_steps: int,
        time_embedding: str = "scalar",
        time_fourier_bands: int = 8,
        time_dim: int = 32,
        time_as_global: bool = False,
        dt: float | None = None,
        **kwargs,
    ):
        self.rollout_steps: int = int(num_time_steps) - 1
        # Bookkeeping only: the Yaris recipes set the frame interval on the
        # dataset side (``yaris.dt``) and the model reads it from there; the
        # backbone itself is time-step agnostic.
        self.dt = dt
        super().__init__(**kwargs)
        self.time_embed = TimeEmbedding(time_embedding, time_fourier_bands, time_dim)
        self.time_as_global = bool(time_as_global)

    def _time_inputs(self, sample: SimSample, num_nodes: int):
        """Per-node time code ``[N, dim]`` and the optional global token."""
        code = self.time_embed(sample.node_features["time"])
        code = code.to(sample.node_features["coords"].dtype)
        global_embedding = global_tokens(sample)
        if self.time_as_global:
            token = code.view(1, 1, -1)
            global_embedding = (
                token
                if global_embedding is None
                else torch.cat((global_embedding, token), dim=-1)
            )
        return code.expand(num_nodes, -1), global_embedding

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        if self.training:
            return self._forward(sample, data_stats)
        else:
            return self._rollout(sample, data_stats)

    def _forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [N, Fo] prediction at time t
        """
        inputs = sample.node_features
        coords = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords.new_zeros((coords.size(0), 0)))
        global_embedding = None
        if sample.global_features is not None:
            global_embedding = (
                torch.stack(
                    [sample.global_features[k] for k in sample.global_features], dim=0
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )  # [1, 1, num_global]

        N = coords.size(0)
        time_code, time_global = self._time_inputs(sample, N)
        if time_global is not None:
            global_embedding = time_global

        fx_t = torch.cat([coords, features, time_code], dim=-1)  # [N, 3+F+time]
        pred = (
            super(GeoTransolverTimeConditional, self)
            .forward(
                local_embedding=fx_t.unsqueeze(0),
                geometry=coords.unsqueeze(0),
                local_positions=coords.unsqueeze(0),
                global_embedding=global_embedding,
            )
            .squeeze(0)
        )  # [N, Fo]

        outputs = coords + pred[:, :3]
        outputs = torch.cat([outputs, pred[:, 3:]], dim=-1)

        return outputs  # [N,3]

    def _rollout(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        """
        Args:
            sample: SimSample containing node_features and node_target
            data_stats: dict containing normalization stats
        Returns:
            [N, T, Fo] rollout of predicted positions
        """
        device = sample.node_features["coords"].device
        outputs: list[torch.Tensor] = []
        # Cases may be shorter than the configured bound (Yaris: 19-31 frames):
        # query exactly the target's horizon while the time label keeps its
        # normalization by rollout_steps, so it means the same physical time.
        target = getattr(sample, "node_target", None)
        steps = (
            int(target.shape[1])
            if target is not None and target.ndim == 3
            else self.rollout_steps
        )
        if steps > self.rollout_steps:
            raise ValueError(
                f"Target has {steps} steps but the model was built for {self.rollout_steps}"
            )
        original = sample.node_features.get("time")
        try:
            for t in range(steps):
                time = torch.tensor(t / self.rollout_steps, device=device)
                sample.node_features["time"] = time
                y_t2 = self._forward(sample, data_stats)
                outputs.append(y_t2)
        finally:
            if original is None:
                sample.node_features.pop("time", None)
            else:
                sample.node_features["time"] = original

        return torch.stack(outputs, dim=0).transpose(0, 1)  # [N,T,3]


# =============================================================================
# One-step rollout models
# =============================================================================


class GeoTransolverOneStepRollout(GeoTransolver):
    """
    One-step rollout:
      - Training: teacher forcing (uses GT for each step, but first step needs backstep)
      - Inference: autoregressive (uses predictions)
    """

    input_kind = "points"

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt", 5e-3)
        self.initial_vel: torch.Tensor = kwargs.pop("initial_vel")
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        inputs = sample.node_features
        coords0 = inputs["coords"]  # [N,3]
        features = inputs.get("features", coords0.new_zeros((coords0.size(0), 0)))
        global_emb = _geo_global_emb(sample)

        # Ground truth sequence [T+1, N, 3] (t0 + rollout steps)
        gt_seq = torch.cat(
            [
                coords0.unsqueeze(0),
                sample.node_target.transpose(0, 1),
            ],  # [N,T,3] -> [T,N,3]
            dim=0,
        )

        outputs: list[torch.Tensor] = []

        # First step: backstep to create y_-1
        y_t0 = gt_seq[0] - self.initial_vel * self.dt
        y_t1 = gt_seq[0]

        for t in range(self.rollout_steps):
            if self.training and t > 0:
                # teacher forcing uses GT pairs
                y_t0, y_t1 = gt_seq[t - 1], gt_seq[t]

            vel = (y_t1 - y_t0) / self.dt
            vel_norm = (vel - data_stats["node"]["norm_vel_mean"]) / (
                data_stats["node"]["norm_vel_std"] + EPS
            )
            fx_t = torch.cat([vel_norm, features], dim=-1)

            def step_fn(local_emb, geometry, local_pos):
                return super(GeoTransolverOneStepRollout, self).forward(
                    local_embedding=local_emb,
                    geometry=geometry,
                    local_positions=local_pos,
                    global_embedding=global_emb,
                )

            if self.training:
                outf = ckpt(
                    step_fn,
                    fx_t.unsqueeze(0),
                    y_t1.unsqueeze(0),
                    y_t1.unsqueeze(0),
                    use_reentrant=False,
                ).squeeze(0)
            else:
                outf = step_fn(
                    fx_t.unsqueeze(0), y_t1.unsqueeze(0), y_t1.unsqueeze(0)
                ).squeeze(0)

            acc = (
                outf * data_stats["node"]["norm_acc_std"]
                + data_stats["node"]["norm_acc_mean"]
            )
            vel_pred = self.dt * acc + vel
            y_t2_pred = self.dt * vel_pred + y_t1

            outputs.append(y_t2_pred)

            if not self.training:
                # autoregressive update for inference
                y_t0, y_t1 = y_t1, y_t2_pred

        return torch.stack(outputs, dim=0).transpose(0, 1)  # [N,T,3]


# ---- Rigid-wall clamp (evaluation-time shape correction) -----------------------------
# Evaluation-time rigid-wall clamp of a predicted shape.
#
# The clamp caps the coordinate of every node at an analytic rigid wall plane
# (the Yaris barrier at ``x = 100`` mm), so no node is predicted inside the wall.
# Its settings are a small frozen dataclass that the models and recipes hold;
# :func:`correct_shape` applies it to one predicted frame in the datapipe's
# normalized coordinates. Nothing here is trained or differentiated.


@dataclass(frozen=True)
class RigidWallClamp:
    """Settings of the rigid-wall clamp.

    Parameters
    ----------
    enabled : bool
        Apply the clamp (the wall position comes from ``graph.rigid_wall``,
        set by ``datapipe.rigid_wall_position``).
    axis : int
        Coordinate axis of the wall normal (0, 1 or 2).
    sign : float
        ``+1`` when the admissible half space is ``x[axis] <= wall``.
    """

    enabled: bool = False
    axis: int = 0
    sign: float = 1.0

    def __post_init__(self) -> None:
        if self.axis not in (0, 1, 2) or float(self.sign) not in (1.0, -1.0):
            raise ValueError("wall_axis must be 0, 1 or 2 and wall_sign +1 or -1")
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "axis", int(self.axis))
        object.__setattr__(self, "sign", float(self.sign))


def wall_position(sample) -> float:
    """The rigid wall plane coordinate the datapipe attached to the graph."""
    wall = getattr(sample.graph, "rigid_wall", None)
    if wall is None:
        raise ValueError(
            "the rigid-wall clamp needs graph.rigid_wall (datapipe.rigid_wall_position)"
        )
    return float(wall.reshape(-1)[0])


@torch.no_grad()
def correct_shape(
    prediction: torch.Tensor,
    sample,
    data_stats,
    *,
    clamp: RigidWallClamp | None = None,
) -> torch.Tensor:
    """Apply the rigid-wall clamp to one predicted frame ``[N, 3]`` when enabled.

    The clamp caps the wall coordinate in the datapipe's normalized coordinates;
    the wall position comes from ``graph.rigid_wall``. Returns ``prediction``
    itself when the clamp is off.
    """
    if clamp is None or not clamp.enabled:
        return prediction
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        std = data_stats["node"]["pos_std"].reshape(-1)[clamp.axis].to(prediction)
        mean = data_stats["node"]["pos_mean"].reshape(-1)[clamp.axis].to(prediction)
        limit = (wall_position(sample) - mean) / (std + EPS)
        return clamp_to_wall(prediction, limit, axis=clamp.axis, sign=clamp.sign)


# ---- DeFormer: time-conditional network wrapper --------------------------------------
# Time-conditional DeFormer: initial mesh and a time code to the deformed shape.
#
# For a requested future frame the model maps the initial mesh, its static
# features and a time code to the deformed positions:
#
#     inputs (X0, features, tau) -> pre-processor -> FLARE backbone
#         -> post stage A -> coarse head  (provisional shape)
#         -> post stage B -> final head   (correction added to the provisional shape)
#
# * The **time code** is a learned Fourier embedding of the normalized query time
#   that enters every node's input and, with ``time_as_global``, the global token
#   of the backbone.
# * The **local stages** are the base model's local operator (static kNN
#   attention in the DeFormer recipe), FiLM-conditioned on the
#   mesh context: a pre-processor before the backbone and two post stages after
#   it. Stage A feeds a zero-initialised coarse head (the provisional shape,
#   supervised with ``training.auxiliary_loss_weight``); stage B is an independent,
#   re-initialised copy of stage A whose gated correction gives the final shape.
# * There is no recurrent state and no BPTT: training samples independent
#   (case, time) pairs, evaluation queries every future frame from the initial
#   mesh and returns the trajectory ``[N, T, 3]``.
#
# The evaluation-time rigid-wall clamp (:class:`RigidWallClamp`) is available
# and off by default. The checkpoint format is the base model's
# (``physicsnemo.Module.from_checkpoint``).


class TimeConditionalOutput(NamedTuple):
    """Training output of a time-conditional model: the final position and the
    auxiliary (provisional) position of stage A, both normalized ``[N, 3]``; the
    trainer supervises the auxiliary one with ``training.auxiliary_loss_weight``."""

    prediction: torch.Tensor
    auxiliary_prediction: torch.Tensor


class DeFormerTimeConditional(DeFormer):
    """Time-conditional DeFormer (the "network" of the DeFormer paper).

    Parameters
    ----------
    num_time_steps : int
        Frames per trajectory in the datapipe; ``num_time_steps - 1`` future
        frames are queried at evaluation, labelled ``k / (num_time_steps - 1)``
        for ``k = 0, ...`` (the datapipe's ``one_time_step`` convention: label
        0 is the first future frame).
    time_embedding : {"scalar", "fourier"}
        Time code entering the node inputs. ``functional_dim`` must count its
        width (1 for scalar, ``time_dim`` for fourier).
    time_fourier_bands, time_dim : int
        Fourier harmonics and output width of the learned time code.
    time_as_global : bool
        Also feed the time code as the global token; ``global_dim`` must then
        equal ``time_dim`` plus the number of dataset global features.
    dt : float, optional
        Frame interval of the datapipe in seconds. Bookkeeping only: the Yaris
        recipes derive ``datapipe.dt`` from ``model.dt`` and the prediction
        never integrates in time.
    clamp_to_wall, wall_axis, wall_sign
        Evaluation-time rigid-wall clamp of the prediction (see
        :class:`RigidWallClamp`); needs ``graph.rigid_wall``
        from ``datapipe.rigid_wall_position``.
    **kwargs
        :class:`~physicsnemo.experimental.models.deformer.DeFormer`
        settings (backbone, local operator, context fusion, activation
        checkpointing). ``out_dim`` must be 3 and ``use_te`` false. The base
        model is built without its output head (``output_head=False``):
        ``num_post_processor_layers``, ``num_post_processor_checkpoint_segments``
        and ``mesh_post_residual_gate_init`` size the two stages here.
    """

    input_kind = "graph"

    def __init__(
        self,
        *,
        num_time_steps: int,
        time_embedding: str = "fourier",
        time_fourier_bands: int = 8,
        time_dim: int = 32,
        time_as_global: bool = True,
        dt: float | None = None,
        clamp_to_wall: bool = False,
        wall_axis: int = 0,
        wall_sign: float = 1.0,
        **kwargs,
    ) -> None:
        if num_time_steps < 2:
            raise ValueError("num_time_steps must be at least 2")
        if dt is not None and not dt > 0:
            raise ValueError("dt must be positive when given")
        if kwargs.get("out_dim", 3) != 3:
            raise ValueError("This recipe predicts positions only (out_dim=3)")
        if kwargs.get("use_te", False):
            raise ValueError("The time-conditional DeFormer requires use_te=False")
        if "output_head" in kwargs:
            raise ValueError(
                "DeFormerTimeConditional builds its own decoder stages; "
                "output_head is not a setting of this model"
            )
        # Validated settings of the evaluation-time wall clamp.
        self.wall_clamp = RigidWallClamp(
            enabled=clamp_to_wall, axis=wall_axis, sign=wall_sign
        )
        # The base model supplies the backbone state and the pre-processor only
        # (no output head); the two decoder stages below are this model's.
        super().__init__(output_head=False, **kwargs)
        self.rollout_steps = int(num_time_steps) - 1
        self.dt = None if dt is None else float(dt)
        self.time_embed = TimeEmbedding(time_embedding, time_fourier_bands, time_dim)
        self.time_as_global = bool(time_as_global)
        # Full-width backbone output (hidden plus optional local features) to the
        # mesh width, never through the three-channel output.
        effective_dim = self.backbone_width
        self.post_latent_input = (
            nn.Identity()
            if effective_dim == self.mesh_hidden_dim
            else nn.Linear(effective_dim, self.mesh_hidden_dim)
        )
        # Stage A side head: the provisional position. Zero-initialised so the
        # model starts at the undeformed mesh.
        self.coarse_decoder = nn.Sequential(
            nn.LayerNorm(self.mesh_hidden_dim),
            nn.Linear(self.mesh_hidden_dim, self.output_dim),
        )
        self._zero_head(self.coarse_decoder[-1])
        # Stage A: a local processor of the configured post-stage depth on the
        # backbone latent, FiLM-conditioned on the mesh context when fusion is
        # on. Stage B: an independent processor and FiLM of the same shape
        # whose gated correction gives the final shape.
        fusion = self.mesh_context_fusion != "none"
        self.coarse_processor = self.make_local_processor(
            self.num_post_processor_layers, self.num_post_processor_checkpoint_segments
        )
        self.coarse_context_film = (
            MeshContextFiLM(self.mesh_hidden_dim) if fusion else None
        )
        self.refinement_processor = self.make_local_processor(
            self.num_post_processor_layers, self.num_post_processor_checkpoint_segments
        )
        self.refinement_context_film = (
            MeshContextFiLM(self.mesh_hidden_dim) if fusion else None
        )
        self.final_decoder = self.make_node_decoder()
        self.final_gate = nn.Parameter(torch.tensor(self.mesh_post_residual_gate_init))
        last_linear = [
            m for m in self.final_decoder.modules() if isinstance(m, nn.Linear)
        ][-1]
        self._zero_head(last_linear)
        self.__name__ = "DeFormerTimeConditional"

    @staticmethod
    def _zero_head(linear: nn.Linear) -> None:
        nn.init.zeros_(linear.weight)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)

    # ----- building blocks -----
    def _time_inputs(self, sample, num_nodes: int):
        """Per-node time code ``[N, time_dim]`` and the global token(s)."""
        coords = sample.node_features["coords"]
        code = self.time_embed(sample.node_features["time"]).to(coords.dtype)
        global_embedding = global_tokens(sample)
        if self.time_as_global:
            token = code.view(1, 1, -1)
            global_embedding = (
                token
                if global_embedding is None
                else torch.cat((global_embedding, token), dim=-1)
            )
        return code.expand(num_nodes, -1), global_embedding

    def coarse(self, sample):
        """Provisional position ``[N, 3]``, the stage-A latent and the backbone state."""
        coords = sample.node_features["coords"]
        features = sample.node_features.get(
            "features", coords.new_zeros((coords.shape[0], 0))
        )
        time_code, global_embedding = self._time_inputs(sample, coords.shape[0])
        nodes = torch.cat((coords, features, time_code), dim=-1)
        state = DeFormer.forward(
            self,
            node_features=nodes,
            graph=sample.graph,
            geometry=coords,
            global_embedding=global_embedding,
            local_positions=coords,
            return_backbone=True,
        )
        latent = self.post_latent_input(state.point_features)
        if self.coarse_context_film is not None:
            latent = self.coarse_context_film(latent, state.mesh_context)
        latent = self._run_local_processor(
            self.coarse_processor, latent, state.neighborhood
        )
        coarse = coords + self.coarse_decoder(latent).to(coords.dtype)
        return coarse, latent, state

    def refine(self, latent, state):
        """Gated correction ``[N, 3]`` of stage B from the stage-A latent."""
        if self.refinement_context_film is not None:
            latent = self.refinement_context_film(latent, state.mesh_context)
        latent = self._run_local_processor(
            self.refinement_processor, latent, state.neighborhood
        )
        return self.final_gate * self.final_decoder(latent)

    def predict_frame(self, sample, data_stats) -> TimeConditionalOutput:
        """Final and provisional position at the sample's time label.

        In evaluation mode the final position passes through the rigid-wall
        clamp when it is enabled.
        """
        coarse, latent, state = self.coarse(sample)
        shape = coarse + self.refine(latent, state).to(coarse.dtype)
        if not self.training:
            shape = correct_shape(shape, sample, data_stats, clamp=self.wall_clamp)
        return TimeConditionalOutput(shape, coarse)

    # ----- recipe entry point -----
    def forward(self, sample, data_stats, return_auxiliary: bool = False):
        """One frame in training (``[N, 3]``, or the :class:`TimeConditionalOutput`
        with ``return_auxiliary``), the trajectory ``[N, T, 3]`` in evaluation.

        Evaluation queries exactly the target's horizon (cases may be shorter
        than the configured bound) while the time label keeps its normalization
        by ``rollout_steps``, so a label means the same physical time everywhere.
        """
        if self.training:
            time = float(sample.node_features["time"])
            if not 0.0 <= time <= (self.rollout_steps - 1) / self.rollout_steps + 1e-6:
                raise ValueError(
                    "time is outside the datapipe's normalized future-frame range"
                )
            output = self.predict_frame(sample, data_stats)
            return output if return_auxiliary else output.prediction
        target = getattr(sample, "node_target", None)
        steps = (
            int(target.shape[1])
            if target is not None and target.ndim == 3
            else self.rollout_steps
        )
        if steps > self.rollout_steps:
            raise ValueError(
                f"Target has {steps} steps but the model was built for {self.rollout_steps}"
            )
        features = sample.node_features
        original = features.get("time")
        outputs = []
        try:
            for k in range(steps):
                features["time"] = torch.tensor(
                    k / self.rollout_steps, device=features["coords"].device
                )
                outputs.append(self.predict_frame(sample, data_stats).prediction)
        finally:
            if original is None:
                features.pop("time", None)
            else:
                features["time"] = original
        return torch.stack(outputs, dim=1)  # [N, T, 3]


# ---- DeFormer: teacher-plus-corrector wrapper ----------------------------------------
# DeFormer: a frozen time-conditional network followed by its corrector.
#
# The teacher is a finished :class:`models.rollout.DeFormerTimeConditional`,
# loaded from its checkpoint and frozen; the student is a
# :class:`physicsnemo.experimental.models.deformer.DeFormerCorrector` trained on
# top of it. The teacher's prediction at the queried time is recomputed on the fly;
# this wrapper assembles the student's node inputs (coordinates, static features,
# the time code and the teacher's displacement), hands them to the student, which
# adds its correction, and loops over frames at evaluation. The trainings are
# decoupled (the teacher never receives a gradient) and no prediction files are
# stored. Inference is the teacher followed by the student.
#
# ``final_clamp_to_wall`` applies the evaluation-time rigid-wall clamp
# (:class:`RigidWallClamp`) to the corrected shape (the teacher's own
# prediction stays as the student was trained on).


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
        (:class:`RigidWallClamp`); needs
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
    overwritten by the state dict.
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
