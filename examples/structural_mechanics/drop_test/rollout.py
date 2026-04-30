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

import torch
from torch.utils.checkpoint import checkpoint as ckpt

from physicsnemo.models.transolver import Transolver
from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.models.figconvnet.figconvunet import FIGConvUNet
from physicsnemo.experimental.models.geotransolver import GeoTransolver

from datapipe import SimSample

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


def _oneshot_output(pred_flat: torch.Tensor, N: int, T: int, Fo: int) -> torch.Tensor:
    """Validate and reshape to [N, T, Fo]."""
    if pred_flat.shape[-1] < T * Fo:
        raise ValueError(
            f"Model output dim {pred_flat.shape[-1]} smaller than T*Fo={T * Fo}"
        )
    return pred_flat[:, : T * Fo].view(N, T, Fo)


def _oneshot_add_coords(pred: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Add initial coords to position slice only. pred [N,T,Fo], coords [N,3]. Fo >= 3."""
    pred = pred.clone()
    pred[:, :, :_POS_DIM] += coords.unsqueeze(1)  # [N,1,3] broadcasts to [N,T,3]
    return pred


class GeoTransolverOneShot(GeoTransolver):
    """GeoTransolver model with one-shot training."""

    def __init__(self, *args, **kwargs):
        self.rollout_steps = _oneshot_init(kwargs, "out_dim")
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict) -> torch.Tensor:
        coords, features, N, T, Fo = _oneshot_inputs(sample, self.rollout_steps)
        fx = torch.cat([coords, features], dim=-1)
        global_emb = None
        if sample.global_features is not None:
            g = torch.stack(
                [sample.global_features[k] for k in sample.global_features], dim=0
            )
            global_emb = g.unsqueeze(0).unsqueeze(0)  # [1, 1, G]
        raw = (
            super()
            .forward(
                local_embedding=fx.unsqueeze(0),
                geometry=coords.unsqueeze(0),
                local_positions=coords.unsqueeze(0),
                global_embedding=global_emb,
            )
            .squeeze(0)
        )
        pred = _oneshot_add_coords(_oneshot_output(raw, N, T, Fo), coords)
        return pred


class TransolverOneShot(Transolver):
    """Transolver model with one-shot training."""

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
    if sample.global_features is None:
        return None
    g = torch.stack([sample.global_features[k] for k in sample.global_features], dim=0)
    return g.unsqueeze(0).unsqueeze(0)  # [1, 1, G]


class GeoTransolverAutoregressiveRolloutTraining(GeoTransolver):
    """
    GeoTransolver model with autoregressive rollout training.

    Predicts sequence by autoregressively updating velocity and position
    using predicted accelerations. Supports gradient checkpointing during training.
    """

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
        N = coords.size(0)
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


class GeoTransolverTimeConditional(GeoTransolver):
    """
    GeoTransolver model with time-conditional rollout training.

    Predicts each time step independently, conditioned on normalized time.
    """

    def __init__(self, *args, **kwargs):
        self.rollout_steps: int = kwargs.pop("num_time_steps") - 1
        super().__init__(*args, **kwargs)

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

        N, T = coords.size(0), self.rollout_steps
        Fo = sample.node_target.shape[-1]  # 3 + sum(C_k)

        fx_t = torch.cat(
            [coords, features, inputs["time"].unsqueeze(0).repeat(N, 1)], dim=-1
        )  # [N, 3+F+1]
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
        # Shallow-copy node_features so the per-step `time` key doesn't mutate the
        # caller's SimSample. The SimSample dataclass is reused across the validation
        # loop, and a leaked `time` key would be visible to the next iteration.
        original_features = sample.node_features
        try:
            outputs: list[torch.Tensor] = []
            for t in range(self.rollout_steps):
                time = torch.tensor(t / self.rollout_steps, device=device)
                sample.node_features = {**original_features, "time": time}
                y_t2 = self._forward(sample, data_stats)
                outputs.append(y_t2)
        finally:
            sample.node_features = original_features

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
        N = coords0.size(0)
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
