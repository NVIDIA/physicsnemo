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

import math

import torch
from datapipe import SimSample
from torch.utils.checkpoint import checkpoint as ckpt

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
    return global_tokens(sample)


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

    def __init__(
        self,
        *,
        num_time_steps: int,
        time_embedding: str = "scalar",
        time_fourier_bands: int = 8,
        time_dim: int = 32,
        time_as_global: bool = False,
        dt: float | None = None,
        args=(),
        **kwargs,
    ):
        if len(args):
            raise TypeError("GeoTransolverTimeConditional takes keyword arguments only")
        self.rollout_steps: int = int(num_time_steps) - 1
        # Accepted for recipe compatibility (the Yaris datapipe derives its frame
        # interval from ``model.dt``); the backbone itself is time-step agnostic.
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
