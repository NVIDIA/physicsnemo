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

import os
import re
import numpy as np
import torch
from typing import Any, Callable, Optional

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.datapipes.gnn.utils import load_json, save_json

# Lazy imports for graph datapipe (PyG only loaded when DropTestGraphDataset is used)
_pyg_data = OptionalImport("torch_geometric.data")
_pyg_utils = OptionalImport("torch_geometric.utils")
from physicsnemo.utils.logging import PythonLogger

NODE_STATS_FILE = "node_stats.json"
FEATURE_STATS_FILE = "feature_stats.json"
EDGE_STATS_FILE = "edge_stats.json"
DYNAMIC_TARGET_STATS_FILE = "dynamic_target_stats.json"
EPS = 1e-8  # numerical stability for std


class SimSample:
    """
    Unified representation for Simulation data (graph or point cloud).

    Attributes
    ---------
    node_features: dict[str, Tensor] with at least:
      - 'coords': FloatTensor [N, 3]
      - any other feature keys configured, e.g., 'thickness': [N, Fk]
    node_target   : FloatTensor [N, T, Fo] where T=rollout steps, Fo=3+sum(C_k)
    target_series : Optional[dict[str, Tensor]] mapping name -> [T, N] or [T, N, C]
    graph         : PyG Data or None
    """

    def __init__(
        self,
        node_features: dict[str, torch.Tensor],
        node_target: torch.Tensor,
        graph=None,
        global_features: Optional[dict[str, torch.Tensor]] = None,
        target_series: Optional[dict[str, torch.Tensor]] = None,
    ):
        assert isinstance(node_features, dict), "node_features must be a dict"
        assert "coords" in node_features, "node_features must contain 'coords'"
        assert (
            node_features["coords"].ndim == 2 and node_features["coords"].shape[1] == 3
        ), f"'coords' must be [N,3], got {node_features['coords'].shape}"
        self.node_features = node_features
        self.node_target = node_target
        self.graph = graph  # PyG Data or None
        self.global_features = global_features
        self.target_series = target_series

    def to(self, device: torch.device):
        """Move all contained tensors (and graph) to ``device`` in place."""
        for k, v in self.node_features.items():
            self.node_features[k] = v.to(device)
        self.node_target = self.node_target.to(device)
        if self.graph is not None:
            self.graph = self.graph.to(device)
        if self.global_features is not None:
            self.global_features = {
                k: v.to(device) for k, v in self.global_features.items()
            }
        return self

    def is_graph(self) -> bool:
        """Whether this sample carries a PyG graph (vs. point cloud only)."""
        return self.graph is not None

    def __repr__(self) -> str:
        n = self.node_features["coords"].shape[0]
        keys = {k: tuple(v.shape) for k, v in self.node_features.items()}
        din = 3
        for k, v in self.node_features.items():
            if k != "coords":
                din += v.shape[1]
        dout = (
            self.node_target.shape[1]
            if self.node_target.ndim == 2
            else tuple(self.node_target.shape[1:])
        )
        e = 0 if self.graph is None else self.graph.num_edges
        gf = (
            ""
            if self.global_features is None
            else f", global_features={list(self.global_features.keys())}"
        )
        ts = (
            ""
            if self.target_series is None
            else f", target_series={list(self.target_series.keys())}"
        )
        return f"SimSample(N={n}, keys={list(self.node_features.keys())}, Din={din}, Dout={dout}, E={e}{gf}{ts})"


class DropTestBaseDataset:
    """
    Shared base for Drop Test datasets (graph and point-cloud).

    Responsibilities:
      - Load raw records via reader (VTU)
      - Compute/load node and feature stats (cached under <data_dir>/stats)
      - Normalize position trajectories and features
      - Provide common x/y builder to keep training interchangeable
    """

    def __init__(
        self,
        name: str = "dataset",
        reader: Optional[Callable] = None,
        data_dir: Optional[str] = None,
        global_features_filepath: Optional[str] = None,
        global_features: Optional[list[str]] = None,
        split: str = "train",
        num_samples: int = 1000,
        num_steps: int = 400,
        static_features: Optional[list[str]] = None,
        dynamic_features: Optional[list[str]] = None,
        dynamic_targets: Optional[list[str]] = None,
        logger=None,
        dt: float = 5e-3,
        stats_dir: str = "stats",
        sample_type: str = "all_time_steps",
        log_transform_targets: bool = False,
    ):
        super().__init__()
        self.name = name
        self.data_dir = data_dir or "."
        self.global_features_filepath = global_features_filepath
        self.global_features_keys = global_features
        self.split = split
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.static_features = static_features if static_features is not None else []
        self.dynamic_features = dynamic_features or []
        self.dynamic_targets = dynamic_targets or []
        self.length = num_samples
        self.logger = logger or PythonLogger()
        self.dt = dt
        self.sample_type = sample_type
        self.log_transform_targets = log_transform_targets

        if sample_type not in ["all_time_steps", "one_time_step"]:
            raise ValueError(
                f"Invalid sample type: {sample_type} Expected 'all_time_steps' or 'one_time_step'"
            )

        # Precompute batch_idx logic
        rollout_steps = num_steps - 1
        if sample_type == "one_time_step":
            self._max_idx = num_samples * rollout_steps
            self._resolve_idx = lambda idx: (idx // rollout_steps, idx % rollout_steps)
        else:
            self._max_idx = num_samples
            self._resolve_idx = lambda idx: (idx, None)

        self.logger.info(
            f"[{self.__class__.__name__}] Preparing the {split} dataset..."
        )

        # Prepare stats dir
        self._stats_dir = stats_dir
        os.makedirs(self._stats_dir, exist_ok=True)

        # Load raw records via provided reader callable (Hydra can pass a class/callable)
        if reader is None:
            raise ValueError("Data reader function is not specified.")

        # Require global_features_filepath when global_features keys are configured
        if global_features and len(global_features) > 0:
            if (
                not global_features_filepath
                or not str(global_features_filepath).strip()
            ):
                raise ValueError(
                    "datapipe.global_features is configured but training.global_features_filepath "
                    "is not set or is empty. Set it via config or CLI, e.g. "
                    "training.global_features_filepath=/path/to/global_features.json"
                )
            if str(global_features_filepath).strip() == "???":
                raise ValueError(
                    "datapipe.global_features is configured but training.global_features_filepath "
                    "is unresolved (???). Set it via config or CLI, e.g. "
                    "training.global_features_filepath=/path/to/global_features.json"
                )

        self.srcs, self.dsts, point_data, global_features = reader(
            data_dir=self.data_dir,
            num_samples=num_samples,
            split=split,
            global_features_filepath=self.global_features_filepath,
            logger=self.logger,
        )

        # Reconcile num_samples with what the reader actually returned: the reader may
        # yield fewer records than requested (e.g. a val/test dir with fewer files),
        # and every downstream loop (stats computation, normalization, __getitem__)
        # must iterate over the real count, not the requested count.
        actual = len(point_data)
        if actual < num_samples:
            self.logger.warning(
                f"Reader returned {actual} records but num_samples={num_samples} was "
                f"requested; using {actual}."
            )
            self.num_samples = actual
            rollout_steps = num_steps - 1
            if sample_type == "one_time_step":
                self._max_idx = self.num_samples * rollout_steps
            else:
                self._max_idx = self.num_samples
            self.length = self.num_samples
        # Check if any global features are present
        has_global = global_features and any(gf for gf in global_features)
        if not has_global:
            self.global_features = None
        else:
            if self.global_features_keys is None:
                raise ValueError(
                    "global_features_filepath is set, but no global_features keys were specified"
                )

            for i, gf in enumerate(global_features):
                missing = set(self.global_features_keys) - gf.keys()
                if missing:
                    raise KeyError(
                        f"Missing global features {missing} "
                        f"for sample {i}. Available: {list(gf.keys())}"
                    )
                global_features[i] = {k: gf[k] for k in self.global_features_keys}

            self.global_features = global_features

        # Storage for per-sample tensors
        self.mesh_pos_seq: list[torch.Tensor] = []  # [T,N,3]
        self.node_features_data: list[torch.Tensor] = []  # [N,F]
        self._feature_slices: dict[str, tuple[int, int]] = {}
        self.target_series_data: list[dict[str, torch.Tensor]] = []

        for rec in point_data:
            if "coords" not in rec:
                raise KeyError(f"Missing coordinates key 'coords' in reader record")
            coords_np = rec["coords"][:num_steps]
            assert coords_np.ndim == 3 and coords_np.shape[-1] == 3, (
                f"coords must be [T,N,3], got {coords_np.shape}"
            )
            self.mesh_pos_seq.append(torch.as_tensor(coords_np, dtype=torch.float32))

            parts = []
            for k in self.static_features:
                arr = self._get_static_feature(rec, k)
                if arr.ndim == 1:
                    arr = arr[:, None]
                parts.append(arr)
            T = coords_np.shape[0]
            for k in self.dynamic_features:
                dyn = self._get_dynamic_feature(rec, k, T)
                if dyn.ndim == 2:
                    dyn_flat = dyn.transpose(1, 0)
                else:
                    dyn_flat = dyn.transpose(1, 0, 2).reshape(dyn.shape[1], -1)
                parts.append(dyn_flat)

            feats_np = (
                np.concatenate(parts, axis=-1)
                if len(parts) > 0
                else np.zeros((coords_np.shape[1], 0), dtype=np.float32)
            )
            assert feats_np.ndim == 2 and feats_np.shape[0] == coords_np.shape[1], (
                f"features must be [N,F], got {feats_np.shape}, N mismatch with {coords_np.shape}"
            )

            if len(self._feature_slices) == 0:
                start = 0
                for k in self.static_features:
                    arr_k = self._get_static_feature(rec, k)
                    width = arr_k.shape[1] if arr_k.ndim > 1 else 1
                    self._feature_slices[k] = (start, start + width)
                    start += width
                for k in self.dynamic_features:
                    dyn_k = self._get_dynamic_feature(rec, k, T)
                    width = (
                        dyn_k.shape[0]
                        if dyn_k.ndim == 2
                        else dyn_k.shape[0] * dyn_k.shape[2]
                    )
                    self._feature_slices[k] = (start, start + width)
                    start += width

            self.node_features_data.append(
                torch.as_tensor(feats_np, dtype=torch.float32)
            )

            target_series_rec: dict[str, torch.Tensor] = {}
            for k in self.dynamic_targets:
                dyn = self._get_dynamic_feature(rec, k, T)
                t = torch.as_tensor(dyn, dtype=torch.float32)
                if self.log_transform_targets:
                    t = torch.log1p(t.clamp(min=0.0))
                target_series_rec[k] = t
            self.target_series_data.append(target_series_rec)

        if self.log_transform_targets:
            self.logger.info(
                f"[{self.__class__.__name__}] log1p transform applied to dynamic targets: "
                f"{self.dynamic_targets}"
            )

        # Stats (node + generic features)
        node_stats_path = os.path.join(self._stats_dir, NODE_STATS_FILE)
        feat_stats_path = os.path.join(self._stats_dir, FEATURE_STATS_FILE)

        if self.split == "train":
            self.node_stats = self._compute_autoreg_node_stats()
            self.feature_stats = self._compute_feature_stats()
            save_json(self.node_stats, node_stats_path)
            save_json(self.feature_stats, feat_stats_path)
            if self.dynamic_targets:
                self.dynamic_target_stats = self._compute_dynamic_target_stats()
                dyn_stats_path = os.path.join(
                    self._stats_dir, DYNAMIC_TARGET_STATS_FILE
                )
                save_json(self.dynamic_target_stats, dyn_stats_path)
            else:
                self.dynamic_target_stats = {}
        else:
            if os.path.exists(node_stats_path) and os.path.exists(feat_stats_path):
                self.node_stats = load_json(node_stats_path)
                self.feature_stats = load_json(feat_stats_path)
            else:
                raise FileNotFoundError(
                    f"Node stats file {node_stats_path} or feature stats file {feat_stats_path} not found"
                )
            dyn_stats_path = os.path.join(self._stats_dir, DYNAMIC_TARGET_STATS_FILE)
            if self.dynamic_targets and os.path.exists(dyn_stats_path):
                self.dynamic_target_stats = load_json(dyn_stats_path)
            else:
                self.dynamic_target_stats = {}
                if self.dynamic_targets and self.logger:
                    self.logger.warning(
                        f"dynamic_targets={self.dynamic_targets} but "
                        f"{dyn_stats_path} not found. Run training first to generate stats, "
                        "or targets will remain unnormalized."
                    )

        # Normalize trajectories and features
        for i in range(self.num_samples):
            self.mesh_pos_seq[i] = self._normalize_node_tensor(
                self.mesh_pos_seq[i],
                self.node_stats["pos_mean"],
                self.node_stats["pos_std"],
            )
            if self.node_features_data[i].numel() > 0:
                mu = torch.as_tensor(
                    self.feature_stats.get("feature_mean", []), dtype=torch.float32
                )
                std = torch.as_tensor(
                    self.feature_stats.get("feature_std", []), dtype=torch.float32
                )
                if mu.numel() == 0:
                    continue
                self.node_features_data[i] = (
                    self.node_features_data[i] - mu.view(1, -1)
                ) / (std.view(1, -1) + EPS)
            # Normalize dynamic targets
            if self.dynamic_target_stats:
                for k in self.dynamic_targets:
                    if f"{k}_mean" not in self.dynamic_target_stats:
                        continue
                    mu = self.dynamic_target_stats[f"{k}_mean"].flatten()
                    std = self.dynamic_target_stats[f"{k}_std"].flatten()
                    ts = self.target_series_data[i][k]
                    if ts.ndim == 2:
                        ts = ts.unsqueeze(-1)
                    # ts: [T, N, C], mu/std: [C]
                    self.target_series_data[i][k] = (ts - mu.view(1, 1, -1)) / (
                        std.view(1, 1, -1) + EPS
                    )

    def __len__(self):
        return self._max_idx

    def build_xy(self, batch_idx: int, time_idx: int | None):
        """
        x: dict with:
            - 'coords': [N, 3] at t0
            - 'features': [N, F] concatenated (static + flattened dynamic)
        y: [N, T, Fo] where T=rollout steps, Fo=3+sum(C_k) per timestep
        """
        assert 0 <= batch_idx < self.num_samples, f"batch_idx {batch_idx} out of range"
        if time_idx is not None:
            assert 0 <= time_idx < self.num_steps - 1, (
                f"time_idx {time_idx} out of range [0, {self.num_steps - 1})"
            )
        pos_seq = self.mesh_pos_seq[batch_idx]
        feats = self.node_features_data[batch_idx]
        T, N, _ = pos_seq.shape
        F = feats.shape[1]

        pos_t0 = pos_seq[0]
        x = {"coords": pos_t0, "features": feats}

        pos_rollout = pos_seq[1:]

        if len(self.dynamic_targets) > 0:
            ts_rec = self.target_series_data[batch_idx]
            dyn_list = []
            for k in self.dynamic_targets:
                if k not in ts_rec:
                    raise KeyError(
                        f"Missing dynamic target '{k}' for sample {batch_idx}"
                    )
                series = ts_rec[k]
                if series.ndim == 2:
                    series = series.unsqueeze(-1)
                series_rollout = series[1:]
                dyn_list.append(series_rollout)

            y_per_t = torch.cat([pos_rollout] + dyn_list, dim=-1)
        else:
            y_per_t = pos_rollout

        y = y_per_t.transpose(0, 1)

        if time_idx is not None:
            x["time"] = torch.tensor(time_idx / (self.num_steps - 1))
            y = y[:, time_idx]

            Fo = y.shape[-1]
            assert x["coords"].shape == (N, 3) and x["features"].shape == (N, F), (
                f"coords shape {x['coords'].shape}, features shape {x['features'].shape}, expected (N,3)/(N,{F})"
            )
            assert y.shape == (N, Fo), (
                f"target shape {y.shape} does not match expected (N={N}, Fo={Fo})"
            )

        else:
            T_out, Fo = y.shape[1], y.shape[2]
            assert x["coords"].shape == (N, 3) and x["features"].shape == (N, F), (
                f"coords shape {x['coords'].shape}, features shape {x['features'].shape}, expected (N,3)/(N,{F})"
            )
            assert y.shape == (N, T_out, Fo), (
                f"target shape {y.shape} does not match expected (N={N}, T={T_out}, Fo={Fo})"
            )
        return x, y

    def _compute_autoreg_node_stats(self):
        dt = self.dt
        pos_mean = torch.zeros(3, dtype=torch.float32)
        pos_meansqr = torch.zeros(3, dtype=torch.float32)

        for i in range(self.num_samples):
            pos = self.mesh_pos_seq[i]
            pos_mean += torch.mean(pos, dim=(0, 1)) / self.num_samples
            pos_meansqr += torch.mean(pos * pos, dim=(0, 1)) / self.num_samples

        pos_var = torch.clamp(pos_meansqr - pos_mean * pos_mean, min=0.0)
        pos_std = torch.sqrt(pos_var + EPS)

        vel_mean = torch.zeros(3, dtype=torch.float32)
        vel_meansqr = torch.zeros(3, dtype=torch.float32)
        for i in range(self.num_samples):
            pos = self.mesh_pos_seq[i]
            vel = (pos[1:] - pos[:-1]) / dt
            vel = vel / pos_std
            vel_mean += torch.mean(vel, dim=(0, 1)) / self.num_samples
            vel_meansqr += torch.mean(vel * vel, dim=(0, 1)) / self.num_samples
        vel_var = torch.clamp(vel_meansqr - vel_mean * vel_mean, min=0.0)
        vel_std = torch.sqrt(vel_var + EPS)

        acc_mean = torch.zeros(3, dtype=torch.float32)
        acc_meansqr = torch.zeros(3, dtype=torch.float32)
        for i in range(self.num_samples):
            pos = self.mesh_pos_seq[i]
            acc = (pos[:-2] + pos[2:] - 2 * pos[1:-1]) / (dt * dt)
            acc = acc / pos_std
            acc_mean += torch.mean(acc, dim=(0, 1)) / self.num_samples
            acc_meansqr += torch.mean(acc * acc, dim=(0, 1)) / self.num_samples
        acc_var = torch.clamp(acc_meansqr - acc_mean * acc_mean, min=0.0)
        acc_std = torch.sqrt(acc_var + EPS)

        return {
            "pos_mean": pos_mean,
            "pos_std": pos_std,
            "norm_vel_mean": vel_mean,
            "norm_vel_std": vel_std,
            "norm_acc_mean": acc_mean,
            "norm_acc_std": acc_std,
        }

    def _compute_feature_stats(self):
        fdim = self.node_features_data[0].shape[1]
        for t in self.node_features_data:
            assert t.shape[1] == fdim, f"Feature dim mismatch: {t.shape[1]} vs {fdim}"

        if fdim == 0:
            mu = torch.zeros(0, dtype=torch.float32)
            std = torch.ones(0, dtype=torch.float32)
            return {"feature_mean": mu, "feature_std": std}

        feat_mean = torch.zeros(fdim, dtype=torch.float32)
        feat_meansqr = torch.zeros(fdim, dtype=torch.float32)
        for i in range(self.num_samples):
            x = self.node_features_data[i].to(torch.float32)
            m = torch.mean(x, dim=0)
            msq = torch.mean(x * x, dim=0)
            feat_mean += m / self.num_samples
            feat_meansqr += msq / self.num_samples
        feat_var = torch.clamp(feat_meansqr - feat_mean * feat_mean, min=0.0)
        feat_std = torch.sqrt(feat_var + EPS)
        return {"feature_mean": feat_mean, "feature_std": feat_std}

    def _compute_dynamic_target_stats(self):
        """Compute mean and std for each dynamic target over the training set."""
        out = {}
        for k in self.dynamic_targets:
            all_vals = []
            for i in range(self.num_samples):
                ts = self.target_series_data[i][k]
                if ts.ndim == 2:
                    ts = ts.unsqueeze(-1)
                all_vals.append(ts)
            stacked = torch.cat([v.reshape(-1, v.shape[-1]) for v in all_vals], dim=0)
            mu = torch.mean(stacked, dim=0)
            var = torch.clamp(torch.mean(stacked * stacked, dim=0) - mu * mu, min=0.0)
            std = torch.sqrt(var + EPS)
            out[f"{k}_mean"] = mu
            out[f"{k}_std"] = std
        return out

    @staticmethod
    def _normalize_node_tensor(
        invar: torch.Tensor, mu: torch.Tensor, std: torch.Tensor
    ):
        assert invar.shape[-1] == mu.shape[-1] == std.shape[-1] == 3, (
            f"Expected last dim=3, got {invar.shape[-1]} / {mu.shape} / {std.shape}"
        )
        return (invar - mu.view(1, 1, -1)) / (std.view(1, 1, -1) + EPS)

    @staticmethod
    def _get_static_feature(rec: dict, key: str) -> np.ndarray:
        if key in rec:
            return np.asarray(rec[key])
        pd = rec.get("point_data", {})
        if key in pd:
            return np.asarray(pd[key])
        raise KeyError(
            f"Missing static feature key '{key}' in reader record (checked top-level and point_data)"
        )

    @staticmethod
    def _get_dynamic_feature(rec: dict, key: str, T: int) -> np.ndarray:
        pd = rec.get("point_data", {})
        prefix = f"{key}_t"
        names = [name for name in pd.keys() if name.startswith(prefix)]
        if not names:
            raise KeyError(f"Missing dynamic feature series for '{key}' in point_data")

        def natural_key(name):
            return [
                int(s) if s.isdigit() else s.lower()
                for s in re.findall(r"\d+|\D+", name)
            ]

        names = sorted(names, key=natural_key)
        series = [np.asarray(pd[n]) for n in names]
        series = [x[:, None] if x.ndim == 1 else x for x in series]
        N = series[0].shape[0]
        C = series[0].shape[1]
        for x in series:
            assert x.shape[0] == N and x.shape[1] == C, (
                f"Inconsistent shapes in dynamic feature '{key}': "
                f"expected (N={N},C={C}), got {x.shape}"
            )
        arr = np.stack(series, axis=0)
        Tprime = arr.shape[0]
        if Tprime >= T:
            arr = arr[:T]
        else:
            pad = np.repeat(arr[-1:], repeats=(T - Tprime), axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        return arr


class DropTestGraphDataset(DropTestBaseDataset):
    """
    Graph version for drop test:
      - Builds PyG graphs (create_graph + add_self_loop)
      - Computes/loads edge stats and normalizes edge features
      - Returns SimSample with graph, global_features, and target_series
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        _srcs, _dsts = [], []
        for src, dst in zip(self.srcs, self.dsts):
            mask = src != dst
            _srcs.append(np.asarray(src)[mask])
            _dsts.append(np.asarray(dst)[mask])
        self.srcs, self.dsts = _srcs, _dsts

        Data = _pyg_data.Data
        self.graphs = []
        for i in range(self.num_samples):
            g = self.create_graph(
                self.srcs[i],
                self.dsts[i],
                num_nodes=self.mesh_pos_seq[i][0].shape[0],
                dtype=torch.long,
            )
            pos0 = self.mesh_pos_seq[i][0]
            g = self.add_edge_features(g, pos0)
            self.graphs.append(g)

        edge_stats_path = os.path.join(self._stats_dir, EDGE_STATS_FILE)
        if self.split == "train":
            self.edge_stats = self._compute_edge_stats()
            save_json(self.edge_stats, edge_stats_path)
        else:
            if os.path.exists(edge_stats_path):
                self.edge_stats = load_json(edge_stats_path)
            else:
                raise FileNotFoundError(f"Edge stats file {edge_stats_path} not found")

        self.edge_stats["edge_mean"] = torch.as_tensor(
            self.edge_stats["edge_mean"], dtype=torch.float32
        )
        self.edge_stats["edge_std"] = torch.as_tensor(
            self.edge_stats["edge_std"], dtype=torch.float32
        )

        for i in range(self.num_samples):
            self.graphs[i].edge_attr = self._normalize_edge(
                self.graphs[i].edge_attr,
                self.edge_stats["edge_mean"],
                self.edge_stats["edge_std"],
            )

    def __getitem__(self, idx: int):
        assert 0 <= idx < self._max_idx, f"Index {idx} out of range"
        batch_idx, time_idx = self._resolve_idx(idx)
        g = self.graphs[batch_idx]
        x, y = self.build_xy(batch_idx, time_idx)
        if self.global_features is not None:
            gf = {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in self.global_features[batch_idx].items()
            }
        else:
            gf = None
        ts = None if time_idx is not None else self.target_series_data[batch_idx]
        return SimSample(
            node_features=x,
            node_target=y,
            graph=g,
            global_features=gf,
            target_series=ts,
        )

    @staticmethod
    def create_graph(src, dst, num_nodes: int, dtype=torch.long):
        """Build a coalesced, self-loop-augmented undirected PyG graph from ``src``/``dst``."""
        src = torch.as_tensor(src, dtype=dtype)
        dst = torch.as_tensor(dst, dtype=dtype)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        edge_index, _ = _pyg_utils.coalesce(edge_index, None, num_nodes=num_nodes)
        edge_index, _ = _pyg_utils.add_self_loops(edge_index, num_nodes=num_nodes)
        return _pyg_data.Data(edge_index=edge_index, num_nodes=num_nodes)

    @staticmethod
    def add_edge_features(data, pos: torch.Tensor):
        """Attach per-edge (displacement, |displacement|) features to ``data.edge_attr``."""
        row, col = data.edge_index
        pos_t = torch.as_tensor(pos, dtype=torch.float32)
        disp = pos_t[row] - pos_t[col]
        disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
        data.edge_attr = torch.cat((disp, disp_norm), dim=1)
        return data

    def _compute_edge_stats(self):
        edge_dim = self.graphs[0].edge_attr.shape[-1]
        edge_mean = torch.zeros(edge_dim, dtype=torch.float32)
        edge_meansqr = torch.zeros(edge_dim, dtype=torch.float32)
        for i in range(self.num_samples):
            x_e = self.graphs[i].edge_attr.to(torch.float32)
            m = torch.mean(x_e, dim=0)
            msq = torch.mean(x_e * x_e, dim=0)
            edge_mean += m / self.num_samples
            edge_meansqr += msq / self.num_samples

        edge_var = torch.clamp(edge_meansqr - edge_mean * edge_mean, min=0.0)
        edge_std = torch.sqrt(edge_var + EPS)
        return {"edge_mean": edge_mean, "edge_std": edge_std}

    @staticmethod
    def _normalize_edge(edge_x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor):
        assert edge_x.shape[-1] == mu.shape[-1] == std.shape[-1], (
            f"Edge feature dim mismatch: {edge_x.shape[-1]} vs {mu.shape[-1]} / {std.shape[-1]}"
        )
        return (edge_x - mu.view(1, -1)) / (std.view(1, -1) + EPS)


class DropTestPointCloudDataset(DropTestBaseDataset):
    """
    Point-cloud version for drop test:
      - No graphs or edges
      - Returns SimSample with node_features, node_target
      - Provides empty edge_stats dict for compatibility
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_stats: dict[str, Any] = {}

    def __getitem__(self, idx: int):
        assert 0 <= idx < self._max_idx, f"Index {idx} out of range"
        batch_idx, time_idx = self._resolve_idx(idx)
        x, y = self.build_xy(batch_idx, time_idx)
        if self.global_features is not None:
            gf = {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in self.global_features[batch_idx].items()
            }
        else:
            gf = None
        ts = None if time_idx is not None else self.target_series_data[batch_idx]
        return SimSample(
            node_features=x,
            node_target=y,
            global_features=gf,
            target_series=ts,
        )


def simsample_collate(batch: list[SimSample]) -> list[SimSample]:
    """
    Keep samples as a list (variable N per item is common here).
    Models should iterate the list or implement internal padding.
    """
    return batch
