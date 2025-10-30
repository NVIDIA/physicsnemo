# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
import numpy as np
import torch
from typing import Any, Callable, Optional

from torch_geometric.data import Data
from torch_geometric.utils import coalesce, add_self_loops

from physicsnemo.datapipes.gnn.utils import load_json, save_json
from physicsnemo.launch.logging import PythonLogger

STATS_DIRNAME = "stats"
NODE_STATS_FILE = "node_stats.json"
FEATURE_STATS_FILE = "feature_stats.json"
EDGE_STATS_FILE = "edge_stats.json"
EPS = 1e-8  # numerical stability for std


class SimSample:
    """
    Unified representation for Simulation data (graph or point cloud).

    Attributes
    ---------
    node_features: dict[str, Tensor] with at least:
      - 'coords': FloatTensor [N, 3]
      - any other feature keys configured, e.g., 'thickness': [N, Fk]
    node_target   : FloatTensor [N, Dout] or [N, (T-1)*3] depending on task
    graph         : PyG Data or None
    """

    def __init__(
        self,
        node_features: dict[str, torch.Tensor],
        node_target: torch.Tensor,
        graph: Optional[Data] = None,
    ):
        assert isinstance(node_features, dict), "node_features must be a dict"
        assert "coords" in node_features, "node_features must contain 'coords'"
        assert (
            node_features["coords"].ndim == 2 and node_features["coords"].shape[1] == 3
        ), f"'coords' must be [N,3], got {node_features['coords'].shape}"
        self.node_features = node_features
        self.node_target = node_target
        self.graph = graph  # PyG Data or None

    def to(self, device: torch.device):
        for k, v in self.node_features.items():
            self.node_features[k] = v.to(device)
        self.node_target = self.node_target.to(device)
        if self.graph is not None:
            self.graph = self.graph.to(device)
        return self

    def is_graph(self) -> bool:
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
        return f"SimSample(N={n}, keys={list(self.node_features.keys())}, Din={din}, Dout={dout}, E={e})"


class CrashBaseDataset:
    """
    Shared base for Crash datasets (graph and point-cloud).

    Responsibilities:
      - Load raw records via `process_d3plot_data`
      - Compute/load node and thickness stats (cached under <data_dir>/stats)
      - Normalize position trajectories and thickness
      - Provide common x/y builder to keep training interchangeable
    """

    def __init__(
        self,
        name: str = "dataset",
        reader: Optional[Callable] = None,
        data_dir: Optional[str] = None,
        split: str = "train",
        num_samples: int = 1000,
        num_steps: int = 400,
        features: Optional[list[str]] = None,
        logger=None,
        dt: float = 5e-3,
    ):
        super().__init__()
        self.name = name
        self.data_dir = data_dir or "."
        self.split = split
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.features = features
        self.length = num_samples
        self.logger = logger or PythonLogger()
        self.dt = dt

        self.logger.info(
            f"[{self.__class__.__name__}] Preparing the {split} dataset..."
        )

        self.features = features or []

        # Prepare stats dir
        self._stats_dir = STATS_DIRNAME
        os.makedirs(STATS_DIRNAME, exist_ok=True)

        # Load raw records via provided reader callable (Hydra can pass a class/callable)
        if reader is None:
            raise ValueError("Data reader function is not specified.")
        self.srcs, self.dsts, point_data = reader(
            data_dir=self.data_dir,
            num_samples=num_samples,
            split=split,
            logger=self.logger,
        )

        # Storage for per-sample tensors
        self.mesh_pos_seq: list[torch.Tensor] = []  # [T,N,3]
        self.node_features_data: list[torch.Tensor] = []  # [N,F]
        self._feature_slices: dict[
            str, tuple[int, int]
        ] = {}  # per-sample feature slices

        for rec in point_data:
            # Coordinates
            if "coords" not in rec:
                raise KeyError(f"Missing coordinates key 'coords' in reader record")
            coords_np = rec["coords"][:num_steps]
            assert coords_np.ndim == 3 and coords_np.shape[-1] == 3, (
                f"coords must be [T,N,3], got {coords_np.shape}"
            )
            self.mesh_pos_seq.append(torch.as_tensor(coords_np, dtype=torch.float32))

            # Features: concatenate requested keys if present; allow empty
            parts = []
            for k in self.features:
                if k not in rec:
                    raise KeyError(f"Missing feature key '{k}' in reader record")
                arr = rec[k]
                if arr.ndim == 1:
                    arr = arr[:, None]
                parts.append(arr)

            feats_np = (
                np.concatenate(parts, axis=-1)
                if len(parts) > 0
                else np.zeros((coords_np.shape[1], 0), dtype=np.float32)
            )
            assert feats_np.ndim == 2 and feats_np.shape[0] == coords_np.shape[1], (
                f"features must be [N,F], got {feats_np.shape}, N mismatch with {coords_np.shape}"
            )

            # build slice map on first record to make future slicing trivial
            if len(self._feature_slices) == 0:
                start = 0
                for k in self.features:
                    width = rec[k].shape[1] if rec[k].ndim > 1 else 1
                    self._feature_slices[k] = (start, start + width)
                    start += width

            self.node_features_data.append(
                torch.as_tensor(feats_np, dtype=torch.float32)
            )

        # Stats (node + generic features)
        node_stats_path = os.path.join(self._stats_dir, NODE_STATS_FILE)
        feat_stats_path = os.path.join(self._stats_dir, FEATURE_STATS_FILE)

        if self.split == "train":
            self.node_stats = self._compute_autoreg_node_stats()
            self.feature_stats = self._compute_feature_stats()
            save_json(self.node_stats, node_stats_path)
            save_json(self.feature_stats, feat_stats_path)
        else:
            if os.path.exists(node_stats_path) and os.path.exists(feat_stats_path):
                self.node_stats = load_json(node_stats_path)
                self.feature_stats = load_json(feat_stats_path)
            else:
                raise FileNotFoundError(
                    f"Node stats file {node_stats_path} or feature stats file {feat_stats_path} not found"
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

    def __len__(self):
        return self.length

    def _xy_shapes(self, idx: int) -> tuple[int, int]:
        T, N, _ = self.mesh_pos_seq[idx].shape
        F = self.node_features_data[idx].shape[1]
        Din = 3 + F
        Dout = (T - 1) * 3
        return Din, Dout

    # Common x/y construction used by both datasets
    def build_xy(self, idx: int):
        """
        x: dict with two keys:
            - 'coords': [N, 3] at t0
            - 'features': [N, F] concatenated in the order given by self.features
        y: [N, (T-1)*3]
        """
        assert 0 <= idx < self.num_samples, f"Index {idx} out of range"
        pos_seq = self.mesh_pos_seq[idx]  # [T,N,3]
        feats = self.node_features_data[idx]  # [N,F]
        T, N, _ = pos_seq.shape
        F = feats.shape[1]

        pos_t0 = pos_seq[0]  # [N,3]
        x = {"coords": pos_t0, "features": feats}

        # Flatten all future positions along feature dim
        y = pos_seq[1:].transpose(0, 1).flatten(start_dim=1)  # [N,(T-1)*3]

        _, Dout = self._xy_shapes(idx)
        assert x["coords"].shape == (N, 3) and x["features"].shape == (N, F), (
            f"coords shape {x['coords'].shape}, features shape {x['features'].shape}, expected (N,3)/(N,{F})"
        )
        assert y.shape == (N, Dout), (
            f"y shape mismatch: expected {(N, Dout)}, got {y.shape}"
        )
        return x, y

    # ---- stats helpers ----
    def _compute_autoreg_node_stats(self):
        """
        Compute per-coordinate stats of normalized kinematics.
        pos_mean/std are computed in raw space then used to normalize velocity/acc.
        """
        dt = self.dt
        pos_mean = torch.zeros(3, dtype=torch.float32)
        pos_meansqr = torch.zeros(3, dtype=torch.float32)

        for i in range(self.num_samples):
            pos = self.mesh_pos_seq[i]  # [T,N,3]
            pos_mean += torch.mean(pos, dim=(0, 1)) / self.num_samples
            pos_meansqr += torch.mean(pos * pos, dim=(0, 1)) / self.num_samples

        pos_var = torch.clamp(pos_meansqr - pos_mean * pos_mean, min=0.0)
        pos_std = torch.sqrt(pos_var + EPS)

        # normalized velocity stats (pos already normalized by pos_std)
        vel_mean = torch.zeros(3, dtype=torch.float32)
        vel_meansqr = torch.zeros(3, dtype=torch.float32)
        for i in range(self.num_samples):
            pos = self.mesh_pos_seq[i]
            vel = (pos[1:] - pos[:-1]) / dt
            vel = vel / pos_std  # normalize per coord
            vel_mean += torch.mean(vel, dim=(0, 1)) / self.num_samples
            vel_meansqr += torch.mean(vel * vel, dim=(0, 1)) / self.num_samples
        vel_var = torch.clamp(vel_meansqr - vel_mean * vel_mean, min=0.0)
        vel_std = torch.sqrt(vel_var + EPS)

        # normalized acceleration stats
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
        # If no features, return empty stats compatible with normalization branch
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

    @staticmethod
    def _normalize_node_tensor(
        invar: torch.Tensor, mu: torch.Tensor, std: torch.Tensor
    ):
        # invar: [T,N,3], mu/std: [3]
        assert invar.shape[-1] == mu.shape[-1] == std.shape[-1] == 3, (
            f"Expected last dim=3, got {invar.shape[-1]} / {mu.shape} / {std.shape}"
        )
        return (invar - mu.view(1, 1, -1)) / (std.view(1, 1, -1) + EPS)

    @staticmethod
    def _normalize_thickness_tensor(
        thickness: torch.Tensor, mu: torch.Tensor, std: torch.Tensor
    ):
        # thickness: [N], mu/std: scalar tensors
        return (thickness - mu) / (std + EPS)


class CrashGraphDataset(CrashBaseDataset):
    """
    Graph version:
      - Builds PyG graphs (create_graph + add_self_loop)
      - Computes/loads edge stats and normalizes edge features
      - Returns SimSample with edge_index/edge_features
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Filter self-edges and create graphs
        _srcs, _dsts = [], []
        for src, dst in zip(self.srcs, self.dsts):
            mask = src != dst
            _srcs.append(np.asarray(src)[mask])
            _dsts.append(np.asarray(dst)[mask])
        self.srcs, self.dsts = _srcs, _dsts

        self.graphs: list[Data] = []
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

        # Edge stats
        edge_stats_path = os.path.join(self._stats_dir, EDGE_STATS_FILE)
        if self.split == "train":
            self.edge_stats = self._compute_edge_stats()
            save_json(self.edge_stats, edge_stats_path)
        else:
            if os.path.exists(edge_stats_path):
                self.edge_stats = load_json(edge_stats_path)
            else:
                raise FileNotFoundError(f"Edge stats file {edge_stats_path} not found")

        # Convert loaded stats to tensors
        self.edge_stats["edge_mean"] = torch.as_tensor(
            self.edge_stats["edge_mean"], dtype=torch.float32
        )
        self.edge_stats["edge_std"] = torch.as_tensor(
            self.edge_stats["edge_std"], dtype=torch.float32
        )

        # Normalize edge features
        for i in range(self.num_samples):
            self.graphs[i].edge_attr = self._normalize_edge(
                self.graphs[i].edge_attr,
                self.edge_stats["edge_mean"],
                self.edge_stats["edge_std"],
            )

    def __getitem__(self, idx: int):
        assert 0 <= idx < self.num_samples, f"Index {idx} out of range"
        g = self.graphs[idx]
        x, y = self.build_xy(idx)  # [N,3+F], [N,(T-1)*3]

        return SimSample(
            node_features=x,
            node_target=y,
            graph=g,
        )

    # ----- graph-specific helpers -----
    @staticmethod
    def create_graph(src, dst, num_nodes: int, dtype=torch.long):
        src = torch.as_tensor(src, dtype=dtype)
        dst = torch.as_tensor(dst, dtype=dtype)
        edge_index = torch.stack(
            [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
        )  # [2, E]
        edge_index, _ = coalesce(edge_index, None, num_nodes=num_nodes)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        return Data(edge_index=edge_index, num_nodes=num_nodes)

    @staticmethod
    def add_edge_features(data: Data, pos: torch.Tensor) -> Data:
        # pos: [N,3] (denormalized)
        row, col = data.edge_index
        pos_t = torch.as_tensor(pos, dtype=torch.float32)
        disp = pos_t[row] - pos_t[col]  # [E,3]
        disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)  # [E,1]
        data.edge_attr = torch.cat((disp, disp_norm), dim=1)  # [E,4]
        return data

    def _compute_edge_stats(self):
        edge_mean = None
        edge_meansqr = None
        for i in range(self.num_samples):
            x_e = self.graphs[i].edge_attr.to(torch.float32)  # [E,De]
            m = torch.mean(x_e, dim=0)
            msq = torch.mean(x_e * x_e, dim=0)
            edge_mean = m if edge_mean is None else edge_mean + m / self.num_samples
            edge_meansqr = (
                msq if edge_meansqr is None else edge_meansqr + msq / self.num_samples
            )

        edge_var = torch.clamp(edge_meansqr - edge_mean * edge_mean, min=0.0)
        edge_std = torch.sqrt(edge_var + EPS)
        return {
            "edge_mean": edge_mean,
            "edge_std": edge_std,
        }

    @staticmethod
    def _normalize_edge(edge_x: torch.Tensor, mu: torch.Tensor, std: torch.Tensor):
        assert edge_x.shape[-1] == mu.shape[-1] == std.shape[-1], (
            f"Edge feature dim mismatch: {edge_x.shape[-1]} vs {mu.shape[-1]} / {std.shape[-1]}"
        )
        return (edge_x - mu.view(1, -1)) / (std.view(1, -1) + EPS)


class CrashPointCloudDataset(CrashBaseDataset):
    """
    Point-cloud version:
      - No graphs or edges
      - Returns SimSample with node_features, node_target
      - Provides empty edge_stats dict for compatibility
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_stats: dict[str, Any] = {}

    def __getitem__(self, idx: int):
        assert 0 <= idx < self.num_samples, f"Index {idx} out of range"
        x, y = self.build_xy(idx)
        return SimSample(node_features=x, node_target=y)


def simsample_collate(batch: list[SimSample]) -> list[SimSample]:
    """
    Keep samples as a list (variable N per item is common here).
    Models should iterate the list or implement internal padding.
    """
    return batch
