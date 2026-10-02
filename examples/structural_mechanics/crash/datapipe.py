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

import copy
import os
import re
from typing import Any, Callable, Optional

import numpy as np
import torch

from physicsnemo.core.version_check import OptionalImport
from physicsnemo.datapipes.gnn.utils import load_json, save_json

# Lazy imports for graph datapipe (PyG only loaded when CrashGraphDataset is used)
_pyg_data = OptionalImport("torch_geometric.data")
_pyg_utils = OptionalImport("torch_geometric.utils")
from physicsnemo.utils.logging import PythonLogger

NODE_STATS_FILE = "node_stats.json"
FEATURE_STATS_FILE = "feature_stats.json"
EDGE_STATS_FILE = "edge_stats.json"
GLOBAL_STATS_FILE = "global_stats.json"
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
        """Move sample tensors without mutating the cached graph attribute store."""
        for k, v in self.node_features.items():
            self.node_features[k] = v.to(device)
        self.node_target = self.node_target.to(device)
        if self.graph is not None:
            # PyG Data.to mutates its attribute store in place. The graph may
            # belong to CrashGraphDataset.graphs, so isolate the store before
            # transfer; otherwise every visited graph remains cached on GPU.
            # Data.__copy__ copies stores without duplicating tensor storage.
            self.graph = copy.copy(self.graph).to(device)
        if self.global_features is not None:
            self.global_features = {
                k: v.to(device) for k, v in self.global_features.items()
            }
        return self

    def is_graph(self) -> bool:
        """Return whether this sample carries structural graph connectivity."""
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
        stats_mode: str = "compute",
        sample_type: str = "all_time_steps",
        initial_history_steps: int = 1,
        rollout_window_steps: Optional[int] = None,
        windows_per_sample: int = 1,
        window_seed: int | None = None,
        contact_exclusion_hops: int | None = None,
        contact_require_elements: bool = False,
        contact_surface: bool = False,
        contact_surface_exclusion: str = "incidence",
        contact_geodesic_distance_scale: float = 2.0**0.5,
        contact_geodesic_gap_scale: float = 1.0,
        contact_geodesic_gap_min: float = 0.0,
        contact_geodesic_max_pairs: int = 16_000_000,
        contact_geodesic_cache_dir: Optional[str] = None,
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
        self.initial_history_steps = int(initial_history_steps)
        self.rollout_window_steps = (
            None if rollout_window_steps is None else int(rollout_window_steps)
        )
        self.windows_per_sample = int(windows_per_sample)
        self.window_seed = window_seed
        self.epoch = 0
        self.contact_exclusion_hops = contact_exclusion_hops
        self.contact_require_elements = bool(contact_require_elements)
        self.contact_surface = bool(contact_surface)
        if contact_surface_exclusion not in (
            "incidence",
            "one_ring",
            "reference_geodesic",
        ):
            raise ValueError(
                "contact_surface_exclusion must be incidence, one_ring or reference_geodesic"
            )
        if not contact_surface and contact_surface_exclusion != "incidence":
            raise ValueError("surface exclusions require contact_surface=True")
        self.contact_surface_exclusion = contact_surface_exclusion
        self.contact_geodesic_options = dict(
            distance_scale=contact_geodesic_distance_scale,
            gap_scale=contact_geodesic_gap_scale,
            gap_min=contact_geodesic_gap_min,
            max_pairs=contact_geodesic_max_pairs,
        )
        self.contact_geodesic_cache_dir = contact_geodesic_cache_dir
        if contact_surface and contact_exclusion_hops is not None:
            raise ValueError(
                "surface contact uses contact_surface_exclusion, not node hop exclusions"
            )
        if contact_exclusion_hops is not None and contact_exclusion_hops not in (1, 2):
            raise ValueError("contact_exclusion_hops must be 1, 2, or None")
        if (
            contact_require_elements
            and contact_exclusion_hops is None
            and not contact_surface
        ):
            raise ValueError("contact_require_elements requires contact_exclusion_hops")
        if stats_mode not in {"compute", "load"}:
            raise ValueError("stats_mode must be either 'compute' or 'load'")
        self.stats_mode = stats_mode

        valid_sample_types = {
            "all_time_steps",
            "one_time_step",
            "random_time_window",
        }
        if sample_type not in valid_sample_types:
            raise ValueError(
                f"Invalid sample type: {sample_type}. Expected one of "
                f"{sorted(valid_sample_types)}"
            )
        if not 1 <= self.initial_history_steps < num_steps:
            raise ValueError(
                "initial_history_steps must be in [1, num_steps); got "
                f"{self.initial_history_steps} for num_steps={num_steps}"
            )
        if self.initial_history_steps > 2:
            raise ValueError(
                "Only one- and two-frame initialization are currently supported"
            )
        if self.windows_per_sample <= 0:
            raise ValueError("windows_per_sample must be positive")

        # Precompute batch_idx logic
        rollout_steps = num_steps - self.initial_history_steps
        if sample_type == "one_time_step":
            self._max_idx = num_samples * rollout_steps
            self._resolve_idx = lambda idx: (idx // rollout_steps, idx % rollout_steps)
        elif sample_type == "random_time_window":
            if self.rollout_window_steps is None:
                raise ValueError(
                    "rollout_window_steps is required for random_time_window"
                )
            if not 1 <= self.rollout_window_steps <= rollout_steps:
                raise ValueError(
                    "rollout_window_steps must be in [1, "
                    f"{rollout_steps}], got {self.rollout_window_steps}"
                )
            self._max_idx = num_samples * self.windows_per_sample
            self._resolve_idx = lambda idx: (idx % num_samples, None)
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
        self.mesh_cells = []
        reference_cells = None
        for record in point_data:
            cells = record.get("mesh_cells")
            if (contact_require_elements or contact_surface) and cells is None:
                raise ValueError(
                    "Contact requires element connectivity; enable reader.include_contact_topology"
                )
            if cells is not None:
                if reference_cells is None:
                    reference_cells = cells
                elif np.array_equal(cells, reference_cells):
                    cells = reference_cells
            self.mesh_cells.append(cells)
        # Check if any global features are present
        # global_features is a list of dictionaries, each containing the global features for a sample
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

        global_stats_path = os.path.join(self._stats_dir, GLOBAL_STATS_FILE)
        if self.global_features is None:
            self.global_stats = {
                "global_mean": torch.zeros(0, dtype=torch.float32),
                "global_std": torch.ones(0, dtype=torch.float32),
            }
        elif self.split == "train" and self.stats_mode == "compute":
            self.global_stats = self._compute_global_stats()
            save_json(self.global_stats, global_stats_path)
        elif os.path.exists(global_stats_path):
            self.global_stats = load_json(global_stats_path)
        else:
            raise FileNotFoundError(
                f"Global stats file {global_stats_path} not found. "
                "Build the training split before validation or inference."
            )
        self.global_stats = {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in self.global_stats.items()
        }

        # Storage for per-sample tensors
        self.mesh_pos_seq: list[torch.Tensor] = []  # [T,N,3]
        self.contact_reference_positions: list[torch.Tensor] = []
        self.node_features_data: list[torch.Tensor] = []  # [N,F]
        self.shell_thickness_data: list[torch.Tensor] = []  # physical [N]
        self._feature_slices: dict[
            str, tuple[int, int]
        ] = {}  # per-sample feature slices
        self.target_series_data: list[dict[str, torch.Tensor]] = []

        for rec in point_data:
            # Coordinates
            if "coords" not in rec:
                raise KeyError(f"Missing coordinates key 'coords' in reader record")
            coords_np = rec["coords"][:num_steps]
            assert coords_np.ndim == 3 and coords_np.shape[-1] == 3, (
                f"coords must be [T,N,3], got {coords_np.shape}"
            )
            self.mesh_pos_seq.append(torch.as_tensor(coords_np, dtype=torch.float32))
            if self.contact_surface_exclusion == "reference_geodesic":
                # Freeze physical INITIAL geometry before normalization. Neither
                # the sampled window start nor future ground truth defines d0.
                self.contact_reference_positions.append(
                    self.mesh_pos_seq[-1][0].clone()
                )

            try:
                raw_thickness = self._get_static_feature(rec, "thickness")
            except KeyError:
                raw_thickness = None
            if raw_thickness is None:
                if self.contact_surface:
                    raise ValueError(
                        "surface contact requires supplied physical thickness"
                    )
                thickness_np = np.zeros(coords_np.shape[1], dtype=np.float32)
            else:
                thickness_np = np.asarray(raw_thickness, dtype=np.float32)
                if thickness_np.ndim == 2 and thickness_np.shape[-1] == 1:
                    thickness_np = thickness_np[:, 0]
                if thickness_np.shape != (coords_np.shape[1],):
                    raise ValueError(
                        "thickness must have shape [N] or [N, 1], got "
                        f"{thickness_np.shape}"
                    )
            self.shell_thickness_data.append(torch.from_numpy(thickness_np.copy()))

            # Features: concatenate requested keys if present; allow empty
            parts = []
            # Static features: use as-is (N,[C])
            for k in self.static_features:
                arr = self._get_static_feature(rec, k)
                if arr.ndim == 1:
                    arr = arr[:, None]
                parts.append(arr)
            # Dynamic features: collect series up to num_steps, flatten to [N, T*C]
            T = coords_np.shape[0]
            for k in self.dynamic_features:
                dyn = self._get_dynamic_feature(rec, k, T)
                # dyn: [T,N] or [T,N,C] -> [N, T*C]
                if dyn.ndim == 2:
                    dyn_flat = dyn.transpose(1, 0)  # [N,T]
                else:
                    dyn_flat = dyn.transpose(1, 0, 2).reshape(
                        dyn.shape[1], -1
                    )  # [N,T*C]
                parts.append(dyn_flat)

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
                    # After flattening to [N, width]
                    self._feature_slices[k] = (start, start + width)
                    start += width

            self.node_features_data.append(
                torch.as_tensor(feats_np, dtype=torch.float32)
            )

            # Collect dynamic target series (kept as [T,N] or [T,N,C])
            target_series_rec: dict[str, torch.Tensor] = {}
            for k in self.dynamic_targets:
                dyn = self._get_dynamic_feature(rec, k, T)  # [T,N] or [T,N,C]
                target_series_rec[k] = torch.as_tensor(dyn, dtype=torch.float32)
            self.target_series_data.append(target_series_rec)

        # Stats (node + generic features)
        node_stats_path = os.path.join(self._stats_dir, NODE_STATS_FILE)
        feat_stats_path = os.path.join(self._stats_dir, FEATURE_STATS_FILE)

        if self.split == "train" and self.stats_mode == "compute":
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
        return self._max_idx

    # Common x/y construction
    def set_epoch(self, epoch: int):
        """Set before constructing each epoch's (non-persistent) worker iterator."""
        self.epoch = int(epoch)

    def build_xy(
        self, batch_idx: int, time_idx: int | None, sample_idx: int | None = None
    ):
        """
        x: dict with:
            - 'coords': [N, 3] at the current state
            - optional 'previous_coords': [N, 3] for two-frame initialization
            - 'features': [N, F] concatenated (static + flattened dynamic)
        y: [N, T, Fo] where T=rollout steps, Fo=3+sum(C_k) per timestep
        """
        assert 0 <= batch_idx < self.num_samples, f"batch_idx {batch_idx} out of range"
        if time_idx is not None:
            assert 0 <= time_idx < self.num_steps - 1, (
                f"time_idx {time_idx} out of range [0, {self.num_steps - 1})"
            )
        pos_seq = self.mesh_pos_seq[batch_idx]  # [T,N,3]
        feats = self.node_features_data[batch_idx]  # [N,F]
        T, N, _ = pos_seq.shape
        F = feats.shape[1]

        current_idx = self.initial_history_steps - 1
        target_start = self.initial_history_steps
        target_end = T
        if self.sample_type == "random_time_window":
            window_steps = self.rollout_window_steps
            if window_steps is None:
                raise RuntimeError(
                    "random_time_window was not initialized with a window length"
                )
            max_current_idx = T - window_steps - 1
            current_idx = int(
                torch.randint(
                    self.initial_history_steps - 1,
                    max_current_idx + 1,
                    (1,),
                    generator=self._window_generator(
                        batch_idx if sample_idx is None else sample_idx
                    ),
                ).item()
            )
            target_start = current_idx + 1
            target_end = target_start + window_steps

        x = {"coords": pos_seq[current_idx], "features": feats}
        if self.initial_history_steps == 2:
            x["previous_coords"] = pos_seq[current_idx - 1]

        pos_rollout = pos_seq[target_start:target_end]

        if len(self.dynamic_targets) > 0:
            # Collect dynamic targets [T-1, N, C_k] for each target
            ts_rec = self.target_series_data[batch_idx]
            dyn_list = []
            for k in self.dynamic_targets:
                if k not in ts_rec:
                    raise KeyError(
                        f"Missing dynamic target '{k}' for sample {batch_idx}"
                    )
                series = ts_rec[k]  # Tensor [T,N] or [T,N,C]
                if series.ndim == 2:
                    series = series.unsqueeze(-1)  # [T,N,1]
                series_rollout = series[target_start:target_end]
                dyn_list.append(series_rollout)

            # Concatenate along feature dim per timestep: [T-1, N, 3+sum(C_k)]
            y_per_t = torch.cat([pos_rollout] + dyn_list, dim=-1)  # [T-1, N, Fo]
        else:
            y_per_t = pos_rollout  # [T-1, N, 3]

        # [N, T, Fo] where T = rollout steps
        y = y_per_t.transpose(0, 1)  # [N, T-1, Fo]

        if time_idx is not None:
            x["time"] = torch.tensor(
                time_idx / (self.num_steps - self.initial_history_steps)
            )
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

    def _window_generator(self, sample_idx):
        """Stateless draw keyed by seed, epoch and sample/window slot, not rank/RNG."""
        import hashlib

        if getattr(self, "window_seed", None) is None:
            return None  # Explicit legacy sampling, for old recipes/checkpoints.
        key = f"crash-window-v2:{self.window_seed}:{self.epoch}:{sample_idx}".encode()
        seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
        return torch.Generator().manual_seed(seed)

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

    def _compute_global_stats(self):
        if self.global_features is None or not self.global_features_keys:
            return {
                "global_mean": torch.zeros(0, dtype=torch.float32),
                "global_std": torch.ones(0, dtype=torch.float32),
            }
        values = torch.tensor(
            [
                [float(features[key]) for key in self.global_features_keys]
                for features in self.global_features
            ],
            dtype=torch.float32,
        )
        global_mean = values.mean(dim=0)
        global_var = torch.clamp(
            (values * values).mean(dim=0) - global_mean**2, min=0.0
        )
        return {
            "global_mean": global_mean,
            "global_std": torch.sqrt(global_var + EPS),
        }

    def _normalized_global_features(self, batch_idx: int):
        if self.global_features is None:
            return None
        mean = self.global_stats["global_mean"]
        std = self.global_stats["global_std"]
        return {
            key: (
                torch.tensor(self.global_features[batch_idx][key], dtype=torch.float32)
                - mean[index]
            )
            / (std[index] + EPS)
            for index, key in enumerate(self.global_features_keys)
        }

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
    def _get_static_feature(rec: dict, key: str) -> np.ndarray:
        """
        Fetch a per-point static feature from the reader record.
        Supports:
          - direct top-level fields: rec[key]
          - rec['point_data'][key]
        """
        # Direct field
        if key in rec:
            return np.asarray(rec[key])

        # From point_data dict
        pd = rec.get("point_data", {})
        if key in pd:
            return np.asarray(pd[key])

        raise KeyError(
            f"Missing static feature key '{key}' in reader record (checked top-level and point_data)"
        )

    @staticmethod
    def _get_dynamic_feature(rec: dict, key: str, T: int) -> np.ndarray:
        """
        Fetch a per-point dynamic feature time series from the reader record.
        Expects point_data keys like '<key>_t...' per timestep.
        Returns array of shape [T, N] or [T, N, C]; pads by repeating last frame if fewer than T.
        """
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
        # Normalize shapes to [N,C]
        series = [x[:, None] if x.ndim == 1 else x for x in series]
        N = series[0].shape[0]
        C = series[0].shape[1]
        for x in series:
            assert x.shape[0] == N and x.shape[1] == C, (
                f"Inconsistent shapes in dynamic feature '{key}': "
                f"expected (N={N},C={C}), got {x.shape}"
            )
        # Stack to [T', N, C]
        arr = np.stack(series, axis=0)
        Tprime = arr.shape[0]
        if Tprime >= T:
            arr = arr[:T]
        else:
            # pad by repeating last frame
            pad = np.repeat(arr[-1:], repeats=(T - Tprime), axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        return arr


class CrashGraphDataset(CrashBaseDataset):
    """
    Graph version:
      - Builds PyG graphs (create_graph + add_self_loop)
      - Computes/loads edge stats and normalizes edge features
      - Returns SimSample with graph, global_features, and target_series (parity with point-cloud).
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

        Data = _pyg_data.Data
        self.graphs = []
        reference_src = None
        reference_dst = None
        reference_num_nodes = None
        reference_edge_index = None
        reference_component_id = None
        reference_contact_exclusions = None
        self.geodesic_cache = None
        if self.contact_surface_exclusion == "reference_geodesic":
            from surface_geodesic_cache import SurfaceGeodesicCache

            self.geodesic_cache = SurfaceGeodesicCache(
                self.contact_geodesic_cache_dir, logger=self.logger
            )
        for i in range(self.num_samples):
            num_nodes = int(self.mesh_pos_seq[i][0].shape[0])
            shares_reference_topology = (
                reference_src is not None
                and num_nodes == reference_num_nodes
                and np.array_equal(self.srcs[i], reference_src)
                and np.array_equal(self.dsts[i], reference_dst)
            )
            if shares_reference_topology:
                # Crash ensembles normally deform the same mesh. Reuse its
                # immutable connectivity and component labels instead of
                # repeating coalescing and a Python union-find for every run.
                g = Data(edge_index=reference_edge_index, num_nodes=num_nodes)
                g.component_id = reference_component_id
            else:
                g = self.create_graph(
                    self.srcs[i],
                    self.dsts[i],
                    num_nodes=num_nodes,
                    dtype=torch.long,
                )
                g.component_id = self.connected_component_ids(
                    g.edge_index, int(g.num_nodes)
                )
                if reference_src is None:
                    reference_src = self.srcs[i]
                    reference_dst = self.dsts[i]
                    reference_num_nodes = num_nodes
                    reference_edge_index = g.edge_index
                    reference_component_id = g.component_id
            pos0 = self.mesh_pos_seq[i][0]
            g = self.add_edge_features(g, pos0)
            g.shell_thickness = self.shell_thickness_data[i]
            if self.contact_surface:
                from surface_topology import (
                    SurfaceContactData,
                    surface_faces_from_cells,
                    surface_one_ring_exclusions,
                )

                g = SurfaceContactData(**g.to_dict())
                if (
                    shares_reference_topology
                    and self.mesh_cells[i] is self.mesh_cells[0]
                ):
                    g.contact_faces = self.graphs[0].contact_faces
                    if self.contact_surface_exclusion == "one_ring":
                        g.contact_surface_exclusions = self.graphs[
                            0
                        ].contact_surface_exclusions
                else:
                    g.contact_faces = surface_faces_from_cells(
                        self.mesh_cells[i], num_nodes
                    )
                    if self.contact_surface_exclusion == "one_ring":
                        g.contact_surface_exclusions = surface_one_ring_exclusions(
                            g.contact_faces, num_nodes
                        )
            if self.contact_exclusion_hops is not None:
                from material_contact import material_contact_exclusions

                if (
                    shares_reference_topology
                    and self.mesh_cells[i] is self.mesh_cells[0]
                ):
                    g.contact_exclusion_edges = reference_contact_exclusions
                else:
                    g.contact_exclusion_edges = material_contact_exclusions(
                        g.edge_index,
                        num_nodes,
                        self.contact_exclusion_hops,
                        self.mesh_cells[i],
                    )
                    if i == 0:
                        reference_contact_exclusions = g.contact_exclusion_edges
            if self.contact_surface_exclusion == "reference_geodesic":
                g.contact_surface_exclusions = self.geodesic_cache.get(
                    self.contact_reference_positions[i],
                    g.contact_faces,
                    g.shell_thickness,
                    self.contact_geodesic_options,
                )
            self.graphs.append(g)

        # Edge stats
        edge_stats_path = os.path.join(self._stats_dir, EDGE_STATS_FILE)
        if self.split == "train" and self.stats_mode == "compute":
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
        assert 0 <= idx < self._max_idx, f"Index {idx} out of range"
        batch_idx, time_idx = self._resolve_idx(idx)
        g = self.graphs[batch_idx]
        x, y = self.build_xy(batch_idx, time_idx, sample_idx=idx)
        gf = self._normalized_global_features(batch_idx)
        # Truncated training samples do not need the full target series on device.
        ts = (
            None
            if time_idx is not None or self.sample_type == "random_time_window"
            else self.target_series_data[batch_idx]
        )
        return SimSample(
            node_features=x,
            node_target=y,
            graph=g,
            global_features=gf,
            target_series=ts,
        )

    # ----- graph-specific helpers (use _pyg_data / _pyg_utils so PyG loads only when used) -----
    @staticmethod
    def create_graph(src, dst, num_nodes: int, dtype=torch.long):
        """Build coalesced bidirectional structural edges and node self-loops."""
        src = torch.as_tensor(src, dtype=dtype)
        dst = torch.as_tensor(dst, dtype=dtype)
        edge_index = torch.stack(
            [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
        )  # [2, E]
        edge_index, _ = _pyg_utils.coalesce(edge_index, None, num_nodes=num_nodes)
        edge_index, _ = _pyg_utils.add_self_loops(edge_index, num_nodes=num_nodes)
        return _pyg_data.Data(edge_index=edge_index, num_nodes=num_nodes)

    @staticmethod
    def connected_component_ids(
        edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Return contiguous structural component IDs using union-find."""

        parents = np.arange(num_nodes, dtype=np.int64)
        ranks = np.zeros(num_nodes, dtype=np.int8)

        def find(node: int) -> int:
            while parents[node] != node:
                parents[node] = parents[parents[node]]
                node = int(parents[node])
            return node

        for source, destination in edge_index.detach().cpu().numpy().T:
            root_source = find(int(source))
            root_destination = find(int(destination))
            if root_source == root_destination:
                continue
            if ranks[root_source] < ranks[root_destination]:
                root_source, root_destination = root_destination, root_source
            parents[root_destination] = root_source
            if ranks[root_source] == ranks[root_destination]:
                ranks[root_source] += 1

        root_to_component: dict[int, int] = {}
        labels = np.empty(num_nodes, dtype=np.int64)
        for node in range(num_nodes):
            root = find(node)
            component = root_to_component.setdefault(root, len(root_to_component))
            labels[node] = component
        return torch.from_numpy(labels)

    @staticmethod
    def add_edge_features(data, pos: torch.Tensor):
        # data: PyG Data; pos: [N,3]
        """Attach reference displacement and distance to each structural edge."""
        row, col = data.edge_index
        pos_t = torch.as_tensor(pos, dtype=torch.float32)
        disp = pos_t[row] - pos_t[col]  # [E,3]
        disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)  # [E,1]
        data.edge_attr = torch.cat((disp, disp_norm), dim=1)  # [E,4]
        return data

    def _compute_edge_stats(self):
        if self.num_samples <= 0:
            raise ValueError("Cannot compute edge statistics without samples")
        edge_dim = self.graphs[0].edge_attr.shape[-1]
        edge_mean = torch.zeros(edge_dim, dtype=torch.float32)
        edge_meansqr = torch.zeros(edge_dim, dtype=torch.float32)
        for i in range(self.num_samples):
            x_e = self.graphs[i].edge_attr.to(torch.float32)  # [E,De]
            m = torch.mean(x_e, dim=0)
            msq = torch.mean(x_e * x_e, dim=0)
            edge_mean += m / self.num_samples
            edge_meansqr += msq / self.num_samples

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
        assert 0 <= idx < self._max_idx, f"Index {idx} out of range"
        batch_idx, time_idx = self._resolve_idx(idx)
        x, y = self.build_xy(batch_idx, time_idx, sample_idx=idx)
        gf = self._normalized_global_features(batch_idx)
        # Truncated training samples do not need the full target series on device.
        ts = (
            None
            if time_idx is not None or self.sample_type == "random_time_window"
            else self.target_series_data[batch_idx]
        )
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
