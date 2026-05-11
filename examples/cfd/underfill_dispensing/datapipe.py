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

"""
Data Pipeline for Transient Epoxy VOF Prediction (Static Mesh).

Loads VTP files containing:
    - Static coordinates: [N, 3]
    - Time-varying epoxy_vof: [T, N, 1]

Output format (per sample):
    - node_features["coords"]:   [N, 3] normalized coordinates at t=0
    - node_features["features"]: [N, 1] epoxy_vof at t=0 (raw, unnormalized)
    - node_target:               [N, T-1] future epoxy_vof (t=1 to T-1, raw)

Where T = num_steps (total timesteps including initial).

Note on VOF normalization:
    Feature (epoxy_vof) normalization is intentionally set to identity
    (mean=0, std=1) so that the model operates on raw VOF values in [0, 1].
    Empirically, the dataset has mean ≈ 0.510 and std ≈ 0.485, which is
    close to a 50/50 bimodal distribution of 0s and 1s. Since the data is
    already in a well-scaled range and the affine transform would be nearly
    symmetric (mapping [0,1] → [−1.05, +1.01]), the benefit of z-score
    normalization is marginal. Keeping raw values simplifies post-processing
    (clamping, thresholding) and maintains physical interpretability.

    To re-enable VOF normalization, set USE_VOF_NORMALIZATION = True below
    and ensure that inference.py denormalization remains consistent.

Note on stats directory:
    Normalization statistics are saved to and loaded from a ``stats/``
    subdirectory inside the Hydra output directory (``hydra.run.dir``).
    With the default config (``hydra.run.dir: ./outputs/``,
    ``hydra.job.chdir: True``), this resolves to ``./outputs/stats/``,
    keeping stats co-located with checkpoints, logs, and predictions.
    The path can be overridden via the ``stats_dir`` constructor argument.
"""

import os
import json
import numpy as np
import torch
from typing import Callable, Optional

from physicsnemo.utils.logging import PythonLogger

from vtp_reader import Reader

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

STATS_DIRNAME = "stats"
NODE_STATS_FILE = "node_stats.json"
FEATURE_STATS_FILE = "feature_stats.json"
EPS = 1e-8

# Set to True to enable z-score normalization of VOF values.
# When False, feature_stats are identity (mean=0, std=1) making
# normalization a no-op. See module docstring for rationale.
USE_VOF_NORMALIZATION = False


# ═══════════════════════════════════════════════════════════════════════════════
# Stats Serialization
# ═══════════════════════════════════════════════════════════════════════════════


def save_stats_json(stats: dict, filepath: str) -> None:
    """
    Save a stats dict (with torch tensor values) to JSON.

    Tensors are converted to Python lists so the file has no
    numpy/torch dependency and can be inspected by hand.
    """
    serializable = {
        k: v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v
        for k, v in stats.items()
    }
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)


def load_stats_json(filepath: str, dtype: torch.dtype = torch.float32) -> dict:
    """Load a stats dict from JSON and convert lists back to tensors."""
    with open(filepath, "r") as f:
        raw = json.load(f)
    return {k: torch.as_tensor(v, dtype=dtype) for k, v in raw.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Stats Directory Resolution
# ═══════════════════════════════════════════════════════════════════════════════


def _resolve_stats_dir(stats_dir: Optional[str] = None) -> str:
    """
    Resolve the stats directory to a stable absolute path.

    Resolution order:
      1. Explicit ``stats_dir`` if provided (from config override).
      2. ``<hydra.run.dir>/stats/`` via Hydra's runtime config.
         With default config (``hydra.run.dir: ./outputs/``), this
         resolves to ``<project>/outputs/stats/``.
      3. ``./stats/`` as fallback when not running under Hydra
         (e.g., unit tests, notebooks).

    Using Hydra's output directory rather than relying on ``os.getcwd()``
    ensures that training, validation, and inference always find the same
    stats files, even if ``hydra.job.chdir`` is changed to False or
    ``hydra.run.dir`` is modified.

    Args:
        stats_dir: Explicit override path, or None for auto-resolution.

    Returns:
        Absolute path to the stats directory.
    """
    if stats_dir is not None:
        return os.path.abspath(stats_dir)

    # Derive from Hydra's output directory when available
    try:
        from hydra.core.hydra_config import HydraConfig

        hydra_output = HydraConfig.get().runtime.output_dir
        return os.path.abspath(os.path.join(hydra_output, STATS_DIRNAME))
    except Exception:
        # Not running under Hydra (unit tests, notebooks, standalone scripts)
        return os.path.abspath(STATS_DIRNAME)


# ═══════════════════════════════════════════════════════════════════════════════
# SimSample Class
# ═══════════════════════════════════════════════════════════════════════════════


class SimSample:
    """
    Point cloud sample for Transolver (no graph structure).

    Attributes:
        node_features: Dictionary containing:
            - "coords":   [N, 3] static mesh coordinates (normalized)
            - "features": [N, 1] epoxy_vof at t=0 (raw unless USE_VOF_NORMALIZATION)
        node_target: [N, T-1] future epoxy_vof values

    Used by:
        - rollout.py: Accesses node_features["coords"] and node_features["features"]
        - train.py: Accesses node_target for loss computation
    """

    def __init__(
        self,
        node_features: dict[str, torch.Tensor],
        node_target: torch.Tensor,
    ):
        self.node_features = node_features
        self.node_target = node_target

    def to(self, device: torch.device) -> "SimSample":
        """Move all tensors to specified device."""
        self.node_features = {k: v.to(device) for k, v in self.node_features.items()}
        self.node_target = self.node_target.to(device)
        return self

    def is_graph(self) -> bool:
        """Return False - this is a point cloud, not a graph."""
        return False

    def get_info(self) -> dict:
        """Return sample information for debugging."""
        return {
            "num_nodes": self.node_features["coords"].shape[0],
            "coords_shape": tuple(self.node_features["coords"].shape),
            "features_shape": tuple(
                self.node_features.get("features", torch.tensor([])).shape
            ),
            "target_shape": tuple(self.node_target.shape),
        }

    def __repr__(self) -> str:
        N = self.node_features["coords"].shape[0]
        F = self.node_features.get("features", torch.tensor([[]])).shape[-1]
        T_target = self.node_target.shape[-1] if self.node_target.ndim > 1 else 0
        return f"SimSample(N={N}, features={F}, T_target={T_target})"


# ═══════════════════════════════════════════════════════════════════════════════
# UnderfillDataset Class
# ═══════════════════════════════════════════════════════════════════════════════


class UnderfillDataset:
    """
    Dataset for Transolver training on transient epoxy VOF prediction.

    Handles:
        - Loading VTP files with static mesh and time-varying epoxy_vof
        - Computing normalization statistics (saved for validation/inference)
        - Preparing input/target pairs for autoregressive training

    Expected VTP data format:
        - coords: [N, 3]
        - epoxy_vof: [T, N, 1] with T timesteps

    Statistics exposed (used by train.py and inference.py):
        - node_stats: {"pos_mean": [3], "pos_std": [3]}
        - feature_stats: {"feature_mean": [1], "feature_std": [1]}
    """

    NUM_FEATURES = 1  # Scalar field (epoxy_vof)

    def __init__(
        self,
        name: str = "dataset",
        reader: Optional[Callable] = None,
        data_dir: Optional[str] = None,
        split: str = "train",
        num_samples: int = 1000,
        num_steps: int = 20,
        logger=None,
        dt: float = 5e-3,
        debug: bool = False,
        stats_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize dataset.

        Args:
            name: Dataset name for logging
            reader: VTP reader callable (default: vtp_reader.Reader)
            data_dir: Directory containing VTP files
            split: "train", "validation", or "test"
            num_samples: Maximum number of samples to load
            num_steps: Total time steps (T), including initial state
            logger: Logger instance
            dt: Time step size (for reference, not used in computation)
            debug: Enable verbose output
            stats_dir: Explicit path for normalization statistics directory.
                       If None (default), resolves to ``<hydra.run.dir>/stats/``
                       (i.e., ``./outputs/stats/`` with the default config).
        """
        self.name = name
        self.data_dir = data_dir or "."
        self.split = split
        self.num_samples = num_samples
        self.num_steps = num_steps  # Max 20 steps (step00 to step19)
        self.logger = logger or PythonLogger()
        self.dt = dt
        self.debug = debug

        self._log(f"\n{'=' * 70}")
        self._log(f"Initializing {self.__class__.__name__}")
        self._log(f"{'=' * 70}")
        self._log(f"  Name:        {name}")
        self._log(f"  Split:       {split}")
        self._log(f"  Data dir:    {self.data_dir}")
        self._log(f"  Num samples: {num_samples}")
        self._log(f"  Num steps:   {self.num_steps} (T)")
        self._log(f"  Rollout:     {self.num_steps - 1} steps (T-1)")
        self._log(f"  Feature:     epoxy_vof (scalar)")
        self._log(
            f"  VOF norm:    {'z-score' if USE_VOF_NORMALIZATION else 'identity (no-op)'}"
        )

        # Resolve stats directory to a stable absolute path.
        # Default: <hydra.run.dir>/stats/ → ./outputs/stats/
        self._stats_dir = _resolve_stats_dir(stats_dir)
        os.makedirs(self._stats_dir, exist_ok=True)
        self._log(f"  Stats dir:   {self._stats_dir}")

        # Initialize reader
        if reader is None:
            reader = Reader(debug=debug)

        # Load raw data from VTP files
        point_data = reader(
            data_dir=self.data_dir,
            num_samples=num_samples,
            split=split,
            logger=self.logger,
        )

        if not point_data:
            raise ValueError(f"No data loaded from {self.data_dir}")

        # Storage for processed data
        self.mesh_pos_seq: list[torch.Tensor] = []  # List of [T, N, 3]
        self.epoxy_vof_seq: list[torch.Tensor] = []  # List of [T, N, 1]

        self._log(f"\n  Processing {len(point_data)} records...")

        for i, rec in enumerate(point_data):
            self._process_record(i, rec)

        self._log(f"  Loaded {len(self.mesh_pos_seq)} samples successfully")

        # Compute or load statistics
        self._setup_statistics()

        # Apply normalization to all data
        self._apply_normalization()

        self.length = len(self.mesh_pos_seq)

        # Print summary and verify shapes
        self._print_summary()

    def _log(self, msg: str):
        """Log message to logger and optionally print for debug."""
        if self.debug:
            print(msg)
        if self.logger:
            self.logger.info(msg)

    def _process_record(self, idx: int, rec: dict):
        """
        Process a single VTP record.

        The reader contract returns numpy arrays, so we convert directly
        to torch tensors without defensive wrappers.

        Args:
            idx: Record index
            rec: Dictionary with "coords" [N, 3] and "epoxy_vof" [T, N, 1]
        """
        coords = np.asarray(rec["coords"])  # [N, 3]
        N_nodes = coords.shape[0]

        if "epoxy_vof" not in rec:
            raise ValueError(f"Record {idx} missing 'epoxy_vof' field")

        epoxy_vof = np.asarray(rec["epoxy_vof"])  # [T_file, N, 1]

        T_file = epoxy_vof.shape[0]
        T = min(T_file, self.num_steps)

        if self.debug:
            print(f"\n    Record {idx}: N={N_nodes}, T={T} (available: {T_file})")

        # Static coords replicated for all timesteps: [T, N, 3]
        coords_seq = np.tile(coords[np.newaxis, :, :], (T, 1, 1))
        self.mesh_pos_seq.append(torch.from_numpy(coords_seq.copy()))

        # Slice epoxy_vof to desired time steps: [T, N, 1]
        epoxy_vof_sliced = epoxy_vof[:T].copy()
        self.epoxy_vof_seq.append(torch.from_numpy(epoxy_vof_sliced))

        if self.debug:
            print(f"      coords: {coords_seq.shape}")
            print(
                f"      epoxy_vof: {epoxy_vof_sliced.shape}, "
                f"range [{epoxy_vof_sliced.min():.4f}, {epoxy_vof_sliced.max():.4f}]"
            )

    def _setup_statistics(self):
        """Compute or load normalization statistics.

        Position (coordinate) statistics are always computed via z-score
        normalization since coordinate ranges vary across geometries.

        VOF feature statistics are controlled by USE_VOF_NORMALIZATION:
          - False (default): identity transform (mean=0, std=1). VOF values
            remain in their natural [0, 1] range. This is appropriate because
            the data is bimodal (mean ≈ 0.51, std ≈ 0.49) and already
            well-scaled. Raw values simplify clamping and thresholding.
          - True: z-score normalization computed from training data. Maps
            [0, 1] → [≈ −1.05, ≈ +1.01]. May improve convergence if the
            model architecture benefits from zero-centered inputs.

        Statistics are stored in self._stats_dir (absolute path resolved
        from hydra.run.dir), ensuring training, validation, and inference
        always reference the same files.
        """
        node_stats_path = os.path.join(self._stats_dir, NODE_STATS_FILE)
        feat_stats_path = os.path.join(self._stats_dir, FEATURE_STATS_FILE)

        if self.split == "train":
            self._log("\n  Computing statistics from training data...")
            self.node_stats = self._compute_node_stats()

            if USE_VOF_NORMALIZATION:
                self.feature_stats = self._compute_feature_stats()
                self._log("  VOF normalization: z-score (computed from data)")
            else:
                self.feature_stats = self._identity_feature_stats()
                self._log("  VOF normalization: identity (no-op)")

            # Save for validation/inference
            save_stats_json(self.node_stats, node_stats_path)
            save_stats_json(self.feature_stats, feat_stats_path)
            self._log(f"  Saved statistics to {self._stats_dir}/")

        else:
            # Load from saved training stats
            if os.path.exists(node_stats_path) and os.path.exists(feat_stats_path):
                self._log(f"\n  Loading statistics from {self._stats_dir}/")
                self.node_stats = load_stats_json(node_stats_path)

                if USE_VOF_NORMALIZATION:
                    self.feature_stats = load_stats_json(feat_stats_path)
                    self._log("  VOF normalization: z-score (loaded from file)")
                else:
                    self.feature_stats = self._identity_feature_stats()
                    self._log("  VOF normalization: identity (no-op)")
            else:
                self._log(
                    f"\n  WARNING: No saved statistics found at {self._stats_dir}/"
                )
                self._log("           Expected files:")
                self._log(f"             {node_stats_path}")
                self._log(f"             {feat_stats_path}")
                self._log("           Run training first to generate statistics!")
                self._log("           Computing from current split as fallback.")
                self.node_stats = self._compute_node_stats()

                if USE_VOF_NORMALIZATION:
                    self.feature_stats = self._compute_feature_stats()
                else:
                    self.feature_stats = self._identity_feature_stats()

        self._log_statistics()

    @staticmethod
    def _identity_feature_stats() -> dict:
        """Return identity (no-op) feature statistics: mean=0, std=1."""
        return {
            "feature_mean": torch.zeros(1, dtype=torch.float32),
            "feature_std": torch.ones(1, dtype=torch.float32),
        }

    def _log_statistics(self):
        """Log the computed/loaded statistics."""
        pos_mean = self.node_stats["pos_mean"]
        pos_std = self.node_stats["pos_std"]
        feat_mean = self.feature_stats["feature_mean"]
        feat_std = self.feature_stats["feature_std"]

        self._log(f"\n  Statistics (from {self._stats_dir}):")
        self._log(
            f"    pos_mean:     [{pos_mean[0].item():.6f}, {pos_mean[1].item():.6f}, {pos_mean[2].item():.6f}]"
        )
        self._log(
            f"    pos_std:      [{pos_std[0].item():.6f}, {pos_std[1].item():.6f}, {pos_std[2].item():.6f}]"
        )
        self._log(f"    feature_mean: {feat_mean.item():.6f}")
        self._log(f"    feature_std:  {feat_std.item():.6f}")

        if not USE_VOF_NORMALIZATION:
            self._log(
                f"    (VOF normalization is identity — model sees raw [0, 1] values)"
            )

    def _compute_node_stats(self) -> dict:
        """Compute position statistics over all samples and time steps."""
        all_pos = torch.cat([p.reshape(-1, 3) for p in self.mesh_pos_seq], dim=0)

        mean = torch.mean(all_pos, dim=0)
        std = torch.std(all_pos, dim=0)
        std = torch.clamp(std, min=EPS)

        return {"pos_mean": mean, "pos_std": std}

    def _compute_feature_stats(self) -> dict:
        """Compute epoxy_vof statistics over all samples and time steps.

        Used when USE_VOF_NORMALIZATION is True. Maps VOF from [0, 1] to
        approximately [-1, +1] via z-score normalization.

        When USE_VOF_NORMALIZATION is False (default), this method is not
        called and identity stats (mean=0, std=1) are used instead.
        """
        all_vof = torch.cat([f.reshape(-1, 1) for f in self.epoxy_vof_seq], dim=0)

        mean = torch.mean(all_vof, dim=0)
        std = torch.std(all_vof, dim=0)
        std = torch.clamp(std, min=EPS)

        return {"feature_mean": mean, "feature_std": std}

    def _apply_normalization(self):
        """Apply normalization to all loaded data.

        Coordinates are always z-score normalized.
        VOF values are normalized only if USE_VOF_NORMALIZATION is True;
        otherwise the identity transform is applied (no change).
        """
        self._log("\n  Applying normalization...")

        pos_mean = self.node_stats["pos_mean"]
        pos_std = self.node_stats["pos_std"]
        feat_mean = self.feature_stats["feature_mean"]
        feat_std = self.feature_stats["feature_std"]

        for i in range(len(self.mesh_pos_seq)):
            # Normalize positions: [T, N, 3]
            self.mesh_pos_seq[i] = (
                self.mesh_pos_seq[i] - pos_mean.view(1, 1, -1)
            ) / pos_std.view(1, 1, -1)

            # Normalize epoxy_vof: [T, N, 1]
            # When USE_VOF_NORMALIZATION is False, this is (x - 0) / 1 = x
            self.epoxy_vof_seq[i] = (
                self.epoxy_vof_seq[i] - feat_mean.view(1, 1, -1)
            ) / feat_std.view(1, 1, -1)

    def _print_summary(self):
        """Print dataset summary and verify tensor shapes."""
        self._log(f"\n{'=' * 70}")
        self._log(f"Dataset Summary: {self.name} ({self.split})")
        self._log(f"{'=' * 70}")
        self._log(f"  Total samples:     {self.length}")
        self._log(f"  Time steps (T):    {self.num_steps}")
        self._log(f"  Target steps:      {self.num_steps - 1} (T-1)")
        self._log(f"  Feature:           epoxy_vof")
        self._log(f"  Feature dimension: {self.NUM_FEATURES}")
        self._log(
            f"  VOF normalization: {'z-score' if USE_VOF_NORMALIZATION else 'identity (no-op)'}"
        )
        self._log(f"  Stats directory:   {self._stats_dir}")

        if self.length > 0:
            sample = self[0]
            N = sample.node_features["coords"].shape[0]

            self._log(f"\n  Sample 0 shapes:")
            self._log(
                f"    coords:   {sample.node_features['coords'].shape}  (expected: [N, 3])"
            )
            self._log(
                f"    features: {sample.node_features['features'].shape}  (expected: [N, 1])"
            )
            self._log(f"    target:   {sample.node_target.shape}  (expected: [N, T-1])")

            # Verify dimensions match expectations
            T_target = self.num_steps - 1
            expected_target = (N, T_target)
            actual_target = tuple(sample.node_target.shape)

            if actual_target == expected_target:
                self._log(f"\n  ✓ All shapes correct")
            else:
                self._log(f"\n  ✗ Target shape mismatch!")
                self._log(f"    Expected: {expected_target}")
                self._log(f"    Actual:   {actual_target}")

        self._log(f"{'=' * 70}\n")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> SimSample:
        """
        Get a sample for training/inference.

        Args:
            idx: Sample index

        Returns:
            SimSample with:
                - node_features["coords"]: [N, 3] normalized coordinates
                - node_features["features"]: [N, 1] VOF at t=0
                - node_target: [N, T-1] future VOF values

        Shape flow:
            pos_seq:  [T, N, 3] -> coords:   [N, 3] (take t=0)
            vof_seq:  [T, N, 1] -> features: [N, 1] (take t=0)
                                -> target:   [N, T-1] (take t=1 to T-1)
        """
        pos_seq = self.mesh_pos_seq[idx]  # [T, N, 3]
        vof_seq = self.epoxy_vof_seq[idx]  # [T, N, 1]

        # Input: initial state (t=0)
        node_features = {
            "coords": pos_seq[0],  # [N, 3]
            "features": vof_seq[0],  # [N, 1]
        }

        # Target: future states (t=1, t=2, ..., t=T-1)
        T = vof_seq.shape[0]
        if T > 1:
            # vof_seq[1:] -> [T-1, N, 1]
            # squeeze(-1) -> [T-1, N]
            # transpose   -> [N, T-1]
            node_target = vof_seq[1:].squeeze(-1).transpose(0, 1)
        else:
            # No future timesteps available
            N = pos_seq.shape[1]
            node_target = torch.zeros((N, 0), dtype=torch.float32)

        return SimSample(node_features=node_features, node_target=node_target)


# ═══════════════════════════════════════════════════════════════════════════════
# Collate Function
# ═══════════════════════════════════════════════════════════════════════════════


def simsample_collate(batch: list[SimSample]) -> list[SimSample]:
    """
    Custom collate function - returns list of SimSamples.

    Since samples may have different numbers of nodes (N varies),
    we cannot stack them into a single tensor. Instead, we return
    the list and process samples individually in the training loop.
    """
    return batch
