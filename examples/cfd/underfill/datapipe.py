# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Data Pipeline for Transient Epoxy VOF Prediction (Static Mesh).

Loads VTP files containing:
    - Static coordinates: [N, 3]
    - Time-varying epoxy_vof: [T, N, 1]

Output format (per sample):
    - node_features["coords"]:   [N, 3] normalized coordinates at t=0
    - node_features["features"]: [N, 1] normalized epoxy_vof at t=0
    - node_target:               [N, T-1] normalized future epoxy_vof (t=1 to T-1)

Where T = num_steps (total timesteps including initial).
"""

import os
import json
import numpy as np
import torch
from typing import Callable, Optional, Any

from physicsnemo.utils.logging import PythonLogger

from vtp_reader import Reader

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

STATS_DIRNAME = "stats"
NODE_STATS_FILE = "node_stats.json"
FEATURE_STATS_FILE = "feature_stats.json"
EPS = 1e-8


# ═══════════════════════════════════════════════════════════════════════════════
# JSON Utilities 
# ═══════════════════════════════════════════════════════════════════════════════

def save_json(data: dict, filepath: str) -> None:
    """Save dictionary to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> dict:
    """Load dictionary from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# Type Conversion Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _to_python_native(value: Any) -> Any:
    """
    Recursively convert tensor/numpy values to Python native types.
    
    This ensures JSON serialization works without any numpy/torch dependencies.
    
    Args:
        value: Any value (tensor, numpy array, list, dict, scalar, etc.)
        
    Returns:
        Python native type (list, dict, float, int, etc.)
    """
    if isinstance(value, torch.Tensor):
        # Convert tensor to Python list
        return value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        # Convert numpy array to Python list
        return value.tolist()
    elif isinstance(value, (np.floating, np.float32, np.float64)):
        # Convert numpy float to Python float
        return float(value)
    elif isinstance(value, (np.integer, np.int32, np.int64)):
        # Convert numpy int to Python int
        return int(value)
    elif isinstance(value, dict):
        # Recursively convert dict values
        return {k: _to_python_native(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        # Recursively convert list/tuple elements
        return [_to_python_native(v) for v in value]
    elif hasattr(value, 'item'):
        # Handle any other type with .item() method (scalars)
        return value.item()
    else:
        # Already a Python native type
        return value


def _to_tensor(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Safely convert a value to a torch tensor.
    
    Handles: torch.Tensor, numpy.ndarray, list, scalar values.
    
    Args:
        value: Input value to convert
        dtype: Target dtype
        
    Returns:
        torch.Tensor
    """
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    elif isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy()).to(dtype=dtype)
    elif isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=dtype)
    else:
        return torch.tensor(value, dtype=dtype)


def _to_numpy(value: Any) -> np.ndarray:
    """
    Safely convert a value to a numpy array.
    
    Args:
        value: Input value (tensor, array, list, etc.)
        
    Returns:
        numpy.ndarray
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        return value
    else:
        return np.asarray(value)


def _stats_to_serializable(stats: dict) -> dict:
    """
    Convert stats dict to JSON-serializable format (pure Python types).
    
    Args:
        stats: Dictionary with tensor/array values
        
    Returns:
        Dictionary with Python native types (lists, floats)
    """
    return _to_python_native(stats)


def _stats_from_serializable(stats: dict, dtype: torch.dtype = torch.float32) -> dict:
    """
    Convert stats dict from JSON format back to tensors.
    
    Args:
        stats: Dictionary (with list values from JSON)
        dtype: Target tensor dtype
        
    Returns:
        Dictionary with tensor values
    """
    return {k: _to_tensor(v, dtype=dtype) for k, v in stats.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# SimSample Class
# ═══════════════════════════════════════════════════════════════════════════════

class SimSample:
    """
    Point cloud sample for Transolver (no graph structure).
    
    Attributes:
        node_features: Dictionary containing:
            - "coords":   [N, 3] static mesh coordinates (normalized)
            - "features": [N, 1] epoxy_vof at t=0 (normalized)
        node_target: [N, T-1] future epoxy_vof values (normalized)
    
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
        self.node_features = {
            k: v.to(device) for k, v in self.node_features.items()
        }
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
            "features_shape": tuple(self.node_features.get("features", torch.tensor([])).shape),
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
    
    Statistics exposed (used by train.py):
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
        **kwargs
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
        """
        self.name = name
        self.data_dir = data_dir or "."
        self.split = split
        self.num_samples = num_samples
        self.num_steps = min(num_steps, 20)  # Max 20 steps (step00 to step19)
        self.logger = logger or PythonLogger()
        self.dt = dt
        self.debug = debug

        self._log(f"\n{'='*70}")
        self._log(f"Initializing {self.__class__.__name__}")
        self._log(f"{'='*70}")
        self._log(f"  Name:        {name}")
        self._log(f"  Split:       {split}")
        self._log(f"  Data dir:    {self.data_dir}")
        self._log(f"  Num samples: {num_samples}")
        self._log(f"  Num steps:   {self.num_steps} (T)")
        self._log(f"  Rollout:     {self.num_steps - 1} steps (T-1)")
        self._log(f"  Feature:     epoxy_vof (scalar)")

        # Create stats directory
        self._stats_dir = STATS_DIRNAME
        os.makedirs(self._stats_dir, exist_ok=True)

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
        self.mesh_pos_seq: list[torch.Tensor] = []    # List of [T, N, 3]
        self.epoxy_vof_seq: list[torch.Tensor] = []   # List of [T, N, 1]

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
        
        Args:
            idx: Record index
            rec: Dictionary with "coords" [N, 3] and "epoxy_vof" [T, N, 1]
        """
        # Extract coordinates - handle both numpy and tensor
        coords = _to_numpy(rec["coords"])  # [N, 3]
        N_nodes = coords.shape[0]

        # Extract epoxy_vof
        if "epoxy_vof" not in rec:
            raise ValueError(f"Record {idx} missing 'epoxy_vof' field")
        
        epoxy_vof = _to_numpy(rec["epoxy_vof"])  # [T_file, N, 1]
        
        T_file = epoxy_vof.shape[0]
        T = min(T_file, self.num_steps)

        if self.debug:
            print(f"\n    Record {idx}: N={N_nodes}, T={T} (available: {T_file})")

        # Static coords replicated for all timesteps: [T, N, 3]
        coords_seq = np.tile(coords[np.newaxis, :, :], (T, 1, 1))
        self.mesh_pos_seq.append(torch.from_numpy(coords_seq.copy()).float())

        # Slice epoxy_vof to desired time steps: [T, N, 1]
        epoxy_vof_sliced = epoxy_vof[:T].copy()
        self.epoxy_vof_seq.append(torch.from_numpy(epoxy_vof_sliced).float())

        if self.debug:
            print(f"      coords: {coords_seq.shape}")
            print(f"      epoxy_vof: {epoxy_vof_sliced.shape}, "
                  f"range [{epoxy_vof_sliced.min():.4f}, {epoxy_vof_sliced.max():.4f}]")

    def _setup_statistics(self):
        """Compute or load normalization statistics."""
        node_stats_path = os.path.join(self._stats_dir, NODE_STATS_FILE)
        feat_stats_path = os.path.join(self._stats_dir, FEATURE_STATS_FILE)

        if self.split == "train":
            self._log("\n  Computing statistics from training data...")
            self.node_stats = self._compute_node_stats()
            #self.feature_stats = self._compute_feature_stats()
            # Hardcode feature stats to make normalization a no-op
            self.feature_stats = {
                "feature_mean": torch.zeros(1, dtype=torch.float32),
                "feature_std": torch.ones(1, dtype=torch.float32),
            }
            
            # Save for validation/inference (convert to pure Python types)
            node_stats_serializable = _stats_to_serializable(self.node_stats)
            feat_stats_serializable = _stats_to_serializable(self.feature_stats)
            
            save_json(node_stats_serializable, node_stats_path)
            save_json(feat_stats_serializable, feat_stats_path)
            self._log(f"  Saved statistics to {self._stats_dir}/")
            
        else:
            # Load from saved training stats
            if os.path.exists(node_stats_path) and os.path.exists(feat_stats_path):
                self._log(f"\n  Loading statistics from {self._stats_dir}/")
                self.node_stats = _stats_from_serializable(load_json(node_stats_path))
                #self.feature_stats = _stats_from_serializable(load_json(feat_stats_path))
                # Hardcode feature stats to make normalization a no-op
                self.feature_stats = {
                    "feature_mean": torch.zeros(1, dtype=torch.float32),
                    "feature_std": torch.ones(1, dtype=torch.float32),
                }
            else:
                self._log("\n  WARNING: No saved statistics found, computing from current split")
                self._log("           Run training first to generate statistics!")
                self.node_stats = self._compute_node_stats()
                #self.feature_stats = self._compute_feature_stats()
                # Hardcode feature stats to make normalization a no-op
                self.feature_stats = {
                    "feature_mean": torch.zeros(1, dtype=torch.float32),
                    "feature_std": torch.ones(1, dtype=torch.float32),
                }

        # Log statistics
#        self._log_statistics()

    def _log_statistics(self):
        """Log the computed/loaded statistics."""
        pos_mean = self.node_stats['pos_mean']
        pos_std = self.node_stats['pos_std']
        feat_mean = self.feature_stats['feature_mean']
        feat_std = self.feature_stats['feature_std']
        
        self._log(f"\n  Statistics:")
        
        # Handle both tensor and list formats
        if isinstance(pos_mean, torch.Tensor):
            self._log(f"    pos_mean:     [{pos_mean[0].item():.6f}, {pos_mean[1].item():.6f}, {pos_mean[2].item():.6f}]")
            self._log(f"    pos_std:      [{pos_std[0].item():.6f}, {pos_std[1].item():.6f}, {pos_std[2].item():.6f}]")
        else:
            self._log(f"    pos_mean:     {pos_mean}")
            self._log(f"    pos_std:      {pos_std}")
            
        if isinstance(feat_mean, torch.Tensor):
            self._log(f"    feature_mean: {feat_mean.item():.6f}")
            self._log(f"    feature_std:  {feat_std.item():.6f}")
        else:
            self._log(f"    feature_mean: {feat_mean}")
            self._log(f"    feature_std:  {feat_std}")

    def _compute_node_stats(self) -> dict:
        """Compute position statistics over all samples and time steps."""
        # Flatten all positions: [Total_points, 3]
        all_pos = torch.cat([p.reshape(-1, 3) for p in self.mesh_pos_seq], dim=0)
        
        mean = torch.mean(all_pos, dim=0)
        std = torch.std(all_pos, dim=0)
        std = torch.clamp(std, min=EPS)  # Prevent division by zero
        
        return {"pos_mean": mean, "pos_std": std}

    def _compute_feature_stats(self) -> dict:
        """Compute epoxy_vof statistics over all samples and time steps."""
        # Flatten all epoxy_vof: [Total_points, 1]
        all_vof = torch.cat([f.reshape(-1, 1) for f in self.epoxy_vof_seq], dim=0)
        
        mean = torch.mean(all_vof, dim=0)
        std = torch.std(all_vof, dim=0)
        std = torch.clamp(std, min=EPS)  # Prevent division by zero
        
        return {"feature_mean": mean, "feature_std": std}

    def _apply_normalization(self):
        """Apply normalization to all loaded data."""
        self._log("\n  Applying normalization...")

        # Ensure stats are tensors
        pos_mean = _to_tensor(self.node_stats["pos_mean"])
        pos_std = _to_tensor(self.node_stats["pos_std"])
        feat_mean = _to_tensor(self.feature_stats["feature_mean"])
        feat_std = _to_tensor(self.feature_stats["feature_std"])

        for i in range(len(self.mesh_pos_seq)):
            # Normalize positions: [T, N, 3]
            # Formula: x_norm = (x - mean) / std
            self.mesh_pos_seq[i] = (
                (self.mesh_pos_seq[i] - pos_mean.view(1, 1, -1)) 
                / pos_std.view(1, 1, -1)
            )
            
            # Normalize epoxy_vof: [T, N, 1]
            # Formula: vof_norm = (vof - mean) / std
            self.epoxy_vof_seq[i] = (
                (self.epoxy_vof_seq[i] - feat_mean.view(1, 1, -1)) 
                / feat_std.view(1, 1, -1)
            )

    def _print_summary(self):
        """Print dataset summary and verify tensor shapes."""
        self._log(f"\n{'='*70}")
        self._log(f"Dataset Summary: {self.name} ({self.split})")
        self._log(f"{'='*70}")
        self._log(f"  Total samples:     {self.length}")
        self._log(f"  Time steps (T):    {self.num_steps}")
        self._log(f"  Target steps:      {self.num_steps - 1} (T-1)")
        self._log(f"  Feature:           epoxy_vof")
        self._log(f"  Feature dimension: {self.NUM_FEATURES}")

        if self.length > 0:
            sample = self[0]
            N = sample.node_features['coords'].shape[0]
            
            self._log(f"\n  Sample 0 shapes:")
            self._log(f"    coords:   {sample.node_features['coords'].shape}  (expected: [N, 3])")
            self._log(f"    features: {sample.node_features['features'].shape}  (expected: [N, 1])")
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

        self._log(f"{'='*70}\n")

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
                - node_features["features"]: [N, 1] normalized VOF at t=0
                - node_target: [N, T-1] normalized future VOF values
        
        Shape flow:
            pos_seq:  [T, N, 3] -> coords:   [N, 3] (take t=0)
            vof_seq:  [T, N, 1] -> features: [N, 1] (take t=0)
                                -> target:   [N, T-1] (take t=1 to T-1)
        """
        pos_seq = self.mesh_pos_seq[idx]    # [T, N, 3]
        vof_seq = self.epoxy_vof_seq[idx]   # [T, N, 1]

        # Input: initial state (t=0)
        node_features = {
            "coords": pos_seq[0],           # [N, 3]
            "features": vof_seq[0]          # [N, 1]
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
