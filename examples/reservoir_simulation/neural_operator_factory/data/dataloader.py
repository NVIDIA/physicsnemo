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
Unified dataset loaders for reservoir simulation neural operators.

Supports both 3D (2D spatial + time) and 4D (3D spatial + time) datasets:
- 3D: Input (N, H, W, T, C), Output (N, H, W, T) - e.g., CO2 sequestration
- 4D: Input (N, X, Y, Z, T, C), Output (N, X, Y, Z, T) - e.g., Norne field
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from data.file_resolution import resolve_data_files
from data.mask_detection import MaskResult, detect_mask
from data.normalization import (
    NormStats,
    compute_norm_stats,
    identity_norm_stats,
    normalize_sample,
)


def _log_message(msg: str, rank_zero_only: bool = True):
    """Print message, optionally only on rank 0 in distributed mode."""
    try:
        from physicsnemo.distributed import DistributedManager

        dist = DistributedManager()
        if not rank_zero_only or dist.rank == 0:
            print(msg)
    except Exception:
        print(msg)


def _load_tensor(path: Path) -> torch.Tensor:
    """Load a ``.pt`` tensor into CPU memory.

    Uses a standard bulk read which is optimal when the full tensor
    will be scanned (e.g. for normalization statistics).
    """
    return torch.load(path, map_location="cpu")


class ReservoirDataset(Dataset):
    """Unified dataset for reservoir simulation modeling.

    Automatically detects and handles both 3D and 4D data:
    - 3D: (N, H, W, T, C) input, (N, H, W, T) output
    - 4D: (N, X, Y, Z, T, C) input, (N, X, Y, Z, T) output

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to the data directory.
    mode : str
        Dataset split: ``'train'``, ``'val'``, or ``'test'``.
    input_file : str, optional
        Input filename pattern (supports ``{mode}`` placeholder).
    output_file : str, optional
        Output filename pattern (supports ``{mode}`` placeholder).
    variable : str, optional
        ``'pressure'`` or ``'saturation'`` for CO2 naming convention.
    normalize : bool
        Z-score normalize using training-set statistics (default ``True``).
    expected_dimensions : str, optional
        ``'3d'`` or ``'4d'``. Raises on mismatch with loaded data.
    use_mask : bool
        Enable inactive-cell mask detection (default ``False``).
    mask_channel : int, optional
        Explicit mask channel index (overrides auto-detection).
    num_timesteps : int, optional
        Truncate the time axis to the first *N* steps (train/val only).
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        mode: str = "train",
        input_file: Optional[str] = None,
        output_file: Optional[str] = None,
        variable: Optional[str] = None,
        normalize: bool = True,
        expected_dimensions: Optional[str] = None,
        use_mask: bool = False,
        mask_channel: Optional[int] = None,
        num_timesteps: Optional[int] = None,
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.mode = mode.lower()
        self.normalize = normalize
        self.variable = variable
        self.expected_dimensions = (
            expected_dimensions.lower() if expected_dimensions else None
        )
        self.use_mask = use_mask
        self._config_mask_channel = mask_channel
        self._num_timesteps = num_timesteps

        if self.mode not in ("train", "val", "test"):
            raise ValueError(f"Mode must be 'train', 'val', or 'test', got {mode}")

        # --- File resolution (delegated) ---
        self.input_file, self.output_file = resolve_data_files(
            self.data_path, self.mode, input_file, output_file, variable
        )

        # --- Load data ---
        self._load_data()

        if self._num_timesteps is not None:
            T = self._num_timesteps
            self.input_data = self.input_data[..., :T, :]
            self.output_data = self.output_data[..., :T]
            _log_message(f"  Truncated to {T} timesteps")

        # --- Dimension detection ---
        self._detect_dimensions()

        # --- Mask detection (delegated) ---
        self.mask_channel: Optional[int] = None
        self.mask_per_sample: bool = False
        self.static_mask: Optional[torch.Tensor] = None
        if self.use_mask:
            self._apply_mask_detection()

        # --- Normalization (delegated) ---
        self._norm_stats: Optional[NormStats] = None
        if self.normalize:
            self._init_normalization()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self):
        """Load input/output tensors from disk."""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        if not self.output_file.exists():
            raise FileNotFoundError(f"Output file not found: {self.output_file}")

        _log_message(
            f"Loading {self.mode} data: "
            f"{self.input_file.name} -> {self.output_file.name}"
        )

        self.input_data = _load_tensor(self.input_file)
        self.output_data = _load_tensor(self.output_file)

        _log_message(
            f"  Loaded {len(self.input_data)} samples | "
            f"Input: {tuple(self.input_data.shape)} | "
            f"Output: {tuple(self.output_data.shape)}"
        )

    # ------------------------------------------------------------------
    # Dimension detection
    # ------------------------------------------------------------------

    def _detect_dimensions(self):
        """Detect 3D vs 4D from tensor shapes and validate against config."""
        input_ndim = self.input_data.dim()
        output_ndim = self.output_data.dim()

        if input_ndim == 5 and output_ndim == 4:
            self.dimensions = "3d"
            self.spatial_dims = 2
            self.dim_names = ("H", "W", "T")
        elif input_ndim == 6 and output_ndim == 5:
            self.dimensions = "4d"
            self.spatial_dims = 3
            self.dim_names = ("X", "Y", "Z", "T")
        else:
            raise ValueError(
                f"Unsupported data dimensions!\n"
                f"  Input: {input_ndim}D {tuple(self.input_data.shape)}\n"
                f"  Output: {output_ndim}D {tuple(self.output_data.shape)}\n"
                f"Expected:\n"
                f"  3D: Input (N, H, W, T, C), Output (N, H, W, T)\n"
                f"  4D: Input (N, X, Y, Z, T, C), Output (N, X, Y, Z, T)"
            )

        if (
            self.expected_dimensions is not None
            and self.dimensions != self.expected_dimensions
        ):
            raise ValueError(
                f"Dimension mismatch!\n"
                f"   Config expects: {self.expected_dimensions}\n"
                f"   Data has: {self.dimensions}\n"
                f"   Input shape: {tuple(self.input_data.shape)}\n"
                f"   Please update arch.dimensions in config to "
                f"'{self.dimensions}' "
                f"or use a dataset with {self.expected_dimensions} data."
            )

        self.num_samples = self.input_data.shape[0]
        self.spatial_shape = tuple(self.input_data.shape[1:-2])
        self.time_steps = self.input_data.shape[-2]
        self.num_channels = self.input_data.shape[-1]

        _log_message(
            f"  Detected: {self.dimensions.upper()} | "
            f"Spatial: {self.spatial_shape} | "
            f"T: {self.time_steps} | C: {self.num_channels}"
        )

    # ------------------------------------------------------------------
    # Mask detection (delegates to data.mask_detection)
    # ------------------------------------------------------------------

    def _apply_mask_detection(self):
        """Run mask detection and store results on self."""
        result: MaskResult = detect_mask(
            self.input_data, self.output_data, self._config_mask_channel
        )
        self.mask_channel = result.channel
        self.mask_per_sample = result.per_sample
        self.static_mask = result.static_mask

        if result.method == "none":
            _log_message("  Mask: none (all cells active)")
        else:
            pct = 100 * result.n_active / result.n_total if result.n_total else 0
            ps = " (per-sample)" if result.per_sample else ""
            _log_message(
                f"  Mask [{result.method} ch {result.channel}]: "
                f"{result.n_active}/{result.n_total} active "
                f"({pct:.1f}%){ps}"
            )

    def get_static_mask(self):
        """Return static spatial mask or None."""
        return self.static_mask

    # ------------------------------------------------------------------
    # Normalization (delegates to data.normalization)
    # ------------------------------------------------------------------

    def _init_normalization(self):
        """Compute or prepare normalization statistics."""
        if self.mode == "train":
            self._norm_stats = compute_norm_stats(self.input_data, self.output_data)
            _log_message(
                f"  Normalization: Output "
                f"mean={self._norm_stats.output_mean.item():.4f}, "
                f"std={self._norm_stats.output_std.item():.4f}"
            )
        else:
            self._norm_stats = identity_norm_stats(
                self.input_data.dim(), self.num_channels
            )

    # Backward-compatible properties so existing code that reads
    # ds.input_mean / ds.input_std / ds.output_mean / ds.output_std
    # continues to work.

    @property
    def input_mean(self):
        """Input channel means (broadcastable)."""
        return self._norm_stats.input_mean if self._norm_stats else None

    @input_mean.setter
    def input_mean(self, value):
        """Set input channel means."""
        if self._norm_stats is None:
            self._norm_stats = NormStats(value, value, value, value)
        self._norm_stats.input_mean = value

    @property
    def input_std(self):
        """Input channel standard deviations (broadcastable)."""
        return self._norm_stats.input_std if self._norm_stats else None

    @input_std.setter
    def input_std(self, value):
        """Set input channel standard deviations."""
        if self._norm_stats is not None:
            self._norm_stats.input_std = value

    @property
    def output_mean(self):
        """Scalar output mean."""
        return self._norm_stats.output_mean if self._norm_stats else None

    @output_mean.setter
    def output_mean(self, value):
        """Set scalar output mean."""
        if self._norm_stats is not None:
            self._norm_stats.output_mean = value

    @property
    def output_std(self):
        """Scalar output standard deviation."""
        return self._norm_stats.output_std if self._norm_stats else None

    @output_std.setter
    def output_std(self, value):
        """Set scalar output standard deviation."""
        if self._norm_stats is not None:
            self._norm_stats.output_std = value

    def set_normalization(
        self,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor,
    ):
        """Set normalization parameters from an external source."""
        self._norm_stats = NormStats(input_mean, input_std, output_mean, output_std)

    def get_normalization_stats(self) -> Tuple[torch.Tensor, ...]:
        """Return ``(input_mean, input_std, output_mean, output_std)``."""
        if self._norm_stats is None:
            raise RuntimeError("Normalization not initialized")
        return self._norm_stats.as_tuple()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single ``(input, output)`` sample.

        Returns
        -------
        Tuple[Tensor, Tensor]
            3D: ``(H, W, T, C)``, ``(H, W, T)``
            4D: ``(X, Y, Z, T, C)``, ``(X, Y, Z, T)``
        """
        inp = self.input_data[idx]
        out = self.output_data[idx]

        if self.normalize and self._norm_stats is not None:
            inp, out = normalize_sample(inp, out, self._norm_stats)

        return inp, out


# =====================================================================
# Collate
# =====================================================================


def collate_fn(batch):
    """Stack samples along the batch dimension (3D and 4D agnostic)."""
    inputs = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    return inputs, targets


# =====================================================================
# Dataloader factory
# =====================================================================


def create_dataloaders(
    data_path: Union[str, Path],
    batch_size: int = 4,
    normalize: bool = True,
    num_workers: int = 4,
    device: Union[str, torch.device] = "cuda",
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    variable: Optional[str] = None,
    expected_dimensions: Optional[str] = None,
    use_mask: bool = False,
    mask_channel: Optional[int] = None,
    num_timesteps: Optional[int] = None,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Create train, validation, and test dataloaders.

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to the data directory.
    batch_size : int
        Batch size per GPU (default 4).
    normalize : bool
        Z-score normalize (default ``True``).
    num_workers : int
        DataLoader worker processes (default 4).
    device : Union[str, torch.device]
        Target device for ``pin_memory`` (default ``"cuda"``).
    input_file, output_file : str, optional
        Filename patterns with ``{mode}`` placeholder.
    variable : str, optional
        ``'pressure'`` or ``'saturation'`` for CO2 convention.
    expected_dimensions : str, optional
        ``'3d'`` or ``'4d'``; raises on mismatch.
    use_mask : bool
        Enable mask detection (default ``False``).
    mask_channel : int, optional
        Explicit mask channel (overrides auto-detect).
    num_timesteps : int, optional
        Truncate train/val time axis; test keeps all.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        ``(train_loader, val_loader, test_loader)``
    """
    from torch.utils.data import DataLoader

    try:
        from physicsnemo.distributed import DistributedManager

        dist = DistributedManager()
        is_distributed = dist.world_size > 1
    except Exception:
        is_distributed = False

    dataset_kwargs: dict = {
        "data_path": data_path,
        "input_file": input_file,
        "output_file": output_file,
        "variable": variable,
        "normalize": normalize,
        "expected_dimensions": expected_dimensions,
        "use_mask": use_mask,
        "mask_channel": mask_channel,
    }

    train_dataset = ReservoirDataset(
        mode="train", num_timesteps=num_timesteps, **dataset_kwargs
    )
    val_dataset = ReservoirDataset(
        mode="val", num_timesteps=num_timesteps, **dataset_kwargs
    )
    test_dataset = ReservoirDataset(mode="test", **dataset_kwargs)

    if normalize:
        norm_stats = train_dataset.get_normalization_stats()

        if is_distributed:
            import torch.distributed as dist_torch

            gpu_stats = []
            for stat in norm_stats:
                s = stat.cuda()
                dist_torch.broadcast(s, src=0)
                gpu_stats.append(s.cpu())
            norm_stats = tuple(gpu_stats)

        val_dataset.set_normalization(*norm_stats)
        test_dataset.set_normalization(*norm_stats)

    use_pin_memory = (isinstance(device, torch.device) and device.type == "cuda") or (
        isinstance(device, str) and device == "cuda"
    )

    train_sampler = val_sampler = test_sampler = None
    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False)

    loader_kwargs: dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
        "persistent_workers": num_workers > 0,
        "collate_fn": collate_fn,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, sampler=val_sampler, **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, sampler=test_sampler, **loader_kwargs
    )

    _log_message(
        f"Created dataloaders: {train_dataset.dimensions.upper()} data | "
        f"Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, "
        f"Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader


# =====================================================================
# Utility
# =====================================================================


def get_dataset_info(data_path: Union[str, Path], **kwargs) -> Dict:
    """Quick dataset introspection without full loading overhead.

    Returns
    -------
    dict
        Keys: dimensions, spatial_shape, time_steps, num_channels, num_samples.
    """
    ds = ReservoirDataset(data_path, mode="train", normalize=False, **kwargs)
    return {
        "dimensions": ds.dimensions,
        "spatial_shape": ds.spatial_shape,
        "time_steps": ds.time_steps,
        "num_channels": ds.num_channels,
        "num_samples": {
            "train": len(ds),
            "val": len(
                ReservoirDataset(data_path, mode="val", normalize=False, **kwargs)
            ),
            "test": len(
                ReservoirDataset(data_path, mode="test", normalize=False, **kwargs)
            ),
        },
    }
