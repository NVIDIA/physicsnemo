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

"""
Dataset loaders for CO2 Sequestration data.
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import torch
from torch.utils.data import Dataset


class CO2SequestrationDataset(Dataset):
    """Dataset for CO2 sequestration modeling.

    This dataset loads pre-computed CO2 flow simulations for training
    neural operators. The data consists of:
    - Input (u): Initial conditions and reservoir properties
    - Output (a): Temporal evolution of CO2 plume (pressure or saturation)

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to the data directory containing .pt files
    mode : str
        Dataset split: 'train', 'val', or 'test'
    variable : str
        Variable to predict: 'pressure' (dP) or 'saturation' (sg)
    normalize : bool, optional
        Whether to normalize the data, by default True
    device : Union[str, torch.device], optional
        Device to load data onto, by default "cuda"

    Example
    -------
    >>> from pathlib import Path
    >>> data_dir = Path("data_lustre")
    >>> train_dataset = CO2SequestrationDataset(
    ...     data_path=data_dir,
    ...     mode='train',
    ...     variable='pressure',
    ...     normalize=True
    ... )
    >>> print(f"Dataset size: {len(train_dataset)}")
    >>> input, output = train_dataset[0]
    >>> print(f"Input shape: {input.shape}, Output shape: {output.shape}")

    Note
    ----
    The dataset expects the following file structure:
    - {variable}_train_a.pt, {variable}_train_u.pt  (Training)
    - {variable}_val_a.pt, {variable}_val_u.pt      (Validation)
    - {variable}_test_a.pt, {variable}_test_u.pt    (Testing)

    Where variable is either 'dP' (pressure) or 'sg' (saturation).
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        mode: str = "train",
        variable: str = "pressure",
        normalize: bool = True,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.mode = mode.lower()
        self.normalize = normalize

        # Set up device
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        self.device = device

        # Map variable name to file prefix
        var_map = {
            "pressure": "dP",
            "saturation": "sg",
            "dP": "dP",
            "sg": "sg",
        }

        if variable.lower() not in var_map:
            raise ValueError(
                f"Variable must be 'pressure' or 'saturation', got {variable}"
            )

        self.variable = var_map[variable.lower()]

        # Validate mode
        if self.mode not in ["train", "val", "test"]:
            raise ValueError(f"Mode must be 'train', 'val', or 'test', got {mode}")

        # Load data files
        self._load_data()

        # Compute normalization statistics
        if self.normalize:
            self._compute_normalization()

    def _load_data(self):
        """Load the .pt files from disk."""
        # Construct file paths
        # NOTE: _a.pt files contain INPUT (12 physical quantities)
        #       _u.pt files contain OUTPUT (1 channel: dP or sg to predict)
        input_file = self.data_path / f"{self.variable}_{self.mode}_a.pt"
        output_file = self.data_path / f"{self.variable}_{self.mode}_u.pt"

        # Check if files exist
        if not input_file.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_file}\n"
                f"Please ensure dataset is downloaded to {self.data_path}"
            )
        if not output_file.exists():
            raise FileNotFoundError(
                f"Output file not found: {output_file}\n"
                f"Please ensure dataset is downloaded to {self.data_path}"
            )

        # Minimal loading message (only on rank 0 if distributed)
        try:
            from physicsnemo.distributed import DistributedManager

            dist = DistributedManager()
            if dist.rank == 0:
                print(
                    f"Loading {self.mode} data: {input_file.name} -> {output_file.name}"
                )
        except:
            print(f"Loading {self.mode} data: {input_file.name} -> {output_file.name}")

        # Load tensors
        self.input_data = torch.load(input_file, map_location="cpu")
        self.output_data = torch.load(output_file, map_location="cpu")

        # Data format: (Height × Width × Time × Channels)
        # INPUT (_a.pt):  (N, H, W, T, C=12) - 12 physical quantities as input channels
        # OUTPUT (_u.pt): (N, H, W, T) - single output (pressure dP OR saturation sg)

        # Validate input data shape - must be (N, H, W, T, C) where C can vary
        if self.input_data.dim() != 5:
            raise ValueError(
                f"Input data must be 5D (N, H, W, T, C), got {self.input_data.dim()}D "
                f"with shape {self.input_data.shape}"
            )

        # Validate output data shape - must be (N, H, W, T) scalar field
        if self.output_data.dim() != 4:
            raise ValueError(
                f"Output data must be 4D (N, H, W, T), got {self.output_data.dim()}D "
                f"with shape {self.output_data.shape}"
            )

        # Only print on rank 0 if distributed
        try:
            from physicsnemo.distributed import DistributedManager

            dist = DistributedManager()
            if dist.rank == 0:
                print(
                    f"  Loaded {len(self.input_data)} samples | Input: {self.input_data.shape} | Output: {self.output_data.shape}"
                )
        except:
            print(
                f"  Loaded {len(self.input_data)} samples | Input: {self.input_data.shape} | Output: {self.output_data.shape}"
            )

    def _compute_normalization(self):
        """Compute normalization statistics (mean and std) from training data."""
        # For training data, compute statistics
        # Data shape: Input: (N, H, W, T, 12), Output: (N, H, W, T)
        if self.mode == "train":
            # Compute mean and std across batch, spatial, and temporal dimensions
            # Keep channel dimension separate for independent normalization (inputs only)
            self.input_mean = self.input_data.mean(dim=(0, 1, 2, 3), keepdim=True)
            self.input_std = self.input_data.std(dim=(0, 1, 2, 3), keepdim=True)
            # Output is scalar field (N, H, W, T), compute single mean/std
            self.output_mean = self.output_data.mean()
            self.output_std = self.output_data.std()

            # Avoid division by zero
            self.input_std = torch.where(
                self.input_std > 1e-6, self.input_std, torch.ones_like(self.input_std)
            )
            if self.output_std < 1e-6:
                self.output_std = torch.tensor(1.0)

            # Only print on rank 0 if distributed
            try:
                from physicsnemo.distributed import DistributedManager

                dist = DistributedManager()
                if dist.rank == 0:
                    print(
                        f"  Normalization: Output mean={self.output_mean.item():.4f}, std={self.output_std.item():.4f}"
                    )
            except:
                print(
                    f"  Normalization: Output mean={self.output_mean.item():.4f}, std={self.output_std.item():.4f}"
                )
        else:
            # For val/test, initialize with identity normalization
            # (Should be set from training set in practice)
            # Input: 5D to match data shape (1, 1, 1, 1, C)
            # Output: scalar (no dimensions)
            self.input_mean = torch.zeros((1, 1, 1, 1, self.input_data.shape[-1]))
            self.input_std = torch.ones((1, 1, 1, 1, self.input_data.shape[-1]))
            self.output_mean = torch.tensor(0.0)
            self.output_std = torch.tensor(1.0)

    def set_normalization(
        self,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor,
    ):
        """Set normalization parameters from external source (e.g., training set).

        Parameters
        ----------
        input_mean : torch.Tensor
            Mean of input data
        input_std : torch.Tensor
            Standard deviation of input data
        output_mean : torch.Tensor
            Mean of output data
        output_std : torch.Tensor
            Standard deviation of output data
        """
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std

    def get_normalization_stats(self) -> Tuple[torch.Tensor, ...]:
        """Return normalization statistics.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            (input_mean, input_std, output_mean, output_std)
        """
        return (self.input_mean, self.input_std, self.output_mean, self.output_std)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.input_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset.

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (input_tensor, output_tensor)
            input: (H, W, T, C) where C=12 (12 physical quantities)
            output: (H, W, T) - single scalar field (pressure or saturation)
        """
        # Get data (keep on CPU for now)
        input_sample = self.input_data[
            idx
        ]  # (H=96, W=200, T=24, C=12) - 12 input channels
        output_sample = self.output_data[idx]  # (H=96, W=200, T=24) - scalar output

        # Normalize if enabled
        if self.normalize:
            # Ensure normalization stats are on same device as data
            input_mean = self.input_mean.to(input_sample.device)
            input_std = self.input_std.to(input_sample.device)
            output_mean = self.output_mean.to(output_sample.device)
            output_std = self.output_std.to(output_sample.device)

            input_sample = (input_sample - input_mean) / input_std
            output_sample = (output_sample - output_mean) / output_std

        return input_sample, output_sample


def collate_fn_3d(batch):
    """Custom collate function for 3D FNO data.

    Properly stacks samples into batches without adding extra dimensions.
    - inputs: (B, H, W, T, C) where C=12 input channels
    - targets: (B, H, W, T) single scalar field (no channel dimension)
    """
    inputs = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1] for item in batch], dim=0)
    return inputs, targets


def create_dataloaders(
    data_path: Union[str, Path],
    variable: str = "pressure",
    batch_size: int = 4,
    normalize: bool = True,
    num_workers: int = 4,
    device: Union[str, torch.device] = "cuda",
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """Create train, validation, and test dataloaders.

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to the data directory
    variable : str, optional
        Variable to predict ('pressure' or 'saturation'), by default 'pressure'
    batch_size : int, optional
        Batch size, by default 4
    normalize : bool, optional
        Whether to normalize data, by default True
    num_workers : int, optional
        Number of dataloader workers, by default 4
    device : Union[str, torch.device], optional
        Device to load data onto, by default "cuda"

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        (train_loader, val_loader, test_loader)

    Example
    -------
    >>> train_loader, val_loader, test_loader = create_dataloaders(
    ...     data_path="data_lustre",
    ...     variable="pressure",
    ...     batch_size=8,
    ... )
    """
    from torch.utils.data import DataLoader

    # Check if running in distributed mode
    try:
        from physicsnemo.distributed import DistributedManager

        dist = DistributedManager()
        is_distributed = dist.world_size > 1
    except:
        is_distributed = False

    # Create datasets
    train_dataset = CO2SequestrationDataset(
        data_path=data_path,
        mode="train",
        variable=variable,
        normalize=normalize,
        device="cpu",  # Keep on CPU, let DataLoader handle transfer
    )
    val_dataset = CO2SequestrationDataset(
        data_path=data_path,
        mode="val",
        variable=variable,
        normalize=normalize,
        device="cpu",
    )
    test_dataset = CO2SequestrationDataset(
        data_path=data_path,
        mode="test",
        variable=variable,
        normalize=normalize,
        device="cpu",
    )

    # Share normalization statistics from training set
    if normalize:
        norm_stats = train_dataset.get_normalization_stats()

        # Synchronize normalization statistics across all ranks in distributed training
        if is_distributed:
            import torch.distributed as dist_torch

            for stat in norm_stats:
                # Broadcast normalization stats from rank 0 to all other ranks
                dist_torch.broadcast(stat, src=0)

        # Apply synchronized normalization to validation and test sets
        val_dataset.set_normalization(*norm_stats)
        test_dataset.set_normalization(*norm_stats)

    # Determine if we should use pin_memory for faster CPU->GPU transfer
    use_pin_memory = isinstance(device, torch.device) and device.type == "cuda"
    if isinstance(device, str):
        use_pin_memory = device == "cuda"

    # Create distributed sampler if running in multi-GPU mode
    train_sampler = None
    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            drop_last=False,
        )

    # Create distributed samplers for val/test if in distributed mode
    val_sampler = None
    test_sampler = None
    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler

        val_sampler = DistributedSampler(
            val_dataset,
            shuffle=False,
            drop_last=False,
        )
        test_sampler = DistributedSampler(
            test_dataset,
            shuffle=False,
            drop_last=False,
        )

    # Create dataloaders with optimized settings and custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        collate_fn=collate_fn_3d,  # Custom collate for 3D data
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn_3d,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn_3d,
    )

    return train_loader, val_loader, test_loader
