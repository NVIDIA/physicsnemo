# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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


from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Union
import warnings

import numpy as np
import torch
import json
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.datapipes.datapipe import Datapipe
from physicsnemo.datapipes.meta import DatapipeMetaData


def _read_npz_sample(filename: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Read a single .npz file containing multiple fields (vs, vp, vx, vz, etc.).

    Parameters
    ----------
    filename : Union[str, Path]
        Path to the .npz file

    Returns
    -------
        Dictionary containing the data fields
    """
    with np.load(filename) as data:
        # Create a copy of the data to avoid issues with the file being closed
        return {key: data[key] for key in data.keys()}


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "EFWIDatapipe"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = True


class EFWIDataset(Dataset):
    """
    Dataset for E-FWI.

    Parameters
    ----------
    data_dir : str | Path
        Path to the dataset directory containing the .npz files. The files are
        expected to be of the form 'data_dir/samples/train_sample_<idx>.npz' if
        ``phase`` is 'train' or 'data_dir/samples/test_sample_<idx>.npz' if
        ``phase`` is 'test'.
    phase : Literal["train", "test"]
        Phase to load the dataset from.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        phase: Literal["train", "test"],
    ) -> None:
        # Parameters validation and input pre-processing
        if isinstance(data_dir, str):
            data_dir: Path = Path(data_dir)
        self.data_dir: Path = data_dir.expanduser() / "samples"
        if not self.data_dir.exists():
            raise AssertionError(f"Path {self.data_dir} does not exist")
        if not self.data_dir.is_dir():
            raise AssertionError(f"Path {self.data_dir} is not a directory")
        if phase not in ["train", "test"]:
            raise AssertionError(
                f"phase should be one of ['train', 'test'], got {phase}"
            )
        self.phase = phase

        self.sample_files: list[Path] = sorted(
            self.data_dir.glob(f"{self.phase}_sample_*.npz"),
            key=lambda x: int(x.stem.split("_")[-1]),
        )

        if len(self.sample_files) == 0:
            raise AssertionError(f"No samples found in {self.data_dir}")

        # Load dataset statistics
        stats_file: Path = self.data_dir.parent / "stats.json"
        if stats_file.exists():
            with open(stats_file, "r") as f:
                self.stats = json.load(f)
        else:
            self.stats = None
            warnings.warn(
                f"Stats file {stats_file} not found. Normalization will not be available."
            )

    def __len__(self) -> int:
        return len(self.sample_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Read and load samples from .npz files.
        Each .npz file contains multiple fields (vs, vp, vx, vz).

        Parameters
        ----------
        idx : int
            Index or indices of samples to load

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of tensors where keys are the variables in the dataset
            and values are the corresponding tensors.
        """

        # Initialize data dictionary
        data: Dict[str, torch.Tensor] = {}

        # Load the sample file
        sample_data: Dict[str, np.ndarray] = _read_npz_sample(self.sample_files[idx])

        # Initialize data arrays on first iteration
        for key, value in sample_data.items():
            data[key] = torch.tensor(value, dtype=torch.float32, device="cpu")

        return data


class EFWIDatapipe(DataLoader):
    """
    Datapipe for E-FWI dataset.

    Parameters
    ----------
    data_dir : str | Path
        Path to the dataset directory containing the .npz files. The files are
        expected to be of the form ``data_dir/samples/train_sample_<idx>.npz``
        if ``phase`` is ``train`` or ``data_dir/samples/test_sample_<idx>.npz``
        if ``phase`` is ``test``.
    phase : Literal["train", "test"]
        Phase to load the dataset from.
    batch_size_per_device : int
        Batch size per device.
    seed : int, optional, default=0
        Random seed. Used when shuffling the dataset.
    shuffle : bool, optional, default=True
        Whether to shuffle the dataset.
    num_workers : int, optional, default=1
        Number of workers to use for loading the dataset.
    device : str | torch.device, optional, default="cuda"
        Device to use for loading the dataset.
    process_rank : int, optional, default=0
        Rank of the process. Used for distributed training.
    world_size : int, optional, default=1
        Total number of processes. Used for distributed training.
    prefetch_factor : int, optional, default=2
        Number of batches to prefetch.
    use_sharding : bool, optional, default=None
        Whether to use sharding. If None, sharding is used if world_size > 1.
    """

    def __init__(
        self,
        data_dir: str | Path,
        phase: Literal["train", "test"],
        batch_size_per_device: int,
        seed: int = 0,
        shuffle: bool = True,
        num_workers: int = 1,
        device: Union[str, torch.device] = "cuda",
        process_rank: int = 0,
        world_size: int = 1,
        prefetch_factor: Optional[int] = 2,
        use_sharding: Optional[bool] = None,
    ) -> None:
        if isinstance(device, str):
            device: torch.device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device: torch.device = torch.device("cuda:0")
        self.device = device

        dataset: EFWIDataset = EFWIDataset(data_dir=data_dir, phase=phase)
        self.phase = phase

        # Determine whether to use sharding
        should_shard: bool = use_sharding if use_sharding is not None else True

        if should_shard and world_size > 1:
            sampler: DistributedSampler = DistributedSampler(
                dataset=dataset,
                num_replicas=world_size,
                rank=process_rank,
                shuffle=shuffle,
                seed=seed,
                drop_last=False,
            )
            shuffle: bool | None = None
            generator = None
        else:
            sampler: DistributedSampler | None = None
            generator = torch.Generator("cpu").manual_seed(seed)

        super().__init__(
            dataset=dataset,
            batch_size=batch_size_per_device,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=(pin_memory := device.type == "cuda"),
            shuffle=shuffle,
            timeout=0,
            worker_init_fn=None,
            multiprocessing_context=None,
            prefetch_factor=prefetch_factor,
            pin_memory_device=(str(self.device) if pin_memory else ""),
            generator=generator,
        )

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for the datapipe. Used for shuffling in distributed
        training.

        Parameters
        ----------
        epoch : int
            The epoch number.
        """
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def get_stats(
        self,
        metric: Literal["mean", "std", "min", "max"],
        phase: Literal["train", "test"] = "train",
    ) -> Dict[str, float]:
        """Return training statistics for each variable.

        Parameters
        ----------
        - metric : str
            One of "mean", "std", "min", "max" corresponding to the
            statistics stored in the YAML file.
        - phase : Literal["train", "test"]
            The phase to get the statistics for.
        """

        if self.dataset.stats is None:
            raise RuntimeError("Statistics file not available for dataset.")

        if metric not in {"mean", "std", "min", "max"}:
            raise ValueError(f"Unknown metric '{metric}'.")

        if phase not in {"train", "test"}:
            raise ValueError(f"Unknown phase '{phase}'.")

        return {k: v[phase][metric] for k, v in self.dataset.stats.items()}

    def __iter__(self):
        for data in super().__iter__():
            for key in data.keys():
                data[key] = data[key].to(self.device)
            yield data
