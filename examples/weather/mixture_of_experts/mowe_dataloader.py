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

import h5py
import numpy as np
import torch

try:
    import nvidia.dali as dali
    import nvidia.dali.plugin.pytorch as dali_pth
except ImportError:
    raise ImportError(
        "DALI dataset requires NVIDIA DALI package to be installed. "
        + "The package can be installed at:\n"
        + "https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html"
    )

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pytz

# Assuming these utility files exist in the specified paths
from physicsnemo.datapipes.climate.utils.invariant import latlon_grid
from physicsnemo.datapipes.climate.utils.zenith_angle import cos_zenith_angle

from physicsnemo.datapipes.datapipe import Datapipe
from physicsnemo.datapipes.meta import DatapipeMetaData

Tensor = torch.Tensor


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "ModelERA5HDF5"
    auto_device: bool = True
    cuda_graphs: bool = True
    ddp_sharding: bool = True


class MoWEDatapipe(Datapipe):
    """
    DALI data pipeline for loading multiple model forecasts and ERA5 data from HDF5 files.
    This pipeline handles variable timestep offsets, multi-step forecasts, and different file lengths.


    Parameters
    -----------
    data_dirs (List[str]): A list of directory paths. The first path must
        point to the ERA5 (ground truth) data, and subsequent paths
        point to the data for each model forecast.
    stats_dir (Union[str, None]): Path to the directory containing
        'global_means.npy' and 'global_stds.npy' for data normalization.
        If None, normalization is skipped.
    in_channels (List[int]): List of channel indices to select from the model
        HDF5 files.
    out_channels (List[int]): List of channel indices to select from the
        ERA5 HDF5 file. These are the target variables.
    orig_index (List[int]): A list that maps the sorted `out_channels` back
        to their original order corresponding to `in_channels`. This is
        used to align the ground truth channels with the input channels
        after data loading.
    model_time_offsets (List[int]): A list of integer offsets (in time steps)
        for each model. This accounts for delays in model initialization
        relative to the ground truth time. Length must be `len(data_dirs) - 1`.
    max_lead_time (int): The maximum forecast lead time (in time steps) to
        sample from. A random lead time between 1 and this value will be
        chosen for each sample.
    batch_size (int, optional): The batch size. Defaults to 1.
    shuffle_model_idx (Union[int, None], optional): The index of the model
        (from the model list) whose input channels should be shuffled.
        Requires `shuffle_channel_order`. Defaults to None.
    shuffle_channel_order (Union[List[int], None], optional): The new order
        of channels to apply to the model specified by `shuffle_model_idx`.
        Defaults to None.
    num_samples_per_year (Union[int, None], optional): The number of time
        steps to use from each year's HDF5 file. If None, it's inferred
        from the first file. Defaults to 1459 (for 6-hourly data in a leap year).
    latlon_resolution (Union[Tuple[int, int], None], optional): The (latitude,
        longitude) resolution of the data. Required if `use_cos_zenith` is True.
        Defaults to None.
    interpolation_type (Union[str, None], optional): DALI interpolation
        method as a string (e.g., "INTERP_LINEAR"). If None, no resizing
        is performed. Defaults to None.
    patch_size (Union[Tuple[int, int], int, None], optional): If specified,
        crops the image to be divisible by the patch size. Not compatible
        with `use_cos_zenith`. Defaults to None.
    use_cos_zenith (bool, optional): If True, computes and adds the cosine
        of the solar zenith angle to the output. Defaults to False.
    cos_zenith_args (Dict, optional): Dictionary of arguments for the
        cosine zenith angle calculation. Defaults to {}.
    use_time_of_year_index (bool, optional): If True, adds a time-of-year
        index to the output. Defaults to False.
    shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
    num_workers (int, optional): The number of parallel workers for data
        loading. Defaults to 1.
    device (Union[str, torch.device], optional): The device to place
        tensors on ('cuda' or 'cpu'). Defaults to "cuda".
    process_rank (int, optional): The rank of the current process for
        distributed training. Defaults to 0.
    world_size (int, optional): The total number of processes for
        distributed training. Defaults to 1.
    mode (str, optional): The dataset mode, 'train' or 'val'.
        Defaults to 'train'.
    ratio (float, optional): The train/validation split ratio based on
        years of data. Defaults to 0.8.

    Raises:
    -------
    ValueError: If input parameters are inconsistent or invalid.
    IOError: If data or statistics directories/files do not exist.
    FileNotFoundError: If no common HDF5 files are found across `data_dirs`.
    """

    def __init__(
        self,
        data_dirs: List[str],
        stats_dir: Union[str, None],
        in_channels: List[int],
        out_channels: List[int],
        orig_index: List[int],
        model_time_offsets: List[int],
        max_lead_time: int,
        batch_size: int = 1,
        shuffle_model_idx: Union[int, None] = None,
        shuffle_channel_order: Union[List[int], None] = None,
        num_samples_per_year: Union[int, None] = 1459,
        latlon_resolution: Union[Tuple[int, int], None] = None,
        interpolation_type: Union[str, None] = None,
        patch_size: Union[Tuple[int, int], int, None] = None,
        use_cos_zenith: bool = False,
        cos_zenith_args: Dict = {},
        use_time_of_year_index: bool = False,
        shuffle: bool = True,
        num_workers: int = 1,
        device: Union[str, torch.device] = "cuda",
        process_rank: int = 0,
        world_size: int = 1,
        mode: str = "train",
        ratio=0.8,
    ):
        """Initializes the MoeERA5HDF5DatapipeMulti."""
        super().__init__(meta=MetaData())
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.data_dirs = [Path(d) for d in data_dirs]
        self.stats_dir = Path(stats_dir) if stats_dir is not None else None

        self.in_channels = sorted(in_channels)
        self.out_channels = sorted(out_channels)
        self.orig_index = orig_index
        self.model_time_offsets = model_time_offsets
        self.max_lead_time = max_lead_time

        self.shuffle_model_idx = shuffle_model_idx
        self.shuffle_channel_order = shuffle_channel_order

        self.num_samples_per_year = num_samples_per_year
        self.latlon_resolution = latlon_resolution
        self.interpolation_type = interpolation_type
        self.use_cos_zenith = use_cos_zenith
        self.cos_zenith_args = cos_zenith_args
        self.use_time_of_year_index = use_time_of_year_index
        self.process_rank = process_rank
        self.world_size = world_size
        self.n_models = len(self.data_dirs) - 1
        self.mode = mode
        self.ratio = ratio

        if self.n_models <= 0:
            raise ValueError(
                "`data_dirs` must contain at least two paths: one for ERA5 and one for a model."
            )
        if len(self.model_time_offsets) != self.n_models:
            raise ValueError(
                f"Length of `model_time_offsets` ({len(self.model_time_offsets)}) "
                f"must match the number of models ({self.n_models})."
            )
        if self.shuffle_model_idx is not None:
            if self.shuffle_channel_order is None:
                raise ValueError(
                    "`shuffle_channel_order` must be provided if `shuffle_model_idx` is set."
                )
            if len(self.shuffle_channel_order) != len(self.in_channels):
                raise ValueError(
                    "Length of `shuffle_channel_order` must match the number of `in_channels`."
                )
            if not (0 <= self.shuffle_model_idx < self.n_models):
                raise ValueError(
                    f"`shuffle_model_idx` must be between 0 and {self.n_models - 1}."
                )

        if use_cos_zenith:
            cos_zenith_args["dt"] = cos_zenith_args.get("dt", 6.0)
            cos_zenith_args["latlon_bounds"] = cos_zenith_args.get(
                "latlon_bounds", ((90, -90), (0, 360))
            )
        self.latlon_bounds = cos_zenith_args.get("latlon_bounds")

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size

        if isinstance(device, str):
            device = torch.device(device)
        self.device = torch.device(device.type, device.index or 0)

        for data_dir in self.data_dirs:
            if not data_dir.is_dir():
                raise IOError(f"Error, data directory {data_dir} does not exist")
        if self.stats_dir is not None and not self.stats_dir.is_dir():
            raise IOError(f"Error, stats directory {self.stats_dir} does not exist")

        if self.interpolation_type:
            valid_interpolation = [
                "INTERP_NN",
                "INTERP_LINEAR",
                "INTERP_CUBIC",
                "INTERP_LANCZOS3",
                "INTERP_TRIANGULAR",
                "INTERP_GAUSSIAN",
            ]
            if self.interpolation_type not in valid_interpolation:
                raise ValueError(
                    f"Interpolation type {self.interpolation_type} not supported"
                )
            self.interpolation_type = getattr(dali.types, self.interpolation_type)

        self.layout = "FCHW"
        self.output_keys = ["invar", "outvar", "lead_time"]  # Added lead_time
        if self.use_cos_zenith:
            if not self.latlon_resolution:
                raise ValueError("latlon_resolution must be set for cos zenith angle")
            self.data_latlon = np.stack(
                latlon_grid(bounds=self.latlon_bounds, shape=self.latlon_resolution),
                axis=0,
            )
            self.latlon_dali = dali.types.Constant(self.data_latlon)
            self.output_keys += ["cos_zenith"]
        if self.use_time_of_year_index:
            self.output_keys += ["time_of_year_idx"]

        self.parse_dataset_files()
        self.load_statistics()
        self.pipe = self._create_pipeline()

    def parse_dataset_files(self) -> None:
        file_sets = [set(p.name for p in d.glob("????.h5")) for d in self.data_dirs]
        common_filenames = sorted(list(set.intersection(*file_sets)))

        if not common_filenames:
            raise FileNotFoundError(
                "No common year HDF5 files found across all data_dirs."
            )

        total_len = len(common_filenames)
        train_len = int(self.ratio * total_len)
        if self.mode == "train":
            self.year_filenames = common_filenames[:train_len]
        else:
            self.year_filenames = common_filenames[train_len:]
        print(self.year_filenames)
        self.n_years = len(self.year_filenames)
        self.logger.info(f"Found {self.n_years} common years across all data sources.")

        self.data_paths = [
            [d / fname for fname in self.year_filenames] for d in self.data_dirs
        ]

        samples_in_first_year = []
        first_year_paths = [paths[0] for paths in self.data_paths]
        with h5py.File(first_year_paths[0], "r") as f:
            samples_in_first_year.append(f["fields"].shape[0])
            self.img_shape = f["fields"].shape[2:]  # From ERA5 (4D)
            num_channels_available = f["fields"].shape[1]

        min_samples_per_year = min(samples_in_first_year)
        self.logger.info(
            f"Found minimum of {min_samples_per_year} samples across all sources for the first year."
        )

        if self.num_samples_per_year is None:
            self.num_samples_per_year = min_samples_per_year
        elif self.num_samples_per_year > min_samples_per_year:
            self.logger.warning(
                f"Requested num_samples_per_year ({self.num_samples_per_year}) is greater "
                f"than the minimum available ({min_samples_per_year}). Using the smaller, safer value."
            )
            self.num_samples_per_year = min_samples_per_year

        if self.patch_size is not None:
            if self.use_cos_zenith:
                raise ValueError("Patching is not supported with cos zenith angle")
            self.img_shape = [
                s - s % self.patch_size[i] for i, s in enumerate(self.img_shape)
            ]

        if (
            max(self.in_channels) >= num_channels_available
            or max(self.out_channels) >= num_channels_available
        ):
            raise ValueError(
                f"Channel index out of bounds. Available channels: {num_channels_available}"
            )

        # Effective samples per year is reduced by max offset and max lead time
        effective_samples_per_year = (
            self.num_samples_per_year
            - self.max_lead_time
            - max(self.model_time_offsets)
        )
        if effective_samples_per_year <= 0:
            raise ValueError(
                "`max_lead_time` and `model_time_offsets` are too large for `num_samples_per_year`"
            )
        self.total_length = self.n_years * effective_samples_per_year
        self.length = self.total_length
        self.logger.info(f"Using {self.num_samples_per_year} samples per year.")
        self.logger.info(f"Input image shape: {self.img_shape}")

    def load_statistics(self) -> None:
        if self.stats_dir is None:
            self.mu = None
            self.std = None
            return
        mean_stat_file = self.stats_dir / Path("global_means.npy")
        std_stat_file = self.stats_dir / Path("global_stds.npy")

        if not mean_stat_file.exists():
            raise IOError(f"Mean statistics file {mean_stat_file} not found")
        if not std_stat_file.exists():
            raise IOError(f"Std statistics file {std_stat_file} not found")

        mu_sort = np.load(str(mean_stat_file))[0, self.out_channels]
        sd_sort = np.load(str(std_stat_file))[0, self.out_channels]
        self.mu = mu_sort.copy()
        self.sd = sd_sort.copy()
        for idx, o_c in enumerate(self.orig_index):
            self.mu[o_c, :, :] = mu_sort[idx, :, :]
            self.sd[o_c, :, :] = sd_sort[idx, :, :]
        if not self.mu.shape == self.sd.shape == (len(self.out_channels), 1, 1):
            raise AssertionError("Error, normalisation arrays have wrong shape")

    def _create_pipeline(self) -> dali.Pipeline:
        """Constructs the DALI pipeline."""
        pipe = dali.Pipeline(
            batch_size=self.batch_size,
            num_threads=2,
            prefetch_queue_depth=2,
            py_num_workers=self.num_workers,
            device_id=self.device.index,
            py_start_method="spawn",
        )

        with pipe:
            # External source operator reads data
            source = MoWE5DaliExternalSource(
                data_paths=self.data_paths,
                year_filenames=self.year_filenames,
                num_samples=self.total_length,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                orig_index=self.orig_index,
                model_time_offsets=self.model_time_offsets,
                n_models=self.n_models,
                num_samples_per_year=self.num_samples_per_year,
                max_lead_time=self.max_lead_time,
                shuffle_model_idx=self.shuffle_model_idx,
                shuffle_channel_order=self.shuffle_channel_order,
                use_cos_zenith=self.use_cos_zenith,
                cos_zenith_args=self.cos_zenith_args,
                use_time_of_year_index=self.use_time_of_year_index,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                process_rank=self.process_rank,
                world_size=self.world_size,
                mode=self.mode,
            )
            # The length of the dataloader is the number of samples in this shard
            self.length = len(source) // self.batch_size

            num_outputs = (
                self.n_models + 4
            )  # models, outvar, timestamp, time_idx, lead_time
            raw_outputs = dali.fn.external_source(
                source, num_outputs=num_outputs, parallel=True, batch=False
            )

            model_inputs_raw = raw_outputs[: self.n_models]
            outvar_raw, timestamps, time_of_year_idx, lead_time = raw_outputs[
                self.n_models :
            ]

            # Move data to GPU if specified
            if self.device.type == "cuda":
                model_inputs_raw = [m.gpu() for m in model_inputs_raw]
                outvar_raw = outvar_raw.gpu()
                lead_time = lead_time.gpu()

            h, w = self.img_shape

            # Process each model's input tensor
            model_inputs_processed = []
            for model_in in model_inputs_raw:
                processed = model_in[:, :h, :w]
                if self.stats_dir is not None:
                    processed = dali.fn.normalize(
                        processed, mean=self.mu, stddev=self.sd
                    )
                if self.interpolation_type is not None:
                    processed = dali.fn.resize(
                        processed,
                        resize_x=w,
                        resize_y=h,
                        interp_type=self.interpolation_type,
                        antialias=False,
                    )
                model_inputs_processed.append(processed)

            # Stack model inputs into a single tensor: [N_MODELS, C, H, W]
            invar = dali.fn.stack(*model_inputs_processed, axis=0)
            invar.layout = "FCHW"

            # Process the ground truth tensor(ERA5) and crop to same shape as the model (720, 1440)
            outvar = outvar_raw[:, :h, :w]
            if self.stats_dir is not None:
                outvar = dali.fn.normalize(outvar, mean=self.mu, stddev=self.sd)
            if self.interpolation_type is not None:
                outvar = dali.fn.resize(
                    outvar,
                    resize_x=w,
                    resize_y=h,
                    interp_type=self.interpolation_type,
                    antialias=False,
                )

            # Assemble the final list of outputs
            outputs = [invar, outvar, lead_time]
            if self.use_cos_zenith:
                cos_zenith = dali.fn.cast(
                    cos_zenith_angle(timestamps, latlon=self.latlon_dali),
                    dtype=dali.types.FLOAT,
                )
                if self.device.type == "cuda":
                    cos_zenith = cos_zenith.gpu()
                outputs.append(cos_zenith)
            if self.use_time_of_year_index:
                outputs.append(time_of_year_idx)

            pipe.set_outputs(*outputs)
        return pipe

    def __iter__(self):
        """Returns an iterator for the DALI pipeline."""
        self.pipe.reset()
        return dali_pth.DALIGenericIterator([self.pipe], self.output_keys)

    def __len__(self):
        """Returns the number of batches in the dataloader."""
        return self.length


class MoWE5DaliExternalSource:
    """
    DALI External Source for loading multi-step model forecasts against ERA5 data.
    """

    def __init__(
        self,
        data_paths: list[list[str]],
        year_filenames: list[str],
        num_samples: int,
        in_channels: list[int],
        out_channels: list[int],
        orig_index: list[int],
        model_time_offsets: list[int],
        n_models: int,
        num_samples_per_year: int,
        max_lead_time: int,
        shuffle_model_idx: Union[int, None],
        shuffle_channel_order: Union[List[int], None],
        use_cos_zenith: bool,
        cos_zenith_args: dict,
        use_time_of_year_index: bool,
        batch_size: int = 1,
        shuffle: bool = True,
        process_rank: int = 0,
        world_size: int = 1,
        mode: str = "train",
    ):
        self.data_paths = data_paths
        self.year_filenames = year_filenames
        self.num_samples = num_samples
        self.in_chans = in_channels
        self.out_chans = out_channels
        self.orig_index = orig_index
        self.model_time_offsets = model_time_offsets
        self.n_models = n_models
        self.num_samples_per_year = num_samples_per_year
        self.max_lead_time = max_lead_time
        self.shuffle_model_idx = shuffle_model_idx
        self.shuffle_channel_order = shuffle_channel_order
        self.use_cos_zenith = use_cos_zenith
        self.cos_zenith_args = cos_zenith_args
        self.use_time_of_year_index = use_time_of_year_index
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.data_files = None
        self.current_year_idx = -1

        self.rng = np.random.default_rng(seed=43)
        max_offset = max(self.model_time_offsets) if self.model_time_offsets else 0

        # --- Pre-compute valid sample indices ---
        n_years = len(self.year_filenames)
        total_samples_in_split = n_years * self.num_samples_per_year
        all_potential_indices = np.arange(total_samples_in_split)

        # Create a mask to filter out indices that don't have enough history for offsets
        # or enough future data for the maximum lead time.
        per_year_indices = all_potential_indices % self.num_samples_per_year
        valid_indices_mask = (per_year_indices >= max_offset) & (
            per_year_indices < self.num_samples_per_year - self.max_lead_time
        )
        all_indices = all_potential_indices[valid_indices_mask]

        # Distribute indices among processes for DDP
        if world_size > 1:
            self.indices = np.array_split(all_indices, world_size)[process_rank]
        else:
            self.indices = all_indices

        self.num_batches = len(self.indices) // self.batch_size
        self.last_epoch = None

    def __call__(self, sample_info: dali.types.SampleInfo) -> tuple[np.ndarray, ...]:
        """Provides a single data sample to the DALI pipeline.

        This method is called by DALI for each sample. It determines the file and
        time index, loads the data, and returns it as a tuple.

        Args:
            sample_info (dali.types.SampleInfo): Information about the current
                sample being requested by DALI.

        Returns:
            tuple[np.ndarray, ...]: A tuple containing the model inputs, ground
                truth, and any additional metadata (timestamps, etc.).
        """
        if sample_info.iteration >= self.num_batches:
            raise StopIteration()

        # Shuffle indices at the beginning of each epoch
        if self.shuffle and sample_info.epoch_idx != self.last_epoch:
            np.random.default_rng(seed=sample_info.epoch_idx).shuffle(self.indices)
            self.last_epoch = sample_info.epoch_idx

        # Get the global index for the current sample
        target_idx = self.indices[sample_info.idx_in_epoch]
        target_year_idx = target_idx // self.num_samples_per_year
        target_in_idx = target_idx % self.num_samples_per_year

        # Randomly select a lead time for this sample
        if self.mode == "train":
            lead_time = np.random.randint(1, self.max_lead_time + 1)
        else:
            rng = np.random.default_rng(seed=target_idx)
            lead_time = rng.integers(1, self.max_lead_time + 1)
            # lead_time = (target_in_idx % self.max_lead_time) + 1

        # Open HDF5 files for the correct year if we've moved to a new one
        if target_year_idx != self.current_year_idx:
            if self.data_files:
                for source_files in self.data_files:
                    handle = source_files[0]
                    handle.close()

            year_paths = [paths[target_year_idx] for paths in self.data_paths]
            self.data_files = [[h5py.File(path, "r")] for path in year_paths]
            self.current_year_idx = target_year_idx

        timestamp_out = np.array([])
        if self.use_cos_zenith:
            dt_hours = self.cos_zenith_args.get("dt", 6.0)
            year_str = self.year_filenames[target_year_idx].split(".")[0]
            year = int(year_str)
            # Timestamp corresponds to the ground truth time
            truth_time_idx = target_in_idx + lead_time
            start_time = datetime(year, 1, 1, tzinfo=pytz.utc) + timedelta(
                hours=int(truth_time_idx) * dt_hours
            )
            timestamp_out = np.array([start_time.timestamp()])

        time_of_year_idx = (
            np.array([target_in_idx]) if self.use_time_of_year_index else np.array([-1])
        )

        # --- Load Ground Truth (ERA5) ---
        # Ground truth is at the initial condition time + lead time
        ground_truth_idx = target_in_idx + lead_time
        outvar = self.data_files[0][0]["fields"][
            ground_truth_idx,
            self.out_chans,
            :720,
        ]

        # Re-sort the channels to match the input channel order
        outvar_copy = outvar.copy()
        for idx, o_c in enumerate(self.orig_index):
            outvar_copy[o_c] = outvar[idx]
        outvar = outvar_copy

        # --- Load Model Inputs ---
        model_inputs = []
        for i in range(self.n_models):
            offset = self.model_time_offsets[i]
            # The model's initial condition time is offset from the target time
            model_in_idx = target_in_idx - offset

            # Model data is 5D: [initial_condition, lead_time, C, H, W]
            invar_i = self.data_files[i + 1][0]["fields"][
                model_in_idx,
                lead_time,
                self.in_chans,
                :720,
            ]

            # Apply channel shuffling augmentation if specified
            if i == self.shuffle_model_idx:
                invar_i = invar_i[self.shuffle_channel_order, :, :]

            model_inputs.append(invar_i)

        return (
            *model_inputs,
            outvar,
            timestamp_out,
            time_of_year_idx,
            np.array([lead_time], dtype=np.int32),
        )

    def __len__(self):
        """Returns the number of samples in this worker's shard."""
        return len(self.indices)

    def __del__(self):
        """Ensures HDF5 file handles are closed when the object is destroyed."""
        if self.data_files:
            for source_files in self.data_files:
                for handle in source_files:
                    handle.close()
