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

import hashlib
import json
import os
import threading
import warnings
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Literal

import numpy as np
import xarray as xr

from physicsnemo.utils.zenith_angle import zenith_azimuth_angles_from_timestamp

from .dataset import StormCastDataset

# reserved array name of the solar angles, which are computed rather than read
SOLAR_ANGLES = "solar_angles"
# in the order returned by zenith_azimuth_angles_from_timestamp
SOLAR_ANGLE_CHANNELS = ["sin_zenith", "cos_zenith", "sin_azimuth", "cos_azimuth"]


@dataclass
class AccessPattern:
    """Which channels and time steps to read from one data array.

    Parameters
    ----------
    channels : list[str] | None
        Names of the channels to read, or ``None`` to read all channels.
    time_steps : list[int]
        Time steps to read, relative to the sample time and in units of the time step,
        in the order they appear in the sample. For example, ``[0]`` reads only the
        sample time and ``[-6, -3, 0]`` reads the sample time and two earlier times.
        Must be non-empty and without duplicates.

    Attributes
    ----------
    channel_index : list[int] | slice
        Indices of ``channels`` along the channel dimension of the array, set by the
        dataset when indexing the files; ``slice(None)`` if all channels are read.
    """

    channels: list[str] | None
    time_steps: list[int]
    channel_index: list[int] | slice = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.time_steps = [int(step) for step in self.time_steps]
        if not self.time_steps:
            raise ValueError("time_steps must contain at least one time step.")
        if len(set(self.time_steps)) < len(self.time_steps):
            raise ValueError(
                f"time_steps must not contain duplicates, got {self.time_steps}."
            )


class _XarrayDataset(StormCastDataset):
    """Generic reader for regional training data stored in xarray-compatible files.

    Which arrays, channels and time steps make up the ``"state"`` and ``"background"``
    of each sample is configured with access patterns, so the same reader can serve
    e.g. downscaling, nowcasting and hybrid models. The class can also be used as a
    starting point for custom datasets, by subclassing or copying.

    Each data file must contain a ``"time"`` coordinate decodable to
    ``numpy.datetime64`` and one or more data arrays with dimensions
    ``("time", channel_dim, y_dim, x_dim)``, where ``channel_dim`` has a string
    coordinate with the channel names. Dimensions are identified by position, so only
    ``"time"`` must have that name. A sample at time ``t`` is built as follows: for
    each array in the access patterns, the selected channels are read at each time
    step in ``time_steps`` and normalized; the time and channel dimensions are then
    merged and the arrays concatenated along the channel axis, in the order of the
    access patterns. Channel names get a ``_t{step:+d}`` suffix, e.g. ``t2m_t+0``.

    Only samples whose every accessed time exists, is valid and lies within the time
    range are used. Files may contain an optional boolean ``"valid"`` variable along
    ``"time"`` to mark bad samples. The files may have different numbers of times and
    need not be in chronological order, but each data array must have the same
    channels, in the same order, and the same spatial shape in every file.

    Parameters
    ----------
    data_paths : str | list[str]
        Path(s) or URL(s) to the data files, in any format readable by
        ``xarray.open_dataset``.
    state_access_patterns : dict[str, dict]
        Maps data array names to ``AccessPattern`` arguments (``channels`` and
        ``time_steps``) defining the ``"state"`` (training target).
    background_access_patterns : dict[str, dict], optional, default {}
        As ``state_access_patterns``, but defining the ``"background"`` (conditioning).
        When empty, samples have no ``"background"``. The reserved array name
        ``"solar_angles"`` computes the sin/cos of the solar zenith and azimuth angles,
        with channels ``"sin_zenith"``, ``"cos_zenith"``, ``"sin_azimuth"`` and
        ``"cos_azimuth"``. ``lat_lon_variables`` must be set for ``"solar_angles"`` to
        work.
    invariants_path : str | None, optional, default None
        Path to a file with time-independent data: invariants, latitude/longitude and
        mask, each with dimensions ``(y_dim, x_dim)`` matching the data arrays. May be
        one of ``data_paths``.
    stats_path : str
        Path to a JSON file with normalization statistics, formatted as
        ``{array_name: {channel_name: {"mean": float, "std": float}}}``. Invariant
        statistics go under the ``"invariant"`` key.
    time_step_seconds : int | None, optional, default None
        Length of one time step in seconds. When ``None``, inferred as the smallest
        interval between the times in the data.
    invariant_variables : list[str], optional, default []
        Names of the variables in ``invariants_path`` to use as invariants.
    lat_lon_variables : tuple[str, str] | None, optional, default None
        Names of the latitude and longitude (degrees) variables in ``invariants_path``.
    mask_variable : str | None, optional, default None
        Name of a static mask variable in ``invariants_path``, nonzero where valid.
    nan_mask : bool, optional, default False
        If ``True``, mask out pixels where any state or background channel is
        non-finite, and replace non-finite values with 0.
    open_dataset_kwargs : dict[str, Any], optional, default {}
        Keyword arguments for ``xarray.open_dataset`` when opening the data files, e.g.
        ``{"engine": "zarr"}``. Not used for ``invariants_path``.
    time_range_train : tuple[str, str] | None, optional, default None
        Half-open interval ``[start, end)`` of ISO 8601 times (e.g.
        ``"2025-10-26T03:00:00"``) to use when ``train=True``. ``None`` uses all times.
    time_range_valid : tuple[str, str] | None, optional, default None
        As ``time_range_train``, but used when ``train=False``. Defaults to
        ``time_range_train``.
    exclude_times : list[str], optional, default []
        ISO 8601 times to exclude, e.g. known bad samples.
    crop_window : tuple[tuple[int, int], tuple[int, int]] | None, optional, default None
        ``((y_start, y_end), (x_start, x_end))`` crop applied to all spatial data.
    num_threads : int, optional, default 16
        Number of threads used to read and normalize each sample.
    cache_location : str | None, optional, default None
        Experimental. Cache chunks of Zarr data files read during training in this
        location. If cache_location is a local directory (e.g. on node-local disk),
        use disk-based caching. Alternatively, ``cache_location="memory"``. caches
        data in RAM for even faster access. Useful when the data storage is slow,
        e.g. a bucket store, or when the entire dataset fits in RAM. ``None`` disables
        caching. See the Notes for requirements and limitations.
    train : bool, optional, default True
        Selects ``time_range_train`` if ``True``, ``time_range_valid`` otherwise.

    Notes
    -----
    Caching relies on the experimental ``zarr.experimental.cache_store.CacheStore``.
    Local and URL (via ``fsspec``) Zarr stores are supported; for URLs, pass
    credentials etc. as ``storage_options`` in ``open_dataset_kwargs``. Other file
    formats are not cached.

    The cache is never evicted, so it must be large enough for all the data read on
    each node; otherwise the job fails when the disk or memory runs out. Limitations:

    - The ``"memory"`` cache is separate for each process, including each DataLoader
      worker. With ``training.num_data_workers > 1`` this multiplies memory use and
      lowers the hit rate, so prefer ``training.num_data_workers=1``, which still
      loads data concurrently with training, and parallelize reads with
      ``num_threads``.
    - A directory cache is shared by all processes using it, and is reused by later
      runs. It is not invalidated if the source data changes; clear it manually in
      that case. Use a node-local disk rather than a shared file system.
    - Byte-range reads, which include all reads from sharded Zarr v3 stores, are only
      cached in memory, even when ``cache_location`` is a directory.

    Examples
    --------
    A dataset config (e.g. ``config/dataset/my_data.yaml``) for a hybrid model that
    predicts three HRRR variables from their values one time step earlier and from
    ERA5 forcing at the target time, with ERA5 regridded to the HRRR grid:

    .. code-block:: yaml

        name: xarray.XarrayDataset
        data_paths: ["/data/hrrr_era5_2020.zarr", "/data/hrrr_era5_2021.zarr"]
        open_dataset_kwargs: {engine: zarr}
        invariants_path: /data/invariants.nc
        invariant_variables: [orography, land_sea_mask]
        lat_lon_variables: [lat, lon]
        stats_path: /data/stats.json
        state_access_patterns:
          hrrr: {channels: [t2m, u10, v10], time_steps: [1]}
        background_access_patterns:
          hrrr: {channels: [t2m, u10, v10], time_steps: [0]}
          era5: {channels: null, time_steps: [1]}
        time_range_train: ["2020-01-01T00:00:00", "2021-07-01T00:00:00"]
        time_range_valid: ["2021-07-01T00:00:00", "2022-01-01T00:00:00"]

    Time step 0 is the last past step, so the state channels are then ``t2m_t+1``,
    ``u10_t+1`` and ``v10_t+1``, and the background channels ``t2m_t+0``,
    ``u10_t+0`` and ``v10_t+0`` followed by all ERA5 channels with a ``_t+1`` suffix.
    """

    def __init__(
        self,
        *,
        data_paths: str | list[str],
        state_access_patterns: dict[str, dict[str, Any]],
        background_access_patterns: dict[str, dict[str, Any]] = {},
        invariants_path: str | None = None,
        stats_path: str,
        time_step_seconds: int | None = None,
        invariant_variables: list[str] = [],
        lat_lon_variables: tuple[str, str] | None = None,
        mask_variable: str | None = None,
        nan_mask: bool = False,
        open_dataset_kwargs: dict[str, Any] = {},
        time_range_train: tuple[str, str] | None = None,
        time_range_valid: tuple[str, str] | None = None,
        exclude_times: list[str] = [],
        crop_window: tuple[tuple[int, int], tuple[int, int]] | None = None,
        num_threads: int = 16,
        cache_location: str | None = None,
        train: bool = True,
    ) -> None:
        if not isinstance(data_paths, list):
            data_paths = [data_paths]
        if not data_paths:
            raise ValueError(
                f"Need at least one path in {data_paths}, but received none."
            )
        self.data_paths = data_paths
        self.state_access_patterns = {
            array: AccessPattern(**p) for (array, p) in state_access_patterns.items()
        }
        self.background_access_patterns = {
            array: AccessPattern(**p)
            for (array, p) in background_access_patterns.items()
        }
        self.invariants_path = invariants_path
        self.stats_path = stats_path
        self.time_step = (
            None
            if time_step_seconds is None
            else np.timedelta64(time_step_seconds, "s")
        )
        self.invariant_variables = invariant_variables
        self.lat_lon_variables = lat_lon_variables
        self.mask_variable = mask_variable
        self.nan_mask = nan_mask
        if SOLAR_ANGLES in self.state_access_patterns:
            raise ValueError(f"'{SOLAR_ANGLES}' can only be used in the background.")
        if (
            SOLAR_ANGLES in self.background_access_patterns
            and lat_lon_variables is None
        ):
            raise ValueError(f"lat_lon_variables must be set to use '{SOLAR_ANGLES}'.")
        self.open_dataset_kwargs = open_dataset_kwargs
        if time_range_valid is None:
            time_range_valid = time_range_train
        self.time_range = time_range_train if train else time_range_valid
        if self.time_range is not None:
            self.time_range = tuple(np.datetime64(t) for t in self.time_range)
        self.exclude_times = np.array(
            [np.datetime64(t) for t in exclude_times], dtype="datetime64[s]"
        )
        self.crop_slice = (
            (slice(None), slice(None))
            if crop_window is None
            else tuple(slice(*w) for w in crop_window)
        )
        self.num_threads = num_threads
        self.cache_location = cache_location
        if cache_location is not None:
            uncached = [p for p in data_paths if not _is_zarr_path(p)]
            if uncached:
                warnings.warn(
                    "Caching is only supported for Zarr stores; these paths will "
                    f"not be cached: {uncached}"
                )
        self.executor = None

        self._index_files()
        self._resolve_channels()
        self._select_times()
        self._load_invariants()
        self._load_normalization()

    def _index_files(self) -> None:
        """Map each valid time to its file and index, checking that each file contains
        all accessed arrays."""
        self.file_and_index_for_time = {}
        arrays = {array for (array, _) in self._file_access_patterns()}
        for path in self.data_paths:
            with xr.open_dataset(path, **self.open_dataset_kwargs) as ds:
                missing = arrays - set(ds.variables)
                if missing:
                    raise ValueError(f"Arrays {sorted(missing)} not found in '{path}'.")
                times = ds["time"].values
                idx = np.arange(len(times))
                if "valid" in ds.variables:
                    valid = ds["valid"].values.astype(bool)
                    times = times[valid]
                    idx = idx[valid]
                if len(np.unique(times)) < len(times):
                    warnings.warn(
                        f"'{path}' contains duplicate time stamps. Only the last "
                        "occurrence of each will be used."
                    )
                duplicates = [t for t in times if t in self.file_and_index_for_time]
                if duplicates:
                    warnings.warn(
                        f"{len(duplicates)} time stamps in '{path}' were already found "
                        f"in earlier files (first: {duplicates[0]}). The data from "
                        f"'{path}' will be used for these."
                    )
                self.file_and_index_for_time.update(
                    {t: (path, k) for (t, k) in zip(times, idx)}
                )

    def _resolve_channels(self) -> None:
        """Find the image shape and the channels read by each access pattern, checking
        that they exist. Channels are taken from the first file, as all files must
        have the same channels."""
        self.used_channels = {}
        with xr.open_dataset(self.data_paths[0], **self.open_dataset_kwargs) as ds:
            for array, pattern in self._file_access_patterns():
                da = ds[array]
                self.img_shape = da[..., *self.crop_slice].shape[-2:]
                channels = [str(ch) for ch in da[da.dims[-3]].values]
                _resolve_channel_index(
                    pattern, channels, f"'{array}' in '{self.data_paths[0]}'"
                )
                self.used_channels.setdefault(array, {}).update(
                    dict.fromkeys(pattern.channels)
                )
        if (pattern := self.background_access_patterns.get(SOLAR_ANGLES)) is not None:
            _resolve_channel_index(pattern, SOLAR_ANGLE_CHANNELS, f"'{SOLAR_ANGLES}'")

    def _select_times(self) -> None:
        """Select the sample times: those in the time range, not excluded, and for
        which all times read by the access patterns are available."""
        self.times = np.array(sorted(self.file_and_index_for_time))
        if self.time_step is None:
            self.time_step = (self.times[1:] - self.times[:-1]).min()

        if self.time_range is not None:
            in_range = (self.times >= self.time_range[0]) & (
                self.times < self.time_range[1]
            )
            self.times = self.times[in_range]
        self.times = self.times[~np.isin(self.times, self.exclude_times)]
        keep_times = np.ones(len(self.times), dtype=bool)
        for _, pattern in self._file_access_patterns():
            for step in pattern.time_steps:
                keep_times &= np.isin(self.times + step * self.time_step, self.times)
        self.times = self.times[keep_times]
        if len(self.times) == 0:
            raise ValueError(
                "No samples left after selecting times. Check the time range, "
                "exclude_times, time_step_seconds and the time_steps of the access "
                "patterns."
            )

    def _load_invariants(self) -> None:
        """Load the invariants, latitude/longitude and mask, if configured."""
        self.invariants = self.lat = self.lon = self.mask = None
        if self.invariants_path is None:
            if self.invariant_variables or self.lat_lon_variables or self.mask_variable:
                raise ValueError(
                    "invariants_path must be set to use invariant_variables, "
                    "lat_lon_variables or mask_variable."
                )
            return

        with xr.open_dataset(self.invariants_path) as ds:

            def read(name: str, dtype: type = np.float32) -> np.ndarray:
                return ds[name][..., *self.crop_slice].astype(dtype).values

            if self.lat_lon_variables is not None:
                (self.lat, self.lon) = (read(v) for v in self.lat_lon_variables)
            if self.mask_variable is not None:
                self.mask = read(self.mask_variable, bool)
                if self.mask.ndim == 2:
                    self.mask = self.mask[None, :, :]
            if self.invariant_variables:
                self.invariants = np.stack([read(v) for v in self.invariant_variables])

    def _load_normalization(self) -> None:
        """Load the normalization statistics, normalize the invariants, and build the
        channel names and statistics of the collated samples."""
        with open(self.stats_path, "r") as f:
            stats = json.load(f)
        self.mean = {}
        self.std = {}
        for array, channels in self.used_channels.items():
            (self.mean[array], self.std[array]) = _load_stats(
                stats, list(channels), array
            )
        if self.invariants is not None:
            (self.mean["invariant"], self.std["invariant"]) = _load_stats(
                stats, self.invariant_variables, "invariant"
            )
            for x, v in zip(self.invariants, self.invariant_variables):
                self.in_place_normalize(x, "invariant", v)
        # solar angles are already in [-1, 1], so they are not normalized; these
        # identity stats keep them unchanged in normalize/denormalize_background
        self.mean[SOLAR_ANGLES] = dict.fromkeys(SOLAR_ANGLE_CHANNELS, np.float32(0.0))
        self.std[SOLAR_ANGLES] = dict.fromkeys(SOLAR_ANGLE_CHANNELS, np.float32(1.0))

        (self.state_variables, self.state_mean, self.state_std) = self._collated_layout(
            self.state_access_patterns
        )
        (self.background_variables, self.background_mean, self.background_std) = (
            self._collated_layout(self.background_access_patterns)
        )

    def _collated_layout(
        self, access_patterns: dict[str, AccessPattern]
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Channel names and per-channel mean and std, of shape :math:`(C, 1, 1)`, for
        the collated sample layout of ``access_patterns``."""
        # must match the (array, time step, channel) ordering used in __getitem__
        names = []
        mean = []
        std = []
        for array, pattern in access_patterns.items():
            for step in pattern.time_steps:
                for ch in pattern.channels:
                    names.append(f"{ch}_t{step:+d}")
                    mean.append(self.mean[array][ch])
                    std.append(self.std[array][ch])
        mean = np.array(mean, dtype=np.float32)[:, None, None]
        std = np.array(std, dtype=np.float32)[:, None, None]
        return (names, mean, std)

    def _file_access_patterns(self) -> Iterator[tuple[str, AccessPattern]]:
        """``(array, pattern)`` pairs of the state and then the background access
        patterns, except the computed solar angles."""
        return (
            (array, pattern)
            for (array, pattern) in chain(
                self.state_access_patterns.items(),
                self.background_access_patterns.items(),
            )
            if array != SOLAR_ANGLES
        )

    def _open_dataset(self, path: str) -> xr.Dataset:
        """Open a data file, wrapping Zarr stores in a chunk cache if caching is
        enabled."""
        if self.cache_location is None or not _is_zarr_path(path):
            return xr.open_dataset(path, **self.open_dataset_kwargs)

        from zarr.experimental.cache_store import CacheStore
        from zarr.storage import FsspecStore, LocalStore, MemoryStore

        kwargs = self.open_dataset_kwargs.copy()
        # zarr rejects storage_options when given a store object instead of a URL
        storage_options = kwargs.pop("storage_options", None)
        if _is_url(path):
            source_store = FsspecStore.from_url(
                path, storage_options=storage_options, read_only=True
            )
        else:
            source_store = LocalStore(path, read_only=True)

        if self.cache_location == "memory":
            if _num_dataloader_workers() > 1:
                warnings.warn(
                    "The in-memory sample cache is not shared between DataLoader "
                    "worker processes, which multiplies memory use and lowers the hit "
                    "rate. Consider training.num_data_workers=1; the dataset already "
                    "reads in parallel using dataset.num_threads."
                )
            cache_store = MemoryStore()
        else:
            cache_store = LocalStore(
                os.path.join(self.cache_location, _cache_dir_name(path))
            )

        store = CacheStore(source_store, cache_store=cache_store)
        # engine="zarr" is required: guess_can_open() only recognizes str/PathLike
        kwargs["engine"] = "zarr"
        return xr.open_dataset(store, **kwargs)

    def _get_dataset(self, path: str) -> xr.Dataset:
        """Return the open dataset for ``path``, opening it on first use."""
        if (ds := self._datasets.get(path)) is None:
            # one handle per file, so that all threads share the same in-memory cache
            with self._open_lock:
                if (ds := self._datasets.get(path)) is None:
                    ds = self._open_dataset(path)
                    self._datasets[path] = ds
        return ds

    def _get_data_for_time(
        self,
        array: str,
        time: np.datetime64,
        channel_index: list[int] | slice,
        buffer: np.ndarray,
    ) -> None:
        """Read the channels ``channel_index`` of ``array`` at ``time``, cropped to the
        crop window, into ``buffer``, or compute them if ``array`` is the solar
        angles."""
        if array == SOLAR_ANGLES:
            timestamp = (time - np.datetime64(0, "s")) / np.timedelta64(1, "s")
            angles = zenith_azimuth_angles_from_timestamp(timestamp, self.lon, self.lat)
            buffer[:] = np.stack(angles)[channel_index]
            return

        (dataset_path, time_idx) = self.file_and_index_for_time[time]
        da = self._get_dataset(dataset_path)[array]
        (time_dim, channel_dim, y_dim, x_dim) = da.dims
        index = {
            time_dim: time_idx,
            channel_dim: channel_index,
            y_dim: self.crop_slice[0],
            x_dim: self.crop_slice[1],
        }
        buffer[:] = da.isel(index).values

    def _submit_reads(
        self, time: np.datetime64, access_patterns: dict[str, AccessPattern]
    ) -> tuple[np.ndarray, dict[Future, tuple[str, list[str], np.ndarray]]]:
        """Allocate the collated array of ``access_patterns`` at ``time`` and submit
        a read of each (array, time step) into its block of channels.

        Returns the array, which is filled once the reads are done, and a dict
        mapping each read task to its ``(array, channels, block)``."""
        num_channels = sum(
            len(p.channels) * len(p.time_steps) for p in access_patterns.values()
        )
        out = np.empty((num_channels, *self.img_shape), dtype=np.float32)
        reads = {}
        offset = 0
        # the order must match _collated_layout
        for array, pattern in access_patterns.items():
            for step in pattern.time_steps:
                block = out[offset : offset + len(pattern.channels)]
                task = self.executor.submit(
                    self._get_data_for_time,
                    array,
                    time + step * self.time_step,
                    pattern.channel_index,
                    block,
                )
                reads[task] = (array, pattern.channels, block)
                offset += len(pattern.channels)
        return (out, reads)

    def __getitem__(
        self, idx: int
    ) -> dict[Literal["background", "state", "mask"], np.ndarray]:
        """Get the sample at an index.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        dict[str, np.ndarray]
            Normalized ``"state"`` of shape :math:`(C_{state}, H, W)`, and if
            configured, ``"background"`` of shape :math:`(C_{background}, H, W)` and
            boolean ``"mask"`` of shape :math:`(1, H, W)`, ``True`` where valid.
        """
        if self.executor is None:
            self._datasets = {}
            self._open_lock = threading.Lock()
            self.executor = ThreadPoolExecutor(max_workers=self.num_threads)

        time = self.times[idx]
        sample = {}
        (sample["state"], reads) = self._submit_reads(time, self.state_access_patterns)
        if self.background_access_patterns:
            (sample["background"], background_reads) = self._submit_reads(
                time, self.background_access_patterns
            )
            reads.update(background_reads)

        # normalize each block in parallel by channel, as soon as its read is done
        normalize_tasks = []
        for read in as_completed(reads):
            read.result()
            (array, channels, block) = reads[read]
            if array == SOLAR_ANGLES:  # solar angles are not normalized
                continue
            for j, ch in enumerate(channels):
                normalize_tasks.append(
                    self.executor.submit(self.in_place_normalize, block[j], array, ch)
                )
        for task in normalize_tasks:
            task.result()

        mask = self.mask
        if self.nan_mask:
            finite = np.ones((1, *self.img_shape), dtype=bool)
            for x in sample.values():
                finite &= np.isfinite(x).all(axis=0, keepdims=True)
                np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            mask = finite if mask is None else mask & finite
        if mask is not None:
            sample["mask"] = mask

        return sample

    def __len__(self) -> int:
        """Number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.times)

    def in_place_normalize(self, x: np.ndarray, array: str, var: str) -> None:
        """Normalize one channel from physical units, in place.

        This default implementation performs shift-scale normalization with mean and
        standard deviation from the statistics loaded from ``stats_path``. Override
        this function to implement e.g. nonlinear normalization.

        Parameters
        ----------
        x : np.ndarray
            Channel of shape :math:`(H, W)` in physical units, normalized in place.
        array : str
            Name of the data array of the channel, or ``"invariant"`` for invariants.
            Not called for the solar angles, which are not normalized.
        var : str
            Name of the channel or invariant variable.

        Returns
        -------
        None
        """
        inv_std = 1.0 / self.std[array][var]
        x -= self.mean[array][var]
        x *= inv_std

    def longitude(self) -> np.ndarray:
        """Longitude of each grid point.

        Returns
        -------
        np.ndarray
            Array of shape :math:`(H, W)`, all NaN if ``lat_lon_variables`` is not set.
        """
        return np.full(self.img_shape, np.nan) if self.lon is None else self.lon

    def latitude(self) -> np.ndarray:
        """Latitude of each grid point.

        Returns
        -------
        np.ndarray
            Array of shape :math:`(H, W)`, all NaN if ``lat_lon_variables`` is not set.
        """
        return np.full(self.img_shape, np.nan) if self.lat is None else self.lat

    def background_channels(self) -> list[str]:
        """Names of the background channels.

        Returns
        -------
        list[str]
            One name per background channel, in sample order.
        """
        return list(self.background_variables)

    def state_channels(self) -> list[str]:
        """Names of the state channels.

        Returns
        -------
        list[str]
            One name per state channel, in sample order.
        """
        return list(self.state_variables)

    def get_invariants(self) -> np.ndarray | None:
        """Normalized invariants.

        Returns
        -------
        np.ndarray | None
            Array of shape :math:`(C_{invariant}, H, W)`, or ``None`` if no invariants
            are used.
        """
        return self.invariants

    def image_shape(self) -> tuple[int, int]:
        """Spatial shape of the data after cropping.

        Returns
        -------
        tuple[int, int]
            Height and width.
        """
        return self.img_shape

    def normalize_background(self, x: np.ndarray) -> np.ndarray:
        """Normalize background data from physical units.

        Parameters
        ----------
        x : np.ndarray
            Background of shape :math:`(C_{background}, H, W)` or
            :math:`(B, C_{background}, H, W)`, in physical units.

        Returns
        -------
        np.ndarray
            Normalized background, same shape as ``x``.
        """
        return (x - self.background_mean) / self.background_std

    def denormalize_background(self, x: np.ndarray) -> np.ndarray:
        """Convert normalized background data to physical units.

        Parameters
        ----------
        x : np.ndarray
            Normalized background of shape :math:`(C_{background}, H, W)` or
            :math:`(B, C_{background}, H, W)`.

        Returns
        -------
        np.ndarray
            Background in physical units, same shape as ``x``.
        """
        return x * self.background_std + self.background_mean

    def normalize_state(self, x: np.ndarray) -> np.ndarray:
        """Normalize state data from physical units.

        Parameters
        ----------
        x : np.ndarray
            State of shape :math:`(C_{state}, H, W)` or :math:`(B, C_{state}, H, W)`,
            in physical units.

        Returns
        -------
        np.ndarray
            Normalized state, same shape as ``x``.
        """
        return (x - self.state_mean) / self.state_std

    def denormalize_state(self, x: np.ndarray) -> np.ndarray:
        """Convert normalized state data to physical units.

        Parameters
        ----------
        x : np.ndarray
            Normalized state of shape :math:`(C_{state}, H, W)` or
            :math:`(B, C_{state}, H, W)`.

        Returns
        -------
        np.ndarray
            State in physical units, same shape as ``x``.
        """
        return x * self.state_std + self.state_mean


def _resolve_channel_index(
    pattern: AccessPattern, available: list[str], source: str
) -> None:
    """Set ``pattern.channel_index`` from the ``available`` channels of ``source``,
    and ``pattern.channels`` to all of them if it is ``None``."""
    if pattern.channels is None:
        pattern.channels = list(available)
        pattern.channel_index = slice(None)
        return
    unknown = set(pattern.channels) - set(available)
    if unknown:
        raise ValueError(
            f"Channels {unknown} of {source} not found. Available channels: "
            f"{list(available)}"
        )
    pattern.channel_index = [available.index(ch) for ch in pattern.channels]


def _is_url(path: str) -> bool:
    """Whether ``path`` is a URL (e.g. ``s3://bucket/data.zarr``), not a local path."""
    return "://" in path


def _is_zarr_path(path: str) -> bool:
    """Whether ``path`` points to a Zarr store, judging by its suffix."""
    return path.rstrip("/").endswith(".zarr")


def _cache_dir_name(path: str) -> str:
    """Cache directory name unique to ``path``, including the store name for
    readability."""
    path = (
        path.rstrip("/") if _is_url(path) else os.path.abspath(os.path.normpath(path))
    )
    path_hash = hashlib.sha256(path.encode()).hexdigest()[:16]
    return f"{os.path.basename(path)}-{path_hash}"


def _num_dataloader_workers() -> int:
    """Number of workers of the PyTorch DataLoader this process is a worker of, or 0."""
    import torch.utils.data

    worker_info = torch.utils.data.get_worker_info()
    return 0 if worker_info is None else worker_info.num_workers


def _load_stats(
    stats: dict, variables: list[str], array: str
) -> tuple[dict[str, np.float32], dict[str, np.float32]]:
    """Look up the mean and standard deviation of each variable of an array.

    Parameters
    ----------
    stats : dict
        Statistics loaded from the stats JSON file.
    variables : list[str]
        Variable (channel) names to look up.
    array : str
        Array name, i.e. the top-level key in ``stats``.

    Returns
    -------
    tuple[dict[str, np.float32], dict[str, np.float32]]
        Means and standard deviations keyed by variable name.
    """
    missing = [v for v in variables if v not in stats.get(array, {})]
    if missing:
        raise ValueError(
            f"Normalization statistics for '{array}' are missing for: {missing}"
        )
    mean = {str(v): np.float32(stats[array][v]["mean"]) for v in variables}
    std = {str(v): np.float32(stats[array][v]["std"]) for v in variables}
    return (mean, std)


class XarrayDataset(_XarrayDataset):
    """:class:`_XarrayDataset` with the ``(params, train)`` constructor expected by
    the training recipe; ``params`` holds the keyword arguments of
    :class:`_XarrayDataset`."""

    def __init__(self, params: dict[str, Any], train: bool) -> None:
        super().__init__(train=train, **params)


def write_example_files(
    directory: str,
    arrays: dict[str, list[str]] = {
        "hrrr": ["t2m", "u10", "v10"],
        "era5": ["msl", "tcwv"],
    },
    num_files: int = 2,
    times_per_file: int = 24,
    start_time: str | np.datetime64 = "2020-01-01T00:00:00",
    time_step: np.timedelta64 = np.timedelta64(1, "h"),
    image_shape: tuple[int, int] = (32, 32),
    invariant_variables: list[str] = ["orography", "land_sea_mask"],
    file_format: Literal["zarr", "netcdf"] = "zarr",
    seed: int = 0,
) -> dict[str, Any]:
    r"""Write data files filled with random data in the format read by
    :class:`XarrayDataset`, along with the stats file and, optionally, the
    invariants file.

    The files serve as a template for formatting your own data, and as temporary
    data for tests. The data files are consecutive in time and contain one array
    per entry of ``arrays``, with dimensions ``("time", "<array>_channel", "y",
    "x")``, plus an all-true ``"valid"`` variable. Each channel has random values
    around its own mean and scale; ``stats.json`` holds the statistics computed
    from the data. If ``invariant_variables`` is non-empty, an invariants file is
    written with these variables and ``"lat"``, ``"lon"`` and ``"mask"``, each with
    dimensions ``("y", "x")``.

    Parameters
    ----------
    directory : str
        Directory to write the files to, created if needed. Existing files with the
        same names are overwritten.
    arrays : dict[str, list[str]], optional
        Maps array names to channel names. Defaults to a ``"hrrr"`` array with 3
        channels and an ``"era5"`` array with 2 channels.
    num_files : int, optional, default 2
        Number of data files.
    times_per_file : int, optional, default 24
        Number of time steps in each data file.
    start_time : str | np.datetime64, optional, default "2020-01-01T00:00:00"
        First time in the data.
    time_step : np.timedelta64, optional, default np.timedelta64(1, "h")
        Interval between consecutive times, a whole number of seconds.
    image_shape : tuple[int, int], optional, default (32, 32)
        Height and width of the data.
    invariant_variables : list[str], optional, default ["orography", "land_sea_mask"]
        Names of the invariants to write. If empty, no invariants file is written,
        so there is also no latitude, longitude or mask.
    file_format : {"zarr", "netcdf"}, optional, default "zarr"
        Format of the data and invariants files. Data arrays are chunked by time
        step, with all channels in one chunk. NetCDF files are written as NetCDF4,
        which requires the ``netCDF4`` or ``h5netcdf`` package.
    seed : int, optional, default 0
        Seed of the random number generator.

    Returns
    -------
    dict[str, Any]
        Keyword arguments of :class:`_XarrayDataset` describing the files:
        ``data_paths``, ``stats_path`` and ``time_step_seconds``, plus
        ``invariants_path``, ``invariant_variables``, ``lat_lon_variables`` and
        ``mask_variable`` if an invariants file is written.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp_dir:
    ...     files = write_example_files(tmp_dir)
    ...     dataset = _XarrayDataset(
    ...         state_access_patterns={"hrrr": {"channels": None, "time_steps": [1]}},
    ...         background_access_patterns={
    ...             "hrrr": {"channels": None, "time_steps": [0]},
    ...             "era5": {"channels": ["msl"], "time_steps": [1]},
    ...         },
    ...         **files,
    ...     )
    ...     sample = dataset[0]
    >>> len(dataset), sample["state"].shape, sample["background"].shape
    (47, (3, 32, 32), (4, 32, 32))
    """
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(seed)
    suffix = ".zarr" if file_format == "zarr" else ".nc"
    (height, width) = image_shape

    def write(ds: xr.Dataset, path: str, chunks: dict[str, tuple] = {}) -> None:
        if file_format == "zarr":
            encoding = {var: {"chunks": c} for (var, c) in chunks.items()}
            with warnings.catch_warnings():
                # consolidated metadata is not yet part of the Zarr v3 specification
                warnings.filterwarnings("ignore", message="Consolidated metadata")
                ds.to_zarr(path, mode="w", encoding=encoding)
        else:
            encoding = {var: {"chunksizes": c} for (var, c) in chunks.items()}
            ds.to_netcdf(path, format="NETCDF4", encoding=encoding)

    # random data around a different mean and scale for each channel
    num_times = num_files * times_per_file
    times = np.datetime64(start_time, "s") + time_step * np.arange(num_times)
    data = {}
    for array, channels in arrays.items():
        mean = rng.uniform(-10.0, 10.0, size=(1, len(channels), 1, 1))
        std = rng.uniform(0.5, 5.0, size=(1, len(channels), 1, 1))
        noise = rng.standard_normal((num_times, len(channels), height, width))
        data[array] = (mean + std * noise).astype(np.float32)

    data_paths = []
    for i in range(num_files):
        file_times = slice(i * times_per_file, (i + 1) * times_per_file)
        ds = xr.Dataset(
            {
                array: (("time", f"{array}_channel", "y", "x"), data[array][file_times])
                for array in arrays
            },
            coords={
                "time": times[file_times],
                **{f"{array}_channel": ch for (array, ch) in arrays.items()},
            },
        )
        ds["valid"] = ("time", np.ones(times_per_file, dtype=bool))
        path = os.path.join(directory, f"data_{i:03d}{suffix}")
        # one chunk per time step with all channels, so a read needs one request
        chunks = {array: (1, len(ch), height, width) for (array, ch) in arrays.items()}
        write(ds, path, chunks)
        data_paths.append(path)
    files = {
        "data_paths": data_paths,
        "time_step_seconds": int(time_step // np.timedelta64(1, "s")),
    }

    stats = {
        array: {
            ch: {
                "mean": float(data[array][:, j].mean()),
                "std": float(data[array][:, j].std()),
            }
            for (j, ch) in enumerate(channels)
        }
        for (array, channels) in arrays.items()
    }

    if invariant_variables:
        (lat, lon) = np.meshgrid(
            np.linspace(30.0, 50.0, height, dtype=np.float32),
            np.linspace(-110.0, -80.0, width, dtype=np.float32),
            indexing="ij",
        )
        invariants = xr.Dataset(
            {
                "lat": (("y", "x"), lat),
                "lon": (("y", "x"), lon),
                "mask": (("y", "x"), rng.random(image_shape) > 0.1),
                **{
                    v: (("y", "x"), rng.standard_normal(image_shape, np.float32))
                    for v in invariant_variables
                },
            }
        )
        path = os.path.join(directory, f"invariants{suffix}")
        write(invariants, path)
        stats["invariant"] = {
            v: {"mean": float(invariants[v].mean()), "std": float(invariants[v].std())}
            for v in invariant_variables
        }
        files.update(
            invariants_path=path,
            invariant_variables=list(invariant_variables),
            lat_lon_variables=("lat", "lon"),
            mask_variable="mask",
        )

    path = os.path.join(directory, "stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    files["stats_path"] = path

    return files
