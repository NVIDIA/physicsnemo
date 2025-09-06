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

import os
import json
from datetime import datetime, timedelta
from typing import Any, List
from importlib.metadata import version
import queue
import threading

import xarray as xr
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger

os.makedirs("outputs", exist_ok=True)

from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import pandas as pd

from physicsnemo.distributed import DistributedManager
from earth2studio.data import DataSource, fetch_data
from earth2studio.utils.type import TimeArray, VariableArray
from earth2studio.utils.time import to_time_array
from earth2studio.utils.coords import map_coords, split_coords

# Dealing with zarr 3.0 API breaks and type checking
try:
    zarr_version = version("zarr")
    zarr_major_version = int(zarr_version.split(".")[0])
except Exception:
    zarr_major_version = 2

load_dotenv()  # TODO: make common example prep function


class h5DataSetFile:
    """A local xarray dataset file data source. This file should be compatable with
    xarray. For example, a netCDF file.

    Parameters
    ----------
    file_path : str
        Path to xarray dataset compatible file.
    metadata_file_path : str
        Path to h5 metadata file, should provide
        metadata for coords.
    array_name : str
        Data array name in xarray dataset

    Parameters
    ----------
    time : datetime | list[datetime] | TimeArray
        Timestamps to return data for.
    variable : str | list[str] | VariableArray
        Strings or list of strings that refer to variables to return.

    Returns
    -------
    xr.DataArray
        Loaded data array
    """

    def __init__(
        self,
        file_dir: str,
        metadata_file_path: str,
        array_name: str,
        years: int | list[int],
        **xr_args: Any,
    ):
        self.file_dir = file_dir
        self.metadata_file_path = metadata_file_path

        # Open metadata
        with open(metadata_file_path) as f:
            metadata = json.load(f)

        dims = list(metadata["dims"])
        dims = ["variable" if d == "channel" else d for d in dims]
        time_step = timedelta(hours=6)  # metadata['dhours']

        if isinstance(years, int):
            years = [years]

        self.ds = []
        for year in years:
            file_path = os.path.join(file_dir, f"{year}.h5")
            ds = xr.open_dataset(
                file_path, engine="h5netcdf", phony_dims="sort", **xr_args
            )[array_name]

            ds = ds.rename(dict(zip(ds.dims, dims)))
            ds = ds.assign_coords(
                time=[
                    datetime(year, 1, 1) + time_step * int(i)
                    for i in range(len(ds.time))
                ],
                **dict(
                    [
                        ["variable", metadata["coords"]["channel"]],
                        ["lat", metadata["coords"]["lat"]],
                        ["lon", metadata["coords"]["lon"]],
                    ]
                ),
            )
            ds = ds.chunk(dict(time=4))
            self.ds.append(ds)

        self.ds = xr.concat(self.ds, dim="time")

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Function to get data."""
        return self.ds.sel(time=time, variable=variable)


class AsyncDataFetcher:
    """
    An asynchronous data prefetcher that streams batches from a data source in a background thread.

    This class is designed to asynchronously fetch data (e.g., weather or scientific datasets)
    from a `DataSource` using a worker thread, buffering the results in a queue to overlap
    I/O or preprocessing with model training. It behaves like an iterator and can be used
    in a `for` loop to consume data sequentially.

    Parameters
    ----------
    data_source : DataSource
        The backend data source object used to retrieve samples (must be compatible with `fetch_data`).
    time_arrays : List[np.ndarray]
        A list of time indices or timestamps to fetch data for.
    variables : np.ndarray
        Array of variables to extract from the data source (e.g., temperature, pressure).
    lead_times : np.ndarray
        Lead times for which forecasts or lagged values should be fetched.
    device : torch.device
        The device to which the fetched tensors should be moved (e.g., `torch.device("cuda")`).
    max_prefetch : int, optional (default=4)
        Maximum number of batches to prefetch and hold in the internal queue.
    **fetch_kwargs
        Additional keyword arguments passed to the underlying `fetch_data` call.

    Notes
    -----
    - The class spawns a background thread that sequentially calls `fetch_data`
      on the provided `time_arrays` and enqueues the results.
    - Iteration stops automatically when all items are consumed or if `close()` is called.
    - Exceptions raised in the worker thread are propagated to the main thread
      during iteration.

    Examples
    --------
    >>> fetcher = AsyncDataFetcher(
    ...     data_source=my_source,
    ...     time_arrays=[np.array([0, 1]), np.array([2, 3])],
    ...     variables=np.array(["t2m", "w10m"]),
    ...     lead_times=np.array([0, 6]),
    ...     device=torch.device("cuda"),
    ... )
    >>> for batch in fetcher:
    ...     # batch is ready on GPU
    ...     train_step(batch)
    >>> fetcher.close()
    """

    def __init__(
        self,
        data_source: DataSource,
        time_arrays: List[np.array],
        variables: np.array,
        lead_times: np.array,
        device: torch.device,
        max_prefetch=4,
        **fetch_kwargs,
    ):
        self.data_source = data_source
        self.time_arrays = time_arrays
        self.variables = variables
        self.lead_times = lead_times
        self.device = device

        self.max_prefetch = max_prefetch
        self._queue = queue.Queue(max_prefetch)
        self._thread = threading.Thread(target=self._worker)
        self._stop = threading.Event()
        self._started = False

    def _worker(self):
        for time in self.time_arrays:
            if self._stop.is_set():
                break
            try:
                data = fetch_data(
                    source=self.data_source,
                    time=time,
                    variable=self.variables,
                    lead_time=self.lead_times,
                    device=self.device,
                )
                self._queue.put(data)
            except Exception as e:
                self._queue.put(e)
        self._queue.put(StopIteration)

    def __iter__(self):
        if not self._started:
            self._thread.start()
            self._started = True
        return self

    def __next__(self):
        item = self._queue.get()
        if isinstance(item, Exception):
            raise item
        if item is StopIteration:
            raise StopIteration
        return item

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg: DictConfig):
    DistributedManager.initialize()
    manager = DistributedManager()
    device = manager.device
    rank = manager.rank
    world_size = manager.world_size

    logger.info(f"Running on rank {rank} with world size {world_size}")

    # Load the default model package which downloads the check point from NGC
    model = instantiate(cfg.model)
    model.to(device)
    logger.info(f"Model loaded on rank {rank}")

    # Create the data source
    data_source = h5DataSetFile(
        file_dir=cfg.data_source.file_dir,
        metadata_file_path=cfg.data_source.metadata_path,
        array_name=cfg.data_source.array_name,
        years=cfg.data_source.years,
    )
    logger.info(f"Data source loaded on rank {rank}")

    # Create the IO handler, store in memory
    io = instantiate(cfg.io)
    logger.info(f"IO handler loaded on rank {rank}")

    output_coords = instantiate(cfg.output_coords)
    for key, value in output_coords.items():
        output_coords[key] = np.asarray(value)

    times = pd.date_range(start=cfg.start_time, end=cfg.end_time, freq=cfg.freq)
    times = to_time_array(times)

    rank_times = times[rank::world_size]

    if cfg.initialize_io:
        if rank == 0:
            logger.info(f"Initializing IO on rank {rank}")
            # Set up IO backend
            total_coords = model.output_coords(model.input_coords()).copy()
            for key, value in model.output_coords(
                model.input_coords()
            ).items():  # Scrub batch dims
                if value.shape == (0,):
                    del total_coords[key]

            total_coords["time"] = times
            total_coords["lead_time"] = np.asarray(
                [
                    model.output_coords(model.input_coords())["lead_time"] * i
                    for i in range(cfg.nsteps + 1)
                ]
            ).flatten()
            total_coords.move_to_end("lead_time", last=False)
            total_coords.move_to_end("time", last=False)

            for key, value in total_coords.items():
                total_coords[key] = output_coords.get(key, value)
            var_names = total_coords.pop("variable")
            io.add_array(total_coords, var_names)

            # Completed timesteps
            io.add_array(
                {"time": total_coords["time"]},
                "timesteps_completed",
                data=torch.zeros(len(total_coords["time"]), dtype=torch.int32),
            )

    torch.distributed.barrier()

    # Create batches of initial conditions
    bt = []
    for i in range(0, len(rank_times), cfg.batch_size):
        time = rank_times[i : i + cfg.batch_size]
        timesteps_completed, _ = io.read({"time": time}, "timesteps_completed")
        if torch.all(timesteps_completed == 1):
            logger.info(
                f"Skipping timestep {time} because it has already been completed on rank {rank}"
            )
            continue
        else:
            bt.append(rank_times[i : i + cfg.batch_size])

    logger.info(f"Running {len(bt)} batches on rank {rank}")

    fetcher = AsyncDataFetcher(
        data_source=data_source,
        time_arrays=bt,
        variables=model.input_coords()["variable"],
        lead_times=model.input_coords()["lead_time"],
        device=device,
    )

    for x, coords in tqdm(
        fetcher, desc=f"Processing timesteps on rank {rank}", total=len(bt)
    ):
        logger.info(f"Running model on rank {rank} at time {coords['time']}")
        x, coords = map_coords(x, coords, model.input_coords())

        iterator = model.create_iterator(x, coords)

        # Write data to IO backend
        for step, (x, coords) in enumerate(iterator):
            x, coords = map_coords(x, coords, output_coords)
            x = x.cpu()
            torch.cuda.synchronize()
            io.write(*split_coords(x, coords))
            if step == cfg.nsteps:
                break

        # Update completed timesteps
        io.write(
            torch.ones(x.shape[0], dtype=torch.int32),
            {"time": coords["time"]},
            "timesteps_completed",
        )
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
