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

from datetime import datetime, timedelta
import time

import numpy as np
import nvidia.dali as dali

from physicsnemo.datapipes.climate.climate import (
    ClimateDatapipe,
    ClimateHDF5DaliExternalSource,
)


class InterpHDF5DaliExternalSource(ClimateHDF5DaliExternalSource):
    """
    DALI source for reading HDF5 formatted climate data files.

    Specialized for interpolation training with HDF5 climate data.

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to parent classes.
    all_steps : bool, optional
        Whether to return all steps in the sequence. Default is False.
    **kwargs : dict
        Keyword arguments passed to parent classes.
    """

    def __init__(self, *args, all_steps: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_steps = all_steps

    def __call__(
        self, sample_info: dali.types.SampleInfo
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get data from source.

        Parameters
        ----------
        sample_info : dali.types.SampleInfo
            Information about the sample to retrieve.

        Returns
        -------
        state_seq : np.ndarray
            Sequence of training data.
        timestamps : np.ndarray
            Accompanying timestamps for the sequence.
        """

        if sample_info.iteration >= self.num_batches:
            raise StopIteration()

        # Shuffle before the next epoch starts
        if self.shuffle and sample_info.epoch_idx != self.last_epoch:
            print("Shuffling indices")
            np.random.shuffle(self.indices)
            self.last_epoch = sample_info.epoch_idx

        # Get local indices from global index
        # TODO: This is very hacky, but it works for now
        idx = self.indices[sample_info.idx_in_epoch]
        year_idx = idx // self.num_samples_per_year
        in_idx = idx % self.num_samples_per_year

        # quasi-unique deterministic seed for each sample
        seed = (
            (sample_info.epoch_idx << 32)
            + (sample_info.idx_in_epoch << 16)
            + sample_info.idx_in_batch
        )

        interp_idx = np.random.default_rng(seed=seed).integers(self.stride + 1)
        if self.all_steps:
            steps = np.arange(self.stride + 1)
        else:
            steps = np.array([0, self.stride, interp_idx])
        state_seq = self._load_sequence(year_idx, in_idx, steps)

        # Load sequence of timestamps
        year = self.start_year + year_idx
        start_time = datetime(year, 1, 1) + timedelta(hours=int(in_idx) * self.dt)
        timestamps = np.array(
            [(start_time + timedelta(hours=i * self.dt)).timestamp() for i in steps]
        )
        return state_seq, timestamps

    def __len__(self) -> int:
        return len(self.indices) // self.stride

    def _get_read_buffer(self, steps: list[int], data) -> np.ndarray:
        """Get memory buffer for reading data."""
        shape = (len(steps), len(self.chans)) + data.shape[-2:]
        return np.empty(shape, dtype=np.float32)

    def _load_sequence(
        self, year_idx: int, idx: int, steps: np.ndarray, num_retries: int = 10
    ) -> np.ndarray:
        """
        Load sequence of data for interpolation training.

        Parameters
        ----------
        year_idx : int
            The index of the yearly data file.
        idx : int
            The starting index of the data sequence in the yearly file.
        steps : np.ndarray
            Array of index offsets relative to idx (e.g. [0, 6, 2]).
        num_retries : int, optional
            Number of times to retry in case of IO failure. Default is 10.

        Returns
        -------
        np.ndarray
            Data of shape (len(steps), num_channels, height, width).
        """

        # the data is returned in a (time, channels, height, width) shape
        data = self._get_data_file(year_idx)["fields"]

        seq = self._get_read_buffer(steps, data)
        steps = list(steps)  # so we can use .index()
        for step_idx, s in enumerate(steps):
            first_step_idx = steps.index(s)
            if first_step_idx != step_idx:
                # when two steps are the same, copy previous to avoid redundant data I/O
                seq[step_idx] = seq[first_step_idx]
            else:
                for retry_num in range(num_retries + 1):
                    try:
                        # equivalent to: seq[step_idx] = data[idx + s]
                        data.read_direct(seq, np.s_[idx + s], np.s_[step_idx])
                        break
                    except BlockingIOError:
                        # Some systems have had occasional IO issues that can often be
                        # resolved by retrying
                        if retry_num == num_retries:
                            raise
                        else:
                            print(
                                f"IO error reading year_idx={year_idx} idx={idx}, retrying in 5 sec..."
                            )
                            time.sleep(5)
        return seq


class InterpClimateDatapipe(ClimateDatapipe):
    """
    Extends ClimateDatapipe to use interpolation source.
    """

    def _source_cls_from_type(
        self, source_type: str
    ) -> type[InterpHDF5DaliExternalSource]:
        """
        Get the external source class based on a string descriptor.

        Parameters
        ----------
        source_type : str
            String identifier for the source type (e.g., 'hdf5').

        Returns
        -------
        type[InterpHDF5DaliExternalSource]
            The appropriate external source class for the given type.
        """
        return {
            "hdf5": InterpHDF5DaliExternalSource,
        }[source_type]
