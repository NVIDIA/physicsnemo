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
from typing import Callable, Iterable, Tuple, Union

import torch

from physicsnemo.datapipes.climate import ClimateDatapipe, ClimateDataSourceSpec
from physicsnemo.datapipes.climate.utils import invariant
from physicsnemo.distributed import DistributedManager


def setup_datapipes(
    train_specs: Iterable[ClimateDataSourceSpec],
    valid_specs: Iterable[ClimateDataSourceSpec],
    dist_manager: DistributedManager,
    geopotential_filename: Union[str, None] = None,
    geopotential_variable: str = "Z",
    lsm_filename: Union[str, None] = None,
    lsm_variable: str = "LSM",
    use_latlon: bool = True,
    num_samples_per_year_train: int = 1456,
    num_samples_per_year_valid: int = 4,
    batch_size_train: int = 2,  # TODO: enable setting global batch size
    batch_size_valid: Union[int, None] = None,
    crop_window: Union[Tuple[Tuple[float, float], Tuple[float, float]], None] = None,
    num_workers: int = 8,
):
    if batch_size_valid is None:
        batch_size_valid = batch_size_train

    invariants = {}
    if geopotential_filename is not None:
        invariants["geopotential"] = invariant.FileInvariant(
            geopotential_filename, geopotential_variable, normalize=True
        )
    if lsm_filename is not None:
        invariants["land_sea_mask"] = invariant.FileInvariant(
            lsm_filename, lsm_variable
        )
    if use_latlon:
        invariants["sincos_latlon"] = invariant.LatLon()

    datapipe_kwargs = {
        "crop_window": crop_window,
        "invariants": invariants,
        "num_workers": num_workers,
        "device": dist_manager.device,
    }

    train_datapipe = ClimateDatapipe(
        train_specs,
        num_samples_per_year=num_samples_per_year_train,
        batch_size=batch_size_train,
        process_rank=dist_manager.rank,
        world_size=dist_manager.world_size,
        **datapipe_kwargs,
    )

    valid_datapipe = ClimateDatapipe(
        valid_specs,
        num_samples_per_year=num_samples_per_year_valid,
        batch_size=1,
        shuffle=False,
        **datapipe_kwargs,
    )

    return (train_datapipe, valid_datapipe)


def data_source_specs(
    state_params: dict,
    diag_params: dict,
    train_dir: str = "train",
    valid_dir: str = "test",
):
    """Initialize data source specs for both training and validation."""
    state_dir = state_params.pop("data_dir")
    diag_dir = diag_params.pop("data_dir")

    state_train_spec = ClimateDataSourceSpec(
        data_dir=os.path.join(state_dir, train_dir), **state_params
    )
    diag_train_spec = ClimateDataSourceSpec(
        data_dir=os.path.join(diag_dir, train_dir), **diag_params
    )

    state_valid_spec = ClimateDataSourceSpec(
        data_dir=os.path.join(state_dir, valid_dir), **state_params
    )
    diag_valid_spec = ClimateDataSourceSpec(
        data_dir=os.path.join(diag_dir, valid_dir), **diag_params
    )

    return ([state_train_spec, diag_train_spec], [state_valid_spec, diag_valid_spec])


def batch_converter(
    state_spec: ClimateDataSourceSpec,
    diag_spec: ClimateDataSourceSpec,
    datapipe: ClimateDatapipe,
    diag_norm: Union[Callable, None] = None,
) -> Callable:
    """Create a function to convert a batch of data coming from the datapipe
    in dict format to a tuple of input and output tensors."""

    state_name = state_spec.name
    diag_name = diag_spec.name

    @torch.no_grad()
    def _input_output_from_batch_data(batch):
        batch = batch[0]

        # concatenate all input variables to a single tensor
        state = [batch[f"state_seq-{state_name}"]]
        if state_spec.use_cos_zenith:
            state.append(batch[f"cos_zenith-{state_name}"])
        if "sincos_latlon" in datapipe.invariants:
            state.append(torch.unsqueeze(batch["sincos_latlon"], dim=1))
        if "geopotential" in datapipe.invariants:
            state.append(torch.unsqueeze(batch["geopotential"], dim=1))
        if "land_sea_mask" in datapipe.invariants:
            state.append(torch.unsqueeze(batch["land_sea_mask"], dim=1))
        state = torch.cat(state, dim=2)
        state = torch.squeeze(state, dim=1)  # drop time dimension

        diag = batch[f"state_seq-{diag_name}"]
        diag = torch.squeeze(diag, dim=1)  # drop time dimension
        if diag_norm is not None:
            diag = diag_norm.normalize(diag)  # custom normalization

        return (state, diag)

    return _input_output_from_batch_data
