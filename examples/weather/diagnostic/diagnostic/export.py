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

import json
import os
from typing import List, Union

import numpy as np

from physicsnemo import Module
from physicsnemo.datapipes.climate import ClimateDataSourceSpec, ClimateDatapipe


def export_diagnostic_e2mip(
    out_dir: str,
    model: Module,
    datapipe: ClimateDatapipe,
    in_source: ClimateDataSourceSpec,
    out_source: ClimateDataSourceSpec,
    extra_in_variables: Union[List[str], None] = None,
    extra_out_variables: Union[List[str], None] = None,
    model_name: Union[str, None] = None,
):
    """Convert a diagnostic model module into an Earth-2 MIP model package."""

    # automatically add extra variables from datapipe
    use_latlon = "sincos_latlon" in datapipe.invariants
    add_extra_variables = {
        "uvcossza": in_source.use_cos_zenith,
        "sinlat": use_latlon,
        "coslat": use_latlon,
        "sinlon": use_latlon,
        "coslon": use_latlon,
        "z": "geopotential" in datapipe.invariants,
        "lsm": "land_sea_mask" in datapipe.invariants,
    }
    if extra_in_variables is None:
        extra_in_variables = []
    extra_in_variables = extra_in_variables + [
        var for (var, add_var) in add_extra_variables.items() if add_var
    ]

    # get channel names and shape from sources
    in_channel_names = in_source.variables
    if extra_in_variables is not None:
        in_channel_names += extra_in_variables
    out_channel_names = out_source.variables
    if extra_out_variables is not None:
        out_channel_names += extra_out_variables
    grid_shape = in_source.cropped_data_shape

    if model_name is None:
        model_name = model.meta.name

    _create_e2mip_package(
        out_dir,
        model_name,
        model,
        in_channel_names,
        out_channel_names,
        grid_shape=grid_shape,
        input_stats={"mean": in_source.mu, "std": in_source.sd},
        output_stats={"mean": out_source.mu, "std": out_source.sd},
    )


def _create_e2mip_package(
    out_dir,
    model_name,
    model,
    in_channel_names,
    out_channel_names,
    grid_shape=(720, 1440),
    input_stats=None,
    output_stats=None,
):
    model_dir = os.path.join(out_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    model_fn = os.path.join(model_dir, f"{model_name}.mdlus")
    model.save(model_fn)

    metadata = {
        "grid": "x".join(str(n) for n in grid_shape),
        "in_channel_names": in_channel_names,
        "out_channel_names": out_channel_names,
    }
    metadata_fn = os.path.join(model_dir, "metadata.json")
    with open(metadata_fn, "w") as f:
        json.dump(metadata, f, indent=2)

    if input_stats is not None:
        in_mean_fn = os.path.join(model_dir, "input_means.npy")
        np.save(in_mean_fn, input_stats["mean"])
        in_std_fn = os.path.join(model_dir, "input_stds.npy")
        np.save(in_std_fn, input_stats["std"])
    if output_stats is not None:
        out_mean_fn = os.path.join(model_dir, "output_means.npy")
        np.save(out_mean_fn, output_stats["mean"])
        out_std_fn = os.path.join(model_dir, "output_stds.npy")
        np.save(out_std_fn, output_stats["std"])
