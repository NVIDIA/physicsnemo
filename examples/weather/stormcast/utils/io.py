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

import numpy as np
import xarray as xr
import zarr


def init_inference_results_zarr(
    dataset,  # dataset object
    rundir,  # directory to save results
    output_state_channels,  # list of channel names
    n_steps,  # number of time steps
):
    # initialize zarr
    zarr_output_path = os.path.join(rundir, "data.zarr")
    group = zarr.open_group(zarr_output_path, mode="w")
    group.array("latitude", data=dataset.latitude())
    group.array("longitude", data=dataset.longitude())

    edm_prediction_group = group.create_group("edm_prediction")
    noedm_prediction_group = group.create_group("noedm_prediction")
    target_group = group.create_group("target")
    state, _ = dataset[0]["state"]
    assert state.ndim == 3

    grid_size = state.shape[1:]

    for name in output_state_channels:
        target_group.empty(
            name, shape=(n_steps,) + grid_size, chunks=[1, *grid_size], compressor=None
        )
        edm_prediction_group.empty(
            name, shape=(n_steps,) + grid_size, chunks=[1, *grid_size], compressor=None
        )
        noedm_prediction_group.empty(
            name, shape=(n_steps,) + grid_size, chunks=[1, *grid_size], compressor=None
        )

    return (group, target_group, edm_prediction_group, noedm_prediction_group)


def write_inference_results_zarr(
    denorm_state_pred_edm,  # predictions with diffusion
    denorm_state_pred_noedm,  # predictions without diffusion
    denorm_state_target,  # target data
    edm_prediction_group,  # zarr group for diffusion data
    noedm_prediction_group,  # zarr group for no-diffusion data
    target_group,  # zarr group for target data
    output_state_channels,  # list of channel names
    vardict_state,  # dict mapping channel names to indices
    step,  # time step
):
    for name in output_state_channels:
        k = vardict_state[name]
        edm_prediction_group[name][step] = denorm_state_pred_edm[k]
        noedm_prediction_group[name][step] = denorm_state_pred_noedm[k]
        target_group[name][step] = denorm_state_target[k]


def save_inference_results_netcdf(
    ds_out_path,  # directory where results are stored
    zarr_group,  # zarr group to write out
    vertical_vars,  # variables with multiple vertical levels
    level_names,  # level names for vertical_vars
    horizontal_vars,  # single-level variables
    val_times,  # validation times
):
    lons = zarr_group["longitude"][:, :]
    lats = zarr_group["latitude"][:, :]

    def convert_strings_to_ints(string_list):
        return [int(i) for i in string_list]

    model_levels = convert_strings_to_ints(level_names)

    ds_pred_edm = xr.Dataset()
    ds_pred_noedm = xr.Dataset()
    ds_targ = xr.Dataset()

    for var in vertical_vars:
        dsp_edm = xr.Dataset()
        dsp_noedm = xr.Dataset()
        dst = xr.Dataset()

        for i, level in enumerate(level_names):
            key = f"{var}{level}"

            if key in zarr_group["edm_prediction"]:
                # Extract the data from zarr_group
                data_pred_edm = zarr_group["edm_prediction"][key][:, :]
            else:
                data_pred_edm = np.full(
                    (len(val_times), lats.shape[0], lats.shape[1]), np.nan
                )
                print(f"Key {key} not found in Zarr group. Filling with NaNs.")
            if key in zarr_group["noedm_prediction"]:
                # Extract the data from zarr_group
                data_pred_noedm = zarr_group["noedm_prediction"][key][:, :]
            else:
                data_pred_noedm = np.full(
                    (len(val_times), lats.shape[0], lats.shape[1]), np.nan
                )
                print(f"Key {key} not found in Zarr group. Filling with NaNs.")

            if key in zarr_group["target"]:
                data_targ = zarr_group["target"][key][:, :]
            else:
                data_targ = np.full(
                    (len(val_times), lats.shape[0], lats.shape[1]), np.nan
                )
                print(f"Key {key} not found in Zarr group. Filling with NaNs.")

            dap_edm = xr.DataArray(
                data_pred_edm,
                dims=("time", "y", "x"),
                coords={
                    "time": val_times,
                    "y": np.arange(lats.shape[0]),
                    "x": np.arange(lats.shape[1]),
                    "levels": model_levels[i],
                },
            )
            dap_noedm = xr.DataArray(
                data_pred_noedm,
                dims=("time", "y", "x"),
                coords={
                    "time": val_times,
                    "y": np.arange(lats.shape[0]),
                    "x": np.arange(lats.shape[1]),
                    "levels": model_levels[i],
                },
            )
            dat = xr.DataArray(
                data_targ,
                dims=("time", "y", "x"),
                coords={
                    "time": val_times,
                    "y": np.arange(lats.shape[0]),
                    "x": np.arange(lats.shape[1]),
                    "levels": model_levels[i],
                },
            )

            dsp_edm[f"var_{i}"] = dap_edm
            dsp_noedm[f"var_{i}"] = dap_noedm
            dst[f"var_{i}"] = dat

        combined_pred_edm = xr.concat(
            [dsp_edm[var] for var in dsp_edm.data_vars], dim="levels"
        )
        combined_pred_noedm = xr.concat(
            [dsp_noedm[var] for var in dsp_noedm.data_vars], dim="levels"
        )
        combined_targ = xr.concat([dst[var] for var in dst.data_vars], dim="levels")

        reshaped_pred_edm = combined_pred_edm.transpose("time", "levels", "y", "x")
        reshaped_pred_noedm = combined_pred_noedm.transpose("time", "levels", "y", "x")
        reshaped_targ = combined_targ.transpose("time", "levels", "y", "x")

        ds_pred_edm[f"{var}_comb"] = reshaped_pred_edm
        ds_pred_noedm[f"{var}_comb"] = reshaped_pred_noedm
        ds_targ[f"{var}_comb"] = reshaped_targ

    for var in horizontal_vars:
        data_pred_edm = zarr_group["edm_prediction"][var][:, :]
        data_pred_noedm = zarr_group["noedm_prediction"][var][:, :]
        data_targ = zarr_group["target"][var][:, :]
        dap_edm = xr.DataArray(
            data_pred_edm,
            dims=("time", "y", "x"),
            coords={
                "time": val_times,
                "y": np.arange(lats.shape[0]),
                "x": np.arange(lats.shape[1]),
            },
        )
        dap_noedm = xr.DataArray(
            data_pred_noedm,
            dims=("time", "y", "x"),
            coords={
                "time": val_times,
                "y": np.arange(lats.shape[0]),
                "x": np.arange(lats.shape[1]),
            },
        )
        dat = xr.DataArray(
            data_targ,
            dims=("time", "y", "x"),
            coords={
                "time": val_times,
                "y": np.arange(lats.shape[0]),
                "x": np.arange(lats.shape[1]),
            },
        )
        ds_pred_edm.update({var: dap_edm})
        ds_pred_noedm.update({var: dap_noedm})
        ds_targ.update({var: dat})

    ds_pred_edm["longitude"] = xr.DataArray(lons, dims=("y", "x"))
    ds_pred_edm["latitude"] = xr.DataArray(lats, dims=("y", "x"))
    ds_pred_edm = ds_pred_edm.assign_coords(levels=model_levels)

    ds_pred_noedm["longitude"] = xr.DataArray(lons, dims=("y", "x"))
    ds_pred_noedm["latitude"] = xr.DataArray(lats, dims=("y", "x"))
    ds_pred_noedm = ds_pred_noedm.assign_coords(levels=model_levels)

    ds_targ["longitude"] = xr.DataArray(lons, dims=("y", "x"))
    ds_targ["latitude"] = xr.DataArray(lats, dims=("y", "x"))
    ds_targ = ds_targ.assign_coords(levels=model_levels)

    ds_pred_edm.to_netcdf(f"{ds_out_path}/ds_pred_edm.nc", format="NETCDF4")
    ds_pred_noedm.to_netcdf(f"{ds_out_path}/ds_pred_noedm.nc", format="NETCDF4")
    ds_targ.to_netcdf(f"{ds_out_path}/ds_targ.nc", format="NETCDF4")
