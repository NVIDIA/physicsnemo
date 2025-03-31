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
import datetime
import glob
import json
import logging
from urllib.parse import urlparse

import numpy as np
import torch
import xarray

import physicsnemo  # noqa: F401 for docs
from physicsnemo.models.dlwp import DLWP
from physicsnemo.models.graphcast.graph_cast_net import GraphCastNet
from physicsnemo.utils import filesystem
from physicsnemo.utils.zenith_angle import cos_zenith_angle

logger = logging.getLogger(__name__)


# class _DummyModule(torch.nn.Module):
#     """Hack to handle that checkpoint parameter names begin with "module." """

#     def __init__(self, model):
#         super().__init__()
#         self.module = model


class _CosZenWrapper(torch.nn.Module):
    def __init__(self, model, lon, lat):
        super().__init__()
        self.model = model
        self.lon = lon
        self.lat = lat

    def forward(self, x, time):
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        cosz = cos_zenith_angle(time, lon_grid, lat_grid)
        cosz = cosz.astype(np.float32)
        z = torch.from_numpy(cosz).to(device=x.device)
        x, z = torch.broadcast_tensors(x, z)
        x = torch.cat([x, z], dim=1)
        return self.model(x)


# def sfno(package: filesystem.Package, pretrained: bool = True) -> torch.nn.Module:
#     """Load SFNO model from checkpoints trained with era5_wind"""
#     path = package.get("config.json")
#     params = ParamsBase.from_json(path)
#     model = sfnonet.SphericalFourierNeuralOperatorNet(params)
#     logger.info(str(params.to_dict()))

#     if pretrained:
#         weights = package.get("weights.tar")
#         checkpoint = torch.load(weights)
#         load_me = _DummyModule(model)
#         state = checkpoint["model_state"]
#         state = {"module.device_buffer": model.device_buffer, **state}
#         load_me.load_state_dict(state)

#     if params.add_zenith:
#         nlat = params.img_shape_x
#         nlon = params.img_shape_y
#         lat = 90 - np.arange(nlat) * 0.25
#         lon = np.arange(nlon) * 0.25
#         model = _CosZenWrapper(model, lon, lat)

#     return model


class _GraphCastWrapper(torch.nn.Module):
    def __init__(self, model, dtype):
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, x):
        x = x.to(self.dtype)
        y = self.model(x)
        return y


def graphcast_34ch(
    package: filesystem.Package, pretrained: bool = True
) -> torch.nn.Module:
    """Load Graphcast 34 channel model from a checkpoint"""
    num_channels = 34

    icospheres_path = package.get("icospheres.json")
    static_data_path = package.get("static", recursive=True)

    # instantiate the model, set dtype
    base_model = (
        GraphCastNet(
            meshgraph_path=icospheres_path,
            static_dataset_path=static_data_path,
            input_dim_grid_nodes=num_channels,
            input_dim_mesh_nodes=3,
            input_dim_edges=4,
            output_dim_grid_nodes=num_channels,
            processor_layers=16,
            hidden_dim=512,
            do_concat_trick=True,
        )
        .to(dtype=torch.bfloat16)
        .to("cuda")  # TODO hardcoded
    )

    # set model to inference mode
    base_model.eval()

    model = _GraphCastWrapper(base_model, torch.bfloat16)

    if pretrained:
        path = package.get("weights.tar")
        checkpoint = torch.load(path)
        weights = checkpoint["model_state_dict"]
        weights = _fix_state_dict_keys(weights, add_module=False)
        model.model.load_state_dict(weights, strict=True)

    return model


class _DLWPWrapper(torch.nn.Module):
    def __init__(
        self,
        model,
        lsm,
        longrid,
        latgrid,
        topographic_height,
        ll_to_cs_mapfile_path,
        cs_to_ll_mapfile_path,
    ):
        super(_DLWPWrapper, self).__init__()
        self.model = model
        self.lsm = lsm
        self.longrid = longrid
        self.latgrid = latgrid
        self.topographic_height = topographic_height

        # load map weights
        # Note: these map files are created using TempestRemap library
        # https://github.com/ClimateGlobalChange/tempestremap
        # To generate the maps, the below sequence of commands can be
        # executed once TempestRemap is installed.

        # GenerateRLLMesh --lat 721 --lon 1440 --file out_latlon.g --lat_begin 90 --lat_end -90 --out_format Netcdf4
        # GenerateCSMesh --res <desired-res> --file out_cubedsphere.g --out_format Netcdf4
        # GenerateOverlapMesh --a out_latlon.g --b out_cubedsphere.g --out overlap_latlon_cubedsphere.g --out_format Netcdf4
        # GenerateOfflineMap --in_mesh out_latlon.g --out_mesh out_cubedsphere.g --ov_mesh overlap_latlon_cubedsphere.g --in_np 1 --in_type FV --out_type FV --out_map map_LL_CS.nc --out_format Netcdf4
        # GenerateOverlapMesh --a out_cubedsphere.g --b out_latlon.g --out overlap_cubedsphere_latlon.g --out_format Netcdf4
        # GenerateOfflineMap --in_mesh out_cubedsphere.g --out_mesh out_latlon.g --ov_mesh overlap_cubedsphere_latlon.g --in_np 1 --in_type FV --out_type FV --out_map map_CS_LL.nc --out_format Netcdf4
        self.input_map_wts = xarray.open_dataset(ll_to_cs_mapfile_path)
        self.output_map_wts = xarray.open_dataset(cs_to_ll_mapfile_path)

    def prepare_input(self, input, time):
        device = input.device
        dtype = input.dtype

        i = self.input_map_wts.row.values - 1
        j = self.input_map_wts.col.values - 1
        data = self.input_map_wts.S.values
        M = torch.sparse_coo_tensor(np.array((i, j)), data).type(dtype).to(device)

        N, T, C = input.shape[0], input.shape[1], input.shape[2]
        input = (M @ input.reshape(N * T * C, -1).T).T
        S = int((M.shape[0] / 6) ** 0.5)
        input = input.reshape(N, T, C, 6, S, S)
        input_list = list(torch.split(input, 1, dim=1))
        input_list = [tensor.squeeze(1) for tensor in input_list]
        repeat_vals = (input.shape[0], -1, -1, -1, -1)  # repeat along batch dimension
        for i in range(len(input_list)):
            tisr = np.maximum(
                cos_zenith_angle(
                    time
                    - datetime.timedelta(hours=6 * (input.shape[1] - 1))
                    + datetime.timedelta(hours=6 * i),
                    self.longrid,
                    self.latgrid,
                ),
                0,
            ) - (
                1 / np.pi
            )  # subtract mean value
            tisr = (
                torch.tensor(tisr, dtype=dtype)
                .to(device)
                .unsqueeze(dim=0)
                .unsqueeze(dim=0)
            )  # add channel and batch size dimension
            tisr = tisr.expand(*repeat_vals)  # TODO - find better way to batch TISR
            input_list[i] = torch.cat(
                (input_list[i], tisr), dim=1
            )  # concat along channel dim

        input_model = torch.cat(
            input_list, dim=1
        )  # concat the time dimension into channels

        lsm_tensor = torch.tensor(self.lsm, dtype=dtype).to(device).unsqueeze(dim=0)
        lsm_tensor = lsm_tensor.expand(*repeat_vals)
        topographic_height_tensor = (
            torch.tensor((self.topographic_height - 3.724e03) / 8.349e03, dtype=dtype)
            .to(device)
            .unsqueeze(dim=0)
        )
        topographic_height_tensor = topographic_height_tensor.expand(*repeat_vals)

        input_model = torch.cat(
            (input_model, lsm_tensor, topographic_height_tensor), dim=1
        )
        return input_model

    def prepare_output(self, output):
        device = output.device
        dtype = output.dtype
        output = torch.split(output, output.shape[1] // 2, dim=1)
        output = torch.stack(output, dim=1)  # add time dimension back in
        i = self.output_map_wts.row.values - 1
        j = self.output_map_wts.col.values - 1
        data = self.output_map_wts.S.values
        M = torch.sparse_coo_tensor(np.array((i, j)), data).type(dtype).to(device)

        N, T, C = output.shape[0], 2, output.shape[2]
        output = (M @ output.reshape(N * T * C, -1).T).T
        output = output.reshape(N, T, C, 721, 1440)

        return output

    def forward(self, x, time):
        x = self.prepare_input(x, time)
        y = self.model(x)
        return self.prepare_output(y)


def dlwp(package, pretrained=True):
    # load static datasets
    lsm = xarray.open_dataset(package.get("land_sea_mask_rs_cs.nc"))["lsm"].values
    topographic_height = xarray.open_dataset(package.get("geopotential_rs_cs.nc"))[
        "z"
    ].values
    latlon_grids = xarray.open_dataset(package.get("latlon_grid_field_rs_cs.nc"))
    latgrid, longrid = latlon_grids["latgrid"].values, latlon_grids["longrid"].values

    # load maps
    parsed_uri = urlparse(package.root)
    if parsed_uri.scheme == "file":
        root_path = parsed_uri.path
    else:
        root_path = package.root

    ll_to_cs_file = glob.glob(root_path + package.seperator + "map_LL*_CS*.nc")
    cs_to_ll_file = glob.glob(root_path + package.seperator + "map_CS*_LL*.nc")

    if ll_to_cs_file:
        file_path = ll_to_cs_file[0]  # take the first match
        if parsed_uri.scheme == "file":
            ll_to_cs_relative_path = file_path[len(root_path) :].lstrip(
                package.seperator
            )
        else:
            ll_to_cs_relative_path = file_path[len(root_path) :]

    if cs_to_ll_file:
        file_path = cs_to_ll_file[0]
        if parsed_uri.scheme == "file":
            cs_to_ll_relative_path = file_path[len(root_path) :].lstrip(
                package.seperator
            )
        else:
            cs_to_ll_relative_path = file_path[len(root_path) :]

    ll_to_cs_mapfile_path = package.get(ll_to_cs_relative_path)
    cs_to_ll_mapfile_path = package.get(cs_to_ll_relative_path)

    with open(package.get("config.json")) as json_file:
        config = json.load(json_file)
        core_model = DLWP(
            nr_input_channels=config["nr_input_channels"],
            nr_output_channels=config["nr_output_channels"],
        )

        if pretrained:
            weights_path = package.get("weights.pt")
            weights = torch.load(weights_path)
            fixed_weights = _fix_state_dict_keys(weights, add_module=False)
            core_model.load_state_dict(fixed_weights)

        model = _DLWPWrapper(
            core_model,
            lsm,
            longrid,
            latgrid,
            topographic_height,
            ll_to_cs_mapfile_path,
            cs_to_ll_mapfile_path,
        )

        model.eval()

    return model


def _fix_state_dict_keys(state_dict, add_module=False):
    """Add or remove 'module.' from state_dict keys

    Parameters
    ----------
    state_dict : Dict
        Model state_dict
    add_module : bool, optional
        If True, will add 'module.' to keys, by default False

    Returns
    -------
    Dict
        Model state_dict with fixed keys
    """
    fixed_state_dict = {}
    for key, value in state_dict.items():
        if add_module:
            new_key = "module." + key
        else:
            new_key = key.replace("module.", "")
        fixed_state_dict[new_key] = value
    return fixed_state_dict
