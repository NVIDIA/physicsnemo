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

import pytest
import torch
from pytest_utils import import_or_fail

from . import common

Tensor = torch.Tensor


@pytest.fixture
def data_dir(nfs_data_dir):
    return nfs_data_dir.joinpath("datasets/hdf5/test")


@pytest.fixture
def stats_files(nfs_data_dir):
    return {
        "mean": nfs_data_dir.joinpath("datasets/hdf5/stats/global_means.npy"),
        "std": nfs_data_dir.joinpath("datasets/hdf5/stats/global_stds.npy"),
    }


@pytest.fixture
def metadata_path(nfs_data_dir):
    return nfs_data_dir.joinpath("datasets/hdf5/data.json")


@pytest.fixture
def lsm_filename(nfs_data_dir):
    return nfs_data_dir.joinpath("datasets/hdf5/static/land_sea_mask.nc")


@pytest.fixture
def geopotential_filename(nfs_data_dir):
    """Geopotential file."""
    return nfs_data_dir.joinpath("datasets/hdf5/static/geopotential.nc")


# default keyword args for DRY
spec_kwargs = dict(
    name="era5",
    file_type="hdf5",
    channels=None,
    use_cos_zenith=True,
    num_steps=1,
    stride=1,
)
datapipe_kwargs = dict(
    dt=1,
    start_year=2018,
    num_samples_per_year=None,
    batch_size=1,
    num_workers=1,
    shuffle=False,
)

# Skip CPU tests because too slow
@import_or_fail("netCDF4")
@pytest.mark.parametrize("device", ["cuda:0"])
def test_climate_hdf5_constructor(
    data_dir,
    stats_files,
    metadata_path,
    lsm_filename,
    geopotential_filename,
    device,
    pytestconfig,
):

    from physicsnemo.datapipes.climate import ClimateDatapipe, ClimateDataSourceSpec
    from physicsnemo.datapipes.climate.utils import invariant

    # construct data pipe
    spec = ClimateDataSourceSpec(
        data_dir=data_dir,
        stats_files=stats_files,
        metadata_path=metadata_path,
        **spec_kwargs
    )
    invariants = {
        "land_sea_mask": invariant.FileInvariant(lsm_filename, "lsm"),
        "geopotential": invariant.FileInvariant(geopotential_filename, "z"),
        "cos_latlon": invariant.LatLon(),
    }
    datapipe = ClimateDatapipe(
        [spec], invariants=invariants, device=torch.device(device), **datapipe_kwargs
    )

    # iterate datapipe is iterable
    common.check_datapipe_iterable(datapipe)

    # check for failure from invalid dir
    try:
        # init datapipe with empty path
        # if datapipe throws an IO error then this should pass
        spec = ClimateDataSourceSpec(
            data_dir="/null_path",
            stats_files=stats_files,
            metadata_path=metadata_path,
            **spec_kwargs
        )
        datapipe = ClimateDatapipe(
            [spec],
            invariants=invariants,
            device=torch.device(device),
            **datapipe_kwargs
        )
        raise IOError("Failed to raise error given null data path")
    except IOError:
        pass

    # check for failure from invalid dir
    try:
        # init datapipe with empty path
        # if datapipe throws an IO error then this should pass
        spec = ClimateDataSourceSpec(
            data_dir=data_dir,
            stats_files={"mean": "/null_path", "std": "/null_path"},
            metadata_path=metadata_path,
            **spec_kwargs
        )
        datapipe = ClimateDatapipe(
            [spec],
            invariants=invariants,
            device=torch.device(device),
            **datapipe_kwargs
        )
        raise IOError("Failed to raise error given null stats path")
    except IOError:
        pass

    # check for failure from invalid num_samples_per_year
    try:
        spec = ClimateDataSourceSpec(
            data_dir=data_dir,
            stats_files=stats_files,
            metadata_path=metadata_path,
            **spec_kwargs
        )
        datapipe = ClimateDatapipe(
            [spec],
            invariants=invariants,
            device=torch.device(device),
            **{**datapipe_kwargs, **{"num_samples_per_year": 5}}
        )
        raise ValueError("Failed to raise error given invalid num_samples_per_year")
    except ValueError:
        pass

    # check invalid channel
    try:
        spec = ClimateDataSourceSpec(
            data_dir=data_dir,
            stats_files=stats_files,
            metadata_path=metadata_path,
            **{**spec_kwargs, **{"channels": [1]}}
        )
        datapipe = ClimateDatapipe(
            [spec],
            invariants=invariants,
            device=torch.device(device),
            **{**datapipe_kwargs, **{"num_samples_per_year": 5}}
        )
        raise ValueError("Failed to raise error given invalid channel id")
    except ValueError:
        pass


@import_or_fail("netCDF4")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_climate_hdf5_device(
    data_dir,
    stats_files,
    metadata_path,
    lsm_filename,
    geopotential_filename,
    device,
    pytestconfig,
):

    from physicsnemo.datapipes.climate import ClimateDatapipe, ClimateDataSourceSpec
    from physicsnemo.datapipes.climate.utils import invariant

    # construct data pipe
    spec = ClimateDataSourceSpec(
        data_dir=data_dir,
        stats_files=stats_files,
        metadata_path=metadata_path,
        **spec_kwargs
    )
    invariants = {
        "land_sea_mask": invariant.FileInvariant(lsm_filename, "lsm"),
        "geopotential": invariant.FileInvariant(geopotential_filename, "z"),
        "cos_latlon": invariant.LatLon(),
    }
    datapipe = ClimateDatapipe(
        [spec], invariants=invariants, device=torch.device(device), **datapipe_kwargs
    )

    # test single sample
    for data in datapipe:
        common.check_datapipe_device(data[0]["state_seq-era5"], device)
        common.check_datapipe_device(data[0]["timestamps-era5"], device)
        common.check_datapipe_device(data[0]["land_sea_mask"], device)
        common.check_datapipe_device(data[0]["geopotential"], device)
        common.check_datapipe_device(data[0]["cos_latlon"], device)
        common.check_datapipe_device(data[0]["cos_zenith-era5"], device)
        break


# Skip CPU tests because too slow
@import_or_fail("netCDF4")
@pytest.mark.parametrize("data_channels", [[0, 1]])
@pytest.mark.parametrize("num_steps", [2])
@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("device", ["cuda:0"])
def test_climate_hdf5_shape(
    data_dir,
    stats_files,
    metadata_path,
    lsm_filename,
    geopotential_filename,
    data_channels,
    num_steps,
    batch_size,
    device,
    pytestconfig,
):

    from physicsnemo.datapipes.climate import ClimateDatapipe, ClimateDataSourceSpec
    from physicsnemo.datapipes.climate.utils import invariant

    # construct data pipe
    spec = ClimateDataSourceSpec(
        data_dir=data_dir,
        stats_files=stats_files,
        metadata_path=metadata_path,
        name="era5",
        file_type="hdf5",
        channels=data_channels,
        use_cos_zenith=True,
        num_steps=num_steps,
        stride=1,
    )
    invariants = {
        "land_sea_mask": invariant.FileInvariant(lsm_filename, "lsm"),
        "geopotential": invariant.FileInvariant(geopotential_filename, "z"),
        "cos_latlon": invariant.LatLon(),
    }
    datapipe = ClimateDatapipe(
        [spec],
        invariants=invariants,
        device=torch.device(device),
        dt=1,
        start_year=2018,
        num_samples_per_year=None,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
    )

    # test single sample
    for data in datapipe:
        state_seq = data[0]["state_seq-era5"]
        timestamps = data[0]["timestamps-era5"]
        land_sea_mask = data[0]["land_sea_mask"]
        geopotential = data[0]["geopotential"]
        cos_latlon = data[0]["cos_latlon"]
        cos_zenith = data[0]["cos_zenith-era5"]

        # check batch size
        assert common.check_batch_size(
            [
                state_seq,
                timestamps,
                land_sea_mask,
                geopotential,
                cos_latlon,
                cos_zenith,
            ],
            batch_size,
        )

        # check seq length
        assert common.check_seq_length([state_seq, timestamps, cos_zenith], num_steps)

        # check channels
        if data_channels is None:
            nr_channels = 3
        else:
            nr_channels = len(data_channels)
        assert common.check_channels(state_seq, nr_channels, axis=2)

        # check grid dims
        grid_size = (721, 1440)
        assert common.check_grid([state_seq, cos_zenith], grid_size, axis=(3, 4))
        assert common.check_grid(
            [land_sea_mask, geopotential, cos_latlon], grid_size, axis=(2, 3)
        )
        break


# Skip CPU tests because too slow
@import_or_fail("netCDF4")
@pytest.mark.parametrize("num_steps", [1, 2])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("device", ["cuda:0"])
def test_era5_hdf5_sequence(
    data_dir,
    stats_files,
    lsm_filename,
    geopotential_filename,
    metadata_path,
    num_steps,
    stride,
    device,
    pytestconfig,
):

    from physicsnemo.datapipes.climate import ClimateDatapipe, ClimateDataSourceSpec
    from physicsnemo.datapipes.climate.utils import invariant

    # construct data pipe
    spec = ClimateDataSourceSpec(
        data_dir=data_dir,
        stats_files=stats_files,
        metadata_path=metadata_path,
        name="era5",
        file_type="hdf5",
        channels=None,
        use_cos_zenith=True,
        num_steps=num_steps,
        stride=stride,
    )
    invariants = {
        "land_sea_mask": invariant.FileInvariant(lsm_filename, "lsm"),
        "geopotential": invariant.FileInvariant(geopotential_filename, "z"),
        "cos_latlon": invariant.LatLon(),
    }
    datapipe = ClimateDatapipe(
        [spec], invariants=invariants, device=torch.device(device), **datapipe_kwargs
    )

    # get single sample
    # TODO generalize tests for sequence type datapipes
    for data in datapipe:
        state_seq = data[0]["state_seq-era5"]
        break

    # check if tensor has correct shape
    assert common.check_sequence(
        state_seq, start_index=0, step_size=stride, seq_length=num_steps, axis=1
    )


# Skip CPU tests because too slow
@import_or_fail("netCDF4")
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_era5_hdf5_shuffle(
    data_dir,
    stats_files,
    lsm_filename,
    geopotential_filename,
    metadata_path,
    shuffle,
    stride,
    device,
    pytestconfig,
):

    from physicsnemo.datapipes.climate import ClimateDatapipe, ClimateDataSourceSpec
    from physicsnemo.datapipes.climate.utils import invariant

    # construct data pipe
    spec = ClimateDataSourceSpec(
        data_dir=data_dir,
        stats_files=stats_files,
        metadata_path=metadata_path,
        name="era5",
        file_type="hdf5",
        channels=None,
        use_cos_zenith=True,
        num_steps=1,
        stride=stride,
    )
    invariants = {
        "land_sea_mask": invariant.FileInvariant(lsm_filename, "lsm"),
        "geopotential": invariant.FileInvariant(geopotential_filename, "z"),
        "cos_latlon": invariant.LatLon(),
    }
    datapipe = ClimateDatapipe(
        [spec],
        invariants=invariants,
        device=torch.device(device),
        dt=1,
        start_year=2018,
        num_samples_per_year=None,
        batch_size=1,
        num_workers=1,
        shuffle=shuffle,
    )

    # get all samples
    # TODO generalize this
    tensors = [data[0]["state_seq-era5"] for data in datapipe]

    # check sample order
    assert common.check_shuffle(tensors, shuffle, stride, 8)


@import_or_fail("netCDF4")
@pytest.mark.parametrize("device", ["cuda:0"])
def test_era5_hdf5_cudagraphs(
    data_dir,
    stats_files,
    lsm_filename,
    geopotential_filename,
    device,
    metadata_path,
    pytestconfig,
):

    from physicsnemo.datapipes.climate import ClimateDatapipe, ClimateDataSourceSpec
    from physicsnemo.datapipes.climate.utils import invariant

    # Preprocess function to convert dataloader output into Tuple of tensors
    def input_fn(data) -> Tensor:
        return data[0]["state_seq-era5"]

    # construct data pipe
    spec = ClimateDataSourceSpec(
        data_dir=data_dir,
        stats_files=stats_files,
        metadata_path=metadata_path,
        **spec_kwargs
    )
    invariants = {
        "land_sea_mask": invariant.FileInvariant(lsm_filename, "lsm"),
        "geopotential": invariant.FileInvariant(geopotential_filename, "z"),
        "cos_latlon": invariant.LatLon(),
    }
    datapipe = ClimateDatapipe(
        [spec], invariants=invariants, device=torch.device(device), **datapipe_kwargs
    )

    assert common.check_cuda_graphs(datapipe, input_fn)
