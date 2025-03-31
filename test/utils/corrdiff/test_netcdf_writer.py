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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest_utils import import_or_fail


@pytest.fixture
def mock_ncfile():
    """Fixture to provide a mock NetCDF file object."""
    mock_file = MagicMock()
    # Set up mock for dimensions
    mock_file.createDimension = MagicMock()
    # Set up mock for variables
    mock_file.createVariable = MagicMock(return_value=MagicMock())
    # Set up mock for groups
    mock_file.createGroup = MagicMock(return_value=MagicMock())

    # Set up mock variables
    mock_file["lat"] = MagicMock()
    mock_file["lon"] = MagicMock()
    mock_file["time"] = MagicMock()

    return mock_file


@import_or_fail("cftime")
def test_init(mock_ncfile, pytestconfig):

    from physicsnemo.utils.corrdiff import NetCDFWriter

    lat = np.array([[1.0, 2.0], [3.0, 4.0]])
    lon = np.array([[5.0, 6.0], [7.0, 8.0]])
    input_channels = []
    output_channels = []

    # Create some mock variables for output_channels
    mock_output_channel = MagicMock()
    mock_output_channel.name = "var"
    mock_output_channel.level = "1"
    output_channels.append(mock_output_channel)

    # Create instance
    _ = NetCDFWriter(mock_ncfile, lat, lon, input_channels, output_channels)

    # Assert dimensions were created
    mock_ncfile.createDimension.assert_any_call("time")
    mock_ncfile.createDimension.assert_any_call("ensemble")
    mock_ncfile.createDimension.assert_any_call("x", 2)
    mock_ncfile.createDimension.assert_any_call("y", 2)

    # Assert variables for lat and lon were created
    mock_ncfile.createVariable.assert_any_call("lat", "f", dimensions=("y", "x"))
    mock_ncfile.createVariable.assert_any_call("lon", "f", dimensions=("y", "x"))
    mock_ncfile.createVariable.assert_any_call("time", "i8", ("time"))

    # Check creation of groups and variables
    mock_ncfile.createGroup.assert_any_call("truth")
    mock_ncfile.createGroup.assert_any_call("prediction")
    mock_ncfile.createGroup.assert_any_call("input")

    # Assert variables in truth and prediction groups
    mock_ncfile.createGroup("truth").createVariable.assert_called_with(
        "var1", "f", dimensions=("ensemble", "time", "y", "x")
    )
    mock_ncfile.createGroup("prediction").createVariable.assert_called_with(
        "var1", "f", dimensions=("ensemble", "time", "y", "x")
    )
    mock_ncfile.createGroup("input").createVariable.assert_called_with(
        "var1", "f", dimensions=("ensemble", "time", "y", "x")
    )


@import_or_fail("cftime")
def test_write_input(mock_ncfile, pytestconfig):

    from physicsnemo.utils.corrdiff import NetCDFWriter

    lat = np.array([[1.0, 2.0], [3.0, 4.0]])
    lon = np.array([[5.0, 6.0], [7.0, 8.0]])
    input_channels = []
    output_channels = []

    writer = NetCDFWriter(mock_ncfile, lat, lon, input_channels, output_channels)

    # Mock input channel
    channel_name = "var1"
    time_index = 0
    val = np.array([[1.0, 2.0], [3.0, 4.0]])

    writer.write_input(channel_name, time_index, val)

    # Assert write_input method was called correctly
    mock_ncfile["input"][channel_name][time_index] = val
    mock_ncfile["input"][channel_name].__setitem__.assert_called_with(time_index, val)


@import_or_fail("cftime")
def test_write_truth(mock_ncfile, pytestconfig):

    from physicsnemo.utils.corrdiff import NetCDFWriter

    lat = np.array([[1.0, 2.0], [3.0, 4.0]])
    lon = np.array([[5.0, 6.0], [7.0, 8.0]])
    input_channels = []
    output_channels = []

    writer = NetCDFWriter(mock_ncfile, lat, lon, input_channels, output_channels)

    # Mock truth channel
    channel_name = "var1"
    time_index = 0
    val = np.array([[1.0, 2.0], [3.0, 4.0]])

    writer.write_truth(channel_name, time_index, val)

    # Assert write_truth method was called correctly
    mock_ncfile["truth"][channel_name][time_index] = val
    mock_ncfile["truth"][channel_name].__setitem__.assert_called_with(time_index, val)


@import_or_fail("cftime")
def test_write_prediction(mock_ncfile, pytestconfig):

    from physicsnemo.utils.corrdiff import NetCDFWriter

    lat = np.array([[1.0, 2.0], [3.0, 4.0]])
    lon = np.array([[5.0, 6.0], [7.0, 8.0]])
    input_channels = []
    output_channels = []

    writer = NetCDFWriter(mock_ncfile, lat, lon, input_channels, output_channels)

    # Mock prediction channel
    channel_name = "var1"
    time_index = 0
    ensemble_index = 0
    val = np.array([[1.0, 2.0], [3.0, 4.0]])

    writer.write_prediction(channel_name, time_index, ensemble_index, val)

    # Assert write_prediction method was called correctly
    mock_ncfile["prediction"][channel_name][ensemble_index, time_index] = val
    mock_ncfile["prediction"][channel_name].__setitem__.assert_called_with(
        (ensemble_index, time_index), val
    )


@import_or_fail("cftime")
def test_write_time(mock_ncfile, pytestconfig):

    from physicsnemo.utils.corrdiff import NetCDFWriter

    lat = np.array([[1.0, 2.0], [3.0, 4.0]])
    lon = np.array([[5.0, 6.0], [7.0, 8.0]])
    input_channels = []
    output_channels = []

    writer = NetCDFWriter(mock_ncfile, lat, lon, input_channels, output_channels)

    # Mock time write
    time_index = 0
    time = datetime.datetime(2024, 1, 1, 0, 0, 0)

    with patch("cftime.date2num") as mock_date2num:
        mock_date2num.return_value = 0  # Mocked value
        writer.write_time(time_index, time)
        mock_date2num.assert_called_with(
            time, mock_ncfile["time"].units, mock_ncfile["time"].calendar
        )
        mock_ncfile["time"][time_index] = 0
        mock_ncfile["time"].__setitem__.assert_called_with(time_index, 0)
