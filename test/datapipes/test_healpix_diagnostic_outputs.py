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

"""Self-contained tests for HEALPix dataloader handling of models with extra
(diagnostic) outputs.

A model has *extra outputs* when the number of target channels does not match
the number of prognostic input channels (i.e. ``channel_in`` minus any coupled
fields). In that case the input scaling must be selected from ``channel_in``
(rather than ``channel_out``) so its size matches the loaded input array, and
the coupled portion of that scaling must be sliced off before normalizing the
(prognostic-only) inputs.

These tests build tiny in-memory xarray datasets so they exercise the real
``TimeSeriesDataset``/``CoupledTimeSeriesDataset`` code paths without requiring
the NFS-backed test dataset.
"""

import numpy as np
import pytest

from physicsnemo.datapipes.healpix.coupledtimeseries_dataset import (  # noqa: E402
    CoupledTimeSeriesDataset,
)
from physicsnemo.datapipes.healpix.timeseries_dataset import (  # noqa: E402
    TimeSeriesDataset,
)
from test.conftest import requires_module

# per-variable (mean, std); values are distinct so we can tell which scaling
# entry (and thus which channel axis) was selected.
_SCALING = {
    "a": (1.0, 2.0),
    "b": (3.0, 4.0),
    "cpl": (10.0, 5.0),
    "diag": (7.0, 8.0),
}

_FACE, _HEIGHT, _WIDTH = 2, 3, 4


def _scaling_config(variables):
    omegaconf = pytest.importorskip("omegaconf")
    return omegaconf.OmegaConf.create(
        {v: {"mean": _SCALING[v][0], "std": _SCALING[v][1]} for v in variables}
    )


def _make_dataset(channel_in, channel_out, n_time=8):
    """Build a minimal classic-style HEALPix dataset.

    Input channel ``i`` is filled with the constant ``i + 1`` so that the
    normalized value is deterministic and easy to assert against.
    """
    pd = pytest.importorskip("pandas")
    xr = pytest.importorskip("xarray")
    time = pd.date_range("1979-01-01", periods=n_time, freq="3h")

    inputs = np.empty(
        (n_time, len(channel_in), _FACE, _HEIGHT, _WIDTH), dtype="float32"
    )
    for i in range(len(channel_in)):
        inputs[:, i] = float(i + 1)

    targets = np.zeros(
        (n_time, len(channel_out), _FACE, _HEIGHT, _WIDTH), dtype="float32"
    )

    return xr.Dataset(
        data_vars={
            "inputs": (
                ("time", "channel_in", "face", "height", "width"),
                inputs,
            ),
            "targets": (
                ("time", "channel_out", "face", "height", "width"),
                targets,
            ),
        },
        coords={
            "time": time,
            "channel_in": list(channel_in),
            "channel_out": list(channel_out),
            "face": np.arange(_FACE),
            "height": np.arange(_HEIGHT),
            "width": np.arange(_WIDTH),
        },
    )


def _constant_coupler_config(variables, batch_size):
    return [
        {
            "coupler": "ConstantCoupler",
            "params": {
                "batch_size": batch_size,
                "variables": list(variables),
                "input_times": ["0h"],
                "input_time_dim": 1,
                "output_time_dim": 1,
                "presteps": 0,
                "prepared_coupled_data": True,
            },
        }
    ]


def _expected_normalized(channel):
    """Normalized value of the constant-filled input for ``channel``.

    Channel ``a`` is the first input channel (constant 1.0), ``b`` the second
    (constant 2.0).
    """
    value = {"a": 1.0, "b": 2.0}[channel]
    mean, std = _SCALING[channel]
    return (value - mean) / std


@requires_module("xarray")  # for dataset creation via _make_dataset
@requires_module("pandas")  # for dataset creation via _make_dataset
@requires_module("omegaconf")  # for scaling config via _scaling_config
def test_timeseries_scaling_uses_channel_out_without_diagnostics():
    """No diagnostic outputs: input scaling is selected from channel_out."""
    ds = _make_dataset(["a", "b"], ["a", "b"])
    dset = TimeSeriesDataset(
        ds,
        scaling=_scaling_config(["a", "b"]),
        input_time_dim=1,
        output_time_dim=1,
        data_time_step="3h",
        time_step="3h",
        batch_size=2,
    )
    assert dset.input_scaling["mean"].shape[1] == 2
    np.testing.assert_allclose(dset.input_scaling["mean"].ravel(), [1.0, 3.0])
    np.testing.assert_allclose(dset.input_scaling["std"].ravel(), [2.0, 4.0])


@requires_module("xarray")  # for dataset creation via _make_dataset
@requires_module("pandas")  # for dataset creation via _make_dataset
@requires_module("omegaconf")  # for scaling config via _scaling_config
def test_timeseries_scaling_uses_channel_in_with_diagnostics():
    """Diagnostic output ('diag' in channel_out only) forces channel_in
    scaling, so its size matches the loaded input array."""
    ds = _make_dataset(["a", "b"], ["a", "b", "diag"])
    dset = TimeSeriesDataset(
        ds,
        scaling=_scaling_config(["a", "b", "diag"]),
        input_time_dim=1,
        output_time_dim=1,
        data_time_step="3h",
        time_step="3h",
        batch_size=2,
    )
    # input scaling covers channel_in (a, b) -- not channel_out (a, b, diag)
    assert dset.input_scaling["mean"].shape[1] == 2
    np.testing.assert_allclose(dset.input_scaling["mean"].ravel(), [1.0, 3.0])
    np.testing.assert_allclose(dset.input_scaling["std"].ravel(), [2.0, 4.0])
    # target scaling still covers all output channels
    assert dset.target_scaling["mean"].shape[1] == 3


@requires_module("xarray")  # for dataset creation via _make_dataset
@requires_module("pandas")  # for dataset creation via _make_dataset
@requires_module("omegaconf")  # for scaling config via _scaling_config
def test_timeseries_getitem_normalizes_with_channel_in_scaling():
    """With diagnostic outputs, __getitem__ normalizes the full input array
    against the channel_in scaling and returns channel_in-many channels."""
    ds = _make_dataset(["a", "b"], ["a", "b", "diag"])
    dset = TimeSeriesDataset(
        ds,
        scaling=_scaling_config(["a", "b", "diag"]),
        input_time_dim=1,
        output_time_dim=1,
        data_time_step="3h",
        time_step="3h",
        batch_size=2,
        add_insolation=False,
    )
    inputs_result, _ = dset[0]
    inputs = inputs_result[0]  # [B, F, T, C, H, W]
    assert inputs.shape[3] == 2
    np.testing.assert_allclose(inputs[:, :, :, 0], _expected_normalized("a"))
    np.testing.assert_allclose(inputs[:, :, :, 1], _expected_normalized("b"))


@requires_module("xarray")  # for dataset creation via _make_dataset
@requires_module("pandas")  # for dataset creation via _make_dataset
@requires_module("omegaconf")  # for scaling config via _scaling_config
def test_coupled_getitem_slices_coupled_scaling_with_diagnostics():
    """Coupled model with an extra output: input scaling is taken from
    channel_in (prognostic + coupled) and the coupled tail is sliced off so the
    prognostic inputs are normalized correctly."""
    ds = _make_dataset(["a", "b", "cpl"], ["a", "b", "diag"])
    dset = CoupledTimeSeriesDataset(
        ds,
        scaling=_scaling_config(["a", "b", "cpl", "diag"]),
        input_variables=["a", "b"],
        output_variables=["a", "b", "diag"],
        input_time_dim=1,
        output_time_dim=1,
        data_time_step="3h",
        time_step="3h",
        batch_size=2,
        add_insolation=False,
        couplings=_constant_coupler_config(["cpl"], batch_size=2),
    )
    # diagnostic path -> input scaling covers channel_in (a, b, cpl)
    assert dset.input_scaling["mean"].shape[1] == 3

    inputs_result, _ = dset[0]
    inputs = inputs_result[0]  # [B, F, T, C, H, W]
    # only the prognostic inputs are returned, normalized by their own scaling
    assert inputs.shape[3] == 2
    np.testing.assert_allclose(inputs[:, :, :, 0], _expected_normalized("a"))
    np.testing.assert_allclose(inputs[:, :, :, 1], _expected_normalized("b"))


@requires_module("xarray")  # for dataset creation via _make_dataset
@requires_module("pandas")  # for dataset creation via _make_dataset
@requires_module("omegaconf")  # for scaling config via _scaling_config
def test_coupled_getitem_without_diagnostics_uses_channel_out_scaling():
    """Coupled model with no extra outputs: channel_out scaling is used and no
    slicing is applied."""
    ds = _make_dataset(["a", "b", "cpl"], ["a", "b"])
    dset = CoupledTimeSeriesDataset(
        ds,
        scaling=_scaling_config(["a", "b", "cpl"]),
        input_variables=["a", "b"],
        output_variables=["a", "b"],
        input_time_dim=1,
        output_time_dim=1,
        data_time_step="3h",
        time_step="3h",
        batch_size=2,
        add_insolation=False,
        couplings=_constant_coupler_config(["cpl"], batch_size=2),
    )
    # non-diagnostic path -> input scaling covers channel_out (a, b)
    assert dset.input_scaling["mean"].shape[1] == 2

    inputs_result, _ = dset[0]
    inputs = inputs_result[0]
    assert inputs.shape[3] == 2
    np.testing.assert_allclose(inputs[:, :, :, 0], _expected_normalized("a"))
    np.testing.assert_allclose(inputs[:, :, :, 1], _expected_normalized("b"))
