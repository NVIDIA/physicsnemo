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

"""Self-contained tests for the HEALPix couplers.

``ConstantCoupler.set_coupled_fields`` no longer validates the batch dimension
of the provided fields against the configured ``batch_size`` -- the caller is
now responsible for providing consistently shaped fields, and the coupler sizes
its buffer from the field it is given. These tests exercise that behavior with
tiny in-memory datasets so they run without the NFS-backed test dataset.
"""

import numpy as np
import pytest
import torch  # noqa: E402

from physicsnemo.datapipes.healpix.couplers import ConstantCoupler  # noqa: E402
from test.conftest import requires_module

_FACE, _HEIGHT, _WIDTH = 2, 3, 4


def _make_coupler_dataset(channel_in, n_time=8):
    """Minimal dataset; the coupler only reads ``inputs.shape[2:]`` from it to
    determine its spatial dimensions."""
    xr = pytest.importorskip("xarray")
    pd = pytest.importorskip("pandas")

    return xr.Dataset(
        data_vars={
            "inputs": (
                ("time", "channel_in", "face", "height", "width"),
                np.zeros(
                    (n_time, len(channel_in), _FACE, _HEIGHT, _WIDTH),
                    dtype="float32",
                ),
            )
        },
        coords={
            "time": pd.date_range("1979-01-01", periods=n_time, freq="3h"),
            "channel_in": list(channel_in),
            "face": np.arange(_FACE),
            "height": np.arange(_HEIGHT),
            "width": np.arange(_WIDTH),
        },
    )


def _make_coupler(batch_size):
    ds = _make_coupler_dataset(["c0", "c1", "x"])
    coupler = ConstantCoupler(
        dataset=ds,
        batch_size=batch_size,
        variables=["c0", "c1"],
        input_times=["0h"],
        input_time_dim=1,
        output_time_dim=1,
        presteps=0,
    )
    # normally assigned by setup_coupling(); the two coupled channels map to the
    # two coupled variables.
    coupler.coupled_channel_indices = [0, 1]
    return coupler


def _coupled_fields(batch):
    """Coupled field in the expected ``[B, F, T, C, H, W]`` layout.

    ``ConstantCoupler`` broadcasts the first time step and sizes its buffer from
    the provided field, so the caller is responsible for providing a
    consistently shaped field.

    The time dimension is deliberately distinct from ``batch`` so the buffer's
    batch axis (index 1 after the coupler's permute) is pinned to the batch and
    not accidentally satisfied by the time axis.
    """
    n_time = batch + 3
    return torch.rand(batch, _FACE, n_time, len([0, 1]), _HEIGHT, _WIDTH)


@requires_module("xarray")
@requires_module("pandas")
def test_set_coupled_fields_adapts_to_provided_batch_size():
    """The old ``batch_size`` mismatch ``ValueError`` was removed: providing a
    field whose batch differs from the configured ``batch_size`` no longer
    raises, and the buffer is sized from the provided field."""
    configured_batch_size = 2
    coupler = _make_coupler(batch_size=configured_batch_size)

    provided_batch = configured_batch_size * 2
    coupler.set_coupled_fields(_coupled_fields(provided_batch))

    assert coupler.coupled_mode
    # buffer batch dim reflects the provided field, not the configured batch_size
    assert coupler.preset_coupled_fields.shape[1] == provided_batch
    # in coupled mode, construct_integrated_couplings returns the preset buffer
    assert coupler.construct_integrated_couplings() is coupler.preset_coupled_fields


@requires_module("xarray")
@requires_module("pandas")
def test_set_coupled_fields_matches_configured_batch_size():
    """Sanity: a field whose batch equals the configured batch_size still
    works."""
    coupler = _make_coupler(batch_size=3)
    coupler.set_coupled_fields(_coupled_fields(3))
    assert coupler.preset_coupled_fields.shape[1] == 3
