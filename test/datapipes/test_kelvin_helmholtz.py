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

from typing import Tuple

import pytest
import torch
from pytest_utils import import_or_fail

from . import common

Tensor = torch.Tensor


@import_or_fail("warp")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_kelvin_helmholtz_2d_constructor(device, pytestconfig):

    from physicsnemo.datapipes.benchmarks.kelvin_helmholtz import KelvinHelmholtz2D

    # construct data pipe
    datapipe = KelvinHelmholtz2D(
        resolution=32,
        batch_size=1,
        seq_length=2,
        nr_perturbation_freq=5,
        perturbation_range=0.1,
        nr_snapshots=4,
        iteration_per_snapshot=8,
        gamma=5.0 / 3.0,
        normaliser={"density": (0, 1), "velocity": (0, 1), "pressure": (0, 1)},
        device=device,
    )

    # iterate datapipe is iterable
    assert common.check_datapipe_iterable(datapipe)


@import_or_fail("warp")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_kelvin_helmholtz_2d_device(device, pytestconfig):

    from physicsnemo.datapipes.benchmarks.kelvin_helmholtz import KelvinHelmholtz2D

    # construct data pipe
    datapipe = KelvinHelmholtz2D(
        resolution=32,
        batch_size=1,
        seq_length=2,
        nr_perturbation_freq=5,
        perturbation_range=0.1,
        nr_snapshots=4,
        iteration_per_snapshot=32,
        gamma=5.0 / 3.0,
        normaliser={"density": (0, 1), "velocity": (0, 1), "pressure": (0, 1)},
        device=device,
    )

    # iterate datapipe is iterable
    for data in datapipe:
        assert common.check_datapipe_device(data["density"], device)
        assert common.check_datapipe_device(data["velocity"], device)
        assert common.check_datapipe_device(data["pressure"], device)
        break


@import_or_fail("warp")
@pytest.mark.parametrize("resolution", [32, 64])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("seq_length", [2, 3])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_kelvin_helmholtz_2d_shape(
    resolution, batch_size, seq_length, device, pytestconfig
):

    from physicsnemo.datapipes.benchmarks.kelvin_helmholtz import KelvinHelmholtz2D

    # construct data pipe
    datapipe = KelvinHelmholtz2D(
        resolution=resolution,
        batch_size=batch_size,
        seq_length=seq_length,
        nr_perturbation_freq=5,
        perturbation_range=0.1,
        nr_snapshots=4,
        iteration_per_snapshot=8,
        gamma=5.0 / 3.0,
        normaliser={"density": (0, 1), "velocity": (0, 1), "pressure": (0, 1)},
        device=device,
    )

    # test single sample
    for data in datapipe:
        rho = data["density"]
        vel = data["velocity"]
        p = data["pressure"]

        # check batch size
        assert common.check_batch_size([rho, vel, p], batch_size)

        # check sequence length
        assert common.check_seq_length([rho, vel, p], seq_length, axis=1)

        # check sequence length
        assert common.check_channels([rho, p], 1, axis=2)
        assert common.check_channels(vel, 2, axis=2)

        # check grid dims
        assert common.check_grid([rho, vel, p], (resolution, resolution), axis=(3, 4))
        break


@import_or_fail("warp")
@pytest.mark.parametrize("device", ["cuda:0"])
def test_kelvin_helmholtz_cudagraphs(device, pytestconfig):

    from physicsnemo.datapipes.benchmarks.kelvin_helmholtz import KelvinHelmholtz2D

    # Preprocess function to convert dataloader output into Tuple of tensors
    def input_fn(data) -> Tuple[Tensor, ...]:
        return (data["density"], data["velocity"], data["pressure"])

    # construct data pipe
    datapipe = KelvinHelmholtz2D(
        resolution=32,
        batch_size=1,
        seq_length=2,
        nr_perturbation_freq=5,
        perturbation_range=0.1,
        nr_snapshots=4,
        iteration_per_snapshot=8,
        gamma=5.0 / 3.0,
        normaliser={"density": (0, 1), "velocity": (0, 1), "pressure": (0, 1)},
        device=device,
    )

    assert common.check_cuda_graphs(datapipe, input_fn)
