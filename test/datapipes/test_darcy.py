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
def test_darcy_2d_constructor(device, pytestconfig):

    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

    # construct data pipe
    datapipe = Darcy2D(
        resolution=64,
        batch_size=1,
        nr_permeability_freq=5,
        max_permeability=2.0,
        min_permeability=0.5,
        max_iterations=300,
        convergence_threshold=1e-4,
        iterations_per_convergence_check=5,
        nr_multigrids=4,
        normaliser={"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
        device=device,
    )

    # iterate datapipe is iterable
    assert common.check_datapipe_iterable(datapipe)


@import_or_fail("warp")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_darcy_2d_device(device, pytestconfig):

    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

    # construct data pipe
    datapipe = Darcy2D(
        resolution=64,
        batch_size=1,
        nr_permeability_freq=5,
        max_permeability=2.0,
        min_permeability=0.5,
        max_iterations=300,
        convergence_threshold=1e-4,
        iterations_per_convergence_check=5,
        nr_multigrids=4,
        normaliser={"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
        device=device,
    )

    # iterate datapipe is iterable
    for data in datapipe:
        assert common.check_datapipe_device(data["permeability"], device)
        assert common.check_datapipe_device(data["darcy"], device)
        break


@import_or_fail("warp")
@pytest.mark.parametrize("resolution", [128, 64])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_darcy_2d_shape(resolution, batch_size, device, pytestconfig):

    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

    # construct data pipe
    datapipe = Darcy2D(
        resolution=resolution,
        batch_size=batch_size,
        nr_permeability_freq=5,
        max_permeability=2.0,
        min_permeability=0.5,
        max_iterations=300,
        convergence_threshold=1e-4,
        iterations_per_convergence_check=5,
        nr_multigrids=3,
        normaliser={"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
        device=device,
    )

    # test single sample
    for data in datapipe:
        permeability = data["permeability"]
        darcy = data["darcy"]

        # check batch size
        assert common.check_batch_size([permeability, darcy], batch_size)

        # check channels
        assert common.check_channels([permeability, darcy], 1, axis=1)

        # check grid dims
        assert common.check_grid(
            [permeability, darcy], (resolution, resolution), axis=(2, 3)
        )
        break


@import_or_fail("warp")
@pytest.mark.parametrize("device", ["cuda:0"])
def test_darcy_cudagraphs(device, pytestconfig):

    from physicsnemo.datapipes.benchmarks.darcy import Darcy2D

    # Preprocess function to convert dataloader output into Tuple of tensors
    def input_fn(data) -> Tuple[Tensor, ...]:
        return (data["permeability"], data["darcy"])

    # construct data pipe
    datapipe = Darcy2D(
        resolution=64,
        batch_size=1,
        nr_permeability_freq=5,
        max_permeability=2.0,
        min_permeability=0.5,
        max_iterations=300,
        convergence_threshold=1e-4,
        iterations_per_convergence_check=5,
        nr_multigrids=4,
        normaliser={"permeability": (0.0, 1.0), "darcy": (0.0, 1.0)},
        device=device,
    )

    assert common.check_cuda_graphs(datapipe, input_fn)
