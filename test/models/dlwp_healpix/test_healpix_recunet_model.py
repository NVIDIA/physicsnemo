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
# ruff: noqa: E402
import os
import sys

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import common
import pytest
import torch
from graphcast.utils import fix_random_seeds
from pytest_utils import import_or_fail

from physicsnemo.models.dlwp_healpix import HEALPixRecUNet

omegaconf = pytest.importorskip("omegaconf")


@pytest.fixture
def conv_next_block_dict(in_channels=3, out_channels=1):
    activation_block = {
        "_target_": "physicsnemo.models.layers.activations.CappedGELU",
        "cap_value": 10,
    }
    conv_block = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.ConvNeXtBlock",
        "in_channels": in_channels,
        "out_channels": out_channels,
        "activation": activation_block,
        "kernel_size": 3,
        "dilation": 1,
        "upscale_factor": 4,
        "_recursive_": True,
    }
    return conv_block


@pytest.fixture
def down_sampling_block_dict():
    down_sampling_block = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.AvgPool",
        "pooling": 2,
    }
    return down_sampling_block


@pytest.fixture
def encoder_dict(conv_next_block_dict, down_sampling_block_dict, recurrent_block_dict):
    encoder = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.UNetEncoder",
        "conv_block": conv_next_block_dict,
        "down_sampling_block": down_sampling_block_dict,
        "recurrent_block": recurrent_block_dict,
        "_recursive_": False,
        "n_channels": [136, 68, 34],
        "dilations": [1, 2, 4],
    }
    return encoder


@pytest.fixture
def up_sampling_block_dict(in_channels=3, out_channels=1):
    """Block dict fixture."""
    activation_block = {
        "_target_": "physicsnemo.models.layers.activations.CappedGELU",
        "cap_value": 10,
    }
    up_sampling_block = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.TransposedConvUpsample",
        "in_channels": in_channels,
        "out_channels": out_channels,
        "activation": activation_block,
        "upsampling": 2,
    }
    return omegaconf.DictConfig(up_sampling_block)


@pytest.fixture
def output_layer_dict(in_channels=3, out_channels=2):
    output_layer = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.BasicConvBlock",
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": 1,
        "dilation": 1,
        "n_layers": 1,
    }
    return omegaconf.DictConfig(output_layer)


@pytest.fixture
def recurrent_block_dict(in_channels=3):
    recurrent_block = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.ConvGRUBlock",
        "in_channels": in_channels,
        "kernel_size": 1,
        "_recursive_": False,
    }
    return omegaconf.DictConfig(recurrent_block)


@pytest.fixture
def decoder_dict(
    conv_next_block_dict,
    up_sampling_block_dict,
    output_layer_dict,
    recurrent_block_dict,
):
    decoder = {
        "_target_": "physicsnemo.models.dlwp_healpix_layers.UNetDecoder",
        "conv_block": conv_next_block_dict,
        "up_sampling_block": up_sampling_block_dict,
        "recurrent_block": recurrent_block_dict,
        "output_layer": output_layer_dict,
        "_recursive_": False,
        "n_channels": [34, 68, 136],
        "dilations": [4, 2, 1],
    }
    return omegaconf.DictConfig(decoder)


@pytest.fixture
def test_data():
    # create dummy data
    def generate_test_data(
        batch_size=8, time_dim=1, channels=7, img_size=16, device="cpu"
    ):
        test_data = torch.randn(batch_size, 12, time_dim, channels, img_size, img_size)

        return test_data.to(device)

    return generate_test_data


@pytest.fixture
def constant_data():
    # create dummy data
    def generate_constant_data(channels=2, img_size=16, device="cpu"):
        constants = torch.randn(12, channels, img_size, img_size)

        return constants.to(device)

    return generate_constant_data


@pytest.fixture
def insolation_data():
    # create dummy data
    def generate_insolation_data(batch_size=8, time_dim=1, img_size=16, device="cpu"):
        insolation = torch.randn(batch_size, 12, time_dim, 1, img_size, img_size)

        return insolation.to(device)

    return generate_insolation_data


@import_or_fail("omegaconf")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_HEALPixRecUNet_initialize(device, encoder_dict, decoder_dict, pytestconfig):
    in_channels = 3
    out_channels = 3
    n_constants = 1
    decoder_input_channels = 1
    input_time_dim = 2
    output_time_dim = 4

    model = HEALPixRecUNet(
        encoder=encoder_dict,
        decoder=decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
    ).to(device)
    assert isinstance(model, HEALPixRecUNet)

    # test fail case for bad input and output time dims
    with pytest.raises(
        ValueError, match=("'output_time_dim' must be a multiple of 'input_time_dim'")
    ):
        model = HEALPixRecUNet(
            encoder=encoder_dict,
            decoder=decoder_dict,
            input_channels=in_channels,
            output_channels=out_channels,
            n_constants=n_constants,
            decoder_input_channels=decoder_input_channels,
            input_time_dim=2,
            output_time_dim=3,
        ).to(device)

    # test fail case for couplings with no constants or decoder input channels
    with pytest.raises(
        NotImplementedError,
        match=("support for coupled models with no constant field"),
    ):
        model = HEALPixRecUNet(
            encoder=encoder_dict,
            decoder=decoder_dict,
            input_channels=in_channels,
            output_channels=out_channels,
            input_time_dim=2,
            output_time_dim=3,
            decoder_input_channels=2,
            n_constants=0,
            couplings=["t2m", "v10m"],
        ).to(device)

    # test fail case for couplings with no decoder input channels
    with pytest.raises(
        NotImplementedError,
        match=("support for coupled models with no decoder inputs"),
    ):
        model = HEALPixRecUNet(
            encoder=encoder_dict,
            decoder=decoder_dict,
            input_channels=in_channels,
            output_channels=out_channels,
            input_time_dim=2,
            output_time_dim=3,
            decoder_input_channels=0,
            n_constants=2,
            couplings=["t2m", "v10m"],
        ).to(device)

    with pytest.raises(
        NotImplementedError, match=("support for coupled models with no decoder")
    ):
        model = HEALPixRecUNet(
            encoder=encoder_dict,
            decoder=decoder_dict,
            input_channels=in_channels,
            output_channels=out_channels,
            input_time_dim=2,
            output_time_dim=3,
            decoder_input_channels=0,
            n_constants=2,
            couplings=["t2m", "v10m"],
        ).to(device)

    with pytest.raises(
        NotImplementedError,
        match=("support for models with no constant fields and no decoder"),
    ):
        model = HEALPixRecUNet(
            encoder=encoder_dict,
            decoder=decoder_dict,
            input_channels=in_channels,
            output_channels=out_channels,
            input_time_dim=2,
            output_time_dim=3,
            decoder_input_channels=0,
            n_constants=0,
        ).to(device)

    del model
    torch.cuda.empty_cache()


@import_or_fail("omegaconf")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_HEALPixRecUNet_integration_steps(
    device, encoder_dict, decoder_dict, pytestconfig
):
    in_channels = 2
    out_channels = 2
    n_constants = 1
    decoder_input_channels = 0
    input_time_dim = 2
    output_time_dim = 4

    model = HEALPixRecUNet(
        encoder=encoder_dict,
        decoder=decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
    ).to(device)

    assert model.integration_steps == output_time_dim // input_time_dim
    del model
    torch.cuda.empty_cache()


@import_or_fail("omegaconf")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@torch.no_grad()
def test_HEALPixRecUNet_reset(
    device,
    encoder_dict,
    decoder_dict,
    test_data,
    insolation_data,
    constant_data,
    pytestconfig,
):
    # create a smaller version of the dlwp healpix model
    in_channels = 2
    out_channels = 2
    n_constants = 2
    decoder_input_channels = 1
    input_time_dim = 2
    output_time_dim = 4
    size = 16

    fix_random_seeds(seed=42)
    x = test_data(
        time_dim=2 * input_time_dim, channels=in_channels, img_size=size, device=device
    )
    decoder_inputs = insolation_data(
        time_dim=2 * output_time_dim, img_size=size, device=device
    )
    constants = constant_data(channels=n_constants, img_size=size, device=device)
    inputs = [x, decoder_inputs, constants]

    model = HEALPixRecUNet(
        encoder=encoder_dict,
        decoder=decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        enable_healpixpad=True,
        delta_time="6h",
    ).to(device)

    out_var = model(inputs)
    model.reset()

    assert common.compare_output(out_var, model(inputs))

    del model, inputs, out_var
    torch.cuda.empty_cache()


@import_or_fail("omegaconf")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@torch.no_grad()
def test_HEALPixRecUNet_forward(
    device,
    encoder_dict,
    decoder_dict,
    test_data,
    insolation_data,
    constant_data,
    pytestconfig,
):
    # create a smaller version of the dlwp healpix model
    in_channels = 2
    out_channels = 2
    n_constants = 2
    decoder_input_channels = 1
    input_time_dim = 2
    output_time_dim = 4
    batch_size = 2
    size = 16

    fix_random_seeds(seed=42)
    x = test_data(
        batch_size=batch_size,
        time_dim=2 * input_time_dim,
        channels=in_channels,
        img_size=size,
        device=device,
    )
    decoder_inputs = insolation_data(
        batch_size=batch_size,
        time_dim=2 * output_time_dim,
        img_size=size,
        device=device,
    )
    constants = constant_data(channels=n_constants, img_size=size, device=device)
    inputs = [x, decoder_inputs, constants]

    model = HEALPixRecUNet(
        encoder=encoder_dict,
        decoder=decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        enable_healpixpad=True,
        delta_time="6h",
        reset_cycle="6h",
    ).to(device)

    # one forward step to initialize recurrent states
    output = model(inputs)

    expected_shape = [batch_size, 12, output_time_dim, out_channels, size, size]
    assert list(output.shape) == expected_shape

    assert common.validate_forward_accuracy(
        model,
        (inputs,),
        file_name="dlwp_healpix.pth",
        rtol=1e-2,
    )

    output = model(inputs, output_only_last=True)
    expected_shape = [batch_size, 12, input_time_dim, out_channels, size, size]
    assert list(output.shape) == expected_shape

    # no decoder inputs
    inputs = [x, constants]
    model = HEALPixRecUNet(
        encoder=encoder_dict,
        decoder=decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=0,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        enable_healpixpad=True,
        delta_time="6h",
    ).to(device)

    # one forward step to initialize recurrent states
    model(inputs)

    assert common.validate_forward_accuracy(
        model,
        (inputs,),
        file_name="dlwp_healpix_const.pth",
        rtol=1e-2,
    )

    # no constants
    inputs = [x, decoder_inputs]
    model = HEALPixRecUNet(
        encoder=encoder_dict,
        decoder=decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=0,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        enable_healpixpad=True,
        delta_time="6h",
    ).to(device)

    # one forward step to initialize recurrent states
    model(inputs)

    assert common.validate_forward_accuracy(
        model,
        (inputs,),
        file_name="dlwp_healpix_decoder.pth",
        rtol=1e-2,
    )

    del model, inputs
    torch.cuda.empty_cache()
