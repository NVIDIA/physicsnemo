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
# ruff: noqa: E402

import omegaconf
import pytest
import torch

from physicsnemo.models.dlwp_healpix import HEALPixUNet
from test import common
from test.models.graphcast.utils import fix_random_seeds


@pytest.fixture
def conv_next_block_dict(in_channels=3, out_channels=1):
    activation_block = {
        "_target_": "physicsnemo.nn.CappedGELU",
        "cap_value": 10,
    }
    conv_block = {
        "_target_": "physicsnemo.models.dlwp_healpix.layers.ConvNeXtBlock",
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
        "_target_": "physicsnemo.nn.HEALPixAvgPool",
        "pooling": 2,
    }
    return down_sampling_block


@pytest.fixture
def up_sampling_block_dict(in_channels=3, out_channels=1):
    """Upsampling dict fixture."""
    activation_block = {
        "_target_": "physicsnemo.nn.CappedGELU",
        "cap_value": 10,
    }
    up_sampling_block = {
        "_target_": "physicsnemo.models.dlwp_healpix.layers.TransposedConvUpsample",
        "in_channels": in_channels,
        "out_channels": out_channels,
        "activation": activation_block,
        "upsampling": 2,
    }
    return omegaconf.DictConfig(up_sampling_block)


@pytest.fixture
def output_layer_dict(in_channels=3, out_channels=2):
    output_layer = {
        "_target_": "physicsnemo.models.dlwp_healpix.layers.BasicConvBlock",
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": 1,
        "dilation": 1,
        "n_layers": 1,
    }
    return omegaconf.DictConfig(output_layer)


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


@pytest.fixture
def cln_conv_block_dict(channels=4, cond_dim=8):
    """``Multi_SymmetricConvNeXtBlock`` config with an always-on conditional
    layer norm, used to exercise the ``conditions_cln`` wiring through
    ``HEALPixUNet.forward``.
    """
    return omegaconf.DictConfig(
        {
            "_target_": "physicsnemo.models.dlwp_healpix.layers.Multi_SymmetricConvNeXtBlock",
            "in_channels": channels,
            "conditional_layer_norm": {
                "_target_": "physicsnemo.models.dlwp_healpix.layers.ConditionalLayerNorm",
                "_partial_": True,
                "condition_shape": cond_dim,
            },
        }
    )


@pytest.fixture
def cln_up_sampling_block_dict(channels=4):
    up_sampling_block = {
        "_target_": "physicsnemo.models.dlwp_healpix.layers.TransposedConvUpsample",
        "in_channels": channels,
        "out_channels": channels,
        "activation": {"_target_": "physicsnemo.nn.CappedGELU", "cap_value": 10},
        "upsampling": 2,
    }
    return omegaconf.DictConfig(up_sampling_block)


@pytest.fixture
def cln_output_layer_dict(channels=4, out_channels=2):
    output_layer = {
        "_target_": "physicsnemo.models.dlwp_healpix.layers.BasicConvBlock",
        "in_channels": channels,
        "out_channels": out_channels,
        "kernel_size": 1,
        "dilation": 1,
        "n_layers": 1,
    }
    return omegaconf.DictConfig(output_layer)


@pytest.fixture
def cln_unet_encoder_dict(cln_conv_block_dict, down_sampling_block_dict):
    """CLN-enabled encoder dict fixture (small, self-contained channel count)."""
    encoder = {
        "_target_": "physicsnemo.models.dlwp_healpix.layers.UNetEncoder",
        "conv_block": cln_conv_block_dict,
        "down_sampling_block": down_sampling_block_dict,
        "_recursive_": False,
        "n_channels": [4, 4, 4],
        "dilations": [1, 2, 4],
    }
    return encoder


@pytest.fixture
def cln_unet_decoder_dict(
    cln_conv_block_dict, cln_up_sampling_block_dict, cln_output_layer_dict
):
    """CLN-enabled decoder dict fixture (small, self-contained channel count)."""
    decoder = {
        "_target_": "physicsnemo.models.dlwp_healpix.layers.UNetDecoder",
        "conv_block": cln_conv_block_dict,
        "up_sampling_block": cln_up_sampling_block_dict,
        "output_layer": cln_output_layer_dict,
        "_recursive_": False,
        "n_channels": [4, 4, 4],
        "dilations": [4, 2, 1],
    }
    return omegaconf.DictConfig(decoder)


@pytest.fixture
def coupling_data():
    # create dummy coupling data: a list (one entry per required coupling
    # index) of (B, C_coupled, F, H, W) tensors, matching the shape expected
    # by ``HEALPixUNet._reshape_inputs`` when ``couplings_time_first=True``.
    def generate_coupling_data(
        steps=1, channels=4, batch_size=8, img_size=16, device="cpu"
    ):
        return [
            torch.randn(batch_size, channels, 12, img_size, img_size).to(device)
            for _ in range(steps)
        ]

    return generate_coupling_data


@pytest.fixture
def unet_encoder_dict(conv_next_block_dict, down_sampling_block_dict):
    """Encoder dict fixture."""
    encoder = {
        "_target_": "physicsnemo.models.dlwp_healpix.layers.UNetEncoder",
        "conv_block": conv_next_block_dict,
        "down_sampling_block": down_sampling_block_dict,
        "_recursive_": False,
        "n_channels": [136, 68, 34],
        "dilations": [1, 2, 4],
    }
    return encoder


@pytest.fixture
def unet_decoder_dict(
    conv_next_block_dict,
    up_sampling_block_dict,
    output_layer_dict,
):
    """Decoder dict fixture."""
    decoder = {
        "_target_": "physicsnemo.models.dlwp_healpix.layers.UNetDecoder",
        "conv_block": conv_next_block_dict,
        "up_sampling_block": up_sampling_block_dict,
        "output_layer": output_layer_dict,
        "_recursive_": False,
        "n_channels": [34, 68, 136],
        "dilations": [4, 2, 1],
    }
    return omegaconf.DictConfig(decoder)


def test_HEALPixUNet_initialize(
    device, unet_encoder_dict, unet_decoder_dict, pytestconfig
):
    in_channels = 7
    out_channels = 7
    n_constants = 1
    decoder_input_channels = 1
    input_time_dim = 2
    output_time_dim = 4

    model = HEALPixUNet(
        encoder=unet_encoder_dict,
        decoder=unet_decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
    ).to(device)
    assert isinstance(model, HEALPixUNet)

    # test fail case for bad input and output time dims
    with pytest.raises(
        ValueError, match=("'output_time_dim' must be a multiple of 'input_time_dim'")
    ):
        model = HEALPixUNet(
            encoder=unet_encoder_dict,
            decoder=unet_decoder_dict,
            input_channels=in_channels,
            output_channels=out_channels,
            n_constants=n_constants,
            decoder_input_channels=decoder_input_channels,
            input_time_dim=2,
            output_time_dim=3,
        ).to(device)

    # test fail case for couplings with no constants
    with pytest.raises(
        NotImplementedError,
        match=("support for coupled models with no constant field"),
    ):
        model = HEALPixUNet(
            encoder=unet_encoder_dict,
            decoder=unet_decoder_dict,
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
        model = HEALPixUNet(
            encoder=unet_encoder_dict,
            decoder=unet_decoder_dict,
            input_channels=in_channels,
            output_channels=out_channels,
            input_time_dim=2,
            output_time_dim=3,
            decoder_input_channels=0,
            n_constants=2,
            couplings=["t2m", "v10m"],
        ).to(device)

    del model
    torch.cuda.empty_cache()


def test_HEALPixUNet_integration_steps(
    device, unet_encoder_dict, unet_decoder_dict, pytestconfig
):
    in_channels = 2
    out_channels = 2
    n_constants = 1
    decoder_input_channels = 0
    input_time_dim = 2
    output_time_dim = 4

    model = HEALPixUNet(
        encoder=unet_encoder_dict,
        decoder=unet_decoder_dict,
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


def test_HEALPixUNet_forward(
    device,
    unet_encoder_dict,
    unet_decoder_dict,
    test_data,
    insolation_data,
    constant_data,
    pytestconfig,
):
    # create a smaller version of the dlwp healpix model
    in_channels = 3
    out_channels = 3
    n_constants = 2
    decoder_input_channels = 1
    input_time_dim = 2
    output_time_dim = 4
    size = 16

    fix_random_seeds(seed=42)
    x = test_data(
        time_dim=input_time_dim, channels=in_channels, img_size=size, device=device
    )
    decoder_inputs = insolation_data(
        time_dim=output_time_dim, img_size=size, device=device
    )
    constants = constant_data(channels=n_constants, img_size=size, device=device)
    inputs = [x, decoder_inputs, constants]

    model = HEALPixUNet(
        encoder=unet_encoder_dict,
        decoder=unet_decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        enable_healpixpad=True,
    ).to(device)

    # one forward step to initialize recurrent states
    model(inputs)

    assert common.validate_forward_accuracy(
        model,
        (inputs,),
        file_name="models/dlwp_healpix/data/dlwp_healpix_unet.pth",
        rtol=1e-2,
    )

    # no decoder inputs
    inputs = [x, constants]
    model = HEALPixUNet(
        encoder=unet_encoder_dict,
        decoder=unet_decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=0,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        enable_healpixpad=True,
    ).to(device)

    # one forward step to initialize recurrent states
    model(inputs)

    assert common.validate_forward_accuracy(
        model,
        (inputs,),
        file_name="models/dlwp_healpix/data/dlwp_healpix_unet_const.pth",
        rtol=1e-2,
    )

    # no constants
    inputs = [x, decoder_inputs]
    model = HEALPixUNet(
        encoder=unet_encoder_dict,
        decoder=unet_decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=0,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        enable_healpixpad=True,
    ).to(device)

    # one forward step to initialize recurrent states
    model(inputs)

    assert common.validate_forward_accuracy(
        model,
        (inputs,),
        file_name="models/dlwp_healpix/data/dlwp_healpix_unet_decoder.pth",
        rtol=1e-2,
    )

    # no constants and no decoder inputs
    inputs = [x]
    model = HEALPixUNet(
        encoder=unet_encoder_dict,
        decoder=unet_decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=0,
        decoder_input_channels=0,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        enable_healpixpad=True,
    ).to(device)

    # one forward step to initialize recurrent states
    model(inputs)

    assert common.validate_forward_accuracy(
        model,
        (inputs,),
        file_name="models/dlwp_healpix/data/dlwp_healpix_unet_no_decoder_no_const.pth",
        rtol=1e-2,
    )

    del inputs, model
    torch.cuda.empty_cache()


def test_HEALPixUNet_forward_invalid_ndim(
    device, unet_encoder_dict, unet_decoder_dict, pytestconfig
):
    """``forward`` must reject prognostics that aren't shaped (B, F, T, C, H, W)."""
    model = HEALPixUNet(
        encoder=unet_encoder_dict,
        decoder=unet_decoder_dict,
        input_channels=2,
        output_channels=2,
        n_constants=1,
        decoder_input_channels=1,
        input_time_dim=2,
        output_time_dim=2,
    ).to(device)

    bad_prognostics = torch.randn(2, 12, 2, 16, 16).to(device)
    with pytest.raises(ValueError, match="expects prognostics shaped"):
        model([bad_prognostics])

    del model
    torch.cuda.empty_cache()


@torch.no_grad()
def test_HEALPixUNet_forward_output_only_last(
    device,
    unet_encoder_dict,
    unet_decoder_dict,
    test_data,
    insolation_data,
    constant_data,
    pytestconfig,
):
    """With ``output_only_last=True`` only the final integration step's
    outputs should be returned, rather than all steps concatenated.
    """
    in_channels = 2
    out_channels = 2
    n_constants = 1
    decoder_input_channels = 1
    input_time_dim = 2
    output_time_dim = 4
    batch_size = 2
    size = 16

    fix_random_seeds(seed=42)
    x = test_data(
        batch_size=batch_size,
        time_dim=input_time_dim,
        channels=in_channels,
        img_size=size,
        device=device,
    )
    decoder_inputs = insolation_data(
        batch_size=batch_size, time_dim=output_time_dim, img_size=size, device=device
    )
    constants = constant_data(channels=n_constants, img_size=size, device=device)
    inputs = [x, decoder_inputs, constants]

    model = HEALPixUNet(
        encoder=unet_encoder_dict,
        decoder=unet_decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
    ).to(device)

    full_output = model(inputs)
    last_output = model(inputs, output_only_last=True)

    expected_shape = [batch_size, 12, input_time_dim, out_channels, size, size]
    assert list(last_output.shape) == expected_shape
    assert common.compare_output(last_output, full_output[:, :, -input_time_dim:, ...])

    del model, inputs
    torch.cuda.empty_cache()


@torch.no_grad()
def test_HEALPixUNet_forward_residual_prediction(
    device,
    unet_encoder_dict,
    unet_decoder_dict,
    test_data,
    insolation_data,
    constant_data,
    pytestconfig,
):
    """When ``residual_prediction=True`` the model output must equal the
    ``residual_prediction=False`` output plus the raw input tensor (verified
    with identical weights, so the only difference is the residual add).
    Requires ``in_channels == out_channels`` and a single integration step,
    since ``HEALPixUNet`` adds the residual to the full (unsplit) decodings.
    """
    channels = 2
    n_constants = 1
    decoder_input_channels = 1
    input_time_dim = 2
    output_time_dim = 2
    batch_size = 2
    size = 16

    fix_random_seeds(seed=42)
    x = test_data(
        batch_size=batch_size,
        time_dim=input_time_dim,
        channels=channels,
        img_size=size,
        device=device,
    )
    decoder_inputs = insolation_data(
        batch_size=batch_size, time_dim=output_time_dim, img_size=size, device=device
    )
    constants = constant_data(channels=n_constants, img_size=size, device=device)
    inputs = [x, decoder_inputs, constants]

    model_no_residual = HEALPixUNet(
        encoder=unet_encoder_dict,
        decoder=unet_decoder_dict,
        input_channels=channels,
        output_channels=channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        residual_prediction=False,
    ).to(device)
    model_residual = HEALPixUNet(
        encoder=unet_encoder_dict,
        decoder=unet_decoder_dict,
        input_channels=channels,
        output_channels=channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        residual_prediction=True,
    ).to(device)
    model_residual.load_state_dict(model_no_residual.state_dict())

    out_no_residual = model_no_residual(inputs)
    out_residual = model_residual(inputs)

    assert common.compare_output(out_residual, out_no_residual + x)

    del model_no_residual, model_residual, inputs
    torch.cuda.empty_cache()


@torch.no_grad()
def test_HEALPixUNet_forward_constraints(
    device,
    unet_encoder_dict,
    unet_decoder_dict,
    test_data,
    insolation_data,
    constant_data,
    pytestconfig,
):
    """``set_constraints`` should apply the configured constraint modules to
    the output only for the targeted channel(s), leaving other channels
    unaffected (verified with identical weights to an unconstrained model).
    """
    in_channels = 2
    out_channels = 2
    n_constants = 1
    decoder_input_channels = 1
    input_time_dim = 2
    output_time_dim = 2
    batch_size = 2
    size = 16

    fix_random_seeds(seed=42)
    x = test_data(
        batch_size=batch_size,
        time_dim=input_time_dim,
        channels=in_channels,
        img_size=size,
        device=device,
    )
    decoder_inputs = insolation_data(
        batch_size=batch_size, time_dim=output_time_dim, img_size=size, device=device
    )
    constants = constant_data(channels=n_constants, img_size=size, device=device)
    inputs = [x, decoder_inputs, constants]

    constraints = omegaconf.OmegaConf.create(
        {
            "non_negative": {
                "_target_": "physicsnemo.models.dlwp_healpix.layers.NonnegativeConstraint",
                "variables": ["var0"],
                "channels": ["var0", "var1"],
                "scaling": {"var0": {"mean": 0.0, "std": 1.0}},
            }
        }
    )

    model_unconstrained = HEALPixUNet(
        encoder=unet_encoder_dict,
        decoder=unet_decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
    ).to(device)
    model_constrained = HEALPixUNet(
        encoder=unet_encoder_dict,
        decoder=unet_decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        constraints=constraints,
    ).to(device)
    model_constrained.load_state_dict(model_unconstrained.state_dict())

    out_unconstrained = model_unconstrained(inputs)
    out_constrained = model_constrained(inputs)

    # unconstrained "var0" should have some negative values to make this a
    # meaningful test of the clamp
    assert out_unconstrained[:, :, :, 0].min() < 0
    assert torch.all(out_constrained[:, :, :, 0] >= 0)
    assert common.compare_output(
        out_unconstrained[:, :, :, 1], out_constrained[:, :, :, 1]
    )

    del model_unconstrained, model_constrained, inputs
    torch.cuda.empty_cache()


@torch.no_grad()
def test_HEALPixUNet_forward_conditions_cln(
    device,
    cln_unet_encoder_dict,
    cln_unet_decoder_dict,
    test_data,
    insolation_data,
    constant_data,
    pytestconfig,
):
    """``conditions_cln`` must be required when the encoder/decoder are
    CLN-enabled, and different conditions must yield different outputs.
    """
    channels = 4
    out_channels = 2
    n_constants = 1
    decoder_input_channels = 1
    input_time_dim = 2
    output_time_dim = 2
    cond_dim = 8
    batch_size = 2
    size = 16

    fix_random_seeds(seed=42)
    x = test_data(
        batch_size=batch_size,
        time_dim=input_time_dim,
        channels=channels,
        img_size=size,
        device=device,
    )
    decoder_inputs = insolation_data(
        batch_size=batch_size, time_dim=output_time_dim, img_size=size, device=device
    )
    constants = constant_data(channels=n_constants, img_size=size, device=device)
    inputs = [x, decoder_inputs, constants]

    model = HEALPixUNet(
        encoder=cln_unet_encoder_dict,
        decoder=cln_unet_decoder_dict,
        input_channels=channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
    ).to(device)

    with pytest.raises(ValueError, match="Conditional inputs are required"):
        model(inputs)

    conditions_a = [
        torch.randn(batch_size, cond_dim).to(device)
        for _ in range(model.integration_steps)
    ]
    conditions_b = [
        torch.randn(batch_size, cond_dim).to(device)
        for _ in range(model.integration_steps)
    ]

    out_a = model(inputs, conditions_cln=conditions_a)
    out_b = model(inputs, conditions_cln=conditions_b)

    assert not common.compare_output(out_a, out_b)

    del model, inputs
    torch.cuda.empty_cache()


@torch.no_grad()
def test_HEALPixUNet_forward_couplings(
    device,
    unet_encoder_dict,
    unet_decoder_dict,
    test_data,
    insolation_data,
    constant_data,
    coupling_data,
    pytestconfig,
):
    """A valid, non-empty ``couplings`` configuration should be accepted and
    routed through ``_reshape_inputs``/``forward`` without error, producing
    the expected output shape.
    """
    in_channels = 3
    out_channels = 3
    n_constants = 2
    decoder_input_channels = 1
    input_time_dim = 2
    output_time_dim = 4
    batch_size = 2
    size = 16

    couplings = [
        {"params": {"variables": ["v1", "v2"], "input_times": ["24h", "48h"]}},
    ]

    fix_random_seeds(seed=42)
    model = HEALPixUNet(
        encoder=unet_encoder_dict,
        decoder=unet_decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        couplings=couplings,
    ).to(device)

    assert model.coupled_channels == 4

    x = test_data(
        batch_size=batch_size,
        time_dim=input_time_dim,
        channels=in_channels,
        img_size=size,
        device=device,
    )
    decoder_inputs = insolation_data(
        batch_size=batch_size, time_dim=output_time_dim, img_size=size, device=device
    )
    constants = constant_data(channels=n_constants, img_size=size, device=device)
    couplings_tensor = coupling_data(
        steps=model.integration_steps,
        channels=model.coupled_channels,
        batch_size=batch_size,
        img_size=size,
        device=device,
    )
    inputs = [x, decoder_inputs, constants, couplings_tensor]

    output = model(inputs)
    expected_shape = [batch_size, 12, output_time_dim, out_channels, size, size]
    assert list(output.shape) == expected_shape

    del model, inputs
    torch.cuda.empty_cache()


def test_HEALPixUNet_forward_is_diagnostic(
    device,
    unet_encoder_dict,
    unet_decoder_dict,
    test_data,
    constant_data,
    pytestconfig,
):
    """With ``is_diagnostic=True`` (and ``output_time_dim == 1``) the model runs
    in diagnostic mode: the usual ``output_time_dim % input_time_dim == 0``
    requirement is bypassed and the output has a single time step regardless of
    ``input_time_dim``.
    """
    in_channels = 2
    out_channels = 2
    n_constants = 1
    input_time_dim = 2
    output_time_dim = 1
    batch_size = 2
    size = 16

    model = HEALPixUNet(
        encoder=unet_encoder_dict,
        decoder=unet_decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=0,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        is_diagnostic=True,
    ).to(device)

    assert model.is_diagnostic
    assert model.integration_steps == 1

    x = test_data(
        batch_size=batch_size,
        time_dim=input_time_dim,
        channels=in_channels,
        img_size=size,
        device=device,
    )
    constants = constant_data(channels=n_constants, img_size=size, device=device)
    inputs = [x, constants]

    with torch.no_grad():
        output = model(inputs)

    expected_shape = [batch_size, 12, output_time_dim, out_channels, size, size]
    assert list(output.shape) == expected_shape

    del model, inputs
    torch.cuda.empty_cache()


def test_HEALPixUNet_diagnostic_residual_prediction_raises(
    device, unet_encoder_dict, unet_decoder_dict, pytestconfig
):
    """A diagnostic model (``is_diagnostic=True``) cannot predict a residual, so
    requesting ``residual_prediction=True`` must raise ``ValueError``; the same
    configuration with ``residual_prediction=False`` must construct
    successfully.
    """
    common_kwargs = dict(
        encoder=unet_encoder_dict,
        decoder=unet_decoder_dict,
        input_channels=2,
        output_channels=2,
        n_constants=1,
        decoder_input_channels=0,
        input_time_dim=2,
        output_time_dim=1,
        is_diagnostic=True,
    )

    with pytest.raises(
        ValueError, match="A diagnostic model cannot predict a residual"
    ):
        HEALPixUNet(**common_kwargs, residual_prediction=True).to(device)

    # the same diagnostic configuration is valid without residual prediction
    model = HEALPixUNet(**common_kwargs, residual_prediction=False).to(device)
    assert model.is_diagnostic
    assert not model.residual_prediction

    del model
    torch.cuda.empty_cache()


def test_HEALPixUNet_diagnostic_requires_output_time_dim_one(
    device, unet_encoder_dict, unet_decoder_dict, pytestconfig
):
    """``is_diagnostic=True`` requires ``output_time_dim == 1``; any other value
    must raise ``ValueError``, while ``output_time_dim == 1`` constructs
    successfully."""
    common_kwargs = dict(
        encoder=unet_encoder_dict,
        decoder=unet_decoder_dict,
        input_channels=2,
        output_channels=2,
        n_constants=1,
        decoder_input_channels=0,
        input_time_dim=2,
        is_diagnostic=True,
    )

    with pytest.raises(ValueError, match="must have output_time_dim == 1"):
        HEALPixUNet(**common_kwargs, output_time_dim=2).to(device)

    model = HEALPixUNet(**common_kwargs, output_time_dim=1).to(device)
    assert model.is_diagnostic

    del model
    torch.cuda.empty_cache()


@torch.no_grad()
def test_HEALPixUNet_forward_diagnostic_output_channel(
    device,
    unet_encoder_dict,
    unet_decoder_dict,
    test_data,
    insolation_data,
    constant_data,
    pytestconfig,
):
    """A model whose output includes an extra diagnostic channel
    (``output_channels > input_channels``) must run across multiple
    integration steps: only the prognostic channels of each step's output are
    fed back into the next step, so the extra diagnostic channel does not
    break the autoregressive loop.
    """
    in_channels = 2
    out_channels = 3  # one extra diagnostic channel beyond the prognostic inputs
    n_constants = 2
    decoder_input_channels = 1
    input_time_dim = 2
    output_time_dim = 4  # 2 integration steps -> exercises the step>0 feedback path
    batch_size = 2
    size = 16

    fix_random_seeds(seed=42)
    x = test_data(
        batch_size=batch_size,
        time_dim=input_time_dim,
        channels=in_channels,
        img_size=size,
        device=device,
    )
    decoder_inputs = insolation_data(
        batch_size=batch_size,
        time_dim=output_time_dim,
        img_size=size,
        device=device,
    )
    constants = constant_data(channels=n_constants, img_size=size, device=device)
    inputs = [x, decoder_inputs, constants]

    model = HEALPixUNet(
        encoder=unet_encoder_dict,
        decoder=unet_decoder_dict,
        input_channels=in_channels,
        output_channels=out_channels,
        n_constants=n_constants,
        decoder_input_channels=decoder_input_channels,
        input_time_dim=input_time_dim,
        output_time_dim=output_time_dim,
        residual_prediction=False,
    ).to(device)

    # more than one integration step, and a genuine diagnostic output channel
    assert model.integration_steps == 2
    assert not model.is_diagnostic

    output = model(inputs)

    expected_shape = [batch_size, 12, output_time_dim, out_channels, size, size]
    assert list(output.shape) == expected_shape
    assert torch.isfinite(output).all()

    del model, inputs
    torch.cuda.empty_cache()


def test_HEALPixUNet_backward_compat_arg_mapper():
    """Legacy (version ``0.1.0``) checkpoints used ``dlwp_healpix_layers``
    Hydra targets; ``_backward_compat_arg_mapper`` must remap them to the
    current ``physicsnemo.models.dlwp_healpix.layers`` module, and must be a
    no-op for any other (current) version.
    """
    legacy_args = {
        "encoder": {
            "_target_": "physicsnemo.models.dlwp_healpix_layers.healpix_encoder.UNetEncoder",
            "conv_block": {
                "_target_": "physicsnemo.models.dlwp_healpix_layers.healpix_blocks.ConvNeXtBlock",
            },
        },
        "decoder": {
            "_target_": "physicsnemo.models.dlwp_healpix_layers.healpix_decoder.UNetDecoder",
        },
        "input_channels": 3,
    }

    remapped = HEALPixUNet._backward_compat_arg_mapper("0.1.0", legacy_args)
    assert (
        remapped["encoder"]["_target_"]
        == "physicsnemo.models.dlwp_healpix.layers.UNetEncoder"
    )
    assert (
        remapped["decoder"]["_target_"]
        == "physicsnemo.models.dlwp_healpix.layers.UNetDecoder"
    )
    assert (
        remapped["encoder"]["conv_block"]["_target_"]
        == "physicsnemo.models.dlwp_healpix.layers.ConvNeXtBlock"
    )
    assert remapped["input_channels"] == 3

    # any other version is a no-op (the base class's default implementation)
    unchanged = HEALPixUNet._backward_compat_arg_mapper("9.9.9", legacy_args)
    assert unchanged == legacy_args


@torch.no_grad()
def test_HEALPixUNet_checkpoint(
    device,
    unet_encoder_dict,
    unet_decoder_dict,
    test_data,
    insolation_data,
    constant_data,
    pytestconfig,
):
    """MOD-008c: save one ``HEALPixUNet`` and restore its state into a second,
    differently-initialized model via both ``Module.load`` and
    ``Module.from_checkpoint``, verifying the forward outputs match."""
    in_channels = 3
    out_channels = 3
    n_constants = 2
    decoder_input_channels = 1
    input_time_dim = 2
    output_time_dim = 4
    size = 16

    fix_random_seeds(seed=42)
    x = test_data(
        time_dim=input_time_dim, channels=in_channels, img_size=size, device=device
    )
    decoder_inputs = insolation_data(
        time_dim=output_time_dim, img_size=size, device=device
    )
    constants = constant_data(channels=n_constants, img_size=size, device=device)
    inputs = [x, decoder_inputs, constants]

    # ``Module.save`` serializes the captured init args to JSON, so pass plain
    # containers rather than ``DictConfig`` (which is not JSON-serializable).
    encoder = omegaconf.OmegaConf.to_container(
        omegaconf.OmegaConf.create(unet_encoder_dict), resolve=True
    )
    decoder = omegaconf.OmegaConf.to_container(
        omegaconf.OmegaConf.create(unet_decoder_dict), resolve=True
    )

    def build_model():
        return HEALPixUNet(
            encoder=encoder,
            decoder=decoder,
            input_channels=in_channels,
            output_channels=out_channels,
            n_constants=n_constants,
            decoder_input_channels=decoder_input_channels,
            input_time_dim=input_time_dim,
            output_time_dim=output_time_dim,
        ).to(device)

    model_1 = build_model()
    model_2 = build_model()

    # Perturb model_2's weights so its output differs from model_1 before the
    # checkpoint is loaded (validate_checkpoint asserts the initial mismatch).
    with torch.no_grad():
        for param in model_2.parameters():
            param.add_(0.1)

    assert common.validate_checkpoint(model_1, model_2, (inputs,), rtol=1e-2)

    del model_1, model_2, inputs
    torch.cuda.empty_cache()
