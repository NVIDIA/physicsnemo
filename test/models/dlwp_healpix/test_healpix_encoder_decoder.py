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

from test import common


def _cln_conv_block(in_channels: int, cond_dim: int) -> omegaconf.DictConfig:
    """Hydra-instantiable ``Multi_SymmetricConvNeXtBlock`` config with an
    always-on conditional layer norm, used to exercise the ``per_level_cln``
    and ``conditions_cln`` wiring in ``UNetEncoder``/``UNetDecoder``.

    Must be an ``omegaconf.DictConfig`` (not a plain ``dict``) because
    ``UNetEncoder``/``UNetDecoder`` access ``block_config.conditional_layer_norm``
    via attribute lookup when deciding whether to disable CLN for a given level.
    """
    return omegaconf.DictConfig(
        {
            "_target_": "physicsnemo.models.dlwp_healpix.layers.Multi_SymmetricConvNeXtBlock",
            "in_channels": in_channels,
            "conditional_layer_norm": {
                "_target_": "physicsnemo.models.dlwp_healpix.layers.ConditionalLayerNorm",
                "_partial_": True,
                "condition_shape": cond_dim,
            },
        }
    )


def test_UNetEncoder_initialize(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import ConvNeXtBlock, UNetEncoder
    from physicsnemo.nn.module.hpx import HEALPixMaxPool

    channels = 2
    n_channels = (16, 32, 64)

    # Dicts for block configs used by encoder
    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": channels,
    }
    down_sampling_block = {
        "_target_": HEALPixMaxPool,
        "pooling": 2,
    }

    encoder = UNetEncoder(
        conv_block=conv_block,
        down_sampling_block=down_sampling_block,
        n_channels=n_channels,
        input_channels=channels,
    ).to(device)
    assert isinstance(encoder, UNetEncoder)

    # with dilations
    encoder = UNetEncoder(
        conv_block=conv_block,
        down_sampling_block=down_sampling_block,
        n_channels=n_channels,
        input_channels=channels,
        dilations=(1, 1, 1),
    ).to(device)
    assert isinstance(encoder, UNetEncoder)

    del encoder
    torch.cuda.empty_cache()


def test_UNetEncoder_forward(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import ConvNeXtBlock, UNetEncoder
    from physicsnemo.nn.module.hpx import HEALPixMaxPool

    channels = 2
    hw_size = 16
    b_size = 12
    n_channels = (16, 32, 64)

    # Dicts for block configs used by encoder
    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": channels,
    }
    down_sampling_block = {
        "_target_": HEALPixMaxPool,
        "pooling": 2,
    }

    encoder = UNetEncoder(
        conv_block=conv_block,
        down_sampling_block=down_sampling_block,
        n_channels=n_channels,
        input_channels=channels,
    ).to(device)

    tensor_size = [b_size, channels, hw_size, hw_size]
    invar = torch.rand(tensor_size).to(device)
    outvar = encoder(invar)

    # doesn't do anything
    encoder.reset()

    # outvar is a module list
    for idx, out_tensor in enumerate(outvar):
        # verify the channels and h dim are correct
        assert out_tensor.shape[1] == n_channels[idx]
        # default behaviour is to half the h/w size after first
        assert out_tensor.shape[2] == tensor_size[2] // (2**idx)

    del encoder, invar, outvar
    torch.cuda.empty_cache()


def test_UNetEncoder_reset(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import ConvNeXtBlock, UNetEncoder
    from physicsnemo.nn.module.hpx import HEALPixMaxPool

    channels = 2
    n_channels = (16, 32, 64)

    # Dicts for block configs used by encoder
    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": channels,
    }
    down_sampling_block = {
        "_target_": HEALPixMaxPool,
        "pooling": 2,
    }

    encoder = UNetEncoder(
        conv_block=conv_block,
        down_sampling_block=down_sampling_block,
        n_channels=n_channels,
        input_channels=channels,
    ).to(device)

    # doesn't do anything
    encoder.reset()
    assert isinstance(encoder, UNetEncoder)

    del encoder
    torch.cuda.empty_cache()


def test_UNetEncoder_per_level_cln_length_mismatch(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import ConvNeXtBlock, UNetEncoder
    from physicsnemo.nn.module.hpx import HEALPixMaxPool

    channels = 2
    n_channels = (16, 32, 64)

    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": channels,
    }
    down_sampling_block = {
        "_target_": HEALPixMaxPool,
        "pooling": 2,
    }

    with pytest.raises(ValueError, match="per_level_cln must be a list of booleans"):
        UNetEncoder(
            conv_block=conv_block,
            down_sampling_block=down_sampling_block,
            n_channels=n_channels,
            input_channels=channels,
            # wrong length: 2 entries for 3 levels
            per_level_cln=[True, False],
        )


def test_UNetEncoder_per_level_checkpointing_length_mismatch(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import ConvNeXtBlock, UNetEncoder
    from physicsnemo.nn.module.hpx import HEALPixMaxPool

    channels = 2
    n_channels = (16, 32, 64)

    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": channels,
    }
    down_sampling_block = {
        "_target_": HEALPixMaxPool,
        "pooling": 2,
    }

    with pytest.raises(
        ValueError, match="per_level_checkpointing must be a list of booleans"
    ):
        UNetEncoder(
            conv_block=conv_block,
            down_sampling_block=down_sampling_block,
            n_channels=n_channels,
            input_channels=channels,
            # wrong length: 2 entries for 3 levels
            per_level_checkpointing=[True, False],
        )


@pytest.mark.parametrize(
    "mask", [[True, False, True], [False, True, False]], ids=["tft", "ftf"]
)
def test_UNetEncoder_per_level_cln_disables_expected_levels(device, mask, pytestconfig):
    """Verify per_level_cln is honored independently at each level, not applied
    uniformly to all levels of the encoder.
    """
    from physicsnemo.models.dlwp_healpix.layers import UNetEncoder
    from physicsnemo.nn.module.hpx import HEALPixMaxPool

    channels = 8
    cond_dim = 8
    n_cond = 2
    hw_size = 8
    n_channels = (8, 8, 8)

    conv_block = _cln_conv_block(channels, cond_dim)
    down_sampling_block = {
        "_target_": HEALPixMaxPool,
        "pooling": 2,
    }

    encoder = UNetEncoder(
        conv_block=conv_block,
        down_sampling_block=down_sampling_block,
        n_channels=n_channels,
        input_channels=channels,
        per_level_cln=mask,
    ).to(device)

    # instantiation-time check: every level's cln_enabled flag must exactly
    # match the requested mask at that index
    for n in range(len(n_channels)):
        conv_module = encoder.encoder[n][-1]
        assert conv_module.cln_enabled == mask[n], (
            f"level {n}: expected cln_enabled={mask[n]}, got {conv_module.cln_enabled}"
        )

    # forward-time check: call each level's conv module directly (bypassing
    # the encoder's sequential level-to-level dependency, since an earlier
    # CLN-sensitive level would otherwise contaminate the input to every
    # later level regardless of that later level's own per_level_cln value)
    for n in range(len(n_channels)):
        conv_module = encoder.encoder[n][-1]
        fixed_input = torch.rand([12 * n_cond, channels, hw_size, hw_size]).to(device)
        cond_a = torch.randn(n_cond, cond_dim).to(device)
        cond_b = torch.randn(n_cond, cond_dim).to(device)

        with torch.no_grad():
            out_a = conv_module(fixed_input, conditions_cln=cond_a)
            out_b = conv_module(fixed_input, conditions_cln=cond_b)

        same = common.compare_output(out_a, out_b)
        if mask[n]:
            assert not same, f"level {n} should be sensitive to conditions_cln"
        else:
            assert same, f"level {n} should be unaffected by conditions_cln"

    del encoder
    torch.cuda.empty_cache()


def test_UNetEncoder_forward_conditions_cln_required(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import UNetEncoder
    from physicsnemo.nn.module.hpx import HEALPixMaxPool

    channels = 4
    cond_dim = 8
    hw_size = 16
    n_channels = (8, 8, 8)

    conv_block = _cln_conv_block(channels, cond_dim)
    down_sampling_block = {
        "_target_": HEALPixMaxPool,
        "pooling": 2,
    }

    # default per_level_cln (None) enables CLN at every level
    encoder = UNetEncoder(
        conv_block=conv_block,
        down_sampling_block=down_sampling_block,
        n_channels=n_channels,
        input_channels=channels,
    ).to(device)

    invar = torch.rand([12, channels, hw_size, hw_size]).to(device)

    with pytest.raises(ValueError, match="Conditional inputs are required"):
        encoder(invar, conditions_cln=None)

    del encoder, invar
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    ("mask", "expected_calls"),
    [([True, False, True], 2), ([False, True, False], 1)],
    ids=["tft-2calls", "ftf-1call"],
)
def test_UNetEncoder_per_level_checkpointing_applied_only_at_configured_levels(
    device, monkeypatch, mask, expected_calls, pytestconfig
):
    """Verify per_level_checkpointing routes only the configured levels through
    torch.utils.checkpoint.checkpoint(), not all (or none) of the encoder levels.
    """
    from torch.utils.checkpoint import checkpoint as real_checkpoint

    from physicsnemo.models.dlwp_healpix.layers import ConvNeXtBlock, UNetEncoder
    from physicsnemo.nn.module.hpx import HEALPixMaxPool

    channels = 2
    hw_size = 16
    n_channels = (8, 8, 8)

    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": channels,
    }
    down_sampling_block = {
        "_target_": HEALPixMaxPool,
        "pooling": 2,
    }

    call_count = {"n": 0}

    def spy_checkpoint(*args, **kwargs):
        call_count["n"] += 1
        return real_checkpoint(*args, **kwargs)

    monkeypatch.setattr(
        "physicsnemo.models.dlwp_healpix.layers.healpix_encoder.checkpoint",
        spy_checkpoint,
    )

    encoder = UNetEncoder(
        conv_block=conv_block,
        down_sampling_block=down_sampling_block,
        n_channels=n_channels,
        input_channels=channels,
        per_level_checkpointing=mask,
    ).to(device)

    invar = torch.rand([12, channels, hw_size, hw_size]).to(device)
    encoder(invar)

    assert call_count["n"] == expected_calls, (
        f"mask={mask}: expected {expected_calls} checkpoint() call(s), "
        f"got {call_count['n']}"
    )

    del encoder, invar
    torch.cuda.empty_cache()


def test_UNetEncoder_per_level_checkpointing_matches_no_checkpointing(
    device, pytestconfig
):
    """A mixed (non-uniform) per_level_checkpointing pattern must produce the
    same forward output and gradients as no checkpointing at all, using the
    exact same model weights.
    """
    from physicsnemo.models.dlwp_healpix.layers import ConvNeXtBlock, UNetEncoder
    from physicsnemo.nn.module.hpx import HEALPixMaxPool

    channels = 2
    hw_size = 16
    b_size = 12
    n_channels = (8, 8, 8)

    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": channels,
    }
    down_sampling_block = {
        "_target_": HEALPixMaxPool,
        "pooling": 2,
    }

    encoder = UNetEncoder(
        conv_block=conv_block,
        down_sampling_block=down_sampling_block,
        n_channels=n_channels,
        input_channels=channels,
        per_level_checkpointing=[False, False, False],
    ).to(device)

    invar = torch.rand([b_size, channels, hw_size, hw_size]).to(device)
    invar.requires_grad_(True)

    baseline = encoder(invar)
    sum(o.sum() for o in baseline).backward()
    baseline_grad = invar.grad.clone()
    baseline_out = [o.detach().clone() for o in baseline]

    invar.grad = None
    encoder.zero_grad()

    # mixed pattern applied to the *same* model instance/weights
    encoder.per_level_checkpointing = [True, False, True]

    checkpointed = encoder(invar)
    sum(o.sum() for o in checkpointed).backward()

    for n in range(len(n_channels)):
        assert common.compare_output(baseline_out[n], checkpointed[n])

    assert torch.isfinite(invar.grad).all()
    assert torch.allclose(baseline_grad, invar.grad, rtol=1e-4, atol=1e-5)

    del encoder, invar, baseline, checkpointed
    torch.cuda.empty_cache()


def test_UNetDecoder_initilization(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        BasicConvBlock,  # for the output layer
        ConvGRUBlock,  # for the recurrent layer
        ConvNeXtBlock,  # for convolutional layer
        TransposedConvUpsample,  # for upsampling
        UNetDecoder,
    )

    in_channels = 2
    out_channels = 1
    n_channels = (64, 32, 16)

    # Dicts for block configs used by decoder
    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": in_channels,
    }

    up_sampling_block = {
        "_target_": TransposedConvUpsample,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "upsampling": 2,
    }

    output_layer = {
        "_target_": BasicConvBlock,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": 1,
        "dilation": 1,
        "n_layers": 1,
    }

    recurrent_block = {
        "_target_": ConvGRUBlock,
        "in_channels": 2,
        "kernel_size": 1,
    }

    decoder = UNetDecoder(
        conv_block=conv_block,
        up_sampling_block=up_sampling_block,
        output_layer=output_layer,
        recurrent_block=recurrent_block,
        n_channels=n_channels,
    ).to(device)

    assert isinstance(decoder, UNetDecoder)

    # without the recurrent block and with dilations
    decoder = UNetDecoder(
        conv_block=conv_block,
        up_sampling_block=up_sampling_block,
        output_layer=output_layer,
        recurrent_block=None,
        n_channels=n_channels,
        dilations=(1, 1, 1),
    ).to(device)
    assert isinstance(decoder, UNetDecoder)

    del decoder
    torch.cuda.empty_cache()


def test_UNetDecoder_forward(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        BasicConvBlock,  # for the output layer
        ConvGRUBlock,  # for the recurrent layer
        ConvNeXtBlock,  # for convolutional layer
        TransposedConvUpsample,  # for upsampling
        UNetDecoder,
    )

    in_channels = 2
    out_channels = 1
    hw_size = 32
    b_size = 12
    n_channels = (64, 32, 16)

    # Dicts for block configs used by decoder
    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": in_channels,
    }

    up_sampling_block = {
        "_target_": TransposedConvUpsample,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "upsampling": 2,
    }

    output_layer = {
        "_target_": BasicConvBlock,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": 1,
        "dilation": 1,
        "n_layers": 1,
    }

    recurrent_block = {
        "_target_": ConvGRUBlock,
        "in_channels": 2,
        "kernel_size": 1,
    }

    decoder = UNetDecoder(
        conv_block=conv_block,
        up_sampling_block=up_sampling_block,
        output_layer=output_layer,
        recurrent_block=recurrent_block,
        n_channels=n_channels,
    ).to(device)

    expected_size = torch.Size([b_size, out_channels, hw_size, hw_size])

    # build the list of tensors for the decoder
    invars = []
    # decoder has an algorithm that goes back to front
    for idx in range(len(n_channels) - 1, -1, -1):
        tensor_size = [b_size, n_channels[idx], hw_size, hw_size]
        invars.append(torch.rand(tensor_size).to(device))
        hw_size = hw_size // 2

    outvar = decoder(invars)
    assert outvar.shape == expected_size

    # make sure history is taken into account with ConvGRU
    outvar_hist = decoder(invars)
    assert not common.compare_output(outvar, outvar_hist)

    # check with no recurrent
    decoder = UNetDecoder(
        conv_block=conv_block,
        up_sampling_block=up_sampling_block,
        output_layer=output_layer,
        recurrent_block=None,
        n_channels=n_channels,
        dilations=(1, 1, 1),
    ).to(device)

    outvar = decoder(invars)
    assert outvar.shape == expected_size

    del decoder, outvar, invars
    torch.cuda.empty_cache()


def test_UNetDecoder_reset(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        BasicConvBlock,  # for the output layer
        ConvGRUBlock,  # for the recurrent layer
        ConvNeXtBlock,  # for convolutional layer
        TransposedConvUpsample,  # for upsampling
        UNetDecoder,
    )

    in_channels = 2
    out_channels = 1
    hw_size = 32
    b_size = 12
    n_channels = (64, 32, 16)

    # Dicts for block configs used by decoder
    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": in_channels,
    }

    up_sampling_block = {
        "_target_": TransposedConvUpsample,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "upsampling": 2,
    }

    output_layer = {
        "_target_": BasicConvBlock,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": 1,
        "dilation": 1,
        "n_layers": 1,
    }

    recurrent_block = {
        "_target_": ConvGRUBlock,
        "in_channels": 2,
        "kernel_size": 1,
    }

    decoder = UNetDecoder(
        conv_block=conv_block,
        up_sampling_block=up_sampling_block,
        output_layer=output_layer,
        recurrent_block=recurrent_block,
        n_channels=n_channels,
    ).to(device)

    # build the list of tensors for the decoder
    invars = []
    # decoder has an algorithm that goes back to front
    for idx in range(len(n_channels) - 1, -1, -1):
        tensor_size = [b_size, n_channels[idx], hw_size, hw_size]
        invars.append(torch.rand(tensor_size).to(device))
        hw_size = hw_size // 2

    outvar = decoder(invars)

    # make sure history is taken into account with ConvGRU
    outvar_hist = decoder(invars)
    assert not common.compare_output(outvar, outvar_hist)

    # make sure after reset we get the same result
    decoder.reset()
    outvar_reset = decoder(invars)
    assert common.compare_output(outvar, outvar_reset)

    # test reset without recurrent block
    decoder = UNetDecoder(
        conv_block=conv_block,
        up_sampling_block=up_sampling_block,
        output_layer=output_layer,
        recurrent_block=None,
        n_channels=n_channels,
    ).to(device)

    outvar = decoder(invars)

    # without the recurrent block should be the same
    outvar_hist = decoder(invars)
    assert common.compare_output(outvar, outvar_hist)

    # make sure after reset we get the same result
    decoder.reset()
    outvar_reset = decoder(invars)
    assert common.compare_output(outvar, outvar_reset)

    del decoder, outvar, invars
    torch.cuda.empty_cache()


def test_UNetDecoder_per_level_cln_length_mismatch(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        BasicConvBlock,
        ConvNeXtBlock,
        TransposedConvUpsample,
        UNetDecoder,
    )

    in_channels = 2
    out_channels = 1
    n_channels = (64, 32, 16)

    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": in_channels,
    }
    up_sampling_block = {
        "_target_": TransposedConvUpsample,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "upsampling": 2,
    }
    output_layer = {
        "_target_": BasicConvBlock,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": 1,
        "dilation": 1,
        "n_layers": 1,
    }

    with pytest.raises(ValueError, match="per_level_cln must be a list of booleans"):
        UNetDecoder(
            conv_block=conv_block,
            up_sampling_block=up_sampling_block,
            output_layer=output_layer,
            recurrent_block=None,
            n_channels=n_channels,
            # wrong length: 2 entries for 3 levels
            per_level_cln=[True, False],
        )


def test_UNetDecoder_per_level_checkpointing_length_mismatch(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        BasicConvBlock,
        ConvNeXtBlock,
        TransposedConvUpsample,
        UNetDecoder,
    )

    in_channels = 2
    out_channels = 1
    n_channels = (64, 32, 16)

    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": in_channels,
    }
    up_sampling_block = {
        "_target_": TransposedConvUpsample,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "upsampling": 2,
    }
    output_layer = {
        "_target_": BasicConvBlock,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": 1,
        "dilation": 1,
        "n_layers": 1,
    }

    with pytest.raises(
        ValueError, match="per_level_checkpointing must be a list of booleans"
    ):
        UNetDecoder(
            conv_block=conv_block,
            up_sampling_block=up_sampling_block,
            output_layer=output_layer,
            recurrent_block=None,
            n_channels=n_channels,
            # wrong length: 2 entries for 3 levels
            per_level_checkpointing=[True, False],
        )


@pytest.mark.parametrize(
    "mask", [[True, False, True], [False, True, False]], ids=["tft", "ftf"]
)
def test_UNetDecoder_per_level_cln_disables_expected_levels(device, mask, pytestconfig):
    """Verify per_level_cln is honored independently at each level, not applied
    uniformly to all levels of the decoder.
    """
    from physicsnemo.models.dlwp_healpix.layers import (
        BasicConvBlock,
        TransposedConvUpsample,
        UNetDecoder,
    )

    cond_dim = 8
    n_cond = 2
    hw_size = 8
    n_channels = (8, 8, 8)
    out_channels = 4

    conv_block = _cln_conv_block(n_channels[0], cond_dim)
    up_sampling_block = {
        "_target_": TransposedConvUpsample,
        "in_channels": n_channels[0],
        "out_channels": n_channels[0],
        "upsampling": 2,
    }
    output_layer = {
        "_target_": BasicConvBlock,
        "in_channels": n_channels[0],
        "out_channels": out_channels,
        "kernel_size": 1,
        "dilation": 1,
        "n_layers": 1,
    }

    decoder = UNetDecoder(
        conv_block=conv_block,
        up_sampling_block=up_sampling_block,
        output_layer=output_layer,
        recurrent_block=None,
        n_channels=n_channels,
        output_channels=out_channels,
        per_level_cln=mask,
    ).to(device)

    # instantiation-time check: every level's cln_enabled flag must exactly
    # match the requested mask at that index
    for n in range(len(n_channels)):
        conv_module = decoder.decoder[n]["conv"]
        assert conv_module.cln_enabled == mask[n], (
            f"level {n}: expected cln_enabled={mask[n]}, got {conv_module.cln_enabled}"
        )

    # forward-time check: call each level's conv module directly (bypassing
    # the decoder's sequential level-to-level dependency, since an earlier
    # CLN-sensitive level would otherwise contaminate the input to every
    # later level regardless of that later level's own per_level_cln value)
    for n in range(len(n_channels)):
        conv_module = decoder.decoder[n]["conv"]
        # level 0 has no skip-connection concat, so its conv sees n_channels[0]
        # input channels; every subsequent level concatenates the upsampled
        # skip connection, doubling the channel count
        in_ch = n_channels[n] * 2 if n > 0 else n_channels[n]
        fixed_input = torch.rand([12 * n_cond, in_ch, hw_size, hw_size]).to(device)
        cond_a = torch.randn(n_cond, cond_dim).to(device)
        cond_b = torch.randn(n_cond, cond_dim).to(device)

        with torch.no_grad():
            out_a = conv_module(fixed_input, conditions_cln=cond_a)
            out_b = conv_module(fixed_input, conditions_cln=cond_b)

        same = common.compare_output(out_a, out_b)
        if mask[n]:
            assert not same, f"level {n} should be sensitive to conditions_cln"
        else:
            assert same, f"level {n} should be unaffected by conditions_cln"

    del decoder
    torch.cuda.empty_cache()


def test_UNetDecoder_forward_conditions_cln_required(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        BasicConvBlock,
        TransposedConvUpsample,
        UNetDecoder,
    )

    cond_dim = 8
    hw_size = 16
    n_channels = (8, 8, 8)
    out_channels = 4

    conv_block = _cln_conv_block(n_channels[0], cond_dim)
    up_sampling_block = {
        "_target_": TransposedConvUpsample,
        "in_channels": n_channels[0],
        "out_channels": n_channels[0],
        "upsampling": 2,
    }
    output_layer = {
        "_target_": BasicConvBlock,
        "in_channels": n_channels[0],
        "out_channels": out_channels,
        "kernel_size": 1,
        "dilation": 1,
        "n_layers": 1,
    }

    # default per_level_cln (None) enables CLN at every level
    decoder = UNetDecoder(
        conv_block=conv_block,
        up_sampling_block=up_sampling_block,
        output_layer=output_layer,
        recurrent_block=None,
        n_channels=n_channels,
        output_channels=out_channels,
    ).to(device)

    invars = []
    size = hw_size
    for idx in range(len(n_channels) - 1, -1, -1):
        invars.append(torch.rand([12, n_channels[idx], size, size]).to(device))
        size = size // 2

    with pytest.raises(ValueError, match="Conditional inputs are required"):
        decoder(invars, conditions_cln=None)

    del decoder, invars
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    ("mask", "expected_calls"),
    [([True, False, True], 2), ([False, True, False], 1)],
    ids=["tft-2calls", "ftf-1call"],
)
def test_UNetDecoder_per_level_checkpointing_applied_only_at_configured_levels(
    device, monkeypatch, mask, expected_calls, pytestconfig
):
    """Verify per_level_checkpointing routes only the configured levels through
    torch.utils.checkpoint.checkpoint(), not all (or none) of the decoder levels.
    """
    from torch.utils.checkpoint import checkpoint as real_checkpoint

    from physicsnemo.models.dlwp_healpix.layers import (
        BasicConvBlock,
        ConvNeXtBlock,
        TransposedConvUpsample,
        UNetDecoder,
    )

    in_channels = 2
    out_channels = 1
    hw_size = 16
    n_channels = (8, 8, 8)

    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": in_channels,
    }
    up_sampling_block = {
        "_target_": TransposedConvUpsample,
        "in_channels": in_channels,
        "out_channels": in_channels,
        "upsampling": 2,
    }
    output_layer = {
        "_target_": BasicConvBlock,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": 1,
        "dilation": 1,
        "n_layers": 1,
    }

    call_count = {"n": 0}

    def spy_checkpoint(*args, **kwargs):
        call_count["n"] += 1
        return real_checkpoint(*args, **kwargs)

    monkeypatch.setattr(
        "physicsnemo.models.dlwp_healpix.layers.healpix_decoder.checkpoint",
        spy_checkpoint,
    )

    decoder = UNetDecoder(
        conv_block=conv_block,
        up_sampling_block=up_sampling_block,
        output_layer=output_layer,
        recurrent_block=None,
        n_channels=n_channels,
        per_level_checkpointing=mask,
    ).to(device)

    invars = []
    size = hw_size
    for idx in range(len(n_channels) - 1, -1, -1):
        invars.append(torch.rand([12, n_channels[idx], size, size]).to(device))
        size = size // 2

    decoder(invars)

    assert call_count["n"] == expected_calls, (
        f"mask={mask}: expected {expected_calls} checkpoint() call(s), "
        f"got {call_count['n']}"
    )

    del decoder, invars
    torch.cuda.empty_cache()


def test_UNetDecoder_per_level_checkpointing_matches_no_checkpointing(
    device, pytestconfig
):
    """A mixed (non-uniform) per_level_checkpointing pattern must produce the
    same forward output and gradients as no checkpointing at all, using the
    exact same model weights.
    """
    from physicsnemo.models.dlwp_healpix.layers import (
        BasicConvBlock,
        ConvNeXtBlock,
        TransposedConvUpsample,
        UNetDecoder,
    )

    in_channels = 2
    out_channels = 1
    hw_size = 16
    b_size = 12
    n_channels = (8, 8, 8)

    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": in_channels,
    }
    up_sampling_block = {
        "_target_": TransposedConvUpsample,
        "in_channels": in_channels,
        "out_channels": in_channels,
        "upsampling": 2,
    }
    output_layer = {
        "_target_": BasicConvBlock,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": 1,
        "dilation": 1,
        "n_layers": 1,
    }

    decoder = UNetDecoder(
        conv_block=conv_block,
        up_sampling_block=up_sampling_block,
        output_layer=output_layer,
        recurrent_block=None,
        n_channels=n_channels,
        per_level_checkpointing=[False, False, False],
    ).to(device)

    invars = []
    size = hw_size
    for idx in range(len(n_channels) - 1, -1, -1):
        invar = torch.rand([b_size, n_channels[idx], size, size]).to(device)
        invar.requires_grad_(True)
        invars.append(invar)
        size = size // 2

    baseline = decoder(invars)
    baseline.sum().backward()
    baseline_grads = [v.grad.clone() for v in invars]
    baseline_out = baseline.detach().clone()

    for v in invars:
        v.grad = None
    decoder.zero_grad()

    # mixed pattern applied to the *same* model instance/weights
    decoder.per_level_checkpointing = [True, False, True]

    checkpointed = decoder(invars)
    checkpointed.sum().backward()

    assert common.compare_output(baseline_out, checkpointed)

    for v, baseline_grad in zip(invars, baseline_grads):
        assert torch.isfinite(v.grad).all()
        assert torch.allclose(baseline_grad, v.grad, rtol=1e-4, atol=1e-5)

    del decoder, invars, baseline, checkpointed
    torch.cuda.empty_cache()
