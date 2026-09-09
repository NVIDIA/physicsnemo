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
from functools import partial

import pytest
import torch

from test import common


@pytest.fixture
def test_data():
    # create dummy data
    def generate_test_data(faces=12, channels=2, img_size=16, device="cpu"):
        test_data = torch.eye(img_size).to(device)
        test_data = test_data[(None,) * 2]
        test_data = test_data.expand([faces, channels, -1, -1])

        return test_data

    return generate_test_data


def _cln_factory(cond_dim):
    """Build a ``ConditionalLayerNorm`` factory bound to ``cond_dim``, ready
    to pass as the ``conditional_layer_norm`` argument of a block."""
    from physicsnemo.models.dlwp_healpix.layers.normalization import (
        ConditionalLayerNorm,
    )

    return partial(ConditionalLayerNorm, condition_shape=cond_dim)


def _assert_dropout_present(module):
    """Assert at least one ``Dropout2d`` was inserted somewhere in ``module``."""
    n_dropout = sum(1 for m in module.modules() if isinstance(m, torch.nn.Dropout2d))
    assert n_dropout > 0


def _assert_dropout_eval_deterministic_train_stochastic(module, invar, out_shape):
    """Dropout must be a no-op in ``eval()`` (repeated calls match) and
    stochastic in ``train()`` (repeated calls with identical input diverge)."""
    module.eval()
    outvar_1 = module(invar)
    outvar_2 = module(invar)
    assert outvar_1.shape == out_shape
    assert common.compare_output(outvar_1, outvar_2)

    module.train()
    outvar_3 = module(invar)
    outvar_4 = module(invar)
    assert outvar_3.shape == out_shape
    assert not common.compare_output(outvar_3, outvar_4)


def test_ConvGRUBlock_initialization(device, test_data, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        ConvGRUBlock,
    )

    in_channels = 2
    conv_gru_func = ConvGRUBlock(in_channels=in_channels).to(device)
    assert isinstance(conv_gru_func, ConvGRUBlock)


def test_ConvGRUBlock_forward(device, test_data, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        ConvGRUBlock,
    )

    in_channels = 2
    tensor_size = 16
    conv_gru_func = ConvGRUBlock(in_channels=in_channels).to(device)

    invar = test_data(img_size=tensor_size, device=device)

    out_shape = torch.Size([12, in_channels, tensor_size, tensor_size])

    outvar = conv_gru_func(invar)
    assert outvar.shape == out_shape

    # check if tracking history
    outvar_hist = conv_gru_func(invar)
    assert not common.compare_output(outvar_hist, outvar)


def test_ConvGRUBlock_reset(device, test_data, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        ConvGRUBlock,
    )

    in_channels = 2
    tensor_size = 16
    conv_gru_func = ConvGRUBlock(in_channels=in_channels).to(device)

    invar = test_data(img_size=tensor_size, device=device)

    # first call establishes the hidden state after being zero-initialized
    first_call = conv_gru_func(invar)

    # subsequent call with the same input diverges because hidden state
    # is now non-zero
    second_call = conv_gru_func(invar)
    assert not common.compare_output(first_call, second_call)

    # resetting the hidden state to zero should reproduce the first call
    conv_gru_func.reset()
    reset_call = conv_gru_func(invar)
    assert common.compare_output(first_call, reset_call)


def test_ConvNeXtBlock_initialization(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        ConvNeXtBlock,
    )

    in_channels = 2
    convnext_block = ConvNeXtBlock(in_channels=in_channels).to(device)
    assert isinstance(convnext_block, ConvNeXtBlock)

    in_channels = 2
    out_channels = 2
    convnext_block = ConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert isinstance(convnext_block, ConvNeXtBlock)


def test_ConvNeXtBlock_forward(device, test_data, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        ConvNeXtBlock,
    )

    in_channels = 2
    out_channels = 1
    tensor_size = 16
    convnext_block = ConvNeXtBlock(in_channels=in_channels).to(device)

    invar = test_data(img_size=tensor_size, device=device)

    out_shape = torch.Size([12, 1, tensor_size, tensor_size])

    outvar = convnext_block(invar)
    assert outvar.shape == out_shape

    out_channels = 2
    convnext_block = ConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert outvar.shape == out_shape


def test_DoubleConvNeXtBlock_initialization(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        DoubleConvNeXtBlock,
    )

    in_channels = 2
    out_channels = 1
    latent_channels = 1
    doubleconvnextblock = DoubleConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
    ).to(device)
    assert isinstance(doubleconvnextblock, DoubleConvNeXtBlock)

    latent_channels = 2
    doubleconvnextblock = DoubleConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert isinstance(doubleconvnextblock, DoubleConvNeXtBlock)


def test_DoubleConvNeXtBlock_forward(device, test_data, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        DoubleConvNeXtBlock,
    )

    in_channels = 2
    out_channels = 1
    latent_channels = 1
    tensor_size = 16
    doubleconvnextblock = DoubleConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
    ).to(device)

    invar = test_data(img_size=tensor_size, device=device)

    out_shape = torch.Size([12, 1, tensor_size, tensor_size])

    outvar = doubleconvnextblock(invar)
    assert outvar.shape == out_shape

    latent_channels = 2
    doubleconvnextblock = DoubleConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
    ).to(device)

    outvar = doubleconvnextblock(invar)
    assert outvar.shape == out_shape


def test_DoubleConvNeXtBlock_dropout(device, test_data, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        DoubleConvNeXtBlock,
    )

    in_channels = 2
    out_channels = 1
    latent_channels = 2
    tensor_size = 16
    doubleconvnextblock = DoubleConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
        dropout=0.5,
    ).to(device)

    # dropout is inserted after every conv/norm/activation step of both
    # internal convnext blocks
    _assert_dropout_present(doubleconvnextblock)

    invar = test_data(img_size=tensor_size, device=device)
    out_shape = torch.Size([12, out_channels, tensor_size, tensor_size])
    _assert_dropout_eval_deterministic_train_stochastic(
        doubleconvnextblock, invar, out_shape
    )


def test_DoubleConvNeXtBlock_conditional_layer_norm(device, test_data, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        DoubleConvNeXtBlock,
    )

    in_channels = 2
    out_channels = 1
    latent_channels = 2
    cond_dim = 4
    tensor_size = 16
    conditional_layer_norm = _cln_factory(cond_dim)

    doubleconvnextblock = DoubleConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
        conditional_layer_norm=conditional_layer_norm,
    ).to(device)
    assert doubleconvnextblock.cln_enabled

    invar = test_data(img_size=tensor_size, device=device)
    out_shape = torch.Size([12, out_channels, tensor_size, tensor_size])

    conditions_a = torch.randn(1, cond_dim).to(device)
    conditions_b = torch.randn(1, cond_dim).to(device)

    outvar_a = doubleconvnextblock(invar, conditions_cln=conditions_a)
    outvar_b = doubleconvnextblock(invar, conditions_cln=conditions_b)

    assert outvar_a.shape == out_shape
    # different conditions must produce different normalization affine
    # parameters, and thus different outputs
    assert not common.compare_output(outvar_a, outvar_b)


def test_SymmetricConvNeXtBlock_initialization(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        SymmetricConvNeXtBlock,
    )

    in_channels = 2
    latent_channels = 1
    symmetric_convnextblock = SymmetricConvNeXtBlock(
        in_channels=in_channels,
        latent_channels=latent_channels,
    ).to(device)
    assert isinstance(symmetric_convnextblock, SymmetricConvNeXtBlock)

    latent_channels = 2
    symmetric_convnextblock = SymmetricConvNeXtBlock(
        in_channels=in_channels,
        latent_channels=latent_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert isinstance(symmetric_convnextblock, SymmetricConvNeXtBlock)


def test_SymmetricConvNeXtBlock_forward(device, test_data, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        SymmetricConvNeXtBlock,
    )

    in_channels = 2
    latent_channels = 1
    tensor_size = 16
    symmetric_convnextblock = SymmetricConvNeXtBlock(
        in_channels=in_channels,
        latent_channels=latent_channels,
    ).to(device)

    invar = test_data(img_size=tensor_size, device=device)

    out_shape = torch.Size([12, 1, tensor_size, tensor_size])

    outvar = symmetric_convnextblock(invar)
    assert outvar.shape == out_shape


def test_SymmetricConvNeXtBlock_identity_skip(device, test_data, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        SymmetricConvNeXtBlock,
    )

    channels = 2
    latent_channels = 1
    tensor_size = 16
    # in_channels == out_channels triggers the identity skip_module path
    symmetric_convnextblock = SymmetricConvNeXtBlock(
        in_channels=channels,
        out_channels=channels,
        latent_channels=latent_channels,
    ).to(device)

    invar = test_data(channels=channels, img_size=tensor_size, device=device)
    out_shape = torch.Size([12, channels, tensor_size, tensor_size])

    # identity skip_module must return the input unchanged
    assert symmetric_convnextblock.skip_module(invar) is invar

    outvar = symmetric_convnextblock(invar)
    assert outvar.shape == out_shape


def test_SymmetricConvNeXtBlock_no_skip_connection(device, test_data, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        SymmetricConvNeXtBlock,
    )

    in_channels = 2
    out_channels = 1
    latent_channels = 1
    tensor_size = 16

    with_skip = SymmetricConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
        use_block_skip_connection=True,
    ).to(device)
    without_skip = SymmetricConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
        use_block_skip_connection=False,
    ).to(device)
    # share weights so the only difference between the two blocks is
    # whether the skip connection is added. `use_block_skip_connection=False`
    # never registers a `skip_module` submodule, so load non-strictly.
    without_skip.load_state_dict(with_skip.state_dict(), strict=False)

    invar = test_data(channels=in_channels, img_size=tensor_size, device=device)

    out_with_skip = with_skip(invar)
    out_without_skip = without_skip(invar)

    assert not common.compare_output(out_with_skip, out_without_skip)
    assert common.compare_output(
        out_with_skip, out_without_skip + with_skip.skip_module(invar)
    )


def test_SymmetricConvNeXtBlock_dropout(device, test_data, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        SymmetricConvNeXtBlock,
    )

    in_channels = 2
    latent_channels = 1
    tensor_size = 16
    symmetric_convnextblock = SymmetricConvNeXtBlock(
        in_channels=in_channels,
        latent_channels=latent_channels,
        dropout=0.5,
    ).to(device)

    _assert_dropout_present(symmetric_convnextblock)

    invar = test_data(img_size=tensor_size, device=device)
    out_shape = torch.Size([12, 1, tensor_size, tensor_size])
    _assert_dropout_eval_deterministic_train_stochastic(
        symmetric_convnextblock, invar, out_shape
    )


def test_SymmetricConvNeXtBlock_conditional_layer_norm(device, test_data, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        SymmetricConvNeXtBlock,
    )

    in_channels = 2
    latent_channels = 1
    cond_dim = 4
    tensor_size = 16
    conditional_layer_norm = _cln_factory(cond_dim)

    symmetric_convnextblock = SymmetricConvNeXtBlock(
        in_channels=in_channels,
        latent_channels=latent_channels,
        conditional_layer_norm=conditional_layer_norm,
    ).to(device)
    assert symmetric_convnextblock.cln_enabled

    invar = test_data(img_size=tensor_size, device=device)
    out_shape = torch.Size([12, 1, tensor_size, tensor_size])

    conditions_a = torch.randn(1, cond_dim).to(device)
    conditions_b = torch.randn(1, cond_dim).to(device)

    outvar_a = symmetric_convnextblock(invar, conditions_cln=conditions_a)
    outvar_b = symmetric_convnextblock(invar, conditions_cln=conditions_b)

    assert outvar_a.shape == out_shape
    assert not common.compare_output(outvar_a, outvar_b)


def test_Multi_SymmetricConvNeXtBlock_initialization(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        Multi_SymmetricConvNeXtBlock,
    )

    in_channels = 2
    latent_channels = 1
    multi_symmetric_convnextblock = Multi_SymmetricConvNeXtBlock(
        in_channels=in_channels,
        latent_channels=latent_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert isinstance(multi_symmetric_convnextblock, Multi_SymmetricConvNeXtBlock)


def test_Multi_SymmetricConvNeXtBlock_forward(device, test_data, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        Multi_SymmetricConvNeXtBlock,
    )

    in_channels = 2
    latent_channels = 1
    tensor_size = 16
    multi_symmetric_convnextblock = Multi_SymmetricConvNeXtBlock(
        in_channels=in_channels,
        latent_channels=latent_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert isinstance(multi_symmetric_convnextblock, Multi_SymmetricConvNeXtBlock)

    invar = test_data(img_size=tensor_size, device=device)

    out_shape = torch.Size([12, 1, tensor_size, tensor_size])

    outvar = multi_symmetric_convnextblock(invar)
    assert outvar.shape == out_shape


def test_Multi_SymmetricConvNeXtBlock_n_layers(device, test_data, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        Multi_SymmetricConvNeXtBlock,
        SymmetricConvNeXtBlock,
    )

    in_channels = 2
    out_channels = 3
    latent_channels = 1
    n_layers = 3
    tensor_size = 16
    multi_symmetric_convnextblock = Multi_SymmetricConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
        n_layers=n_layers,
    ).to(device)

    assert len(multi_symmetric_convnextblock.blocks) == n_layers
    for block in multi_symmetric_convnextblock.blocks:
        assert isinstance(block, SymmetricConvNeXtBlock)
    # only the first block consumes in_channels, all subsequent blocks
    # operate on out_channels
    first_conv = multi_symmetric_convnextblock.blocks[0].convblock[0].layers[-1]
    assert first_conv.in_channels == in_channels
    later_conv = multi_symmetric_convnextblock.blocks[1].convblock[0].layers[-1]
    assert later_conv.in_channels == out_channels

    invar = test_data(channels=in_channels, img_size=tensor_size, device=device)
    out_shape = torch.Size([12, out_channels, tensor_size, tensor_size])

    outvar = multi_symmetric_convnextblock(invar)
    assert outvar.shape == out_shape


def test_Multi_SymmetricConvNeXtBlock_dropout(device, test_data, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        Multi_SymmetricConvNeXtBlock,
    )

    in_channels = 2
    latent_channels = 1
    n_layers = 2
    tensor_size = 16
    multi_symmetric_convnextblock = Multi_SymmetricConvNeXtBlock(
        in_channels=in_channels,
        latent_channels=latent_channels,
        n_layers=n_layers,
        dropout=0.5,
    ).to(device)

    # dropout must be forwarded to every wrapped SymmetricConvNeXtBlock
    _assert_dropout_present(multi_symmetric_convnextblock)

    invar = test_data(channels=in_channels, img_size=tensor_size, device=device)
    out_shape = torch.Size([12, 1, tensor_size, tensor_size])
    _assert_dropout_eval_deterministic_train_stochastic(
        multi_symmetric_convnextblock, invar, out_shape
    )


def test_Multi_SymmetricConvNeXtBlock_conditional_layer_norm(
    device, test_data, pytestconfig
):
    from physicsnemo.models.dlwp_healpix.layers import (
        Multi_SymmetricConvNeXtBlock,
    )

    in_channels = 2
    latent_channels = 1
    cond_dim = 4
    n_layers = 2
    tensor_size = 16
    conditional_layer_norm = _cln_factory(cond_dim)

    multi_symmetric_convnextblock = Multi_SymmetricConvNeXtBlock(
        in_channels=in_channels,
        latent_channels=latent_channels,
        n_layers=n_layers,
        conditional_layer_norm=conditional_layer_norm,
    ).to(device)
    assert multi_symmetric_convnextblock.cln_enabled
    # cln must be propagated to every sub-block
    for block in multi_symmetric_convnextblock.blocks:
        assert block.cln_enabled

    invar = test_data(channels=in_channels, img_size=tensor_size, device=device)
    out_shape = torch.Size([12, 1, tensor_size, tensor_size])

    conditions_a = torch.randn(1, cond_dim).to(device)
    conditions_b = torch.randn(1, cond_dim).to(device)

    outvar_a = multi_symmetric_convnextblock(invar, conditions_cln=conditions_a)
    outvar_b = multi_symmetric_convnextblock(invar, conditions_cln=conditions_b)

    assert outvar_a.shape == out_shape
    assert not common.compare_output(outvar_a, outvar_b)


def test_BasicConvBlock_initialization(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        BasicConvBlock,
    )

    in_channels = 3
    out_channels = 1
    latent_channels = 2
    conv_block = BasicConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
    ).to(device)
    assert isinstance(conv_block, BasicConvBlock)

    # test w/ activation and latent channels
    conv_block = BasicConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert isinstance(conv_block, BasicConvBlock)


def test_BasicConvBlock_forward(device, test_data, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        BasicConvBlock,
    )

    in_channels = 3
    out_channels = 1
    tensor_size = 16
    conv_block = BasicConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
    ).to(device)

    invar = test_data(
        channels=in_channels, faces=24, img_size=tensor_size, device=device
    )

    outvar = conv_block(invar)
    out_shape = torch.Size([24, out_channels, tensor_size, tensor_size])

    assert outvar.shape == out_shape


def test_TransposedConvUpsample_initialization(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        TransposedConvUpsample,  #
    )

    transposed_conv_upsample_block = TransposedConvUpsample().to(device)
    assert isinstance(transposed_conv_upsample_block, TransposedConvUpsample)

    transposed_conv_upsample_block = TransposedConvUpsample(
        activation=torch.nn.ReLU()
    ).to(device)
    assert isinstance(transposed_conv_upsample_block, TransposedConvUpsample)


def test_TransposedConvUpsample_forward(device, test_data, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        TransposedConvUpsample,
    )

    in_channels = 2
    out_channels = 1
    size = 16

    transposed_conv_upsample_block = TransposedConvUpsample(
        in_channels=in_channels,
        out_channels=out_channels,
    ).to(device)

    invar = test_data(faces=1, channels=in_channels, img_size=size, device=device)
    outsize = torch.Size([1, out_channels, size * 2, size * 2])

    outvar = transposed_conv_upsample_block(invar)
    assert outvar.shape == outsize

    transposed_conv_upsample_block = TransposedConvUpsample(
        activation=torch.nn.ReLU()
    ).to(device)

    invar = test_data(faces=1, channels=(in_channels + 1), img_size=size, device=device)
    outvar = transposed_conv_upsample_block(invar)
    assert outvar.shape == outsize


def test_Interpolate_initialization(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        Interpolate,
    )

    scale = 2
    mode = "linear"
    interpolation_block = Interpolate(scale_factor=scale, mode=mode).to(device)
    assert isinstance(interpolation_block, Interpolate)


def test_Interpolate_forward(device, pytestconfig):
    from physicsnemo.models.dlwp_healpix.layers import (
        Interpolate,
    )

    scale = 2
    mode = "linear"
    interpolation_block = Interpolate(scale_factor=scale, mode=mode).to(device)

    tensor_size = torch.randint(low=2, high=4, size=(3,)).tolist()
    invar = torch.rand(tensor_size).to(device)

    outvar = torch.nn.functional.interpolate(
        invar,
        scale_factor=scale,
        mode=mode,
    ).to(device)

    assert common.compare_output(outvar, interpolation_block(invar))
