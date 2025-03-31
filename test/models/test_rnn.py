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

import random

import pytest
import torch

from physicsnemo.models.rnn.rnn_one2many import One2ManyRNN
from physicsnemo.models.rnn.rnn_seq2seq import Seq2SeqRNN

from . import common


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_conv_rnn_one2many_forward(device, dimension):
    """Test model forward pass"""
    torch.manual_seed(0)
    # Construct model
    model = One2ManyRNN(
        input_channels=1,
        dimension=dimension,
        nr_latent_channels=8,
        activation_fn="relu",
        nr_downsamples=2,
        nr_tsteps=4,
    ).to(device)

    bsize = 2
    if dimension == 2:
        invar = torch.randn(bsize, 1, 1, 8, 8).to(device)
    elif dimension == 3:
        invar = torch.randn(bsize, 1, 1, 8, 8, 8).to(device)
    else:
        print("Dimension not supported")

    assert common.validate_forward_accuracy(
        model,
        (invar,),
        file_name=f"conv_rnn_one2many_{dimension}d_output.pth",
        atol=1e-4,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_conv_rnn_one2many_checkpoint(device, dimension):
    """Test model checkpoint save/load"""
    # Construct the RNN models
    model_1 = One2ManyRNN(
        input_channels=1,
        dimension=dimension,
        nr_latent_channels=4,
        activation_fn="relu",
        nr_downsamples=2,
        nr_tsteps=8,
    ).to(device)

    model_2 = One2ManyRNN(
        input_channels=1,
        dimension=dimension,
        nr_latent_channels=4,
        activation_fn="relu",
        nr_downsamples=2,
        nr_tsteps=8,
    ).to(device)

    bsize = random.randint(1, 2)
    if dimension == 2:
        invar = torch.randn(bsize, 1, 1, 8, 8).to(device)
    elif dimension == 3:
        invar = torch.randn(bsize, 1, 1, 8, 8, 8).to(device)
    else:
        print("Dimension not supported")

    assert common.validate_checkpoint(model_1, model_2, (invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_conv_rnn_one2many_optimizations(device, dimension):
    """Test model optimizations"""

    def setup_model():
        "Sets up fresh model for each optimization test"
        model = One2ManyRNN(
            input_channels=1,
            dimension=dimension,
            nr_latent_channels=8,
            activation_fn="relu",
            nr_downsamples=2,
            nr_tsteps=2,
        ).to(device)

        bsize = random.randint(1, 2)
        if dimension == 2:
            invar = torch.randn(bsize, 1, 1, 8, 8).to(device)
        elif dimension == 3:
            invar = torch.randn(bsize, 1, 1, 8, 8, 8).to(device)
        else:
            print("Dimension not supported")

        return model, invar

    # Check AMP
    model, invar = setup_model()
    assert common.validate_amp(model, (invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_conv_rnn_one2many_constructor(device):
    """Test model constructor"""

    # Define dictionary of constructor args
    arg_list = [
        {
            "input_channels": 1,
            "dimension": dimension,
            "nr_latent_channels": random.randint(4, 8),
            "activation_fn": "relu",
            "nr_downsamples": random.randint(2, 3),
            "nr_tsteps": random.randint(8, 16),
        }
        for dimension in [2, 3]
    ]

    for kw_args in arg_list:
        # Construct model
        model = One2ManyRNN(**kw_args).to(device)

        bsize = random.randint(1, 4)
        if kw_args["dimension"] == 2:
            invar = torch.randn(bsize, kw_args["input_channels"], 1, 8, 8).to(device)
        else:
            invar = torch.randn(bsize, kw_args["input_channels"], 1, 8, 8, 8).to(device)

        outvar = model(invar)
        assert outvar.shape == (
            bsize,
            kw_args["input_channels"],
            kw_args["nr_tsteps"],
            *invar.shape[3:],
        )

    # Also test failure case
    try:
        model = One2ManyRNN(
            input_channels=1,
            dimension=4,
            nr_latent_channels=32,
            activation_fn="relu",
            nr_downsamples=2,
            nr_tsteps=2,
        ).to(device)
        raise AssertionError("Failed to error for invalid dimension")
    except ValueError:
        pass


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_conv_rnn_seq2seq_forward(device, dimension):
    """Test model forward pass"""
    torch.manual_seed(0)
    # Construct model
    model = Seq2SeqRNN(
        input_channels=1,
        dimension=dimension,
        nr_latent_channels=4,
        activation_fn="relu",
        nr_downsamples=2,
        nr_tsteps=8,
    ).to(device)

    bsize = 2
    if dimension == 2:
        invar = torch.randn(bsize, 1, 8, 8, 8).to(device)
    elif dimension == 3:
        invar = torch.randn(bsize, 1, 8, 8, 8, 8).to(device)
    else:
        print("Dimension not supported")

    assert common.validate_forward_accuracy(
        model,
        (invar,),
        file_name=f"conv_rnn_seq2seq_{dimension}d_output.pth",
        atol=1e-4,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_conv_rnn_seq2seq_checkpoint(device, dimension):
    """Test model checkpoint save/load"""
    # Construct the RNN models
    model_1 = Seq2SeqRNN(
        input_channels=1,
        dimension=dimension,
        nr_latent_channels=8,
        activation_fn="relu",
        nr_downsamples=2,
        nr_tsteps=8,
    ).to(device)

    model_2 = Seq2SeqRNN(
        input_channels=1,
        dimension=dimension,
        nr_latent_channels=8,
        activation_fn="relu",
        nr_downsamples=2,
        nr_tsteps=8,
    ).to(device)

    bsize = random.randint(1, 2)
    if dimension == 2:
        invar = torch.randn(bsize, 1, 8, 8, 8).to(device)
    elif dimension == 3:
        invar = torch.randn(bsize, 1, 8, 8, 8, 8).to(device)
    else:
        print("Dimension not supported")

    assert common.validate_checkpoint(model_1, model_2, (invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_conv_rnn_seq2seq_optimizations(device, dimension):
    """Test model optimizations"""

    def setup_model():
        "Sets up fresh model for each optimization test"
        model = Seq2SeqRNN(
            input_channels=1,
            dimension=dimension,
            nr_latent_channels=4,
            activation_fn="relu",
            nr_downsamples=2,
            nr_tsteps=2,
        ).to(device)

        bsize = random.randint(1, 2)
        if dimension == 2:
            invar = torch.randn(bsize, 1, 2, 8, 8).to(device)
        elif dimension == 3:
            invar = torch.randn(bsize, 1, 2, 8, 8, 8).to(device)
        else:
            print("Dimension not supported")

        return model, invar

    # Check AMP
    model, invar = setup_model()
    assert common.validate_amp(model, (invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_conv_rnn_seq2seq_constructor(device):
    """Test model constructor"""

    # Define dictionary of constructor args
    arg_list = [
        {
            "input_channels": 1,
            "dimension": dimension,
            "nr_latent_channels": random.randint(4, 8),
            "activation_fn": "relu",
            "nr_downsamples": random.randint(2, 3),
            "nr_tsteps": random.randint(2, 4),
        }
        for dimension in [2, 3]
    ]

    for kw_args in arg_list:
        # Construct model
        model = One2ManyRNN(**kw_args).to(device)

        bsize = random.randint(1, 4)
        if kw_args["dimension"] == 2:
            invar = torch.randn(
                bsize, kw_args["input_channels"], kw_args["nr_tsteps"], 16, 16
            ).to(device)
        else:
            invar = torch.randn(
                bsize, kw_args["input_channels"], kw_args["nr_tsteps"], 16, 16, 16
            ).to(device)

        outvar = model(invar)
        assert outvar.shape == (
            bsize,
            kw_args["input_channels"],
            kw_args["nr_tsteps"],
            *invar.shape[3:],
        )

    # Also test failure case
    try:
        model = Seq2SeqRNN(
            input_channels=1,
            dimension=4,
            nr_latent_channels=4,
            activation_fn="relu",
            nr_downsamples=2,
            nr_tsteps=2,
        ).to(device)
        raise AssertionError("Failed to error for invalid dimension")
    except ValueError:
        pass
