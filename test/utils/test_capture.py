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


import logging

import pytest
import torch
import torch.nn as nn

from physicsnemo.distributed import DistributedManager
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.utils import StaticCaptureEvaluateNoGrad, StaticCaptureTraining
from physicsnemo.utils.capture import _StaticCapture

optimizers = pytest.importorskip("apex.optimizers")

Tensor = torch.Tensor
logger = logging.getLogger("__name__")


@pytest.fixture
def model():
    return FullyConnected(2, 64, 2)


@pytest.fixture
def model2():
    return FullyConnected(2, 32, 2)


@pytest.fixture
def logger():
    logger = logging.getLogger("launch")
    formatter = logging.Formatter(
        "[%(asctime)s - %(name)s - %(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    streamhandler.setLevel(logging.INFO)
    logger.addHandler(streamhandler)
    return logger


@pytest.mark.parametrize(
    "optim_type, device",
    [("pytorch", "cuda:0"), ("apex", "cuda:0"), ("pytorch", "cpu")],
)
@pytest.mark.parametrize("use_graphs", [True, False])
@pytest.mark.parametrize(
    "use_amp, amp_type",
    [(True, torch.float16), (True, torch.bfloat16), (False, torch.float16)],
)
@pytest.mark.parametrize("gradient_clip_norm", [None, 0.10])
def test_capture_training(
    model,
    logger,
    device,
    optim_type,
    use_graphs,
    use_amp,
    amp_type,
    gradient_clip_norm,
):
    # Initialize the DistributedManager first since StaticCaptureTraining uses it
    DistributedManager.initialize()

    model = model.to(device)
    input = torch.rand(8, 2).to(device)
    output = torch.rand(8, 2).to(device)
    # Set up optimizer
    if optim_type == "pytorch":
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        if optimizers:
            optim = optimizers.FusedAdam(model.parameters(), lr=0.001)
        else:
            logger.warning("Apex not installed, skipping fused Adam tests")
            return

    # Create training step function with optimization wrapper
    @StaticCaptureTraining(
        model=model,
        optim=optim,
        logger=logger,
        use_graphs=use_graphs,
        use_amp=use_amp,
        cuda_graph_warmup=1,
        amp_type=amp_type,
        gradient_clip_norm=gradient_clip_norm,
    )
    def training_step(invar, outvar):
        predvar = model(invar)
        loss = torch.sum(torch.pow(predvar - outvar, 2))
        return loss

    # Sample training loop
    for i in range(3):
        loss = training_step(input, output)
        input.copy_(torch.rand(8, 2).to(device))
        assert loss > 0, "MSE loss should always be larger than zero"

        for param in model.parameters():
            is_nan = torch.any(torch.isnan(param.grad.data))
            if gradient_clip_norm is not None and not is_nan:
                assert param.grad.data.norm(2) < gradient_clip_norm


@pytest.mark.parametrize(
    "optim_type, device",
    [("pytorch", "cuda:0"), ("apex", "cuda:0"), ("pytorch", "cpu")],
)
@pytest.mark.parametrize("use_graphs", [True, False])
@pytest.mark.parametrize(
    "use_amp, amp_type",
    [(True, torch.float16), (True, torch.bfloat16), (False, torch.float16)],
)
@pytest.mark.parametrize("gradient_clip_norm", [None, 0.10])
def test_capture_training_meta(
    model,
    logger,
    device,
    optim_type,
    use_graphs,
    use_amp,
    amp_type,
    gradient_clip_norm,
):
    # Initialize the DistributedManager first since StaticCaptureTraining uses it
    DistributedManager.initialize()

    model = model.to(device)
    input = torch.rand(8, 2).to(device)
    output = torch.rand(8, 2).to(device)
    # Set up optimizer
    if optim_type == "pytorch":
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        if optimizers:
            optim = optimizers.FusedAdam(model.parameters(), lr=0.001)
        else:
            logger.warning("Apex not installed, skipping fused Adam tests")
            return

    # Test control via meta data
    model.meta.cuda_graphs = use_graphs
    model.meta.amp_gpu = use_amp
    model.meta.amp_cpu = use_amp

    # Create training step function with optimization wrapper
    @StaticCaptureTraining(
        model=model,
        optim=optim,
        logger=logger,
        cuda_graph_warmup=1,
        gradient_clip_norm=gradient_clip_norm,
    )
    def training_step(invar, outvar):
        predvar = model(invar)
        loss = torch.sum(torch.pow(predvar - outvar, 2))
        return loss

    # Sample training loop
    for i in range(3):
        loss = training_step(input, output)
        input.copy_(torch.rand(8, 2).to(device))
        assert loss > 0, "MSE loss should always be larger than zero"

        for param in model.parameters():
            is_nan = torch.any(torch.isnan(param.grad.data))
            if gradient_clip_norm is not None and not is_nan:
                assert param.grad.data.norm(2) < gradient_clip_norm


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_graphs", [True, False])
@pytest.mark.parametrize(
    "use_amp, amp_type",
    [(True, torch.float16), (True, torch.bfloat16), (False, torch.float16)],
)
def test_capture_evaluate(
    model,
    logger,
    device,
    use_graphs,
    use_amp,
    amp_type,
):
    model = model.to(device)
    input = torch.rand(8, 2).to(device)

    # Create eval step function with optimization wrapper
    @StaticCaptureEvaluateNoGrad(
        model=model,
        logger=logger,
        use_graphs=use_graphs,
        use_amp=use_amp,
        cuda_graph_warmup=1,
        amp_type=amp_type,
    )
    def eval_step(invar):
        predvar = model(invar)
        return predvar

    # Sample eval loop
    for i in range(3):
        predvar = eval_step(input)
        input.copy_(torch.rand(8, 2).to(device))
        assert predvar.shape == torch.Size((8, 2))


def test_capture_errors():
    # Test fail cases when capture should error
    model = torch.nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )
    try:
        StaticCaptureEvaluateNoGrad(model=model)
        raise AssertionError(
            "Static capture should error if model is not PhysicsNeMo.Module"
        )
    except ValueError:
        pass


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_capture_scaler_checkpointing(model, model2, device):
    # Testing the class variables of AMP grad scaler for checkpointing
    #
    model = model.to(device)
    model2 = model2.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    optim2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    _StaticCapture.reset_state()
    # Test if it can ignore invalid scalar dicts
    capture1 = StaticCaptureTraining(model=model, optim=optim)
    capture2 = StaticCaptureTraining(model=model2, optim=optim2)
    state_dict = _StaticCapture.state_dict().copy()

    # Reset state
    del capture1
    del capture2
    _StaticCapture.reset_state()

    # Load state dict
    _StaticCapture.load_state_dict(state_dict)
    StaticCaptureTraining(model=model, optim=optim)
    StaticCaptureTraining(model=model2, optim=optim2)

    assert state_dict == _StaticCapture.state_dict()


@pytest.mark.parametrize("device", ["cuda:0"])
def test_capture_scaler_checkpointing_ordering(model, model2, device):
    # Testing the class variables of AMP grad scaler for checkpointing
    #
    model = model.to(device)
    model2 = model2.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    optim2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    _StaticCapture.reset_state()
    # Hard code some non-default attributes for testing
    capture1a = StaticCaptureTraining(model=model, optim=optim, label="capture1")
    capture1a.scaler._init_scale = 2.0
    capture1a.scaler._growth_factor = 1.0
    capture2a = StaticCaptureTraining(model=model2, optim=optim2, label="capture2")
    capture2a.scaler._init_scale = 3.0
    capture2a.scaler._growth_factor = 4.0
    state_dict = _StaticCapture.state_dict().copy()

    # Reset state
    _StaticCapture.reset_state()

    # Create new captures and make sure they are not the same
    # Change instantiation order
    capture2b = StaticCaptureTraining(model=model2, optim=optim2, label="capture2")
    capture1b = StaticCaptureTraining(model=model, optim=optim, label="capture1")
    assert not capture1a.scaler.state_dict() == capture1b.scaler.state_dict()
    assert not capture2a.scaler.state_dict() == capture2b.scaler.state_dict()
    # Load state dict
    _StaticCapture.load_state_dict(state_dict)

    # Compar
    assert capture1a.scaler.state_dict() == capture1b.scaler.state_dict()
    assert capture2a.scaler.state_dict() == capture2b.scaler.state_dict()
