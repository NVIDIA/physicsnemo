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
from contextlib import nullcontext
from typing import Tuple

import torch

import physicsnemo

from .utils import compare_output, dummy_loss_fn

Tensor = torch.Tensor
logger = logging.getLogger("__name__")


@torch.no_grad()
def validate_jit(
    model: physicsnemo.Module,
    in_args: Tuple[Tensor] = (),
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> bool:
    """Check network's JIT compatibility

    This test checks if JIT works on the provided neural network.
    JIT compilation is checked as well as making sure the original
    and JIT model produce the same output.

    Parameters
    ----------
    model : physicsnemo.Module
        PhysicsNeMo module
    in_args : Tuple[Tensor], optional
        Input arguments, by default ()
    rtol : float, optional
        Relative tolerance of error allowed, by default 1e-5
    atol : float, optional
        Absolute tolerance of error allowed, by default 1e-5

    Returns
    -------
    bool
        Test passed

    Note
    ----
    JIT must be turned on in the model's meta data for this test to run.
    """
    if not model.meta.jit:
        logger.warning("Model not marked as supporting JIT, skipping")
        return True

    output = model.forward(*in_args)
    jit_model = torch.jit.script(model)
    output_jit = jit_model(*in_args)

    return compare_output(output, output_jit, rtol, atol)


def validate_cuda_graphs(
    model: physicsnemo.Module,
    in_args: Tuple[Tensor] = (),
    rtol: float = 1e-5,
    atol: float = 1e-5,
    warmup_length: int = 3,
) -> bool:
    """Check network's CUDA graphs compatibility

    This test checks if CUDA graphs works on the provided neural network.
    CUDA graph callable compiling is checked as well as making sure the original
    and CUDA graph model produce the same output.

    Parameters
    ----------
    model : physicsnemo.Module
        PhysicsNeMo module
    in_args : Tuple[Tensor], optional
        Input arguments, keywords not supported, by default ()
    rtol : float, optional
        Relative tolerance of error allowed, by default 1e-5
    atol : float, optional
        Absolute tolerance of error allowed, by default 1e-5
    warmup_length: int, optional
        Number of warm-up iterations when making graph callable

    Returns
    -------
    bool
        Test passed

    Note
    ----
    CUDA graphs graphs must be turned on in the model's meta data for this test to run.

    Note
    ----
    PyTorch's graph for the model and inputs must be completely clear! Meaning if you have
    and inputs / outputs / parameters that are not detached this will cause an error.
    """
    if not model.meta.cuda_graphs:
        logger.warning("Model not marked as supporting CUDA graphs, skipping")
        return True
    if str(model.device) == "cpu":
        logger.warning("Model on CPU, skipping cuda graph test.")
        return True

    # Regular forward pass
    with torch.no_grad():
        output = model.forward(*in_args)

    # Create callable
    graph_module = torch.cuda.make_graphed_callables(
        model, sample_args=in_args, num_warmup_iters=warmup_length
    )
    output_graph = graph_module(*in_args)

    return compare_output(output, output_graph, rtol, atol)


def validate_amp(
    model: physicsnemo.Module,
    in_args: Tuple[Tensor] = (),
    iterations: int = 3,
) -> bool:
    """Check network's AMP compatibility

    This test checks if AMP works on the provided neural network.

    Parameters
    ----------
    model : physicsnemo.Module
        PhysicsNeMo module
    in_args : Tuple[Tensor], optional
        Input arguments, keywords not supported, by default ()
    iterations: int, optional
        Number of iterations to test AMP with

    Returns
    -------
    bool
        Test passed

    Note
    ----
    AMP must be turned on in the model's meta data for this test to run.
    """
    if not model.meta.amp_cpu and str(model.device) == "cpu":
        logger.warning("Model not marked as supporting AMP CPU, skipping")
        return True
    elif not model.meta.amp_gpu:
        logger.warning("Model not marked as supporting AMP GPU, skipping")
        return True

    # Only bfloat 16 supported for CPU, also no scalar backward
    if str(model.device) == "cpu":
        amp_device = "cpu"
        amp_dtype = torch.bfloat16
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    else:
        amp_device = "cuda"
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i in range(iterations):
        with torch.autocast(amp_device, enabled=True, dtype=amp_dtype):
            optimizer.zero_grad()
            output = model.forward(*in_args)
            loss = dummy_loss_fn(output)
            # Backward call (scalar if GPU)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    return True


def validate_torch_fx() -> bool:
    """TODO"""
    return True


def validate_combo_optims(
    model: physicsnemo.Module,
    in_args: Tuple[Tensor] = (),
    iterations: int = 2,
    warmup_length: int = 11,
) -> bool:
    """Tests all model supported optimizations together

     This test will dynamically change what optimizations are used based on the model's
     meta data. This test should be regarded as the final. The goal is to just run and
     clear the method without errors.

    Parameters
     ----------
     model : physicsnemo.Module
         PhysicsNeMo module
     in_args : Tuple[Tensor], optional
         Input arguments, keywords not supported, by default ()
     iterations : int, optional
         Number of training iterations, by default 2
     warmup_length : int, optional
         Number of earm-up iterations before CUDA graph recording, by default 11

     Returns
     -------
     bool
         Test passed

     Note
     ----
     JIT will likely be phased out with Torch FX which will take priority in future.
    """

    # Override warm up length with iterations if no cuda graphs
    warmup_length = warmup_length if model.meta.cuda_graphs else iterations
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Only bfloat 16 supported for CPU, also no scalar backward
    if str(model.device) == "cpu":
        amp_enabled = model.meta.amp_cpu
        cuda_graphs_enabled = False
        amp_device = "cpu"
        amp_dtype = torch.bfloat16
        scaler = torch.cuda.amp.GradScaler(enabled=False)  # Always false on CPU
    else:
        amp_enabled = model.meta.amp_gpu
        cuda_graphs_enabled = model.meta.cuda_graphs
        amp_device = "cuda"
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # Torch script, need to save it as seperate model since TS model doesnt have meta
    if model.meta.jit:
        fwd_model = torch.jit.script(model)
    else:
        fwd_model = model

    def foward(in_args):
        """Mini-forward function to capture in cuda graph if needed"""
        # Test AMP
        # This is a conditional context statement: https://stackoverflow.com/a/34798330
        with torch.autocast(
            amp_device, enabled=True, dtype=amp_dtype
        ) if model.meta.amp else nullcontext():
            optimizer.zero_grad()
            output = fwd_model(*in_args)
            loss = dummy_loss_fn(output)
            scaler.scale(loss).backward()

    # Warmup stream (if cuda graphs)
    with torch.cuda.stream(
        torch.cuda.Stream()
    ) if cuda_graphs_enabled else nullcontext():
        for i in range(warmup_length):
            foward(in_args)
            scaler.step(optimizer)
            scaler.update()

    # Test Cuda graphs
    if cuda_graphs_enabled:
        # Record cuda graphs
        g = torch.cuda.CUDAGraph()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(g):
            foward(in_args)
        # Optimizer step outside for AMP support
        scaler.step(optimizer)
        scaler.update()

        # Replay graph
        for i in range(iterations):
            g.replay()
            scaler.step(optimizer)
            scaler.update()

    return True
