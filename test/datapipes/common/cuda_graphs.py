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
from typing import Callable, Tuple, Union

import torch

import physicsnemo

Tensor = torch.Tensor
logger = logging.getLogger("__name__")


def dummy_loss_fn(data: Union[Tensor, Tuple[Tensor, ...]]):
    """Trivial summation loss for testing"""
    # Output of tensor
    if isinstance(data, torch.Tensor):
        loss = data.sum()
    # Output of tuple of tensors
    elif isinstance(data, tuple):
        # Loop through tuple of outputs
        loss = 0
        for data_tensor in data:
            # If tensor use allclose
            if isinstance(data_tensor, Tensor):
                loss = data_tensor.sum()
    else:
        logger.error(
            "Model returned invalid type for unit test, should be Tensor or Tuple[Tensor]"
        )
        loss = None
    return loss


class MiniNetwork(torch.nn.Module):
    """Mini network with one parameter for testing cuda graph support of data pipes"""

    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(1))

    def forward(self, inputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        output = tuple(self.param * invar for invar in inputs)
        return output


def check_cuda_graphs(
    datapipe: "physicsnemo.Datapipe",
    input_fn: Union[Callable, None] = None,
    iterations: int = 5,
    warmup_length: int = 3,
) -> bool:
    """Tests if a datapipe is compatable with cuda graphs

    Parameters
    ----------
    datapipe : physicsnemo.Datapipe
        PhysicsNeMo data pipe to test
    input_fn : Union[Callable, None], optional
        Input pre-processing function to produce a tuple of tensors for model inputs, by default None
    iterations : int, optional
         Number of training iterations, by default 5
     warmup_length : int, optional
         Number of earm-up iterations before CUDA graph recording, by default 3

    Returns
    -------
    bool
        Test passed

    Note
    ----
    A torch module that accepts a tuple of tensors is used for testing cuda graphs with
    the provided datapipe. If the datapipe does not provide a tuple of tensors by default,
    one should use the `input_fn` to preprocess a batch to that form.
    """
    if not datapipe.meta.cuda_graphs:
        logger.warn("Datapipe does not support cuda graphs, skipping")
        return True

    model = MiniNetwork().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def foward(in_args):
        optimizer.zero_grad()
        output = model(in_args)
        loss = dummy_loss_fn(output)
        loss.backward()

    # Warmup stream (if cuda graphs)
    warmup_stream = torch.cuda.Stream()
    with torch.cuda.stream(warmup_stream):
        for _ in range(warmup_length):
            inputs = next(iter(datapipe))
            if input_fn:
                inputs = input_fn(inputs)
            foward(inputs)
            optimizer.step()

    # Record and replay cuda graphs
    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    for i in range(iterations):
        inputs = next(iter(datapipe))
        if input_fn:
            inputs = input_fn(inputs)

        if i == 0:  # Record
            with torch.cuda.graph(g):
                foward(inputs)
        else:  # Replay
            g.replay()
        # Optimizer step outside for AMP support
        optimizer.step()

    return True
