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


import torch
import torch.nn as nn
from torch.amp import autocast
import contextlib

import numpy as np


def benchmark_model(
    model,
    x,
    target,
    optimizer,
    num_warmup=5,
    num_iterations=10,
    use_mixed_precision=False,
):
    """Benchmark forward pass and training step performance.

    Args:
        model: The model to benchmark
        x: Input tensor
        target: Target tensor for loss computation
        optimizer: Optimizer for training step
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        use_mixed_precision: Whether to use mixed precision training

    Returns:
        Tuple of (forward_time, training_time) in seconds
    """

    # Making a flexible context here to enable us to flip mixed precision on/off easily.
    if use_mixed_precision:
        context = autocast("cuda")
    else:
        context = contextlib.nullcontext()

    # HEADS UP:
    # You would use a grad scalar to do stable mixed precision in real training!
    # https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
    # With only a few iterations of training here, on synthetic data, we won't worry about it.

    # Warmup runs
    for _ in range(num_warmup):
        # Inference only
        with torch.no_grad():
            with context:
                _ = model(x)

        # Training warmup step
        optimizer.zero_grad()
        with context:
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

    # Benchmark forward pass
    torch.cuda.synchronize()
    forward_times = []

    for _ in range(num_iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        with torch.no_grad():
            with context:
                _ = model(x)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = (
            start_event.elapsed_time(end_event) / 1000.0
        )  # Convert ms to seconds
        forward_times.append(elapsed_time)

    avg_forward_time = np.mean(forward_times)

    # Benchmark training step
    torch.cuda.synchronize()
    training_times = []

    for _ in range(num_iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        optimizer.zero_grad()
        with context:
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = (
            start_event.elapsed_time(end_event) / 1000.0
        )  # Convert ms to seconds
        training_times.append(elapsed_time)

    avg_training_time = np.mean(training_times)

    return avg_forward_time, avg_training_time
