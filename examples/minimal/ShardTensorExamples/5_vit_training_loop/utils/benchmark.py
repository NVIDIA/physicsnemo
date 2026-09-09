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


import torch

import torch.optim as optim
from torch.distributed.tensor import DTensor

from .measure_perf import benchmark_model
from .measure_memory import get_model_memory_usage


def end_to_end_benchmark(args, model, inputs, full_img_size, device, num_classes):
    """Run a full latency and memory benchmark for one model configuration.

    Measures forward/training time and peak memory for both inference and
    training modes, then tears down the model and frees GPU memory.
    Any error raised during benchmarking propagates to the caller with its
    full traceback (errors are no longer caught and converted to sentinels).

    Args:
        args: Parsed CLI arguments (controls warmup iters, precision, etc.).
        model: The nn.Module to benchmark.
        inputs: Tuple of (input_tensor, target_tensor).
        full_img_size: Original image dimensions used to label results.
        device: Torch device the model lives on.
        num_classes: Number of output classes (unused directly but passed
            for consistency with callers that construct the model).

    Returns:
        Dict with keys: image_size, params, forward_time, training_time,
        inference_memory, training_memory, mixed_precision.
    """
    x, target = inputs

    inference_only = getattr(args, "inference_only", False)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Create optimizer (only needed for training)
    optimizer = None
    if not inference_only:
        # AdamW's foreach path batches every param of a group into single
        # _foreach_* ops, which cannot mix plain tensors with DTensors (or
        # DTensors on different meshes).  Foreach batching never crosses param
        # groups, so split params by tensor type / mesh to keep each group
        # homogeneous while retaining the fast foreach path.
        param_groups = {}
        for p in model.parameters():
            key = p.device_mesh if isinstance(p, DTensor) else None
            param_groups.setdefault(key, []).append(p)
        optimizer = optim.AdamW(
            [{"params": params} for params in param_groups.values()],
            lr=1e-3,
            weight_decay=0.05,
        )

    # Benchmark model
    forward_time, training_time = benchmark_model(
        model,
        x,
        target,
        optimizer,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        use_mixed_precision=args.use_mixed_precision,
        inference_only=inference_only,
    )

    # Memory usage - always measure inference
    inference_memory = get_model_memory_usage(
        model, x, mode="inference", use_mixed_precision=args.use_mixed_precision
    )

    # Only measure training memory if not inference-only
    training_memory = None
    if not inference_only:
        training_memory = get_model_memory_usage(
            model,
            x,
            target,
            optimizer,
            mode="training",
            use_mixed_precision=args.use_mixed_precision,
        )

    # Store results
    results = {
        "image_size": full_img_size[0],
        "params": num_params,
        "forward_time": forward_time,
        "training_time": training_time,
        "inference_memory": inference_memory,
        "training_memory": training_memory,
        "mixed_precision": args.use_mixed_precision and torch.cuda.is_available(),
        "compile": getattr(args, "compile", False),
    }

    # Clear cache to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    del model
    if optimizer is not None:
        del optimizer

    return results
