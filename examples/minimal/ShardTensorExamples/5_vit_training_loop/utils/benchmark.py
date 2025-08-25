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

import torch.optim as optim

from .measure_perf import benchmark_model
from .measure_memory import get_model_memory_usage


def end_to_end_benchmark(args, model, inputs, full_img_size, device, num_classes):
    x, target = inputs

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

    try:
        # Benchmark model
        forward_time, training_time = benchmark_model(
            model, x, target, optimizer, use_mixed_precision=args.use_mixed_precision
        )

        # Memory usage - measure both inference and training
        inference_memory = get_model_memory_usage(
            model, x, mode="inference", use_mixed_precision=args.use_mixed_precision
        )
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
        }

    except RuntimeError as e:
        print(f"    Error: {e}")
        # Store failed result
        results = {
            "image_size": full_img_size[0],
            "params": num_params,
            "forward_time": float("inf"),
            "training_time": float("inf"),
            "inference_memory": float("inf"),
            "training_memory": float("inf"),
            "mixed_precision": args.use_mixed_precision and torch.cuda.is_available(),
        }

    # Clear cache to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    del model, optimizer

    return results
