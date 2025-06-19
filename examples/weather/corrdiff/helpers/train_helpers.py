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

import warnings

import torch
import numpy as np
from omegaconf import ListConfig


def set_patch_shape(img_shape, patch_shape):
    img_shape_y, img_shape_x = img_shape
    patch_shape_y, patch_shape_x = patch_shape
    if (patch_shape_x is None) or (patch_shape_x > img_shape_x):
        patch_shape_x = img_shape_x
    if (patch_shape_y is None) or (patch_shape_y > img_shape_y):
        patch_shape_y = img_shape_y
    if patch_shape_x == img_shape_x and patch_shape_y == img_shape_y:
        use_patching = False
    else:
        use_patching = True
    if use_patching:
        if patch_shape_x != patch_shape_y:
            warnings.warn(
                f"You are using rectangular patches "
                f"of shape {(patch_shape_y, patch_shape_x)}, "
                f"which are an experimental feature."
            )
            raise NotImplementedError("Rectangular patch not supported yet")
        if patch_shape_x % 32 != 0 or patch_shape_y % 32 != 0:
            raise ValueError("Patch shape needs to be a multiple of 32")
    return use_patching, (img_shape_y, img_shape_x), (patch_shape_y, patch_shape_x)


def set_seed(rank):
    """
    Set seeds for NumPy and PyTorch to ensure reproducibility in distributed settings
    """
    np.random.seed(rank % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))


def configure_cuda_for_consistent_precision():
    """
    Configures CUDA and cuDNN settings to ensure consistent precision by
    disabling TensorFloat-32 (TF32) and reduced precision settings.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def compute_num_accumulation_rounds(total_batch_size, batch_size_per_gpu, world_size):
    """
    Calculate the total batch size per GPU in a distributed setting, log the batch size per GPU, ensure it's within valid limits,
    determine the number of accumulation rounds, and validate that the global batch size matches the expected value.
    """
    batch_gpu_total = total_batch_size // world_size
    batch_size_per_gpu = batch_size_per_gpu
    if batch_size_per_gpu is None or batch_size_per_gpu > batch_gpu_total:
        batch_size_per_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_size_per_gpu
    if total_batch_size != batch_size_per_gpu * num_accumulation_rounds * world_size:
        raise ValueError(
            "total_batch_size must be equal to batch_size_per_gpu * num_accumulation_rounds * world_size"
        )
    return batch_gpu_total, num_accumulation_rounds


def handle_and_clip_gradients(model, grad_clip_threshold=None):
    """
    Handles NaNs and infinities in the gradients and optionally clips the gradients.

    Parameters:
    - model (torch.nn.Module): The model whose gradients need to be processed.
    - grad_clip_threshold (float, optional): The threshold for gradient clipping. If None, no clipping is performed.
    """
    # Replace NaNs and infinities in gradients
    for param in model.parameters():
        if param.grad is not None:
            torch.nan_to_num(
                param.grad, nan=0.0, posinf=1e5, neginf=-1e5, out=param.grad
            )

    # Clip gradients if a threshold is provided
    if grad_clip_threshold is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_threshold)


def parse_model_args(args):
    """Convert ListConfig values in args to tuples."""
    return {k: tuple(v) if isinstance(v, ListConfig) else v for k, v in args.items()}


def is_time_for_periodic_task(
    cur_nimg, freq, done, batch_size, rank, rank_0_only=False
):
    """Should we perform a task that is done every `freq` samples?"""
    if rank_0_only and rank != 0:
        return False
    elif done:  # Run periodic tasks also at the end of training
        return True
    else:
        return cur_nimg % freq < batch_size
