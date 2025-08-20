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


def get_model_memory_usage(model, x, target=None, optimizer=None, mode='inference', use_mixed_precision=False):
    """Estimate model memory usage for inference or training.
    
    Args:
        model: The model to measure
        x: Input tensor
        target: Target tensor (required for training mode)
        optimizer: Optimizer (required for training mode)
        mode: 'inference' or 'training'
        use_mixed_precision: Whether to use mixed precision
        
    Returns:
        Peak memory usage in GB
    """
    
    
    if use_mixed_precision:
        context = autocast("cuda")
    else:
        context = contextlib.nullcontext()
        
    torch.cuda.reset_peak_memory_stats()
    
    if mode == 'inference':
        with torch.no_grad():
            with context:
                _ = model(x)
                
    elif mode == 'training':
        if target is None or optimizer is None:
            raise ValueError("target and optimizer must be provided for training mode")
        
        optimizer.zero_grad()
        
        with context:
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
    return torch.cuda.max_memory_allocated() / 1024**3  # GB
