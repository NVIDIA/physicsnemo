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

import logging
from typing import Any, List, Optional, Union

import torch

try:
    import torch_tensorrt
    _TORCH_TRT_AVAILABLE = True
except ImportError:
    _TORCH_TRT_AVAILABLE = False

logger = logging.getLogger(__name__)


def compile_to_trt(
    model: torch.nn.Module,
    input_signature: List[torch.Tensor],
    enabled_precisions: Optional[Set[torch.dtype]] = None,
    workspace_size: int = 1 << 30,
    min_block_size: int = 3,
    **kwargs: Any,
) -> torch.nn.Module:
    """Compile a PyTorch module to TensorRT for optimized inference.

    This utility provides a high-level wrapper around Torch-TensorRT to optimize
    PhysicsNeMo models. It handles standard compilation parameters and provides
    graceful fallbacks.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to compile.
    input_signature : List[torch.Tensor]
        A list of example input tensors that define the input shapes and types.
    enabled_precisions : Set[torch.dtype], optional
        Set of precisions to enable (e.g., {torch.float32, torch.float16}).
        Defaults to {torch.float32}.
    workspace_size : int, optional
        Maximum workspace size for TensorRT in bytes, by default 1GB.
    min_block_size : int, optional
        Minimum number of operators in a sub-graph to be converted to TensorRT,
        by default 3.
    **kwargs : Any
        Additional arguments passed to torch_tensorrt.compile.

    Returns
    -------
    torch.nn.Module
        The compiled TensorRT-optimized model.

    Raises
    ------
    ImportError
        If torch_tensorrt is not installed.
    """
    if not _TORCH_TRT_AVAILABLE:
        raise ImportError(
            "torch_tensorrt is required for TensorRT compilation. "
            "Please install it using 'pip install torch-tensorrt'."
        )

    if enabled_precisions is None:
        enabled_precisions = {torch.float32}

    logger.info(f"Compiling model {model.__class__.__name__} to TensorRT...")

    # Set up compilation arguments for Torch-TRT
    compile_spec = {
        "inputs": input_signature,
        "enabled_precisions": enabled_precisions,
        "workspace_size": workspace_size,
        "min_block_size": min_block_size,
        **kwargs,
    }

    try:
        # Use torch.compile with tensorrt backend if using PyTorch 2.x style
        # or fall back to torch_tensorrt.compile for explicit conversion.
        # Here we prefer the explicit torch_tensorrt.compile for better control
        # over the conversion process in static inference scenarios.
        trt_model = torch_tensorrt.compile(model, **compile_spec)
        logger.info("TensorRT compilation successful.")
        return trt_model
    except Exception as e:
        logger.error(f"TensorRT compilation failed: {e}")
        raise e


def is_trt_available() -> bool:
    """Check if TensorRT support is available in the current environment.

    Returns
    -------
    bool
        True if torch_tensorrt is installed, False otherwise.
    """
    return _TORCH_TRT_AVAILABLE
