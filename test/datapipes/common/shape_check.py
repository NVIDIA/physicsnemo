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
from typing import Tuple, Union

import torch

import physicsnemo  # noqa: F401 for docs

Tensor = torch.Tensor
logger = logging.getLogger("__name__")


def check_batch_size(
    tensors: Union[Tensor, Tuple[Tensor, ...]], batch_size: int
) -> bool:
    """Checks if tensor has correct batch size

    Parameters
    ----------
    tensors : Union[Tensor, Tuple[Tensor, ...]]
        tensors to check
    batch_size : int
        correct batch size

    Returns
    -------
    bool
        Test passed
    """
    if isinstance(tensors, Tensor):
        tensors = (tensors,)

    # Check batch size for each tensor
    for t in tensors:
        t_batch_size = t.shape[0]
        if t_batch_size != batch_size:
            logger.warning("Batch size incorrect for tensor")
            logger.warning(f"Expected Batch Size: {batch_size}")
            logger.warning(f"Tensor Batch Size: {t_batch_size}")
            return False
    return True


def check_seq_length(
    tensors: Union[Tensor, Tuple[Tensor, ...]], seq_length: int, axis: int = 1
) -> bool:
    """Checks if tensor has correct seq length

    Parameters
    ----------
    tensors : Union[Tensor, Tuple[Tensor, ...]]
        tensors to check
    seq_length : int
        correct seq length
    axis : int
        axis to check seq lenth on, default is 1

    Returns
    -------
    bool
        Test passed
    """
    if isinstance(tensors, Tensor):
        tensors = (tensors,)

    # Check seq length for each tensor
    for t in tensors:
        t_seq_length = t.shape[axis]
        if t_seq_length != seq_length:
            logger.warning("Sequence length incorrect for tensor")
            logger.warning(f"Expected Sequence Length: {seq_length}")
            logger.warning(f"Tensor Sequence Length: {t_seq_length}")
            return False
    return True


def check_channels(
    tensors: Union[Tensor, Tuple[Tensor, ...]], channels: int, axis: int = 1
) -> bool:
    """Checks if tensor has correct channels

    Parameters
    ----------
    tensors : Union[Tensor, Tuple[Tensor, ...]]
        tensors to check
    channels : int
        correct number of channels
    axis : int
        axis to check channels on, default is 1

    Returns
    -------
    bool
        Test passed
    """
    if isinstance(tensors, Tensor):
        tensors = (tensors,)

    # Check channels for each tensor
    for t in tensors:
        t_channels = t.shape[axis]
        if t_channels != channels:
            logger.warning("Number of channels incorrect for tensor")
            logger.warning(f"Expected Channels: {channels}")
            logger.warning(f"Tensor Channels: {t_channels}")
            return False
    return True


def check_grid(
    tensors: Union[Tensor, Tuple[Tensor, ...]],
    grid: Tuple[int, int],
    axis: Tuple[int, int] = (2, 3),
) -> bool:
    """Checks if tensor has correct grid dimension

    Parameters
    ----------
    tensors : Union[Tensor, Tuple[Tensor, ...]]
        tensors to check
    grid : Tuple[int, int]
        correct grid dimension
    axis : Tuple[int, int]
        axis to check grid dimension on, default is 2, 3

    Returns
    -------
    bool
        Test passed
    """
    if isinstance(tensors, Tensor):
        tensors = (tensors,)

    # Check channels for each tensor
    for t in tensors:
        t_grid = (t.shape[axis[0]], t.shape[axis[1]])
        if t_grid != grid:
            logger.warning("Grid dimension incorrect for tensor")
            logger.warning(f"Expected Grid: {grid}")
            logger.warning(f"Tensor Channels: {t_grid}")
            return False
    return True
