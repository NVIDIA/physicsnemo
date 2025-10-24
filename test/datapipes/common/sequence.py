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

import torch

import physicsnemo  # noqa: F401 for docs

Tensor = torch.Tensor
logger = logging.getLogger("__name__")


def check_sequence(
    tensor: Tensor, start_index: int, step_size: int, seq_length: int, axis: int = 0
) -> bool:
    """Checks if tensor has correct sequence. The tensor is expected to have a dimension that represents the sequence. Indexing this dimension should give a tensor of constant ints with the correct sequence number.

    Parameters
    ----------
    tensor : Tensor
        tensor to check sequence on.
    start_index : int
        expected value of first tensor in sequence
    step_size : int
        step size in sequence
    seq_length : int
        expected sequence length
    axis : int
        axis of tensor to check sequence on

    Returns
    -------
    bool
        Test passed
    """

    # convert tensors to int list
    tensor_tags = [
        int(tensor.select(axis, i).flatten()[0]) for i in range(tensor.shape[axis])
    ]

    # correct seq
    correct_seq = [step_size * i + start_index for i in range(seq_length)]

    # check if seq matches epected
    if correct_seq != tensor_tags:
        logger.warning("Sequence does not match expected")
        logger.warning(f"Expected Sequence: {correct_seq}")
        logger.warning(f"Sequence order: {tensor_tags}")
        return False
    return True
