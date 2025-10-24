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
from typing import Tuple

import torch

import physicsnemo  # noqa: F401 for docs

Tensor = torch.Tensor
logger = logging.getLogger("__name__")


def check_shuffle(
    tensors: Tuple[Tensor, ...], shuffle: bool, stride: int, correct_length: int
) -> bool:
    """Checks if list of tensors is shuffled or not

    Parameters
    ----------
    tensors : Tuple[Tensor, ...]
        tuple of tensors. Each tensor is expected to be constant and have an int value
        coresponding to its place in the dataset.
    stride : int
        stride of the datapipe
    shuffle : bool
        if the list of tensors is expected to be shuffled or not
    correct_length: int
        Expected number of tensors

    Returns
    -------
    bool
        Test passed
    """

    # convert tensors to int list
    tensor_tags = [int(t.flatten()[0]) for t in tensors]

    # check if number of samples has correct length
    if correct_length != len(tensor_tags):
        logger.warning("Number of samples not matching expected")
        logger.warning(f"Expected Number of Samples: {correct_length}")
        logger.warning(f"Number of Samples: {len(tensor_tags)}")
        return False

    expected_tags = list(range(correct_length))

    # check if shuffle is false
    if not shuffle:
        if tensor_tags != expected_tags:
            logger.warning("Shuffle is set to False however samples are not in order")
            logger.warning(f"Expected order: {expected_tags}")
            logger.warning(f"Sample order: {tensor_tags}")
            return False

    # check if shuffle is True
    if shuffle:
        if sorted(tensor_tags) != expected_tags:
            logger.warning(
                "Shuffle is set to True however sorted samples don't match expected"
            )
            logger.warning(f"Expected order: {expected_tags}")
            logger.warning(f"Sorted Sample order: {sorted(tensor_tags)}")
            return False
    return True
