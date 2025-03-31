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

# File for common tools in shard patching
from collections.abc import Iterable

from physicsnemo.utils.version_check import check_module_requirements

check_module_requirements("physicsnemo.distributed.shard_tensor")


class UndeterminedShardingError(Exception):
    """Exception raised when operator strategy cannot be determined from input sharding."""

    pass


class MissingShardPatch(NotImplementedError):
    """Exception raised when a required sharding patch implementation is missing."""

    pass


def promote_to_iterable(input_obj, target_iterable):
    """
    Promotes an input to an iterable of the same type as a target iterable,
    unless the input is already an iterable (excluding strings).

    Args:
        input_obj: The object to promote.
        target_iterable: The target iterable whose type determines the result.

    Returns:
        An iterable of the same type as the target iterable.
    """

    # If input_obj is a string or not iterable, wrap it in the target's type.
    if isinstance(input_obj, str) or not isinstance(input_obj, Iterable):
        # Also extend it with copies to the same length:
        ret = type(target_iterable)([input_obj]) * len(target_iterable)
        return ret

    # If input_obj is already an iterable, return it as-is.
    if len(input_obj) != len(target_iterable):
        raise ValueError("Input iterable length must match target iterable length")

    return input_obj
