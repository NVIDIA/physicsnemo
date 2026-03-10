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

r"""Common utilities and exceptions for ShardTensor operation patching.

This module provides base classes and utilities used across the shard patching
system, including custom exception types and helper functions for argument
handling.
"""

from collections.abc import Iterable
from typing import Any, TypeVar

T = TypeVar("T")


class UndeterminedShardingError(Exception):
    r"""Exception raised when operator strategy cannot be determined from input sharding.

    This exception is raised when a ShardTensor operation cannot determine
    the appropriate sharding strategy based on the input tensor placements.
    This typically occurs when input types are mismatched or invalid.
    """

    pass


class MissingShardPatch(NotImplementedError):
    r"""Exception raised when a required sharding patch implementation is missing.

    This exception is raised when an operation is attempted on a ShardTensor
    but the necessary sharding implementation for that operation does not exist
    or is not supported for the given configuration (e.g., kernel size, stride).
    """

    pass


def promote_to_iterable(input_obj: T, target_iterable: Any) -> T:
    r"""Promote an input to an iterable matching the type and length of a target.

    Promotes an input to an iterable of the same type as a target iterable,
    unless the input is already an iterable (excluding strings). This is useful
    for normalizing scalar arguments to match multi-dimensional parameters.

    Parameters
    ----------
    input_obj : T
        The object to promote. Can be a scalar or iterable.
    target_iterable : Any
        The target iterable whose type and length determine the result.

    Returns
    -------
    T
        An iterable of the same type as the target iterable, with the same
        length. If ``input_obj`` is a scalar, it is repeated to match the
        target length.

    Raises
    ------
    ValueError
        If ``input_obj`` is already an iterable but its length doesn't match
        the target iterable length.

    Examples
    --------
    >>> promote_to_iterable(3, (1, 2, 3))
    (3, 3, 3)
    >>> promote_to_iterable((1, 2, 3), (4, 5, 6))
    (1, 2, 3)
    """
    # Don't do anything to strings:
    if isinstance(input_obj, str):
        return input_obj

    # If input_obj is a string or not iterable, wrap it in the target's type.
    if isinstance(input_obj, str) or not isinstance(input_obj, Iterable):
        # Also extend it with copies to the same length:
        ret = type(target_iterable)([input_obj]) * len(target_iterable)
        return ret

    # If input_obj is already an iterable, return it as-is.
    if len(input_obj) != len(target_iterable):
        raise ValueError("Input iterable length must match target iterable length")

    return input_obj
