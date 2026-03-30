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

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch


# Clone one value recursively while preserving tensor leaf semantics.
def clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        cloned = value.detach().clone()
        if value.requires_grad:
            cloned.requires_grad_(True)
        return cloned
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: clone_value(item) for key, item in value.items()}
    return value


# Clone the positional/keyword arguments for two independent backend calls.
def clone_case(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    cloned_args = clone_value(tuple(args))
    cloned_kwargs = clone_value(dict(kwargs))
    return cloned_args, cloned_kwargs


# Compare optional values while handling the shared None case.
def assert_optional_match(
    output: Any,
    reference: Any,
    compare: Callable[[Any, Any], None],
    *,
    mismatch_message: str,
) -> None:
    if output is None or reference is None:
        assert output is None and reference is None, mismatch_message
        return
    compare(output, reference)
