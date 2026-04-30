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

from collections.abc import Callable, Sequence
from itertools import combinations

import torch

_SUPPORTED_DERIVATIVE_ORDERS = (1, 2)


def _normalize_orders_core(
    derivative_orders: int | Sequence[int],
    *,
    function_name: str,
) -> tuple[int, ...]:
    """Normalize a derivative-order request into canonical order."""
    if isinstance(derivative_orders, bool):
        raise TypeError(
            f"{function_name} derivative_orders must be int or sequence[int], got bool"
        )

    if isinstance(derivative_orders, int):
        orders = (int(derivative_orders),)
    elif isinstance(derivative_orders, Sequence):
        orders = tuple(derivative_orders)
    else:
        raise TypeError(
            f"{function_name} derivative_orders must be int or sequence[int], "
            f"got {type(derivative_orders)}"
        )

    if len(orders) == 0:
        raise ValueError(f"{function_name} derivative_orders cannot be empty")

    normalized: set[int] = set()
    for order in orders:
        if isinstance(order, bool) or not isinstance(order, int):
            raise TypeError(
                f"{function_name} derivative_orders entries must be integers, got {type(order)}"
            )
        if order not in _SUPPORTED_DERIVATIVE_ORDERS:
            raise ValueError(
                f"{function_name} supports derivative orders {list(_SUPPORTED_DERIVATIVE_ORDERS)}, "
                f"got {order}"
            )
        normalized.add(order)

    return tuple(order for order in _SUPPORTED_DERIVATIVE_ORDERS if order in normalized)


def normalize_derivative_orders(
    *,
    derivative_orders: int | Sequence[int],
    function_name: str,
) -> tuple[int, ...]:
    """Resolve derivative-order request to canonical tuple."""
    return _normalize_orders_core(derivative_orders, function_name=function_name)


def normalize_include_mixed(
    *,
    include_mixed: bool,
    function_name: str,
) -> bool:
    """Validate mixed-derivative flag."""
    if not isinstance(include_mixed, bool):
        raise TypeError(f"{function_name} include_mixed must be a bool")
    return include_mixed


def validate_mixed_request(
    *,
    derivative_orders: tuple[int, ...],
    include_mixed: bool,
    ndim: int,
    function_name: str,
) -> None:
    """Validate that mixed-derivative requests are structurally valid."""
    if include_mixed and 2 not in derivative_orders:
        raise ValueError(
            f"{function_name} include_mixed is only valid when requesting 2nd derivatives"
        )
    if include_mixed and ndim < 2:
        raise ValueError(
            f"{function_name} mixed derivatives require at least 2D inputs"
        )


def compose_derivative_outputs(
    *,
    field: torch.Tensor,
    requested_orders: tuple[int, ...],
    include_mixed: bool,
    single_order_fn: Callable[[torch.Tensor, int], torch.Tensor],
) -> torch.Tensor:
    """Compose first/second/mixed derivative outputs from single-order calls."""
    outputs: list[torch.Tensor] = []
    first_terms: torch.Tensor | None = None

    if 1 in requested_orders:
        first_terms = single_order_fn(field, 1)
        outputs.extend(first_terms.unbind(0))

    if 2 in requested_orders:
        pure_second_terms = single_order_fn(field, 2)
        outputs.extend(pure_second_terms.unbind(0))

        if include_mixed:
            if first_terms is None:
                first_terms = single_order_fn(field, 1)

            for axis_i, axis_j in combinations(range(field.ndim), 2):
                mixed_ij = single_order_fn(first_terms[axis_i], 1)[axis_j]
                outputs.append(mixed_ij)

    return torch.stack(outputs, dim=0)
