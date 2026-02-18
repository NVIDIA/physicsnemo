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


from collections.abc import Iterable, Sequence
import torch
import functools

_SEQUENCE_BLOCKLIST = (torch.Tensor, str, bytes)


def _is_tensor_sequence(x):
    return isinstance(x, Sequence) and not isinstance(x, _SEQUENCE_BLOCKLIST)


def _coerce_iterable(arg):
    """
    Normalize iterable inputs so tensorwise can unzip any sequence-like object,
    even if it is only an iterator (e.g., zip objects of strings or constants).
    """
    if _is_tensor_sequence(arg):
        return arg, True
    if isinstance(arg, Iterable) and not isinstance(arg, _SEQUENCE_BLOCKLIST):
        return tuple(arg), True
    return arg, False


def tensorwise(fn):
    """
    Decorator: allow fn(tensor, ...) or fn(list-of-tensors, ...).
    If any argument is a sequence of tensors, apply fn elementwise. Non-sequence
    iterables (zip objects, generators of strings, etc.) are automatically
    materialized so they can participate in the elementwise zip as well.
    All sequences must be the same length.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Detect sequences while allowing generic iterables (e.g., zip objects)
        normalized_args = []
        seq_flags = []
        for arg in args:
            normalized_arg, is_seq = _coerce_iterable(arg)
            normalized_args.append(normalized_arg)
            seq_flags.append(is_seq)

        normalized_kwargs = {}
        kw_seq_flags = {}
        for key, value in kwargs.items():
            normalized_value, is_seq = _coerce_iterable(value)
            normalized_kwargs[key] = normalized_value
            kw_seq_flags[key] = is_seq

        any_seq = any(seq_flags) or any(kw_seq_flags.values())

        if not any_seq:
            # Nothing is a sequence — call normally
            return fn(*normalized_args, **normalized_kwargs)

        # All sequence arguments must be sequences of the same length
        # Collect all sequences (positional + keyword)
        seq_lengths = {len(a) for a, flag in zip(normalized_args, seq_flags) if flag}
        seq_lengths.update(
            len(normalized_kwargs[k]) for k, flag in kw_seq_flags.items() if flag
        )
        lengths = seq_lengths
        if len(lengths) != 1:
            raise ValueError(
                f"Sequence arguments must have same length; got lengths {lengths}."
            )

        L = lengths.pop()

        outs = []
        for i in range(L):
            # Rebuild ith positional args
            ith_args = [
                (a[i] if is_s else a) for a, is_s in zip(normalized_args, seq_flags)
            ]
            # Rebuild ith keyword args
            ith_kwargs = {
                k: (v[i] if kw_seq_flags[k] else v)
                for k, v in normalized_kwargs.items()
            }
            outs.append(fn(*ith_args, **ith_kwargs))

        return outs

    return wrapper
