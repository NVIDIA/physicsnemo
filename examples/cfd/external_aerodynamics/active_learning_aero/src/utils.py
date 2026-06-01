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

"""Local helpers for the active learning example.

The training-loop helpers (tensorwise / CombinedOptimizer / autocast +
loss) are copied verbatim from
``examples/cfd/external_aerodynamics/transformer_models/src/{utils.py,train.py}``
to keep this example self-contained. If you change them upstream, sync
manually.

Contents:

* ``tensorwise`` — decorator that allows a function to take either a single
  tensor or a sequence of tensors.
* ``CombinedOptimizer`` — wraps multiple ``torch.optim.Optimizer`` instances
  behind a single Optimizer-like interface.
* ``get_autocast_context`` — returns the autocast context for a given
  precision string (``"float16"`` / ``"bfloat16"`` / ``"float8"``).
* ``cast_precisions`` — casts a tensor (or list of tensors) to the given
  precision.
* ``loss_fn`` — MSE loss used during pre-training and AL fine-tuning.
* ``padded_all_gather`` — DDP helper that gathers 2-D tensors of
  potentially different per-rank row counts into a single tensor.
"""

import functools
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.amp import autocast
from torch.optim import Optimizer

from physicsnemo.core.version_check import check_version_spec


# ---------------------------------------------------------------------------
# tensorwise decorator (copied from transformer_models/src/utils.py)
# ---------------------------------------------------------------------------

_SEQUENCE_BLOCKLIST = (torch.Tensor, str, bytes)


def _is_tensor_sequence(x):
    return isinstance(x, Sequence) and not isinstance(x, _SEQUENCE_BLOCKLIST)


def _coerce_iterable(arg):
    """Normalize iterable inputs so ``tensorwise`` can unzip any sequence-like
    object, even if it is only an iterator (e.g., zip objects of strings or
    constants).
    """
    if _is_tensor_sequence(arg):
        return arg, True
    if isinstance(arg, Iterable) and not isinstance(arg, _SEQUENCE_BLOCKLIST):
        return tuple(arg), True
    return arg, False


def tensorwise(fn):
    """Allow ``fn(tensor, ...)`` or ``fn(list-of-tensors, ...)``.

    If any argument is a sequence of tensors, apply ``fn`` elementwise.
    Non-sequence iterables (zip objects, generators of strings, etc.) are
    automatically materialized so they can participate in the elementwise
    zip as well. All sequences must be the same length.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
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
            return fn(*normalized_args, **normalized_kwargs)

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
            ith_args = [
                (a[i] if is_s else a) for a, is_s in zip(normalized_args, seq_flags)
            ]
            ith_kwargs = {
                k: (v[i] if kw_seq_flags[k] else v)
                for k, v in normalized_kwargs.items()
            }
            outs.append(fn(*ith_args, **ith_kwargs))

        return outs

    return wrapper


# ---------------------------------------------------------------------------
# Transformer Engine availability check (for fp8 autocast).
# ---------------------------------------------------------------------------

TE_AVAILABLE = check_version_spec("transformer_engine", hard_fail=False)

if TE_AVAILABLE:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
else:
    te, Format, DelayedScaling = None, None, None


# ---------------------------------------------------------------------------
# CombinedOptimizer (copied from transformer_models/src/train.py)
# ---------------------------------------------------------------------------


class CombinedOptimizer(Optimizer):
    """Combine multiple PyTorch optimizers into a single Optimizer-like interface.

    The wrapper concatenates the *param_groups* from all contained optimizers so
    that learning-rate schedulers (e.g., ReduceLROnPlateau, CosineAnnealingLR)
    operate transparently across every parameter. Only a minimal subset of the
    *torch.optim.Optimizer* API is implemented—extend as needed.

    Note:
        This will get upstreamed to physicsnemo shortly. Don't count on this
        class existing here in the future!
    """

    def __init__(
        self,
        optimizers: Sequence[Optimizer],
        torch_compile_kwargs: dict[str, Any] | None = None,
    ):
        if not optimizers:
            raise ValueError("`optimizers` must contain at least one optimizer.")

        self.optimizers = optimizers

        param_groups = [g for opt in optimizers for g in opt.param_groups]
        super().__init__(param_groups, defaults={})

        if torch_compile_kwargs is None:
            self.step_fns: list[Callable] = [opt.step for opt in optimizers]
        else:
            self.step_fns: list[Callable] = [
                torch.compile(opt.step, **torch_compile_kwargs) for opt in optimizers
            ]

    def zero_grad(self, *args, **kwargs) -> None:
        """Nullify gradients."""
        for opt in self.optimizers:
            opt.zero_grad(*args, **kwargs)

    def step(self, closure=None) -> None:
        """Execute a single optimization step across all wrapped optimizers."""
        for step_fn in self.step_fns:
            if closure is None:
                step_fn()
            else:
                step_fn(closure)

    def state_dict(self):
        """Return combined state dict from all wrapped optimizers."""
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        """Restore state dicts to all wrapped optimizers."""
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)

        self.param_groups = [g for opt in self.optimizers for g in opt.param_groups]


# ---------------------------------------------------------------------------
# Autocast / precision helpers (copied from transformer_models/src/train.py)
# ---------------------------------------------------------------------------


def get_autocast_context(precision: str) -> nullcontext:
    """Return the autocast context for the given precision string.

    Supported values:

    * ``"float16"``  → ``torch.amp.autocast("cuda", dtype=torch.float16)``
    * ``"bfloat16"`` → ``torch.amp.autocast("cuda", dtype=torch.bfloat16)``
    * ``"float8"``   → Transformer Engine fp8 autocast (only if TE is available)
    * anything else  → ``nullcontext()``
    """
    if precision == "float16":
        return autocast("cuda", dtype=torch.float16)
    elif precision == "bfloat16":
        return autocast("cuda", dtype=torch.bfloat16)
    elif precision == "float8" and TE_AVAILABLE:
        fp8_format = Format.HYBRID
        fp8_recipe = DelayedScaling(
            fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max"
        )
        return te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
    else:
        return nullcontext()


@tensorwise
def cast_precisions(tensor: torch.Tensor, precision: str) -> torch.Tensor:
    """Cast tensor(s) to the given precision.

    Accepts either a single tensor or a list of tensors and returns the same
    structure.
    """
    match precision:
        case "float16":
            dtype = torch.float16
        case "bfloat16":
            dtype = torch.bfloat16
        case _:
            dtype = None

    if dtype is not None:
        return tensor.to(dtype)
    else:
        return tensor


# ---------------------------------------------------------------------------
# Loss (copied from transformer_models/src/train.py)
# ---------------------------------------------------------------------------


@tensorwise
def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE loss used during pre-training and AL fine-tuning."""
    return torch.nn.functional.mse_loss(outputs, targets)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def padded_all_gather(local_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """All-gather 2-D tensors that may have different row counts per rank.

    Pads each rank's tensor to the global max row count with NaN, runs
    ``dist.all_gather``, then strips the padding rows. Returns the local
    tensor unchanged when distributed mode is not initialised or world
    size is 1. Assumes the input is shape ``(N_local, cols)``; padding
    rows are filled with NaN so they can be filtered from the gathered
    output by checking the first column.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return local_tensor

    local_size = torch.tensor([local_tensor.shape[0]], dtype=torch.long, device=device)
    all_sizes = [
        torch.zeros(1, dtype=torch.long, device=device)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(all_sizes, local_size)
    max_size = max(s.item() for s in all_sizes)

    cols = local_tensor.shape[1]
    padded = torch.full(
        (max_size, cols), float("nan"), dtype=local_tensor.dtype, device=device
    )
    padded[: local_tensor.shape[0]] = local_tensor

    gathered = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, padded)
    all_data = torch.cat(gathered, dim=0)

    valid_mask = ~torch.isnan(all_data[:, 0])
    return all_data[valid_mask]
