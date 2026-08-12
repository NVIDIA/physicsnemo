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

"""Weighted and uniform multinomial index sampling."""

from typing import Literal

import torch

from physicsnemo.core.function_spec import FunctionSpec

SamplingStrategy = Literal["exact", "poisson_gap"]

_SAMPLE_CHUNK_SIZE = 1 << 22
_RANDPERM_POPULATION_LIMIT = 1 << 24


def _sample_exact_without_replacement(
    population_size: int,
    num_samples: int,
    *,
    weights: torch.Tensor | None,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Run exact sampling with randperm or a chunked exponential race."""
    if weights is None and population_size <= _RANDPERM_POPULATION_LIMIT:
        return torch.randperm(
            population_size,
            device=device,
            generator=generator,
        )[:num_samples]

    candidate_keys = []
    candidate_indices = []
    key_dtype = weights.dtype if weights is not None else torch.get_default_dtype()

    for start in range(0, population_size, _SAMPLE_CHUNK_SIZE):
        stop = min(start + _SAMPLE_CHUNK_SIZE, population_size)
        keys = torch.empty(stop - start, dtype=key_dtype, device=device)
        keys.exponential_(generator=generator)
        if weights is not None:
            chunk_weights = weights[start:stop]
            keys.div_(chunk_weights)
            keys.masked_fill_(chunk_weights == 0, torch.inf)

        local_count = min(num_samples, stop - start)
        local_keys, local_indices = torch.topk(
            keys,
            local_count,
            largest=False,
            sorted=True,
        )
        candidate_keys.append(local_keys)
        candidate_indices.append(local_indices + start)

    if len(candidate_keys) == 1:
        return candidate_indices[0]

    keys = torch.cat(candidate_keys)
    indices = torch.cat(candidate_indices)
    selected = torch.topk(keys, num_samples, largest=False, sorted=True).indices
    return indices[selected]


def _sample_poisson_gap(
    population_size: int,
    num_samples: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Run the ordered, near-uniform Poisson-gap approximation."""
    if num_samples == population_size:
        return torch.arange(population_size, device=device)

    # Float64 preserves the minimum one-index separation for large populations.
    gaps = torch.empty(num_samples, device=device, dtype=torch.float64)
    gaps.exponential_(generator=generator)
    gaps *= (population_size - num_samples) / gaps.sum()
    gaps += 1.0

    indices = torch.cumsum(gaps, dim=0)
    indices -= gaps[0]
    return torch.clamp(indices.floor().long(), min=0, max=population_size - 1)


class WeightedMultinomial(FunctionSpec):
    r"""Sample indices from a weighted or uniform multinomial distribution.

    The core call follows :func:`torch.multinomial`: pass ``input``,
    ``num_samples``, ``replacement``, and optionally ``generator``. A
    one-dimensional tensor input contains relative sampling weights. Unlike
    :func:`torch.multinomial`, ``input`` may also be an integer population
    size, which represents uniform weights without materializing a tensor.
    Batched weight tensors and the ``out`` argument are not supported.

    Without replacement, the default ``"exact"`` strategy produces exact
    weighted or uniform samples. Moderate uniform populations use
    :func:`torch.randperm`; weighted and very large uniform populations use a
    chunked exponential race without the :func:`torch.multinomial`
    :math:`2^{24}` category limit. Chunking avoids a temporary allocation
    proportional to the full population when only a small sample is requested.

    The opt-in ``"poisson_gap"`` strategy draws and normalizes exponential
    gaps. It uses :math:`O(\text{num_samples})` memory regardless of population
    size, but returns ordered, near-uniform coverage rather than an exact
    uniform random subset. It is only available for integer ``input`` without
    replacement.

    With replacement, integer input uses :func:`torch.randint` and tensor input
    uses :func:`torch.multinomial`.

    Parameters
    ----------
    input : int or torch.Tensor
        An integer uniform population size or one-dimensional floating-point
        relative weights.
    num_samples : int
        Number of indices to sample.
    replacement : bool, default=False
        Whether an index may be sampled more than once.
    generator : torch.Generator, optional
        Generator used for random draws. Its device must match the sampling
        device.
    strategy : {"exact", "poisson_gap"}, default="exact"
        Sampling strategy. Approximate Poisson-gap sampling must be requested
        explicitly and is only valid for uniform sampling without replacement.
    device : torch.device or str, optional
        Output device when ``input`` is an integer. When weights are supplied,
        their device is used and an explicitly supplied device must match it.
    implementation : {"torch"}, optional
        Backend implementation. ``None`` selects the default implementation.

    Returns
    -------
    torch.Tensor
        Sampled indices with shape ``(num_samples,)`` and dtype
        ``torch.int64``. Exact samples are returned in random draw order;
        Poisson-gap samples are returned in increasing index order.
    """

    _BENCHMARK_CASES = (
        ("exact-uniform-n1m-k16k", 1 << 20, 1 << 14, "exact", False, False),
        ("poisson-gap-n1m-k16k", 1 << 20, 1 << 14, "poisson_gap", False, False),
        ("exact-weighted-n1m-k16k", 1 << 20, 1 << 14, "exact", True, False),
        (
            "replacement-uniform-n1m-k16k",
            1 << 20,
            1 << 14,
            "exact",
            False,
            True,
        ),
        (
            "replacement-weighted-n1m-k16k",
            1 << 20,
            1 << 14,
            "exact",
            True,
            True,
        ),
        (
            "exact-uniform-over-2pow24-n16m-k64k",
            (1 << 24) + 1,
            1 << 16,
            "exact",
            False,
            False,
        ),
        (
            "poisson-gap-over-2pow24-n16m-k64k",
            (1 << 24) + 1,
            1 << 16,
            "poisson_gap",
            False,
            False,
        ),
        (
            "exact-weighted-over-2pow24-n16m-k64k",
            (1 << 24) + 1,
            1 << 16,
            "exact",
            True,
            False,
        ),
    )

    @FunctionSpec.register(name="torch", rank=0, baseline=True)
    def torch_forward(
        input: int | torch.Tensor,
        num_samples: int,
        replacement: bool = False,
        *,
        generator: torch.Generator | None = None,
        strategy: SamplingStrategy = "exact",
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """PyTorch implementation of weighted multinomial sampling."""
        if not isinstance(num_samples, int) or isinstance(num_samples, bool):
            raise TypeError(
                f"num_samples must be an int, got {type(num_samples).__name__}."
            )
        if num_samples < 0:
            raise ValueError(f"num_samples must be non-negative, got {num_samples}.")
        if not isinstance(replacement, bool):
            raise TypeError(
                f"replacement must be a bool, got {type(replacement).__name__}."
            )
        if strategy not in ("exact", "poisson_gap"):
            raise ValueError(
                f"strategy must be 'exact' or 'poisson_gap', got {strategy!r}."
            )

        requested_device = torch.device(device) if device is not None else None
        weights: torch.Tensor | None
        if isinstance(input, torch.Tensor):
            weights = input
            if weights.ndim != 1:
                raise ValueError(
                    f"input weights must be 1D, got {weights.ndim}D with "
                    f"{weights.shape=}."
                )
            if not weights.is_floating_point():
                raise TypeError(
                    f"input weights must be floating point, got {weights.dtype=}."
                )
            if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
                raise RuntimeError(
                    "input weights must contain only finite, non-negative values."
                )
            positive_count = int(torch.count_nonzero(weights > 0))
            if num_samples and positive_count == 0:
                raise RuntimeError(
                    "input weights must contain at least one positive value."
                )
            if not replacement and num_samples > positive_count:
                raise RuntimeError(
                    "cannot sample more indices than positive input weights without "
                    f"replacement, got {num_samples=} and {positive_count=}."
                )
            if strategy == "poisson_gap":
                raise ValueError(
                    "poisson_gap sampling requires an integer uniform population."
                )
            if requested_device is not None and requested_device != weights.device:
                raise ValueError(
                    "device must match input.device when weights are supplied, got "
                    f"{requested_device} and {weights.device}."
                )
            population_size = weights.shape[0]
            sample_device = weights.device
        elif isinstance(input, int) and not isinstance(input, bool):
            weights = None
            population_size = input
            if population_size < 0:
                raise ValueError(
                    f"input population size must be non-negative, got {input}."
                )
            if not replacement and num_samples > population_size:
                raise ValueError(
                    "num_samples cannot exceed the input population size without "
                    f"replacement, got {num_samples=} and {population_size=}."
                )
            if replacement and num_samples and population_size == 0:
                raise ValueError(
                    "cannot sample from an empty input population with replacement."
                )
            sample_device = requested_device or torch.device("cpu")
        else:
            raise TypeError(
                "input must be an integer population size or a torch.Tensor of "
                f"weights, got {type(input).__name__}."
            )

        if strategy == "poisson_gap" and replacement:
            raise ValueError("poisson_gap sampling does not support replacement.")
        if generator is not None and torch.device(generator.device) != sample_device:
            raise ValueError(
                "generator.device must match the sampling device, got "
                f"{generator.device} and {sample_device}."
            )

        if num_samples == 0:
            return torch.empty(0, dtype=torch.long, device=sample_device)
        if replacement:
            if weights is None:
                return torch.randint(
                    population_size,
                    (num_samples,),
                    device=sample_device,
                    generator=generator,
                )
            return torch.multinomial(
                weights,
                num_samples,
                replacement=True,
                generator=generator,
            )
        if strategy == "poisson_gap":
            return _sample_poisson_gap(
                population_size,
                num_samples,
                device=sample_device,
                generator=generator,
            )
        return _sample_exact_without_replacement(
            population_size,
            num_samples,
            weights=weights,
            device=sample_device,
            generator=generator,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield exact, approximate, and replacement cases for benchmarking."""
        device = torch.device(device)
        for (
            label,
            population_size,
            num_samples,
            strategy,
            weighted,
            replacement,
        ) in cls._BENCHMARK_CASES:
            input = (
                torch.rand(population_size, device=device)
                if weighted
                else population_size
            )
            yield (
                label,
                (input, num_samples),
                {
                    "replacement": replacement,
                    "strategy": strategy,
                    "device": device,
                },
            )


weighted_multinomial = WeightedMultinomial.make_function("weighted_multinomial")


__all__ = [
    "SamplingStrategy",
    "WeightedMultinomial",
    "weighted_multinomial",
]
