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

"""Sampling indices without replacement."""

from typing import Literal

import torch

from physicsnemo.core.function_spec import FunctionSpec

SamplingStrategy = Literal["exact", "poisson_gap"]

_SAMPLE_CHUNK_SIZE = 1 << 22
_RANDPERM_POPULATION_LIMIT = 1 << 24


def _sample_exact(
    population_size: int,
    count: int,
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
        )[:count]

    candidate_keys = []
    candidate_indices = []
    key_dtype = weights.dtype if weights is not None else torch.get_default_dtype()
    tiny = torch.finfo(key_dtype).tiny

    for start in range(0, population_size, _SAMPLE_CHUNK_SIZE):
        stop = min(start + _SAMPLE_CHUNK_SIZE, population_size)
        keys = torch.empty(stop - start, dtype=key_dtype, device=device)
        keys.exponential_(generator=generator)
        if weights is not None:
            keys.div_(weights[start:stop].clamp_min(tiny))

        local_count = min(count, stop - start)
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
    selected = torch.topk(keys, count, largest=False, sorted=True).indices
    return indices[selected]


def _sample_poisson_gap(
    population_size: int,
    count: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Run the ordered, near-uniform Poisson-gap approximation."""
    if count == population_size:
        return torch.arange(population_size, device=device)

    # Float64 preserves the minimum one-index separation for large populations.
    gaps = torch.empty(count, device=device, dtype=torch.float64)
    gaps.exponential_(generator=generator)
    gaps *= (population_size - count) / gaps.sum()
    gaps += 1.0

    indices = torch.cumsum(gaps, dim=0)
    indices -= gaps[0]
    return torch.clamp(indices.floor().long(), min=0, max=population_size - 1)


class SampleWithoutReplacement(FunctionSpec):
    r"""Sample unique population indices with exact or approximate semantics.

    The default ``"exact"`` strategy produces exact uniform samples when
    ``weights`` is ``None`` and exact weighted samples otherwise. Moderate
    unweighted populations use :func:`torch.randperm`; weighted and very large
    unweighted populations use a chunked exponential race without the
    :func:`torch.multinomial` :math:`2^{24}` category limit. Chunking avoids a
    temporary allocation proportional to the full population when only a small
    sample is requested.

    The opt-in ``"poisson_gap"`` strategy draws and normalizes exponential
    gaps. It uses :math:`O(\text{count})` memory regardless of population size,
    but returns ordered, near-uniform coverage rather than an exact uniform
    random subset. It does not support weights.

    Parameters
    ----------
    population_size : int
        Number of candidate indices in the population.
    count : int
        Number of unique indices to sample.
    weights : torch.Tensor, optional
        One-dimensional, floating-point weights with ``population_size``
        entries. Only supported by the ``"exact"`` strategy.
    strategy : {"exact", "poisson_gap"}, default="exact"
        Sampling strategy. Approximate Poisson-gap sampling must be requested
        explicitly.
    device : torch.device or str, optional
        Output device when ``weights`` is ``None``. When weights are supplied,
        their device is used and an explicitly supplied device must match it.
    generator : torch.Generator, optional
        Generator used for random draws. Its device must match the sampling
        device.
    implementation : {"torch"}, optional
        Backend implementation. ``None`` selects the default implementation.

    Returns
    -------
    torch.Tensor
        Sampled indices with shape ``(count,)`` and dtype ``torch.int64``.
        Exact samples are returned in random draw order; Poisson-gap samples
        are returned in increasing index order.
    """

    _BENCHMARK_CASES = (
        ("exact-uniform-n1m-k16k", 1 << 20, 1 << 14, "exact", False),
        ("poisson-gap-n1m-k16k", 1 << 20, 1 << 14, "poisson_gap", False),
        ("exact-weighted-n1m-k16k", 1 << 20, 1 << 14, "exact", True),
        (
            "exact-uniform-over-2pow24-n16m-k64k",
            (1 << 24) + 1,
            1 << 16,
            "exact",
            False,
        ),
        (
            "poisson-gap-over-2pow24-n16m-k64k",
            (1 << 24) + 1,
            1 << 16,
            "poisson_gap",
            False,
        ),
        (
            "exact-weighted-over-2pow24-n16m-k64k",
            (1 << 24) + 1,
            1 << 16,
            "exact",
            True,
        ),
    )

    @FunctionSpec.register(name="torch", rank=0, baseline=True)
    def torch_forward(
        population_size: int,
        count: int,
        *,
        weights: torch.Tensor | None = None,
        strategy: SamplingStrategy = "exact",
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """PyTorch implementation of index sampling without replacement."""
        if population_size < 0:
            raise ValueError(
                f"population_size must be non-negative, got {population_size}."
            )
        if count < 0 or count > population_size:
            raise ValueError(
                f"count must be between 0 and population_size, got "
                f"{count=} and {population_size=}."
            )
        if strategy not in ("exact", "poisson_gap"):
            raise ValueError(
                f"strategy must be 'exact' or 'poisson_gap', got {strategy!r}."
            )

        requested_device = torch.device(device) if device is not None else None
        if weights is not None:
            if weights.ndim != 1:
                raise ValueError(
                    f"weights must be 1D, got {weights.ndim}D with {weights.shape=}."
                )
            if weights.shape[0] != population_size:
                raise ValueError(
                    "weights must contain population_size entries, got "
                    f"{weights.shape[0]} and {population_size}."
                )
            if not weights.is_floating_point():
                raise TypeError(
                    f"weights must be floating point, got {weights.dtype=}."
                )
            if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
                raise RuntimeError(
                    "weights must contain only finite, non-negative values."
                )
            if count and not bool((weights > 0).any()):
                raise RuntimeError("weights must contain at least one positive value.")
            if strategy == "poisson_gap":
                raise ValueError("poisson_gap sampling does not support weights.")
            if requested_device is not None and requested_device != weights.device:
                raise ValueError(
                    "device must match weights.device when weights are supplied, got "
                    f"{requested_device} and {weights.device}."
                )
            sample_device = weights.device
        else:
            sample_device = requested_device or torch.device("cpu")

        if generator is not None and torch.device(generator.device) != sample_device:
            raise ValueError(
                "generator.device must match the sampling device, got "
                f"{generator.device} and {sample_device}."
            )

        if count == 0:
            return torch.empty(0, dtype=torch.long, device=sample_device)
        if strategy == "poisson_gap":
            return _sample_poisson_gap(
                population_size,
                count,
                device=sample_device,
                generator=generator,
            )
        return _sample_exact(
            population_size,
            count,
            weights=weights,
            device=sample_device,
            generator=generator,
        )

    @classmethod
    def make_inputs_forward(cls, device: torch.device | str = "cpu"):
        """Yield exact and approximate cases for ASV benchmarking."""
        device = torch.device(device)
        for label, population_size, count, strategy, weighted in cls._BENCHMARK_CASES:
            weights = torch.rand(population_size, device=device) if weighted else None
            yield (
                label,
                (population_size, count),
                {
                    "weights": weights,
                    "strategy": strategy,
                    "device": device,
                },
            )


sample_without_replacement = SampleWithoutReplacement.make_function(
    "sample_without_replacement"
)


__all__ = [
    "SampleWithoutReplacement",
    "SamplingStrategy",
    "sample_without_replacement",
]
