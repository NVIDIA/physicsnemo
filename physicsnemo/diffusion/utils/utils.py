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

from typing import Any, Iterator, Sequence

import numpy as np
import torch


class StackedRandomGenerator:
    """
    Wrapper for ``torch.Generator`` that allows specifying a different random
    seed for each sample in a minibatch.

    Parameters
    ----------
    device : torch.device
        Device to use for the random number generator.
    seeds : Sequence[int]
        Sequence (e.g. list or tuple) of random seeds for each sample in the
        minibatch. Its length defines the batch size of generated samples.
    """

    def __init__(self, device: torch.device, seeds: Sequence[int]):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(
        self,
        size: torch.Size | Sequence[int],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate stacked samples from a standard normal distribution. Each sample is
        generated using a different random seed.

        Parameters
        ----------
        size : Sequence[int] | torch.Size
            Size of the output tensor. Accepts any sequence of integers or a
            ``torch.Size`` instance. First dimension must match the number of
            random seeds.
        **kwargs : Any
            Additional arguments to pass to ``torch.randn``.

        Returns
        -------
        torch.Tensor
            Stacked samples from a standard normal distribution. Shape matches
            ``size``.
        """
        if size[0] != len(self.generators):
            raise ValueError(
                f"Expected first dimension of size {len(self.generators)}, got {size[0]}"
            )
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randt(
        self,
        nu: int,
        size: torch.Size | Sequence[int],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate stacked samples from a standard Student-t distribution with
        ``nu`` degrees of freedom. This is useful when sampling from heavy-tailed
        diffusion models.

        Parameters
        ----------
        nu : int
            Degrees of freedom for the Student-t distribution. Must be > 2.
        size : Sequence[int] | torch.Size
            Size of the output tensor. Accepts any sequence of integers or a
            ``torch.Size`` instance. First dimension must match the number of
            random seeds.
        **kwargs : Any
            Additional arguments to pass to ``torch.randn``.

        Returns
        -------
        torch.Tensor
            Stacked samples from a standard Student-t distribution. Shape matches
            ``size``.
        """
        # Size validation
        if size[0] != len(self.generators):
            raise ValueError(
                f"Expected first dimension of size {len(self.generators)}, got {size[0]}"
            )
        # Validation for nu
        if nu <= 2:
            raise ValueError(f"Expected nu > 2, but got {nu}.")

        # Generate samples from Student-t distribution
        chi_dist = torch.distributions.Chi2(nu)
        kappa = (
            (chi_dist.sample((len(self.generators),)) / nu)
            .view(-1, *([1] * len(size[1:])))
            .to(self.generators[0].device)
        )
        eps = torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )
        return eps / torch.sqrt(kappa)

    def randn_like(self, input: torch.Tensor) -> torch.Tensor:
        """
        Generate stacked samples from a standard normal distribution with the same
        shape and data type as the input tensor.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor to match the shape, data type, memory layout, and
            device of.

        Returns
        -------
        torch.Tensor
            Stacked samples from a standard normal distribution. Shape matches
            ``input.shape``.
        """
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(
        self,
        *args: Any,
        size: torch.Size | Sequence[int],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate stacked samples from a uniform distribution over the integers.

        Parameters
        ----------
        *args : Any
            Required positional arguments to pass to ``torch.randint``.
        size : Sequence[int] | torch.Size
            Size of the output tensor. Accepts any sequence of integers or a
            ``torch.Size`` instance. First dimension must match the number of
            random seeds.
        **kwargs : Any
            Additional keyword arguments to pass to ``torch.randint``.

        Returns
        -------
        torch.Tensor
            Stacked samples from a uniform distribution over the integers. Shape
            matches ``size``.
        """
        if size[0] != len(self.generators):
            raise ValueError(
                f"Expected first dimension of size {len(self.generators)}, got {size[0]}"
            )
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )


class InfiniteSampler(torch.utils.data.Sampler[int]):
    """Sampler for torch.utils.data.DataLoader that loops over the dataset indefinitely.

    This sampler yields indices indefinitely, optionally shuffling items as it goes.
    It can also perform distributed sampling when `rank` and `num_replicas` are
    specified.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to sample from
    rank : int, default=0
        The rank of the current process within num_replicas processes
    num_replicas : int, default=1
        The number of processes participating in distributed sampling
    shuffle : bool, default=True
        Whether to shuffle the indices
    seed : int, default=0
        Random seed for reproducibility when shuffling
    window_size : float, default=0.5
        Fraction of dataset to use as window for shuffling. Must be between 0 and 1.
        A larger window means more thorough shuffling but slower iteration.
    start_idx : int, default=0
        The initial index to use for the sampler. This is used for resuming training.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        rank: int = 0,
        num_replicas: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        window_size: float = 0.5,
        start_idx: int = 0,
    ):
        if not len(dataset) > 0:
            raise ValueError("Dataset must contain at least one item")
        if not num_replicas > 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be non-negative and less than num_replicas")
        if not 0 <= window_size <= 1:
            raise ValueError("window_size must be between 0 and 1")
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size
        self.start_idx = start_idx

    def __iter__(self) -> Iterator[int]:
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = self.start_idx
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1
