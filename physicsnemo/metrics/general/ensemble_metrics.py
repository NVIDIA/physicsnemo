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

from abc import ABC
from typing import List, Tuple, Union

import torch
import torch.distributed as dist

from physicsnemo.distributed.manager import DistributedManager

Tensor = torch.Tensor


class EnsembleMetrics(ABC):
    """Abstract class for ensemble performance related metrics

    Can be helpful for distributed and sequential computations of metrics.

    Parameters
    ----------
    input_shape : Union[Tuple[int,...], List]
        Shape of input tensors without batched dimension.
    device : torch.device, optional
        Pytorch device model is on, by default 'cpu'
    dtype : torch.dtype, optional
        Standard dtype to initialize any tensor with
    """

    def __init__(
        self,
        input_shape: Union[Tuple[int, ...], List[int]],
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.input_shape = list(input_shape)
        self.device = torch.device(device)
        self.dtype = dtype

    def _check_shape(self, inputs: Tensor) -> None:
        """
        Check input shapes for non-batched dimension.
        """
        if not all([i == s for (i, s) in zip(inputs.shape[1:], self.input_shape)]):
            raise ValueError(
                "Expected new input to have compatible shape with existing shapes but got"
                + str(inputs.shape)
                + "and"
                + str(self.input_shape)
                + "."
            )

    def __call__(self, *args):
        """
        Initial calculation for stored information. Will also overwrite previously stored data.
        """
        raise NotImplementedError("Class member must implement a __call__ method.")

    def update(self, *args):
        """
        Update initial or stored calculation with additional information.
        """
        raise NotImplementedError("Class member must implement an update method.")

    def finalize(self, *args):
        """
        Marks the end of the sequential calculation, used to finalize any computations.
        """
        raise NotImplementedError("Class member must implement a finalize method.")


def _update_mean(
    old_sum: Tensor,
    old_n: Union[int, Tensor],
    inputs: Tensor,
    batch_dim: Union[int, None] = 0,
) -> Tuple[Tensor, Union[int, Tensor]]:
    """Updated mean sufficient statistics given new data

    This method updates a running sum and number of samples with new (possibly batched)
    inputs

    Parameters
    ----------
    old_sum : Tensor
        Current, or old, running sum
    old_n : Union[int, Tensor]
        Current, or old, number of samples
    input : Tensor
        New input to add to current/old sum. May be batched, in which case the batched
        dimension must be flagged by passing an int to batch_dim.
    batch_dim : Union[int, None], optional
        Whether the new inputs contain a batch of new inputs and what dimension they are
        stored along. Will reduce all elements if None, by default 0.

    Returns
    -------
    Tuple[Tensor, Union[int, Tensor]]
        Updated (rolling sum, number of samples)
    """
    if batch_dim is None:
        inputs = torch.unsqueeze(inputs, 0)
        batch_dim = 0

    new_sum = old_sum + torch.sum(inputs, dim=batch_dim)
    new_n = old_n + inputs.shape[batch_dim]

    return new_sum, new_n


class Mean(EnsembleMetrics):
    """Utility class that computes the mean over a batched or ensemble dimension

    This is particularly useful for distributed environments and sequential computation.

    Parameters
    ----------
    input_shape : Union[Tuple, List]
        Shape of broadcasted dimensions
    """

    def __init__(self, input_shape: Union[Tuple, List], **kwargs):
        super().__init__(input_shape, **kwargs)
        self.sum = torch.zeros(self.input_shape, dtype=self.dtype, device=self.device)
        self.n = torch.zeros([1], dtype=torch.int32, device=self.device)

    def __call__(self, inputs: Tensor, dim: int = 0) -> Tensor:
        """Calculate an initial mean

        Parameters
        ----------
        inputs : Tensor
            Input data
        dim : Int
            Dimension of batched data

        Returns
        -------
        Tensor
            Mean value
        """
        if inputs.device != self.device:
            raise AssertionError(
                f"Input device, {inputs.device}, and Module device, {self.device}, must be the same."
            )
        self.sum = torch.sum(inputs, dim=dim)
        self.n = torch.as_tensor([inputs.shape[dim]], device=self.device)
        # TODO(Dallas) Move distributed calls into finalize.

        if (
            DistributedManager.is_initialized() and dist.is_initialized()
        ):  # pragma: no cover
            dist.all_reduce(self.sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.n, op=dist.ReduceOp.SUM)

        return self.sum / self.n

    def update(self, inputs: Tensor, dim: int = 0) -> Tensor:
        """Update current mean and essential statistics with new data

        Parameters
        ----------
        inputs : Tensor
            Inputs tensor
        dim : int
            Dimension of batched data

        Returns
        -------
        Tensor
            Current mean value
        """
        self._check_shape(inputs)
        if inputs.device != self.device:
            raise AssertionError(
                f"Input device, {inputs.device}, and Module device, {self.device}, must be the same."
            )

        # TODO(Dallas) Move distributed calls into finalize.
        if (
            DistributedManager.is_initialized() and dist.is_initialized()
        ):  # pragma: no cover
            # Collect local sums, n
            sums = torch.sum(inputs, dim=dim)
            n = torch.as_tensor([inputs.shape[dim]], device=self.device)

            # Reduce
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(n, op=dist.ReduceOp.SUM)

            # Update
            self.sum += sums
            self.n += n
        else:
            self.sum, self.n = _update_mean(self.sum, self.n, inputs, batch_dim=dim)
        return self.sum / self.n

    def finalize(
        self,
    ) -> Tensor:
        """Compute and store final mean

        Returns
        -------
        Tensor
            Final mean value
        """
        self.mean = self.sum / self.n

        return self.mean


def _update_var(
    old_sum: Tensor,
    old_sum2: Tensor,
    old_n: Union[int, Tensor],
    inputs: Tensor,
    batch_dim: Union[int, None] = 0,
) -> Tuple[Tensor, Tensor, Union[int, Tensor]]:
    """Updated variance sufficient statistics given new data

    This method updates a running running sum, squared sum, and number of samples with
    new (possibly batched) inputs

    Parameters
    ----------
    old_sum : Tensor
        Current, or old, running sum
    old_sum2 : Tensor
        Current, or old, running squared sum
    old_n : Union[int, Tensor]
        Current, or old, number of samples
    inputs : Tensor
        New input to add to current/old sum. May be batched, in which case the batched
        dimension must be flagged by passing an int to batch_dim.
    batch_dim : Union[int, None], optional
        Whether the new inputs contain a batch of new inputs and what dimension they are
        stored along. Will reduce all elements if None, by default 0.

    Returns
    -------
    Tuple[Tensor, Tensor, Union[int, Tensor]]
        Updated (rolling sum, rolling squared sum, number of samples)

    Note
    ----
    See "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances"
    by Chan et al.
    http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
    for details.
    """

    if batch_dim is None:
        inputs = torch.unsqueeze(inputs, 0)
        batch_dim = 0

    temp_n = inputs.shape[batch_dim]
    temp_sum = torch.sum(inputs, dim=batch_dim)
    temp_sum2 = torch.sum((inputs - temp_sum / temp_n) ** 2, dim=batch_dim)

    delta = old_sum * temp_n / old_n - temp_sum

    new_sum = old_sum + temp_sum
    new_sum2 = old_sum2 + temp_sum2
    new_sum2 += old_n / temp_n / (old_n + temp_n) * delta**2
    new_n = old_n + temp_n

    return new_sum, new_sum2, new_n


class Variance(EnsembleMetrics):
    """Utility class that computes the variance over a batched or ensemble dimension

    This is particularly useful for distributed environments and sequential computation.

    Parameters
    ----------
    input_shape : Union[Tuple, List]
        Shape of broadcasted dimensions

    Note
    ----
    See "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances"
    by Chan et al.
    http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
    for details.
    """

    def __init__(self, input_shape: Union[Tuple, List], **kwargs):
        super().__init__(input_shape, **kwargs)
        self.n = torch.zeros([1], dtype=torch.int32, device=self.device)
        self.sum = torch.zeros(self.input_shape, dtype=self.dtype, device=self.device)
        self.sum2 = torch.zeros(self.input_shape, dtype=self.dtype, device=self.device)

    def __call__(self, inputs: Tensor, dim: int = 0) -> Tensor:
        """Calculate an initial variance

        Parameters
        ----------
        inputs : Tensor
            Input data
        dim : Int
            Dimension of batched data

        Returns
        -------
        Tensor
            Unbiased variance values
        """

        if inputs.device != self.device:
            raise AssertionError(
                f"Input device, {inputs.device}, and Module device, {self.device}, must be the same."
            )
        self.sum = torch.sum(inputs, dim=dim)
        self.n = torch.as_tensor([inputs.shape[0]], device=self.device)

        if (
            DistributedManager.is_initialized() and dist.is_initialized()
        ):  # pragma: no cover
            # Compute mean and send around.
            dist.all_reduce(self.sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.n, op=dist.ReduceOp.SUM)

            self.sum2 = torch.sum((inputs - self.sum / self.n) ** 2, dim=dim)
            dist.all_reduce(self.sum2, op=dist.ReduceOp.SUM)
        else:
            self.sum2 = torch.sum((inputs - self.sum / self.n) ** 2, dim=dim)

        if self.n < 2.0:
            return self.sum2
        else:
            return self.sum2 / (self.n - 1.0)

    def update(self, inputs: Tensor) -> Tensor:
        """Update current variance value and essential statistics with new data

        Parameters
        ----------
        inputs : Tensor
            Input data

        Returns
        -------
        Tensor
            Unbiased variance tensor
        """

        self._check_shape(inputs)
        if inputs.device != self.device:
            raise AssertionError(
                f"Input device, {inputs.device}, and Module device, {self.device}, must be the same."
            )

        new_n = torch.as_tensor([inputs.shape[0]], device=self.device)
        new_sum = torch.sum(inputs, dim=0)
        # TODO(Dallas) Move distributed calls into finalize.
        if (
            DistributedManager.is_initialized() and dist.is_initialized()
        ):  # pragma: no cover
            dist.all_reduce(new_n, op=dist.ReduceOp.SUM)
            dist.all_reduce(new_sum, op=dist.ReduceOp.SUM)
            new_sum2 = torch.sum((inputs - new_sum / new_n) ** 2, dim=0)
            dist.all_reduce(new_sum2, op=dist.ReduceOp.SUM)

        else:
            # Calculate new statistics
            new_sum2 = torch.sum((inputs - new_sum / new_n) ** 2, dim=0)

        delta = self.sum * new_n / self.n - new_sum
        # Update
        self.sum += new_sum
        self.sum2 += new_sum2
        self.sum2 += self.n / new_n / (self.n + new_n) * (delta) ** 2
        self.n += new_n
        if self.n < 2.0:
            return self.sum2
        else:
            return self.sum2 / (self.n - 1.0)

    @property
    def mean(self) -> Tensor:
        """Mean value"""
        return self.sum / self.n

    def finalize(self, std: bool = False) -> Tuple[Tensor, Tensor]:
        """Compute and store final mean and unbiased variance / std

        Parameters
        ----------
        std : bool, optional
            Compute standard deviation, by default False

        Returns
        -------
        Tensor
            Final (mean, variance/std) value
        """
        if not (self.n > 1.0):
            raise ValueError(
                "Error! In order to finalize, there needs to be at least 2 samples."
            )
        self.var = self.sum2 / (self.n - 1.0)
        if std:
            self.std = torch.sqrt(self.var)
            return self.std
        else:
            return self.var
