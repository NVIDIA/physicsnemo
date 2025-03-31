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


from typing import Tuple, Union

import torch
import torch.distributed as dist

from physicsnemo.distributed.manager import DistributedManager

from .ensemble_metrics import EnsembleMetrics

Tensor = torch.Tensor


@torch.jit.script
def linspace(start: Tensor, stop: Tensor, num: int) -> Tensor:  # pragma: no cover
    """Element-wise multi-dimensional linspace

    Replicates the bahaviour of numpy.linspace over all elements of multi-dimensional
    tensors in PyTorch.

    Parameters
    ----------
    start : Tensor
        Starting input Tensor
    stop : Tensor
        Ending input Tensor, should be of same size a input
    num : int
        Number of steps between start and end values between each element

    Returns
    -------
    Tensor
        Tensor of evenly spaced numbers over defined interval [num, *start.shape]
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num + 1, dtype=torch.float32, device=start.device) / (num)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but
    #  torchscript "cannot statically infer the expected size of a list in this contex",
    #  hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out


@torch.jit.script
def _low_memory_bin_reduction_counts(
    inputs: Tensor, bin_edges: Tensor, counts: Tensor, number_of_bins: int
):  # pragma: no cover
    """Computes low-memory bin counts

    This function calculates a low-memory bin count of the inputs tensor and adding the
    result to counts,  in place. The low-memory usage comes at the cost of iterating
    over the batched dimension. This bin count is done with respect to computing a
    pmf/pdf.

    Parameters
    ----------
    inputs : Tensor
        Inputs to be binned, has dimension [B, ...] where B is the batch dimension that
        the binning is done over
    bin_edges : Tensor
        Bin edges with dimension [N+1, ...] where N is the number of bins
    counts : Tensor
        Existing bin count tensor with dimension [N, ...] where N is the number of bins
    number_of_bins : int
        Number of bins

    Returns
    -------
    Tensor
        PDF bin count tensor [N, ...]
    """
    for j in range(inputs.shape[0]):
        counts[0] += (inputs[j] < bin_edges[1]).int()
    for i in range(1, number_of_bins - 1):
        for j in range(inputs.shape[0]):
            counts[i] += (inputs[j] < bin_edges[i + 1]).int() - (
                inputs[j] < bin_edges[i]
            ).int()

    for j in range(inputs.shape[0]):
        counts[number_of_bins - 1] += (
            1 - (inputs[j] < bin_edges[number_of_bins - 1]).int()
        )

    return counts


@torch.jit.script
def _high_memory_bin_reduction_counts(
    inputs: Tensor, bin_edges: Tensor, counts: Tensor, number_of_bins: int
) -> Tensor:  # pragma: no cover
    """Computes high-memory bin counts

    This function calculates a high-memory bin count of the inputs tensor and adding the
    result to counts, in place. The high-memory usage comes from computing the entire
    reduction in memory. See _low_memory_bin_reduction for an alternative. This bin
    count is done with respect to computing a pmf/pdf.

    Parameters
    ----------
    inputs : Tensor
        Inputs to be binned, has dimension [B, ...] where B is the batch dimension that the binning is done over.
    bin_edges : Tensor
        Bin edges with dimension [N+1, ...] where N is the number of bins.
    counts : Tensor
        Existing bin count tensor with dimension [N, ...] where N is the number of bins.
    number_of_bins : int
        Number of bins

    Returns
    -------
    Tensor
        PDF bin count tensor [N, ...]
    """
    counts[0] += torch.count_nonzero(inputs < bin_edges[1], dim=0)
    for i in range(1, number_of_bins - 1):
        counts[i] += torch.count_nonzero(
            inputs < bin_edges[i + 1], dim=0
        ) - torch.count_nonzero(inputs < bin_edges[i], dim=0)
    counts[number_of_bins - 1] += inputs.shape[0] - torch.count_nonzero(
        inputs < bin_edges[number_of_bins - 1], dim=0
    )
    return counts


@torch.jit.script
def _low_memory_bin_reduction_cdf(
    inputs: Tensor, bin_edges: Tensor, counts: Tensor, number_of_bins: int
) -> Tensor:  # pragma: no cover
    """Computes low-memory cumulative bin counts

    This function calculates a low-memory cumulative bin count of the inputs tensor and adding the
    result to counts, in place. The low-memory usage comes at the cost of iterating over
    the batched dimension. This bin count is done with respect to computing a cmf/cdf.

    Parameters
    ----------
    inputs : Tensor
        Inputs to be binned, has dimension [B, ...] where B is the batch dimension that
        the binning is done over
    bin_edges : Tensor
        Bin edges with dimension [N+1, ...] where N is the number of bins
    counts : Tensor
        Existing bin count tensor with dimension [N, ...] where N is the number of bins
    number_of_bins : int
        Number of bins

    Returns
    -------
    Tensor
        CDF bin count tensor [N, ...]
    """
    for i in range(number_of_bins - 1):
        for j in range(inputs.shape[0]):
            counts[i] += (inputs[j] < bin_edges[i + 1]).int()
    counts[number_of_bins - 1] += inputs.shape[0]
    return counts


@torch.jit.script
def _high_memory_bin_reduction_cdf(
    inputs: torch.Tensor,
    bin_edges: torch.Tensor,
    counts: torch.Tensor,
    number_of_bins: int,
) -> Tensor:  # pragma: no cover
    """Computes high-memory cumulative bin counts

    This function calculates a high-memory cumulative bin countof the inputs tensor and
    adding the result to counts, in place. The high-memory usage comes from computing
    the entire reduction in memory. This bin count is done with respect to computing a
    cmf/cdf.

    Parameters
    ----------
    inputs : torch.Tensor
        Inputs to be binned, has dimension [B, ...] where B is the batch dimension that
        the binning is done over.
    bin_edges : torch.Tensor
        Bin edges with dimension [N+1, ...] where N is the number of bins
    counts : torch.Tensor
        Existing bin count tensor with dimension [N, ...] where N is the number of bins
    number_of_bins : int
        Number of bins

    Returns
    -------
    Tensor
        CDF bin count tensor [N, ...]
    """
    for i in range(number_of_bins - 1):
        counts[i] += torch.count_nonzero(inputs < bin_edges[i + 1], dim=0)
    counts[number_of_bins - 1] = inputs.shape[0]
    return counts


def _count_bins(
    input: torch.Tensor,
    bin_edges: torch.Tensor,
    counts: Union[torch.Tensor, None] = None,
    cdf: bool = False,
) -> Tensor:
    """Computes (un)Cumulative bin counts of input tensor

    This function calculates the bin count of the inputs tensor and adding the result to
    counts, in place. Attempts to use a _high_memory_bin_reduction for performance
    reasons, but will fall back to less performant, less memory intensive routine.

    Parameters
    ----------
    input : Tensor
        Inputs to be binned, has dimension [B, ...] where B is the batch dimension that
        the binning is done over
    bin_edges : Tensor
        Bin edges with dimension [N+1, ...] where N is the number of bins
    counts : Union[torch.Tensor, None]
        Existing bin count tensor with dimension [N, ...] where N is the number of bins.
        If no counts is passed then we construct an empty counts, by default None
    cdf : bool, optional
        Compute a counts or cumulative counts; will calculate unnormalized counts function otherwise, by default False

    Returns
    -------
    Tensor
        CDF bin count tensor [N, ...]
    """
    bins_shape = bin_edges.shape
    number_of_bins = bins_shape[0] - 1
    if bins_shape[1:] != input.shape[1:]:
        raise ValueError(
            "Expected bin_edges and inputs to have compatible non-leading dimensions."
        )

    if counts is None:
        counts = torch.zeros(
            (number_of_bins, *bins_shape[1:]), dtype=torch.int64, device=input.device
        )
    else:
        if bins_shape[1:] != counts.shape[1:]:
            raise ValueError(
                "Expected bin_edges and counts to have compatible non-leading dimensions."
            )

    if cdf:
        try:
            counts = _high_memory_bin_reduction_cdf(
                input, bin_edges, counts, number_of_bins
            )
        except RuntimeError:
            counts = _low_memory_bin_reduction_cdf(
                input, bin_edges, counts, number_of_bins
            )
    else:
        try:
            counts = _high_memory_bin_reduction_counts(
                input, bin_edges, counts, number_of_bins
            )
        except RuntimeError:
            counts = _low_memory_bin_reduction_counts(
                input, bin_edges, counts, number_of_bins
            )

    return counts


def _get_mins_maxs(*inputs: Tensor, axis: int = 0) -> Tuple[Tensor, Tensor]:
    """Get max and min value across specified dimension

    Parameters
    ----------
    inputs : (Tensor ...)
        Input tensor(s)
    axis : int, optional
        Axis to calc min/max values with, by default 0

    Returns
    -------
    Tuple[Tensor, Tensor]
        (Minimum, Maximum) values of inputs
    """
    if len(inputs) <= 0:
        raise ValueError("At least one tensor much be provided")

    input = inputs[0]
    inputs = list(inputs)[1:]
    # Check shape consistency
    s = input.shape
    for x in inputs:
        if s[1:] != x.shape[1:]:
            raise ValueError()

    # Compute low and high for input
    low = torch.min(input, axis=axis)[0]
    high = torch.max(input, axis=axis)[0]

    # Iterative over any and all args, storing the latest inf/sup.
    for x in inputs:
        low0 = torch.min(x, axis=axis)[0]
        high0 = torch.max(x, axis=axis)[0]

        low = torch.where(low < low0, low, low0)
        high = torch.where(high > high0, high, high0)

    return low, high


def _update_bins_counts(
    input: Tensor,
    bin_edges: Tensor,
    counts: Tensor,
    cdf: bool = False,
    tol: float = 1e-2,
) -> Tuple[Tensor, Tensor]:
    """Utility for updating an existing histogram with new inputs

    Parameters
    ----------
    input : Tensor
        Input tensor of updated data for binning
    bin_edges : Tensor
        Current bin range tensor [N+1, ...] where N is the number of bins
    counts : Tensor
        Existing bin count tensor with dimension [N, ...] where N is the number of bins
    cdf : bool, optional
        Compute a histogram or a cumulative density function; will calculate probability
        density function otherwise, by default False
    tol : float, optional
        Binning tolerance, by default 1e-4

    Returns
    -------
    Tuple[Tensor, Tensor]
        Updated (bin, count) tensors
    """

    # Compute new lows and highs, compare against old bins
    low, high = _get_mins_maxs(input)

    # If in distributed environment, reduce to get extrema min and max
    if (
        DistributedManager.is_initialized() and dist.is_initialized()
    ):  # pragma: no cover
        dist.all_reduce(low, op=dist.ReduceOp.MIN)
        dist.all_reduce(high, op=dist.ReduceOp.MAX)

    low = torch.where(low < bin_edges[0], low, bin_edges[0])
    high = torch.where(high > bin_edges[-1], high, bin_edges[-1])

    # Test if bin_edges is a superset and do not recompute bin_edges.
    if ~(torch.all(low == bin_edges[0]) & torch.all(high == bin_edges[-1])):
        # There are extrema in inputs/args that are outside of bin_edges and we must recompute bin_edges and counts.

        ## Need to make sure that the new bin_edges are consistent with the old bin_edges.
        # Need to compute   dbin_edges = bin_edges[1] - bin_edges[0]
        #         find minimum k s.t. bin_edges[0] - k*dbin_edges < low
        #           set start = bin_edges[0] - k*dbin_edges
        #         find minimum k s.t. bin_edges[-1] + k*dbin_edges > high
        #          set end = bin_edges[-1] + k*dbin_edges
        dbin_edges = bin_edges[1] - bin_edges[0]
        old_number_of_bins = bin_edges.shape[0] - 1
        number_of_bins = old_number_of_bins

        lk = torch.max(torch.ceil((bin_edges[0] - low) / dbin_edges)).int().item()
        start = bin_edges[0] - lk * dbin_edges
        number_of_bins += lk

        uk = torch.max(torch.ceil((high - bin_edges[-1]) / dbin_edges)).int().item()
        end = bin_edges[-1] + uk * dbin_edges
        number_of_bins += uk

        bin_edges = linspace(start, end, number_of_bins)
        new_counts = torch.zeros(
            (number_of_bins, *bin_edges.shape[1:]),
            dtype=torch.int64,
            device=bin_edges.device,
        )

        new_counts[lk : lk + old_number_of_bins] += counts
        counts = new_counts

    # Count inputs to bins
    partial_counts = _count_bins(input, bin_edges, counts=None, cdf=cdf)
    # If in distributed environment, reduce to get extrema min and max
    if DistributedManager.is_initialized() and dist.is_initialized():
        dist.all_reduce(partial_counts, op=dist.ReduceOp.SUM)
    counts += partial_counts
    # Finally, combine the new partial counts with the existing counts
    return bin_edges, counts


def _compute_counts_cdf(
    *inputs: Tensor,
    bins: Union[int, Tensor] = 10,
    counts: Union[None, Tensor] = None,
    cdf: bool = False,
    tol: float = 1e-2,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Computes the (un)Cumulative histograms of a tensor(s) over the leading dimension

    Parameters
    ----------
    inputs : (Tensor ...)
        Input data tensor(s) [B, ...]
    bins : Union[int, Tensor], optional
        Either the number of bins, or a tensor of bin edges with dimension [N+1, ...]
        where N is the number of bins. If counts is passed, then bins is interpreted to
        be the bin edges for the counts tensor, by default 10
    counts : Union[None, Tensor], optional
        Existing count tensor to combine results with. Must have dimensions
        [N, ...] where N is the number of bins. Passing a tensor may also require
        recomputing the passed bins to make sure inputs and bins are compatible., by
        default None
    cdf : bool, optional
        Compute a histogram or a cumulative density function; will calculate probability
        density function otherwise, by default False
    tol : float, optional
        Binning tolerance, by default 1e-4
    verbose : bool, optional
        Verbose printing, by default False

    Returns
    -------
    Tuple[Tensor, Tensor]
        The calculated (bin edges [N+1, ...], count [N, ...]) tensors
    """
    # Check shapes of inputs
    s = inputs[0].shape
    for input in inputs[1:]:
        if s[1:] != input.shape[1:]:
            raise ValueError()

    if isinstance(bins, int):
        if verbose:
            print("Bins is passed as an int.")
        # Compute largest bins needed
        low, high = _get_mins_maxs(*inputs)
        number_of_bins = bins
        bin_edges = linspace(low, high, number_of_bins)

        # Bin inputs
        counts = None
        for input in inputs:
            counts = _count_bins(input, bin_edges, counts=counts, cdf=cdf)
        return bin_edges, counts

    elif isinstance(bins, torch.Tensor):
        bin_edges = bins
        if verbose:
            print("Bins is passed as a tensor")
        if counts is None:  # Do not need to update counts
            if verbose:
                print("No counts are passed.")

            number_of_bins = bin_edges.shape[0] - 1
            # Get largest bin edges needed from input/args
            low, high = _get_mins_maxs(*inputs)
            # Compare against existing bin_edges
            low = torch.where(low < bin_edges[0], low, bin_edges[0])
            high = torch.where(high > bin_edges[-1], high, bin_edges[-1])

            # Update, if necessary
            if torch.any(low != bin_edges[0]) | torch.any(high != bin_edges[-1]):
                bin_edges = linspace(low, high, number_of_bins)

            # Bin inputs
            counts = None
            for input in inputs:
                counts = _count_bins(input, bin_edges, counts=counts, cdf=cdf)
            return bin_edges, counts
        else:  # Counts do need to be update
            if verbose:
                print("Counts are being updated.")
            for input in inputs:
                bin_edges, counts = _update_bins_counts(
                    input, bin_edges, counts, cdf=cdf
                )
            return bin_edges, counts
    else:
        raise ValueError("Input bin type is not supported.")


def histogram(
    *inputs: Tensor,
    bins: Union[int, Tensor] = 10,
    counts: Union[None, Tensor] = None,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Computes the histogram of a set of tensors over the leading dimension

    This function will compute bin edges and bin counts of given input tensors. If existing bin edges
    or count tensors are supplied, this function will update these existing statistics
    with the new data.

    Parameters
    ----------
    inputs : (Tensor ...)
        Input data tensor(s) [B, ...]
    bins : Union[int, Tensor], optional
        Either the number of bins, or a tensor of bin edges with dimension [N+1, ...]
        where N is the number of bins. If counts is passed, then bins is interpreted to
        be the bin edges for the counts tensor, by default 10
    counts : Union[None, Tensor], optional
        Existing count tensor to combine results with. Must have dimensions
        [N, ...] where N is the number of bins. Passing a tensor may also require
        recomputing the passed bins to make sure inputs and bins are compatible, by
        default None
    verbose : bool, optional
        Verbose printing, by default False

    Returns
    -------
    Tuple[Tensor, Tensor]
        The calculated (bin edges [N+1, ...], count [N, ...]) tensors
    """
    return _compute_counts_cdf(
        *inputs, bins=bins, counts=counts, cdf=False, verbose=verbose
    )


def cdf(
    *inputs: Tensor,
    bins: Union[int, Tensor] = 10,
    counts: Union[None, Tensor] = None,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Computes the cumulative density function  of a set of tensors over the leading
    dimension

    This function will compute CDF bins of given input tensors. If existing bins
    or count tensors are supplied, this function will update these existing statistics
    with the new data.

    Parameters
    ----------
    inputs : (Tensor ...)
        Input data tensor(s) [B, ...]
    bins : Union[int, Tensor], optional
        Either the number of bins, or a tensor of bin edges with dimension [N+1, ...]
        where N is the number of bins. If counts is passed, then bins is interpreted to
        be the bin edges for the counts tensor, by default 10
    counts : Union[None, Tensor], optional
        Existing count tensor to combine results with. Must have dimensions
        [N, ...] where N is the number of bins. Passing a tensor may also require
        recomputing the passed bins to make sure inputs and bins are compatible, by
        default None
    verbose : bool, optional
        Verbose printing, by default False

    Returns
    -------
    Tuple[Tensor, Tensor]
        The calculated (bin edges [N+1, ...], cdf [N, ...]) tensors
    """
    bin_edges, counts = _compute_counts_cdf(
        *inputs, bins=bins, counts=counts, cdf=True, verbose=verbose
    )
    cdf = counts / counts[-1]  # Normalize
    return bin_edges, cdf


class Histogram(EnsembleMetrics):
    """
    Convenience class for computing histograms, possibly as a part of a distributed or
    iterative environment

    Parameters
    ----------
    input_shape : Tuple[int]
        Input data shape
    bins : Union[int, Tensor], optional
        Initial bin edges or number of bins to use, by default 10
    tol : float, optional
        Bin edge tolerance, by default 1e-3
    """

    def __init__(
        self,
        input_shape: Tuple[int],
        bins: Union[int, Tensor] = 10,
        tol: float = 1e-2,
        **kwargs,
    ):
        super().__init__(input_shape, **kwargs)
        if isinstance(bins, int):
            self.number_of_bins = bins
        else:
            self.number_of_bins = bins.shape[0] - 1

            if self.input_shape[1:] != bins.shape[1:]:
                raise ValueError()

        self.counts_shape = self.input_shape
        self.counts_shape[0] = self.number_of_bins
        self.tol = tol
        # Initialize bins
        start = -1.0 * torch.ones(
            self.input_shape[1:], device=self.device, dtype=self.dtype
        )
        end = torch.ones(self.input_shape[1:], device=self.device, dtype=self.dtype)
        self.bin_edges = linspace(start, end, self.number_of_bins)
        self.counts = torch.zeros(
            self.counts_shape, device=self.device, dtype=torch.int64
        )

    def __call__(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        """Calculate histogram

        Parameters
        ----------
        inputs : Tensor
            Input data tensor [B, ...]

        Returns
        -------
        Tuple[Tensor, Tensor]
            The calculated (bin edges [N+1, ...], counts [N, ...]) tensors
        """
        # Build bin_edges
        if DistributedManager.is_initialized() and dist.is_initialized():
            start, _ = torch.min(input, axis=0)
            end, _ = torch.max(input, axis=0)
            # We assume that the start/end across the distributed environments
            # need to be combined.
            dist.all_reduce(start, op=dist.ReduceOp.MIN)
            dist.all_reduce(end, op=dist.ReduceOp.MAX)
            self.bin_edges = linspace(start, end, self.number_of_bins)

            self.counts = _count_bins(input, self.bin_edges)
            dist.all_reduce(self.counts, op=dist.ReduceOp.SUM)
            return self.bin_edges, self.counts

        else:
            self.bin_edges, self.counts = histogram(input, bins=self.number_of_bins)
            return self.bin_edges, self.counts

    def update(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        """Update current histogram with new data

        Parameters
        ----------
        inputs : Tensor
            Input data tensor [B, ...]

        Returns
        -------
        Tuple[Tensor, Tensor]
            The calculated (bin edges [N+1, ...], counts [N, ...]) tensors
        """
        # TODO(Dallas) Move distributed calls into finalize.
        self.bin_edges, self.counts = _update_bins_counts(
            input, self.bin_edges, self.counts
        )
        self.number_of_bins = self.bin_edges.shape[0]
        return self.bin_edges, self.counts

    def finalize(self, cdf: bool = False) -> Tuple[Tensor, Tensor]:
        """Finalize the histogram, which computes the pdf and cdf

        Parameters
        ----------
        cdf : bool, optional
            Compute a cumulative density function; will calculate
            probability density function otherwise, by default False

        Returns
        -------
        Tuple[Tensor, Tensor]
            The calculated (bin edges [N+1, ...], PDF or CDF [N, ...]) tensors
        """
        # Normalize counts
        hist_norm = self.counts.sum(dim=0)
        self.pdf = self.counts / hist_norm
        if cdf:
            self.cdf = torch.cumsum(self.pdf, dim=0)
            return self.bin_edges, self.cdf
        else:
            return self.bin_edges, self.pdf


def normal_pdf(
    mean: Tensor, std: Tensor, bin_edges: Tensor, grid: str = "midpoint"
) -> Tensor:
    """Computes the probability density function of a normal variable with given mean
    and standard deviation. This PDF is given at the locations given by the midpoint
    of the bin_edges.

    This function uses the standard formula:

    .. math::

        \\frac{1}{\\sqrt{2*\\pi} std } \\exp( -\\frac{1}{2} (\\frac{x-mean}{std})^2 )

    where erf is the error function.

    Parameters
    ----------
    mean : Tensor
        mean tensor
    std : Tensor
        standard deviation tensor
    bins : Tensor
        The tensor of bin edges with dimension [N+1, ...]
        where N is the number of bins.
    grid : str
        A string that indicates where in the bins should the cdf be defined.
        Can be one of {"midpoint", "left", "right"}.
    Returns
    -------
    Tensor
        The calculated cdf tensor with dimension [N, ...]
    """
    if grid == "midpoint":
        bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    elif grid == "right":
        bin_mids = bin_edges[1:]
    elif grid == "left":
        bin_mids = bin_edges[:-1]
    else:
        raise ValueError(
            "This type of grid is not defined. Choose one of {'mids', 'right', 'left'}."
        )
    return (
        torch.exp(-0.5 * ((bin_mids - mean[None, ...]) / std[None, ...]) ** 2)
        / std[None, ...]
        / torch.sqrt(torch.as_tensor(2.0 * torch.pi))
    )


def normal_cdf(
    mean: Tensor,
    std: Tensor,
    bin_edges: Tensor,
    grid: str = "midpoint",
) -> Tensor:
    """Computes the cumulative density function of a normal variable with given mean
    and standard deviation. This CDF is given at the locations given by the midpoint
    of the bin_edges.

    This function uses the standard formula:

    .. math::

        \\frac{1}{2} ( 1 + erf( \\frac{x-mean}{std \\sqrt{2}}) ) )

    where erf is the error function.

    Parameters
    ----------
    mean : Tensor
        mean tensor
    std : Tensor
        standard deviation tensor
    bins : Tensor
        The tensor of bin edges with dimension [N+1, ...]
        where N is the number of bins.
    grid : str
        A string that indicates where in the bins should the cdf be defined.
        Can be one of {"mids", "left", "right"}.
    Returns
    -------
    Tensor
        The calculated cdf tensor with dimension [N, ...]
    """
    if grid == "midpoint":
        bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    elif grid == "right":
        bin_mids = bin_edges[1:]
    elif grid == "left":
        bin_mids = bin_edges[:-1]
    else:
        raise ValueError(
            "This type of grid is not defined. Choose one of {'mids', 'right', 'left'}."
        )
    return 0.5 + 0.5 * torch.erf(
        (bin_mids - mean[None, ...])
        / (torch.sqrt(torch.as_tensor([2.0], device=mean.device)) * std[None, ...])
    )
