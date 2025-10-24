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

# TODO(Dallas) Introduce Distributed Class for computation.

import torch

from physicsnemo.metrics.general.entropy import entropy_from_counts
from physicsnemo.metrics.general.histogram import normal_cdf

Tensor = torch.Tensor


def efi_gaussian(
    pred_cdf: Tensor,
    bin_edges: Tensor,
    climatology_mean: Tensor,
    climatology_std: Tensor,
) -> Tensor:
    """Calculates the Extreme Forecast Index (EFI) for an ensemble forecast against
    a climatological distribution.

    Parameters
    ----------
    pred_cdf : Tensor
        Cumulative distribution function of predictions of shape [N, ...]
        where N is the number of bins. This cdf must be defined over the
        passed bin_edges.
    bin_edges : Tensor
        Tensor of bin edges with shape [N+1, ...]
        where N is the number of bins.
    climatology_mean : Tensor
        Tensor of climatological mean with shape [...]
    climatology_std : Tensor
        Tensor of climatological std with shape [...]

    Returns
    -------
    Tensor
        EFI values of each of the batched dimensions.

    Note
    ----
    Reference: https://www.atmos.albany.edu/daes/atmclasses/atm401/spring_2016/ppts_pdfs/ECMWF_EFI.pdf
    """

    clim_cdf = normal_cdf(climatology_mean, climatology_std, bin_edges, grid="right")

    return (
        2.0
        / torch.pi
        * torch.trapz(
            (clim_cdf - pred_cdf) / torch.sqrt(1e-8 + clim_cdf * (1.0 - clim_cdf)),
            clim_cdf,
            dim=0,
        )
    )


def efi(bin_edges: Tensor, counts: Tensor, quantiles: Tensor) -> Tensor:
    """Compute the Extreme Forecast Index for the given histogram.

    The histogram is assumed to correspond with the given quantiles.
    That is, the bin midpoints must align with the quantiles.

    Parameters
    ----------
    bin_edges : Tensor
        The bin edges of the histogram over which the data distribution
        is defined. Assumed to be monotonically increasing but not evenly
        spaced.
    counts : Tensor
        The counts of the histogram over which the data distributed is defined.
        Not assumed to be normalized.
    quantiles : Tensor
        The quantiles of the climatological or reference distribution. The quantiles
        must match the midpoints of the histogram bins.
    See physicsnemo/metrics/climate/efi for more details.
    """
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    pred_cdf = torch.cumsum(counts * bin_widths, dim=0) / torch.sum(
        counts * bin_widths, dim=0
    )
    return (
        2.0
        / torch.pi
        * torch.trapz(
            (quantiles - pred_cdf) / torch.sqrt(1e-8 + quantiles * (1.0 - quantiles)),
            quantiles,
            dim=0,
        )
    )


def normalized_entropy(
    pred_pdf: Tensor,
    bin_edges: Tensor,
    climatology_pdf: Tensor,
) -> Tensor:
    """Calculates the relative entropy, or surprise, of using the prediction
    distribution with respect to the climatology distribution.

    Parameters
    ----------
    pred_cdf : Tensor
        Cumulative distribution function of predictions of shape [N, ...]
        where N is the number of bins. This cdf must be defined over the
        passed bin_edges.
    bin_edges : Tensor
        Tensor of bin edges with shape [N+1, ...]
        where N is the number of bins.
    climatology_pdf : Tensor
        Tensor of climatological probability function shape [N, ...]

    Returns
    -------
    Tensor
        Relative Entropy values of each of the batched dimensions.

    """

    if pred_pdf.shape != climatology_pdf.shape:
        raise ValueError(
            "Prediction PDF and Climatological PDF must have the same shapes"
            + f"but recieved {pred_pdf.shape} and {climatology_pdf.shape}."
        )

    return 1.0 - entropy_from_counts(pred_pdf, bin_edges) / entropy_from_counts(
        climatology_pdf, bin_edges
    )
