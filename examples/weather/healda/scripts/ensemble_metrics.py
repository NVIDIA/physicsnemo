# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

# Copied from modulus
import torch

Tensor = torch.Tensor


@torch.jit.script
def _kernel_crps_implementation(pred: Tensor, obs: Tensor, biased: bool) -> Tensor:
    """An O(m log m) implementation of the kernel CRPS formulas"""
    skill = torch.abs(pred - obs[..., None]).mean(-1)
    pred, _ = torch.sort(pred)

    # derivation of fast implementation of spread-portion of CRPS formula when x is sorted
    # sum_(i,j=1)^m |x_i - x_j| = sum_(i<j) |x_i -x_j| + sum_(i > j) |x_i - x_j|
    #                           = 2 sum_(i <= j) |x_i -x_j|
    #                           = 2 sum_(i <= j) (x_j - x_i)
    #                           = 2 sum_(i <= j) x_j - 2 sum_(i <= j) x_i
    #                           = 2 sum_(j=1)^m j x_j - 2 sum (m - i + 1) x_i
    #                           = 2 sum_(i=1)^m (2i - m - 1) x_i
    m = pred.size(-1)
    i = torch.arange(1, m + 1, device=pred.device, dtype=pred.dtype)
    denom = m * m if biased else m * (m - 1)
    factor = (2 * i - m - 1) / denom
    spread = torch.sum(factor * pred, dim=-1)
    return skill - spread


def kcrps(
    pred: Tensor, obs: Tensor, dim: int = 0, biased: bool = False, chunk_size: int = 16
):
    """Estimate the CRPS from a finite ensemble in batched/streaming fashion

    Computes the local Continuous Ranked Probability Score (CRPS) by using
    the kernel version of CRPS. The cost is O(m log m).

    Creates a map of CRPS and does not accumulate over lat/lon regions.
    Approximates:
    .. math::
        CRPS(X, y) = E[X - y] - 0.5 E[X-X']

    with
    .. math::
        sum_i=1^m |X_i - y| / m - 1/(2m^2) sum_i,j=1^m |x_i - x_j|

    Parameters
    ----------
    pred : Tensor
        Tensor containing the ensemble predictions. The ensemble dimension
        is assumed to be the leading dimension unless 'dim' is specified.
    obs : Union[Tensor, np.ndarray]
        Tensor or array containing an observation over which the CRPS is computed
        with respect to.
    dim : int, optional
        The dimension over which to compute the CRPS, assumed to be 0.
    biased :
        When False, uses the unbiased estimators described in (Zamo and Naveau, 2018)::

            E|X-y|/m - 1/(2m(m-1)) sum_(i,j=1)|x_i - x_j|

        Unlike ``crps`` this is fair for finite ensembles. Non-fair ``crps`` favors less
        dispersive ensembles since it is biased high by E|X- X'|/ m where m is the
        ensemble size.

    Returns
    -------
    Tensor
        Map of CRPS
    """
    pred = torch.movedim(pred, dim, -1)
    return _kernel_crps_implementation(pred, obs, biased=biased)


def unbiased_ensemble_metrics(prediction, truth):
    """Unbiased ensemble metrics

    When averaged over many forecasts these formulas are unbiased
    even for size 2 ensembles.

    Args:
        prediction: shaped (*, e) - e is the ensemble dimension
        truth: shaped (*)
    """
    scores = {}
    scores["mse"] = mse = (prediction.mean(-1) - truth) ** 2
    # ensemble scores
    if prediction.size(-1) > 1:
        scores["variance"] = variance = prediction.var(-1)
        scores["crps"] = kcrps(prediction, truth, dim=-1, biased=False)

        # unbias the ensemble mean mse formula
        # per reviewer of Brenowitz, et. al. 2024 , A practical benchmark for probabilistic scoring
        #  RMSE for the ensemble mean can also be debiased to match the
        # limit of infinite ensemble size, using the same math from Fortin, DebiasedMSE =
        # MSE - (1/n) Var, where Var is the debiased estimate of the ensemble variance.
        R = prediction.size(-1)
        scores["mse"] = mse - variance / R
        scores["mse_biased"] = mse

        # Fortin. eq 15
        scores["spread_error"] = torch.sqrt(variance / mse * (R + 1) / R)
    return scores
