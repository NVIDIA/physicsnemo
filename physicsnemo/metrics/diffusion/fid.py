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

import torch

from physicsnemo.metrics.general.wasserstein import wasserstein_from_normal


def calculate_fid_from_inception_stats(
    mu: torch.Tensor, sigma: torch.Tensor, mu_ref: torch.Tensor, sigma_ref: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the Fréchet Inception Distance (FID) between two sets
    of Inception statistics.

    The Fréchet Inception Distance is a measure of the similarity between two datasets
    based on their Inception features (mu and sigma). It is commonly used to evaluate
    the quality of generated images in generative models.

    Parameters
    ----------
    mu:  torch.Tensor:
        Mean of Inception statistics for the generated dataset.
    sigma: torch.Tensor:
        Covariance matrix of Inception statistics for the generated dataset.
    mu_ref: torch.Tensor
        Mean of Inception statistics for the reference dataset.
    sigma_ref: torch.Tensor
        Covariance matrix of Inception statistics for the reference dataset.

    Returns
    -------
    float
        The Fréchet Inception Distance (FID) between the two datasets.
    """
    return wasserstein_from_normal(mu, sigma, mu_ref, sigma_ref)
