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

r"""Training helpers for the variational GP heads.

The heads in this package define the model; turning them into a useful,
distance-aware uncertainty signal also takes a training recipe (see
*Predictive Uncertainty for Neural CAE Surrogates*, App. A). These are the
pieces of that recipe that do not depend on a particular training loop:

* :func:`linear_warmup` — the 0 → 1 ramp used for KL annealing and for the
  weight on the whole negative ELBO.
* :func:`subsample_points` — a random subset of points per GP step.
* :func:`latent_distance_penalty` — a one-sided hinge that keeps points with
  different targets apart in kernel space, so a learned feature map cannot
  collapse unfamiliar inputs onto familiar ones.
* :func:`kmeans_medoids` — k-means++ / Lloyd clustering that returns data
  points (medoids), used by
  :meth:`~physicsnemo.experimental.uq.gp.FieldVariationalGPHead.seed_inducing_points`.

None of them require ``gpytorch``.
"""

from __future__ import annotations

import torch
from jaxtyping import Float, Int


def linear_warmup(epoch: int, start: int, end: int) -> float:
    r"""Linear ramp: 0 before ``start``, 0 → 1 over ``[start, end)``, 1 after.

    Used for KL annealing and for the weight on the whole negative ELBO, so
    the data-fit terms shape the mean before the KL pulls the variational
    posterior toward the prior. An empty or inverted window
    (``end <= start``) disables the ramp and returns 1.0 throughout.

    Parameters
    ----------
    epoch : int
        Current epoch.
    start : int
        First epoch of the ramp.
    end : int
        Epoch at which the weight reaches 1.0.

    Returns
    -------
    float
        Weight in ``[0, 1]``.

    Examples
    --------
    >>> linear_warmup(15, 10, 20)
    0.5
    >>> linear_warmup(5, 10, 10)
    1.0
    """
    if end <= start:
        return 1.0
    if epoch < start:
        return 0.0
    if epoch >= end:
        return 1.0
    return (epoch - start) / (end - start)


def subsample_points(
    features: Float[torch.Tensor, "batch points dim"],
    target: Float[torch.Tensor, "batch points tasks"],
    n_points: int | None,
) -> tuple[
    Float[torch.Tensor, "batch sub_points dim"],
    Float[torch.Tensor, "batch sub_points tasks"],
]:
    r"""Keep the same random ``n_points`` points of every sample.

    A minibatch of points, not a smaller dataset: keep the ELBO's ``num_data``
    at the full point count. Points are drawn from torch's global RNG, so a
    seeded run draws the same subsets.

    Parameters
    ----------
    features : torch.Tensor
        Per-point features of shape :math:`(B, N, D)`.
    target : torch.Tensor
        Per-point targets of shape :math:`(B, N, T)`.
    n_points : int | None
        Points to keep. ``None``, ``0`` or ``n_points >= N`` returns the inputs
        unchanged.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Features :math:`(B, n, D)` and targets :math:`(B, n, T)`.
    """
    n_total = features.shape[1]
    if not n_points or n_points >= n_total:
        return features, target
    idx = torch.randperm(n_total, device=features.device)[:n_points]
    return features[:, idx], target[:, idx]


def latent_distance_penalty(
    gp_feats: Float[torch.Tensor, "batch points gp_dim"],
    target: Float[torch.Tensor, "batch points tasks"],
    n_pairs: int,
    margin: float,
) -> Float[torch.Tensor, ""]:
    r"""Within-sample latent distance penalty (bi-Lipschitz-style, target-aware).

    For random point pairs inside each sample, penalizes pairs whose
    kernel-space distance is smaller than ``margin`` times their target-space
    distance:

    .. math::
        \frac{1}{B}\sum_b \frac{1}{P}\sum_{(i,j)}
        \operatorname{relu}\big(m\,\tilde d_y(i,j) - \tilde d_h(i,j)\big),

    where both distances are divided by their (detached) per-sample mean, so
    the penalty shapes relative geometry rather than absolute scale. It is
    one-sided: pairs already far apart are never pulled together. Without it a
    jointly trained feature map is free to map different-looking regions onto
    the same few inducing points, and the posterior variance stops responding
    to them.

    Parameters
    ----------
    gp_feats : torch.Tensor
        Features in kernel space, shape :math:`(B, P, D)` (the output of
        ``transform_features``, with gradient). Other shapes are treated as a
        single sample of :math:`(P, D)` points.
    target : torch.Tensor
        Per-point targets, shape :math:`(B, P, T)`.
    n_pairs : int
        Random pairs per sample, drawn from torch's global RNG.
    margin : float
        Multiplier on the normalized target distance.

    Returns
    -------
    torch.Tensor
        Scalar penalty, averaged over pairs and then over samples. Zero when
        there are fewer than two points or ``n_pairs < 1``.
    """
    if gp_feats.dim() != 3:
        gp_feats = gp_feats.reshape(1, -1, gp_feats.shape[-1])
        target = target.reshape(1, -1, target.shape[-1])
    b, p, _ = gp_feats.shape
    if p < 2 or n_pairs < 1:
        return gp_feats.new_zeros(())
    target = target.to(gp_feats.dtype)

    i = torch.randint(0, p, (b, n_pairs), device=gp_feats.device)
    j = torch.randint(0, p, (b, n_pairs), device=gp_feats.device)
    rows = torch.arange(b, device=gp_feats.device).unsqueeze(1)
    d_lat = (
        (gp_feats[rows, i] - gp_feats[rows, j]).pow(2).sum(-1).clamp_min(1e-12).sqrt()
    )
    d_tgt = (target[rows, i] - target[rows, j]).pow(2).sum(-1).clamp_min(1e-12).sqrt()
    # Scale-free per sample, so a sample with a wide target range does not
    # dominate the batch.
    d_lat = d_lat / (d_lat.mean(dim=1, keepdim=True).detach() + 1e-12)
    d_tgt = d_tgt / (d_tgt.mean(dim=1, keepdim=True).detach() + 1e-12)
    return torch.relu(margin * d_tgt - d_lat).mean(dim=1).sum() / max(b, 1)


@torch.no_grad()
def kmeans_medoids(
    x: Float[torch.Tensor, "points dim"],
    k: int,
    n_iter: int = 25,
    generator: torch.Generator | None = None,
) -> Int[torch.Tensor, " k"]:
    r"""Indices of ``k`` distinct rows of ``x`` nearest to the k-means centroids.

    k-means++ initialization (distance-weighted, so sparse but distinct
    regions get centres even when a dense region outnumbers them), then
    ``n_iter`` Lloyd iterations. Medoids rather than centroids because a
    centroid in a transformed space may have no preimage in the raw space the
    caller needs; a medoid is always one of the inputs.

    Parameters
    ----------
    x : torch.Tensor
        Points of shape :math:`(N, D)`.
    k : int
        Number of medoids.
    n_iter : int, optional
        Lloyd iterations. Default is 25.
    generator : torch.Generator | None, optional
        CPU generator for the random draws, for reproducibility. Default is
        ``None`` (torch's global RNG).

    Returns
    -------
    torch.Tensor
        ``k`` distinct indices into ``x`` (all ``N`` indices if ``k >= N``),
        on ``x``'s device.

    Examples
    --------
    >>> x = torch.cat([torch.zeros(90, 2), torch.ones(10, 2) * 5])
    >>> idx = kmeans_medoids(x, 2, generator=torch.Generator().manual_seed(0))
    >>> sorted(x[idx, 0].tolist())
    [0.0, 5.0]
    """
    n = x.shape[0]
    if k >= n:
        return torch.arange(n, device=x.device)
    x = x.float()
    first = int(torch.randint(0, n, (1,), generator=generator))
    centres = [x[first]]
    d2 = (x - x[first]).pow(2).sum(-1)
    for _ in range(1, k):
        probs = (d2 / d2.sum().clamp_min(1e-30)).cpu()
        nxt = int(torch.multinomial(probs, 1, generator=generator))
        centres.append(x[nxt])
        d2 = torch.minimum(d2, (x - x[nxt]).pow(2).sum(-1))
    c = torch.stack(centres)
    for _ in range(n_iter):
        assign = torch.cdist(x, c).argmin(dim=1)
        sums = torch.zeros_like(c).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k)
        # An empty cluster keeps its previous centre.
        c = torch.where(
            (counts == 0).unsqueeze(1),
            c,
            sums / counts.clamp_min(1).unsqueeze(1).to(x.dtype),
        )
    uniq = torch.unique(torch.cdist(c, x).argmin(dim=1))
    # Two centroids can share a nearest point; top up with unused points so
    # exactly k distinct indices come back.
    if uniq.numel() < k:
        mask = torch.ones(n, dtype=torch.bool, device=x.device)
        mask[uniq] = False
        extra = mask.nonzero().squeeze(1)
        pick = torch.randperm(extra.numel(), generator=generator)[: k - uniq.numel()]
        uniq = torch.cat([uniq, extra[pick.to(extra.device)]])
    return uniq
