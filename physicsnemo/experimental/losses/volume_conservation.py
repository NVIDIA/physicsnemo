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

r"""Volume-conservation enforcement loss.

Penalizes the discrepancy between the (optionally volume-weighted) spatial
integral of a prediction and a target field at each step of a trailing
feature/time axis. Matching the spatially-integrated quantity is a weak
conservation constraint used to regularize neural-operator surrogates.

.. important::

    This enforces conservation of a *volume-weighted spatial integral*
    :math:`\int u \, \mathrm{d}V`, not true mass conservation: no material
    density or porosity is applied, so the constraint is physically meaningful
    only for quantities whose spatial integral is (approximately) conserved.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn

from physicsnemo.metrics.general.mse import mse

Tensor = torch.Tensor


class VolumeConservationLoss(nn.Module):
    r"""Enforces conservation of a field's volume-weighted spatial integral.

    For a prediction and target of shape :math:`(B, *\mathrm{spatial}, T)`, the
    loss integrates each field over the spatial dimensions using per-cell
    volume weights, producing a conserved-quantity time series of shape
    :math:`(B, T)`, and compares the two series with a data-fitting ``metric``:

    .. math::

        Q_b^{\,t} = \sum_{\mathrm{spatial}} w \, u_b^{\,t}, \qquad
        \mathcal{L} = \mathrm{mean}_b\; \mathrm{metric}\big(Q^{\,\mathrm{pred}}_b,
        Q^{\,\mathrm{true}}_b\big),

    where :math:`w` are the cell volumes (uniform when not provided). The loss
    is dimension-agnostic (2-D or 3-D spatial fields) and supports an optional
    inactive-cell mask for sparse grids.

    Parameters
    ----------
    metric : Callable[..., Tensor], optional
        Per-sample data-fitting metric applied to the conserved-quantity time
        series. It is called as ``metric(pred, target, dim=-1)`` and must return
        one value per sample; any :mod:`physicsnemo.metrics.general` function
        (e.g. :func:`~physicsnemo.metrics.general.mse.mse`) or a compatible
        callable (such as a relative-error metric) can be injected. Defaults to
        :func:`~physicsnemo.metrics.general.mse.mse`.

    Forward
    -------
    pred : Tensor
        Predicted field of shape :math:`(B, *\mathrm{spatial}, T)`.
    target : Tensor
        Target field with the same shape as ``pred``.
    cell_volumes : Tensor, optional
        Per-cell volume weights of shape ``(*spatial)``. When ``None`` uniform
        weights of one are used.
    mask : Tensor, optional
        Boolean tensor marking active cells, either static ``(*spatial)`` (shared
        across the batch) or per-sample ``(B, *spatial)`` (each sample uses its
        own mask). Inactive cells receive zero weight in that sample's integral.

    Outputs
    -------
    Tensor
        Scalar volume-conservation loss.

    Example
    -------
    >>> import torch
    >>> from physicsnemo.experimental.losses import VolumeConservationLoss
    >>> loss = VolumeConservationLoss()
    >>> pred = torch.randn(2, 8, 16, 4)
    >>> target = torch.randn(2, 8, 16, 4)
    >>> volumes = torch.ones(8, 16)
    >>> value = loss(pred, target, volumes)
    >>> value.ndim
    0
    """

    def __init__(
        self,
        metric: Callable[..., Tensor] = mse,
    ):
        super().__init__()
        self.metric = metric

    @staticmethod
    def _is_per_sample_mask(mask: Tensor, pred: Tensor) -> bool:
        """True when ``mask`` has a leading batch dim matching ``pred``.

        A per-sample mask has shape ``(B, *spatial)``; a static mask has shape
        ``(*spatial)`` and is shared across the batch.
        """
        return mask.dim() == pred.dim() - 1 and mask.shape[0] == pred.shape[0]

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        cell_volumes: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Integrate over all spatial axes between the batch axis and the
        # trailing feature/time axis.
        spatial_dims = tuple(range(1, pred.dim() - 1))
        spatial_shape = tuple(pred.shape[1 : pred.dim() - 1])

        if not torch.compiler.is_compiling():
            if pred.shape != target.shape:
                raise ValueError(
                    f"pred and target must have the same shape, got "
                    f"{tuple(pred.shape)} and {tuple(target.shape)}"
                )
            if pred.dim() < 3:
                raise ValueError(
                    "pred must have shape (B, *spatial, T) with at least one "
                    f"spatial dimension, got {tuple(pred.shape)}"
                )
            # cell_volumes must span exactly the spatial dims, otherwise the
            # weight broadcast would mismatch the field extent.
            if cell_volumes is not None and tuple(cell_volumes.shape) != spatial_shape:
                raise ValueError(
                    f"cell_volumes must have shape {spatial_shape} (pred's spatial "
                    f"dims), got {tuple(cell_volumes.shape)}"
                )
            # mask must be static (*spatial) or per-sample (B, *spatial).
            if mask is not None and tuple(mask.shape) not in (
                spatial_shape,
                (pred.shape[0], *spatial_shape),
            ):
                raise ValueError(
                    f"mask must have shape {spatial_shape} (static) or "
                    f"{(pred.shape[0], *spatial_shape)} (per-sample), got "
                    f"{tuple(mask.shape)}"
                )

        if cell_volumes is None:
            vol = torch.ones(spatial_shape, device=pred.device, dtype=pred.dtype)
        else:
            vol = cell_volumes

        # Build per-cell weights, applying the mask per-sample when a
        # ``(B, *spatial)`` mask is given so a cell inactive in one sample does
        # not contribute to that sample's integral.
        if mask is not None and self._is_per_sample_mask(mask, pred):
            # (*spatial) * (B, *spatial) -> (B, *spatial, 1)
            w = (vol.unsqueeze(0) * mask.to(vol.dtype)).unsqueeze(-1)
        elif mask is not None:
            # static (*spatial) mask -> (1, *spatial, 1)
            w = (vol * mask.to(vol.dtype)).unsqueeze(0).unsqueeze(-1)
        else:
            w = vol.unsqueeze(0).unsqueeze(-1)

        m_pred = (pred * w).sum(dim=spatial_dims)  # (B, T)
        m_true = (target * w).sum(dim=spatial_dims)  # (B, T)

        per_sample = self.metric(m_pred, m_true, dim=-1)
        return per_sample.mean()
