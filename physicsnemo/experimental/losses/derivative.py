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

r"""Spatial-derivative regularization loss.

Penalizes the discrepancy between the spatial gradients of a prediction and a
target field. This is a common regularizer for neural-operator surrogates on
structured grids, where matching the field *and* its derivatives yields sharper,
more physical solutions.

.. note::

    PhysicsNeMo already provides several grid derivative operators in
    :mod:`physicsnemo.nn.functional.derivatives` (e.g. ``rectilinear_grid_gradient``,
    ``uniform_grid_gradient``). Those operators are *periodic*, coordinate-based,
    and act on a single scalar field, which makes them unsuitable for this loss:
    the regularizer must operate on *batched* fields with a trailing feature/time
    axis, use *non-periodic* interior central differences (so boundary cells are
    not wrapped), and support an inactive-cell *mask* (sparse grids such as the
    Norne reservoir). The self-contained :func:`central_difference` /
    :func:`cell_centre_distance` helpers below cover exactly that regime.
"""

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from physicsnemo.metrics.general.mse import mse

Tensor = torch.Tensor


def cell_centre_distance(cell_widths: Tensor, min_spacing: float = 1e-6) -> Tensor:
    r"""Distance between the centres of cell :math:`i` and cell :math:`i+2`.

    For a non-uniform 1-D grid with cell widths :math:`w`, the distance used by
    the interior central-difference stencil is

    .. math::

        d_i = \tfrac{1}{2} w_i + w_{i+1} + \tfrac{1}{2} w_{i+2}.

    A minimum floor is applied to avoid division-by-zero when widths are zero
    (inactive cells) or very small (e.g. normalized grids).

    Parameters
    ----------
    cell_widths : Tensor
        1-D tensor of cell widths with shape :math:`(N,)`.
    min_spacing : float, optional
        Lower bound applied to the output distances, by default ``1e-6``.

    Returns
    -------
    Tensor
        Cell-centre distances with shape :math:`(N - 2,)`.
    """
    d = cell_widths[:-2] / 2.0 + cell_widths[1:-1] + cell_widths[2:] / 2.0
    return d.clamp(min=min_spacing)


def central_difference(field: Tensor, axis: int, spacing: Tensor) -> Tensor:
    r"""Interior central-difference derivative of ``field`` along ``axis``.

    Computes :math:`(f_{i+2} - f_i) / d_i` for every interior point, where
    :math:`d` are the cell-centre distances from :func:`cell_centre_distance`.
    The output is *non-periodic*: the differentiated axis is reduced by two
    (no boundary wrap-around).

    Parameters
    ----------
    field : Tensor
        Arbitrary-rank tensor to differentiate.
    axis : int
        Dimension along which to differentiate.
    spacing : Tensor
        1-D cell-centre distances with shape :math:`(N - 2,)`, where
        :math:`N` is ``field.shape[axis]``.

    Returns
    -------
    Tensor
        Derivative tensor with ``field.shape[axis]`` reduced by two.
    """
    n = field.shape[axis]
    f_right = field.narrow(axis, 2, n - 2)
    f_left = field.narrow(axis, 0, n - 2)

    shape = [1] * field.dim()
    shape[axis] = -1
    sp = spacing.reshape(shape)

    return (f_right - f_left) / sp


class SpatialDerivativeLoss(nn.Module):
    r"""Regularization loss on the spatial derivatives of a field.

    For each requested spatial axis, the interior central-difference derivative
    (see :func:`central_difference`) of both ``pred`` and ``target`` is computed
    on a non-uniform grid defined by per-axis cell widths, and the two are
    compared with a data-fitting ``metric``. The per-axis losses are averaged.

    The loss is dimension-agnostic: it works for any number of spatial axes
    (e.g. 2-D :math:`(B, H, W, T)` or 3-D :math:`(B, X, Y, Z, T)` fields with a
    trailing feature/time axis), and it supports an optional boolean mask that
    excludes inactive cells (useful for sparse grids). Inactive cells are zeroed
    before differentiation and any derivative sample whose central-difference
    stencil touches an inactive cell is excluded from the comparison.

    Parameters
    ----------
    metric : Callable[[Tensor, Tensor], Tensor], optional
        Data-fitting metric used to compare the predicted and target
        derivatives. Any of the :mod:`physicsnemo.metrics.general` functions
        (e.g. :func:`~physicsnemo.metrics.general.mse.mse`,
        :func:`~physicsnemo.metrics.general.relative.relative_l2`) or a custom
        callable can be injected. Defaults to
        :func:`~physicsnemo.metrics.general.mse.mse`.
    min_spacing : float, optional
        Lower bound on the cell-centre distances used by the derivative stencil,
        by default ``1e-6``.

    Forward
    -------
    pred : Tensor
        Predicted field of shape :math:`(B, *\mathrm{spatial}, T)`.
    target : Tensor
        Target field with the same shape as ``pred``.
    cell_widths : Dict[int, Tensor]
        Mapping from a ``pred`` tensor axis (a spatial axis, i.e. one of
        :math:`1 \ldots N_\mathrm{spatial}`) to the 1-D cell widths along that
        axis. The keys select which axes are differentiated.
    mask : Tensor, optional
        Boolean tensor marking active cells, either static ``(*spatial)`` (shared
        across the batch) or per-sample ``(B, *spatial)`` (each sample uses its
        own mask). When ``None`` all cells are active.

    Outputs
    -------
    Tensor
        Scalar derivative-regularization loss, averaged over the axes that
        contribute at least one valid derivative sample. An axis whose
        central-difference stencil is fully masked out contributes nothing and
        is excluded from the average; if no axis contributes, the loss is zero.

    Notes
    -----
    When a ``mask`` is supplied, the loss selects active derivative samples via
    boolean indexing, which produces data-dependent shapes. The masked path is
    therefore **not traceable by** :func:`torch.compile` (it triggers a graph
    break / eager fallback); the unmasked path compiles normally.

    Example
    -------
    >>> import torch
    >>> from physicsnemo.experimental.losses import SpatialDerivativeLoss
    >>> loss = SpatialDerivativeLoss()
    >>> pred = torch.randn(2, 8, 16, 4)
    >>> target = torch.randn(2, 8, 16, 4)
    >>> widths = {1: torch.ones(8), 2: torch.ones(16)}  # differentiate H and W
    >>> value = loss(pred, target, widths)
    >>> value.ndim
    0
    """

    def __init__(
        self,
        metric: Callable[[Tensor, Tensor], Tensor] = mse,
        min_spacing: float = 1e-6,
    ):
        super().__init__()
        self.metric = metric
        self.min_spacing = min_spacing

    @staticmethod
    def _is_per_sample_mask(mask: Tensor, pred: Tensor) -> bool:
        """True when ``mask`` has a leading batch dimension matching ``pred``.

        A per-sample mask has shape ``(B, *spatial)`` (rank ``pred.dim() - 1``);
        a static mask has shape ``(*spatial)`` (rank ``pred.dim() - 2``) and is
        shared across the batch.
        """
        return mask.dim() == pred.dim() - 1 and mask.shape[0] == pred.shape[0]

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        cell_widths: Dict[int, Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        if not torch.compiler.is_compiling():
            if pred.shape != target.shape:
                raise ValueError(
                    f"pred and target must have the same shape, got "
                    f"{tuple(pred.shape)} and {tuple(target.shape)}"
                )
            if not cell_widths:
                raise ValueError("cell_widths must specify at least one axis")
            # Keys must be spatial axes: the batch axis (0) and the trailing
            # feature/time axis are excluded, otherwise the stencil would
            # silently differentiate the wrong dimension.
            valid_axes = range(1, pred.dim() - 1)
            for axis, widths in cell_widths.items():
                if axis not in valid_axes:
                    raise ValueError(
                        f"cell_widths keys must be spatial axes in "
                        f"[1, {pred.dim() - 2}] (the batch axis 0 and the "
                        f"trailing feature/time axis are excluded), got {axis}"
                    )
                # The 1-D cell widths must span the full axis, otherwise the
                # derivative stencil/spacing would mismatch the field extent.
                if widths.shape != (pred.shape[axis],):
                    raise ValueError(
                        f"cell_widths[{axis}] must be a 1-D tensor of length "
                        f"{pred.shape[axis]} (pred.shape[{axis}]), got shape "
                        f"{tuple(widths.shape)}"
                    )

        # Masks are applied per-sample when a ``(B, *spatial)`` mask is given, so a
        # cell inactive in one sample never contributes to that sample's loss.
        per_sample_mask = mask is not None and self._is_per_sample_mask(mask, pred)
        if mask is not None:
            # Zero inactive cells so the stencil does not read garbage across the
            # active/inactive boundary. Per-sample: ``(B, *spatial, 1)``;
            # static: ``(1, *spatial, 1)``.
            if per_sample_mask:
                zero_w = mask.unsqueeze(-1)
            else:
                zero_w = mask.unsqueeze(0).unsqueeze(-1)
            pred = pred * zero_w
            target = target * zero_w

        total = torch.zeros((), device=pred.device, dtype=pred.dtype)
        n_contributing = 0
        for axis, widths in cell_widths.items():
            spacing = cell_centre_distance(widths, min_spacing=self.min_spacing)
            d_pred = central_difference(pred, axis, spacing)
            d_target = central_difference(target, axis, spacing)

            if mask is not None:
                # Stencil-safe mask: a derivative sample is valid only when all
                # three stencil cells (i, i+1, i+2) along ``axis`` are active. The
                # mask axis is offset by the batch dim for per-sample masks.
                mask_axis = axis if per_sample_mask else axis - 1
                n = mask.shape[mask_axis]
                m_left = mask.narrow(mask_axis, 0, n - 2)
                m_centre = mask.narrow(mask_axis, 1, n - 2)
                m_right = mask.narrow(mask_axis, 2, n - 2)
                deriv_mask = m_left & m_centre & m_right
                if per_sample_mask:
                    sel = deriv_mask.unsqueeze(-1).expand_as(d_pred)
                else:
                    sel = deriv_mask.unsqueeze(0).unsqueeze(-1).expand_as(d_pred)
                p = d_pred[sel]
                t = d_target[sel]
            else:
                p = d_pred
                t = d_target

            # Skip axes with no active derivative samples (e.g. a fully-masked
            # region) so that reducing an empty tensor cannot produce NaN.
            if p.numel() == 0:
                continue

            total = total + self.metric(p, t)
            n_contributing += 1

        # Average over the axes that actually contributed, so that a fully-masked
        # axis does not silently shrink the loss magnitude. Returns 0 when no
        # axis has any valid derivative sample.
        if n_contributing == 0:
            return total
        return total / n_contributing
