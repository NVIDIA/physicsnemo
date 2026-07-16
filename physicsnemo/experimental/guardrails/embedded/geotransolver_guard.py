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

"""Wrap a GeoTransolver to enable out-of-distribution guarding.

The upstreamed :class:`~physicsnemo.models.geotransolver.GeoTransolver` carries
no guardrail logic of its own.  This module provides a thin wrapper that attaches
an :class:`~physicsnemo.experimental.guardrails.embedded.OODGuard` around an
existing model instance: it observes the two surfaces the guard watches — the
raw ``global_embedding`` forward input and the pooled geometry latent — and
calibrates during training / checks during inference, exactly as the previously
embedded guard did.

The geometry latent is captured non-invasively with a forward hook on the
model's ``context_builder.geometry_tokenizer`` submodule, whose output is the
:math:`(B, H, S, D)` slice-token tensor.  It is pooled to :math:`(B, D)` before
being handed to the guard.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ood_guard import OODGuard, OODGuardConfig

__all__ = ["GuardedGeoTransolver", "attach_ood_guard"]


def _infer_geometry_embed_dim(model: nn.Module) -> int | None:
    """Head dimension of the geometry latent, or ``None`` if geometry is off."""
    tokenizer = getattr(model.context_builder, "geometry_tokenizer", None)
    if tokenizer is None:
        return None
    return int(tokenizer.dim_head)


def _infer_global_dim(model: nn.Module) -> int | None:
    """Channel dimension of the global embedding, or ``None`` if global is off."""
    tokenizer = getattr(model.context_builder, "global_tokenizer", None)
    if tokenizer is None:
        return None
    return int(tokenizer.in_project_x.in_features)


def _extract_global_embedding(args: tuple, kwargs: dict) -> torch.Tensor | None:
    """Pull ``global_embedding`` from a GeoTransolver forward call.

    ``GeoTransolver.forward`` takes ``global_embedding`` as its third positional
    argument (``local_embedding``, ``local_positions``, ``global_embedding``).
    """
    if "global_embedding" in kwargs:
        return kwargs["global_embedding"]
    if len(args) >= 3:
        return args[2]
    return None


class GuardedGeoTransolver(nn.Module):
    """GeoTransolver wrapped with an out-of-distribution guard.

    The wrapper delegates the forward pass to the wrapped model unchanged and,
    as a side effect, feeds the guard.  During training the guard accumulates
    calibration statistics; during inference it checks incoming data against
    them and emits warnings on out-of-distribution inputs.  Switch behaviour via
    the standard :meth:`~torch.nn.Module.train` / :meth:`~torch.nn.Module.eval`
    toggles — the wrapped model's ``training`` flag selects collect vs. check.

    Parameters
    ----------
    model : GeoTransolver
        A constructed GeoTransolver instance.  At least one of the geometry or
        global surfaces must be enabled for the guard to have anything to watch.
    config : OODGuardConfig
        Guard configuration (``buffer_size`` required; ``knn_k`` and
        ``sensitivity`` optional).
    global_dim : int | None, optional
        Channel dimension of the global embedding.  Inferred from the model's
        ``context_builder`` when ``None``.
    geometry_embed_dim : int | None, optional
        Dimensionality of the pooled geometry latent.  Inferred from the model's
        ``context_builder`` when ``None``.

    Raises
    ------
    ValueError
        If neither the geometry nor the global surface is enabled on the model.

    Examples
    --------
    >>> from physicsnemo.models.geotransolver import GeoTransolver
    >>> from physicsnemo.experimental.guardrails.embedded import (
    ...     GuardedGeoTransolver, OODGuardConfig,
    ... )
    >>> model = GeoTransolver(functional_dim=8, out_dim=3, n_hidden=32,
    ...                       n_layers=2, global_dim=4, use_te=False)
    >>> guarded = GuardedGeoTransolver(model, OODGuardConfig(buffer_size=128))
    >>> guarded.train()  # doctest: +SKIP
    >>> _ = guarded(x, global_embedding=g)  # collects  # doctest: +SKIP
    >>> guarded.eval()  # doctest: +SKIP
    >>> _ = guarded(x, global_embedding=g)  # checks  # doctest: +SKIP
    """

    def __init__(
        self,
        model: nn.Module,
        config: OODGuardConfig,
        *,
        global_dim: int | None = None,
        geometry_embed_dim: int | None = None,
    ) -> None:
        super().__init__()
        if global_dim is None:
            global_dim = _infer_global_dim(model)
        if geometry_embed_dim is None:
            geometry_embed_dim = _infer_geometry_embed_dim(model)
        if global_dim is None and geometry_embed_dim is None:
            raise ValueError(
                "GuardedGeoTransolver requires the wrapped model to enable at "
                "least one of the global or geometry surfaces; both are "
                "disabled, so the OOD guard would have nothing to watch."
            )

        self.model = model
        self.ood_guard = OODGuard(
            buffer_size=config.buffer_size,
            global_dim=global_dim,
            geometry_embed_dim=geometry_embed_dim,
            knn_k=config.knn_k,
            sensitivity=config.sensitivity,
        )

        # Captured, pooled geometry latent for the most recent forward pass.
        self._geo_latent: torch.Tensor | None = None
        # Retain the hook handle so it can be removed in close(); dropping it
        # would leak hooks (and keep old wrappers alive) on repeated wrapping.
        self._geo_hook_handle: torch.utils.hooks.RemovableHandle | None = None
        tokenizer = getattr(model.context_builder, "geometry_tokenizer", None)
        if tokenizer is not None:
            self._geo_hook_handle = tokenizer.register_forward_hook(
                self._capture_geometry_latent
            )

    def _capture_geometry_latent(
        self, module: nn.Module, inputs: tuple, output: torch.Tensor
    ) -> None:
        """Forward hook: pool the (B, H, S, D) slice tokens to (B, D)."""
        # Detach so the guard's buffers never keep the backward graph alive.
        self._geo_latent = output.detach().mean(dim=(1, 2))

    def forward(self, *args, **kwargs):
        global_embedding = _extract_global_embedding(args, kwargs)
        self._geo_latent = None
        output = self.model(*args, **kwargs)

        if self.model.training:
            self.ood_guard.collect(global_embedding, self._geo_latent)
        else:
            self.ood_guard.check(global_embedding, self._geo_latent)

        return output

    def close(self) -> None:
        """Remove the geometry forward hook installed on the wrapped model.

        Call this when the wrapper is no longer needed to detach the hook from
        the wrapped model's ``geometry_tokenizer``.  Idempotent and safe to call
        when no geometry surface (and hence no hook) was registered.
        """
        if self._geo_hook_handle is not None:
            self._geo_hook_handle.remove()
            self._geo_hook_handle = None


def attach_ood_guard(
    model: nn.Module,
    config: OODGuardConfig,
    *,
    global_dim: int | None = None,
    geometry_embed_dim: int | None = None,
) -> GuardedGeoTransolver:
    """Convenience alias for :class:`GuardedGeoTransolver`.

    See :class:`GuardedGeoTransolver` for the full parameter description.
    """
    return GuardedGeoTransolver(
        model,
        config,
        global_dim=global_dim,
        geometry_embed_dim=geometry_embed_dim,
    )
