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

r"""Wrap a GeoTransolver to enable out-of-distribution guarding.

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
from jaxtyping import Float

from physicsnemo import Module

from .ood_guard import OODGuard, OODGuardConfig

__all__ = ["GuardedGeoTransolver", "attach_ood_guard"]


def _infer_geometry_embed_dim(model: nn.Module) -> int | None:
    r"""Infer the head dimension of the geometry latent.

    Parameters
    ----------
    model : torch.nn.Module
        GeoTransolver instance whose geometry tokenizer is inspected.

    Returns
    -------
    int | None
        Head dimension of the geometry latent, or ``None`` if geometry is
        disabled.
    """
    tokenizer = getattr(model.context_builder, "geometry_tokenizer", None)
    if tokenizer is None:
        return None
    return int(tokenizer.dim_head)


def _infer_global_dim(model: nn.Module) -> int | None:
    r"""Infer the channel dimension of the global embedding.

    Parameters
    ----------
    model : torch.nn.Module
        GeoTransolver instance whose global tokenizer is inspected.

    Returns
    -------
    int | None
        Channel dimension of the global embedding, or ``None`` if global
        context is disabled.
    """
    tokenizer = getattr(model.context_builder, "global_tokenizer", None)
    if tokenizer is None:
        return None
    return int(tokenizer.in_project_x.in_features)


class GuardedGeoTransolver(Module):
    r"""GeoTransolver wrapped with an out-of-distribution guard.

    The wrapper delegates the forward pass to the wrapped model unchanged and,
    as a side effect, feeds the guard.  During training the guard accumulates
    calibration statistics; during inference it checks incoming data against
    them and emits warnings on out-of-distribution inputs.  Switch behaviour via
    the standard :meth:`~torch.nn.Module.train` / :meth:`~torch.nn.Module.eval`
    toggles — the wrapped model's ``training`` flag selects collect vs. check.

    Parameters
    ----------
    model : :class:`~physicsnemo.models.geotransolver.GeoTransolver`
        Constructed model instance. At least one of the geometry or global
        surfaces must be enabled for the guard to have anything to watch.
    config : :class:`~physicsnemo.experimental.guardrails.embedded.OODGuardConfig`
        Guard configuration (``buffer_size`` required; ``knn_k`` and
        ``sensitivity`` optional).
    global_dim : int | None, optional, default=None
        Channel dimension of the global embedding.  Inferred from the model's
        ``context_builder`` when ``None``.
    geometry_embed_dim : int | None, optional, default=None
        Dimensionality of the pooled geometry latent.  Inferred from the model's
        ``context_builder`` when ``None``.

    Forward
    -------
    local_embedding : torch.Tensor | tuple[torch.Tensor, ...]
        Local input of shape :math:`(B, N, C)` for an unstructured mesh, or the
        structured layouts accepted by the wrapped model.
    local_positions : torch.Tensor | tuple[torch.Tensor, ...] | None, optional, default=None
        Local positions of shape :math:`(B, N, D_p)`.
    global_embedding : torch.Tensor | None, optional, default=None
        Global input of shape :math:`(B, N_g, C_g)`.
    geometry : torch.Tensor | None, optional, default=None
        Geometry input of shape :math:`(B, N, C_{geo})`, or the corresponding
        structured layout accepted by the wrapped model.
    time : torch.Tensor | None, optional, default=None
        Time input forwarded to the wrapped model.
    return_embedding_states : bool, optional, default=False
        Whether the wrapped model also returns its context embedding states.

    Outputs
    -------
    torch.Tensor | tuple[torch.Tensor, ...]
        Output from the wrapped model, unchanged. When
        ``return_embedding_states=True``, the wrapped model returns the output
        together with context states of shape :math:`(B, H, S, D_c)`.

    Notes
    -----
    Construction raises a ``ValueError`` if neither the geometry nor the global
    surface is enabled, because the guard would have nothing to observe.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models.geotransolver import GeoTransolver
    >>> from physicsnemo.experimental.guardrails.embedded import (
    ...     GuardedGeoTransolver, OODGuardConfig,
    ... )
    >>> model = GeoTransolver(
    ...     functional_dim=8,
    ...     out_dim=3,
    ...     geometry_dim=3,
    ...     global_dim=4,
    ...     n_hidden=32,
    ...     n_head=4,
    ...     n_layers=2,
    ...     use_te=False,
    ... )
    >>> guarded = GuardedGeoTransolver(model, OODGuardConfig(buffer_size=128))
    >>> local = torch.randn(2, 16, 8)
    >>> geometry = torch.randn(2, 16, 3)
    >>> global_embedding = torch.randn(2, 1, 4)
    >>> output = guarded(
    ...     local, global_embedding=global_embedding, geometry=geometry
    ... )
    >>> output.shape
    torch.Size([2, 16, 3])
    """

    def __init__(
        self,
        model: nn.Module,
        config: OODGuardConfig,
        *,
        global_dim: int | None = None,
        geometry_embed_dim: int | None = None,
    ) -> None:
        r"""Initialize the guarded GeoTransolver wrapper.

        Parameters
        ----------
        model : :class:`~physicsnemo.models.geotransolver.GeoTransolver`
            Constructed model instance to wrap.
        config : :class:`~physicsnemo.experimental.guardrails.embedded.OODGuardConfig`
            Guard configuration.
        global_dim : int | None, optional, default=None
            Channel dimension of the global embedding. Inferred when ``None``.
        geometry_embed_dim : int | None, optional, default=None
            Dimension of the pooled geometry latent. Inferred when ``None``.

        Returns
        -------
        None
            The wrapper is initialized in place.
        """
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
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: Float[torch.Tensor, "batch heads slices head_dim"],
    ) -> None:
        r"""Pool geometry slice tokens captured by the tokenizer hook.

        Parameters
        ----------
        module : torch.nn.Module
            Geometry tokenizer that produced ``output``.
        inputs : tuple[object, ...]
            Positional inputs received by the geometry tokenizer.
        output : torch.Tensor
            Slice tokens of shape :math:`(B, H, S, D)`.

        Returns
        -------
        None
            The pooled latent of shape :math:`(B, D)` is stored on the wrapper.
        """
        # Detach so the guard's buffers never keep the backward graph alive.
        self._geo_latent = output.detach().mean(dim=(1, 2))

    def forward(
        self,
        local_embedding: (
            Float[torch.Tensor, "batch ... features"]
            | tuple[Float[torch.Tensor, "batch ... features"], ...]
        ),
        local_positions: (
            Float[torch.Tensor, "batch ... spatial_dim"]
            | tuple[Float[torch.Tensor, "batch ... spatial_dim"], ...]
            | None
        ) = None,
        global_embedding: Float[torch.Tensor, "batch global_tokens global_dim"]
        | None = None,
        geometry: Float[torch.Tensor, "batch ... geometry_dim"] | None = None,
        time: Float[torch.Tensor, "batch ..."] | None = None,
        return_embedding_states: bool = False,
    ) -> (
        Float[torch.Tensor, "batch ... out_dim"]
        | tuple[Float[torch.Tensor, "batch ... out_dim"], ...]
        | tuple[
            (
                Float[torch.Tensor, "batch ... out_dim"]
                | tuple[Float[torch.Tensor, "batch ... out_dim"], ...]
            ),
            Float[torch.Tensor, "batch heads slices context_dim"] | None,
        ]
    ):
        r"""Run the wrapped model and update or check the OOD guard.

        Full input and output documentation is provided in the class-level
        ``Forward`` and ``Outputs`` sections.
        """
        ### Input validation
        if not torch.compiler.is_compiling():
            local_tensors = (
                (local_embedding,)
                if isinstance(local_embedding, torch.Tensor)
                else local_embedding
            )
            if not local_tensors:
                raise ValueError("Expected non-empty local_embedding")

            structured_shape = getattr(self.model, "structured_shape", None)
            valid_local_ranks = {3}
            if structured_shape is not None:
                valid_local_ranks.add(len(structured_shape) + 2)
            for index, tensor in enumerate(local_tensors):
                if tensor.ndim not in valid_local_ranks:
                    expected = " or ".join(
                        f"{rank}D" for rank in sorted(valid_local_ranks)
                    )
                    raise ValueError(
                        f"Expected {expected} local_embedding tensor at index "
                        f"{index} but got tensor of shape {tuple(tensor.shape)}"
                    )

            if local_positions is not None:
                position_tensors = (
                    (local_positions,)
                    if isinstance(local_positions, torch.Tensor)
                    else local_positions
                )
                for index, tensor in enumerate(position_tensors):
                    if tensor.ndim != 3:
                        raise ValueError(
                            "Expected local_positions tensor of shape (B, N, D_p) "
                            f"at index {index} but got tensor of shape "
                            f"{tuple(tensor.shape)}"
                        )

            if global_embedding is not None and global_embedding.ndim != 3:
                raise ValueError(
                    "Expected global_embedding tensor of shape (B, N_g, C_g) "
                    f"but got tensor of shape {tuple(global_embedding.shape)}"
                )
            if (
                global_embedding is not None
                and self.ood_guard.global_min is not None
                and global_embedding.shape[-1] != self.ood_guard.global_min.shape[0]
            ):
                expected_dim = self.ood_guard.global_min.shape[0]
                raise ValueError(
                    "Expected global_embedding tensor of shape "
                    f"(B, N_g, {expected_dim}) but got tensor of shape "
                    f"{tuple(global_embedding.shape)}"
                )

            if geometry is not None:
                valid_geometry_ranks = valid_local_ranks
                if geometry.ndim not in valid_geometry_ranks:
                    expected = " or ".join(
                        f"{rank}D" for rank in sorted(valid_geometry_ranks)
                    )
                    raise ValueError(
                        f"Expected {expected} geometry tensor but got tensor of "
                        f"shape {tuple(geometry.shape)}"
                    )

        self._geo_latent = None
        output = self.model(
            local_embedding,
            local_positions=local_positions,
            global_embedding=global_embedding,
            geometry=geometry,
            time=time,
            return_embedding_states=return_embedding_states,
        )

        if self.model.training:
            self.ood_guard.collect(global_embedding, self._geo_latent)
        else:
            self.ood_guard.check(global_embedding, self._geo_latent)

        return output

    def close(self) -> None:
        r"""Remove the geometry forward hook installed on the wrapped model.

        Call this when the wrapper is no longer needed to detach the hook from
        the wrapped model's ``geometry_tokenizer``.  Idempotent and safe to call
        when no geometry surface (and hence no hook) was registered.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The method removes the hook in place.
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
    r"""Attach an out-of-distribution guard to a GeoTransolver.

    Parameters
    ----------
    model : :class:`~physicsnemo.models.geotransolver.GeoTransolver`
        Constructed model instance to wrap.
    config : :class:`~physicsnemo.experimental.guardrails.embedded.OODGuardConfig`
        Guard configuration.
    global_dim : int | None, optional, default=None
        Channel dimension of the global embedding. Inferred when ``None``.
    geometry_embed_dim : int | None, optional, default=None
        Dimension of the pooled geometry latent. Inferred when ``None``.

    Returns
    -------
    :class:`~physicsnemo.experimental.guardrails.embedded.GuardedGeoTransolver`
        Wrapper that observes the model's global and geometry surfaces.
    """
    return GuardedGeoTransolver(
        model,
        config,
        global_dim=global_dim,
        geometry_embed_dim=geometry_embed_dim,
    )
