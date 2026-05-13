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


"""
Autoregressive Rollout for Transolver VOF Prediction.

Training:  Model processes ALL nodes; per-timestep interface masking is
           applied in the loss only (in train.py).
Inference: Model processes ALL nodes; no masking needed.
"""

from typing import Optional

import torch
from torch.utils.checkpoint import checkpoint as ckpt

from physicsnemo.experimental.models.geotransolver import GeoTransolver
from datapipe import SimSample


# ═══════════════════════════════════════════════════════════════════════════════
# Interface-band utility
# ═══════════════════════════════════════════════════════════════════════════════


def compute_interface_band(
    vof: torch.Tensor,
    coords: torch.Tensor,
    vof_lo: float = 0.01,
    vof_hi: float = 0.99,
    band_fraction: float = 0.05,
    interface_axis: int = -1,
    absolute_expansion: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute a boolean mask selecting nodes in or near the VOF interface.

    The "interface" is defined as nodes with partially-filled VOF values
    (``vof_lo < vof < vof_hi``). Nearby nodes along the detected thickness
    axis are included to form a band around the interface.

    Args:
        vof: Volume-of-fluid values, shape ``[N]`` or ``[N, 1]``. Values
             must be in ``[0, 1]``.
        coords: Node coordinates, shape ``[N, 3]`` in the same units.
        vof_lo: Lower threshold for partially-filled cells.
        vof_hi: Upper threshold for partially-filled cells.
        band_fraction: Expansion amount as a fraction of the domain extent
                       along the interface axis. Ignored when
                       ``absolute_expansion`` is provided.
        interface_axis: Axis (0, 1, or 2) along which the interface is
                        thin. If ``-1``, the axis is auto-detected as the
                        one with smallest spread of interface points.
        absolute_expansion: Explicit expansion in coordinate units. Use
                            this when coordinates are normalized and a
                            fraction of the domain extent would be
                            meaningless.

    Returns:
        Boolean mask of shape ``[N]``. All False if no interface exists
        at this timestep.
    """
    if vof.ndim == 2:
        vof = vof[:, 0]

    N = vof.shape[0]
    device = vof.device

    core = (vof > vof_lo) & (vof < vof_hi)

    if not core.any():
        return torch.zeros(N, dtype=torch.bool, device=device)

    # Auto-detect thickness axis if requested
    axis = interface_axis
    if axis == -1:
        iface_coords = coords[core]
        spreads = iface_coords.max(dim=0).values - iface_coords.min(dim=0).values
        axis = int(spreads.argmin().item())

    iface_z = coords[core, axis]
    z_min = iface_z.min()
    z_max = iface_z.max()

    if absolute_expansion is not None:
        expansion = absolute_expansion
    else:
        domain_extent = coords[:, axis].max() - coords[:, axis].min() + 1e-8
        expansion = band_fraction * domain_extent

    band = (coords[:, axis] >= z_min - expansion) & (
        coords[:, axis] <= z_max + expansion
    )
    return band


# ═══════════════════════════════════════════════════════════════════════════════
# Autoregressive rollout model
# ═══════════════════════════════════════════════════════════════════════════════


class TransolverAutoregressiveRollout(GeoTransolver):
    """
    GeoTransolver wrapper with autoregressive rollout for transient
    volume-of-fluid (VOF) prediction.

    At each rollout step, the model:
      1. Builds the input feature vector for every node:
         ``[vof_t, coords, fourier(coords)]``
      2. Runs the base GeoTransolver to produce a delta
      3. Applies a sigmoid to obtain ``vof_{t+1}`` in ``[0, 1]``
      4. Uses ``vof_{t+1}`` as input to the next step

    Training vs. inference:
      - The model always processes ALL nodes — there is no node subsetting
        inside this class.
      - Interface-band masking is applied only in the training loss
        (in ``train.py``), not here.
      - During training, gradient checkpointing is used per rollout step
        to reduce activation memory.

    Args:
        functional_dim: Per-node input dimension passed to the base
                        GeoTransolver. Must equal
                        ``1 + 3 + 2 * 2 * num_fourier_frequencies``
                        (VOF scalar + 3D coords + sin/cos Fourier features
                        on the 2D projection).
        out_dim: Output dimension per node. Should be 1 for VOF.
        num_time_steps: Total number of timesteps including the initial
                        state. Rollout produces ``num_time_steps - 1``
                        future predictions.
        dt: Physical time step (stored for reference; not used in the
            forward pass).
        num_fourier_frequencies: Number of dyadic Fourier frequency bands
                                 applied to the in-plane (x, z) coords.
                                 Output Fourier dimension is
                                 ``2 * 2 * num_fourier_frequencies``.
        fourier_base: Base multiplier for the Fourier frequencies. The
                      frequencies are ``fourier_base * 2^[0, 1, ...]``.
        **kwargs: Additional keyword arguments forwarded to the parent
                  ``GeoTransolver`` constructor (e.g. ``n_hidden``,
                  ``n_layers``, ``slice_num``, ``use_te``).
    """

    def __init__(
        self,
        functional_dim: int,
        out_dim: int,
        *,  # everything after here is keyword-only
        num_time_steps: int = 20,
        dt: float = 5e-3,
        num_fourier_frequencies: int = 3,
        fourier_base: int = 1,
        **kwargs,
    ):
        # Store rollout-specific state
        self.dt = dt
        self.num_time_steps = num_time_steps
        self.rollout_steps = num_time_steps - 1
        self.num_fourier_frequencies = num_fourier_frequencies
        self.fourier_base = fourier_base

        # Initialize the base GeoTransolver with the user-supplied
        # architecture arguments. Any extra kwargs are passed through
        # transparently; this class only intercepts rollout-specific ones.
        super().__init__(
            functional_dim=functional_dim,
            out_dim=out_dim,
            **kwargs,
        )

    def forward(
        self,
        sample: SimSample,
        data_stats: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Run autoregressive rollout on all nodes for ``rollout_steps`` steps.

        Args:
            sample: ``SimSample`` with:
                - ``node_features["coords"]``: ``[N, 3]`` normalized coords.
                - ``node_features["features"]``: ``[N, 1]`` VOF at t=0.
            data_stats: Unused; kept for API compatibility with the shared
                        trainer interface.

        Returns:
            Tensor of shape ``[T, N, 1]`` giving predicted VOF for every
            node at every future timestep (``T = rollout_steps``).
        """
        del data_stats  # unused; included for compatibility

        coords = sample.node_features["coords"]  # [N, 3]
        vof_t = sample.node_features["features"]  # [N, 1]

        outputs: list[torch.Tensor] = []

        for _ in range(self.rollout_steps):
            fourier = self._fourier_features(coords)  # [N, F_fourier]
            fx_t = torch.cat([vof_t, coords, fourier], dim=-1)  # [N, functional_dim]

            if self.training:
                delta = ckpt(
                    self._forward_step,
                    fx_t.unsqueeze(0),
                    coords.unsqueeze(0),
                    use_reentrant=False,
                ).squeeze(0)
            else:
                delta = self._forward_step(
                    fx_t.unsqueeze(0),
                    coords.unsqueeze(0),
                ).squeeze(0)

            vof_next = torch.sigmoid(delta)
            outputs.append(vof_next)
            vof_t = vof_next

        return torch.stack(outputs, dim=0)  # [T, N, 1]

    def _forward_step(
        self,
        fx: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single-step base-model forward pass.

        Args:
            fx: Per-node input features, shape ``[B, N, functional_dim]``.
            coords: Per-node coordinates, shape ``[B, N, 3]``.

        Returns:
            Raw model output (pre-sigmoid delta), shape ``[B, N, out_dim]``.
        """
        return super().forward(
            local_embedding=fx,
            geometry=coords,
            local_positions=coords,
        )

    def _fourier_features(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute dyadic Fourier positional encoding on the in-plane
        ``(x, z)`` coordinates.

        We intentionally encode only two of the three coordinates because
        the problem has a dominant thickness axis along y; encoding y
        alongside the others would waste capacity on a short direction
        with little spatial variation.

        Args:
            coords: Shape ``[N, 3]`` or ``[B, N, 3]``.

        Returns:
            Fourier features of shape
            ``[N, 2 * 2 * num_fourier_frequencies]`` or
            ``[B, N, 2 * 2 * num_fourier_frequencies]``.
        """
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)[:, :, [0, 2]]  # keep (x, z)
            squeeze = True
        else:
            coords = coords[:, :, [0, 2]]
            squeeze = False

        B, N, D = coords.shape
        assert D == 2, f"Expected 2D (x, z) coordinates, got D={D}"

        freqs = self.fourier_base * (
            2.0
            ** torch.arange(
                self.num_fourier_frequencies,
                device=coords.device,
                dtype=coords.dtype,
            )
        )
        phases = coords.unsqueeze(-1) * (2.0 * torch.pi * freqs)  # [B, N, 2, F]
        sin_enc = torch.sin(phases)
        cos_enc = torch.cos(phases)
        enc = torch.cat([sin_enc, cos_enc], dim=-1)  # [B, N, 2, 2F]
        enc = enc.reshape(B, N, 2 * 2 * self.num_fourier_frequencies)

        if squeeze:
            enc = enc.squeeze(0)
        return enc
