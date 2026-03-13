# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Autoregressive Rollout for Transolver VOF Prediction.

Training:  model processes ALL nodes; per-timestep interface masking
           is applied in the loss only (in train.py).
Inference: model processes ALL nodes; no masking needed.
"""

import torch
from torch.utils.checkpoint import checkpoint as ckpt

from physicsnemo.experimental.models.geotransolver import GeoTransolver
from datapipe import SimSample


def compute_interface_band(
    vof: torch.Tensor,
    coords: torch.Tensor,
    vof_lo: float = 0.01,
    vof_hi: float = 0.99,
    band_fraction: float = 0.05,
    interface_axis: int = -1,
    absolute_expansion: float = None,
) -> torch.Tensor:
    """
    Compute a boolean mask selecting nodes in/near the VOF interface.

    If absolute_expansion is set, uses that directly as the ± expansion
    in coordinate units (useful for normalized coordinates).
    Otherwise falls back to band_fraction * domain_extent.

    Returns all-False mask if no interface exists at this timestep.
    """
    if vof.ndim == 2:
        vof = vof[:, 0]

    N = vof.shape[0]
    device = vof.device

    core = (vof > vof_lo) & (vof < vof_hi)

    if not core.any():
        return torch.zeros(N, dtype=torch.bool, device=device)

    # Auto-detect axis
    axis = interface_axis
    if axis == -1:
        iface_coords = coords[core]
        spreads = iface_coords.max(dim=0).values - iface_coords.min(dim=0).values
        axis = spreads.argmin().item()

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




class TransolverAutoregressiveRollout(GeoTransolver):
    """
    GeoTransolver with autoregressive rollout for Epoxy VOF prediction.

    Always processes ALL nodes. Interface masking is handled in the loss
    (train.py), not here.
    """

    def __init__(self, *args, **kwargs):
        self.dt: float = kwargs.pop("dt", 5e-3)
        self.rollout_steps: int = kwargs.pop("num_time_steps", 20) - 1
        self.num_fourier_frequencies: int = kwargs.pop("num_fourier_frequencies", 3)
        self.fourier_base: int = kwargs.pop("fourier_base", 1)
        kwargs.pop("initial_vel", None)
        # Pop any interface params that might be in model config
        kwargs.pop("vof_lo", None)
        kwargs.pop("vof_hi", None)
        kwargs.pop("infer_band_fraction", None)
        kwargs.pop("interface_axis", None)
        super().__init__(*args, **kwargs)

    def forward(self, sample: SimSample, data_stats: dict, **kwargs) -> torch.Tensor:
        """
        Autoregressive rollout on ALL nodes.

        Args:
            sample: SimSample with coords [N,3] and features [N,1]
            data_stats: normalization statistics

        Returns:
            [T, N, 1] predicted VOF for all nodes at all timesteps
        """
        coords = sample.node_features["coords"]    # [N, 3]
        vof_t = sample.node_features["features"]    # [N, 1]

        outputs: list[torch.Tensor] = []

        for t in range(self.rollout_steps):
            fourier = self._fourier_features(coords)
            fx_t = torch.cat([vof_t, coords, fourier], dim=-1)

            if self.training:
                delta = ckpt(
                    self._forward_step,
                    fx_t.unsqueeze(0),
                    coords.unsqueeze(0),
                    use_reentrant=False,
                ).squeeze(0)
            else:
                delta = self._forward_step(
                    fx_t.unsqueeze(0), coords.unsqueeze(0)
                ).squeeze(0)

            vof_next = torch.sigmoid(delta)
            outputs.append(vof_next)
            vof_t = vof_next

        return torch.stack(outputs, dim=0)  # [T, N, 1]

    def _forward_step(self, fx: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return super().forward(
            local_embedding=fx, geometry=coords, local_positions=coords
        )

    def _fourier_features(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)[:, :, [0, 2]]
            squeeze = True
        else:
            squeeze = False

        B, N, D = coords.shape
        assert D == 2, f"Expected 2D coordinates, got D={D}"

        freqs = self.fourier_base * (
            2.0
            ** torch.arange(
                self.num_fourier_frequencies, device=coords.device, dtype=coords.dtype
            )
        )
        phases = coords.unsqueeze(-1) * (2.0 * torch.pi * freqs)
        sin_enc = torch.sin(phases)
        cos_enc = torch.cos(phases)
        enc = torch.cat([sin_enc, cos_enc], dim=-1)
        enc = enc.reshape(B, N, 2 * 2 * self.num_fourier_frequencies)

        if squeeze:
            enc = enc.squeeze(0)
        return enc
