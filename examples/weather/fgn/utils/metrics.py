# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation diagnostics for FGN, drawing on Figures 2 + 3 of arXiv:2506.10772v1.

Canonical metric library: **earth2studio has coord-aware equivalents** of
every metric here under ``earth2studio.statistics`` — ``spread_skill_ratio``,
``rmse``, ``rank_histogram``, ``crps``, ``lsd`` (log spectral distance),
``energy_score``, ``fss``, plus ``weights.lat_weight`` for ``cos(lat)``. Those
classes operate on ``(tensor, CoordSystem)`` pairs and are the right choice
for xarray-style evaluation pipelines. This module intentionally uses
lightweight torch kernels against the FGN trainer's pure
``(B, K, M, C, H, W)`` tensors to keep the inline validation hook cheap and
dependency-free; docstrings below cite the canonical equivalent for each
diagnostic so a later refactor can swap them in.

What we compute per validation rollout:

- **CRPS per variable per lead time** (Figure 2a, without a baseline).
  Delegates to :func:`physicsnemo.metrics.general.crps.kcrps` with
  ``biased=False`` (the Zamo-Naveau fair estimator used for training).
  Canonical coord-aware equivalent: :class:`earth2studio.statistics.crps`.
- **Spread-skill ratio per variable per lead time** (Figure 2 b-f).
  Standard definition: ``spread = sqrt(mean over grid of var across
  members)`` vs ``skill = sqrt(mean over grid of MSE of ensemble mean)``.
  A well-calibrated ensemble sits near 1.
  Canonical: :class:`earth2studio.statistics.spread_skill_ratio`.
- **Ensemble-mean RMSE per variable per lead time** (Figure 2 companion).
  Canonical: :class:`earth2studio.statistics.rmse`.
- **Rank histogram** per variable, aggregated over grid + validation
  batches. A uniform histogram indicates good calibration; U-shaped →
  under-dispersive, hump-shaped → over-dispersive.
  Canonical: :class:`earth2studio.statistics.rank_histogram`.
- **Energy score per lead time** (multivariate CRPS generalisation).
  Computed over the *variable* axis so that cross-channel calibration
  is captured, averaged over a spatially subsampled grid to keep the
  O(M²) pairwise term cheap.  A single scalar per lead; lower is better.
  Added in earth2studio 0.13.0 as :class:`earth2studio.statistics.energy_score`.
- **Azimuthal 1D power spectra** per variable for ensemble-mean vs
  ground truth (Figure 3 e-j). Uses
  :func:`physicsnemo.metrics.general.power_spectrum.power_spectrum` —
  2D-FFT azimuthally averaged, a reasonable proxy for the paper's
  spherical-harmonic spectra without adding a ``torch-harmonics`` dep.
  Honeycomb artifacts at the mesh frequency (Figure 5) would surface here
  as a localised high-frequency bump.
- **Derived-variable CRPS** for `10m wind speed = sqrt(u10m^2 + v10m^2)`
  and `z300 - z500` (Figure 3 c-d), when those component variables are
  present in the state channel list.

Deferred (require more scope or data we don't have yet):

- FGN-vs-baseline scorecards (no baseline model wired in).
- Pooled CRPS (Figure 3 a-b) — pool-size sweep.
- REV for extreme thresholds (Figure 2 g-h) — needs a climatology.
- Cyclone track evaluation (Figure 4) — needs IBTrACS + Tempest Extremes.
- Direct honeycomb-artifact viz (Figure 5) — implicit in the spectra plot.

earth2studio examples (see ``earth2studio/examples/02_medium_range/``) use
raw ``matplotlib.pyplot`` + ``cartopy`` for geospatial plots — there is no
shared plotting wrapper to delegate to. The plot helpers below mirror that
convention: plain ``matplotlib`` with ``Agg`` backend so the hook stays
headless-safe.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from physicsnemo.metrics.general.crps import kcrps
from physicsnemo.metrics.general.power_spectrum import power_spectrum

# ---------------------------------------------------------------------------
# Metric computations. All expect float32 tensors on the same device and
# return CPU numpy arrays for easy plotting / logging.
# ---------------------------------------------------------------------------


def _check_shapes(ensemble: torch.Tensor, target: torch.Tensor) -> None:
    """Validate the shared (B, K, M, C, H, W) vs (B, K, C, H, W) layout."""
    if ensemble.ndim != 6:
        raise ValueError(
            f"ensemble must have shape [B, K, M, C, H, W], got {tuple(ensemble.shape)}"
        )
    if target.ndim != 5:
        raise ValueError(
            f"target must have shape [B, K, C, H, W], got {tuple(target.shape)}"
        )
    if ensemble.shape[0] != target.shape[0] or ensemble.shape[1] != target.shape[1]:
        raise ValueError(
            f"ensemble/target batch + lead dims must match, got {tuple(ensemble.shape)} vs {tuple(target.shape)}"
        )


def crps_per_variable_per_lead(
    ensemble: torch.Tensor, target: torch.Tensor
) -> np.ndarray:
    """Fair CRPS averaged over batch + spatial dims, retaining (K, C).

    Shapes: ensemble (B, K, M, C, H, W), target (B, K, C, H, W).
    Returns numpy array of shape (K, C).
    """
    _check_shapes(ensemble, target)
    B, K, M, C, H, W = ensemble.shape
    flat_ens = ensemble.reshape(B * K, M, C, H, W)
    flat_tgt = target.reshape(B * K, C, H, W)
    per_loc = kcrps(flat_ens, flat_tgt, dim=1, biased=False)  # (B*K, C, H, W)
    per_loc = per_loc.reshape(B, K, C, H, W)
    return per_loc.mean(dim=(0, -2, -1)).detach().cpu().numpy()


def ensemble_rmse_per_variable_per_lead(
    ensemble: torch.Tensor, target: torch.Tensor
) -> np.ndarray:
    """sqrt(mean((mean_n x_n − y)^2)) per lead, per channel. Shape (K, C)."""
    _check_shapes(ensemble, target)
    mean_pred = ensemble.mean(dim=2)  # (B, K, C, H, W)
    sq = (mean_pred - target) ** 2
    mse = sq.mean(dim=(0, -2, -1))  # (K, C)
    return mse.sqrt().detach().cpu().numpy()


def spread_skill_per_variable_per_lead(
    ensemble: torch.Tensor, target: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensemble spread, skill, and their ratio — per lead, per channel.

    Standard definitions:
      skill  = sqrt(mean over grid of MSE of ensemble mean)
      spread = sqrt(mean over grid of ensemble variance across members)

    A well-calibrated ensemble has spread/skill ≈ 1 (paper Figure 2 b-f).

    Returns ``(spread, skill, ratio)`` each of shape (K, C) as numpy.
    """
    _check_shapes(ensemble, target)
    # Variance across member axis (dim=2), unbiased estimator.
    member_var = ensemble.var(dim=2, unbiased=True)  # (B, K, C, H, W)
    spread = member_var.mean(dim=(0, -2, -1)).sqrt()
    skill = ensemble_rmse_per_variable_per_lead(ensemble, target)
    skill_t = torch.from_numpy(skill).to(spread.device)
    ratio = spread / skill_t.clamp_min(1e-12)
    return (
        spread.detach().cpu().numpy(),
        skill,
        ratio.detach().cpu().numpy(),
    )


def rank_histogram_per_variable(
    ensemble: torch.Tensor, target: torch.Tensor, num_bins: int | None = None
) -> np.ndarray:
    """Verification rank histogram per channel, aggregated over batch/lead/grid.

    For each observation, count the rank of the truth among the sorted
    ensemble members; the histogram over many observations reveals
    calibration. With ``M`` members there are ``M + 1`` possible ranks.

    Returns an integer numpy array of shape ``(C, num_bins)`` with
    ``num_bins = M + 1`` by default.
    """
    _check_shapes(ensemble, target)
    B, K, M, C, H, W = ensemble.shape
    bins = num_bins if num_bins is not None else M + 1
    # For each position, rank of target among members is the count of
    # members strictly less than target (breaks ties by assigning the
    # lowest possible rank). Paper-standard convention uses a random
    # tiebreak; with float32 data ties are rare, so strict-less-than is
    # a reasonable MVP.
    less = (ensemble < target.unsqueeze(2)).sum(dim=2)  # (B, K, C, H, W)
    # Rank values are in 0..M inclusive. Scale/clip to 0..bins-1.
    less = less.to(torch.long)
    if bins != M + 1:
        less = (less * bins // (M + 1)).clamp(0, bins - 1)
    hist = torch.zeros(C, bins, dtype=torch.long, device=less.device)
    for c in range(C):
        flat = less[..., c, :, :].reshape(-1)
        hist[c] = torch.bincount(flat, minlength=bins)
    return hist.detach().cpu().numpy()


def derived_variable_crps(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    variables: Sequence[str],
) -> dict[str, np.ndarray]:
    """Paper Figure 3 c-d: CRPS of derived quantities.

    - ``wspd10m = sqrt(u10m^2 + v10m^2)`` when both ``u10m`` and ``v10m``
      are present.
    - ``z300_minus_z500 = z300 - z500`` when both levels are present.

    Returns a dict of ``name -> (K,)`` CRPS arrays; empty dict if neither
    derived quantity is available.
    """
    _check_shapes(ensemble, target)
    idx = {name: i for i, name in enumerate(variables)}
    derived: dict[str, np.ndarray] = {}

    if "u10m" in idx and "v10m" in idx:
        u_ens = ensemble[:, :, :, idx["u10m"]]
        v_ens = ensemble[:, :, :, idx["v10m"]]
        u_tgt = target[:, :, idx["u10m"]]
        v_tgt = target[:, :, idx["v10m"]]
        wspd_ens = torch.sqrt(u_ens**2 + v_ens**2).unsqueeze(3)  # (B,K,M,1,H,W)
        wspd_tgt = torch.sqrt(u_tgt**2 + v_tgt**2).unsqueeze(2)  # (B,K,1,H,W)
        per_loc = kcrps(
            wspd_ens.reshape(-1, wspd_ens.shape[2], 1, *wspd_ens.shape[-2:]),
            wspd_tgt.reshape(-1, 1, *wspd_tgt.shape[-2:]),
            dim=1,
            biased=False,
        )
        B, K = ensemble.shape[0], ensemble.shape[1]
        per_loc = per_loc.reshape(B, K, 1, *per_loc.shape[-2:])
        derived["wspd10m"] = per_loc.mean(dim=(0, 2, -2, -1)).detach().cpu().numpy()

    if "z300" in idx and "z500" in idx:
        dz_ens = ensemble[:, :, :, idx["z300"]] - ensemble[:, :, :, idx["z500"]]
        dz_tgt = target[:, :, idx["z300"]] - target[:, :, idx["z500"]]
        dz_ens = dz_ens.unsqueeze(3)
        dz_tgt = dz_tgt.unsqueeze(2)
        per_loc = kcrps(
            dz_ens.reshape(-1, dz_ens.shape[2], 1, *dz_ens.shape[-2:]),
            dz_tgt.reshape(-1, 1, *dz_tgt.shape[-2:]),
            dim=1,
            biased=False,
        )
        B, K = ensemble.shape[0], ensemble.shape[1]
        per_loc = per_loc.reshape(B, K, 1, *per_loc.shape[-2:])
        derived["z300_minus_z500"] = (
            per_loc.mean(dim=(0, 2, -2, -1)).detach().cpu().numpy()
        )

    return derived


def power_spectra_per_variable(
    ensemble_mean: torch.Tensor, target: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Azimuthal 1D power spectrum of ensemble-mean and target, per channel.

    ``ensemble_mean`` and ``target`` are ``(B, K, C, H, W)`` tensors; the
    spectrum is averaged over batch and any requested lead times by the
    caller (pass in ``[:, lead_idx:lead_idx+1]`` to isolate a single lead).

    Uses :func:`physicsnemo.metrics.general.power_spectrum.power_spectrum`
    — 2D-FFT azimuthal averaging. Not a true spherical-harmonic spectrum
    (the paper uses spherical harmonics), but a cheap proxy that still
    surfaces the mesh-frequency spike described in Figure 3e / Figure 5.

    Returns ``(k_bins, ens_spectra, tgt_spectra)`` with shapes
    ``(nbins,)`` and ``(K, C, nbins)`` respectively.
    """
    if ensemble_mean.shape != target.shape:
        raise ValueError(
            f"ensemble_mean/target shapes must match, got {tuple(ensemble_mean.shape)} vs {tuple(target.shape)}"
        )
    # Average spectrum over batch on a per-lead, per-channel basis.
    B, K, C, H, W = ensemble_mean.shape
    ens_flat = ensemble_mean.reshape(B * K * C, H, W)
    tgt_flat = target.reshape(B * K * C, H, W)
    k, ens_pow = power_spectrum(ens_flat)
    _, tgt_pow = power_spectrum(tgt_flat)
    ens_pow = ens_pow.reshape(B, K, C, -1).mean(dim=0)
    tgt_pow = tgt_pow.reshape(B, K, C, -1).mean(dim=0)
    return (
        k.detach().cpu().numpy(),
        ens_pow.detach().cpu().numpy(),
        tgt_pow.detach().cpu().numpy(),
    )


def energy_score_per_lead(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    spatial_stride: int = 8,
    fair: bool = True,
) -> np.ndarray:
    """Fair energy score (multivariate CRPS) per lead, averaged over variables + grid.

    The energy score is the multivariate generalisation of CRPS:

        ES = E[||X - y||] - (1/2) E[||X - X'||]

    where the norm is taken over the *variable* axis (dim C) at each spatial
    point. This captures cross-channel calibration that per-variable CRPS misses.

    Computing the O(M²) pairwise term over the full 721 × 1440 grid is
    expensive; ``spatial_stride`` sub-samples before computing to keep it fast.
    With the default stride of 8 the spatial footprint is ~91 × 180 = 16 k
    points, well within budget for a validation hook.

    Shapes: ensemble (B, K, M, C, H, W), target (B, K, C, H, W).
    Returns numpy array of shape (K,).

    Canonical coord-aware equivalent: :class:`earth2studio.statistics.energy_score`
    (added in earth2studio 0.13.0, March 2026).
    """
    _check_shapes(ensemble, target)
    B, K, M, C, H, W = ensemble.shape

    ens = ensemble[:, :, :, :, ::spatial_stride, ::spatial_stride]
    tgt = target[:, :, :, ::spatial_stride, ::spatial_stride]
    Hs, Ws = ens.shape[-2], ens.shape[-1]
    N = B * K * Hs * Ws

    # (N, M, C) and (N, C)
    flat_ens = ens.permute(0, 1, 4, 5, 2, 3).reshape(N, M, C).float()
    flat_tgt = tgt.permute(0, 1, 3, 4, 2).reshape(N, C).float()

    # Term 1: (1/M) * sum_m ||x_m - y||_C
    term1 = (flat_ens - flat_tgt.unsqueeze(1)).norm(dim=-1).mean(dim=-1)  # (N,)

    # Term 2: pairwise spread in batches to cap peak memory.
    CHUNK = 65536
    term2_parts: list[torch.Tensor] = []
    for i in range(0, N, CHUNK):
        pw = torch.cdist(flat_ens[i : i + CHUNK], flat_ens[i : i + CHUNK], p=2)
        if fair:
            mask = ~torch.eye(M, device=pw.device, dtype=torch.bool)
            term2_parts.append((pw * mask).sum(dim=(-2, -1)) / (2.0 * M * (M - 1)))
        else:
            term2_parts.append(pw.sum(dim=(-2, -1)) / (2.0 * M * M))
    term2 = torch.cat(term2_parts)  # (N,)

    es = (term1 - term2).reshape(B, K, Hs, Ws).mean(dim=(0, 2, 3))  # (K,)
    return es.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Plotting (matplotlib-gated; skip silently if unavailable so the lightweight
# smoke test still runs in headless CI).
# ---------------------------------------------------------------------------


def _import_matplotlib():
    try:
        import matplotlib  # noqa: F401

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def plot_metric_vs_lead(
    metric: np.ndarray,
    variables: Sequence[str],
    steps: Sequence[float],
    ylabel: str,
    title: str,
    out_path: str,
    hline_y: float | None = None,
    xlabel: str = "lead time (hours)",
) -> bool:
    """One line per channel over lead-time axis. Returns True if plotted."""
    plt = _import_matplotlib()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(10, 5))
    for ci, name in enumerate(variables):
        ax.plot(steps, metric[:, ci], marker="o", markersize=4, label=name)
    if hline_y is not None:
        ax.axhline(hline_y, color="k", linestyle="--", linewidth=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # Legend outside axes so it never overlaps data (70+ channels)
    ax.legend(
        fontsize=7, ncol=2, loc="upper left",
        bbox_to_anchor=(1.01, 1), borderaxespad=0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_rank_histograms(
    histograms: np.ndarray,
    variables: Sequence[str],
    out_path: str,
) -> bool:
    """Grid of rank histograms (one per channel) for calibration inspection."""
    plt = _import_matplotlib()
    if plt is None:
        return False
    C, bins = histograms.shape
    ncols = min(4, C)
    nrows = (C + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3 * ncols, 2.5 * nrows),
        constrained_layout=True, squeeze=False,
    )
    for ci, name in enumerate(variables):
        ax = axes[ci // ncols][ci % ncols]
        ax.bar(np.arange(bins), histograms[ci], color="#2266aa")
        expected = histograms[ci].sum() / bins
        ax.axhline(expected, color="k", linestyle="--", linewidth=0.7)
        ax.set_title(name, fontsize=8, pad=3)
        ax.set_xlabel("rank", fontsize=7)
        ax.set_ylabel("count", fontsize=7)
        ax.tick_params(labelsize=6)
    for j in range(C, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle("Rank histograms (uniform = well calibrated)", fontsize=11)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


_SPECTRA_VARS = ["z500", "q700", "t850", "u850", "v500", "t2m", "u10m", "v10m", "msl"]


def plot_power_spectra(
    k: np.ndarray,
    ens_spectra: np.ndarray,
    tgt_spectra: np.ndarray,
    variables: Sequence[str],
    lead_idx: int,
    out_path: str,
    grid_deg: float = 0.25,
    var_subset: Sequence[str] | None = None,
) -> bool:
    """Figure 3 e-j: log-log spectra with wavelength (km) on x-axis.

    Only plots a curated subset of variables (``var_subset``).  Defaults to
    ``_SPECTRA_VARS``; any names not present in ``variables`` are skipped.
    """
    plt = _import_matplotlib()
    if plt is None:
        return False

    # Build index list for the subset we want to show
    subset = list(var_subset) if var_subset is not None else _SPECTRA_VARS
    var_list = list(variables)
    pairs = [(name, var_list.index(name)) for name in subset if name in var_list]
    if not pairs:
        # Fall back: first 9 channels
        pairs = [(var_list[i], i) for i in range(min(9, len(var_list)))]

    n = len(pairs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    km_per_deg = 111.0
    grid_km = grid_deg * km_per_deg
    kk = k[1:]
    n_cells = round(40030 / grid_km)
    wavelength_km = n_cells * grid_km / kk

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows),
        constrained_layout=True, squeeze=False,
    )
    for plot_i, (name, ci) in enumerate(pairs):
        ax = axes[plot_i // ncols][plot_i % ncols]
        ax.loglog(wavelength_km, tgt_spectra[lead_idx, ci, 1:], label="truth", color="k")
        ax.loglog(wavelength_km, ens_spectra[lead_idx, ci, 1:], label="forecast", color="C0")
        ax.invert_xaxis()
        ax.set_title(name, fontsize=10, pad=4)
        ax.set_xlabel("Wavelength (km)", fontsize=9)
        ax.set_ylabel("Mean power", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(
        f"Power spectra at lead step {lead_idx + 1}  "
        f"({', '.join(p[0] for p in pairs)})",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def pooled_crps_per_lead(
    ensemble: torch.Tensor,
    target: torch.Tensor,
    pool_sizes: Sequence[int],
    pool_type: str = "avg",
) -> np.ndarray:
    """Pooled CRPS at multiple spatial scales (Figure 3 a-b of arXiv:2506.10772).

    Coarsens ensemble and target by pooling P×P grid-cell windows and then
    computes fair CRPS on the coarsened field. Tests calibration at scales
    larger than a single grid point.

    pool_sizes [4, 8, 16, 32] ≈ [120, 240, 480, 960] km at 0.25° resolution.

    Parameters
    ----------
    pool_type : {"avg", "max"}
        ``"avg"`` (Figure 3a) averages over the P×P window; ``"max"``
        (Figure 3b) takes the maximum — tests tail / extreme calibration.

    Shapes: ensemble (B, K, M, C, H, W), target (B, K, C, H, W).
    Returns numpy array of shape (len(pool_sizes), K, C).
    """
    import torch.nn.functional as F

    if pool_type not in ("avg", "max"):
        raise ValueError(f"pool_type must be 'avg' or 'max', got {pool_type!r}")
    _check_shapes(ensemble, target)
    B, K, M, C, H, W = ensemble.shape
    results = []
    for P in pool_sizes:
        ens_flat = ensemble.reshape(B * K * M, C, H, W)
        tgt_flat = target.reshape(B * K, C, H, W)
        if pool_type == "avg":
            ens_p = F.avg_pool2d(ens_flat, kernel_size=P, stride=P, ceil_mode=True)
            tgt_p = F.avg_pool2d(tgt_flat, kernel_size=P, stride=P, ceil_mode=True)
        else:
            ens_p = F.max_pool2d(ens_flat, kernel_size=P, stride=P, ceil_mode=True)
            tgt_p = F.max_pool2d(tgt_flat, kernel_size=P, stride=P, ceil_mode=True)
        Hp, Wp = ens_p.shape[-2], ens_p.shape[-1]
        ens_p = ens_p.reshape(B, K, M, C, Hp, Wp)
        tgt_p = tgt_p.reshape(B, K, C, Hp, Wp)
        results.append(crps_per_variable_per_lead(ens_p, tgt_p))  # (K, C)
    return np.stack(results, axis=0)  # (len(pool_sizes), K, C)


def plot_pooled_crps(
    pooled: np.ndarray,
    pool_sizes: Sequence[int],
    variables: Sequence[str],
    lead_hours: Sequence[float],
    out_path: str,
    title: str = "Pooled CRPS",
    grid_deg: float = 0.25,
) -> bool:
    """Figure 3 a-b: 2-D heatmap — pool size (km) × lead time, per variable.

    Matches the paper's heatmap layout: each panel covers one variable with
    lead time on the x-axis and spatial scale on the y-axis.
    """
    plt = _import_matplotlib()
    if plt is None:
        return False
    _P, _K, C = pooled.shape
    km_per_cell = grid_deg * 111.0  # 1° ≈ 111 km
    km_labels = [f"{int(ps * km_per_cell)}" for ps in pool_sizes]
    ncols = min(4, C)
    nrows = (C + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 2.5 * nrows),
        constrained_layout=True, squeeze=False,
    )
    for ci, name in enumerate(variables):
        ax = axes[ci // ncols][ci % ncols]
        data = pooled[:, :, ci]  # (P, K)
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=[lead_hours[0], lead_hours[-1], -0.5, len(pool_sizes) - 0.5],
            cmap="Blues",
        )
        ax.set_yticks(range(len(pool_sizes)))
        ax.set_yticklabels(km_labels, fontsize=7)
        ax.set_xlabel("lead time (h)", fontsize=8)
        ax.set_ylabel("scale (km)", fontsize=8)
        ax.set_title(name, fontsize=9, pad=3)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    for j in range(C, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def save_summary(metrics: dict[str, Any], out_path: str) -> None:
    """Persist a flat dict of numpy arrays + scalars as a single .npz file."""
    np.savez(out_path, **{k: np.asarray(v) for k, v in metrics.items()})


# ---------------------------------------------------------------------------
# Bad-seed detector — paper §6.2 (Discussion / Weaknesses)
# ---------------------------------------------------------------------------


def flag_bad_seeds(
    forecast_spectra: np.ndarray,
    truth_spectra: np.ndarray,
    tail_fraction: float = 0.2,
    threshold: float = 3.0,
) -> list[int]:
    """Flag seeds whose high-wavenumber power diverges from the truth.

    Paper §6.2 (Discussion): *"we found that a particular training seed
    produced a number of unstable rollouts, which was detected by
    examining the averaged spectra of the validation year forecasts. We
    removed this seed and retrained that particular model with a different
    seed."*

    The diagnostic operates on the azimuthal 1D power spectra computed by
    :func:`power_spectra_per_variable` (or any equivalent pipeline). For
    each seed we compare the mean power in the top ``tail_fraction`` of
    wavenumbers against the same tail of the ground-truth spectrum; a seed
    whose ratio exceeds ``threshold`` on **any** channel/lead pair has
    amplified high-frequency content and is treated as unstable.

    Parameters
    ----------
    forecast_spectra : np.ndarray
        Shape ``(S, K, C, B)`` — S seeds, K lead times, C channels,
        B wavenumber bins. Values are power (>= 0).
    truth_spectra : np.ndarray
        Shape ``(K, C, B)`` — ground-truth spectra shared by all seeds.
    tail_fraction : float, default 0.2
        Fraction of highest-wavenumber bins to average over. 0.2 = top 20%.
    threshold : float, default 3.0
        ``forecast_tail / truth_tail`` ratio above which a seed is flagged.
        Paper does not specify a number; 3x is a defensible starting
        default and is surfaced as a knob so callers can tune it.

    Returns
    -------
    list[int]
        Seed indices (into axis 0 of ``forecast_spectra``) that should be
        dropped before running deep-ensemble inference.
    """
    if forecast_spectra.ndim != 4:
        raise ValueError(
            f"forecast_spectra must have shape (S, K, C, B), got {forecast_spectra.shape}"
        )
    if truth_spectra.shape != forecast_spectra.shape[1:]:
        raise ValueError(
            f"truth_spectra shape {truth_spectra.shape} must match "
            f"forecast_spectra[1:] {forecast_spectra.shape[1:]}"
        )
    if not 0 < tail_fraction <= 1:
        raise ValueError(f"tail_fraction must be in (0, 1], got {tail_fraction}")
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")

    B = forecast_spectra.shape[-1]
    tail_start = max(0, B - max(1, int(round(B * tail_fraction))))

    # Mean power in the high-wavenumber tail — shape (S, K, C) / (K, C).
    fore_tail = forecast_spectra[..., tail_start:].mean(axis=-1)
    truth_tail = truth_spectra[..., tail_start:].mean(axis=-1)

    # Guard against a truth-tail of zero (degenerate variables like a
    # constant field) by clamping before the divide.
    safe_truth = np.where(truth_tail > 0, truth_tail, np.finfo(np.float32).eps)
    ratio = fore_tail / safe_truth  # (S, K, C)

    flagged = [
        s for s in range(forecast_spectra.shape[0]) if np.any(ratio[s] > threshold)
    ]
    return flagged
