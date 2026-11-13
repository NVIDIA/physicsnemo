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

"""ARCO/ERA5 dataset for FGN, following Appendix A.1 of arXiv:2506.10772v1.

Wraps :class:`earth2studio.data.ARCO` so data fetching, caching, and the
compact-name lexicon live in earth2studio. This module only turns a sample
index into a `(history, target, background)` triple at a fixed 6-hour
stride, applies the SST land-NaN imputation described in the paper, and
computes clock / invariant features locally.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import torch

from .dataset import FGNDataset

# Paper Table A.1 atmospheric schema
PAPER_ATMOS_VARS: tuple[str, ...] = ("z", "q", "t", "u", "v", "w")
PAPER_LEVELS: tuple[int, ...] = (
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
)
PAPER_SURFACE_IN_OUT: tuple[str, ...] = ("t2m", "u10m", "v10m", "msl", "sst")

# 6 atmospheric * 13 levels + 5 surface input/predicted = 83 state channels
# `tp` is predicted-only (6-h accumulation) and is not currently handled here;
# it belongs in the target-only output variables.
DEFAULT_STATE: tuple[str, ...] = tuple(
    [f"{v}{lvl}" for v in PAPER_ATMOS_VARS for lvl in PAPER_LEVELS]
    + list(PAPER_SURFACE_IN_OUT)
)

# Static fields available from ARCO directly via the compact-name lexicon.
ARCO_STATIC_VARS: frozenset[str] = frozenset({"z", "lsm"})
# Computed locally from the grid, not from ARCO.
LOCAL_INVARIANTS: frozenset[str] = frozenset({"lat", "lon"})

# Clock features are computed locally from the target timestamp.
CLOCK_CHANNELS: tuple[str, ...] = (
    "local_time_sin",
    "local_time_cos",
    "year_progress_sin",
    "year_progress_cos",
)

ARCO_LAT = np.linspace(90, -90, 721, dtype=np.float32)
ARCO_LON = np.linspace(0, 359.75, 1440, dtype=np.float32)


def _parse_date(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


class ArcoFGNDataset(FGNDataset):
    """ERA5 training dataset for FGN, served through earth2studio ARCO.

    Each sample returns a history of ``history_frames`` state tensors at a
    ``step_hours`` stride and the next state tensor as the target. The
    default state variable list matches Table A.1 of arXiv:2506.10772v1.

    Parameters expected on ``params`` (Hydra DictConfig or similar):

    - ``state_variables`` (list[str], optional): compact ARCO names; defaults
      to the 83-channel Table A.1 list.
    - ``invariant_variables`` (list[str]): subset of ``{"z", "lsm", "lat",
      "lon"}``; defaults to ``["z", "lsm"]``.
    - ``train_start`` / ``train_end`` / ``val_start`` / ``val_end`` (str):
      ISO-8601 dates defining the split.
    - ``step_hours`` (int, default 6): temporal stride between frames.
    - ``history_frames`` (int, default 2): number of prior frames used as
      input. Paper uses 2.
    - ``spatial_stride`` (int, default 1): sub-sample the ARCO 721x1440 grid
      for cheaper dev runs.
    - ``static_date`` (str, default ``"2016-01-01"``): date used to fetch
      the truly-static ``z``/``lsm`` fields once.
    - ``arco_cache`` (bool, default True): earth2studio ARCO local cache.
    - ``tp_accumulation_hours`` (int or None, default None): when set, the
      state variable named ``tp{tp_accumulation_hours}`` (e.g. ``tp06``) is
      treated as a paper §3 predicted-only accumulated precipitation
      channel: history values are forced to zero (matching HRES-fc0
      initialisation and the earth2studio ``gencast_mini`` convention for
      ``tp12``) and target values are computed as the sum of
      ``tp_accumulation_hours`` hourly ARCO ``tp`` values leading up to
      each target timestamp. Requires ``tp{N}`` to appear in
      ``state_variables`` so the channel exists in both input and output
      tensors.
    """

    def __init__(self, params: Any, train: bool) -> None:
        def _get(name: str, default: Any) -> Any:
            return (
                getattr(params, name, default)
                if hasattr(params, name)
                else (
                    params[name]
                    if isinstance(params, dict) and name in params
                    else default
                )
            )

        state = _get("state_variables", None)
        self._state_variables: list[str] = list(state) if state else list(DEFAULT_STATE)
        invariants = _get("invariant_variables", None)
        self._invariant_variables: list[str] = (
            list(invariants) if invariants is not None else ["z", "lsm"]
        )
        for v in self._invariant_variables:
            if v not in ARCO_STATIC_VARS and v not in LOCAL_INVARIANTS:
                raise ValueError(
                    f"invariant_variables entry {v!r} is not supported; "
                    f"expected one of {sorted(ARCO_STATIC_VARS | LOCAL_INVARIANTS)}"
                )

        self.step_hours = int(_get("step_hours", 6))
        self.history_frames = int(_get("history_frames", 2))
        if self.history_frames < 1:
            raise ValueError("history_frames must be >= 1")
        self.future_frames = int(_get("future_frames", 1))
        if self.future_frames < 1:
            raise ValueError("future_frames must be >= 1")

        if train:
            start = _get("train_start", "1979-01-01")
            end = _get("train_end", "2018-01-15")
        else:
            start = _get("val_start", "2018-01-15")
            end = _get("val_end", "2019-01-01")
        self.start: datetime = _parse_date(start)
        self.end: datetime = _parse_date(end)

        self.stride = int(_get("spatial_stride", 1))
        if self.stride < 1:
            raise ValueError(f"spatial_stride must be >= 1, got {self.stride}")
        self.height = len(ARCO_LAT[:: self.stride])
        self.width = len(ARCO_LON[:: self.stride])

        self.static_date = _parse_date(_get("static_date", "2016-01-01"))
        self.arco_cache = bool(_get("arco_cache", True))

        # Paper §3 "Total precipitation" handling. When tp_accumulation_hours
        # is set, the state variable literally named ``tp<N>`` (e.g.
        # ``tp06`` for the paper's 6-hour accumulation) is:
        #   - zeroed in history (predicted-only — matches HRES-fc0 init
        #     and earth2studio gencast_mini's tp12 convention)
        #   - computed in target by summing N hourly ARCO ``tp`` values
        #     leading up to each target timestamp.
        self.tp_accumulation_hours: int | None = None
        self._tp_channel_idx: int | None = None
        self._tp_state_name: str | None = None
        tp_hours = _get("tp_accumulation_hours", None)
        if tp_hours is not None:
            self.tp_accumulation_hours = int(tp_hours)
            if self.tp_accumulation_hours < 1:
                raise ValueError(
                    f"tp_accumulation_hours must be >= 1, got {self.tp_accumulation_hours}"
                )
            tp_name = f"tp{self.tp_accumulation_hours:02d}"
            if tp_name not in self._state_variables:
                raise ValueError(
                    f"tp_accumulation_hours={self.tp_accumulation_hours} requires "
                    f"{tp_name!r} in state_variables; got {self._state_variables}"
                )
            self._tp_state_name = tp_name
            self._tp_channel_idx = self._state_variables.index(tp_name)

        # Optional per-channel z-score stats. File layout: an .npz with
        # arrays ``mean`` and ``std``, each of shape ``(len(state_variables),)``
        # in the same order as ``state_variables``. Mirrors the StormCast
        # convention (means.npy / stds.npy) but packaged as one file to
        # avoid order-mismatch bugs when variable lists change.
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        stats_path = _get("stats_path", None)
        if stats_path:
            self._load_stats(str(stats_path))

        # Sample count: for each target index i,
        #   last_target_time(i) = start + (i + history_frames + future_frames - 1) * step
        # and we need last_target_time <= end.
        total_hours = max(0, int((self.end - self.start).total_seconds() // 3600))
        total_steps = total_hours // self.step_hours
        window = self.history_frames + self.future_frames - 1
        self.num_samples = max(0, total_steps - window)
        if self.num_samples <= 0:
            raise ValueError(
                "Date range is shorter than (history_frames + future_frames) * step_hours; "
                f"start={self.start}, end={self.end}, step_hours={self.step_hours}"
            )

        # Lazy, per-worker ARCO client
        self._arco = None
        self._invariants_cache: np.ndarray | None = None

    # --- FGNDataset interface ------------------------------------------------

    def state_channels(self) -> list[str]:
        return list(self._state_variables)

    def background_channels(self) -> list[str]:
        return list(CLOCK_CHANNELS)

    def image_shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # ParallelHelper.sharded_dataloader yields numpy.int64 indices from a
        # numpy index array; Python 3.12's timedelta rejects np.int64.
        idx = int(idx)
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(idx)

        first_target_time = self._target_time(idx)
        total_frames = self.history_frames + self.future_frames
        # Window start is the time of history[0].
        window_start = first_target_time - timedelta(
            hours=self.history_frames * self.step_hours
        )
        times = [
            window_start + timedelta(hours=k * self.step_hours)
            for k in range(total_frames)
        ]

        # Fetch real state variables from ARCO. The tp{N} placeholder name
        # (e.g. "tp06") is NOT in the ARCOLexicon — that's a derived
        # accumulation the paper §3 defines on top of the hourly ``tp``
        # variable — so we exclude it from the bulk fetch and fill its
        # slot from _fetch_tp_accumulation below.
        if self._tp_channel_idx is not None:
            ci = self._tp_channel_idx
            arco_vars = [v for i, v in enumerate(self._state_variables) if i != ci]
        else:
            ci = None
            arco_vars = list(self._state_variables)

        da = self._ensure_arco()(time=times, variable=arco_vars)
        fetched = np.asarray(da.values, dtype=np.float32)  # (T, V-1 or V, 721, 1440)
        if self.stride > 1:
            fetched = fetched[..., :: self.stride, :: self.stride]

        # Re-embed the fetched channels into the full (T, V, H, W) layout
        # with a zero tp{N} slot we'll overwrite immediately after. This
        # keeps channel indexing in lockstep with ``self._state_variables``.
        T = fetched.shape[0]
        V = len(self._state_variables)
        arr = np.zeros((T, V, self.height, self.width), dtype=np.float32)
        if ci is None:
            arr[:] = fetched
        else:
            arr[:, :ci] = fetched[:, :ci]
            arr[:, ci + 1 :] = fetched[:, ci:]

        self._impute_sst_nans_(arr)

        # Paper §3: replace the tp{N} channel with N-hour accumulation for
        # every target frame, keep history zeroed (predicted-only — "FGN
        # is trained to only output tp, not taking it as input"). Done
        # here before normalization so z-score stats apply to the
        # accumulated values.
        if ci is not None:
            tp_acc = self._fetch_tp_accumulation(times)  # (T, H, W)
            arr[self.history_frames :, ci, :, :] = tp_acc[self.history_frames :]

        if self._mean is not None:
            # Broadcast (V,) stats over (T, V, H, W).
            arr = (arr - self._mean[None, :, None, None]) / self._std[
                None, :, None, None
            ]

        history = arr[: self.history_frames]
        target = arr[self.history_frames :]  # (future_frames, V, H, W)
        # Clock features are computed for the first target step only for now;
        # if downstream code wants per-step clocks, extend to (future_frames, 4, H, W).
        background = self._clock_features(first_target_time)

        return {
            "history": torch.from_numpy(history),
            "target": torch.from_numpy(target),
            "background": torch.from_numpy(background),
            "init_time": first_target_time.isoformat(),
        }

    def output_only_channels(self) -> list[int]:
        if self._tp_channel_idx is None:
            return []
        return [self._tp_channel_idx]

    def get_invariants(self) -> np.ndarray | None:
        if not self._invariant_variables:
            return None
        if self._invariants_cache is not None:
            return self._invariants_cache

        pieces: list[np.ndarray] = []
        arco_vars = [v for v in self._invariant_variables if v in ARCO_STATIC_VARS]
        if arco_vars:
            da = self._ensure_arco()(time=[self.static_date], variable=arco_vars)
            raw = np.asarray(da.values, dtype=np.float32)[0]  # (V, 721, 1440)
            if self.stride > 1:
                raw = raw[..., :: self.stride, :: self.stride]
            pieces.extend(raw[i] for i in range(raw.shape[0]))

        if "lat" in self._invariant_variables:
            pieces.append(
                np.broadcast_to(
                    ARCO_LAT[:: self.stride, None], (self.height, self.width)
                )
                .astype(np.float32)
                .copy()
            )
        if "lon" in self._invariant_variables:
            pieces.append(
                np.broadcast_to(
                    ARCO_LON[None, :: self.stride], (self.height, self.width)
                )
                .astype(np.float32)
                .copy()
            )

        # Reorder to match `self._invariant_variables` order.
        by_name: dict[str, np.ndarray] = {}
        cursor = 0
        for v in arco_vars:
            by_name[v] = pieces[cursor]
            cursor += 1
        if "lat" in self._invariant_variables:
            by_name["lat"] = pieces[cursor]
            cursor += 1
        if "lon" in self._invariant_variables:
            by_name["lon"] = pieces[cursor]
            cursor += 1

        self._invariants_cache = np.stack(
            [by_name[v] for v in self._invariant_variables], axis=0
        ).astype(np.float32)
        return self._invariants_cache

    def normalize_state(
        self, x: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if self._mean is None:
            return x
        mean, std = self._broadcast_stats_for(x)
        return (x - mean) / std

    def denormalize_state(
        self, x: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if self._mean is None:
            return x
        mean, std = self._broadcast_stats_for(x)
        return x * std + mean

    # --- Internals -----------------------------------------------------------

    def _fetch_tp_accumulation(self, frame_times: list[datetime]) -> np.ndarray:
        """Sum the N hourly ARCO ``tp`` values preceding each frame time.

        ARCO stores ``total_precipitation`` as the hourly accumulation
        during ``[t-1h, t]``, matching ECMWF's ERA5 convention. So the
        paper's N-hour accumulation ending at T equals
        ``sum(tp(T-N+1), tp(T-N+2), ..., tp(T))`` — N hourly values. We
        fetch all distinct hourly stamps required by any frame in a single
        earth2studio call to minimise GCS round-trips.
        """
        if self.tp_accumulation_hours is None:
            raise RuntimeError(
                "_fetch_tp_accumulation called without tp_accumulation_hours set"
            )
        N = self.tp_accumulation_hours

        # Union of hours we need across all frames, sorted.
        hourly_set: set[datetime] = set()
        per_frame_hours: list[list[datetime]] = []
        for t in frame_times:
            hours = [t - timedelta(hours=N - 1 - j) for j in range(N)]
            per_frame_hours.append(hours)
            hourly_set.update(hours)
        unique_hours = sorted(hourly_set)
        hour_to_idx = {t: i for i, t in enumerate(unique_hours)}

        da = self._ensure_arco()(time=unique_hours, variable=["tp"])
        hourly = np.asarray(da.values, dtype=np.float32)  # (U, 1, 721, 1440)
        hourly = hourly[:, 0]  # (U, 721, 1440)
        if self.stride > 1:
            hourly = hourly[:, :: self.stride, :: self.stride]

        acc = np.zeros((len(frame_times), self.height, self.width), dtype=np.float32)
        for k, hours_k in enumerate(per_frame_hours):
            acc[k] = sum(hourly[hour_to_idx[h]] for h in hours_k)
        return acc

    def _load_stats(self, stats_path: str) -> None:
        from pathlib import Path as _Path

        path = _Path(stats_path)
        if not path.exists():
            raise FileNotFoundError(f"stats_path does not exist: {path}")
        data = np.load(path)
        if "mean" not in data or "std" not in data:
            raise KeyError(
                f"{path} must contain arrays 'mean' and 'std'; got {list(data.files)}"
            )
        mean = np.asarray(data["mean"], dtype=np.float32)
        std = np.asarray(data["std"], dtype=np.float32)
        expected = (len(self._state_variables),)
        if mean.shape != expected or std.shape != expected:
            raise ValueError(
                f"stats mean/std must have shape {expected} matching "
                f"state_variables; got mean={mean.shape}, std={std.shape}"
            )
        if np.any(std == 0):
            raise ValueError("stats std contains zeros; cannot z-score normalize")
        self._mean = mean
        self._std = std

    def _broadcast_stats_for(self, x: np.ndarray | torch.Tensor) -> tuple[Any, Any]:
        """Reshape `(V,)` stats to broadcast along the channel axis of ``x``.

        Supports ``x`` of shape ``(V, H, W)``, ``(T, V, H, W)``, or
        ``(B, T, V, H, W)`` — channel axis is the third-from-last.
        """
        if x.ndim == 3:
            shape = (-1, 1, 1)
        elif x.ndim == 4:
            shape = (1, -1, 1, 1)
        elif x.ndim == 5:
            shape = (1, 1, -1, 1, 1)
        else:
            raise ValueError(f"unsupported state tensor ndim {x.ndim}")
        if isinstance(x, torch.Tensor):
            mean = torch.as_tensor(self._mean, dtype=x.dtype, device=x.device).reshape(
                shape
            )
            std = torch.as_tensor(self._std, dtype=x.dtype, device=x.device).reshape(
                shape
            )
        else:
            mean = self._mean.reshape(shape)
            std = self._std.reshape(shape)
        return mean, std

    def _ensure_arco(self):
        if self._arco is None:
            from earth2studio.data import ARCO

            self._arco = ARCO(cache=self.arco_cache, verbose=False)
        return self._arco

    def _target_time(self, idx: int) -> datetime:
        return self.start + timedelta(
            hours=(idx + self.history_frames) * self.step_hours
        )

    @staticmethod
    def _impute_sst_nans_(arr: np.ndarray) -> None:
        """Replace NaNs in SST with the global min SST seen in the batch.

        Paper A.1.1: ERA5 represents land in SST with NaNs; we impute with a
        global minimum to keep the tensor dense. This happens over the whole
        fetched window to avoid leaking land-mask information.
        """
        # Implicitly finds any channel whose values contain NaN; SST is the
        # only Table A.1 variable expected to have them.
        if not np.isnan(arr).any():
            return
        for c in range(arr.shape[1]):
            chan = arr[:, c, :, :]
            mask = np.isnan(chan)
            if not mask.any():
                continue
            finite_min = float(np.nanmin(chan))
            chan[mask] = finite_min
            arr[:, c, :, :] = chan

    def _clock_features(self, t: datetime) -> np.ndarray:
        # Year progress in [0, 1)
        year_start = datetime(t.year, 1, 1)
        year_end = datetime(t.year + 1, 1, 1)
        yp = (t - year_start).total_seconds() / (year_end - year_start).total_seconds()
        yp_sin = float(np.sin(2 * np.pi * yp))
        yp_cos = float(np.cos(2 * np.pi * yp))

        utc_hours = t.hour + t.minute / 60.0 + t.second / 3600.0
        lon = ARCO_LON[:: self.stride]
        local_hours = (utc_hours + lon / 15.0) % 24.0
        local_frac = local_hours / 24.0
        lt_sin_row = np.sin(2 * np.pi * local_frac).astype(np.float32)
        lt_cos_row = np.cos(2 * np.pi * local_frac).astype(np.float32)

        lt_sin = np.broadcast_to(lt_sin_row[None, :], (self.height, self.width)).copy()
        lt_cos = np.broadcast_to(lt_cos_row[None, :], (self.height, self.width)).copy()
        yp_sin_field = np.full((self.height, self.width), yp_sin, dtype=np.float32)
        yp_cos_field = np.full((self.height, self.width), yp_cos, dtype=np.float32)

        return np.stack([lt_sin, lt_cos, yp_sin_field, yp_cos_field], axis=0)
