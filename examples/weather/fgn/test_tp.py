# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the paper §3 / Table A.1 predicted-only ``tp06`` channel.

The accumulation helper ``_fetch_tp_accumulation`` is tested with a
monkey-patched ARCO client so we don't touch GCS — the logic we care
about is: (a) union of required hourly stamps, (b) correct per-frame
sum of the N preceding hourly values, (c) zeroing of the history tp
channel vs correct target accumulation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from datasets.arco import ArcoFGNDataset


@pytest.fixture
def cfg() -> SimpleNamespace:
    return SimpleNamespace(
        state_variables=["u10m", "v10m", "t2m", "msl", "tp06"],
        invariant_variables=[],
        step_hours=6,
        history_frames=2,
        future_frames=2,
        train_start="2016-01-01",
        train_end="2016-01-15",
        val_start="2018-01-01",
        val_end="2018-02-01",
        spatial_stride=40,
        static_date="2016-01-01",
        arco_cache=False,
        stats_path=None,
        tp_accumulation_hours=6,
    )


def test_tp_config_enforces_channel_presence():
    bad = SimpleNamespace(
        state_variables=["u10m", "v10m"],  # no tp06
        invariant_variables=[],
        step_hours=6,
        history_frames=2,
        future_frames=1,
        train_start="2016-01-01",
        train_end="2016-01-15",
        val_start="2018-01-01",
        val_end="2018-02-01",
        spatial_stride=40,
        tp_accumulation_hours=6,
    )
    with pytest.raises(ValueError, match="tp06"):
        ArcoFGNDataset(bad, train=True)


def test_tp_channel_index_and_output_only(cfg):
    ds = ArcoFGNDataset(cfg, train=True)
    assert ds.state_channels() == ["u10m", "v10m", "t2m", "msl", "tp06"]
    assert ds._tp_channel_idx == 4
    assert ds.output_only_channels() == [4]
    assert ds.tp_accumulation_hours == 6


def test_tp_no_op_when_disabled():
    c = SimpleNamespace(
        state_variables=["u10m", "v10m"],
        invariant_variables=[],
        step_hours=6,
        history_frames=2,
        future_frames=1,
        train_start="2016-01-01",
        train_end="2016-01-15",
        val_start="2018-01-01",
        val_end="2018-02-01",
        spatial_stride=40,
        tp_accumulation_hours=None,
    )
    ds = ArcoFGNDataset(c, train=True)
    assert ds.output_only_channels() == []


class _FakeARCO:
    """Stand-in for ``earth2studio.data.ARCO`` that returns a deterministic
    value derived from the request timestamp so accumulation sums are
    easy to hand-verify. Only used for the accumulation helper test.
    """

    def __init__(self, reference: datetime):
        self.reference = reference

    def __call__(self, *, time, variable):
        import xarray as xr

        assert variable == ["tp"], variable
        data = np.zeros((len(time), 1, 721, 1440), dtype=np.float32)
        for i, t in enumerate(time):
            # Value = hours since reference (so a 6-hour accumulation ending
            # at T yields sum of the 6 integer offsets preceding T).
            offset = (t - self.reference).total_seconds() / 3600.0
            data[i, 0, :, :] = float(offset)
        return xr.DataArray(
            data,
            dims=("time", "variable", "lat", "lon"),
        )


def test_tp_accumulation_sums_six_hourly_values(cfg):
    ds = ArcoFGNDataset(cfg, train=True)
    reference = datetime(2016, 1, 1, 0, 0)
    ds._arco = _FakeARCO(reference)

    frame_times = [
        reference + timedelta(hours=24),
        reference + timedelta(hours=30),
    ]
    acc = ds._fetch_tp_accumulation(frame_times)
    assert acc.shape == (2, ds.height, ds.width)
    # For T = reference + 24h, 6-hour accumulation sums offsets
    # {19, 20, 21, 22, 23, 24} = 129.
    np.testing.assert_allclose(acc[0], np.full((ds.height, ds.width), 129.0))
    # For T = reference + 30h, sums {25, 26, 27, 28, 29, 30} = 165.
    np.testing.assert_allclose(acc[1], np.full((ds.height, ds.width), 165.0))


def test_tp_getitem_zeros_history_and_accumulates_target(cfg, monkeypatch):
    ds = ArcoFGNDataset(cfg, train=True)

    reference = datetime(2016, 1, 1, 0, 0)
    ds._arco = _FakeARCO(reference)

    # Stub the state-variable ARCO fetch so __getitem__ doesn't hit the
    # network. We return uniform values equal to the channel index to
    # make shape + indexing easy to verify.
    import xarray as xr

    original_call = ds._arco.__call__

    def patched_call(*, time, variable):
        if variable == ["tp"]:
            return original_call(time=time, variable=variable)
        data = np.zeros((len(time), len(variable), 721, 1440), dtype=np.float32)
        for j in range(len(variable)):
            data[:, j] = float(j) + 1.0  # non-zero placeholder
        return xr.DataArray(data, dims=("time", "variable", "lat", "lon"))

    monkeypatch.setattr(
        ds, "_arco", type("Stub", (), {"__call__": staticmethod(patched_call)})()
    )
    # The above monkeypatch replaces `_arco` with an object whose
    # `__call__` is our stub; `_ensure_arco()` returns `self._arco`.

    sample = ds[0]
    history = sample["history"].numpy()
    target = sample["target"].numpy()

    assert history.shape == (2, 5, ds.height, ds.width)
    assert target.shape == (2, 5, ds.height, ds.width)

    # Paper §3: tp06 channel (index 4) is zero throughout history.
    np.testing.assert_array_equal(
        history[:, 4], np.zeros((2, ds.height, ds.width), dtype=np.float32)
    )
    # Target tp06 is the accumulation (non-zero by construction).
    assert np.all(target[:, 4] > 0.0)

    # Non-tp channels in both history and target are unchanged (= j+1 from stub).
    for j in range(4):
        np.testing.assert_allclose(history[:, j], float(j) + 1.0)
        np.testing.assert_allclose(target[:, j], float(j) + 1.0)
