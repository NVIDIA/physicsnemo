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

"""Tests for the ``Reader`` base class helpers shared by concrete readers.

``Reader._window_indices`` is the single place where coordinated subsampling
decides *which* rows to read; every subsampling-capable reader delegates to
it, so its contract (list-order key selection, ``None`` when off or when no
target key is present) is pinned down here against a minimal in-memory
reader rather than through any one storage backend.
"""

import numpy as np
import pytest
import torch

from physicsnemo.datapipes._indexing import _cyclic_block_indices
from physicsnemo.datapipes.readers.base import Reader


class _DictReader(Reader):
    """Minimal in-memory reader: each sample is a dict of CPU tensors."""

    def __init__(self, samples, **kwargs):
        super().__init__(**kwargs)
        self._samples = samples

    def _load_sample(self, index):
        return {k: v.clone() for k, v in self._samples[index].items()}

    def __len__(self):
        return len(self._samples)

    def _get_sample_metadata(self, index):
        return {"name": f"sample_{index}"}


def _reader(config=None, **kwargs):
    samples = [{"x": torch.arange(10.0)}, {"x": torch.arange(10.0) + 1}]
    return _DictReader(samples, coordinated_subsampling=config, **kwargs)


def _row_counts(table: dict[str, int], queried: list[str]):
    """Return a ``row_counts`` callable that records the keys it is asked about."""

    def row_counts(key):
        queried.append(key)
        return table.get(key)

    return row_counts


class TestWindowIndices:
    def test_disabled_returns_none_without_querying(self):
        queried: list[str] = []
        window, keys = _reader()._window_indices(_row_counts({"a": 10}, queried), None)
        assert window is None
        assert keys == set()
        assert queried == []

    def test_first_configured_key_defines_window(self):
        queried: list[str] = []
        reader = _reader({"n_points": 4, "target_keys": ["a", "b"]})
        window, keys = reader._window_indices(
            _row_counts({"a": 10, "b": 20}, queried),
            torch.Generator().manual_seed(2),
        )
        expected = _cyclic_block_indices(
            10, 4, generator=torch.Generator().manual_seed(2)
        ).numpy()
        assert isinstance(window, np.ndarray)
        np.testing.assert_array_equal(window, expected)
        assert keys == {"a", "b"}
        # The window comes from the first *present* key; later keys are not
        # consulted for the row count.
        assert queried == ["a"]

    def test_absent_leading_key_falls_through_in_list_order(self):
        queried: list[str] = []
        reader = _reader({"n_points": 3, "target_keys": ["missing", "present"]})
        window, keys = reader._window_indices(
            _row_counts({"present": 8}, queried), torch.Generator().manual_seed(0)
        )
        assert window is not None
        assert window.shape == (3,)
        assert window.max() < 8
        assert keys == {"missing", "present"}
        assert queried == ["missing", "present"]

    def test_no_target_present_returns_none_but_keeps_keys(self):
        reader = _reader({"n_points": 3, "target_keys": ["x", "y"]})
        window, keys = reader._window_indices(lambda key: None, None)
        assert window is None
        assert keys == {"x", "y"}

    def test_generator_none_draws_valid_window(self):
        reader = _reader({"n_points": 5, "target_keys": ["a"]})
        window, _ = reader._window_indices(lambda key: 12, None)
        assert window.shape == (5,)
        assert window.min() >= 0
        assert window.max() < 12

    def test_seeded_reader_is_reproducible_per_index(self):
        reader = _reader({"n_points": 4, "target_keys": ["a"]})
        reader.set_generator(torch.Generator().manual_seed(7))
        first, _ = reader._window_indices(lambda key: 100, reader._index_generator(1))
        second, _ = reader._window_indices(lambda key: 100, reader._index_generator(1))
        np.testing.assert_array_equal(first, second)
        other, _ = reader._window_indices(lambda key: 100, reader._index_generator(0))
        # Different indices derive different generators (overwhelmingly).
        assert not np.array_equal(first, other)


class TestReaderMetadata:
    @pytest.mark.parametrize("include_index", [True, False])
    def test_include_index_in_metadata(self, include_index):
        _, meta = _reader(include_index_in_metadata=include_index)[1]
        assert meta["name"] == "sample_1"
        assert ("index" in meta) is include_index
        if include_index:
            assert meta["index"] == 1

    def test_negative_index_resolves_and_out_of_range_raises(self):
        reader = _reader()
        data, meta = reader[-1]
        assert meta["index"] == 1
        torch.testing.assert_close(data["x"], torch.arange(10.0) + 1)
        with pytest.raises(IndexError):
            reader[2]
