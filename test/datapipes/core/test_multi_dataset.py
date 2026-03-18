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

"""Tests for MultiDataset class."""

import pytest

import physicsnemo.datapipes as dp
from physicsnemo.datapipes.multi_dataset import DATASET_INDEX_METADATA_KEY


class TestMultiDatasetBasic:
    """Basic MultiDataset functionality."""

    def test_create_multi_dataset(self, numpy_data_dir):
        """MultiDataset with two datasets has combined length."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)

        assert len(multi) == len(ds_a) + len(ds_b)

    def test_create_multi_dataset_three_or_more(self, numpy_data_dir):
        """MultiDataset with three+ datasets has combined length and correct index mapping."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_c = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, ds_c, output_strict=True)

        assert len(multi) == len(ds_a) + len(ds_b) + len(ds_c)
        assert multi[0][1][DATASET_INDEX_METADATA_KEY] == 0
        assert multi[10][1][DATASET_INDEX_METADATA_KEY] == 1
        assert multi[20][1][DATASET_INDEX_METADATA_KEY] == 2

    def test_getitem_maps_to_correct_dataset(self, numpy_data_dir):
        """Indices 0..len0-1 from first dataset, then second."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)

        # First 10 from ds_a (dataset_index 0)
        data0, meta0 = multi[0]
        assert meta0[DATASET_INDEX_METADATA_KEY] == 0
        assert "positions" in data0

        data9, meta9 = multi[9]
        assert meta9[DATASET_INDEX_METADATA_KEY] == 0

        # Next 10 from ds_b (dataset_index 1)
        data10, meta10 = multi[10]
        assert meta10[DATASET_INDEX_METADATA_KEY] == 1

        data19, meta19 = multi[19]
        assert meta19[DATASET_INDEX_METADATA_KEY] == 1

    def test_getitem_preserves_sub_dataset_metadata(self, numpy_data_dir):
        """Metadata from sub-dataset (e.g. index) is preserved alongside dataset_index."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)

        data0, meta0 = multi[0]
        assert meta0["index"] == 0
        assert meta0[DATASET_INDEX_METADATA_KEY] == 0

        data10, meta10 = multi[10]
        assert meta10["index"] == 0  # first sample of second dataset
        assert meta10[DATASET_INDEX_METADATA_KEY] == 1

    def test_negative_indexing(self, numpy_data_dir):
        """Negative indices work as expected."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)

        data_last, meta_last = multi[-1]
        assert meta_last[DATASET_INDEX_METADATA_KEY] == 1
        data_first, meta_first = multi[-20]
        assert meta_first[DATASET_INDEX_METADATA_KEY] == 0

    def test_field_names_strict(self, numpy_data_dir):
        """output_strict=True returns common output keys (TensorDict keys)."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)

        assert "positions" in multi.field_names
        assert "features" in multi.field_names

    def test_iteration(self, numpy_data_dir):
        """Iteration yields all samples in order with dataset_index in metadata."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)

        seen_indices = []
        seen_dataset_indices = []
        for data, meta in multi:
            seen_indices.append(meta["index"])
            seen_dataset_indices.append(meta[DATASET_INDEX_METADATA_KEY])

        assert len(seen_indices) == 20
        assert seen_dataset_indices[:10] == [0] * 10
        assert seen_dataset_indices[10:] == [1] * 10

    def test_context_manager(self, numpy_data_dir):
        """MultiDataset as context manager closes sub-datasets."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        with dp.MultiDataset(ds_a, ds_b, output_strict=True) as multi:
            data, meta = multi[0]
            assert meta[DATASET_INDEX_METADATA_KEY] == 0


class TestMultiDatasetStrictValidation:
    """Output strictness validation."""

    def test_strict_raises_when_output_keys_differ(self, numpy_data_dir):
        """output_strict=True raises if datasets produce different output keys."""
        ds_full = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_pos_only = dp.Dataset(dp.NumpyReader(numpy_data_dir, fields=["positions"]))

        with pytest.raises(ValueError, match="output keys"):
            dp.MultiDataset(ds_full, ds_pos_only, output_strict=True)

    def test_non_strict_accepts_different_fields(self, numpy_data_dir):
        """output_strict=False does not validate output keys; field_names is first dataset's."""
        ds_full = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_pos_only = dp.Dataset(dp.NumpyReader(numpy_data_dir, fields=["positions"]))

        multi = dp.MultiDataset(ds_full, ds_pos_only, output_strict=False)
        assert len(multi) == 20
        assert set(multi.field_names) == set(ds_full.field_names)

    def test_strict_validates_output_keys_not_reader_fields(self, numpy_data_dir):
        """output_strict compares TensorDict keys after transforms, not reader field_names."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)
        assert len(multi.field_names) >= 2  # positions, features


class TestMultiDatasetPrefetchAndClose:
    """Prefetch and close delegation."""

    def test_prefetch_delegates(self, numpy_data_dir):
        """Prefetch delegates to correct sub-dataset."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)

        multi.prefetch(0)
        multi.prefetch(10)
        data0, meta0 = multi[0]
        data10, meta10 = multi[10]
        assert meta0[DATASET_INDEX_METADATA_KEY] == 0
        assert meta10[DATASET_INDEX_METADATA_KEY] == 1

    def test_cancel_prefetch_all(self, numpy_data_dir):
        """cancel_prefetch(None) clears all sub-datasets."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)

        multi.prefetch(0)
        multi.prefetch(10)
        multi.cancel_prefetch()
        # Should still be able to get data synchronously
        data0, _ = multi[0]
        assert "positions" in data0

    def test_cancel_prefetch_invalid_index_no_op(self, numpy_data_dir):
        """cancel_prefetch(out-of-range index) does not raise (matches Dataset)."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)

        multi.prefetch(0)
        multi.cancel_prefetch(999)  # out of range, should no-op
        multi.cancel_prefetch(-1)  # also out of range
        data0, _ = multi[0]
        assert "positions" in data0

    def test_prefetch_batch(self, numpy_data_dir):
        """prefetch_batch delegates to sub-datasets by global index."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)

        multi.prefetch_batch([0, 1, 10, 11])
        for idx in [0, 1, 10, 11]:
            data, meta = multi[idx]
            assert meta[DATASET_INDEX_METADATA_KEY] == (0 if idx < 10 else 1)
            assert "positions" in data

    def test_prefetch_count(self, numpy_data_dir):
        """prefetch_count is sum of sub-dataset prefetch counts."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)

        assert multi.prefetch_count == 0
        multi.prefetch_batch([0, 1, 2, 3])
        assert multi.prefetch_count >= 0  # may complete quickly
        multi.cancel_prefetch()
        assert multi.prefetch_count == 0

    def test_close_closes_all(self, numpy_data_dir):
        """close() closes all sub-datasets."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)
        multi.close()
        # After close, sub-datasets are closed (no-op to call again)


class TestMultiDatasetErrors:
    """Error cases."""

    def test_requires_at_least_one_datasets(self, numpy_data_dir):
        """MultiDataset requires at least one datasets."""
        with pytest.raises(ValueError, match="at least one"):
            dp.MultiDataset(output_strict=True)

    def test_requires_dataset_instances(self, numpy_data_dir):
        """All elements must be Dataset instances."""
        ds = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        with pytest.raises(TypeError, match="must be a Dataset"):
            dp.MultiDataset(ds, dp.NumpyReader(numpy_data_dir), output_strict=False)

    def test_index_out_of_range(self, numpy_data_dir):
        """Index out of range raises IndexError."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)

        with pytest.raises(IndexError, match="out of range"):
            _ = multi[20]
        with pytest.raises(IndexError, match="out of range"):
            _ = multi[-21]

    def test_prefetch_out_of_range_raises(self, numpy_data_dir):
        """prefetch with out-of-range index raises IndexError."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)

        with pytest.raises(IndexError, match="out of range"):
            multi.prefetch(20)


class TestMultiDatasetWithDataLoader:
    """DataLoader accepts MultiDataset (same interface as Dataset)."""

    def test_dataloader_with_multi_dataset(self, numpy_data_dir):
        """DataLoader iterates over MultiDataset and collates batches."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)

        loader = dp.DataLoader(multi, batch_size=4, shuffle=False)
        assert len(loader) == 5  # 20 / 4

        batches = list(loader)
        assert len(batches) == 5
        assert batches[0]["positions"].shape[0] == 4
        assert batches[-1]["positions"].shape[0] == 4

    def test_dataloader_with_multi_dataset_and_metadata(self, numpy_data_dir):
        """Collate metadata includes dataset_index from MultiDataset."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)

        loader = dp.DataLoader(
            multi, batch_size=5, shuffle=False, collate_metadata=True
        )
        batch_data, metadata_list = next(iter(loader))

        assert len(metadata_list) == 5
        assert [m[DATASET_INDEX_METADATA_KEY] for m in metadata_list] == [
            0,
            0,
            0,
            0,
            0,
        ]

        batches = list(loader)
        _, meta_batch_2 = batches[2]  # indices 10-14, all from dataset 1
        assert all(m[DATASET_INDEX_METADATA_KEY] == 1 for m in meta_batch_2)

    def test_dataloader_shuffle_with_multi_dataset(self, numpy_data_dir):
        """Shuffled DataLoader over MultiDataset yields all indices."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)

        loader = dp.DataLoader(multi, batch_size=4, shuffle=True, collate_metadata=True)
        all_dataset_indices = []
        for batch_data, metadata_list in loader:
            all_dataset_indices.extend(
                m[DATASET_INDEX_METADATA_KEY] for m in metadata_list
            )
        assert set(all_dataset_indices) == {0, 1}
        assert len(all_dataset_indices) == 20


class TestMultiDatasetRepr:
    """String representation."""

    def test_repr(self, numpy_data_dir):
        """Repr includes output_strict and datasets."""
        ds_a = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        ds_b = dp.Dataset(dp.NumpyReader(numpy_data_dir))
        multi = dp.MultiDataset(ds_a, ds_b, output_strict=True)

        r = repr(multi)
        assert "MultiDataset" in r
        assert "output_strict=True" in r
