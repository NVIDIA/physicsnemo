# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Tests for utility transforms: Rename, Purge, ConstantField, and ZeroLike."""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.datapipes.core.transforms import ConstantField, Purge, Rename


class TestRename:
    """Tests for the Rename transform."""

    def test_basic_rename(self):
        """Test basic key renaming."""
        data = TensorDict(
            {
                "old_name": torch.randn(100, 3),
                "other": torch.randn(100, 1),
            }
        )

        transform = Rename(mapping={"old_name": "new_name"})
        result = transform(data)

        assert "new_name" in result.keys()
        assert "old_name" not in result.keys()
        assert "other" in result.keys()
        assert torch.allclose(result["new_name"], data["old_name"])

    def test_multiple_renames(self):
        """Test renaming multiple keys at once."""
        data = TensorDict(
            {
                "x": torch.randn(100, 3),
                "y": torch.randn(100, 3),
                "z": torch.randn(100, 1),
            }
        )

        transform = Rename(mapping={"x": "positions", "y": "velocities"})
        result = transform(data)

        assert "positions" in result.keys()
        assert "velocities" in result.keys()
        assert "z" in result.keys()
        assert "x" not in result.keys()
        assert "y" not in result.keys()

    def test_strict_mode_missing_key_raises(self):
        """Test that strict mode raises error for missing keys."""
        data = TensorDict({"existing": torch.randn(10, 3)})

        transform = Rename(mapping={"missing": "new_name"}, strict=True)

        with pytest.raises(KeyError, match="missing"):
            transform(data)

    def test_non_strict_mode_skips_missing(self):
        """Test that non-strict mode skips missing keys."""
        data = TensorDict(
            {
                "existing": torch.randn(10, 3),
                "other": torch.randn(10, 1),
            }
        )

        transform = Rename(
            mapping={"missing": "new_name", "existing": "renamed"}, strict=False
        )
        result = transform(data)

        assert "renamed" in result.keys()
        assert "other" in result.keys()
        assert "existing" not in result.keys()
        # "missing" was skipped, so no "new_name" key either
        assert "new_name" not in result.keys()

    def test_conflict_with_existing_key_raises(self):
        """Test that renaming to an existing key raises error."""
        data = TensorDict(
            {
                "source": torch.randn(10, 3),
                "target": torch.randn(10, 1),
            }
        )

        transform = Rename(mapping={"source": "target"})

        with pytest.raises(ValueError, match="conflict"):
            transform(data)

    def test_rename_preserves_data(self):
        """Test that renaming preserves the tensor data exactly."""
        original_tensor = torch.randn(50, 4)
        data = TensorDict({"original": original_tensor.clone()})

        transform = Rename(mapping={"original": "renamed"})
        result = transform(data)

        assert torch.equal(result["renamed"], original_tensor)

    def test_rename_preserves_batch_size(self):
        """Test that renaming preserves the TensorDict batch_size."""
        data = TensorDict({"x": torch.randn(10, 3)}, batch_size=[10])

        transform = Rename(mapping={"x": "positions"})
        result = transform(data)

        assert result.batch_size == data.batch_size

    def test_extra_repr(self):
        """Test the extra_repr output."""
        transform = Rename(mapping={"a": "b"}, strict=False)
        repr_str = transform.extra_repr()

        assert "mapping" in repr_str
        assert "strict" in repr_str


class TestPurge:
    """Tests for the Purge transform."""

    def test_drop_only_single_key(self):
        """Test dropping a single key."""
        data = TensorDict(
            {
                "keep1": torch.randn(100, 3),
                "keep2": torch.randn(100, 1),
                "drop_me": torch.randn(100, 5),
            }
        )

        transform = Purge(drop_only=["drop_me"])
        result = transform(data)

        assert "keep1" in result.keys()
        assert "keep2" in result.keys()
        assert "drop_me" not in result.keys()

    def test_drop_only_multiple_keys(self):
        """Test dropping multiple keys."""
        data = TensorDict(
            {
                "keep": torch.randn(100, 3),
                "drop1": torch.randn(100, 1),
                "drop2": torch.randn(100, 5),
            }
        )

        transform = Purge(drop_only=["drop1", "drop2"])
        result = transform(data)

        assert "keep" in result.keys()
        assert "drop1" not in result.keys()
        assert "drop2" not in result.keys()

    def test_keep_only_single_key(self):
        """Test keeping only a single key."""
        data = TensorDict(
            {
                "keep": torch.randn(100, 3),
                "remove1": torch.randn(100, 1),
                "remove2": torch.randn(100, 5),
            }
        )

        transform = Purge(keep_only=["keep"])
        result = transform(data)

        assert "keep" in result.keys()
        assert "remove1" not in result.keys()
        assert "remove2" not in result.keys()
        assert len(list(result.keys())) == 1

    def test_keep_only_multiple_keys(self):
        """Test keeping multiple keys."""
        data = TensorDict(
            {
                "positions": torch.randn(100, 3),
                "velocities": torch.randn(100, 3),
                "temp": torch.randn(100, 1),
                "debug": torch.randn(100, 10),
            }
        )

        transform = Purge(keep_only=["positions", "velocities"])
        result = transform(data)

        assert "positions" in result.keys()
        assert "velocities" in result.keys()
        assert "temp" not in result.keys()
        assert "debug" not in result.keys()

    def test_default_drop_nothing(self):
        """Test that default (drop_only=None) drops nothing."""
        data = TensorDict(
            {
                "a": torch.randn(10, 3),
                "b": torch.randn(10, 1),
            }
        )

        transform = Purge()
        result = transform(data)

        assert "a" in result.keys()
        assert "b" in result.keys()
        assert len(list(result.keys())) == 2

    def test_both_options_raises(self):
        """Test that specifying both keep_only and drop_only raises error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            Purge(keep_only=["a"], drop_only=["b"])

    def test_strict_mode_drop_missing_raises(self):
        """Test strict mode raises for missing drop keys."""
        data = TensorDict({"existing": torch.randn(10, 3)})

        transform = Purge(drop_only=["missing"], strict=True)

        with pytest.raises(KeyError, match="missing"):
            transform(data)

    def test_strict_mode_keep_missing_raises(self):
        """Test strict mode raises for missing keep keys."""
        data = TensorDict({"existing": torch.randn(10, 3)})

        transform = Purge(keep_only=["missing"], strict=True)

        with pytest.raises(KeyError, match="missing"):
            transform(data)

    def test_non_strict_drop_skips_missing(self):
        """Test non-strict mode skips missing drop keys."""
        data = TensorDict(
            {
                "keep": torch.randn(10, 3),
                "drop": torch.randn(10, 1),
            }
        )

        transform = Purge(drop_only=["drop", "missing"], strict=False)
        result = transform(data)

        assert "keep" in result.keys()
        assert "drop" not in result.keys()

    def test_non_strict_keep_skips_missing(self):
        """Test non-strict mode skips missing keep keys."""
        data = TensorDict(
            {
                "existing": torch.randn(10, 3),
                "other": torch.randn(10, 1),
            }
        )

        transform = Purge(keep_only=["existing", "missing"], strict=False)
        result = transform(data)

        assert "existing" in result.keys()
        assert "other" not in result.keys()
        assert len(list(result.keys())) == 1

    def test_purge_preserves_data(self):
        """Test that purge preserves the tensor data exactly."""
        original = torch.randn(50, 4)
        data = TensorDict(
            {
                "keep": original.clone(),
                "drop": torch.randn(50, 2),
            }
        )

        transform = Purge(drop_only=["drop"])
        result = transform(data)

        assert torch.equal(result["keep"], original)

    def test_purge_preserves_batch_size(self):
        """Test that purge preserves the TensorDict batch_size."""
        data = TensorDict(
            {"a": torch.randn(10, 3), "b": torch.randn(10, 1)}, batch_size=[10]
        )

        transform = Purge(drop_only=["b"])
        result = transform(data)

        assert result.batch_size == data.batch_size

    def test_extra_repr_keep_only(self):
        """Test extra_repr for keep_only mode."""
        transform = Purge(keep_only=["a", "b"])
        repr_str = transform.extra_repr()

        assert "keep_only" in repr_str
        assert "strict" in repr_str

    def test_extra_repr_drop_only(self):
        """Test extra_repr for drop_only mode."""
        transform = Purge(drop_only=["a", "b"])
        repr_str = transform.extra_repr()

        assert "drop_only" in repr_str
        assert "strict" in repr_str

    def test_extra_repr_default(self):
        """Test extra_repr for default (identity) mode."""
        transform = Purge()
        repr_str = transform.extra_repr()

        assert "identity" in repr_str or "None" in repr_str

    def test_drop_all_keys(self):
        """Test dropping all keys results in empty TensorDict."""
        data = TensorDict(
            {
                "a": torch.randn(10, 3),
                "b": torch.randn(10, 1),
            }
        )

        transform = Purge(drop_only=["a", "b"])
        result = transform(data)

        assert len(list(result.keys())) == 0

    def test_keep_empty_list(self):
        """Test keeping empty list results in empty TensorDict."""
        data = TensorDict(
            {
                "a": torch.randn(10, 3),
                "b": torch.randn(10, 1),
            }
        )

        transform = Purge(keep_only=[])
        result = transform(data)

        assert len(list(result.keys())) == 0


class TestConstantField:
    """Tests for the ConstantField transform."""

    def test_default_zeros(self):
        """Test default fill_value creates zeros."""
        data = TensorDict({"positions": torch.randn(100, 3)})

        transform = ConstantField(reference_key="positions", output_key="sdf")
        result = transform(data)

        assert "sdf" in result.keys()
        assert result["sdf"].shape == (100, 1)
        assert torch.all(result["sdf"] == 0.0)

    def test_custom_fill_value(self):
        """Test custom fill_value."""
        data = TensorDict({"positions": torch.randn(100, 3)})

        transform = ConstantField(
            reference_key="positions", output_key="temperature", fill_value=293.15
        )
        result = transform(data)

        assert result["temperature"].shape == (100, 1)
        assert torch.allclose(
            result["temperature"],
            torch.full((100, 1), 293.15, dtype=data["positions"].dtype),
        )

    def test_ones_fill_value(self):
        """Test fill_value=1.0 creates ones."""
        data = TensorDict({"positions": torch.randn(50, 3)})

        transform = ConstantField(
            reference_key="positions", output_key="mask", fill_value=1.0
        )
        result = transform(data)

        assert torch.all(result["mask"] == 1.0)

    def test_negative_fill_value(self):
        """Test negative fill_value."""
        data = TensorDict({"positions": torch.randn(50, 3)})

        transform = ConstantField(
            reference_key="positions", output_key="indicator", fill_value=-1.0
        )
        result = transform(data)

        assert torch.all(result["indicator"] == -1.0)

    def test_custom_output_dim(self):
        """Test custom output dimension."""
        data = TensorDict({"positions": torch.randn(100, 3)})

        transform = ConstantField(
            reference_key="positions",
            output_key="features",
            fill_value=0.5,
            output_dim=5,
        )
        result = transform(data)

        assert result["features"].shape == (100, 5)
        assert torch.all(result["features"] == 0.5)

    def test_inherits_dtype_from_reference(self):
        """Test that output tensor inherits dtype from reference."""
        data = TensorDict({"positions": torch.randn(100, 3, dtype=torch.float64)})

        transform = ConstantField(reference_key="positions", output_key="sdf")
        result = transform(data)

        assert result["sdf"].dtype == torch.float64

    def test_inherits_device_from_reference(self):
        """Test that output tensor inherits device from reference."""
        data = TensorDict({"positions": torch.randn(100, 3)})

        transform = ConstantField(reference_key="positions", output_key="sdf")
        result = transform(data)

        assert result["sdf"].device == data["positions"].device

    def test_missing_reference_key_raises(self):
        """Test that missing reference key raises KeyError."""
        data = TensorDict({"other": torch.randn(10, 3)})

        transform = ConstantField(reference_key="missing", output_key="sdf")

        with pytest.raises(KeyError, match="missing"):
            transform(data)

    def test_preserves_existing_keys(self):
        """Test that existing keys are preserved."""
        data = TensorDict(
            {
                "positions": torch.randn(100, 3),
                "velocities": torch.randn(100, 3),
            }
        )

        transform = ConstantField(reference_key="positions", output_key="sdf")
        result = transform(data)

        assert "positions" in result.keys()
        assert "velocities" in result.keys()
        assert "sdf" in result.keys()

    def test_preserves_batch_size(self):
        """Test that batch_size is preserved."""
        data = TensorDict({"positions": torch.randn(100, 3)}, batch_size=[100])

        transform = ConstantField(reference_key="positions", output_key="sdf")
        result = transform(data)

        assert result.batch_size == data.batch_size

    def test_extra_repr(self):
        """Test extra_repr output."""
        transform = ConstantField(
            reference_key="pos", output_key="sdf", fill_value=0.5, output_dim=2
        )
        repr_str = transform.extra_repr()

        assert "reference_key" in repr_str
        assert "output_key" in repr_str
        assert "fill_value" in repr_str
        assert "output_dim" in repr_str
