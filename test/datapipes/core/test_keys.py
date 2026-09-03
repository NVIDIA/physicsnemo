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

"""Tests for the nested-key helpers in ``physicsnemo.datapipes.keys``."""

import inspect

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.datapipes.keys import (
    KEY_SEPARATOR,
    as_nested_key,
    as_nested_keys,
    format_leaf_keys,
    get_leaf,
    key_to_str,
    leaf_keys,
    present_keys,
    rename_keys,
    require_keys,
    with_leaf_name,
)


def _nested_td() -> TensorDict:
    return TensorDict(
        {
            "solution": {"p": torch.zeros(4), "v": torch.zeros(4, 3)},
            "sdf": torch.ones(4),
        },
        batch_size=[4],
    )


class TestAsNestedKey:
    def test_plain_string_stays_string(self):
        assert as_nested_key("pressure") == "pressure"

    def test_dotted_string_becomes_tuple(self):
        assert as_nested_key("solution.p") == ("solution", "p")
        assert as_nested_key("a.b.c") == ("a", "b", "c")

    def test_sequence_is_taken_verbatim(self):
        ### A list lets a leaf name that itself contains "." be addressed.
        assert as_nested_key(["solution", "p.mean"]) == ("solution", "p.mean")
        assert as_nested_key(("solution", "p")) == ("solution", "p")

    def test_single_element_sequence_collapses_to_string(self):
        assert as_nested_key(["pressure"]) == "pressure"
        assert as_nested_key(["p.mean"]) == "p.mean"

    def test_separator_matches_tensordict_default(self):
        ### The config separator is deliberately the one TensorDict itself
        ### uses for ``flatten_keys`` / ``unflatten_keys``; fail loudly if
        ### tensordict ever changes it.
        flatten_default = (
            inspect.signature(TensorDict.flatten_keys).parameters["separator"].default
        )
        assert KEY_SEPARATOR == flatten_default == "."

    @pytest.mark.parametrize("bad", ["", "a..b", ".a", "a."])
    def test_empty_component_raises(self, bad):
        with pytest.raises(ValueError, match="non-empty"):
            as_nested_key(bad)

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            as_nested_key(3)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            as_nested_key(["a", 3])  # type: ignore[list-item]

    def test_as_nested_keys_handles_none(self):
        assert as_nested_keys(None) == []
        assert as_nested_keys(["a", "b.c"]) == ["a", ("b", "c")]

    def test_key_to_str_round_trip(self):
        for name in ("pressure", "solution.p", "a.b.c"):
            assert key_to_str(as_nested_key(name)) == name


class TestLeafHelpers:
    def test_leaf_keys_descend_into_groups(self):
        td = _nested_td()
        assert set(leaf_keys(td)) == {("solution", "p"), ("solution", "v"), "sdf"}
        assert format_leaf_keys(td) == ["sdf", "solution.p", "solution.v"]

    def test_get_leaf_nested(self):
        td = _nested_td()
        assert get_leaf(td, ("solution", "v")).shape == (4, 3)
        assert get_leaf(td, "sdf").shape == (4,)

    def test_get_leaf_missing_lists_leaves(self):
        td = _nested_td()
        with pytest.raises(KeyError, match="solution.p"):
            get_leaf(td, ("solution", "missing"))
        ### Descending through a leaf tensor is also "not found".
        with pytest.raises(KeyError):
            get_leaf(td, ("sdf", "x"))

    def test_get_leaf_on_group_raises_type_error(self):
        td = _nested_td()
        with pytest.raises(TypeError, match="group of fields"):
            get_leaf(td, "solution")

    def test_with_leaf_name_keeps_parents(self):
        assert with_leaf_name("v", lambda n: f"knn_{n}") == "knn_v"
        assert with_leaf_name(("solution", "v"), lambda n: f"knn_{n}") == (
            "solution",
            "knn_v",
        )


class TestRenameKeys:
    def test_nested_to_top_and_back(self):
        td = _nested_td()
        out = rename_keys(td, {("solution", "p"): "pressure"}, strict=True)
        assert "pressure" in out
        assert ("solution", "p") not in out
        assert torch.equal(out["pressure"], td["solution", "p"])

        back = rename_keys(out, {"pressure": ("solution", "p")}, strict=True)
        assert ("solution", "p") in back

    def test_rename_group(self):
        td = _nested_td()
        out = rename_keys(td, {"solution": "raw"}, strict=True)
        assert set(leaf_keys(out)) == {("raw", "p"), ("raw", "v"), "sdf"}

    def test_input_not_mutated_and_storage_shared(self):
        td = _nested_td()
        out = rename_keys(td, {("solution", "p"): ("solution", "pp")}, strict=True)
        assert ("solution", "p") in td
        assert ("solution", "pp") not in td
        assert out["solution", "pp"].data_ptr() == td["solution", "p"].data_ptr()

    def test_strict_missing_raises_with_leaf_listing(self):
        td = _nested_td()
        with pytest.raises(KeyError, match="solution.v"):
            rename_keys(td, {("solution", "zz"): "z"}, strict=True)

    def test_non_strict_skips_missing(self):
        td = _nested_td()
        out = rename_keys(td, {("solution", "zz"): "z", "sdf": "d"}, strict=False)
        assert "d" in out and "z" not in out

    def test_conflict_raises(self):
        td = _nested_td()
        with pytest.raises(ValueError, match="conflict"):
            rename_keys(td, {("solution", "p"): "sdf"}, strict=True)

    def test_chained_rename_keeps_every_value(self):
        ### ``a -> b, b -> c`` must move both values, not overwrite ``b``
        ### with ``a`` before ``b`` has been moved on.
        td = TensorDict({"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}, [1])
        out = rename_keys(td, {"a": "b", "b": "c"}, strict=True)
        assert {k: v.item() for k, v in out.items()} == {"b": 1.0, "c": 2.0}

    def test_swap_rename(self):
        td = TensorDict({"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}, [1])
        out = rename_keys(td, {"a": "b", "b": "a"}, strict=True)
        assert {k: v.item() for k, v in out.items()} == {"a": 2.0, "b": 1.0}

    def test_nested_chain_across_groups(self):
        td = _nested_td()
        out = rename_keys(
            td, {("solution", "p"): "sdf", "sdf": ("solution", "p")}, strict=True
        )
        assert torch.equal(out["sdf"], td["solution", "p"])
        assert torch.equal(out["solution", "p"], td["sdf"])

    def test_duplicate_destination_raises(self):
        td = _nested_td()
        with pytest.raises(ValueError, match="same name"):
            rename_keys(td, {("solution", "p"): "x", "sdf": "x"}, strict=True)

    def test_hoisting_last_leaf_prunes_empty_group(self):
        td = TensorDict({"a": {"b": {"d": torch.zeros(3)}}, "x": torch.zeros(3)}, [3])
        out = rename_keys(td, {("a", "b", "d"): "d"}, strict=True)
        assert set(out.keys()) == {"d", "x"}
        ### A group that still has leaves is kept.
        out = rename_keys(_nested_td(), {("solution", "p"): "p"}, strict=True)
        assert set(out.keys(True, True)) == {"p", ("solution", "v"), "sdf"}


class TestKeyGuards:
    def test_require_keys_reports_missing_with_leaf_listing(self):
        td = _nested_td()
        require_keys(td, [("solution", "p"), "sdf"])
        ### A path running through a leaf tensor is "missing", not an AttributeError.
        with pytest.raises(KeyError, match="sdf.x"):
            require_keys(td, [("sdf", "x")])

    def test_present_keys_filters_paths_through_leaves(self):
        td = _nested_td()
        assert present_keys(td, [("sdf", "x"), "sdf", ("solution", "zz")]) == ["sdf"]

    def test_get_leaf_hints_at_literal_dotted_name(self):
        td = TensorDict({"p.mean": torch.zeros(2)}, batch_size=[2])
        with pytest.raises(KeyError, match="one-element list"):
            get_leaf(td, as_nested_key("p.mean"))
        assert get_leaf(td, as_nested_key(["p.mean"])).shape == (2,)
