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

r"""Single-process unit tests for ShardTensor promotion mode.

These tests exercise the pure-Python machinery introduced alongside
plain-tensor auto-promotion and do not require multiple GPUs:

- ``TensorPromotionMode`` and the ``ShardTensor`` promotion-mode accessors
  (``get_promotion_mode`` / ``set_promotion_mode`` / ``promotion_mode``).
- The ``_find_mesh_in_args`` reference-mesh walk.

The positive path of ``_find_mesh_in_args`` (actually locating a mesh on a
distributed tensor) requires a real ``DeviceMesh`` and is covered by the
multigpu integration tests; here we validate that it returns ``None`` when no
distributed tensor is present.
"""

import pytest

from physicsnemo.domain_parallel import shard_tensor as st_mod
from physicsnemo.domain_parallel.shard_tensor import ShardTensor, TensorPromotionMode


@pytest.fixture
def restore_promotion_mode():
    r"""Save and restore the global ShardTensor promotion mode around a test."""
    previous = ShardTensor.get_promotion_mode()
    try:
        yield
    finally:
        ShardTensor.set_promotion_mode(previous)


# ---------------------------------------------------------------------------
# Promotion mode accessors
# ---------------------------------------------------------------------------


def test_default_promotion_mode_is_warn(restore_promotion_mode):
    assert ShardTensor.get_promotion_mode() is TensorPromotionMode.SILENT


@pytest.mark.parametrize(
    "mode",
    [
        TensorPromotionMode.DISABLED,
        TensorPromotionMode.WARN,
        TensorPromotionMode.SILENT,
    ],
)
def test_set_promotion_mode_enum(restore_promotion_mode, mode):
    ShardTensor.set_promotion_mode(mode)
    assert ShardTensor.get_promotion_mode() is mode


@pytest.mark.parametrize(
    "value,expected",
    [
        ("disabled", TensorPromotionMode.DISABLED),
        ("warn", TensorPromotionMode.WARN),
        ("silent", TensorPromotionMode.SILENT),
    ],
)
def test_set_promotion_mode_string_coercion(restore_promotion_mode, value, expected):
    ShardTensor.set_promotion_mode(value)
    assert ShardTensor.get_promotion_mode() is expected


def test_set_promotion_mode_invalid_string_raises(restore_promotion_mode):
    with pytest.raises(ValueError):
        ShardTensor.set_promotion_mode("not-a-mode")


def test_promotion_mode_context_manager_restores(restore_promotion_mode):
    ShardTensor.set_promotion_mode(TensorPromotionMode.WARN)
    with ShardTensor.promotion_mode(TensorPromotionMode.SILENT):
        assert ShardTensor.get_promotion_mode() is TensorPromotionMode.SILENT
    assert ShardTensor.get_promotion_mode() is TensorPromotionMode.WARN


def test_promotion_mode_context_manager_restores_on_exception(restore_promotion_mode):
    ShardTensor.set_promotion_mode(TensorPromotionMode.DISABLED)
    with pytest.raises(RuntimeError):
        with ShardTensor.promotion_mode(TensorPromotionMode.SILENT):
            assert ShardTensor.get_promotion_mode() is TensorPromotionMode.SILENT
            raise RuntimeError("boom")
    assert ShardTensor.get_promotion_mode() is TensorPromotionMode.DISABLED


# ---------------------------------------------------------------------------
# Reference-mesh walk
# ---------------------------------------------------------------------------


def test_find_mesh_in_args_returns_none_for_plain_containers():
    assert st_mod._find_mesh_in_args(1, "x", None) is None
    assert st_mod._find_mesh_in_args({"a": [1, 2]}, (3, [4, {"b": 5}])) is None
