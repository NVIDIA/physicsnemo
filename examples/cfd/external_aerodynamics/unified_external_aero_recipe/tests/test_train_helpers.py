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

"""Unit tests for the private TensorDict-aware walkers in `src/train.py`.

`TensorDict` is not a `dict` subclass; the bare ``isinstance(obj, dict)``
branches in the recipe's three recursive helpers therefore silently skip
TensorDict inputs unless an explicit branch is added. These tests pin
down that explicit handling for:

- :func:`train._recursive_to_device`: must move TensorDict leaves to the
  requested device, including when the TD is nested under a plain dict.
- :func:`train._recursive_cast_floats`: must cast only floating-point
  leaves of a TensorDict, leaving int leaves (e.g., mesh cell indices)
  untouched.
- :func:`train._walk_batch_for_logging`: must yield ``(name, tensor)``
  pairs from TensorDict leaves, both at the top level and when the TD
  is nested under a plain dict.

The regression-test trick used for `_recursive_to_device` is:
``TensorDict(..., batch_size=[N])`` without an explicit ``device`` has
``td.device is None``, while ``td.to("cpu")`` sets ``.device`` to
``torch.device("cpu")``. So the assertion ``result.device == cpu``
distinguishes the post-fix behaviour (TD went through ``.to``) from
the pre-fix behaviour (TD was returned untouched and would still have
``.device is None``).
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

### `train.py` imports `torch.utils.tensorboard.SummaryWriter` at module
### load, which transitively requires the `tensorboard` package. That
### dep is not declared in pyproject.toml; CI / training environments
### have it installed, but bare dev sandboxes might not. Skip cleanly.
pytest.importorskip("tensorboard")

from train import (  # noqa: E402  -- after the importorskip guard
    _recursive_cast_floats,
    _recursive_to_device,
    _walk_batch_for_logging,
)


### ---------------------------------------------------------------------------
### _recursive_to_device
### ---------------------------------------------------------------------------


class TestRecursiveToDevice:
    """Tests for `_recursive_to_device`."""

    def test_tensordict_input_moves_to_device(self):
        """Bare TD input goes through `.to(device)`."""
        cpu = torch.device("cpu")
        td = TensorDict(
            {"pressure": torch.zeros(4), "wss": torch.zeros(4, 3)},
            batch_size=[4],
        )
        ### Baseline: TD with no explicit device has .device is None.
        assert td.device is None

        result = _recursive_to_device(td, cpu)
        assert isinstance(result, TensorDict)
        ### Post-fix: TD went through .to(cpu), which sets .device.
        ### Pre-fix (silent skip), this would still be None.
        assert result.device == cpu
        assert result["pressure"].device == cpu
        assert result["wss"].device == cpu
        assert set(result.keys()) == {"pressure", "wss"}

    def test_dict_with_nested_tensordict(self):
        """Plain dict containing a TD: walker recurses into the dict, then
        the TD branch picks up the inner TD."""
        cpu = torch.device("cpu")
        batch = {
            "forward_kwargs": {"x": torch.zeros(2, 3)},
            "targets": TensorDict({"pressure": torch.zeros(4)}, batch_size=[4]),
        }
        assert batch["targets"].device is None

        result = _recursive_to_device(batch, cpu)
        assert isinstance(result, dict)
        assert isinstance(result["targets"], TensorDict)
        assert result["targets"].device == cpu
        assert result["forward_kwargs"]["x"].device == cpu


### ---------------------------------------------------------------------------
### _recursive_cast_floats
### ---------------------------------------------------------------------------


class TestRecursiveCastFloats:
    """Tests for `_recursive_cast_floats`."""

    def test_tensordict_casts_only_float_leaves(self):
        """Float leaf -> dtype; int leaf -> unchanged."""
        td = TensorDict(
            {
                "f": torch.zeros(3, dtype=torch.float32),
                "i": torch.zeros(3, dtype=torch.int64),
            },
            batch_size=[3],
        )

        result = _recursive_cast_floats(td, torch.bfloat16)
        assert isinstance(result, TensorDict)
        assert result["f"].dtype == torch.bfloat16
        ### Critical regression check: int leaves must NOT be silently cast,
        ### otherwise mesh cell indices (int64) would be corrupted.
        assert result["i"].dtype == torch.int64
        ### Structure (batch_size, key set) is preserved.
        assert result.batch_size == torch.Size([3])
        assert set(result.keys()) == {"f", "i"}


### ---------------------------------------------------------------------------
### _walk_batch_for_logging
### ---------------------------------------------------------------------------


class TestWalkBatchForLogging:
    """Tests for `_walk_batch_for_logging`."""

    def test_yields_from_tensordict_leaves(self):
        """Bare TD input yields one entry per leaf with the leaf path."""
        td = TensorDict(
            {"pressure": torch.zeros(5), "wss": torch.zeros(5, 3)},
            batch_size=[5],
        )

        items = dict(_walk_batch_for_logging(td))
        assert set(items) == {"pressure", "wss"}
        assert items["pressure"].shape == torch.Size([5])
        assert items["wss"].shape == torch.Size([5, 3])

    def test_dict_containing_tensordict_yields_dotted_keys(self):
        """Nested dict -> TD -> leaves: keys come back dot-joined."""
        batch = {
            "targets": TensorDict(
                {"pressure": torch.zeros(5), "wss": torch.zeros(5, 3)},
                batch_size=[5],
            ),
        }

        items = dict(_walk_batch_for_logging(batch))
        ### Without the TD branch in the walker, neither `targets.pressure`
        ### nor `targets.wss` would appear in the output.
        assert set(items) == {"targets.pressure", "targets.wss"}
        assert items["targets.pressure"].shape == torch.Size([5])
