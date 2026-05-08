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

"""Unit tests for `src/train.py`'s private TensorDict-aware walkers and for `src/output_normalize.py`.

``TensorDict`` is not a ``dict`` subclass, so the bare
``isinstance(obj, dict)`` branches in the recipe's recursive helpers
must be paired with explicit ``isinstance(obj, TensorDict)`` branches
for TD inputs to be walked at all. These tests pin that explicit
handling for:

- :func:`train._recursive_to_device`: must move TensorDict leaves to
  the requested device, including when the TD is nested under a plain
  dict. The tests assert ``result.device == cpu`` -- a freshly built
  ``TensorDict(..., batch_size=[N])`` has ``device is None``, and
  ``td.to("cpu")`` updates ``.device``, so this assertion fails if the
  walker skips the TD branch.
- :func:`train._recursive_cast_floats`: ``Mesh``, ``DomainMesh``, and
  ``TensorDict`` propagate the conditional cast via their
  auto-injected ``.apply()`` -- float leaves move to the requested
  dtype, integer leaves (e.g. ``Mesh.cells``) are kept by the same
  ``is_floating_point()`` guard as the bare ``Tensor`` branch. Relies
  on ``DomainMesh.apply`` recursing into ``interior``, ``boundaries``,
  and ``global_data`` (physicsnemo >= the fix in
  NVIDIA/physicsnemo#1621).
- :func:`train._walk_batch_for_logging`: must yield ``(name, tensor)``
  pairs from TensorDict leaves -- including correctly producing dotted
  paths for nested TDs via ``TD.flatten_keys('.')``.
- :func:`output_normalize.normalize_output_to_tensordict`: routes a
  model output (``Mesh`` or ``(B, N, C)`` tensor) to a per-target
  TensorDict, with clear error messages on shape / channel-count
  mismatches.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

### `train.py` imports `torch.utils.tensorboard.SummaryWriter` at module
### load, which transitively requires the `tensorboard` package. That
### dep is not declared in pyproject.toml; CI / training environments
### have it installed, but bare dev sandboxes might not. Skip cleanly.
### `output_normalize` itself is tensorboard-free, so we import it
### directly (no skip).
pytest.importorskip("tensorboard")

from train import (  # noqa: E402  -- after the importorskip guard
    _recursive_cast_floats,
    _recursive_to_device,
    _walk_batch_for_logging,
)
from output_normalize import normalize_output_to_tensordict  # noqa: E402
from physicsnemo.mesh import (  # noqa: E402  -- after the importorskip guard
    DomainMesh,
    Mesh,
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
        ### `.to(cpu)` sets `.device`, so a non-None `.device` here is
        ### proof the walker recursed into the TD branch (a skipped TD
        ### would leave `.device` at its initial `None`).
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

    def test_tensor_aware_containers_cast_floats_via_apply(self):
        """Mesh, DomainMesh, and TensorDict propagate the conditional cast.

        Each container's auto-injected ``.apply()`` recurses through every
        tensor leaf: float leaves move to the new dtype while int leaves
        (e.g. ``Mesh.cells``) are preserved by the same
        ``is_floating_point()`` guard used in the bare-Tensor branch.

        The DomainMesh assertions exercise the upstream invariant that
        ``DomainMesh.apply`` recurses into ``interior``, ``boundaries``,
        and ``global_data`` (provided by physicsnemo >=
        NVIDIA/physicsnemo#1621). If any of those fields stops being
        walked, these assertions are the canary.
        """
        td = TensorDict(
            {
                "f": torch.zeros(3, dtype=torch.float32),
                "i": torch.zeros(3, dtype=torch.int64),
            },
            batch_size=[3],
        )
        mesh = Mesh(
            points=torch.zeros(5, 3),
            cells=torch.zeros(2, 3, dtype=torch.int64),
            point_data={"p": torch.zeros(5)},
            cell_data={"normals": torch.zeros(2, 3)},
        )
        dm = DomainMesh(
            interior=Mesh(points=torch.zeros(3, 3), point_data={"p": torch.zeros(3)}),
            boundaries={
                "v": Mesh(
                    points=torch.zeros(2, 3),
                    cells=torch.zeros(1, 3, dtype=torch.int64),
                )
            },
            global_data={"U_inf": torch.zeros(3)},
        )

        td_cast = _recursive_cast_floats(td, torch.bfloat16)
        assert isinstance(td_cast, TensorDict)
        assert td_cast["f"].dtype == torch.bfloat16
        assert td_cast["i"].dtype == torch.int64

        mesh_cast = _recursive_cast_floats(mesh, torch.bfloat16)
        assert isinstance(mesh_cast, Mesh)
        assert mesh_cast.points.dtype == torch.bfloat16
        assert mesh_cast.cells.dtype == torch.int64
        assert mesh_cast.point_data["p"].dtype == torch.bfloat16
        assert mesh_cast.cell_data["normals"].dtype == torch.bfloat16

        dm_cast = _recursive_cast_floats(dm, torch.bfloat16)
        ### Every leaf inside the DomainMesh -- interior, boundaries, and
        ### global_data alike -- must end up at the requested dtype, with
        ### integer leaves (e.g. cells) untouched.
        assert isinstance(dm_cast, DomainMesh)
        assert dm_cast.interior.points.dtype == torch.bfloat16
        assert dm_cast.interior.point_data["p"].dtype == torch.bfloat16
        assert dm_cast.boundaries["v"].points.dtype == torch.bfloat16
        assert dm_cast.boundaries["v"].cells.dtype == torch.int64
        assert dm_cast.global_data["U_inf"].dtype == torch.bfloat16

    def test_plain_float_tensor_still_cast(self):
        """Plain float tensors cast; plain int tensors pass through unchanged."""
        t = torch.zeros(3, dtype=torch.float32)
        assert _recursive_cast_floats(t, torch.bfloat16).dtype == torch.bfloat16
        ### Int tensors are not cast.
        i = torch.zeros(3, dtype=torch.int64)
        assert _recursive_cast_floats(i, torch.bfloat16).dtype == torch.int64


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

    def test_walk_handles_nested_tensordict_via_flatten_keys(self):
        """A TD nested under another TD: ``flatten_keys`` produces dotted paths.

        This exercises the idiomatic-TD path: ``flatten_keys('.')`` on a
        nested TD returns a flat TD whose keys are dotted leaf paths.
        Without that delegation, a manual ``.items()`` walk would still
        work for flat TDs but would silently mishandle nested ones.
        """
        td = TensorDict(
            {
                "scalar": torch.zeros(3),
                "nested": TensorDict({"x": torch.zeros(3)}, batch_size=[3]),
            },
            batch_size=[3],
        )
        items = dict(_walk_batch_for_logging(td))
        assert set(items) == {"scalar", "nested.x"}
        ### And under a plain dict prefix, paths cascade correctly:
        items_with_prefix = dict(_walk_batch_for_logging({"targets": td}))
        assert set(items_with_prefix) == {"targets.scalar", "targets.nested.x"}


### ---------------------------------------------------------------------------
### normalize_output_to_tensordict
### ---------------------------------------------------------------------------


class TestNormalizeOutputToTensordict:
    """Tests for `normalize_output_to_tensordict`."""

    def test_tensors_output_three_dim_splits_correctly(self):
        """Standard (B, N, total_C) output splits into per-field leaves."""
        target_config = {"pressure": "scalar", "wss": "vector"}
        out = torch.randn(1, 50, 4)  # 1 scalar + 1 vector(3) = 4 channels
        td = normalize_output_to_tensordict(out, target_config, "tensors")
        assert tuple(td["pressure"].shape) == (1, 50)  # squeezed scalar
        assert tuple(td["wss"].shape) == (1, 50, 3)
        assert td.batch_size == torch.Size([1, 50])

    def test_tensors_output_two_dim_raises_clearly(self):
        """Two-D output (missing channel dim) raises a clear shape error.

        A ``(B, N)`` output for a single-scalar target is a config bug:
        without the explicit ``ndim < 3`` guard the per-element axis ``N``
        gets compared to the expected channel count ``C``, yielding a
        confusing "channel dim ``N`` does not match expected ``1``" error.
        The guard surfaces the actual problem (missing trailing channel
        dimension) directly.
        """
        target_config = {"pressure": "scalar"}
        out = torch.randn(1, 50)
        with pytest.raises(ValueError, match=r"expects a \(B, N, C\) tensor"):
            normalize_output_to_tensordict(out, target_config, "tensors")

    def test_tensors_output_channel_mismatch_still_raises(self):
        """Three-D output with wrong channel count still raises the channel error."""
        target_config = {"pressure": "scalar"}
        out = torch.randn(1, 50, 3)  # expected 1 channel
        with pytest.raises(ValueError, match="does not match the expected"):
            normalize_output_to_tensordict(out, target_config, "tensors")

    def test_mesh_output_extracts_target_fields(self):
        """Mesh output: ``point_data.select(*target_config)`` keeps batch_size [N]."""
        target_config = {"pressure": "scalar", "wss": "vector"}
        mesh = Mesh(
            points=torch.randn(7, 3),
            point_data={
                "pressure": torch.randn(7),
                "wss": torch.randn(7, 3),
                ### A non-target field that must NOT appear in the result.
                "extra": torch.randn(7),
            },
        )
        td = normalize_output_to_tensordict(mesh, target_config, "mesh")
        assert set(td.keys()) == {"pressure", "wss"}
        assert td.batch_size == torch.Size([7])

    def test_mesh_output_missing_target_raises(self):
        """Missing target field on a Mesh output is reported clearly."""
        target_config = {"pressure": "scalar"}
        mesh = Mesh(points=torch.randn(7, 3), point_data={"other": torch.randn(7)})
        with pytest.raises(KeyError, match="missing target fields"):
            normalize_output_to_tensordict(mesh, target_config, "mesh")
