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

"""Adapt model outputs into a per-field `TensorDict` keyed by target name.

The recipe's loss / metrics consume `TensorDict[name -> tensor]` predictions
keyed by the field names declared in the dataset YAML's ``targets:`` block.
Different model output conventions need different unpacking:

- **Mesh output** (e.g. GLOBE): ``output.point_data.select(*target_config)``
  yields a TensorDict with batch_size ``[N]`` matching the mesh's points.
- **Tensor output** (e.g. GeoTransolver / Transolver / FLARE): a ``(B, N, C)``
  tensor whose channels are concatenated in ``target_config`` order; we
  slice by per-field dim count and squeeze the trailing 1 for scalars.

Several prediction point sets (``target_sources``) map onto outputs by the
model YAML's ``output_groups`` (one tensor per group of targets, for branch
models) or ``output_sets`` (one Mesh output on the sets' merged points, split
back by set size, for single-decoder models).

Kept in its own module (separate from ``train.py``) so tests can exercise
the slicing logic without paying the cost of importing tensorboard.
"""

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import torch
from jaxtyping import Float
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from utils import FieldType, field_dim

from physicsnemo.datapipes.keys import as_nested_key, format_leaf_keys
from physicsnemo.mesh import Mesh

### Recipe-wide I/O contract literal: every model declares whether its
### `forward()` consumes / returns Mesh-like objects or plain `(B, N, C)`
### tensors. The collate, output normalizer, and forward-pass dispatch
### all key off this same enum.
IOType: TypeAlias = Literal["mesh", "tensors"]


def require_output_type(cfg: DictConfig) -> IOType:
    """Return the model's declared ``output_type``, or raise if missing/invalid.

    Both entry points (``train.py`` / ``infer.py``) require the model YAML
    to declare ``output_type`` (``"mesh"`` or ``"tensors"``) so the forward
    output can be unpacked; this is the shared validation.
    """
    output_type = cfg.get("output_type", None)
    if output_type not in ("mesh", "tensors"):
        raise ValueError(
            f"Model YAML must declare `output_type` as one of 'mesh', "
            f"'tensors'; got {output_type!r}."
        )
    return output_type


def resolve_output_layout(cfg: DictConfig) -> dict[str, Any]:
    """The model YAML's ``output_groups`` or ``output_sets``, as plain containers.

    Returns keyword arguments for :func:`normalize_output_to_tensordict`;
    empty when the model declares neither.
    """
    layout = {
        key: OmegaConf.to_container(cfg[key], resolve=True)
        for key in ("output_groups", "output_sets")
        if cfg.get(key) is not None
    }
    if len(layout) > 1:
        raise ValueError("output_groups and output_sets are mutually exclusive")
    return layout


def _check_partition(groups: list[list[str]], target_config: dict, key: str) -> None:
    """Raise unless *groups* name every target exactly once."""
    names = [name for group in groups for name in group]
    if (
        not groups
        or any(not group for group in groups)
        or len(names) != len(set(names))
        or set(names) != set(target_config)
    ):
        raise ValueError(f"{key} must partition the targets exactly once")


def split_concat_by_target(
    tensor: Float[torch.Tensor, "B N C"],
    target_config: dict[str, FieldType],
    n_spatial_dims: int = 3,
) -> TensorDict:
    """Slice a ``(B, N, C)`` tensor by ``target_config`` into a per-field TensorDict.

    Channels are consumed in the iteration order of ``target_config``: a
    ``"scalar"`` field eats one channel and the trailing dim is squeezed
    (so the leaf is shape ``(B, N)``); a ``"vector"`` field eats
    ``n_spatial_dims`` channels and keeps its trailing dim (leaf shape
    ``(B, N, n_spatial_dims)``).

    Args:
        tensor: Concatenated output tensor, shape ``(B, N, C)`` where
            ``C == sum(field_dim(t, n_spatial_dims) for t in target_config.values())``.
        target_config: Ordered ``{name: "scalar" | "vector"}`` mapping.
        n_spatial_dims: Vector-field dimensionality. Default 3.

    Returns:
        TensorDict with one leaf per target field and ``batch_size == tensor.shape[:2]``.

    Raises:
        ValueError: If the channel count does not match the expected total.
    """
    expected_channels = sum(
        field_dim(t, n_spatial_dims) for t in target_config.values()
    )
    if tensor.shape[-1] != expected_channels:
        raise ValueError(
            f"Output channel dim {tensor.shape[-1]} does not match the "
            f"expected total channels {expected_channels} for "
            f"target_config={target_config!r}."
        )

    ### Build the TD leaf by leaf; ``set`` with a tuple key creates the
    ### nesting, so a ``"solution.p"`` target lands at the same place
    ### ``extract_targets`` reads it from.
    out = TensorDict({}, batch_size=tensor.shape[:2], device=tensor.device)
    idx = 0
    for name, ftype in target_config.items():
        dim = field_dim(ftype, n_spatial_dims)
        slice_ = tensor[..., idx : idx + dim]
        if ftype == "scalar":
            slice_ = slice_.squeeze(-1)
        out.set(as_nested_key(name), slice_)
        idx += dim

    return out


def normalize_output_to_tensordict(
    output: Any,
    target_config: dict[str, FieldType],
    output_type: IOType,
    n_spatial_dims: int = 3,
    *,
    output_groups: list[list[str]] | None = None,
    output_sets: dict[str, dict[str, str]] | None = None,
    targets: TensorDict | None = None,
) -> TensorDict:
    """Adapt a model output into a `TensorDict` keyed by target name.

    For ``output_type == "mesh"``, the output is expected to be a `Mesh`
    whose `.point_data` contains one tensor per target name (e.g. GLOBE);
    we return ``output.point_data.select(*target_config)`` so the result
    inherits the mesh's batch_size (``[N]``) and device.

    For ``output_type == "tensors"``, the output is expected to be a
    ``(B, N, C)`` tensor whose channels are concatenated in
    ``target_config`` order (e.g. GeoTransolver, Transolver, FLARE,
    DoMINO); we slice it via :func:`split_concat_by_target`. DoMINO
    returns a ``(vol, surf)`` tuple; we take the non-None element
    automatically. ``output_groups`` and ``output_sets`` map outputs onto
    several prediction sets (see the module docstring).

    Args:
        output: Raw model output. ``Mesh`` for ``output_type='mesh'``;
            ``Tensor`` (or DoMINO ``(vol, surf)`` tuple) for
            ``output_type='tensors'``.
        target_config: Ordered ``{name: "scalar" | "vector"}`` mapping.
        output_type: ``"mesh"`` or ``"tensors"``.
        n_spatial_dims: Vector-field dimensionality. Default 3.
        output_groups: Ordered target-name lists, one per output tensor.
        output_sets: Ordered ``{set: {target: output field}}`` of a merged
            Mesh output (``output_type='mesh'`` only).
        targets: The batch targets; ``output_sets`` reads each set's size
            from them.

    Returns:
        Per-field TensorDict ready to feed into the loss / metric calculators.
    """
    if output_sets is not None:
        return _split_output_sets(
            output, target_config, output_type, output_sets, targets
        )
    if output_type == "mesh":
        if not isinstance(output, Mesh):
            raise TypeError(
                f"output_type='mesh' but model returned {type(output).__name__}"
            )
        keys = [as_nested_key(name) for name in target_config]
        missing = [
            name
            for name, key in zip(target_config, keys)
            if key not in output.point_data
        ]
        if missing:
            raise KeyError(
                f"Mesh output is missing target fields {missing!r}; "
                f"available: {format_leaf_keys(output.point_data)!r}"
            )
        return output.point_data.select(*keys)

    if output_type == "tensors":
        if output_groups is not None:
            _check_partition(output_groups, target_config, "output_groups")
            if not isinstance(output, tuple) or len(output) != len(output_groups):
                raise ValueError("output_groups requires one output tensor per group")
            if any(not isinstance(t, torch.Tensor) or t.ndim != 3 for t in output):
                raise ValueError("Each output group requires a (B, N, C) tensor")
            return _split_by_groups(
                output, output_groups, target_config, n_spatial_dims
            )
        if isinstance(output, tuple):
            ### DoMINO returns ``(vol, surf)`` with the unused branch as
            ### ``None``; pick the first non-None entry. A tuple of all
            ### Nones means the model was misconfigured for both modes,
            ### which the bare ``next(...)`` would surface as a cryptic
            ### ``StopIteration``; raise an explicit, actionable error here
            ### instead.
            non_none = [o for o in output if o is not None]
            if not non_none:
                raise ValueError(
                    f"output_type='tensors' got a tuple of all-None values; "
                    f"expected at least one non-None tensor in the tuple. "
                    f"Got {output!r}."
                )
            if len(non_none) != 1:
                raise ValueError("Multiple output tensors require output_groups")
            output = non_none[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"output_type='tensors' but model returned {type(output).__name__}"
            )
        ### A 2-D output (B, N) is almost certainly a model that dropped the
        ### channel dim for a single-scalar target. Diagnose that explicitly
        ### before the channel-count check, otherwise the user sees
        ### "expected 1, got N" which mistakes the per-element axis for the
        ### channel axis.
        if output.ndim < 3:
            raise ValueError(
                f"output_type='tensors' expects a (B, N, C) tensor; got "
                f"shape {tuple(output.shape)} (ndim={output.ndim}). If your "
                f"model returns (B, N) for a single-scalar target, add a "
                f"trailing channel dim (e.g. ``out.unsqueeze(-1)``)."
            )
        return split_concat_by_target(output, target_config, n_spatial_dims)

    raise ValueError(f"Unknown output_type {output_type!r}")


def _split_by_groups(
    tensors: Any, groups: list[list[str]], target_config: dict, n_spatial_dims: int
) -> TensorDict:
    """Split each ``(B, N_i, C_i)`` tensor by its group's targets, in order."""
    result = TensorDict({}, batch_size=tensors[0].shape[:1], device=tensors[0].device)
    for tensor, names in zip(tensors, groups):
        group = {n: target_config[n] for n in names}
        fields = split_concat_by_target(tensor, group, n_spatial_dims)
        for name in names:
            result.set(as_nested_key(name), fields[as_nested_key(name)])
    return result


def check_output_sets(
    output_sets: dict[str, dict[str, str]],
    target_sources: dict[str, str] | None,
    forward_kwargs: dict[str, Any] | None = None,
) -> None:
    """Raise unless each ``output_sets`` key is the set its targets come from.

    The split follows the key order, so a key must name its targets' source
    mesh, and a ``concat_point_sets`` list in *forward_kwargs* that merges
    exactly these sets must merge them in that order; several sets need
    ``target_sources``.
    """
    for sets in _concat_point_set_lists(forward_kwargs):
        if set(sets) == set(output_sets) and list(sets) != list(output_sets):
            raise ValueError(
                f"concat_point_sets {sets!r} must list the output_sets keys "
                f"{list(output_sets)!r}, in that order"
            )
    if target_sources is None:
        if len(output_sets) > 1:
            raise ValueError("output_sets with several sets needs target_sources")
        return
    for key, mapping in output_sets.items():
        prefix = f"{key}.point_data."
        wrong = [n for n in mapping if not target_sources.get(n, "").startswith(prefix)]
        if wrong:
            raise ValueError(
                f"output_sets[{key!r}] lists targets {wrong!r} whose "
                f"target_sources do not start with {prefix!r}"
            )


def _concat_point_set_lists(spec: Any) -> list[list[str]]:
    """Every ``concat_point_sets`` list in a ``forward_kwargs`` spec."""
    if isinstance(spec, dict):
        found = [spec["concat_point_sets"]] if "concat_point_sets" in spec else []
        return found + [s for v in spec.values() for s in _concat_point_set_lists(v)]
    if isinstance(spec, list):
        return [s for v in spec for s in _concat_point_set_lists(v)]
    return []


def _split_output_sets(
    output: Any,
    target_config: dict[str, FieldType],
    output_type: IOType,
    output_sets: dict[str, dict[str, str]],
    targets: TensorDict | None,
) -> TensorDict:
    """Split one merged Mesh output back into the prediction sets of ``output_sets``."""
    groups = [list(mapping) for mapping in output_sets.values()]
    _check_partition(groups, target_config, "output_sets")
    if output_type != "mesh" or not isinstance(output, Mesh):
        raise TypeError("output_sets requires output_type: mesh and a Mesh output")
    ### One set covers every output row. Several sets have target_sources,
    ### whose targets' root batch excludes the point axis (see extract_targets).
    sizes = [output.point_data.batch_size[0]]
    if len(groups) > 1:
        if targets is None:
            raise ValueError("output_sets needs the batch targets to size each set")
        axis = targets.batch_dims
        sizes = [targets[as_nested_key(g[0])].shape[axis] for g in groups]
    result = TensorDict({}, device=output.points.device)
    for part, mapping in zip(output.point_data.split(sizes, 0), output_sets.values()):
        for name, field in mapping.items():
            result.set(as_nested_key(name), part[as_nested_key(field)])
    return result
