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

import copy
import importlib
import json
import pickle

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh import FieldSchema, RankSpec


def _mixed_fields(n: int = 4) -> TensorDict:
    return TensorDict(
        {
            "wind": torch.zeros(n, 3),
            "state": {"temperature": torch.zeros(n)},
            "displacement": torch.zeros(n, 3),
            "pressure": torch.zeros(n),
            "stress": torch.zeros(n, 3, 3),
        },
        batch_size=[n],
    )


### RankSpec --------------------------------------------------------------------


def test_rank_spec_parse_accepts_rank_spec_and_mapping():
    spec = RankSpec(rank=2, symmetric=True, parity="odd")
    assert RankSpec.parse(spec) is spec
    assert RankSpec.parse({"rank": 2, "symmetric": True, "parity": "odd"}) == spec
    assert RankSpec.parse({"rank": 0}) == RankSpec(rank=0)

    assert RankSpec(rank=0).shape(3) == ()
    assert RankSpec(rank=0).numel(3) == 1
    assert RankSpec(rank=1).shape(2) == (2,)
    assert RankSpec(rank=2).shape(3) == (3, 3)
    assert RankSpec(rank=2).numel(3) == 9


def test_rank_spec_is_its_canonical_declaration():
    ### A spec is the dict of its non-default attributes, so equal laws are
    ### equal (and hash alike) however they were spelled, and the JSON is the
    ### declaration a user writes.
    spec = RankSpec(2, symmetric=True)
    assert spec == {"rank": 2, "symmetric": True}
    assert spec == RankSpec(rank=2, symmetric=True, parity="even")
    assert hash(spec) == hash(RankSpec.parse({"rank": 2, "symmetric": True}))
    assert (spec.rank, spec.symmetric, spec.parity) == (2, True, "even")
    assert RankSpec(0) == {"rank": 0}
    assert json.dumps(RankSpec(1, parity="odd")) == '{"rank": 1, "parity": "odd"}'
    assert repr(spec) == "RankSpec(rank=2, symmetric=True)"


@pytest.mark.parametrize(
    ("spec", "error", "match"),
    [
        (0, TypeError, r"got int \(write \{'rank': 0\}\)"),
        (True, TypeError, "got bool$"),  # no "(write {'rank': 1})" hint
        (1.0, TypeError, "got float"),
        ({"symmetric": True}, ValueError, "must contain a 'rank' key"),
        ({"rank": 1, "channels": 3}, ValueError, r"unknown keys \['channels'\]"),
        ({"rank": True}, TypeError, "rank must be an integer"),
        ({"rank": -1}, ValueError, "must be non-negative"),
        ({"rank": 1, "parity": "axial"}, ValueError, "parity must be 'even' or 'odd'"),
        ({"rank": 1, "symmetric": True}, ValueError, "only meaningful for rank >= 2"),
        ({"rank": 2, "symmetric": 1}, TypeError, "symmetric must be a bool"),
    ],
)
def test_rank_spec_parse_rejects_invalid_specs(spec, error, match):
    with pytest.raises(error, match=match):
        RankSpec.parse(spec, label="outputs['x']")


### FieldSchema.parse -----------------------------------------------------------


def test_parse_treats_nested_and_dotted_names_alike_and_keeps_order():
    stress = RankSpec(rank=2, symmetric=True)
    nested = {
        "surface": {"pressure": {"rank": 0}, "shear": {"rank": 1}},
        "stress": stress,
        "vorticity": {"rank": 1, "parity": "odd"},
    }
    dotted = {
        "surface.pressure": {"rank": 0},
        "surface.shear": RankSpec(1),
        "stress": {"rank": 2, "symmetric": True},
        "vorticity": RankSpec(rank=1, parity="odd"),
    }
    expected = {
        "surface.pressure": RankSpec(rank=0),
        "surface.shear": RankSpec(rank=1),
        "stress": stress,
        "vorticity": RankSpec(rank=1, parity="odd"),
    }
    schema = FieldSchema.parse(nested)
    assert dict(schema) == expected
    assert list(schema) == list(expected)
    assert FieldSchema.parse(dotted) == schema
    assert schema.ranks == {
        "surface.pressure": 0,
        "surface.shear": 1,
        "stress": 2,
        "vorticity": 1,
    }
    assert (schema.count(0), schema.count(1), schema.count(2)) == (1, 2, 1)
    assert schema["stress"] is stress
    assert "surface.shear" in schema and "surface" not in schema
    assert len(schema) == 4


def test_parse_is_idempotent_and_direct_construction_matches():
    schema = FieldSchema.parse({"pressure": {"rank": 0}})
    assert FieldSchema.parse(schema) is schema
    assert FieldSchema({"pressure": RankSpec(0)}) == schema
    assert FieldSchema.parse({}) == FieldSchema({})
    assert repr(schema) == "FieldSchema({'pressure': RankSpec(rank=0)})"


def test_parse_rejects_integer_leaves():
    with pytest.raises(TypeError, match=r"outputs\['pressure'\] must be a RankSpec"):
        FieldSchema.parse({"pressure": 0}, label="outputs")
    ### Nested, a mapping of bare values reads as a field with unknown keys.
    with pytest.raises(
        ValueError,
        match=r"outputs\['fluid'\] has unknown keys \['velocity'\].*\{'rank': 0\}",
    ):
        FieldSchema.parse({"fluid": {"velocity": 1}}, label="outputs")


def test_fields_and_groups_are_told_apart_by_their_values():
    ### Field attribute names are not reserved: a nested field may be called
    ### "rank", "symmetric" or "parity".
    schema = FieldSchema.parse(
        {"stats": {"rank": {"rank": 0}, "parity": {"rank": 1}, "symmetric": {}}}
    )
    assert schema.ranks == {"stats.rank": 0, "stats.parity": 1}
    ### A field that forgets "rank" is reported as that field.
    with pytest.raises(ValueError, match=r"outputs\['stress'\] mapping must contain"):
        FieldSchema.parse({"stress": {"symmetric": True}}, label="outputs")
    with pytest.raises(ValueError, match=r"outputs\['w'\] has unknown keys \['rnak'\]"):
        FieldSchema.parse({"w": {"rnak": 1}}, label="outputs")
    with pytest.raises(
        ValueError, match=r"outputs\['surface'\] mixes field attributes"
    ):
        FieldSchema.parse(
            {"surface": {"rank": 0, "pressure": {"rank": 0}}}, label="outputs"
        )
    assert FieldSchema.parse({"no_slip": {}}) == FieldSchema({})


def test_parse_rejects_group_leaf_conflicts_and_bad_names():
    with pytest.raises(ValueError, match="'surface' is a group of 'surface.pressure'"):
        FieldSchema.parse({"surface": {"rank": 0}, "surface.pressure": {"rank": 0}})
    with pytest.raises(ValueError, match="is a group of"):
        FieldSchema.parse(
            {
                "surface": {"pressure": {"mean": {"rank": 0}}},
                "surface.pressure": {"rank": 0},
            }
        )
    with pytest.raises(ValueError, match="more than once"):
        FieldSchema.parse(
            {"surface": {"pressure": {"rank": 0}}, "surface.pressure": {"rank": 1}}
        )
    with pytest.raises(ValueError, match="empty path component"):
        FieldSchema.parse({"surface..pressure": {"rank": 0}})
    with pytest.raises(TypeError, match="must be strings"):
        FieldSchema.parse({1: {"rank": 0}})
    with pytest.raises(TypeError, match="fields must be a mapping"):
        FieldSchema.parse([("pressure", {"rank": 0})])
    with pytest.raises(ValueError, match=r"outputs\['shear'\]: parity must be"):
        FieldSchema.parse({"shear": {"rank": 1, "parity": "axial"}}, label="outputs")


def test_conflicts_are_reported_in_declaration_order():
    ### Deterministic messages: the first offending declaration is reported,
    ### independent of hash seeding.
    spec = {name: {"rank": 0} for name in ("c", "c.z", "a", "a.x", "b", "b.y")}
    with pytest.raises(ValueError, match="'c' is a group of 'c.z'"):
        FieldSchema.parse(spec)


def test_direct_construction_is_validated_too():
    with pytest.raises(TypeError, match="must be a RankSpec.*use FieldSchema.parse"):
        FieldSchema({"pressure": {"rank": 0}})
    with pytest.raises(ValueError, match="is a group of"):
        FieldSchema({"a": RankSpec(0), "a.b": RankSpec(0)})
    with pytest.raises(TypeError, match="fields must be a mapping"):
        FieldSchema([("a", RankSpec(0))])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.__setitem__("velocity", RankSpec(1)),
        lambda d: d.__delitem__(next(iter(d))),
        lambda d: d.update(velocity=RankSpec(1)),
        lambda d: d.setdefault("velocity", RankSpec(1)),
        lambda d: d.pop(next(iter(d))),
        lambda d: d.popitem(),
        lambda d: d.clear(),
        lambda d: d.__ior__({"velocity": RankSpec(1)}),
    ],
)
def test_schema_and_spec_are_read_only(mutate):
    schema = FieldSchema.parse({"pressure": {"rank": 0}})
    for obj in (schema, schema["pressure"]):
        with pytest.raises(TypeError, match="is immutable"):
            mutate(obj)
    assert schema == {"pressure": RankSpec(0)}


def test_schema_round_trips_through_json_pickle_and_copy():
    ### Model checkpoints store constructor arguments as JSON, so a schema (or
    ### a spec) passed to a constructor must serialize as it stands and parse
    ### back to an equal schema.
    schema = FieldSchema.parse(
        {
            "surface": {"pressure": {"rank": 0}},
            "stress": RankSpec(2, symmetric=True),
            "vorticity": {"rank": 1, "parity": "odd"},
        }
    )
    text = json.dumps(schema)
    assert json.loads(text) == {
        "surface.pressure": {"rank": 0},
        "stress": {"rank": 2, "symmetric": True},
        "vorticity": {"rank": 1, "parity": "odd"},
    }
    assert FieldSchema.parse(json.loads(text)) == schema
    pickled = pickle.loads(pickle.dumps(schema))  # noqa: S301 - local round trip
    for clone in (pickled, copy.deepcopy(schema)):
        assert clone == schema and type(clone) is FieldSchema
        assert all(type(spec) is RankSpec for spec in clone.values())


### TensorDict interplay --------------------------------------------------------


def test_key_maps_dotted_names_to_tensordict_keys():
    assert FieldSchema.key("pressure") == "pressure"
    assert FieldSchema.key("state.temperature") == ("state", "temperature")
    data = _mixed_fields()
    schema = FieldSchema.parse({"state.temperature": {"rank": 0}, "wind": {"rank": 1}})
    for name in schema:
        assert data[FieldSchema.key(name)].shape[0] == 4


def test_from_tensordict_reads_ranks_from_leaf_shapes():
    schema = FieldSchema.from_tensordict(_mixed_fields())
    assert schema.ranks == {
        "wind": 1,
        "state.temperature": 0,
        "displacement": 1,
        "pressure": 0,
        "stress": 2,
    }
    batched = TensorDict({"velocity": torch.zeros(2, 5, 3)}, batch_size=[2, 5])
    assert FieldSchema.from_tensordict(batched).ranks == {"velocity": 1}


def test_check_accepts_supersets_and_reports_schema_errors():
    schema = FieldSchema.parse(
        {"pressure": {"rank": 0}, "state": {"temperature": {"rank": 0}}}
    )
    schema.check(_mixed_fields(), label="data")  # extra leaves are fine
    ### ... whatever their names: undeclared leaves are never parsed as fields.
    odd_names = TensorDict(
        {k: torch.zeros(3) for k in ("pressure", "a", "a.b", "trailing.")},
        batch_size=[3],
    )
    FieldSchema.parse({"pressure": {"rank": 0}}).check(odd_names, label="data")

    data = TensorDict({"pressure": torch.zeros(3, 2)}, batch_size=[3])
    declared = FieldSchema.parse(
        {"pressure": {"rank": 0}, "velocity": {"rank": 1}, "fluid.T": {"rank": 0}}
    )
    with pytest.raises(ValueError) as error:
        declared.check(data, label="boundary data")
    assert str(error.value) == (
        "boundary data does not contain its declared fields:\n"
        "  - missing field 'velocity' (declared rank 1)\n"
        "  - missing field 'fluid.T' (declared rank 0)\n"
        "  - rank mismatch for 'pressure': declared 0, got 1"
    )


### Names removed in 2.3 --------------------------------------------------------


@pytest.mark.parametrize("module", ["physicsnemo.mesh", "physicsnemo.mesh.fields"])
@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("RankSpecDict", r"FieldSchemaLike"),
        ("flatten_rank_spec", r"FieldSchema\.parse\(spec\)\.ranks"),
        ("rank_counts", r"FieldSchema\.parse\(spec\)\.count\(rank\)"),
        ("ranks_from_tensordict", r"FieldSchema\.from_tensordict\(data\)\.ranks"),
        ("validate_data_contains_ranks", r"\.check\(data, label=source_label\)"),
    ],
)
def test_removed_names_point_to_their_replacements(module, name, replacement):
    with pytest.raises(
        ImportError,
        match=rf"{module}\.{name} was removed in PhysicsNeMo 2\.3.*{replacement}",
    ):
        getattr(importlib.import_module(module), name)


def test_removed_names_keep_their_message_in_import_statements():
    ### An AttributeError here would be replaced by Python's bare "cannot
    ### import name"; the ImportError carries the replacement through.
    with pytest.raises(
        ImportError, match=r"flatten_rank_spec was removed.*\{'rank': n\}"
    ):
        from physicsnemo.mesh import (
            flatten_rank_spec,  # noqa: F401  # ty: ignore[unresolved-import]
        )
    with pytest.raises(ImportError, match=r"validate_data_contains_ranks was removed"):
        from physicsnemo.mesh.fields import (
            validate_data_contains_ranks,  # noqa: F401  # ty: ignore[unresolved-import]
        )
    ### Any other missing name is still an ordinary AttributeError.
    mesh = importlib.import_module("physicsnemo.mesh")
    assert not hasattr(mesh, "not_a_mesh_name")
    with pytest.raises(AttributeError, match="has no attribute 'not_a_mesh_name'"):
        getattr(mesh, "not_a_mesh_name")
