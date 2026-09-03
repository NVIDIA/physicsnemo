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

"""Nested-TensorDict behaviour of the datapipe transforms and collators.

A ``Mesh``'s ``point_data`` / ``cell_data`` / ``global_data`` may hold
nested sub-TensorDicts (``{"solution": {"p": ..., "wss": ...}}``). Every
config-driven field name must be able to reach such a leaf via the
``"group.leaf"`` spelling without the user flattening anything. These
tests pin that contract for each transform that addresses fields by name.
"""

import pytest
import torch
from tensordict import TensorDict

import physicsnemo.datapipes as dp
from physicsnemo.datapipes.transforms.mesh import (
    ComputeSurfaceNormals,
    DropMeshFields,
    MeshToDomainMesh,
    NormalizeMeshFields,
    RenameMeshFields,
    RestructureTensorDict,
    SetGlobalField,
)
from physicsnemo.mesh import Mesh


def _surface_mesh() -> Mesh:
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]])
    return Mesh(
        points=points,
        cells=cells,
        cell_data={
            "solution": {
                "pMeanTrim": torch.tensor([1500.0, -200.0]),
                "wssMeanTrim": torch.tensor([[5.0, -1.0, 0.5], [-3.0, 2.0, -0.25]]),
            },
            "normals": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        },
        point_data={"geom": {"sdf": torch.zeros(4)}, "id": torch.arange(4)},
    )


def _leaves(td: TensorDict) -> set:
    return set(td.keys(include_nested=True, leaves_only=True))


class TestRenameMeshFields:
    def test_nested_source_hoisted_to_top_level(self):
        mesh = _surface_mesh()
        out = RenameMeshFields(
            cell_data={"solution.pMeanTrim": "pressure", "solution.wssMeanTrim": "wss"}
        )(mesh)
        assert _leaves(out.cell_data) == {"pressure", "wss", "normals"}
        assert torch.equal(
            out.cell_data["pressure"], mesh.cell_data["solution", "pMeanTrim"]
        )
        ### Input untouched.
        assert ("solution", "pMeanTrim") in mesh.cell_data

    def test_nested_to_nested_and_top_to_nested(self):
        mesh = _surface_mesh()
        out = RenameMeshFields(
            cell_data={"solution.pMeanTrim": "solution.p", "normals": "geom.n"},
            point_data={"geom.sdf": "sdf"},
        )(mesh)
        assert ("solution", "p") in out.cell_data
        assert ("geom", "n") in out.cell_data
        assert "sdf" in out.point_data and ("geom", "sdf") not in out.point_data

    def test_missing_nested_source_is_skipped(self):
        mesh = _surface_mesh()
        out = RenameMeshFields(cell_data={"solution.nope": "x"})(mesh)
        assert _leaves(out.cell_data) == _leaves(mesh.cell_data)


class TestDropMeshFields:
    def test_nested_leaf_is_dropped(self):
        mesh = _surface_mesh()
        out = DropMeshFields(
            cell_data=["solution.wssMeanTrim"], point_data=["geom.sdf"]
        )(mesh)
        assert ("solution", "wssMeanTrim") not in out.cell_data
        assert ("solution", "pMeanTrim") in out.cell_data
        assert ("geom", "sdf") not in out.point_data

    def test_group_can_be_dropped_whole(self):
        out = DropMeshFields(cell_data=["solution"])(_surface_mesh())
        assert _leaves(out.cell_data) == {"normals"}

    def test_path_through_leaf_is_a_silent_miss_not_a_crash(self):
        ### "normals.x" descends through the leaf tensor "normals"; TensorDict's
        ### own exclude raises AttributeError on that, DropMeshFields must not.
        mesh = _surface_mesh()
        out = DropMeshFields(cell_data=["normals.x"])(mesh)
        assert _leaves(out.cell_data) == _leaves(mesh.cell_data)


class TestNormalizeMeshFields:
    def _stats(self):
        return {
            "solution.pMeanTrim": {"type": "scalar", "mean": -100.0, "std": 800.0},
            "solution.wssMeanTrim": {
                "type": "vector",
                "mean": [0.5, -0.25, 0.1],
                "std": [3.0, 2.0, 0.4],
            },
        }

    def test_forward_normalizes_nested_leaf(self):
        mesh = _surface_mesh()
        norm = NormalizeMeshFields(association="cell_data", fields=self._stats())
        out = norm(mesh)
        expected = (mesh.cell_data["solution", "pMeanTrim"] - (-100.0)) / (
            800.0 + norm._eps
        )
        assert torch.allclose(out.cell_data["solution", "pMeanTrim"], expected)
        assert torch.equal(out.cell_data["normals"], mesh.cell_data["normals"])

    def test_inverse_td_round_trip_nested(self):
        mesh = _surface_mesh()
        norm = NormalizeMeshFields(association="cell_data", fields=self._stats())
        recovered = norm.inverse_td(norm(mesh).cell_data)
        for key in (("solution", "pMeanTrim"), ("solution", "wssMeanTrim")):
            assert torch.allclose(recovered[key], mesh.cell_data[key], atol=1e-4)

    def test_inverse_td_matches_by_full_key_not_leaf_name(self):
        ### Two leaves named "p" at different depths must not share stats:
        ### only the one the forward pass normalized may be un-normalized.
        norm = NormalizeMeshFields(
            association="point_data",
            fields={"p": {"type": "scalar", "mean": 10.0, "std": 2.0}},
        )
        td = TensorDict(
            {"p": torch.zeros(3), "pred": {"p": torch.zeros(3)}}, batch_size=[3]
        )
        out = norm.inverse_td(td)
        assert torch.allclose(out["p"], torch.full((3,), 10.0))
        assert torch.equal(out["pred", "p"], torch.zeros(3))

    def test_inverse_tensor_accepts_dotted_target_config(self):
        norm = NormalizeMeshFields(association="cell_data", fields=self._stats())
        x = torch.zeros(5, 4)
        out = norm.inverse_tensor(
            x, {"solution.pMeanTrim": "scalar", "solution.wssMeanTrim": "vector"}
        )
        assert torch.allclose(out[:, 0], torch.full((5,), -100.0))
        assert torch.allclose(out[0, 1:], torch.tensor([0.5, -0.25, 0.1]))

    def test_stats_property_uses_dotted_names_and_round_trips(self, tmp_path):
        norm = NormalizeMeshFields(association="cell_data", fields=self._stats())
        assert set(norm.stats) == {"solution.pMeanTrim", "solution.wssMeanTrim"}
        path = tmp_path / "stats.pt"
        torch.save(norm.stats, path)
        reloaded = NormalizeMeshFields(association="cell_data", stats_file=str(path))
        assert (
            reloaded(_surface_mesh())
            .cell_data["solution", "pMeanTrim"]
            .allclose(norm(_surface_mesh()).cell_data["solution", "pMeanTrim"])
        )

    def test_stats_setter_replaces_live_stats(self):
        ### ``stats`` returns a copy, so mutating it is a no-op; the setter is
        ### how persisted stats get applied (infer.py relies on this).
        norm = NormalizeMeshFields(association="cell_data", fields=self._stats())
        norm.stats.clear()
        assert norm.stats  # unchanged
        norm.stats = {
            "solution.pMeanTrim": {
                "type": "scalar",
                "mean": torch.tensor(0.0),
                "std": torch.tensor(1.0),
            }
        }
        out = norm(_surface_mesh())
        assert torch.allclose(
            out.cell_data["solution", "pMeanTrim"],
            _surface_mesh().cell_data["solution", "pMeanTrim"] / (1.0 + norm._eps),
        )


class TestSetGlobalField:
    def test_nested_yaml_dict_and_dotted_name(self):
        mesh = _surface_mesh()
        out = SetGlobalField(
            fields={"flow": {"U_inf": [30.0, 0.0, 0.0]}, "ref.L": 2.0}
        )(mesh)
        assert out.global_data["flow", "U_inf"].tolist() == [30.0, 0.0, 0.0]
        assert out.global_data["ref", "L"].item() == 2.0
        assert (
            "flow.U_inf" in SetGlobalField(fields={"flow": {"U_inf": 1.0}}).extra_repr()
        )

    def test_accepts_omegaconf_containers(self):
        ### ``hydra.utils.instantiate`` passes DictConfig / ListConfig (not
        ### dict / list) unless ``_convert_`` is set; the transform must cope.
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"flow": {"U_inf": [30.0, 0.0, 0.0]}, "ref.L": 2.0})
        out = SetGlobalField(fields=cfg)(_surface_mesh())
        assert out.global_data["flow", "U_inf"].tolist() == [30.0, 0.0, 0.0]
        assert out.global_data["ref", "L"].item() == 2.0


class TestComputeSurfaceNormals:
    def test_nested_destination(self):
        out = ComputeSurfaceNormals(store_as="cell_data", field_name="geom.n")(
            _surface_mesh()
        )
        assert out.cell_data["geom", "n"].shape == (2, 3)


class TestMeshToDomainMesh:
    def test_missing_or_leaf_prefix_target_raises_key_error(self):
        with pytest.raises(KeyError, match="solution.pMeanTrim"):
            MeshToDomainMesh(cell_data_targets=["solution.nope"])(_surface_mesh())
        with pytest.raises(KeyError, match="normals.x"):
            MeshToDomainMesh(cell_data_targets=["normals.x"])(_surface_mesh())

    def test_nested_target_moved_to_interior(self):
        mesh = _surface_mesh()
        domain = MeshToDomainMesh(cell_data_targets=["solution.pMeanTrim"])(mesh)
        assert _leaves(domain.interior.point_data) == {("solution", "pMeanTrim")}
        boundary = domain.boundaries["vehicle"]
        assert ("solution", "pMeanTrim") not in boundary.cell_data
        assert ("solution", "wssMeanTrim") in boundary.cell_data
        assert "normals" in boundary.cell_data


class TestRestructureTensorDict:
    def test_nested_source_and_destination(self):
        td = TensorDict(
            {
                "point_data": {"solution": {"p": torch.zeros(3)}},
                "points": torch.zeros(3, 3),
            },
            batch_size=[],
        )
        out = RestructureTensorDict(
            groups={
                "output": {"fields.p": "point_data.solution.p"},
                "input": {"x": "points"},
            }
        )(td)
        assert ("output", "fields", "p") in out
        assert ("input", "x") in out


class TestTensorDictTransforms:
    def _td(self) -> TensorDict:
        return TensorDict(
            {
                "solution": {"p": torch.arange(6.0), "v": torch.ones(6, 3)},
                "x": torch.zeros(6, 3),
                "w": torch.ones(6),
            },
            batch_size=[],
        )

    def test_normalize_nested_with_dotted_stats(self):
        norm = dp.Normalize(
            input_keys=["solution.p"],
            method="mean_std",
            means={"solution.p": 1.0},
            stds={"solution.p": 2.0},
        )
        out = norm(self._td())
        assert torch.allclose(
            out["solution", "p"], (torch.arange(6.0) - 1.0) / (2.0 + norm.eps)
        )
        back = norm.inverse(out)
        assert torch.allclose(back["solution", "p"], torch.arange(6.0), atol=1e-5)

    def test_normalize_state_dict_round_trip_nested(self):
        norm = dp.Normalize(
            input_keys=["solution.p"], method="mean_std", means=0.0, stds=1.0
        )
        other = dp.Normalize(input_keys=["z"], method="mean_std", means=0.0, stds=1.0)
        other.load_state_dict(norm.state_dict())
        assert other.input_keys == [("solution", "p")]

    def test_concat_fields_nested(self):
        out = dp.ConcatFields(input_keys=["x", "solution.v"], output_key="feat.emb")(
            self._td()
        )
        assert out["feat", "emb"].shape == (6, 6)

    def test_subsample_points_nested(self):
        torch.manual_seed(0)
        out = dp.SubsamplePoints(
            input_keys=["x", "solution.p", "solution.v"],
            n_points=3,
            algorithm="uniform",
        )(self._td())
        assert out["x"].shape == (3, 3)
        assert out["solution", "p"].shape == (3,)
        assert out["solution", "v"].shape == (3, 3)

    def test_normalize_and_field_slice_accept_omegaconf(self):
        ### Hydra passes DictConfig / ListConfig with its default ``_convert_``.
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "input_keys": ["solution.p"],
                "means": {"solution.p": 1.0},
                "stds": {"solution.p": 2.0},
            }
        )
        norm = dp.Normalize(
            input_keys=cfg.input_keys, method="mean_std", means=cfg.means, stds=cfg.stds
        )
        out = norm(self._td())
        assert torch.allclose(
            out["solution", "p"], (torch.arange(6.0) - 1.0) / (2.0 + norm.eps)
        )

        sliced = dp.FieldSlice(
            slicing=OmegaConf.create({"solution.v": {"-1": [0, 2]}})
        )(self._td())
        assert sliced["solution", "v"].shape == (6, 2)

    def test_field_slice_nested(self):
        out = dp.FieldSlice(slicing={"solution.v": {"-1": [0, 2]}})(self._td())
        assert out["solution", "v"].shape == (6, 2)

    def test_constant_field_nested_reference_and_output(self):
        out = dp.ConstantField(reference_key="solution.p", output_key="geom.sdf")(
            self._td()
        )
        assert out["geom", "sdf"].shape == (6, 1)

    def test_translate_and_scale_nested(self):
        moved = dp.Translate(
            input_keys=["solution.v"], center_key_or_value=torch.tensor([1.0, 1.0, 1.0])
        )(self._td())
        assert torch.allclose(moved["solution", "v"], torch.full((6, 3), 2.0))
        scaled = dp.Scale(
            input_keys=["solution.v"], scale=torch.tensor([2.0, 2.0, 2.0])
        )(self._td())
        assert torch.allclose(scaled["solution", "v"], torch.full((6, 3), 2.0))

    def test_translate_with_nested_center_key(self):
        td = self._td()
        td.set(("ref", "center"), torch.tensor([1.0, 1.0, 1.0]))
        moved = dp.Translate(
            input_keys=["x"], center_key_or_value="ref.center", subtract=True
        )(td)
        assert torch.allclose(moved["x"], torch.full((6, 3), -1.0))

    def test_group_name_gives_actionable_error(self):
        with pytest.raises(TypeError, match="group of fields"):
            dp.NormalizeVectors(input_keys=["solution"])(self._td())

    def test_missing_nested_key_lists_leaves(self):
        with pytest.raises(KeyError, match="solution.v"):
            dp.NormalizeVectors(input_keys=["solution.nope"])(self._td())

    def test_purge_and_rename_still_nested_aware(self):
        td = self._td()
        kept = dp.Purge(keep_only=["solution.p"])(td)
        assert set(kept.keys(True, True)) == {("solution", "p")}
        renamed = dp.Rename(mapping={"solution.p": "pressure", "x": "pos"})(td)
        assert (
            "pressure" in renamed and "pos" in renamed and ("solution", "v") in renamed
        )
        ### A group and a key inside it cannot share one mapping (order-dependent).
        with pytest.raises(ValueError, match="lies inside"):
            dp.Rename(mapping={"solution.p": "pressure", "solution": "sol"})(td)


class TestCollators:
    def _samples(self):
        def sample(n):
            return (
                TensorDict(
                    {"points": torch.randn(n, 3), "solution": {"p": torch.randn(n)}},
                    batch_size=[],
                ),
                {},
            )

        return [sample(4), sample(6)]

    def test_concat_collator_nested(self):
        out = dp.ConcatCollator(dim=0, add_batch_idx=True)(self._samples())
        assert out["points"].shape == (10, 3)
        assert out["solution", "p"].shape == (10,)
        assert out["batch_idx"].tolist() == [0] * 4 + [1] * 6

    def test_concat_collator_dotted_keys_and_batch_idx(self):
        out = dp.ConcatCollator(
            dim=0, keys=["solution.p"], batch_idx_key="meta.batch_idx"
        )(self._samples())
        assert set(out.keys(True, True)) == {("solution", "p"), ("meta", "batch_idx")}

    def test_default_collator_bad_key_is_key_error(self):
        samples = [(TensorDict({"a": torch.zeros(2)}), {}) for _ in range(2)]
        with pytest.raises(KeyError, match="a.x"):
            dp.DefaultCollator(keys=["a.x"])(samples)

    def test_purge_non_strict_tolerates_path_through_leaf(self):
        td = TensorDict({"x": torch.zeros(6, 3), "w": torch.ones(6)}, batch_size=[])
        assert set(
            dp.Purge(keep_only=["x", "x.y"], strict=False)(td).keys(True, True)
        ) == {"x"}
        assert set(
            dp.Purge(drop_only=["x.y"], strict=False)(td).keys(True, True)
        ) == set(td.keys(True, True))

    def test_default_collator_dotted_keys(self):
        samples = [
            (TensorDict({"a": torch.zeros(2), "g": {"b": torch.ones(2)}}), {})
            for _ in range(3)
        ]
        out = dp.DefaultCollator(keys=["g.b"])(samples)
        assert set(out.keys(True, True)) == {("g", "b")}
        assert out["g", "b"].shape == (3, 2)
