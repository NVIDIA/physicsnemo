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

"""Training on several prediction point sets per sample.

One sample carries the volume interior and a surface point set
(``boundaries.surface_queries``), each with its own targets, addressed by
``target_sources``. Branch models map a tuple of outputs onto them with
``output_groups``; single-decoder models predict on the merged points
(``concat_point_sets``) and ``output_sets`` splits the output back.
"""

from __future__ import annotations

from types import SimpleNamespace

import hydra
import pytest
import torch
from collate import build_collate_fn
from datasets import (
    ChainedNormalizer,
    apply_model_config,
    build_dataset,
    check_target_sources,
    find_normalizer,
)
from forces import ForceContext, surface_force_fields
from forward_kwargs import (
    extract_target_measure,
    extract_targets,
    resolve_forward_kwargs,
)
from infer import attach_and_save, build_redim_field_types, redimensionalize
from loss import LossCalculator
from metrics import MetricCalculator
from nondim import NonDimensionalizeByMetadata
from omegaconf import OmegaConf
from output_normalize import (
    check_output_sets,
    normalize_output_to_tensordict,
    resolve_output_layout,
)
from tensordict import TensorDict
from utils import by_source_fields, resolve_field_weights

from physicsnemo.datapipes.transforms.mesh import MeshTransform, NormalizeMeshFields
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import set_point_measures

N_VOLUME, N_SURFACE = 10, 6
TARGETS = {
    "volume.pressure": "scalar",
    "volume.velocity": "vector",
    "surface.pressure": "scalar",
    "surface.wss": "vector",
}
SOURCES = {
    "volume.pressure": "interior.point_data.pressure",
    "volume.velocity": "interior.point_data.velocity",
    "surface.pressure": "boundaries.surface_queries.point_data.pressure",
    "surface.wss": "boundaries.surface_queries.point_data.wss",
}
MERGED = ["interior", "boundaries.surface_queries"]


def _domain() -> DomainMesh:
    """A volume interior plus a surface point set at the vehicle cell centroids."""
    torch.manual_seed(0)
    vehicle = Mesh(
        points=torch.randn(12, 3),
        cells=torch.randint(0, 12, (N_SURFACE, 3)),
    )
    interior = Mesh(
        points=torch.randn(N_VOLUME, 3),
        point_data={
            "pressure": torch.randn(N_VOLUME),
            "velocity": torch.randn(N_VOLUME, 3),
            "sdf": torch.randn(N_VOLUME),
        },
    )
    surface = Mesh(
        points=vehicle.cell_centroids,
        point_data={
            "pressure": torch.randn(N_SURFACE),
            "wss": torch.randn(N_SURFACE, 3),
            "sdf": torch.zeros(N_SURFACE),
        },
    )
    return DomainMesh(
        interior=interior,
        boundaries={"vehicle": vehicle, "surface_queries": surface},
        global_data={"U_inf": torch.tensor([30.0, 0.0, 0.0])},
    )


class _Pointwise(torch.nn.Module):
    """A pointwise mesh-in/mesh-out decoder: ``(pressure, vec)`` per point."""

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(3, 4)

    def forward(self, prediction_points: Mesh) -> Mesh:
        out = self.head(prediction_points.points)
        return Mesh(
            points=prediction_points.points,
            point_data={"pressure": out[..., 0], "vec": out[..., 1:]},
        )


class _TwoBranch(torch.nn.Module):
    """A branch model returning ``(surface, volume)`` channel tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.surface = torch.nn.Linear(3, 4)
        self.volume = torch.nn.Linear(3, 4)

    def forward(self, surface_points, volume_points):
        return self.surface(surface_points), self.volume(volume_points)


class TestTargetSources:
    """Targets on point sets of different sizes."""

    def test_extract_targets_keeps_per_set_sizes(self):
        """Each target is read from its own set and keeps that set's size."""
        domain = _domain()
        targets = extract_targets(domain, TARGETS, SOURCES)
        assert targets.batch_size == torch.Size([])
        assert targets["volume", "velocity"].shape == (N_VOLUME, 3)
        assert targets["surface", "pressure"].shape == (N_SURFACE,)
        torch.testing.assert_close(
            targets["surface", "wss"],
            domain.boundaries["surface_queries"].point_data["wss"],
        )
        with pytest.raises(ValueError, match="exactly match"):
            extract_targets(
                domain, TARGETS, {"volume.pressure": SOURCES["volume.pressure"]}
            )

    @pytest.mark.parametrize(
        "path", ["boundaries.vehicle.cell_data.pressure", ".point_data.p", "interior"]
    )
    def test_malformed_source_paths_raise(self, path):
        """Only ``<mesh path>.point_data.<field>`` paths are accepted."""
        check_target_sources(SOURCES)
        with pytest.raises(ValueError, match="<mesh path>.point_data.<field>"):
            check_target_sources({**SOURCES, "surface.pressure": path})

    @pytest.mark.parametrize("input_type", ["mesh", "tensors"])
    def test_collate_batches_unequal_sets(self, input_type):
        """Tensor mode adds a root batch dim of 1; leaves keep their set sizes."""
        collate = build_collate_fn(
            input_type, {"points": "interior.points"}, TARGETS, SOURCES
        )
        targets = collate([(_domain(), {})])["targets"]
        lead = (1,) if input_type == "tensors" else ()
        assert targets.batch_size == torch.Size(lead)
        assert targets["volume", "pressure"].shape == (*lead, N_VOLUME)
        assert targets["surface", "wss"].shape == (*lead, N_SURFACE, 3)

    @pytest.mark.parametrize("input_type", ["mesh", "tensors"])
    def test_each_target_takes_its_own_sets_measure(self, input_type):
        """Targets on a set that stores a measure are weighted by it; the
        others count every point alike."""
        domain = _domain()
        areas = torch.rand(N_SURFACE) + 0.1
        set_point_measures(domain.boundaries["surface_queries"], areas, dimension=2)
        collate = build_collate_fn(
            input_type, {"points": "interior.points"}, TARGETS, SOURCES
        )
        batch = collate([(domain, {})])
        lead = (1,) if input_type == "tensors" else ()
        measure = batch["target_measure"]
        assert set(measure.keys(True, True)) == {
            ("surface", "pressure"),
            ("surface", "wss"),
        }
        torch.testing.assert_close(measure["surface", "wss"], areas.reshape(*lead, -1))

        targets = batch["targets"]
        _, losses = LossCalculator(TARGETS, loss_type="mse")(
            targets.apply(torch.zeros_like), targets, measure
        )
        sp, vp = targets["surface", "pressure"], targets["volume", "pressure"]
        torch.testing.assert_close(
            losses["loss/surface.pressure"], (areas * sp**2).sum() / areas.sum()
        )
        torch.testing.assert_close(losses["loss/volume.pressure"], (vp**2).mean())
        ### No set stores a measure: the batch carries none.
        assert extract_target_measure(_domain(), SOURCES) is None
        assert "target_measure" not in collate([(_domain(), {})])


class TestOutputGroups:
    """A branch model's tuple of outputs, one per group of targets."""

    def test_branch_model_loss_and_metrics(self):
        """Each output tensor is split by its group; loss and metrics are finite."""
        cfg = OmegaConf.create(
            {
                "output_groups": [
                    ["surface.pressure", "surface.wss"],
                    ["volume.pressure", "volume.velocity"],
                ]
            }
        )
        spec = {
            "surface_points": "boundaries.surface_queries.points",
            "volume_points": "interior.points",
        }
        batch = build_collate_fn("tensors", spec, TARGETS, SOURCES)([(_domain(), {})])
        output = _TwoBranch()(**batch["forward_kwargs"])
        pred = normalize_output_to_tensordict(
            output, TARGETS, "tensors", **resolve_output_layout(cfg)
        )
        assert pred["surface", "wss"].shape == (1, N_SURFACE, 3)
        assert pred["volume", "pressure"].shape == (1, N_VOLUME)
        torch.testing.assert_close(pred["surface", "pressure"], output[0][..., 0])

        loss, loss_td = LossCalculator(TARGETS)(pred, batch["targets"])
        assert torch.isfinite(loss)
        assert set(loss_td.keys()) == {f"loss/{n}" for n in TARGETS} | {"loss/total"}
        metrics = MetricCalculator(TARGETS)
        metric_td = metrics(pred, batch["targets"])
        assert set(metric_td.keys()) == set(metrics.expected_keys())
        assert all(torch.isfinite(v) for v in metric_td.values())

    def test_several_outputs_need_output_groups(self):
        """Two non-None tensors and no ``output_groups`` raise."""
        output = (torch.randn(1, 4, 4), torch.randn(1, 4, 4))
        with pytest.raises(ValueError, match="output_groups"):
            normalize_output_to_tensordict(output, TARGETS, "tensors")

    def test_layout_keys_are_exclusive(self):
        """``output_groups`` and ``output_sets`` cannot both be set."""
        cfg = OmegaConf.create({"output_groups": [["a"]], "output_sets": {"x": {}}})
        with pytest.raises(ValueError, match="mutually exclusive"):
            resolve_output_layout(cfg)


class TestOutputSets:
    """A single decoder over the merged points of several sets."""

    def test_concat_point_sets_merges_points_and_shared_fields(self):
        """The merged Mesh holds every set's points, and the fields all sets hold."""
        domain = _domain()
        merged = resolve_forward_kwargs({"x": {"concat_point_sets": MERGED}}, domain)[
            "x"
        ]
        assert merged.n_points == N_VOLUME + N_SURFACE
        assert {"pressure", "sdf"} <= set(merged.point_data.keys())
        assert "velocity" not in merged.point_data.keys()
        torch.testing.assert_close(
            merged.points[N_VOLUME:], domain.boundaries["surface_queries"].points
        )
        with pytest.raises(ValueError, match="no other keys"):
            resolve_forward_kwargs(
                {"x": {"concat_point_sets": MERGED, "select": ["pressure"]}}, domain
            )

    OUTPUT_SETS = {
        "interior": {"volume.pressure": "pressure", "volume.velocity": "vec"},
        "boundaries.surface_queries": {
            "surface.pressure": "pressure",
            "surface.wss": "vec",
        },
    }

    def _split(self, domain, targets=TARGETS, sources=SOURCES):
        """Collate *domain*, run the merged model and split its output."""
        spec = {"prediction_points": {"concat_point_sets": MERGED}}
        batch = build_collate_fn("mesh", spec, targets, sources)([(domain, {})])
        model = _Pointwise()
        output = model(**batch["forward_kwargs"])
        layout = resolve_output_layout(
            OmegaConf.create({"output_sets": self.OUTPUT_SETS})
        )
        pred = normalize_output_to_tensordict(
            output, targets, "mesh", targets=batch["targets"], **layout
        )
        return model, batch, pred

    def test_split_predictions_equal_per_set_calls(self):
        """The split merged prediction equals calling the model on each set."""
        domain = _domain()
        model, batch, pred = self._split(domain)
        for path, prefix, vector in (
            ("interior", "volume", "velocity"),
            ("boundaries.surface_queries", "surface", "wss"),
        ):
            points = resolve_forward_kwargs({"p": f"{path}.points"}, domain)["p"]
            alone = model.head(points)
            torch.testing.assert_close(pred[prefix, "pressure"], alone[..., 0])
            torch.testing.assert_close(pred[prefix, vector], alone[..., 1:])
        loss, _ = LossCalculator(TARGETS)(pred, batch["targets"])
        assert torch.isfinite(loss)

    def test_set_sizes_from_scalar_targets_stored_n_by_1(self):
        """A scalar target stored ``(N, 1)`` still sizes its set by points."""
        domain = _domain()
        for mesh in (domain.interior, domain.boundaries["surface_queries"]):
            mesh.point_data["pressure"] = mesh.point_data["pressure"].unsqueeze(-1)
        _, _, pred = self._split(domain)
        assert pred["volume", "pressure"].shape == (N_VOLUME,)
        assert pred["surface", "wss"].shape == (N_SURFACE, 3)

    def test_output_sets_needs_mesh_output(self):
        """Tensor outputs are refused."""
        layout = resolve_output_layout(
            OmegaConf.create({"output_sets": self.OUTPUT_SETS})
        )
        with pytest.raises(TypeError, match="output_type: mesh"):
            normalize_output_to_tensordict(
                torch.randn(1, N_VOLUME + N_SURFACE, 4), TARGETS, "tensors", **layout
            )

    def test_output_sets_keys_must_match_sources(self):
        """Keys in the wrong order, or several sets without sources, raise."""
        check_output_sets(self.OUTPUT_SETS, SOURCES)
        swapped = dict(reversed(list(self.OUTPUT_SETS.items())))
        swapped = dict(zip(self.OUTPUT_SETS, swapped.values()))
        with pytest.raises(
            ValueError, match="do not start with 'interior.point_data.'"
        ):
            check_output_sets(swapped, SOURCES)
        with pytest.raises(ValueError, match="needs target_sources"):
            check_output_sets(self.OUTPUT_SETS, None)

    def test_concat_order_must_match_output_sets(self):
        """Every ``concat_point_sets`` list must merge the sets in key order."""
        spec = {"prediction_points": {"concat_point_sets": MERGED}}
        check_output_sets(self.OUTPUT_SETS, SOURCES, spec)
        spec = {"prediction_points": {"concat_point_sets": MERGED[::-1]}}
        with pytest.raises(ValueError, match="in that order"):
            check_output_sets(self.OUTPUT_SETS, SOURCES, spec)
        nested = {"k": [{"src": {"concat_point_sets": MERGED[::-1]}}]}
        with pytest.raises(ValueError, match="in that order"):
            check_output_sets(self.OUTPUT_SETS, SOURCES, nested)
        ### An input-side merge of other sets is not the prediction set.
        inputs = {"context": {"concat_point_sets": ["boundaries.vehicle", "interior"]}}
        check_output_sets(self.OUTPUT_SETS, SOURCES, inputs)


class _AddSurfaceQueries(MeshTransform):
    """Add the vehicle's cell centroids, with its cell data, as a point set."""

    def __call__(self, mesh: Mesh) -> Mesh:
        return mesh

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        vehicle = domain.boundaries["vehicle"]
        queries = Mesh(
            points=vehicle.cell_centroids, point_data=vehicle.cell_data.clone()
        )
        boundaries = {n: domain.boundaries[n] for n in domain.boundary_names}
        return DomainMesh(
            interior=domain.interior,
            boundaries={**boundaries, "surface_queries": queries},
            global_data=domain.global_data,
        )


class TestExtraTask:
    """A model config's ``extra_task`` adds a surface set to a volume dataset."""

    DATASET = {
        "pipeline": {
            "reader": {"_target_": "unused"},
            "transforms": [
                {
                    "_target_": "${dp:NormalizeMeshFields}",
                    "association": "point_data",
                    "fields": {"pressure": {"type": "scalar", "mean": 2.0, "std": 4.0}},
                }
            ],
        },
        "targets": {"pressure": "scalar"},
    }
    MODEL = {
        "dataset_append_transforms": [{"_target_": "${dp:CenterMesh}"}],
        "training": {"field_weights": {"pressure": 0.5}},
        "extra_task": {
            "transforms": [
                {"_target_": f"{__name__}._AddSurfaceQueries"},
                {
                    "_target_": "${dp:NormalizeMeshFields}",
                    "association": "point_data",
                    "fields": {"surface_p": "pressure"},
                },
            ],
            "targets": {"surface_p": "scalar"},
            "target_sources": {
                "surface_p": "boundaries.surface_queries.point_data.surface_p"
            },
            "field_weights": {"surface_p": 3.0},
        },
    }

    @staticmethod
    def _volume_domain() -> tuple[DomainMesh, torch.Tensor, torch.Tensor]:
        """A volume sample whose vehicle cells carry the extra set's field."""
        torch.manual_seed(0)
        vehicle = Mesh(
            points=torch.randn(12, 3),
            cells=torch.randint(0, 12, (N_SURFACE, 3)),
            cell_data={"surface_p": torch.randn(N_SURFACE)},
        )
        interior = Mesh(
            points=torch.randn(N_VOLUME, 3),
            point_data={"pressure": torch.randn(N_VOLUME)},
        )
        domain = DomainMesh(interior=interior, boundaries={"vehicle": vehicle})
        return domain, interior.point_data["pressure"], vehicle.cell_data["surface_p"]

    @staticmethod
    def _sources(ds) -> dict[str, str]:
        """The dataset's own target defaults to the interior."""
        return {
            "pressure": "interior.point_data.pressure",
            **OmegaConf.to_container(ds.target_sources),
        }

    def test_extra_task_adds_set_targets_weights_and_shared_constants(self):
        """Targets, sources, weights and normalizer constants join the dataset's."""
        ds = apply_model_config(
            OmegaConf.create(self.DATASET), OmegaConf.create(self.MODEL)
        )
        assert dict(ds.targets) == {"pressure": "scalar", "surface_p": "scalar"}
        assert [t._target_.rsplit(".", 1)[-1] for t in ds.pipeline.transforms] == [
            "NormalizeMeshFields",
            "CenterMesh",
            "_AddSurfaceQueries",
            "NormalizeMeshFields",
        ]
        shared = OmegaConf.to_container(ds.pipeline.transforms[-1].fields)
        assert shared == {"surface_p": {"type": "scalar", "mean": 2.0, "std": 4.0}}

        domain, _, raw_sp = self._volume_domain()
        for t_cfg in ds.pipeline.transforms:
            domain = hydra.utils.instantiate(t_cfg).apply_to_domain(domain)
        targets = extract_targets(domain, dict(ds.targets), self._sources(ds))
        assert targets["surface_p"].shape == (N_SURFACE,)
        torch.testing.assert_close(targets["surface_p"], (raw_sp - 2.0) / (4.0 + 1e-8))

        weights = resolve_field_weights(OmegaConf.create(self.MODEL))
        assert weights == {"pressure": 0.5, "surface_p": 3.0}
        pred = targets.apply(torch.zeros_like)
        _, losses = LossCalculator(
            dict(ds.targets), loss_type="mse", field_weights=weights
        )(pred, targets)
        torch.testing.assert_close(
            losses["loss/surface_p"], 3.0 * targets["surface_p"].pow(2).mean()
        )

    def test_extra_set_field_exported_in_physical_units(self, tmp_path):
        """Every NormalizeMeshFields is inverted, the extra task's included."""
        ds = apply_model_config(
            OmegaConf.create(self.DATASET), OmegaConf.create(self.MODEL)
        )
        transforms = [hydra.utils.instantiate(t) for t in ds.pipeline.transforms]
        normalizer = find_normalizer([SimpleNamespace(transforms=transforms)])
        assert isinstance(normalizer, ChainedNormalizer)
        assert set(normalizer.stats) == {"pressure", "surface_p"}
        ### One normalizer stays the NormalizeMeshFields itself, as on main.
        single = find_normalizer([SimpleNamespace(transforms=transforms[:1])])
        assert single is transforms[0]

        domain, raw_p, raw_sp = self._volume_domain()
        raw_p, raw_sp = raw_p.clone(), raw_sp.clone()
        for t in transforms:
            domain = t.apply_to_domain(domain)
        sources = self._sources(ds)
        targets = extract_targets(domain, dict(ds.targets), sources)
        physical = redimensionalize(
            targets,
            normalizer=normalizer,
            nondim=None,
            field_types={},
            global_data=domain.global_data,
        )
        out = tmp_path / "sample.pdmsh"
        attach_and_save(
            domain,
            physical,
            physical,
            dict(ds.targets),
            out,
            rescale_geometry=False,
            target_sources=sources,
        )
        saved = DomainMesh.load(str(out))
        torch.testing.assert_close(
            saved.boundaries["surface_queries"].point_data["true_surface_p"], raw_sp
        )
        torch.testing.assert_close(saved.interior.point_data["pred_pressure"], raw_p)

    def test_only_interior_targets_injected_into_mesh_to_domain_mesh(self, tmp_path):
        """A target sourced from another set is not looked up on the interior."""
        (tmp_path / "run_0").mkdir()
        (tmp_path / "run_0" / "domain_0.pdmsh").write_bytes(b"")
        cfg = OmegaConf.create(
            {
                "pipeline": {
                    "reader": {
                        "_target_": "${dp:DomainMeshReader}",
                        "path": str(tmp_path),
                        "pattern": "run_*/domain_*.pdmsh",
                    },
                    "transforms": [{"_target_": "${dp:MeshToDomainMesh}"}],
                },
                "targets": {
                    "volume.pressure": "scalar",
                    "velocity": "vector",
                    "surface.pressure": "scalar",
                },
                "target_sources": {
                    "volume.pressure": SOURCES["volume.pressure"],
                    "surface.pressure": SOURCES["surface.pressure"],
                },
            }
        )
        dataset = build_dataset(cfg, device=None)
        ### Under the source field names, as the mesh stores them.
        assert dataset.transforms[0]._cell_data_targets == ["pressure", "velocity"]

    def test_unknown_shared_constants_raise(self):
        """Naming a field the dataset normalizer does not hold is refused."""
        model = OmegaConf.create(self.MODEL)
        model.extra_task.transforms[1].fields = {"surface_p": "velocity"}
        with pytest.raises(ValueError, match="velocity"):
            apply_model_config(OmegaConf.create(self.DATASET), model)

    def test_shared_constants_need_inline_fields(self):
        """A dataset normalizer reading a ``stats_file`` cannot share constants."""
        dataset = OmegaConf.create(self.DATASET)
        normalize = dataset.pipeline.transforms[0]
        del normalize["fields"]
        normalize.stats_file = "norm_stats.pt"
        with pytest.raises(ValueError, match="inline, not a stats_file"):
            apply_model_config(dataset, OmegaConf.create(self.MODEL))

    def test_chained_normalizers_refuse_a_repeated_field(self):
        """Two normalizers of one field would share one entry of the stats."""
        stats = {"pressure": {"type": "scalar", "mean": 0.0, "std": 1.0}}
        with pytest.raises(ValueError, match="normalized once"):
            ChainedNormalizer([NormalizeMeshFields(fields=stats)] * 2)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("loss_type", ["huber", "mse", "relative_mse"])
def test_masked_union_set_equals_split_sets(loss_type, weighted):
    """A NaN-padded union set with masking scores like the two sets apart,
    with or without each set's measure."""
    torch.manual_seed(0)
    n_a, n_b = 7, 5
    target_config = {"a": "scalar", "b": "vector"}
    a_true, a_pred = torch.randn(n_a), torch.randn(n_a)
    b_true, b_pred = torch.randn(n_b, 3), torch.randn(n_b, 3)
    split_pred = TensorDict({"a": a_pred, "b": b_pred})
    split_true = TensorDict({"a": a_true, "b": b_true})

    nan = float("nan")
    union_true = TensorDict(
        {
            "a": torch.cat([a_true, torch.full((n_b,), nan)]),
            "b": torch.cat([torch.full((n_a, 3), nan), b_true]),
        },
        batch_size=[n_a + n_b],
    )
    union_pred = TensorDict(
        {
            "a": torch.cat([a_pred, torch.randn(n_b)]),
            "b": torch.cat([torch.randn(n_a, 3), b_pred]),
        },
        batch_size=[n_a + n_b],
    )
    ### Per-field measures apart; one shared measure on the union.
    split_measure = union_measure = None
    if weighted:
        split_measure = TensorDict({"a": torch.rand(n_a), "b": torch.rand(n_b)}) + 0.1
        union_measure = torch.cat([split_measure["a"], split_measure["b"]])
    split_loss, split_td = LossCalculator(target_config, loss_type=loss_type)(
        split_pred, split_true, split_measure
    )
    masked = LossCalculator(
        target_config, loss_type=loss_type, mask_nonfinite_targets=True
    )
    union_loss, union_td = masked(union_pred, union_true, union_measure)
    torch.testing.assert_close(union_loss, split_loss)
    for key in split_td.keys():
        torch.testing.assert_close(union_td[key], split_td[key])

    split_metrics = MetricCalculator(target_config)(
        split_pred, split_true, split_measure
    )
    union_metrics = MetricCalculator(target_config, mask_nonfinite_targets=True)(
        union_pred, union_true, union_measure
    )
    for key in split_metrics.keys():
        torch.testing.assert_close(union_metrics[key], split_metrics[key])

    ### A field with no finite target adds 0 to the loss, and has NaN metrics.
    empty = union_true.clone()
    empty["a"] = torch.full_like(empty["a"], nan)
    _, empty_td = masked(union_pred, empty, union_measure)
    assert empty_td["loss/a"] == 0
    empty_metrics = MetricCalculator(target_config, mask_nonfinite_targets=True)(
        union_pred, empty, union_measure
    )
    assert all(empty_metrics[k].isnan() for k in empty_metrics.keys() if "a_" in k)
    torch.testing.assert_close(empty_metrics["b_l2"], split_metrics["b_l2"])


class TestForcesAndExport:
    """Forces and export follow each target's source set."""

    ### As real runs build them: keyed by the stored field names.
    NONDIM = {"pressure": "pressure", "velocity": "velocity", "wss": "stress"}
    FIELD_TYPES = build_redim_field_types(
        OmegaConf.create(
            {
                "pipeline": {
                    "transforms": [
                        {"_target_": "NonDimensionalizeByMetadata", "fields": NONDIM}
                    ]
                }
            }
        )
    )
    STATS = {
        "pressure": {"type": "scalar", "mean": 2.0, "std": 4.0},
        "velocity": {"type": "vector", "mean": [1.0, 0.0, 0.0], "std": 3.0},
        "wss": {"type": "vector", "mean": [0.0, 0.0, 0.0], "std": 0.5},
    }

    def test_forces_on_surface_set_aligned_with_vehicle(self):
        """Forces integrate the surface set's targets; a misaligned set is skipped."""
        domain = _domain()
        ctx = ForceContext.from_config(
            OmegaConf.create({"enabled": True}),
            self.FIELD_TYPES,
            "cpu",
            target_sources=SOURCES,
        )
        assert (ctx.pressure_field, ctx.shear_field) == (
            "surface.pressure",
            "surface.wss",
        )
        pts = extract_targets(domain, TARGETS, SOURCES)
        forces = ctx.coefficients(domain, pts, pts, None, SOURCES)
        assert forces is not None and forces[0] == forces[1]
        ### Without target_sources, the volume interior is no surface case.
        assert ctx.coefficients(domain, pts, pts, None) is None

        ### Normalized targets are un-normalized under their source names.
        normalizer = NormalizeMeshFields(fields=self.STATS)
        normed = normalizer.apply_to_domain(_domain())
        normed_pts = extract_targets(normed, TARGETS, SOURCES)
        normed_forces = ctx.coefficients(
            normed, normed_pts, normed_pts, normalizer, SOURCES
        )
        assert normed_forces[0] == pytest.approx(forces[0], rel=1e-5, abs=1e-6)

        shifted = domain.boundaries["surface_queries"].translate(torch.ones(3))
        domain.boundaries["surface_queries"] = shifted
        with pytest.warns(UserWarning, match="aligned"):
            assert ctx.coefficients(domain, pts, pts, None, SOURCES) is None

    def test_two_targets_of_one_source_field_raise(self):
        """Two targets on one mesh may not share a source field."""
        sources = {**SOURCES, "surface.wss": SOURCES["surface.pressure"]}
        with pytest.raises(ValueError):
            by_source_fields(
                extract_targets(_domain(), TARGETS, sources), sources, lambda t: t
            )

    def test_redimensionalize_under_source_field_names(self):
        """Targets named apart from their fields are inverted with those fields' rules."""
        global_data = TensorDict(
            {
                "U_inf": torch.tensor([30.0, 0.0, 0.0]),
                "p_inf": torch.tensor(10.0),
                "rho_inf": torch.tensor(1.2),
            }
        )
        raw = _domain()
        nondim = NonDimensionalizeByMetadata(fields=self.NONDIM)
        normalizer = NormalizeMeshFields(fields=self.STATS)
        domain = _domain()
        domain = DomainMesh(
            interior=domain.interior,
            boundaries={n: domain.boundaries[n] for n in domain.boundary_names},
            global_data=global_data,
        )
        domain = normalizer.apply_to_domain(nondim.apply_to_domain(domain))
        physical = redimensionalize(
            extract_targets(domain, TARGETS, SOURCES),
            normalizer=normalizer,
            nondim=nondim,
            field_types=self.FIELD_TYPES,
            global_data=global_data,
            target_sources=SOURCES,
        )
        torch.testing.assert_close(
            physical, extract_targets(raw, TARGETS, SOURCES), rtol=1e-4, atol=1e-4
        )

    def test_forces_pick_the_pair_on_one_source_mesh(self):
        """A volume pressure next to a surface set's pair integrates the surface."""
        field_types = {"pressure": "pressure", "surface_p": "pressure", "wss": "stress"}
        sources = {
            "pressure": "interior.point_data.pressure",
            "surface_p": "boundaries.surface_queries.point_data.surface_p",
            "wss": "boundaries.surface_queries.point_data.wss",
        }
        assert surface_force_fields(field_types) == ("pressure", "wss")
        assert surface_force_fields(field_types, sources) == ("surface_p", "wss")
        ctx = ForceContext.from_config(
            OmegaConf.create({"enabled": True}),
            field_types,
            "cpu",
            target_sources=sources,
        )
        assert (ctx.pressure_field, ctx.shear_field) == ("surface_p", "wss")

    def test_export_writes_onto_each_source_mesh(self, tmp_path):
        """``pred_*`` / ``true_*`` land on each target's source set."""
        domain = _domain()
        true = extract_targets(domain, TARGETS, SOURCES)
        pred = true.apply(lambda t: t + 1)
        out = tmp_path / "sample.pdmsh"
        attach_and_save(
            domain,
            pred,
            true,
            TARGETS,
            out,
            rescale_geometry=False,
            target_sources=SOURCES,
        )
        saved = DomainMesh.load(str(out))
        surface = saved.boundaries["surface_queries"].point_data
        torch.testing.assert_close(surface["pred_wss"], pred["surface", "wss"])
        torch.testing.assert_close(
            surface["true_pressure"], true["surface", "pressure"]
        )
        assert "pressure" not in surface.keys()
        torch.testing.assert_close(
            saved.interior.point_data["pred_velocity"], pred["volume", "velocity"]
        )
        assert "sdf" in saved.interior.point_data.keys()
