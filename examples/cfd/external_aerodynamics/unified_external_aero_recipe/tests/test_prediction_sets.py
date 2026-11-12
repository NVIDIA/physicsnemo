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

import hydra
import pytest
import torch
from collate import build_collate_fn
from datasets import apply_model_config
from forces import ForceContext
from forward_kwargs import extract_targets, resolve_forward_kwargs
from infer import attach_and_save
from loss import LossCalculator
from metrics import MetricCalculator
from omegaconf import OmegaConf
from output_normalize import normalize_output_to_tensordict, resolve_output_layout
from tensordict import TensorDict
from utils import resolve_field_weights

from physicsnemo.datapipes.transforms.mesh import MeshTransform
from physicsnemo.mesh import DomainMesh, Mesh

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
    """A pointwise decoder: one ``(pressure, vector)`` prediction per point."""

    def __init__(self, mesh_output: bool) -> None:
        super().__init__()
        self.mesh_output = mesh_output
        self.head = torch.nn.Linear(3, 4)

    def forward(self, prediction_points: torch.Tensor):
        out = self.head(prediction_points)
        if not self.mesh_output:
            return out
        return Mesh(
            points=prediction_points,
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

    @pytest.mark.parametrize("output_type", ["mesh", "tensors"])
    def test_split_predictions_equal_per_set_calls(self, output_type):
        """The split merged prediction equals calling the model on each set."""
        domain = _domain()
        cfg = OmegaConf.create(
            {
                "output_sets": {
                    "interior": {
                        "volume.pressure": "pressure",
                        "volume.velocity": "vec",
                    },
                    "boundaries.surface_queries": {
                        "surface.pressure": "pressure",
                        "surface.wss": "vec",
                    },
                }
            }
        )
        spec = {"prediction_points": {"concat_point_sets": MERGED, "select": "points"}}
        batch = build_collate_fn(output_type, spec, TARGETS, SOURCES)([(domain, {})])
        model = _Pointwise(mesh_output=output_type == "mesh")
        output = model(**batch["forward_kwargs"])
        pred = normalize_output_to_tensordict(
            output,
            TARGETS,
            output_type,
            targets=batch["targets"],
            **resolve_output_layout(cfg),
        )

        lead = (1,) if output_type == "tensors" else ()
        for path, prefix in (
            ("interior", "volume"),
            ("boundaries.surface_queries", "surface"),
        ):
            points = resolve_forward_kwargs({"p": f"{path}.points"}, domain)["p"]
            alone = model.head(points.reshape(*lead, -1, 3))
            vector = "velocity" if prefix == "volume" else "wss"
            torch.testing.assert_close(pred[prefix, "pressure"], alone[..., 0])
            torch.testing.assert_close(pred[prefix, vector], alone[..., 1:])
        loss, _ = LossCalculator(TARGETS)(pred, batch["targets"])
        assert torch.isfinite(loss)


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

        ### The dataset's own targets default to the interior.
        torch.manual_seed(0)
        vehicle = Mesh(
            points=torch.randn(12, 3),
            cells=torch.randint(0, 12, (N_SURFACE, 3)),
            cell_data={"surface_p": torch.randn(N_SURFACE)},
        )
        domain = DomainMesh(
            interior=Mesh(
                points=torch.randn(N_VOLUME, 3),
                point_data={"pressure": torch.randn(N_VOLUME)},
            ),
            boundaries={"vehicle": vehicle},
        )
        raw_p = vehicle.cell_data["surface_p"].clone()
        for t_cfg in ds.pipeline.transforms:
            domain = hydra.utils.instantiate(t_cfg).apply_to_domain(domain)
        sources = {
            "pressure": "interior.point_data.pressure",
            **OmegaConf.to_container(ds.target_sources),
        }
        targets = extract_targets(domain, dict(ds.targets), sources)
        assert targets["surface_p"].shape == (N_SURFACE,)
        torch.testing.assert_close(targets["surface_p"], (raw_p - 2.0) / (4.0 + 1e-8))

        weights = resolve_field_weights(OmegaConf.create(self.MODEL))
        assert weights == {"pressure": 0.5, "surface_p": 3.0}
        pred = targets.apply(torch.zeros_like)
        _, losses = LossCalculator(
            dict(ds.targets), loss_type="mse", field_weights=weights
        )(pred, targets)
        torch.testing.assert_close(
            losses["loss/surface_p"], 3.0 * targets["surface_p"].pow(2).mean()
        )

    def test_unknown_shared_constants_raise(self):
        """Naming a field the dataset normalizer does not hold is refused."""
        model = OmegaConf.create(self.MODEL)
        model.extra_task.transforms[1].fields = {"surface_p": "velocity"}
        with pytest.raises(ValueError, match="velocity"):
            apply_model_config(OmegaConf.create(self.DATASET), model)


@pytest.mark.parametrize("loss_type", ["huber", "mse", "relative_mse"])
def test_masked_union_set_equals_split_sets(loss_type):
    """A NaN-padded union set with masking scores like the two sets apart."""
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
    split_loss, split_td = LossCalculator(target_config, loss_type=loss_type)(
        split_pred, split_true
    )
    masked = LossCalculator(
        target_config, loss_type=loss_type, mask_nonfinite_targets=True
    )
    union_loss, union_td = masked(union_pred, union_true)
    torch.testing.assert_close(union_loss, split_loss)
    for key in split_td.keys():
        torch.testing.assert_close(union_td[key], split_td[key])

    split_metrics = MetricCalculator(target_config)(split_pred, split_true)
    union_metrics = MetricCalculator(target_config, mask_nonfinite_targets=True)(
        union_pred, union_true
    )
    for key in split_metrics.keys():
        torch.testing.assert_close(union_metrics[key], split_metrics[key])

    ### A field with no finite target contributes 0, not NaN.
    empty = union_true.clone()
    empty["a"] = torch.full_like(empty["a"], nan)
    _, empty_td = masked(union_pred, empty)
    assert empty_td["loss/a"] == 0


class TestForcesAndExport:
    """Forces and export follow each target's source set."""

    FIELD_TYPES = {"surface.pressure": "pressure", "surface.wss": "stress"}

    def test_forces_on_surface_set_aligned_with_vehicle(self):
        """Forces integrate the surface set's targets; a misaligned set raises."""
        domain = _domain()
        ctx = ForceContext.from_config(
            OmegaConf.create({"enabled": True}), self.FIELD_TYPES, "cpu"
        )
        pts = extract_targets(domain, TARGETS, SOURCES)
        forces = ctx.coefficients(domain, pts, pts, None, SOURCES)
        assert forces is not None and forces[0] == forces[1]
        ### Without target_sources, the volume interior is no surface case.
        assert ctx.coefficients(domain, pts, pts, None) is None

        shifted = domain.boundaries["surface_queries"].translate(torch.ones(3))
        domain.boundaries["surface_queries"] = shifted
        with pytest.raises(ValueError, match="aligned"):
            ctx.coefficients(domain, pts, pts, None, SOURCES)

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
