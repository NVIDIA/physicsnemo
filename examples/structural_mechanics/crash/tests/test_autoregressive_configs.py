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


"""Configuration contracts for the supported crash comparisons."""

import hashlib
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONF = Path(__file__).resolve().parents[1] / "conf"
AUTOREGRESSIVE_CONFIGS = (
    "crash_geoflare_autoregressive",
    "crash_deformer_autoregressive",
    "crash_deformer_contact_autoregressive",
)

# Effective recipes captured before config consolidation (revision d66b58ad).
# Only experiment_name is excluded; paths, interpolation strings, and every
# model/datapipe/training setting remain covered without duplicating YAML trees.
REFERENCE_DIGESTS = (
    "f7433cc0e7f412241a9978eac54ed2aa887e04bd65ed13775658262c9819534c",
    "8f5f60a9c8d74213051dd9003282b709de725a0f157b2880350a05ffcc3a2296",
    "cc976b6cb805164a10cdb55415950061be8356cbead2e2c0533da0f66d3b4781",
)


@pytest.mark.parametrize(
    "name,digest", tuple(zip(AUTOREGRESSIVE_CONFIGS, REFERENCE_DIGESTS, strict=True))
)
def test_consolidated_recipes_preserve_reference_settings(name, digest):
    """Packaging changes must not silently change the validated recipes."""
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        config = compose(config_name=name)
    settings = OmegaConf.to_container(config, resolve=False)
    settings.pop("experiment_name")
    assert (
        hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        == digest
    )


def _compose(name, overrides=()):
    """Compose a public entry point without reading datasets."""
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        return compose(
            config_name=name,
            overrides=[
                "training.raw_data_dir=/fixtures/train",
                "training.raw_data_dir_validation=/fixtures/validation",
                "inference.raw_data_dir_test=/fixtures/test",
                *overrides,
            ],
        )


@pytest.mark.parametrize("name", sorted(path.stem for path in CONF.glob("*.yaml")))
def test_all_public_experiments_compose(name):
    """Every advertised experiment has a complete Hydra defaults tree."""
    config = _compose(name)
    assert config.model._target_.startswith("rollout.")
    assert config.datapipe._target_.startswith("datapipe.")
    OmegaConf.to_container(config, resolve=True, throw_on_missing=False)


@pytest.mark.parametrize("name", AUTOREGRESSIVE_CONFIGS)
def test_autoregressive_protocol(name):
    """The three models retain the reference closed-loop BPTT-4 budget."""
    config = _compose(name)
    assert config.model.attention_type == "GALE_FA"
    assert config.model.functional_dim == 4
    assert config.model.out_dim == 3
    assert config.model.initial_velocity_mode == "previous_coords"
    assert config.model.rollout_steps_from_target is True
    assert config.model.teacher_forcing is False
    assert config.datapipe.sample_type == "random_time_window"
    assert config.datapipe.initial_history_steps == 2
    assert config.datapipe.rollout_window_steps == 4
    assert config.datapipe.windows_per_sample == 1
    assert config.training.num_time_steps == 26
    assert config.training.num_training_samples == 127
    assert config.training.num_validation_samples == 8
    assert config.training.epochs == 500
    assert config.training.start_lr == 2.0e-4
    assert config.training.end_lr == 1.0e-6
    assert config.training.optimizer == "muon"
    assert config.training.amp_dtype == "bfloat16"
    assert config.training.validation_freq == 50
    assert config.training.save_checkpoint_freq == 10
    assert config.training.manual_gradient_allreduce is True
    assert config.training.early_stopping_patience == 0
    assert config.training.acceleration_loss_weight == 0.0
    assert "teacher_forcing_schedule" not in config.training


def test_geoflare_and_deformer_share_backbone_and_training():
    """The structural baseline adds mesh processing without shrinking FLARE."""
    geoflare, deformer = map(_compose, AUTOREGRESSIVE_CONFIGS[:2])
    assert geoflare.training == deformer.training
    assert geoflare.reader == deformer.reader
    assert geoflare.inference == deformer.inference
    assert geoflare.datapipe._target_ == "datapipe.CrashPointCloudDataset"
    assert deformer.datapipe._target_ == "datapipe.CrashGraphDataset"
    for key in geoflare.model:
        if key != "_target_":
            assert geoflare.model[key] == deformer.model[key], key
    assert geoflare.model._target_ == "rollout.GeoTransolverAutoregressive"
    assert deformer.model._target_ == "rollout.MeshGeoFLAREAutoregressive"
    assert deformer.model.mesh_context_fusion == "pre_post"
    assert deformer.model.num_pre_processor_layers == 1
    assert deformer.model.num_post_processor_layers == 2
    assert not deformer.model.use_contact
    assert not deformer.model.enable_contact


def test_contact_recipe_changes_only_contact_reproducibility_and_memory():
    """Contact preserves the no-contact backbone, objective, and training budget."""
    baseline = _compose("crash_deformer_autoregressive")
    contact = _compose("crash_deformer_contact_autoregressive")
    for key in baseline.training:
        assert baseline.training[key] == contact.training[key], key
    assert contact.training.deterministic
    assert contact.training.reproducible_sampling
    assert contact.training.sampler_seed == 0
    assert contact.reader.include_contact_topology
    assert contact.datapipe.contact_require_elements
    assert contact.datapipe.contact_surface
    assert contact.datapipe.contact_surface_exclusion == "reference_geodesic"
    assert contact.datapipe.contact_geodesic_gap_min == 5.0
    assert contact.datapipe.contact_exclusion_hops is None
    changed = {
        "use_contact",
        "enable_contact",
        "enable_node_contact",
        "num_pre_processor_checkpoint_segments",
        "num_post_processor_checkpoint_segments",
    }
    for key in baseline.model:
        if key not in changed:
            assert baseline.model[key] == contact.model[key], key
    assert contact.model.use_contact and contact.model.enable_contact
    assert contact.model.contact_graph_backend == "surface"
    assert contact.model.contact_dim == 12
    assert contact.model.contact_include_velocity
    assert contact.model.contact_surface_predictive
    assert contact.model.contact_surface_material_fan
    assert contact.model.contact_surface_max_pairs == 8_000_000
    assert contact.model.contact_activation_distance == 5.0
    assert contact.model.contact_isolate_rng
    assert not contact.model.enable_cylinder_contact
    assert contact.inference == baseline.inference
