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

"""Synthetic end-to-end smoke tests for every training config in ``conf/``.

Each test:

1. Builds a synthetic post-pipeline ``DomainMesh`` matching the structure
   the dataset YAML's pipeline would have produced (target fields in
   ``interior.point_data``; surface boundaries with precomputed normals;
   volume interior with sdf / sdf_normals; ``global_data`` carrying
   ``U_inf`` etc.).
2. Builds the recipe collate from the training YAML's ``forward_kwargs``
   spec.
3. Instantiates the model at shrunk dimensions (small ``n_layers``,
   ``n_hidden``, ``n_head``, ``slice_num``; ``include_local_features``
   off) so each test runs in seconds on CPU.
4. Runs ``model.forward(**batch["forward_kwargs"])`` and verifies the
   output shape matches ``target_config``.
5. Computes the dict-based loss to confirm pred / target shapes line up.

Tests skip if the model class is not importable (e.g., FLARE under
``physicsnemo.experimental`` may be gated, or DoMINO is not yet wired
up). The test set deliberately excludes DoMINO YAMLs because their
``forward_kwargs`` references fields the dataset doesn't expose
(documented in the YAML comments).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import hydra
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from physicsnemo.mesh import DomainMesh, Mesh

from collate import build_collate_fn
from loss import LossCalculator
from output_normalize import normalize_output_to_tensordict

warnings.filterwarnings("ignore", category=DeprecationWarning)


### ---------------------------------------------------------------------------
### Synthetic DomainMesh builders
### ---------------------------------------------------------------------------
###
### These wrap the shared `conftest.make_*_domain_mesh` factories with the
### larger N (80 surface cells / 200 volume points) the synthetic E2E
### pipeline needs in order for the shrunk transformer / GLOBE models to
### produce non-degenerate outputs.

from conftest import make_surface_domain_mesh, make_volume_domain_mesh  # noqa: E402


def _surface_domain_mesh(
    target_config: dict[str, str], n_cells: int = 80
) -> DomainMesh:
    """Surface DomainMesh sized for synthetic E2E testing (80 cells default)."""
    return make_surface_domain_mesh(target_config, n_cells=n_cells)


def _volume_domain_mesh(target_config: dict[str, str], n_pts: int = 200) -> DomainMesh:
    """Volume DomainMesh sized for synthetic E2E testing (200 points default)."""
    return make_volume_domain_mesh(target_config, n_pts=n_pts)


### ---------------------------------------------------------------------------
### Model-shrinking overrides
### ---------------------------------------------------------------------------


### Override knobs that don't change the model's external interface
### (input / output channel counts) but make the model much cheaper to
### run on CPU. Per-class because each model exposes a different set of
### shrinkable knobs.
_MODEL_SHRINK_OVERRIDES: dict[str, dict] = {
    "GeoTransolver": {
        "n_layers": 2,
        "n_hidden": 32,
        "slice_num": 16,
        "n_head": 2,
        # Local-features path needs ball-query ops that are heavy on CPU and
        # do not exercise our changes; disable for the smoke test.
        "include_local_features": False,
    },
    "Transolver": {
        "n_layers": 2,
        "n_hidden": 32,
        "slice_num": 16,
        "n_head": 2,
    },
    "FLARE": {
        "n_layers": 2,
        "n_hidden": 32,
        "slice_num": 16,
        "n_head": 2,
    },
}


def _shrink_model_cfg(model_cfg: DictConfig) -> DictConfig:
    """Apply the per-class shrink overrides to a model config."""
    target = model_cfg._target_
    cls_name = target.split(".")[-1]
    overrides = _MODEL_SHRINK_OVERRIDES.get(cls_name)
    if overrides is None:
        return model_cfg
    return OmegaConf.merge(model_cfg, OmegaConf.create(overrides))


### ---------------------------------------------------------------------------
### Configurations under test
### ---------------------------------------------------------------------------


_RECIPE_ROOT = Path(__file__).resolve().parent.parent

### Each entry: (training-config name, dataset-config name, "surface" or "volume").
### DoMINO is excluded by design (illustrative-only `forward_kwargs.data_dict`;
### dataset doesn't expose all the per-cell neighbor / grid features its
### `forward()` expects).
_TENSOR_INPUT_CONFIGS: list[tuple[str, str, str]] = [
    (
        "train_geotransolver_automotive_surface",
        "drivaer_ml_surface",
        "surface",
    ),
    (
        "train_geotransolver_automotive_volume",
        "drivaer_ml_volume",
        "volume",
    ),
    (
        "train_geotransolver_fa_automotive_surface",
        "drivaer_ml_surface",
        "surface",
    ),
    (
        "train_geotransolver_fa_automotive_volume",
        "drivaer_ml_volume",
        "volume",
    ),
    (
        "train_geotransolver_fa_highlift_surface",
        "highlift_surface",
        "surface",
    ),
    (
        "train_transolver_automotive_surface",
        "drivaer_ml_surface",
        "surface",
    ),
    (
        "train_transolver_automotive_volume",
        "drivaer_ml_volume",
        "volume",
    ),
    (
        "train_flare_automotive_surface",
        "drivaer_ml_surface",
        "surface",
    ),
    (
        "train_flare_automotive_volume",
        "drivaer_ml_volume",
        "volume",
    ),
    (
        "train_highlift_surface",
        "highlift_surface",
        "surface",
    ),
    (
        "train_highlift_volume",
        "highlift_volume",
        "volume",
    ),
]


### ---------------------------------------------------------------------------
### Test driver
### ---------------------------------------------------------------------------


def _output_to_tensordict(
    output, target_config: dict[str, str], n_spatial_dims: int = 3
) -> TensorDict:
    """Mirror the output-normalization step in ``train.forward_pass``.

    Dispatches on output type the same way the production code does:
    ``Mesh`` outputs use ``output.point_data.select(*target_config)``;
    tensor outputs go through :func:`split_concat_by_target` (with
    DoMINO-style ``(vol, surf)`` tuple unwrapping). The choice of
    ``output_type`` is inferred here from the value's runtime type so a
    single helper can drive both the tensor- and mesh-input parametrized
    suites.
    """
    output_type = "mesh" if isinstance(output, Mesh) else "tensors"
    return normalize_output_to_tensordict(
        output, target_config, output_type, n_spatial_dims
    )


@pytest.mark.parametrize(
    "train_name,dataset_name,domain",
    _TENSOR_INPUT_CONFIGS,
    ids=[name for name, _, _ in _TENSOR_INPUT_CONFIGS],
)
def test_tensor_input_config_synthetic_e2e(
    train_name: str, dataset_name: str, domain: str
) -> None:
    """Build a synthetic DomainMesh, run the configured model end-to-end."""
    train_path = _RECIPE_ROOT / "conf" / f"{train_name}.yaml"
    dataset_path = _RECIPE_ROOT / "conf" / "dataset" / f"{dataset_name}.yaml"
    train_cfg = OmegaConf.load(train_path)
    dataset_cfg = OmegaConf.load(dataset_path)

    ### Both YAMLs must declare `input_type` and `output_type`; the
    ### `tensors` value is what `_run_tensor_input_config` is parametrized
    ### over.
    assert OmegaConf.select(train_cfg, "input_type") == "tensors", (
        f"{train_name} input_type is not 'tensors'"
    )
    assert OmegaConf.select(train_cfg, "output_type") == "tensors", (
        f"{train_name} output_type is not 'tensors'"
    )
    target_config = OmegaConf.to_container(dataset_cfg.targets, resolve=True)
    assert isinstance(target_config, dict) and target_config, (
        f"{dataset_name} has no targets:"
    )

    ### Build a synthetic post-pipeline DomainMesh.
    if domain == "surface":
        ds = _surface_domain_mesh(target_config)
    elif domain == "volume":
        ds = _volume_domain_mesh(target_config)
    else:  # pragma: no cover -- table-only typo guard
        raise ValueError(f"Unknown domain {domain!r}")

    ### Build the recipe collate the same way `build_dataloaders` would.
    forward_kwargs_spec = OmegaConf.to_container(train_cfg.forward_kwargs, resolve=True)
    collate = build_collate_fn(
        input_type="tensors",
        forward_kwargs_spec=forward_kwargs_spec,
        target_config=target_config,
    )
    batch = collate([(ds, {})])

    ### Instantiate model with shrunk knobs. Skip if the model class
    ### is not importable in this environment (e.g., experimental gates).
    try:
        small_model_cfg = _shrink_model_cfg(train_cfg.model)
        model = hydra.utils.instantiate(small_model_cfg, _convert_="partial")
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"model class not importable: {e}")

    ### Forward and verify output shape matches the target channel count.
    with torch.no_grad():
        output = model(**batch["forward_kwargs"])
    pred_td = _output_to_tensordict(output, target_config)

    for name, ftype in target_config.items():
        assert name in pred_td.keys(), f"{name} missing from pred"
        pred_t = pred_td[name]
        target_t = batch["targets"][name]
        assert pred_t.shape == target_t.shape, (
            f"shape mismatch for {name}: pred={tuple(pred_t.shape)} "
            f"vs target={tuple(target_t.shape)}"
        )

    ### Loss computes without errors.
    field_weights = (
        OmegaConf.to_container(
            OmegaConf.select(
                train_cfg, "training.field_weights", default=OmegaConf.create({})
            ),
            resolve=True,
        )
        or None
    )
    lc = LossCalculator(
        target_config=target_config,
        loss_type=train_cfg.training.loss_type,
        field_weights=field_weights,
    )
    loss, _ = lc(pred_td, batch["targets"])
    assert torch.isfinite(loss), f"loss not finite: {float(loss)}"


### ---------------------------------------------------------------------------
### GLOBE (mesh-input / mesh-output) configs
### ---------------------------------------------------------------------------


### Different shrink overrides for GLOBE: it has no n_layers / n_hidden in the
### transformer sense; instead, dial down `n_communication_hyperlayers` and
### kernel MLP sizes. `expand_far_targets` is left True (matches default).
_GLOBE_SHRINK_OVERRIDES = {
    "n_communication_hyperlayers": 1,
    "hidden_layer_sizes": [16, 16],
    "n_latent_scalars": 4,
    "n_latent_vectors": 2,
    "n_spherical_harmonics": 2,
}


_MESH_INPUT_CONFIGS: list[tuple[str, str, str]] = [
    (
        "train_globe_automotive_surface",
        "drivaer_ml_surface",
        "surface",
    ),
    (
        "train_globe_automotive_volume",
        "drivaer_ml_volume",
        "volume",
    ),
]


@pytest.mark.parametrize(
    "train_name,dataset_name,domain",
    _MESH_INPUT_CONFIGS,
    ids=[name for name, _, _ in _MESH_INPUT_CONFIGS],
)
def test_mesh_input_config_synthetic_e2e(
    train_name: str, dataset_name: str, domain: str
) -> None:
    """Same shape as the tensor-input test but for GLOBE-style mesh I/O.

    Builds a synthetic post-pipeline DomainMesh, instantiates GLOBE with
    shrunk kernel sizes, runs ``forward()`` with the mesh-native batch
    (no batch dim added), and confirms the output Mesh's ``point_data``
    contains every target field at the right shape.
    """
    train_path = _RECIPE_ROOT / "conf" / f"{train_name}.yaml"
    dataset_path = _RECIPE_ROOT / "conf" / "dataset" / f"{dataset_name}.yaml"
    train_cfg = OmegaConf.load(train_path)
    dataset_cfg = OmegaConf.load(dataset_path)

    assert OmegaConf.select(train_cfg, "input_type") == "mesh"
    assert OmegaConf.select(train_cfg, "output_type") == "mesh"
    target_config = OmegaConf.to_container(dataset_cfg.targets, resolve=True)
    assert isinstance(target_config, dict) and target_config

    if domain == "surface":
        ds = _surface_domain_mesh(target_config)
    elif domain == "volume":
        ds = _volume_domain_mesh(target_config)
    else:  # pragma: no cover
        raise ValueError(f"Unknown domain {domain!r}")

    forward_kwargs_spec = OmegaConf.to_container(train_cfg.forward_kwargs, resolve=True)
    collate = build_collate_fn(
        input_type="mesh",
        forward_kwargs_spec=forward_kwargs_spec,
        target_config=target_config,
    )
    batch = collate([(ds, {})])

    ### Shrink GLOBE while preserving the externally-visible
    ### `output_field_ranks` and `boundary_source_data_ranks`.
    small_model_cfg = OmegaConf.merge(
        train_cfg.model, OmegaConf.create(_GLOBE_SHRINK_OVERRIDES)
    )
    try:
        model = hydra.utils.instantiate(small_model_cfg, _convert_="partial")
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"GLOBE not importable: {e}")

    with torch.no_grad():
        output = model(**batch["forward_kwargs"])
    pred_td = _output_to_tensordict(output, target_config)

    for name, ftype in target_config.items():
        assert name in pred_td.keys(), f"{name} missing from pred"
        pred_t = pred_td[name]
        target_t = batch["targets"][name]
        assert pred_t.shape == target_t.shape, (
            f"shape mismatch for {name}: pred={tuple(pred_t.shape)} "
            f"vs target={tuple(target_t.shape)}"
        )

    field_weights = (
        OmegaConf.to_container(
            OmegaConf.select(
                train_cfg, "training.field_weights", default=OmegaConf.create({})
            ),
            resolve=True,
        )
        or None
    )
    lc = LossCalculator(
        target_config=target_config,
        loss_type=train_cfg.training.loss_type,
        field_weights=field_weights,
    )
    loss, _ = lc(pred_td, batch["targets"])
    assert torch.isfinite(loss), f"loss not finite: {float(loss)}"
