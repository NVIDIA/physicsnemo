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

"""Synthetic end-to-end smoke tests for every recipe-style invocation of train.yaml.

Each test:

1. Composes ``conf/train.yaml`` with ``model=<...> dataset=<...> [+ overrides]``
   to mirror what each previously-named train_*.yaml recipe used to do.
2. Loads the corresponding ``datasets/<dataset>.yaml`` to read the
   ``targets:`` block and injects the resulting channel-count sum as
   ``cfg.out_dim`` (production code does the same in
   ``build_dataloaders``; the synthetic test path bypasses that and so
   has to set it explicitly).
3. Builds a synthetic post-pipeline ``DomainMesh`` matching the structure
   the dataset's pipeline would have produced.
4. Builds the recipe collate from the composed cfg's ``forward_kwargs``
   spec.
5. Instantiates the model at shrunk dimensions (small ``n_layers``,
   ``n_hidden``, ``n_head``, ``slice_num``; ``include_local_features``
   off) so each test runs in seconds on CPU.
6. Runs ``model.forward(**batch["forward_kwargs"])`` and verifies the
   output shape matches ``target_config``.
7. Computes the dict-based loss to confirm pred / target shapes line up.

Tests skip if the model class is not importable (e.g., FLARE under
``physicsnemo.experimental`` may be gated, or DoMINO is not yet wired
up). The test set deliberately excludes DoMINO recipes because their
``forward_kwargs`` references fields the dataset doesn't expose
(documented in the model template comments).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import NamedTuple

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from physicsnemo.mesh import DomainMesh, Mesh

from collate import build_collate_fn
from loss import LossCalculator
from output_normalize import normalize_output_to_tensordict
from utils import field_dim

warnings.filterwarnings("ignore", category=DeprecationWarning)


_RECIPE_ROOT = Path(__file__).resolve().parent.parent


def _load_dataset_targets(dataset_name: str) -> dict[str, str]:
    """Read the ``targets:`` block out of a dataset pipeline YAML."""
    dataset_path = _RECIPE_ROOT / "datasets" / f"{dataset_name}.yaml"
    dataset_cfg = OmegaConf.load(dataset_path)
    targets = OmegaConf.to_container(dataset_cfg.targets, resolve=True)
    assert isinstance(targets, dict) and targets, f"{dataset_name} has no targets:"
    return targets


def _compose_train_cfg(
    model: str, dataset: str, *, overrides: list[str] | None = None
) -> DictConfig:
    """Compose ``conf/train.yaml`` with the given model + dataset selection.

    Mirrors what ``@hydra.main`` does at runtime, then injects
    ``cfg.out_dim`` from the chosen dataset's ``targets:`` block (the
    production launcher does the same inside ``build_dataloaders``;
    synthetic tests bypass that and have to do it themselves so the
    model template's ``${out_dim}`` interpolation resolves).
    """
    targets = _load_dataset_targets(dataset)
    out_dim = sum(field_dim(ftype) for ftype in targets.values())

    cli_overrides = [
        f"model={model}",
        f"dataset={dataset}",
        f"+out_dim={out_dim}",  # not declared in train.yaml; production sets it in build_dataloaders
        *(overrides or []),
    ]
    with initialize_config_dir(
        config_dir=str(_RECIPE_ROOT / "conf"),
        version_base=None,
    ):
        return compose(config_name="train", overrides=cli_overrides)


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
### Recipes under test (the previously-named train_*.yaml files mapped
### to (model, dataset, [overrides]) tuples for the new train.yaml).
### ---------------------------------------------------------------------------


### DoMINO is excluded by design (illustrative-only forward_kwargs.data_dict
### references fields the dataset doesn't expose).


class _RecipeSpec(NamedTuple):
    recipe_id: str
    model: str
    dataset: str
    domain: str
    overrides: list[str]


_TENSOR_INPUT_RECIPES: list[_RecipeSpec] = [
    _RecipeSpec(
        "geotransolver_surface",
        "geotransolver_surface",
        "drivaer_ml_surface",
        "surface",
        [],
    ),
    _RecipeSpec(
        "geotransolver_volume",
        "geotransolver_volume",
        "drivaer_ml_volume",
        "volume",
        [],
    ),
    _RecipeSpec(
        "geotransolver_fa_surface",
        "geotransolver_surface",
        "drivaer_ml_surface",
        "surface",
        ["+model.attention_type=GALE_FA"],
    ),
    _RecipeSpec(
        "geotransolver_fa_volume",
        "geotransolver_volume",
        "drivaer_ml_volume",
        "volume",
        [
            "+model.attention_type=GALE_FA",
            "model.n_layers=20",
            "model.state_mixing_mode=concat_project",
        ],
    ),
    _RecipeSpec(
        "geotransolver_fa_highlift_surface",
        "geotransolver_surface",
        "highlift_surface",
        "surface",
        ["+model.attention_type=GALE_FA"],
    ),
    _RecipeSpec(
        "transolver_surface", "transolver_surface", "drivaer_ml_surface", "surface", []
    ),
    _RecipeSpec(
        "transolver_volume", "transolver_volume", "drivaer_ml_volume", "volume", []
    ),
    _RecipeSpec("flare_surface", "flare_surface", "drivaer_ml_surface", "surface", []),
    _RecipeSpec("flare_volume", "flare_volume", "drivaer_ml_volume", "volume", []),
    _RecipeSpec(
        "highlift_surface", "geotransolver_surface", "highlift_surface", "surface", []
    ),
    _RecipeSpec(
        "highlift_volume",
        "geotransolver_volume_highlift",
        "highlift_volume",
        "volume",
        [],
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
    "recipe_id,model,dataset,domain,overrides",
    _TENSOR_INPUT_RECIPES,
    ids=[r.recipe_id for r in _TENSOR_INPUT_RECIPES],
)
def test_tensor_input_config_synthetic_e2e(
    recipe_id: str,
    model: str,
    dataset: str,
    domain: str,
    overrides: list[str],
) -> None:
    """Build a synthetic DomainMesh, run the configured model end-to-end."""
    train_cfg = _compose_train_cfg(model, dataset, overrides=overrides)

    ### The chosen model template must declare a tensor-input contract
    ### for the test driver below to work (it builds tensor batches via
    ### the collate's `input_type='tensors'` branch).
    assert OmegaConf.select(train_cfg, "input_type") == "tensors", (
        f"{recipe_id}: {model=} has {input_type=}, not 'tensors'"
    )
    assert OmegaConf.select(train_cfg, "output_type") == "tensors", (
        f"{recipe_id}: {model=} has {output_type=}, not 'tensors'"
    )

    target_config = _load_dataset_targets(dataset)

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
        model_inst = hydra.utils.instantiate(small_model_cfg, _convert_="partial")
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"model class not importable: {e}")

    ### Forward and verify output shape matches the target channel count.
    with torch.no_grad():
        output = model_inst(**batch["forward_kwargs"])
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
### GLOBE (mesh-input / mesh-output) recipes
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


_MESH_INPUT_RECIPES: list[tuple[str, str, str, str, list[str]]] = [
    ("globe_surface", "globe_surface", "drivaer_ml_surface", "surface", []),
    ("globe_volume", "globe_volume", "drivaer_ml_volume", "volume", []),
]


@pytest.mark.parametrize(
    "recipe_id,model,dataset,domain,overrides",
    _MESH_INPUT_RECIPES,
    ids=[r[0] for r in _MESH_INPUT_RECIPES],
)
def test_mesh_input_config_synthetic_e2e(
    recipe_id: str,
    model: str,
    dataset: str,
    domain: str,
    overrides: list[str],
) -> None:
    """Same shape as the tensor-input test but for GLOBE-style mesh I/O.

    Builds a synthetic post-pipeline DomainMesh, instantiates GLOBE with
    shrunk kernel sizes, runs ``forward()`` with the mesh-native batch
    (no batch dim added), and confirms the output Mesh's ``point_data``
    contains every target field at the right shape.
    """
    train_cfg = _compose_train_cfg(model, dataset, overrides=overrides)

    assert OmegaConf.select(train_cfg, "input_type") == "mesh"
    assert OmegaConf.select(train_cfg, "output_type") == "mesh"

    target_config = _load_dataset_targets(dataset)

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
        model_inst = hydra.utils.instantiate(small_model_cfg, _convert_="partial")
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"GLOBE not importable: {e}")

    with torch.no_grad():
        output = model_inst(**batch["forward_kwargs"])
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
