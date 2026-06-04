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

"""
Unified External Aerodynamics Inference Script

Companion to ``train.py``.  Loads a trained checkpoint, runs the model
over a chosen split, reports metrics (in training space, matching
validation), re-dimensionalizes predictions to physical units, and
writes each sample back as a native ``.pdmsh`` ``DomainMesh`` (the same
on-disk format the datapipe reads).

The script is model/dataset-agnostic: it keys off the same
``input_type`` / ``output_type`` / ``forward_kwargs`` / ``targets``
contract the trainer uses, so it works for every model in the recipe
(GeoTransolver, Transolver, FLARE, GLOBE, ...) without per-model code.

Usage::

    # HiLift surface (vanilla GeoTransolver)
    python src/infer.py model=geotransolver_surface dataset=highlift_surface \
        run_id=<the trained run_id> infer_split=test

    # HiLift volume
    python src/infer.py model=geotransolver_volume_highlift \
        dataset=highlift_volume run_id=<the trained run_id> infer_split=test

Output layout::

    ${output_dir}/${run_id}/
      predictions/<sample_id>.pdmsh   # DomainMesh: interior carries
                                      # pred_<field> and true_<field>
      metrics.jsonl                   # per-sample + summary records

Caveats:

- Metrics are reported in training space (non-dimensional / normalized),
  identical to the validation loop, so they line up with the numbers
  logged during training.  Re-dimensionalization is applied only to the
  written fields.
- ``CenterMesh``'s per-sample translation offset is not stored, so the
  written geometry is physical-*scale* (when ``rescale_geometry=true``)
  but remains centered at the origin.  Field values are unaffected.
- Inference runs at whatever ``sampling_resolution`` allows (default:
  effectively the full mesh).  Very large volume meshes may need a
  smaller cap to fit in memory; chunked inference is left as future work.
- v1 is intended for single-process inference.  It does not break under
  ``torchrun`` (each rank handles its sampler shard and metrics are
  all-reduced), but distributed writing has not been a focus.
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from collate import build_collate_fn
from datasets import load_dataset_config
from metrics import DEFAULT_METRICS, MetricCalculator, MetricName
from nondim import NonDimensionalizeByMetadata, NondimFieldType, freestream_scales
from omegaconf import DictConfig, OmegaConf
from output_normalize import IOType, normalize_output_to_tensordict
from tabulate import tabulate
from tensordict import TensorDict
from utils import FieldType, set_seed

### Reuse the trainer's dataloader assembly, autocast helper, and the
### Mesh/TensorDict-aware device mover rather than duplicating them. The
### `@hydra.main`-decorated launcher in train.py is not executed on
### import, so importing these is side-effect free.
from train import _recursive_to_device, build_dataloaders, get_autocast_context

from physicsnemo import datapipes  # noqa: F401 - registers ${dp:...} resolver
from physicsnemo.distributed import DistributedManager
from physicsnemo.mesh import DomainMesh
from physicsnemo.utils import load_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

_LOGGER = logging.getLogger("inference")


### ---------------------------------------------------------------------------
### Checkpoint resolution
### ---------------------------------------------------------------------------


def resolve_checkpoint_path(cfg: DictConfig) -> str:
    """Resolve the directory `load_checkpoint` should scan.

    Prefers an explicit ``checkpoint_path``; otherwise reproduces the
    trainer's ``${checkpoint_dir or output_dir}/${run_id}/checkpoints``
    layout. Note ``checkpoint_dir`` should point at the *training* output
    directory (default ``runs``), not this script's ``output_dir``.

    Raises:
        ValueError: If neither ``checkpoint_path`` nor ``run_id`` is set.
    """
    explicit = OmegaConf.select(cfg, "checkpoint_path", default=None)
    if explicit:
        return str(explicit)
    run_id = OmegaConf.select(cfg, "run_id", default=None)
    if not run_id:
        raise ValueError(
            "Set either `checkpoint_path=<dir>` (a checkpoints directory) or "
            "`run_id=<trained run>` (resolved under `checkpoint_dir`, default "
            "'runs') so the trained weights can be located."
        )
    ckpt_root = OmegaConf.select(cfg, "checkpoint_dir", default=None) or cfg.output_dir
    return os.path.join(str(ckpt_root), str(run_id), "checkpoints")


### ---------------------------------------------------------------------------
### Re-dimensionalization
### ---------------------------------------------------------------------------


def build_redim_field_types(ds_yaml: DictConfig) -> dict[str, NondimFieldType]:
    """Map each canonical prediction field to its non-dim recipe.

    Composes the dataset YAML's ``NonDimensionalizeByMetadata.fields``
    (keyed by *raw* field names, e.g. ``pMeanTrim: pressure``) with the
    ``RenameMeshFields`` map (raw -> canonical, e.g. ``pMeanTrim:
    pressure``) so the result is keyed by the canonical names the model
    actually predicts (e.g. ``{pressure: pressure, wss: stress}``).

    Read straight off the config (``resolve=False`` so unrelated
    ``${...}`` interpolations are not forced), which keeps this robust to
    transform-internal attribute names. Returns ``{}`` when the dataset
    declares no ``NonDimensionalizeByMetadata`` transform (e.g. a dataset
    whose fields are already physical), making re-dim a no-op.
    """
    transforms = (
        OmegaConf.to_container(
            OmegaConf.select(ds_yaml, "pipeline.transforms", default=[]),
            resolve=False,
        )
        or []
    )
    nondim_fields: dict[str, str] = {}
    rename_map: dict[str, str] = {}
    for t in transforms:
        if not isinstance(t, dict):
            continue
        target = str(t.get("_target_", ""))
        if "NonDimensionalizeByMetadata" in target:
            nondim_fields = dict(t.get("fields", {}) or {})
        elif "RenameMeshFields" in target:
            ### Rename maps live under per-association sub-blocks; a field
            ### is renamed in whichever association it was declared.
            for assoc in ("cell_data", "point_data"):
                rename_map.update(t.get(assoc, {}) or {})

    return {
        rename_map.get(raw, raw): ftype for raw, ftype in nondim_fields.items()
    }


def redimensionalize(
    td: TensorDict,
    *,
    normalizer: Any | None,
    nondim: NonDimensionalizeByMetadata | None,
    field_types: dict[str, NondimFieldType],
    global_data: TensorDict,
) -> TensorDict:
    """Invert the pipeline's field conditioning back to physical units.

    Reverses the two conditioning stages in pipeline order: first undo
    statistical normalization (``NormalizeMeshFields.inverse_td``, which
    carries its own stats and skips unnormalized fields), then undo
    physics non-dimensionalization (``NonDimensionalizeByMetadata.inverse_td``,
    using freestream scales read from the sample's ``global_data``). Each
    stage is skipped when the corresponding transform was absent from the
    dataset, so a fully un-conditioned dataset returns ``td`` unchanged.
    """
    out = td.float()
    if normalizer is not None:
        out = normalizer.inverse_td(out)
    if nondim is not None and field_types:
        q_inf, p_inf, U_inf_mag, rho_inf, T_inf = freestream_scales(global_data)
        out = nondim.inverse_td(
            out,
            field_types,
            q_inf,
            p_inf,
            U_inf_mag,
            rho_inf=rho_inf,
            T_inf=T_inf,
        )
    return out


### ---------------------------------------------------------------------------
### Output writing
### ---------------------------------------------------------------------------


def _to_pointwise(td: TensorDict, output_type: IOType) -> TensorDict:
    """Drop the leading batch dim so a TD aligns with the interior points.

    Tensor-output models produce ``(1, N)`` / ``(1, N, C)`` leaves
    (batch_size ``[1, N]``); mesh-output models produce per-point leaves
    already (batch_size ``[N]``).
    """
    return td[0] if output_type == "tensors" else td


def _sample_id(metadata: dict[str, Any], idx: int) -> str:
    """Build a filesystem-safe, unique sample id from the source path.

    Uses the case mesh's directory + filename when discoverable (e.g.
    ``geo_LHC001_AoA_4_domain``) and always prefixes the sampler index so
    ids stay unique even if two samples share a name.
    """
    src = metadata.get("source_path", "") if isinstance(metadata, dict) else ""
    hint = ""
    if src:
        parts = Path(src).parts
        mesh_part = next(
            (p for p in parts if p.endswith((".pdmsh", ".pmsh"))), None
        )
        if mesh_part is not None:
            stem = mesh_part.rsplit(".", 1)[0]
            pos = parts.index(mesh_part)
            parent = parts[pos - 1] if pos > 0 else ""
            hint = f"{parent}_{stem}" if parent else stem
        else:
            hint = Path(src).stem
    hint = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in hint)
    return f"{idx:05d}_{hint}" if hint else f"sample_{idx:05d}"


def attach_and_save(
    domain: DomainMesh,
    pred_phys: TensorDict,
    true_phys: TensorDict,
    target_config: dict[str, FieldType],
    out_path: Path,
    *,
    rescale_geometry: bool,
) -> None:
    """Attach physical predictions/targets to the DomainMesh and save it.

    Writes ``pred_<name>`` and ``true_<name>`` onto a copy of the
    interior's ``point_data`` (the training-space target fields are
    dropped to avoid ambiguity with their physical ``true_<name>``
    counterparts; non-target inputs like ``sdf`` are kept). The result is
    saved with :meth:`DomainMesh.save` as a native ``.pdmsh`` tree.

    When *rescale_geometry* is set and ``L_ref`` is available, every mesh
    in the domain is scaled by ``L_ref`` to recover physical-scale
    coordinates (``Mesh.scale`` leaves ``point_data`` untouched, so the
    attached fields are not affected).
    """
    if rescale_geometry and "L_ref" in domain.global_data:
        L_ref = domain.global_data["L_ref"]
        domain = domain.apply_to_meshes(lambda m: m.scale(L_ref))

    interior = domain.interior
    ### Drop training-space targets (replaced by physical true_<name>);
    ### keep non-target inputs such as sdf / sdf_normals for inspection.
    present_targets = [n for n in target_config if n in interior.point_data.keys()]
    new_pd = interior.point_data.exclude(*present_targets).clone()
    for name, val in pred_phys.items():
        new_pd[f"pred_{name}"] = val
    for name, val in true_phys.items():
        new_pd[f"true_{name}"] = val

    ### `Mesh.copy` is the tensorclass shallow copy used by the transforms;
    ### swap in the augmented point_data, mirroring their pattern.
    new_interior = interior.copy()
    new_interior.point_data = new_pd

    out_domain = DomainMesh(
        interior=new_interior,
        boundaries=domain.boundaries,
        global_data=domain.global_data,
    ).to("cpu")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_domain.save(str(out_path))


### ---------------------------------------------------------------------------
### Driver
### ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../conf", config_name="infer")
def main(cfg: DictConfig) -> None:
    """Run checkpoint inference over a split and write physical predictions.

    Args:
        cfg: Hydra config composed from ``conf/infer.yaml`` (see that file
            and this module's docstring for the knobs). Requires
            ``model=``, ``dataset=``, and either ``run_id=`` or
            ``checkpoint_path=`` on the CLI.
    """
    DistributedManager.initialize()
    dist_manager = DistributedManager()
    device = dist_manager.device
    is_rank0 = dist_manager.rank == 0
    logger = RankZeroLoggingWrapper(PythonLogger(name="inference"), dist_manager)

    set_seed(cfg.training.get("seed", None), rank=dist_manager.rank)

    ### Reuse the trainer's loader assembly: this resolves the split via
    ### `val_split` (aliased to `infer_split` in the YAML), returns the
    ### NormalizeMeshFields normalizer, and auto-derives `cfg.out_dim`
    ### from the dataset's `targets:` block (needed before the model
    ### template's `out_dim: ${out_dim}` resolves). The train loader is
    ### built and discarded -- a minor cost for full reuse.
    _train_loader, val_loader, normalizer, dataset_info = build_dataloaders(cfg)
    target_config: dict[str, FieldType] = dataset_info["targets"]
    output_type: IOType | None = cfg.get("output_type", None)
    if output_type is None:
        raise ValueError(
            "Model YAML must declare `output_type` (one of 'mesh', 'tensors')."
        )

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    logger.info(f"Targets (from dataset YAML): {target_config}")

    # -- Output dir + JSONL logging (rank 0) ------------------------------------
    run_id = OmegaConf.select(cfg, "run_id", default=None)
    if not run_id:
        raise ValueError(
            "`run_id=<name>` is required: it identifies the checkpoint run and "
            "namespaces the output directory. (Use `checkpoint_path=<dir>` to "
            "additionally override where the weights are read from.)"
        )
    run_dir = Path(cfg.output_dir) / str(run_id)
    pred_dir = run_dir / "predictions"
    log_jsonl = None
    if is_rank0:
        pred_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.jsonl"

        def log_jsonl(record: dict) -> None:
            record["ts"] = datetime.now(timezone.utc).isoformat()
            with open(metrics_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")

    # -- Model + checkpoint -----------------------------------------------------
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model.__class__.__name__} ({num_params:,} params)")

    ckpt_path = resolve_checkpoint_path(cfg)
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint directory {ckpt_path!r} does not exist. Check "
            f"`run_id` / `checkpoint_dir` (or set `checkpoint_path`)."
        )
    loaded_epoch = load_checkpoint(path=ckpt_path, models=model, device=device)
    logger.info(f"Loaded checkpoint from {ckpt_path!r} (epoch {loaded_epoch}).")

    if cfg.get("compile", False):
        model = torch.compile(model)
    model.eval()

    # -- Collate (rebuilt here so we keep the source DomainMesh in hand) --------
    ### The trainer's collate returns only forward_kwargs + targets and
    ### discards the DomainMesh; we re-index the dataset per sample so the
    ### domain stays available for writing, then run the same collate on a
    ### 1-element list to get the batched forward kwargs.
    collate_fn = build_collate_fn(
        input_type=cfg.input_type,
        forward_kwargs_spec=OmegaConf.to_container(cfg.forward_kwargs, resolve=True),
        target_config=target_config,
    )

    # -- Re-dimensionalization setup --------------------------------------------
    redimensionalize_on = bool(cfg.get("redimensionalize", True))
    recipe_root = Path(__file__).resolve().parent.parent
    ds_yaml = load_dataset_config(recipe_root / "datasets" / f"{cfg.dataset}.yaml")
    field_types = build_redim_field_types(ds_yaml) if redimensionalize_on else {}
    nondim_helper = (
        NonDimensionalizeByMetadata(fields=field_types) if field_types else None
    )
    active_normalizer = normalizer if redimensionalize_on else None
    logger.info(
        f"Re-dimensionalization: {'on' if redimensionalize_on else 'off'} "
        f"(field types: {field_types}, "
        f"normalizer: {'yes' if active_normalizer is not None else 'no'})"
    )

    # -- Metrics ----------------------------------------------------------------
    metrics_cfg = OmegaConf.select(cfg, "metrics", default=None)
    metrics_list: list[MetricName] = (
        list(DEFAULT_METRICS)
        if metrics_cfg is None
        else OmegaConf.to_container(metrics_cfg, resolve=True)
    )
    metric_calculator = MetricCalculator(
        target_config=target_config, metrics=metrics_list
    )

    if is_rank0 and log_jsonl is not None:
        log_jsonl(
            {
                "phase": "config",
                "model": model.__class__.__name__,
                "dataset": cfg.dataset,
                "infer_split": cfg.get("infer_split", cfg.get("val_split")),
                "checkpoint": ckpt_path,
                "epoch": loaded_epoch,
                "redimensionalize": redimensionalize_on,
                "num_parameters": num_params,
            }
        )

    # -- Inference loop ---------------------------------------------------------
    dataset = val_loader.dataset
    sampler = val_loader.sampler
    n_samples = len(sampler)
    log_every = max(1, int(cfg.get("logging", {}).get("log_every_n_steps", 10)))
    logger.info(f"Running inference over {n_samples} sample(s) -> {pred_dir}")

    totals: dict[str, float] = defaultdict(float)
    count = 0
    for i, idx in enumerate(sampler):
        sample = dataset[idx]
        domain, metadata = sample
        batch = _recursive_to_device(collate_fn([sample]), device)

        with torch.no_grad(), get_autocast_context(cfg.precision):
            output = model(**batch["forward_kwargs"])
        pred_td = normalize_output_to_tensordict(output, target_config, output_type)

        ### Metrics in training space (matches the validation numbers).
        sample_metrics = {
            k: float(v.item())
            for k, v in metric_calculator(
                pred_td.float(), batch["targets"].float()
            ).items()
        }
        for k, v in sample_metrics.items():
            totals[k] += v
        count += 1

        ### Re-dimensionalize predictions + reference to physical units,
        ### then write them back onto the DomainMesh.
        pred_phys = redimensionalize(
            _to_pointwise(pred_td, output_type),
            normalizer=active_normalizer,
            nondim=nondim_helper,
            field_types=field_types,
            global_data=domain.global_data,
        )
        true_phys = redimensionalize(
            _to_pointwise(batch["targets"], output_type),
            normalizer=active_normalizer,
            nondim=nondim_helper,
            field_types=field_types,
            global_data=domain.global_data,
        )

        sample_id = _sample_id(metadata, idx)
        if is_rank0:
            attach_and_save(
                domain,
                pred_phys,
                true_phys,
                target_config,
                pred_dir / f"{sample_id}.pdmsh",
                rescale_geometry=bool(cfg.get("rescale_geometry", False)),
            )

        if is_rank0 and (i % log_every == 0 or i == n_samples - 1):
            metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in sample_metrics.items())
            logger.info(f"  [{i + 1}/{n_samples}] {sample_id}  {metrics_str}")
            if log_jsonl is not None:
                log_jsonl(
                    {
                        "phase": "sample",
                        "step": i,
                        "sample_id": sample_id,
                        "metrics": sample_metrics,
                    }
                )

    # -- Aggregate (all-reduce when distributed) --------------------------------
    if dist_manager.world_size > 1 and dist.is_initialized():
        keys = sorted(totals)
        packed = torch.tensor(
            [totals[k] for k in keys] + [float(count)], device=device
        )
        dist.all_reduce(packed)
        *sums, total_count = packed.tolist()
        totals = {k: s for k, s in zip(keys, sums)}
        count = int(total_count)

    averages = {k: totals[k] / max(count, 1) for k in sorted(totals)}
    if is_rank0:
        table = tabulate(
            [[k, f"{v:.6f}"] for k, v in averages.items()],
            headers=["Metric", "Value"],
            tablefmt="pretty",
        )
        logger.info(f"\nInference metrics over {count} samples:\n{table}\n")
        if log_jsonl is not None:
            log_jsonl({"phase": "summary", "num_samples": count, "metrics": averages})

    logger.info(f"Inference complete! Predictions written to {pred_dir}")


if __name__ == "__main__":
    main()
