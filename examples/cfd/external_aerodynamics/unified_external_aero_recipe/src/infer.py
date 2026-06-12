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
  identical to the training / validation loop, so they line up with the
  numbers logged during training.  Physical, actionable quantities come
  from the re-dimensionalized fields written to disk and the integrated
  force / moment coefficients (CD/CL/...), not from the metric tables.
- ``CenterMesh``'s per-sample translation offset is not stored, so the
  written geometry is physical-*scale* (when ``rescale_geometry=true``)
  but remains centered at the origin.  Field values are unaffected.
- Inference runs at whatever ``sampling_resolution`` allows (default:
  effectively the full mesh).  Very large volume meshes may need a
  smaller cap to fit in memory.  Chunked / windowed inference is
  deliberately out of scope for this recipe -- use ``physicsnemo-cfd``
  for that.
- Under ``torchrun`` each rank runs its own sampler shard, writes that
  shard's predictions, and the metrics / force coefficients are
  all-reduced across ranks, so the full split is covered.  Per-sample
  JSONL rows for non-zero ranks land in ``metrics.rank<r>.jsonl``; the
  aggregate summary is written once on rank 0.  When the split size is
  not divisible by the world size, the samplers pad by replaying
  indices, so a few samples are processed on two ranks (both write the
  same prediction path) and the all-reduced averages count them twice.
- When the checkpoint directory carries ``norm_stats.pt`` (persisted by
  the trainer at save time), those training-time normalization stats are
  used -- overriding the dataset YAML's stats on mismatch -- so the
  normalization applied here always matches how the checkpoint was
  trained.
"""

import os
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from datasets import build_dataloaders, find_normalizer, load_dataset_config
from forces import ForceAccumulator, ForceContext
from metrics import MetricCalculator, resolve_metrics
from nondim import NonDimensionalizeByMetadata, NondimFieldType, freestream_scales
from omegaconf import DictConfig, OmegaConf
from output_normalize import IOType, normalize_output_to_tensordict, require_output_type
from tabulate import tabulate
from tensordict import TensorDict
from utils import (
    FieldType,
    get_autocast_context,
    make_jsonl_logger,
    recursive_to_device,
    set_seed,
)

from physicsnemo import datapipes  # noqa: F401 - registers ${dp:...} resolver
from physicsnemo.distributed import DistributedManager
from physicsnemo.mesh import DomainMesh
from physicsnemo.utils import load_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

### ---------------------------------------------------------------------------
### Checkpoint resolution
### ---------------------------------------------------------------------------


def resolve_checkpoint_path(cfg: DictConfig) -> str:
    """Resolve the directory `load_checkpoint` should scan.

    Prefers an explicit ``checkpoint_path``; otherwise reproduces the
    trainer's ``${checkpoint_dir}/${run_id}/checkpoints`` layout. Note
    ``checkpoint_dir`` should point at the *training* output directory
    (default ``runs``), not this script's ``output_dir``. ``run_id`` is
    always required (``main()`` validates it before calling this).
    """
    explicit = OmegaConf.select(cfg, "checkpoint_path", default=None)
    if explicit:
        return str(explicit)
    return os.path.join(str(cfg.checkpoint_dir), str(cfg.run_id), "checkpoints")


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
    ### Select without a default so a dataset with no `pipeline.transforms`
    ### node yields None -> [] (`OmegaConf.to_container` rejects the plain
    ### Python list a `default=[]` would produce).
    transforms_node = OmegaConf.select(ds_yaml, "pipeline.transforms", default=None)
    transforms = (
        OmegaConf.to_container(transforms_node, resolve=False)
        if transforms_node is not None
        else []
    ) or []
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

    return {rename_map.get(raw, raw): ftype for raw, ftype in nondim_fields.items()}


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


def _norm_stats_match(a: dict[str, dict], b: dict[str, dict]) -> bool:
    """True when two ``NormalizeMeshFields.stats`` dicts agree.

    Compares the field-name key sets and, per field, the ``type`` string
    and the ``mean`` / ``std`` values (``allclose``).
    """
    if a.keys() != b.keys():
        return False
    for name in a:
        sa, sb = a[name], b[name]
        if sa["type"] != sb["type"]:
            return False
        for stat in ("mean", "std"):
            va = torch.as_tensor(sa[stat], dtype=torch.float32)
            vb = torch.as_tensor(sb[stat], dtype=torch.float32)
            if va.shape != vb.shape or not torch.allclose(va, vb):
                return False
    return True


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
        mesh_part = next((p for p in parts if p.endswith((".pdmsh", ".pmsh"))), None)
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
### Aggregation
### ---------------------------------------------------------------------------


def _allreduce_sums(
    totals: dict[str, float], count: int, device: torch.device | str
) -> tuple[dict[str, float], int]:
    """All-reduce a ``{key: running_sum}`` dict and the sample count.

    No-op (returns a plain copy) when not running distributed. Folds every
    sum plus the count into a single collective.
    """
    if not (
        dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    ):
        return dict(totals), count
    keys = sorted(totals)
    packed = torch.tensor([totals[k] for k in keys] + [float(count)], device=device)
    dist.all_reduce(packed)
    *sums, total = packed.tolist()
    return {k: s for k, s in zip(keys, sums)}, int(total)


### ---------------------------------------------------------------------------
### Driver
### ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../conf", config_name="infer")
def main(cfg: DictConfig) -> None:
    """Run checkpoint inference over a split and write physical predictions.

    Args:
        cfg: Hydra config composed from ``conf/infer.yaml`` (see that file
            and this module's docstring for the knobs). Requires ``model=``,
            ``dataset=``, and ``run_id=`` on the CLI (``checkpoint_path=``
            optionally overrides where the weights are read from).
    """
    DistributedManager.initialize()
    dist_manager = DistributedManager()
    device = dist_manager.device
    is_rank0 = dist_manager.rank == 0
    logger = RankZeroLoggingWrapper(PythonLogger(name="inference"), dist_manager)

    set_seed(cfg.training.get("seed", None), rank=dist_manager.rank)

    ### Inference runs one dataset at a time: re-dimensionalization keys off
    ### the single ``cfg.dataset`` (its field types + normalization stats),
    ### so folding in ``extra_datasets`` with different conditioning would
    ### silently de-normalize those samples with the wrong scales.
    extra_datasets = list(OmegaConf.select(cfg, "extra_datasets", default=[]) or [])
    if extra_datasets:
        raise ValueError(
            f"Inference does not support `extra_datasets` (got {extra_datasets!r}). "
            f"Re-dimensionalization is per-dataset, so run one dataset at a time: "
            f"set `dataset=<name>` and leave `extra_datasets` empty."
        )

    ### Fail fast on checkpoint misconfiguration before the (expensive)
    ### dataset construction below. `run_id` is always required: it
    ### identifies the checkpoint run and namespaces the output directory
    ### (`checkpoint_path` only overrides where the weights are read from).
    run_id = OmegaConf.select(cfg, "run_id", default=None)
    if not run_id:
        raise ValueError(
            "`run_id=<name>` is required: it identifies the checkpoint run and "
            "namespaces the output directory. (Use `checkpoint_path=<dir>` to "
            "additionally override where the weights are read from.)"
        )
    ckpt_path = resolve_checkpoint_path(cfg)
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint directory {ckpt_path!r} does not exist. Check "
            f"`run_id` / `checkpoint_dir` (or set `checkpoint_path`)."
        )

    ### Reuse the trainer's loader assembly: this resolves the split via
    ### `val_split` (aliased to `infer_split` in the YAML), returns the
    ### NormalizeMeshFields normalizer, and auto-derives `cfg.out_dim`
    ### from the dataset's `targets:` block (needed before the model
    ### template's `out_dim: ${out_dim}` resolves). The train loader is
    ### built and discarded, but the train split must still resolve:
    ### `cfg.train_split` (default 'train') has to exist in the manifest
    ### and match at least one on-disk run even though inference never
    ### iterates it. On a machine holding only the inference split,
    ### override it, e.g. `train_split=test`.
    _train_loader, val_loader, normalizer, dataset_info = build_dataloaders(cfg)
    target_config: dict[str, FieldType] = dataset_info["targets"]
    output_type = require_output_type(cfg)

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    logger.info(f"Targets (from dataset YAML): {target_config}")

    # -- Output dir + JSONL logging ----------------------------------------------
    run_dir = Path(cfg.output_dir) / str(run_id)
    pred_dir = run_dir / "predictions"
    ### Every rank writes its own sampler shard's predictions, so every rank
    ### ensures the output dirs exist (mkdir is race-safe with exist_ok=True).
    pred_dir.mkdir(parents=True, exist_ok=True)
    ### Rank 0 owns the aggregate ``metrics.jsonl`` (config + summary + its
    ### own shard's per-sample rows); every other rank writes its shard's
    ### per-sample rows to ``metrics.rank<r>.jsonl`` so distributed shards
    ### are not silently dropped.
    log_jsonl = make_jsonl_logger(
        run_dir
        / ("metrics.jsonl" if is_rank0 else f"metrics.rank{dist_manager.rank}.jsonl")
    )

    # -- Model + checkpoint -----------------------------------------------------
    model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model.__class__.__name__} ({num_params:,} params)")

    loaded_epoch = load_checkpoint(path=ckpt_path, models=model, device=device)
    ### The trainer always saves with epoch >= 1, so 0 here means nothing
    ### was restored (`load_checkpoint` only logs in that case) -- e.g.
    ### `checkpoint_path` pointing at the run directory instead of its
    ### `checkpoints/` subdirectory. Without this check, inference would
    ### proceed on randomly initialized weights.
    if loaded_epoch == 0:
        raise FileNotFoundError(
            f"No checkpoint restored from {ckpt_path!r}: the directory exists "
            f"but holds no checkpoint for this model. Check `run_id` / "
            f"`checkpoint_dir` (or `checkpoint_path`)."
        )
    logger.info(f"Loaded checkpoint from {ckpt_path!r} (epoch {loaded_epoch}).")

    ### The trainer persists the normalizer's stats next to the weights
    ### (norm_stats.pt) so inference applies the exact training-time
    ### normalization even if the dataset YAML's stats are edited later.
    ### On mismatch the persisted stats win: every live NormalizeMeshFields
    ### is updated in place. `normalizer` comes from the train pipeline,
    ### which the val split shares in manifest mode; a directory-mode
    ### `val_datadir` builds its own instance, so the val pipeline's copy
    ### is updated too. This keeps the forward normalization, the
    ### training-space metrics, and the inverses consistent with training.
    norm_stats_path = Path(ckpt_path) / "norm_stats.pt"
    if normalizer is not None and norm_stats_path.exists():
        saved_stats = torch.load(norm_stats_path, weights_only=True)
        if not _norm_stats_match(normalizer.stats, saved_stats):
            logger.warning(
                f"Dataset-YAML normalization stats differ from the "
                f"training-time stats persisted at {str(norm_stats_path)!r}; "
                f"using the persisted stats."
            )
            val_normalizer = find_normalizer([val_loader.dataset])
            for live in {
                id(n): n for n in (normalizer, val_normalizer) if n is not None
            }.values():
                live.stats.clear()
                live.stats.update(saved_stats)

    if cfg.get("compile", False):
        model = torch.compile(model)
    model.eval()

    # -- Collate ----------------------------------------------------------------
    ### Reuse the loader's own collate (same input_type / forward_kwargs /
    ### targets contract) so inference can never drift from training/validation.
    ### The DomainMesh is kept by indexing the dataset per sample below (which
    ### yields the source ``(DomainMesh, metadata)`` pair), not by the collate;
    ### we run this collate on the 1-element list to get the batched
    ### forward kwargs.
    collate_fn = val_loader.collate_fn

    # -- Re-dimensionalization setup --------------------------------------------
    ### `field_types` is computed unconditionally because the force
    ### integration needs it to identify Cp / Cf even when written fields
    ### stay in training space; `active_*` gate the physical conversion.
    redimensionalize_on = bool(cfg.get("redimensionalize", True))
    recipe_root = Path(__file__).resolve().parent.parent
    ds_yaml = load_dataset_config(recipe_root / "datasets" / f"{cfg.dataset}.yaml")
    field_types = build_redim_field_types(ds_yaml)
    nondim_helper = (
        NonDimensionalizeByMetadata(fields=field_types) if field_types else None
    )
    active_normalizer = normalizer if redimensionalize_on else None
    active_nondim = nondim_helper if redimensionalize_on else None
    logger.info(
        f"Re-dimensionalization: {'on' if redimensionalize_on else 'off'} "
        f"(field types: {field_types}, "
        f"normalizer: {'yes' if normalizer is not None else 'no'})"
    )

    # -- Force / moment coefficient setup (surface cases) -----------------------
    force_cfg = OmegaConf.select(cfg, "force_coefficients", default=None)
    force_ctx = ForceContext.from_config(force_cfg, field_types, device)
    force_acc = ForceAccumulator()
    if force_ctx is not None:
        logger.info(
            f"Force coefficients: integrating Cp='{force_ctx.pressure_field}', "
            f"Cf='{force_ctx.shear_field}' "
            f"(reference_area={force_ctx.reference_area}, reference_length="
            f"{force_ctx.reference_length if force_ctx.reference_length is not None else 'L_ref'})"
        )
    elif force_cfg is not None and force_cfg.get("enabled", False):
        logger.info(
            "Force coefficients enabled but unavailable for this dataset "
            "(no pressure + shear surface fields); skipping."
        )

    # -- Metrics ----------------------------------------------------------------
    metric_calculator = MetricCalculator(
        target_config=target_config, metrics=resolve_metrics(cfg)
    )

    if is_rank0:
        log_jsonl(
            {
                "phase": "config",
                "model": model.__class__.__name__,
                "dataset": cfg.dataset,
                "infer_split": cfg.infer_split,
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

    ### Zero-fill the running sums with the calculator's deterministic key
    ### set so every rank packs the same tensor length into the final
    ### all-reduce -- even a rank whose sampler shard is empty (possible
    ### when world_size exceeds the split size). ForceAccumulator does the
    ### same for the force sums.
    totals: dict[str, float] = {k: 0.0 for k in metric_calculator.expected_keys()}
    count = 0
    sampling_cap = cfg.get("sampling_resolution", None)
    truncation_warned = False
    for i, idx in enumerate(sampler):
        sample = dataset[idx]
        domain, metadata = sample
        batch = recursive_to_device(collate_fn([sample]), device)

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

        pred_pts = _to_pointwise(pred_td, output_type)
        true_pts = _to_pointwise(batch["targets"], output_type)

        ### Integrated force / moment coefficients (surface cases). The
        ### ForceContext un-normalizes to Cp / Cf internally and returns
        ### None for non-surface samples.
        sample_forces = None
        if force_ctx is not None:
            sample_forces = force_ctx.coefficients(
                domain, pred_pts, true_pts, normalizer
            )
            if sample_forces is not None:
                force_acc.update(*sample_forces)
                ### Force magnitudes are only physical at full surface
                ### resolution (see forces.py): a vehicle cell count
                ### sitting exactly at the subsample cap means the surface
                ### was almost certainly truncated by the reader.
                if (
                    not truncation_warned
                    and sampling_cap is not None
                    and domain.boundaries["vehicle"].n_cells == sampling_cap
                ):
                    logger.warning(
                        f"Vehicle surface has exactly sampling_resolution="
                        f"{sampling_cap} cells, so it was likely subsampled; "
                        f"integrated force/moment coefficients cover only the "
                        f"kept cells and their magnitudes are not physical. "
                        f"Raise `sampling_resolution` for absolute CD/CL/CM."
                    )
                    truncation_warned = True

        ### Re-dimensionalize predictions + reference to physical units,
        ### then write them back onto the DomainMesh.
        pred_phys = redimensionalize(
            pred_pts,
            normalizer=active_normalizer,
            nondim=active_nondim,
            field_types=field_types,
            global_data=domain.global_data,
        )
        true_phys = redimensionalize(
            true_pts,
            normalizer=active_normalizer,
            nondim=active_nondim,
            field_types=field_types,
            global_data=domain.global_data,
        )

        ### Every rank writes its own shard. Sample ids embed the dataset
        ### index, so they are unique within and across ranks -- except when
        ### drop_last=False padding replays an index on a second rank, which
        ### then writes the same sample to the same path (see the module
        ### docstring's torchrun caveat). attach_and_save self-creates
        ### parent dirs.
        sample_id = _sample_id(metadata, idx)
        attach_and_save(
            domain,
            pred_phys,
            true_phys,
            target_config,
            pred_dir / f"{sample_id}.pdmsh",
            rescale_geometry=bool(cfg.get("rescale_geometry", False)),
        )

        ### One JSONL row per sample -- the documented metrics.jsonl
        ### contract. Console logging below is throttled by log_every.
        record: dict[str, Any] = {
            "phase": "sample",
            "step": i,
            "sample_id": sample_id,
            "metrics": sample_metrics,
        }
        if sample_forces is not None:
            record["forces"] = {"pred": sample_forces[0], "true": sample_forces[1]}
        log_jsonl(record)

        if is_rank0 and (i % log_every == 0 or i == n_samples - 1):
            metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in sample_metrics.items())
            if sample_forces is not None:
                pred_c, true_c = sample_forces
                metrics_str += (
                    f"  | CD(p/t)={pred_c['CD']:.4f}/{true_c['CD']:.4f}"
                    f"  CL(p/t)={pred_c['CL']:.4f}/{true_c['CL']:.4f}"
                )
            logger.info(f"  [{i + 1}/{n_samples}] {sample_id}  {metrics_str}")

    # -- Aggregate (all-reduce when distributed) --------------------------------
    totals, count = _allreduce_sums(totals, count, device)
    force_acc.totals, force_acc.count = _allreduce_sums(
        force_acc.totals, force_acc.count, device
    )

    averages = {k: totals[k] / max(count, 1) for k in sorted(totals)}
    if is_rank0:
        table = tabulate(
            [[k, f"{v:.6f}"] for k, v in averages.items()],
            headers=["Metric", "Value"],
            tablefmt="pretty",
        )
        logger.info(
            f"\nInference metrics (training space / normalized) over "
            f"{count} samples:\n{table}\n"
        )
        log_jsonl(
            {
                "phase": "summary",
                "space": "training",
                "num_samples": count,
                "metrics": averages,
            }
        )

        ### Force / moment coefficient summary (surface cases only).
        if force_acc.count > 0:
            rows, coeff_summary = force_acc.summary()
            ftable = tabulate(
                rows,
                headers=["Coeff", "pred (mean)", "true (mean)", "MAE"],
                tablefmt="pretty",
            )
            logger.info(
                f"\nForce / moment coefficients over {force_acc.count} samples:\n"
                f"{ftable}\n"
            )
            log_jsonl(
                {
                    "phase": "forces_summary",
                    "num_samples": force_acc.count,
                    "coefficients": coeff_summary,
                }
            )

    logger.info(f"Inference complete! Predictions written to {pred_dir}")


if __name__ == "__main__":
    main()
