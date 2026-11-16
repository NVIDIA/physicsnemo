# Configuration layout

Every experiment is a Hydra recipe at the top of `conf/` that composes the
shared groups (`reader/`, `datapipe/`, `model/`, `training/`, `inference/`) and
overrides what the experiment changes. Print any recipe without starting
training with `python train.py --config-name=<recipe> --cfg job`.

Recipe names start with their dataset: `px_yaris_*` is the full-vehicle
Toyota Yaris dataset (Zarr stores), `biw_yaris_*` the body-in-white Yaris
VTP data and `bumper_*` the bumper beam.

## DeFormer on the full-vehicle Yaris dataset (`px_yaris_*`)

| Recipe | Model | What it trains |
| --- | --- | --- |
| `px_yaris_deformer` | `model/deformer_time_conditional.yaml` (`models.rollout.DeFormerTimeConditional`) | the DeFormer network: 60 epochs, physical MSE, provisional-head loss 0.1 |
| `px_yaris_deformer_corrector` | `model/deformer_corrector.yaml` (`models.rollout.ResidualTimeConditional` around `physicsnemo.experimental.models.deformer.DeFormerCorrector`) | the contact-aware corrector on the frozen network: 30 epochs, one warm-up epoch, wall clamp at evaluation |
| `px_yaris_geotransolver` | `model/geotransolver_time_conditional.yaml` (`models.rollout.GeoTransolverTimeConditional`) | the point-model baseline under the network's protocol |

`px_yaris_deformer` is self-contained (reader `px_yaris_full`, datapipe `px_yaris_full`,
training defaults plus the keys below); the other two inherit it. The four
required paths are the `yaris.*` keys; everything else has a value.

### Keys of `px_yaris_deformer`

| Key | Value | Meaning |
| --- | --- | --- |
| `yaris.data_root`, `yaris.split_manifest`, `yaris.outlier_mask`, `yaris.work_dir` | required | stores, `splits.csv`, `outlier_nodes.npz` (or `null`), run directory (an absolute path: Hydra changes into it) |
| `yaris.stride`, `yaris.dt` | 5, 0.005 | every fifth 1 ms frame; the frame interval in seconds (`stride * reader.raw_dt`), read by the datapipe and, through it, by the time-conditional models |
| `training.trainer_class` | `trainers.px_yaris_trainer.YarisFullTrainer` | exact, horizon-aware validation over every validation design |
| `training.num_training_samples` / `num_validation_samples` | 69 / 20 | the manifest's split sizes |
| `training.num_time_steps` | 31 | frame bound per case (`ceil(152 / 5)`); the time label `k / 30` is normalized by it |
| `training.optimizer`, `muon_min_dim` | `muon`, 9 | Muon for matrices, AdamW for vectors and matrices narrower than 9 |
| `training.start_lr`, `end_lr`, `epochs` | 2e-4, 1e-6, 60 | cosine schedule per epoch |
| `training.lr_warmup_epochs` | 0 | linear per-step warm-up before the cosine (the corrector uses 1) |
| `training.amp`, `amp_dtype`, `manual_gradient_allreduce` | true, bfloat16, true | bf16 autocast; gradients averaged after backward |
| `training.seed`, `sampler_seed`, `reproducible_sampling` | 42, 0, true | paired seeds: `training.seed=43 training.sampler_seed=1` |
| `training.loss`, `loss_space`, `loss_scale_mm` | `mse`, `physical`, 10 | MSE of the position error in millimetres over one 10 mm scale (`training.loss=relative_l2` is the documented alternative) |
| `training.auxiliary_loss_weight` | 0.1 | weight of the position loss on the provisional head |
| `training.validation_freq`, `save_checkpoint_freq` | 5, 5 | epochs |
| `datapipe.sample_type` | `one_time_step` | one (case, future frame) pair per sample, from the initial frame |
| `datapipe.local_knn_k`, `local_knn_metric`, `local_knn_cache_dir` | 32, `euclidean`, `<stats_dir>/local_knn_cache` | the static neighbourhoods, built once and cached by content hash; the kNN-attention models read `k` and the metric from here |
| `datapipe.contact_surface`, `mesh_edges`, `rigid_wall_position` | false, false, null | graph attachments the corrector and the metrics need (see below) |

### Keys of `model/deformer_time_conditional.yaml`

| Key | Value | Meaning |
| --- | --- | --- |
| `functional_dim`, `out_dim`, `global_dim` | 36, 3, 32 | coords + thickness + 32-channel time code; the time code as the global token |
| `n_hidden`, `n_head`, `n_layers`, `slice_num`, `attention_type` | 256, 8, 6, 128, `GALE_FA` | the FLARE backbone (original FLARE, not FLARE++) |
| `time_embedding`, `time_fourier_bands`, `time_dim`, `time_as_global` | `fourier`, 8, 32, true | the learned Fourier time code |
| `num_time_steps`, `dt` | `${training.num_time_steps}`, `${oc.select:datapipe.dt,null}` | frame bound and the dataset's frame interval (null for a datapipe without one) |
| `mesh_hidden_dim`, `num_pre_processor_layers`, `num_post_processor_layers` | 256, 1, 1 | width and depth of the local stages (stage B is an independent copy of stage A) |
| `mesh_pre_residual_gate_init`, `mesh_post_residual_gate_init`, `mesh_context_fusion` | 0, 1, `pre_post` | residual gates and FiLM conditioning on the mesh context |
| `local_neighborhood`, `local_knn_k` | `${datapipe.local_knn_metric}`, `${datapipe.local_knn_k}` | static kNN attention over the datapipe's neighbourhoods (Euclidean for the Yaris stores, whose edges cover shells only) |
| `local_num_heads`, `local_mlp_ratio`, `local_edge_dim`, `local_activation` | 8, 2, 32, `gelu` | attention block |
| `local_attention_fused`, `local_attention_backend` | true, `auto` | the fused kernel (Triton on CUDA) |
| `activation_checkpointing`, `checkpointing_ratio`, `activation_checkpointing_components`, `num_*_processor_checkpoint_segments` | true, 1.0, `[blocks, context]`, 1 | memory policy of the 1.48M-node forward |
| `clamp_to_wall`, `wall_axis`, `wall_sign` | false, 0, 1 | evaluation-time rigid-wall clamp (needs `datapipe.rigid_wall_position`) |

The body-in-white recipe `biw_yaris_deformer` uses the same model file over
the datapipe's geodesic neighbourhoods (`datapipe.local_knn_metric=geodesic`)
without activation checkpointing; its `dt` is null because the VTP datapipe
has no frame interval.

### Keys of `px_yaris_deformer_corrector` and `model/deformer_corrector.yaml`

| Key | Value | Meaning |
| --- | --- | --- |
| `model.teacher` | `model/deformer_time_conditional.yaml` | the network's architecture (frozen, weights from `teacher_weights`) |
| `model.teacher_weights` | required | the network's `.mdlus` checkpoint |
| `model.functional_dim` | 39 | coords + thickness + time code + the network's displacement |
| `model.hidden_dim`, `processor_size`, `aggregation`, `num_processor_checkpoint_segments` | 128, 8, `sum`, 8 | the MeshGraphNet processor |
| `model.use_contact`, `contact_activation_distance`, `contact_reference_clearance`, `node_contact_radius` | true, 5, 10, 10 | contact edges from the crossing-aware search on the predicted shape: 5 mm band, 10 mm reference clearance, features over 10 mm |
| `model.contact_side_from_reference`, `contact_side_symmetric`, `contact_crossing_radius`, `contact_clearance_approach_only` | true, false, null, false | the side test and its corrections (see `physicsnemo.experimental.nn.shell_contact.ShellContactSearch`) |
| `model.contact_bidirectional`, `contact_search_implementation`, `contact_surface_max_pairs` | true, `warp`, 8e6 | both message directions; search backend and budget |
| `model.dt`, `local_knn_k`, `local_neighborhood` | from the datapipe | the teacher's frame interval and neighbourhoods (`${oc.select:datapipe.dt,null}`, `${datapipe.local_knn_k}`, `${datapipe.local_knn_metric}`) |
| `model.final_clamp_to_wall`, `wall_axis`, `wall_sign` | true (recipe), 0, 1 | wall clamp of the corrected shape at evaluation |
| `datapipe.contact_surface`, `mesh_edges`, `rigid_wall_position` | true, true, 100 | shell facets and exclusions, mesh edges, the Yaris barrier plane |
| `training.epochs`, `lr_warmup_epochs`, `validation_freq`, `save_checkpoint_freq` | 30, 1, 2, 1 | the corrector's budget |
| `training.auxiliary_loss_weight` | 0 | no provisional head |
| `training.validation_penetration` | enabled, tolerance 0.5 mm, band 5 mm, clearance 10 mm, no crossing sweep | target-relative penetration metrics in the validation record; the crossing depth the search can still see is bounded (band plus half thicknesses, recorded as `crossing_depth_ceiling_mm`), `crossing_radius_mm` lifts the bound at a large cost |

`px_yaris_geotransolver` sets the point model's time code, width and memory policy
to the network's; the datapipe still builds the neighbourhoods (a dataset
product shared with the network's work directory), which the point model
ignores.

### Evaluation keys (`evaluate_px_yaris.py`)

| Override | Effect |
| --- | --- |
| `training.validation_output_dir=<dir>` | where the record goes (required; the run's own records are never written) |
| `hydra.run.dir=<dir>/hydra` | required inside the output directory: the recipe's run directory is the training run, whose `.hydra` files an evaluation must not overwrite |
| `training.evaluation_split=test` | score the manifest's test split instead of the validation split |
| `datapipe.stats_dir=<run>/stats` | the run's statistics and neighbourhood cache (default `${yaris.work_dir}/stats`), loaded and never recomputed |
| `training.ckpt_path=<run>/checkpoints`, `training.load_epoch=<E>` | the checkpoint to score |
| `datapipe.contact_surface=true` | shell facets for the penetration metrics of a contact-free model |
| `training.validation_penetration.wall_position_mm=100` | wall family for models whose recipe has no `rigid_wall_position` |
| `training.validation_penetration.crossing_radius_mm=<mm>` | crossing sweep of the penetration search: crossings deeper than the band plus the half thicknesses become visible up to this radius (about thirty times the search cost) |
| `training.validation_node_errors=true` | per-node RMSE map and per-region translational share |
| `training.validation_region_bounds=[-2000,-1000]` | reference-x bounds of the regions |
| `training.warm_start_model=<file>.mdlus` | score a model-only checkpoint (no `checkpoint.0.<E>.pt` next to it); `training.ckpt_path` may then be an empty directory |

Every key above is declared in `training/default.yaml`, so plain overrides
compose on every recipe. Only the requested split is built, in load mode (the
frozen statistics, the kNN statistics and the neighbourhood cache are read from
`datapipe.stats_dir`; a missing one fails instead of being rebuilt, and nothing
is written there); no training dataset, optimizer or TensorBoard writer is
created. The record is `<dir>/epoch_<EEEE>.json` (zero-padded,
e.g. `epoch_0030.json`); its `penetration` block carries `complete: false`,
the ranks' `failure_reasons` and `null` values when the probe failed anywhere,
and `evaluate_px_yaris.py` then exits with an error.

### Training-loop keys (`training/default.yaml`)

| Key | Default | Effect |
| --- | --- | --- |
| `training.restore_rng` | true | restore the checkpoint's RNG states on resume; false after a change of the GPU count |
| `training.data_startup_timeout_seconds` | 3600 | how long the ranks wait while rank 0 prepares statistics and caches at startup |
| `training.loss`, `loss_space`, `loss_scale_mm` | `mse`, `normalized`, 10 | the loss (`relative_l2` is the alternative) and its space (`physical`: millimetres over `loss_scale_mm`) |
| `training.auxiliary_loss_weight` | 0 | weight of the loss on a model's auxiliary prediction (the DeFormer's provisional head) |
| `training.reproducible_sampling`, `sampler_seed` | false, 0 | epoch-seeded sample order independent of the rank count |
| `training.deterministic` | false | deterministic cuDNN/cuBLAS kernels |
| `training.manual_gradient_allreduce` | false | average the gradients after backward instead of DDP's bucketed reduction |
| `training.muon_min_dim` | 0 | Muon: 2-D parameters with a dimension below this stay on AdamW |
| `training.trainer_class` | null | dotted path of a `train.Trainer` subclass (null: `train.Trainer`) |
| `training.load_epoch`, `warm_start_model` | null, null | the checkpoint epoch to load (null: the latest); `.mdlus` weights to start from when `ckpt_path` holds no training checkpoint |
| `training.validation_output_dir`, `validation_node_errors`, `validation_region_bounds`, `validation_penetration` | null, false, null, off | the validation suite's keys (see [Evaluation keys](#evaluation-keys-evaluate_px_yarispy)); the recipes that use them set their values |
| `training.evaluation_split` | `validation` | the manifest split `evaluate_px_yaris.py` scores (`validation` or `test`) |
| `reader.include_contact_topology` (`reader/vtp.yaml`) | false | also cache the shell cells of the VTP files, for `datapipe.contact_surface` |
| `datapipe.contact_surface`, `contact_surface_exclusion`, `rigid_wall_position` (`datapipe/graph.yaml`) | false, `incidence`, null | contact topology and rigid wall on the structural graph |
| `datapipe.local_knn_k`, `local_knn_metric`, `local_knn_cache_dir` (`datapipe/graph.yaml`) | null, `geodesic`, null | static kNN neighbourhoods for kNN attention; null keeps the MeshGraphNet graph unchanged |
| `training.num_training_samples`, `num_validation_samples` | `???`, 0 (69 and 20 in `px_yaris_deformer`) | the manifest's split sizes; a larger train split is truncated to its first N cases (the frozen statistics still use every train case), the validation count must equal the manifest's |

Re-running a training command resumes from `<work_dir>/checkpoints`.
`px_yaris_prepare.py` reads the environment variable `YARIS_STATS_WORKERS`
(default 8) for its statistics workers.

## Body-in-white Yaris (VTP) recipes (`biw_yaris_*`)

| Recipe | Model | Purpose |
| --- | --- | --- |
| `biw_yaris_deformer` | `model/deformer_time_conditional.yaml` over the datapipe's geodesic neighbourhoods, no activation checkpointing (`dt` null: the VTP datapipe has no frame interval) | the DeFormer network on the body-in-white data: 26 frames, 127 training and 8 validation cases, 100 epochs, Muon, normalized MSE with provisional-head loss 0.1 |
| `biw_yaris_geoflare_oneshot`, `biw_yaris_geotransolver_oneshot`, `bumper_*` | one-shot and time-conditional point models | the original body-in-white (formerly `crash_*`) and bumper examples |

The recipe is self-contained (reader `vtp`, datapipe `graph`, training defaults
plus its own keys); the required paths are `training.raw_data_dir` and
`training.raw_data_dir_validation` (`inference.raw_data_dir_test` for
`inference.py`). `datapipe.local_knn_metric=euclidean` and `datapipe.local_knn_k=16`
change the neighbourhood metric and size (the model reads both from the
datapipe); the datapipe builds the matching neighbourhoods and caches them
under `datapipe.local_knn_cache_dir`.

`evaluate_biw.py` scores a body-in-white time-conditional checkpoint with
the PX (PhysicsX) Yaris validation suite (one implementation, `metrics/validation_suite.py`,
serves the Yaris trainer and both evaluators; both evaluators require
`hydra.run.dir` inside `training.validation_output_dir`); the penetration
metrics need the shell facets
(`reader.include_contact_topology=true datapipe.contact_surface=true
datapipe.contact_surface_exclusion=one_ring datapipe.rigid_wall_position=-29.8`,
see the script's docstring). Its per-region readouts follow the recipe's
`training.validation_region_bounds`, `[-3000, -1500]` for the body-in-white
mesh (the PX Yaris recipes use `[-2000, -1000]`); the recipe turns on
`training.validation_node_errors` and `training.validation_penetration.enabled`
for the script (its own training validation is the pooled MSE).

## Shared components

| Directory | Purpose |
| --- | --- |
| `model/` | architecture defaults (`deformer_time_conditional`, `deformer_corrector`, `geotransolver_time_conditional`, the one-shot and legacy rollout models) |
| `datapipe/` | `px_yaris_full` (Zarr, full mesh), `graph` and `point_cloud` (VTP) |
| `reader/` | `px_yaris_full` (Zarr stores with a shared mesh; `include_solid_edges`, `connect_isolated_k` for geodesic neighbourhoods), `vtp`, `zarr` |
| `training/` | optimizer, schedule and checkpoint defaults |
| `inference/` | evaluation and VTP export defaults |
