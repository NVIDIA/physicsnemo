# DeFormer: deformation transformer with a contact-aware corrector

This is the model-specific documentation of the crash example; the pipeline it
runs on (readers, datapipes, trainer, inference) is described in the
[example README](../README.md). Commands below run from the example directory,
`examples/structural_mechanics/crash`.

**DeFormer** is the crash surrogate of this example: a time-conditional
deformation transformer (the *network*) followed by a decoupled, contact-aware
MeshGraphNet *corrector*. Given the undeformed mesh, the shell thickness and a
query time, it predicts the deformed shape of a full vehicle at that time; every
frame of a crash is predicted independently from the initial mesh, so there is
no error accumulation and the frames can be computed in parallel. On the
full-resolution Toyota Yaris frontal-impact dataset (99 thickness-variation
designs, 1.48M nodes per design, 5 ms frames) the network reaches a pooled
validation RMSE of 7.6 mm against 10.1 mm for a GeoTransolver baseline of the
same width and depth, and the corrector lowers it to 7.5 mm while cutting the
spurious shell crossings about sevenfold. The paper, "DeFormer: Deformation
Transformers with Contact-Aware Local Corrections for Full-Vehicle Crash
Simulation", is in preparation; a link will follow.

The network (`deformer.time_conditional.DeFormerTimeConditional`,
`conf/model/deformer_time_conditional.yaml`):

* a GeoTransolver/FLARE global stage (`attention_type: GALE_FA`, 6 blocks,
  width 256, 128 slices) made time-conditional with a learned Fourier time code
  (8 bands, 32 channels) that enters every node's input and the global token;
* local stages of static kNN neighbourhood attention over the undeformed mesh
  (k = 32 Euclidean neighbours, the fused Triton kernel of
  `physicsnemo.experimental.models.deformer`): a pre-processor before the
  backbone and a two-stage decoder after it, where stage A emits a provisional
  shape through a zero-initialised head and stage B, an independent copy of
  stage A, adds a gated correction;
* a physical-space MSE loss (position error in millimetres over one 10 mm
  scale for all three axes) plus 0.1 times the same loss on the provisional
  shape; Muon/AdamW, cosine schedule, 60 epochs, bf16 autocast, activation
  checkpointing of the FLARE blocks and the geometry context so a 1.48M-node
  forward fits one GPU.
  9,672,505 parameters.

The corrector (`deformer.residual.ResidualTimeConditional` around
`physicsnemo.experimental.models.deformer.DeFormerCorrector`,
`conf/model/deformer_corrector.yaml`):

* the frozen network is recomputed on the fly (no prediction files); the
  corrector reads its predicted shape through message passing on the shell
  mesh edges (reference geometry, predicted geometry and their difference per
  edge) and on *contact edges* discovered on the predicted shape by the
  crossing-aware node-to-facet search of
  `physicsnemo.experimental.nn.shell_contact` (5 mm band, 10 mm reference
  clearance so that assembled flanges do not count as contact, both message
  directions);
* 8 layers of width 128, zero-initialised decoder (epoch 0 reproduces the
  network), 30 epochs with a one-epoch learning-rate warm-up. 1,250,659
  parameters;
* at evaluation the corrected positions are clamped at the rigid wall plane
  (`model.final_clamp_to_wall`, `datapipe.rigid_wall_position: 100`), which
  removes the remaining wall crossings at unchanged RMSE.

The library pieces DeFormer relies on are PhysicsNeMo's: the fused kNN
attention kernel, activation checkpointing of the FLARE blocks, `physicsnemo.Module`
serialization (`.mdlus` checkpoints) and the Warp contact-search kernels behind
the `shell_contact` functional.

## Recipes

| Recipe | Model | Purpose |
| --- | --- | --- |
| `px_yaris_deformer` | network | the DeFormer network on the Yaris mesh (60 epochs) |
| `px_yaris_deformer_corrector` | network + corrector | the contact-aware corrector on the frozen network (30 epochs) |
| `px_yaris_geotransolver` | GeoTransolver (FLARE variant) | the point-model baseline under the identical protocol |
| `biw_yaris_deformer` | network | the same network on the body-in-white VTP data (26 frames, 100 epochs) |

Recipe names start with their dataset: `px_yaris_*` is the full-vehicle
Toyota Yaris dataset (Zarr stores) and `biw_yaris_*` the body-in-white Yaris
VTP data. The three PX Yaris recipes share one data recipe
(`conf/reader/px_yaris_full.yaml`, `conf/datapipe/px_yaris_full.yaml`) and
differ only in the model group and the corrector's additions; `conf/README.md`
lists every key.

## Data preparation

The Yaris recipes read one Zarr store per design with a shared reference mesh:
`nodes/pos` (`[T, N, 3]` positions in mm at 1 ms), `nodes/thickness` (`[N]`),
`edges` (`[E, 2]` shell edges), `shell/connectivity` (`[S, 4]` quads,
triangles with the last id repeated), optionally `solid/connectivity`
(`[H, 8]` hexahedra) and optionally `nodes/part_id` (`[N]`, copied into the
node metadata written with the node-error maps). The Yaris stores are delivered in
this layout by the dataset's own preparation pipeline, together with the two
files below. It is not the layout PhysicsNeMo-Curator's crash ETL writes
(`mesh_pos`, `thickness`, `edges`, read by `zarr_reader.py` for the VTP-derived
recipes; see [Data Preprocessing](../README.md#data-preprocessing)); a store in another
layout is rejected with a message naming the missing arrays. Two more files are
needed:

* `splits.csv` with columns `split,sample_name` (`train`, `validation`, `test`)
  naming the stores: the dataset creator's split is the authority, and a case
  that must be left out is omitted from this file rather than special-cased in
  code;
* `outlier_nodes.npz` with a boolean `exclude` array over the nodes: rigid-wall
  nodes and fly-away nodes that are dropped from inputs, targets and
  neighbourhoods (pass `yaris.outlier_mask=null` to keep every node).

Every PX Yaris recipe takes the same four paths: `yaris.data_root` (the stores),
`yaris.split_manifest`, `yaris.outlier_mask` and `yaris.work_dir` (statistics,
neighbourhood caches, checkpoints and validation records of the run). Give
`yaris.work_dir` as an absolute path: Hydra changes into it, and a relative
path would be resolved again for the statistics, checkpoint and TensorBoard
directories and nest the outputs under themselves (the trainer rejects it). Prepare
the frozen normalization statistics and the kNN neighbourhood cache once, on a
CPU node (minutes for 1.48M nodes):

```bash
cd examples/structural_mechanics/crash
python px_yaris_prepare.py --config-name px_yaris_deformer \
  yaris.data_root=/data/yaris_zarr yaris.split_manifest=/data/splits.csv \
  yaris.outlier_mask=/data/outlier_nodes.npz yaris.work_dir=/runs/network
```

The statistics carry an identity (split manifest, outlier mask, kept node
count, stride, dt); a changed mask or split invalidates them. The corrector
and the baseline can point `yaris.work_dir` at their own directories and reuse
the network's statistics by copying `stats/` or by overriding
`datapipe.stats_dir`. `px_yaris_prepare.py` reads the environment variable
`YARIS_STATS_WORKERS` (default 8) for its parallel statistics workers. The
prepare step may be skipped: rank 0 then computes the statistics serially when
training starts while the other ranks wait, bounded by
`training.data_startup_timeout_seconds` (one hour by default).

## Train the network

```bash
torchrun --nproc_per_node=<NUM_GPUS> train.py --config-name px_yaris_deformer \
  yaris.data_root=/data/yaris_zarr yaris.split_manifest=/data/splits.csv \
  yaris.outlier_mask=/data/outlier_nodes.npz yaris.work_dir=/runs/network
```

The reference run used 16 GPUs for 60 epochs (about 6.6 h on GB300s). Pair
seeds with `training.seed=43 training.sampler_seed=1`. Checkpoints land in
`<work_dir>/checkpoints/` as `DeFormerTimeConditional.0.<epoch>.mdlus`
plus the optimizer state; the exact held-out validation (every validation
design once, pooled over its own horizon) writes `<work_dir>/validation/epoch_<EEEE>.json`
(zero-padded, e.g. `epoch_0060.json`) every `training.validation_freq` epochs.
Re-running the same command resumes from `<work_dir>/checkpoints` with the
optimizer and RNG states; after a change of the GPU count pass
`training.restore_rng=false`. The reference runs are the only measured
configuration (16 GB300 GPUs with activation checkpointing on); the per-GPU
memory of the 1.48M-node forward on smaller cards has not been measured, and
the exact validation additionally holds about three float64 `[N, T, 3]`
tensors of metric overhead (about 1.1 GB each; about 4.4 with the node-error
maps and regions on) on every device.

At full-vehicle scale the three neighbourhood-attention processors dominate the
activation memory: with the recipe widths under bf16 each one-layer processor
would hold about 20 GB of activations after the forward pass without
checkpointing, about 60 GB for the three. The recipe runs every local
processor as one checkpoint segment (`num_pre_processor_checkpoint_segments`,
`num_post_processor_checkpoint_segments`), and the edge encoders are
recomputed inside the segments, so the held activation memory stays at a few
gigabytes and the peak near 50 GB; without the segments the time-conditional
recipe does not fit an 80 GB card.

## Train the corrector

```bash
torchrun --nproc_per_node=<NUM_GPUS> train.py \
  --config-name px_yaris_deformer_corrector \
  yaris.data_root=/data/yaris_zarr yaris.split_manifest=/data/splits.csv \
  yaris.outlier_mask=/data/outlier_nodes.npz yaris.work_dir=/runs/corrector \
  datapipe.stats_dir=/runs/network/stats \
  model.teacher_weights=/runs/network/checkpoints/DeFormerTimeConditional.0.60.mdlus
```

The recipe turns on the contact topology (`datapipe.contact_surface`), the mesh
edges (`datapipe.mesh_edges`), the rigid wall (`datapipe.rigid_wall_position`)
and the evaluation-time wall clamp.

## Evaluate

`evaluate_px_yaris.py` scores a checkpoint with the full validation suite
without touching the run's own records: pooled and per-frame vector RMSE in
millimetres, per-case and per-region breakdowns, the target-relative
penetration statistics of the shell surface and the rigid wall (only
penetration the ground truth does not have counts against a model) and,
optionally, node-error maps:

```bash
torchrun --nproc_per_node=<NUM_GPUS> evaluate_px_yaris.py \
  --config-name px_yaris_deformer_corrector \
  yaris.data_root=/data/yaris_zarr yaris.split_manifest=/data/splits.csv \
  yaris.outlier_mask=/data/outlier_nodes.npz yaris.work_dir=/runs/corrector \
  datapipe.stats_dir=/runs/network/stats \
  model.teacher_weights=/runs/network/checkpoints/DeFormerTimeConditional.0.60.mdlus \
  training.ckpt_path=/runs/corrector/checkpoints training.load_epoch=30 \
  training.validation_output_dir=/runs/eval/corrector \
  hydra.run.dir=/runs/eval/corrector \
  training.validation_node_errors=true
```

For the network or the baseline use their recipe with `datapipe.contact_surface=true`
(the penetration metrics need the shell facets; the models ignore them) and
`training.validation_penetration.wall_position_mm=100` for the wall family.
The record is `<validation_output_dir>/epoch_<EEEE>.json` (zero-padded). The
penetration metrics are the script's purpose: a failure of the probe on any
rank (pair budget, out of memory, a non-finite frame) leaves `null` values and
the ranks' `failure_reasons` in the record's `penetration` block, and the
script then exits with an error.

The surface crossing depth has a detection ceiling. A node is paired with a
facet only while its unsigned distance to it lies inside the 5 mm band, so a
node that has travelled further through a shell than the band plus the half
thicknesses of the pair (about 6 mm with 1 mm shells) has no pair and reads as
clean; a part pushed deep through another is under-counted. The record states
the ceiling next to the crossing numbers (`crossing_depth_ceiling_mm`, computed
with the mean thickness of the validated designs). The opt-in key
`training.validation_penetration.crossing_radius_mm=<mm>` turns on the crossing
sweep of the search, which keeps candidates beyond the band up to that radius
when the side test certifies a crossing and lifts the ceiling accordingly, at
about thirty times the search cost at full-vehicle scale.

A model-only checkpoint, the `.mdlus` file without the `checkpoint.0.<E>.pt`
training state (the form in which trained weights are distributed), is scored
with `training.warm_start_model=<file>.mdlus`; `training.ckpt_path` may then
point at an empty directory. The corrector's `.mdlus` can also be loaded
directly with
`physicsnemo.Module.from_checkpoint(path, override_args={"teacher_weights": "<network .mdlus>"})`;
without the override it looks for the network's `.mdlus` under the file name
recorded in the checkpoint (the delivered corrector records
`DeFormerTimeConditional.0.60.mdlus`, resolved from the working directory).

## The network on the body-in-white data

`biw_yaris_deformer` trains the same network on the
body-in-white VTP data (26 frames, 127 training and 8 validation cases) with
geodesic neighbourhoods on the structural edges and without activation
checkpointing; the recipe is self-contained and needs only the two data paths:

```bash
torchrun --nproc_per_node=<NUM_GPUS> train.py --config-name biw_yaris_deformer \
  training.raw_data_dir=/data/crash/train training.raw_data_dir_validation=/data/crash/validation
```

`evaluate_biw.py` scores a body-in-white checkpoint with the PX Yaris
validation suite; the command in its docstring turns on the shell facets and
the one-ring exclusions that the penetration metrics need.
