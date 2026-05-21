# Functional Generative Networks (FGN) Plan

This document outlines a plan to add an `examples/weather/fgn/` implementation to PhysicsNeMo, following the style of the existing weather examples and informed by:

- Ferran Alet et al., "Skillful joint probabilistic weather forecasting from marginals" (arXiv:2506.10772v1)
- HTML source: https://arxiv.org/html/2506.10772v1

## Progress

Current status as of 2026-05-21:

- Done: created the initial `examples/weather/fgn/` scaffold with `train.py`,
  `inference.py`, `README.md`, Hydra configs, a synthetic dataset, a
  latent-conditioned PhysicsNeMo `Module`, fair-CRPS loss, and a smoke test.
- Done: aligned the example direction explicitly to WeatherNext 2 as the target
  FGN-style model.
- Done: updated the variable naming plan so internal training and inference
  should use PhysicsNeMo / Earth2Studio compact names such as `u10m`, `t2m`,
  `msl`, `z500`, `q850`, and `w300`.
- Done: surveyed PhysicsNeMo + Earth2Studio for reusable building blocks.
- Done: confirmed StormCast's FSDP + ShardTensor `ParallelHelper` pattern;
  adopted it directly in `utils/parallel.py`.
- Done: local CPU smoke test passes (`pytest examples/weather/fgn/test_training.py`).
- Done: fixed inference/training `latent_dim` mismatch.
- Done: ARCO-backed real dataset (`datasets/arco.py`) using
  `earth2studio.data.ARCO` + `earth2studio.lexicon.arco.ARCOLexicon`; full
  721×1440 0.25° grid, 6-hour step, 83-channel Table A.1 state.
- Done: `datasets/__init__.py` auto-discovery registry + `datasets/dataset.py`
  abstract base class (`FGNDataset`) mirroring StormCast conventions.
- Done: `utils/loss.py` — `fair_crps` (eq. 4), `ensemble_mean_mse`,
  `build_channel_weights`, `build_area_weights` (paper §2.2.3 GraphCast weights).
- Done: `tp06` predicted-only variable handled in dataset + trainer.
- Done: per-channel mean/std normalization stats (`scripts/compute_arco_stats.py`,
  `stats_path` config key, normalize/denormalize helpers).
- Done: FSDP + ShardTensor via `ParallelHelper` wired in trainer; smoke test
  and multi-GPU FSDP path both verified on Slurm H100 nodes.
- Done: AR finetune stage in trainer — rollout of K steps with BPTT,
  averaged fCRPS over rollout, N ensemble trajectories.
- Done: stage-4 AR scheduler (Table A.2): `8k·1AR → 4k·2AR → 1k·{3..8}AR`.
- Done: validation metrics + plots modelled on Figures 2+3 (CRPS, RMSE,
  spread-skill, rank histograms, power spectra, `metrics.npz`).
- Done: energy score metric added alongside fCRPS in `utils/metrics.py`.
- Done: deep ensemble inference across multiple checkpoints (paper §2.2.1).
- Done: bad-seed spectrum-based detector (`scripts/check_spectra.py`).
- Done: Hydra configs (`config/fgn.yaml`, `config/fgn_arco.yaml`);
  Pydantic-validated config types in `utils/config.py`.
- Done: ruff-clean, CHANGELOG, FGNUNet docstring, SPDX headers on all files.
- Done: bf16 AMP around model forward calls in trainer; training loop
  refactored to use `_make_train_iter()` (routes through `sharded_data_iter`
  when domain parallelism is active, matching StormCast convention).
- **In progress**: 5000-step `fgn_2024_long` training run on 2×H100
  (job 99807); batch_size=2 global (1 per GPU), domain_parallel_size=1 DDP,
  bf16 AMP, full 721×1440 resolution, 2024 ARCO data.
  First 10 steps confirmed: `train_loss=4.43` at step 10.

### Sequencing decision (2026-04-19)

Keep the placeholder U-Net + AdaGN modulation for v1. App. A.4 of the paper
shows that the *functional-noise + fCRPS* training regime is what makes FGN
beat GenCast, not the specific backbone (a 16-layer / 512-dim, single-seed,
no-AR FGN already beats GenCast on >90 % of targets). So the plan is:

1. Land the training/data side on top of the existing U-Net backbone.
2. Only then swap to a CLN graph-transformer — by which point the ARCO
   dataset, FSDP/ShardTensor wiring, AR finetune, loss weighting, and
   checkpoint/inference conventions are all settled and we know whether
   the backbone integrates cleanly (e.g. FSDP-compatibility of GNN ops).

Risk of this ordering: architectural integration issues (FSDP + GNN, mesh
graph construction under ShardTensor) are not surfaced until the end; if
they block, we may have to rework parts of the training loop built against
the U-Net.

## Task List (live)

Grouped roughly in execution order. Checked items are done; others are the
open backlog.

### Scaffold and smoke tests (done)
- [x] Example scaffold: `train.py`, `inference.py`, `test_training.py`,
  Hydra configs under `config/`, mock dataset, fCRPS loss, `FGNUNet`.
- [x] Variable naming plan pinned to compact Earth2Studio names.
- [x] `pytest test_training.py` smoke test passes locally (no srun, CPU).
- [x] Inference reads `latent_dim` from the loaded `Module` so test configs
  can't desync from inference configs.

### Training and data infra (next, on top of the current U-Net)
- [x] ARCO-backed real dataset class at `datasets/arco.py` using
  `earth2studio.data.ARCO` + `earth2studio.lexicon.arco.ARCOLexicon`; two
  prior frames, 6-hour step, compact internal names. Keep `MockFGNDataset`
  for CI. Config entry at `config/dataset/arco.yaml`. Defaults to the full
  Table A.1 83-channel state; `spatial_stride` dev knob for cheap local runs.
  Constructor + clock-feature + invariants plumbing unit-tested without
  network; a real network-fetch integration test is still pending.
- [ ] Add a slow/opt-in integration test that does one
  `ArcoFGNDataset.__getitem__` on a tiny grid to catch earth2studio API
  regressions (gated on GCS reachability; not in default smoke test).
- [x] Extend `ArcoFGNDataset` with `tp06` (predicted-only, 6-h accumulated)
  and SST NaN-imputation using the paper's global-min convention
  (pre-computed, not per-sample) to match Appendix A.1 exactly.
- [x] Add optional `stats_path` for per-channel mean/std, applied in
  `__getitem__`, with `normalize_state` / `denormalize_state` round-trip
  support for 3/4/5-D numpy and torch tensors. Stats file is a single
  `.npz` with `mean` + `std` arrays matching `state_variables` order.
  Helper script at `scripts/compute_arco_stats.py` computes stats online
  (Welford) from random ARCO timestamps with SST NaN imputation applied
  first, so training and stats see the same representation. Smoke test
  and normalize/denormalize unit check both pass.
- [ ] Verbose WeatherNext-name → compact-name mapper at the dataset
  boundary (e.g. `300_geopotential` → `z300`, `10m_u_component_of_wind` →
  `u10m`) so external configs can be authored in either convention.
- [x] Adopt StormCast's `ParallelHelper` pattern in `utils/parallel.py`:
  `distribute_model` (FSDP + `distribute_module`), `sharded_dataloader`,
  `sharded_data_iter`, `ShardTensor` scatter helpers. Gated on
  `training.domain_parallel_size` and `training.force_sharding`. Single-
  process smoke test still passes (no regression); the helper is bypassed
  entirely when `world_size == 1 and not use_shard_tensor`, so the no-init
  CPU path keeps working. A real multi-rank / multi-GPU run is not yet
  exercised — the helper is a line-for-line adaptation of StormCast's
  proven implementation (minus the diffusion noise-scheduler wrapping), so
  we trust the structural match and defer live FSDP verification to an
  actual multi-GPU training job.
- [x] Multi-rank sanity test: verified FSDP wrap + sharded loader +
  checkpoint save/load on real 2×H100 Slurm nodes (jobs 99807+).
- [x] AR finetune stage in the trainer: rollout of `K` steps with full
  BPTT, averaged fCRPS over the rollout. `training.ar_steps` (1–8) drives
  both dataset `future_frames` and the trainer's rollout loop; N parallel
  ensemble trajectories diverge as each member's own prediction is fed back
  as the next history frame. Verified locally at `ar_steps=1` (backward-
  compatible: matches pre-AR loss values exactly) and `ar_steps=2` (three
  steps, descending loss, checkpoint saved). Still pending: a config-
  driven stage scheduler mirroring Table A.2
  (`8k·1AR → 4k·2AR → 1k·{3..8}AR`), plus gradient checkpointing for
  memory-efficient AR at large K.
- [x] Multi-stage AR schedule runner: stage-4 Table A.2 scheduler
  (`8k·1AR → 4k·2AR → 1k·{3..8}AR`) implemented in trainer.
- [x] Paper-faithful per-variable / per-level loss weights (GraphCast
  weights with geopotential halved, per §2.2.3 of arXiv:2506.10772v1 +
  `Lam et al., 2022` / `Price et al., 2023`), plumbed through fCRPS.
  Implementation details:
    - fCRPS kernel delegates to
      `physicsnemo.metrics.general.crps.kcrps(..., biased=False)` (the
      core Zamo-Naveau fair estimator, same helper used by MoWE's
      `train_crps.py`), so we don't re-roll the pairwise formula.
    - `build_channel_weights(variables)` derives the scheme from compact
      ARCO names (`t2m`=1.0, other surface=0.1, atmospheric linear-by-
      level with `z{level}` halved); mirrors
      `physicsnemo.metrics.climate.graphcast_loss.GraphCastLossFunction.assign_variable_weights`
      but without the dataset-metadata-JSON dependency.
    - `build_area_weights(H)` gives `cos(lat)` normalized so mean=1 over
      latitudes (preserves loss scale when toggled); related to
      `physicsnemo.metrics.climate.reduction._compute_lat_weights`
      (sum=1 normalization).
    - Two config toggles: `training.loss.use_channel_weights` +
      `training.loss.use_area_weights` (both default false so smoke tests
      keep current behaviour).
  - `test_loss.py` covers eq. (4) against hand-computed references
    (N=2, 3, 5), eq. (5) reduction vs explicit (1/G) Σ a_i fCRPS_i,
    weight-builder invariants, and area-weight normalization.
    17/17 pass; smoke test unchanged.
- [x] Validation metrics + plots modelled on Figures 2 + 3 of
  arXiv:2506.10772v1. When `training.validation_metrics: true` the trainer
  runs an M-member ensemble rollout over all `ar_steps` lead times on a
  validation batch and writes:
    - `crps_vs_lead.png` — per-channel fCRPS vs lead (Figure 2a without
      baseline). Delegates to
      `physicsnemo.metrics.general.crps.kcrps(..., biased=False)`.
    - `rmse_vs_lead.png` — ensemble-mean RMSE per channel, companion to
      Figure 2.
    - `spread_skill_vs_lead.png` — Figure 2 b-f spread-skill ratio per
      channel vs lead, with horizontal line at 1.0.
    - `rank_histograms.png` — classical calibration diagnostic; uniform
      = well calibrated, U = under-dispersive, hump = over-dispersive.
    - `power_spectra_lead<K>.png` — Figure 3 e-j azimuthal 1D spectrum of
      ensemble-mean vs truth (2D-FFT approximation; honeycomb artifacts
      of Figure 5 would show as a mesh-frequency bump).
    - `metrics.npz` with every array + the variable list + lead steps.
    Includes derived-variable CRPS for `wspd10m` (Figure 3c) and
    `z300 - z500` (Figure 3d) when those variables are in the channel
    list. Canonical metric equivalents cited inline:
    `earth2studio.statistics.{crps, rmse, spread_skill_ratio,
    rank_histogram}` (coord-aware; the lightweight torch versions stay
    for inline trainer use). No shared plotting library exists in
    physicsnemo or earth2studio, so plots use matplotlib directly with
    the ``Agg`` backend — same convention as earth2studio examples and
    `examples/weather/stormcast/utils/plots.py`.
- [ ] Scorecards vs a baseline (Figure 2a style). Deferred: we don't
  have a baseline model wired in.
- [ ] Pooled CRPS (Figure 3 a-b). Deferred: needs a pool-size sweep.
- [ ] REV for extreme thresholds (Figure 2 g-h). Deferred: needs a
  pre-computed climatology distribution.
- [ ] Tropical cyclone track prediction (Figure 4). Deferred: external
  Tempest Extremes tool + IBTrACS dataset.
- [ ] Logging layout consistent with other `examples/weather/*` recipes
  (ExperimentLogger or equivalent, `outdir/experiment_name/run_id` tree).
- [ ] Pydantic-validated config types for dataset/model/training/inference,
  following `examples/weather/stormcast/utils/config.py` style.

### Inference and ensembling
- [x] Single-model stochastic rollout: autoregressive, fresh `z_t` each
  step (paper §2.2.2), N trajectories per checkpoint.
- [x] Deep ensemble inference across multiple checkpoints (paper §2.2.1:
  J=4, "same model for all timesteps of a trajectory"; remainder of
  unequal split goes to earlier models). `inference.checkpoints` config
  accepts a list; fallback to `inference.checkpoint` ("latest" or path) for
  the single-model case. Saved payload records per-model member counts and
  checkpoint paths for provenance. `test_fgn_deep_ensemble_inference`
  trains two seeds and verifies the `[3, 2]` allocation on 5 trajectories.
- [x] Bad-seed detection / dropping via spectrum diagnostics (paper §6.2):
  `scripts/check_spectra.py` implemented.
- [ ] Zarr / xarray / earth2studio-compatible output format.

### Backbone upgrade (deferred)
- [ ] Implement `ConditionalLayerNorm` (feature-dim, shared across spatial
  sites) and the 32→32 linear latent encoder from Appendix A.3.
- [ ] Build a CLN graph-transformer processor mirroring
  `GraphCastProcessorGraphTransformer` but with every `LayerNorm` replaced
  by CLN driven by the shared `z`.
- [ ] Compose an `FGNGraphNet(Module)` that re-uses `GraphCastNet`'s
  encoder / decoder / embedders and swaps in the CLN processor; forward
  signature `(history, latent, background, invariants) -> next_state`.
- [ ] Extend CLN to the encoder / decoder LayerNorms per §2.3 (all CLN
  layers share the same `z`); measure impact against the processor-only
  version.
- [ ] Edge-embedding capacity reduction from App. A.3
  (`input_dim_edges`-MLP hidden/output `= 32`).
- [ ] Global features: broadcast year-progress sin/cos across *both* grid
  and mesh (App. A.3).
- [ ] Grid→mesh message function without receiver conditioning.
- [ ] Multi-resolution curriculum: 1° → 0.25° with mesh refinement 5 → 6
  and the grid→mesh message-sum-÷4 correction from Stage 3 of Table A.2.

### Conventions + housekeeping
- [ ] Audit every new file against `examples/weather/stormcast` and
  `examples/weather/graphcast` for naming, module layout, type hints,
  docstring style, and SPDX headers.
- [ ] Requirements pinning: declare `earth2studio`, Pydantic, Hydra, and
  any optional extras (zarr, xarray) in `requirements.txt`.
- [ ] `README.md` update after each milestone with current scope, how to
  run training/inference, and dataset download instructions.

What exists now is a runnable MVP scaffold, not yet a WeatherNext 2-equivalent
data pipeline or model recipe.

## Paper Sections To Implement (verbatim from arXiv:2506.10772v1)

These two paragraphs are the core modelling claim for FGN and should be used
as the spec when we upgrade the MVP backbone / training loop:

**§2.3 Model Architecture.** Each of the constituent models in the FGN
model-ensemble is implemented with a very similar neural network
architecture to that of the denoiser in GenCast (Price et al., 2025), with a
GNN encoder/decoder mapping from the lat/lon grid to a latent space defined
on a spherical 6-times-refined icosahedral mesh, and a graph-transformer
processor which operates on the nodes of this mesh.

There are important differences, however, between the FGN and GenCast
architectures. FGN is a larger model, with ~180M parameters per model seed,
compared to GenCast's ~57M in total. FGN has latent dimension 768 and 24
layers in its graph-transformer processor, compared with latent dimension of
512 and 16 layers in GenCast, and produces forecasts with a 6-hour timestep,
whereas GenCast has a 12-hour timestep.

One key modeling choice was to use conditional normalization — which in
GenCast are used to condition on the corresponding diffusion noise level σ —
as the means of perturbing the model. To be precise, in FGN a global noise
vector `z ~ N(0, I_32)` is passed into all of the network's conditional
layer-norm layers, such that sampling different vectors `z` for each
ensemble member is what generates the variance across the ensemble.

This approach has two important features: (1) a low-dimensional noise
source, which (2) is globally applied across all layers, with learned
conditional normalization parameters shared across the spatial dimensions of
the model (i.e. mesh and grid node dimensions). Together these constrain the
output distribution in a way that encourages the model to generate globally
coherent variability despite being trained on a marginals-only loss.

Lang et al. (2024) also use conditional normalization to inject stochasticity
into their weather model, but with *location-specific* conditioning, which
adds a high-spatial-frequency component to the noise injection — FGN does
**not** do that.

**§2.4 Auto-regressive training.** FGN is initially trained with a
single-step loss only. The final stages of training, however, involve
autoregressive (AR) rollouts as part of each training step, where the model
produces a number of forecast steps autoregressively, and the loss is
averaged over all rollout steps, with gradients propagated back through the
rollout. FGN is finetuned on rollouts up to 8 steps. AR training is helpful
but not essential — skillful joint forecast distributions are also achieved
without it (Appendix A.4).

**Implementation consequences for this example:**

- Noise vector must be single `z ~ N(0, I_32)` per sample, broadcast into
  every CLN layer (not per-node, not per-layer-independent, not per-ε AdaGN).
- CLN scale/offset linear layers must have parameters that are *shared*
  across mesh nodes / grid nodes — i.e. the CLN operates on the feature
  dimension only, with the same learned map applied at every spatial site.
- AR finetune must keep gradients flowing through the rollout (truncated
  BPTT up to 8 steps), with a small tail of AR training compared to the
  single-step pre-training (Table A.2: 8k×1AR → 4k×2AR → 1k×{3..8}AR).
- Deep ensemble (J=4) is applied externally: train multiple seeds, each
  with its own `{θ*_j, Δ_j}`, and route a fixed seed per trajectory at
  inference.

### Appendix A.3 specifics (also load-bearing)

- **Conditioning on input states.** Unlike GenCast, FGN is not a diffusion
  denoiser: it takes only `X_{t-2:t-1}` concatenated along the channel
  dimension — no noisy `Z_σ` input. One forward pass per forecast step.
- **Shared encoder for CLN.** GenCast encodes σ into 16-d via sine/cosine
  Fourier features + a 2-layer MLP. FGN replaces that with the 32-d noise
  vector `z` passed through a *single linear (matrix multiply)* to produce
  another 32-d conditioning vector, which is then fed to all CLN layers.
  The paper notes this linear-only "encoder" was initially an implementation
  detail but empirically helps optimization.
- **Capacity (paper full-size).**
  - Hidden / output size of MLPs: **768** (GenCast: 512).
  - Attention heads: **6** (GenCast: 4).
  - Graph-transformer layers: **24** (GenCast: 16).
  - Exception: the edge-feature MLPs for *grid→mesh* and *mesh→grid* are
    shrunk to hidden/output size **32** (GenCast: 512) — `O(10^5)` edges
    but only 4 edge features, so this saves a lot of compute/memory at no
    cost. This matches `input_dim_edges=4` in `GraphCastNet`.
- **Global features.** Year-progress sin/cos is broadcast across the grid
  (as in GraphCast/GenCast), *and additionally* broadcast across the mesh
  and concatenated with the mesh input features — new in FGN.
- **Grid→mesh encoder GNN.** GenCast/GraphCast's edge message function
  conditions on both sender grid-node features and receiver mesh-node
  features (via gather). FGN drops the *receiver* conditioning from the
  message function — same performance, noticeable compute and memory win.
- **Mesh.** 6-times-refined icosahedral mesh (matches GenCast, not
  GraphCast's multimesh).

### Table A.1 — exact training dataset schema (paper Appendix A.1)

Source: **ERA5** for Stages 1–3 (pre-training), **HRES-fc0** for Stage 4
(fine-tuning). Years `1979-01-01` — `2018-01-15`, downsampled from 1 h to 6 h
by subsampling (except total precipitation, which is accumulated over the
6 h leading up to each downsampled time).

**Pressure levels (13, WeatherBench 13):**
`50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000` hPa.

**Atmospheric variables (6, input + predicted, one tensor per pressure level):**

| Type         | Name                 | Short | ECMWF ID | Role              | Internal name |
|--------------|----------------------|-------|----------|-------------------|---------------|
| Atmospheric  | Geopotential         | `z`   | 129      | Input/Predicted   | `z{level}`    |
| Atmospheric  | Specific humidity    | `q`   | 133      | Input/Predicted   | `q{level}`    |
| Atmospheric  | Temperature          | `t`   | 130      | Input/Predicted   | `t{level}`    |
| Atmospheric  | U component of wind  | `u`   | 131      | Input/Predicted   | `u{level}`    |
| Atmospheric  | V component of wind  | `v`   | 132      | Input/Predicted   | `v{level}`    |
| Atmospheric  | Vertical velocity    | `w`   | 135      | Input/Predicted   | `w{level}`    |

→ 6 × 13 = **78 atmospheric channels**.

**Single-level (surface) variables (6, mostly input + predicted):**

| Type    | Name                    | Short | ECMWF ID | Role                   | Internal name |
|---------|-------------------------|-------|----------|------------------------|---------------|
| Single  | 2 metre temperature     | `2t`  | 167      | Input/Predicted        | `t2m`         |
| Single  | 10 metre u-wind         | `10u` | 165      | Input/Predicted        | `u10m`        |
| Single  | 10 metre v-wind         | `10v` | 166      | Input/Predicted        | `v10m`        |
| Single  | Mean sea level pressure | `msl` | 151      | Input/Predicted        | `msl`         |
| Single  | Sea surface temperature | `sst` | 34       | Input/Predicted        | `sst`         |
| Single  | Total precipitation     | `tp`  | 228      | Predicted only (6 h)   | `tp06`        |

→ 5 input+predicted surface + 1 predicted-only (`tp`) = **6 surface channels**.
Total predicted channels per frame = 78 atmospheric + 6 surface = **84**.

**Static / clock inputs (6, input-only, never predicted):**

| Type    | Name                    | Short | ECMWF ID | Role        | Internal name      |
|---------|-------------------------|-------|----------|-------------|--------------------|
| Static  | Geopotential at surface | `z`   | 129      | Input-only  | `z`                |
| Static  | Land-sea mask           | `lsm` | 172      | Input-only  | `lsm`              |
| Static  | Latitude                | —     | n/a      | Input-only  | `lat`              |
| Static  | Longitude               | —     | n/a      | Input-only  | `lon`              |
| Clock   | Local time of day       | —     | n/a      | Input-only  | `local_time` (sin/cos) |
| Clock   | Elapsed year progress   | —     | n/a      | Input-only  | `year_progress` (sin/cos) |

Per FGN App. A.3 the year-progress sin/cos is broadcast across *both* the
grid and the mesh.

**NaN handling (preprocessing):**
- ERA5 SST has NaNs over land — replace with the global min sea surface
  temperature observed in a subset of ERA5.
- HRES-fc0 uses a placeholder value over land for SST; impute the same
  ERA5 land-NaN positions with the same global-min SST to keep the two
  datasets consistent.

**Shapes at training time:**
- Grid: `721 × 1440` at 0.25° (or `181 × 360` at 1° for Stages 1–2).
- History frames: 2 (prior two timesteps, `X_{t-2}` and `X_{t-1}`).
- Per-node input feature count: `2 × 84` state + 6 static/clock + broadcast
  year-progress features.
- Target: next state `X_t` with the same 84 channels (`tp` is output-only
  so it appears only in the target, never in the history input).

### Table A.2 training schedule (batch size 64, AdamW, weight decay 0.1, cosine LR, 1000 warm-up steps, fCRPS with N=2)

| Stage | Dataset   | Δt    | Resolution | #AR | Peak LR     | Total steps |
|-------|-----------|-------|------------|-----|-------------|-------------|
| 1     | ERA5      | 12 h  | 1°         | 1   | 8e-4        | 400 000     |
| 2     | ERA5      | 6 h   | 1°         | 1   | 8e-5        | 100 000     |
| 3     | ERA5      | 6 h   | 0.25°      | 1   | 8e-5        | 32 000      |
| 4     | HRES-fc0  | 6 h   | 0.25°      | 1–8 | 8e-5 → 8e-7 | 18 000 (8k·1AR + 4k·2AR + 1k·{3..8}AR) |

Stage 3 cross-resolution detail: when switching 1° → 0.25°, mesh refines
from 5× to 6× and each mesh node receives ~16× more grid→mesh messages —
divide the sum of grid→mesh messages by 4 at the start of Stage 3 to keep
the incoming signal scale roughly preserved. (Matches Appendix D of the
GenCast paper.)

## Goal

Add a PhysicsNeMo-native weather example that implements the core FGN ideas:

- epistemic uncertainty via deep ensembles of independently trained models
- aleatoric uncertainty via learned functional perturbations driven by low-dimensional latent noise
- direct marginal training with fair CRPS
- autoregressive rollout for trajectory generation

## External Model Specification Target

If this recipe is later extended toward the public WeatherNext 2 style of
deployment, the target external schema to align with is:

### Model Specifications

| Attribute | Details |
|---|---|
| Architecture | Functional Generative Network ([paper](https://arxiv.org/abs/2506.10772)) |
| Spatial Resolution | 0.25° |
| Temporal Resolution | Down to 1 hour (customizable by user) |
| Forecast Initialization Frequency Times (UTC) | Every 6 hours (00, 06, 12, 18 UTC) |
| Lead Times (Forecast Horizon) | 15 days (default, customizable) |
| Locations | Global |
| Training data | ERA5 / HRES-fc0 (HRES-fc1to5 for 1-hour model) |
| Initialization Inputs for Generating Forecasts | HRES-fc0 |
| Output Format | 64 members, 4 models with 16 members each by default |
| Historical Data | Historical forecasts from January 2024 to present for backtesting |

### Data Schema at 1-Hour Forecast Granularity

| Variable name | Units | Description |
|---|---|---|
| `300_geopotential` | m^2^/s^2^ | Geopotential at 300 hPa |
| `500_geopotential` | m^2^/s^2^ | Geopotential at 500 hPa |
| `300_temperature` | K | Temperature at 300 hPa |
| `500_temperature` | K | Temperature at 500 hPa |
| `1000_u_component_of_wind` | m/s | U wind component at 1000 hPa |
| `1000_v_component_of_wind` | m/s | V wind component at 1000 hPa |
| `100m_u_component_of_wind` | m/s | 100 meter U wind component |
| `100m_v_component_of_wind` | m/s | 100 meter V wind component |
| `10m_u_component_of_wind` | m/s | 10 meter U wind component |
| `10m_v_component_of_wind` | m/s | 10 meter V wind component |
| `2m_temperature` | K | 2 meter temperature |
| `mean_sea_level_pressure` | Pa | Mean sea level pressure |
| `sea_surface_temperature` | K | Sea surface temperature |

### Data Schema at 6-Hour Forecast Granularity

| Variable name | Units | Description | Level |
|---|---|---|---|
| `total_precipitation_6hr` | m | Total accumulated precipitation | Surface (6-hour accumulation) |
| `mean_sea_level_pressure` | Pa | Mean sea level pressure | Sea Level |
| `2m_temperature` | K | Air temperature | 2 meters |
| `10m_u_component_of_wind` | m/s | Eastward wind component | 10 meters |
| `10m_v_component_of_wind` | m/s | Northward wind component | 10 meters |
| `100m_u_component_of_wind` | m/s | Eastward wind component | 100 meters |
| `100m_v_component_of_wind` | m/s | Northward wind component | 100 meters |
| `{level}_geopotential` | m^2^/s^2^ | Geopotential height | 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa |
| `{level}_specific_humidity` | kg/kg | Specific humidity | 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa |
| `{level}_temperature` | K | Temperature | 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa |
| `{level}_u_component_of_wind` | m/s | Eastward wind component | 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa |
| `{level}_v_component_of_wind` | m/s | Northward wind component | 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa |
| `{level}_vertical_velocity` | Pa/s | Vertical velocity | 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa |

## Canonical Internal Variable Convention

For the actual `examples/weather/fgn/` implementation, the canonical internal
schema should follow PhysicsNeMo and Earth2Studio naming, not the verbose
external WeatherNext table names.

This matters because:

- Earth2Studio data sources and lexicons already use compact forecast variable names
- existing PhysicsNeMo weather examples are already written against these names
- it keeps dataset, checkpoint, and inference interfaces consistent across recipes

Recommended rule:

- verbose WeatherNext names are accepted at the dataset boundary
- they are immediately mapped into compact Earth2Studio-style names
- all downstream training, normalization, checkpoint metadata, and inference use
  only the compact names

Examples of the mapping:

| External WeatherNext 2 name | Canonical internal name |
|---|---|
| `300_geopotential` | `z300` |
| `500_geopotential` | `z500` |
| `300_temperature` | `t300` |
| `500_temperature` | `t500` |
| `1000_u_component_of_wind` | `u1000` |
| `1000_v_component_of_wind` | `v1000` |
| `100m_u_component_of_wind` | `u100m` |
| `100m_v_component_of_wind` | `v100m` |
| `10m_u_component_of_wind` | `u10m` |
| `10m_v_component_of_wind` | `v10m` |
| `2m_temperature` | `t2m` |
| `mean_sea_level_pressure` | `msl` |
| `sea_surface_temperature` | `sst` |
| `total_precipitation_6hr` | `tp06` |
| `{level}_specific_humidity` | `q{level}` |
| `{level}_vertical_velocity` | `w{level}` |

Implementation consequence:

- the first real-data dataset class should include a schema mapper that converts
  source names into the internal compact names before channel ordering is fixed
- config files should list variables as `u10m`, `v10m`, `t2m`, `msl`, `z500`,
  `q850`, `w300`, etc.
- any Earth2Studio-based fetch path should operate directly in those compact names

This schema is broader than the MVP synthetic recipe. For the first real-data
version, the cleanest progression is:

1. start with a uniform 6-hour subset
2. map variables into a PhysicsNeMo dataset with explicit channel ordering
3. add mixed-cadence support only after the 6-hour path is training cleanly

The implementation should look and feel like other `examples/weather/*` recipes:

- Hydra/Pydantic config structure
- explicit dataset/model/training/inference modules
- distributed training support
- clean train/inference entrypoints
- testable with small synthetic configs

## Paper Features To Capture

From the paper, the minimum viable FGN recipe should support:

1. Two-frame autoregressive conditioning.
   The paper uses a 2nd-order Markov setup with prediction of the next state from the previous two states.

2. Functional stochasticity in parameter space.
   A low-dimensional latent vector perturbs the function represented by the network, rather than perturbing inputs or outputs directly.

3. Deep ensemble epistemic uncertainty.
   Multiple independently trained models each generate a subset of ensemble trajectories.

4. Marginal probabilistic training with fair CRPS.
   Training is applied to univariate marginals while retaining useful joint structure in generated samples.

5. Ensemble trajectory generation.
   At inference, a trajectory keeps one model identity fixed while resampling aleatoric noise at each step.

## PhysicsNeMo Design Choice

The cleanest path is to build FGN on top of the existing weather recipe structure used by StormCast rather than starting from scratch.

Why:

- `examples/weather/stormcast/` already provides a modern regional/generative weather training stack.
- It already has dataset interfaces, configurable conditions, distributed helpers, inference configs, and trainer organization.
- FGN needs probabilistic training and autoregressive rollout, but not diffusion specifically.

Recommended approach:

- create `examples/weather/fgn/` as a sibling recipe, initially reusing selected abstractions from StormCast
- avoid coupling FGN to diffusion-specific utilities
- use PhysicsNeMo core/module/checkpointing/distributed APIs directly

## Proposed Example Layout

Recommended initial structure:

```text
examples/weather/fgn/
  README.md
  requirements.txt
  train.py
  inference.py
  test_training.py
  config/
    dataset/
    model/
    training/
    inference/
    regression.yaml
    test_regression.yaml
  datasets/
    __init__.py
    dataset.py
    mock.py
    data_loader_*.py
  utils/
    config.py
    loss.py
    nn.py
    trainer.py
    plots.py
    parallel.py
```

Optional later:

- `ensemble_inference.py` for multi-checkpoint inference across independently trained models
- `scripts/` for launching ensemble members

## Model Plan

### Phase 1: Deterministic Backbone + Latent Conditioning

Implement a backbone that predicts the next state from:

- previous state frames: `state[t-1]`, `state[t]`
- optional large-scale conditioning/background
- optional invariants
- latent vector `z`

Preferred first backbone:

- U-Net-like grid model, because it matches existing PhysicsNeMo weather practice and is straightforward to test

How to inject `z`:

- use FiLM/AdaGN/AdaLN style modulation in residual blocks
- produce modulation parameters from a small MLP applied to `z`
- share `z` across the spatial field for a single sample and time step

This matches the paper’s core idea that one latent sample perturbs the function globally, inducing structured spatial variability.

### Phase 2: Ensemble Wrapper

Do not hide the epistemic ensemble inside a single checkpoint.
Instead:

- train one model per ensemble member seed
- save each checkpoint independently
- inference script loads a list of checkpoints and allocates ensemble trajectories across them

This maps directly to the paper’s deep ensemble design and keeps the implementation simple.

## Loss Plan

### Primary Training Loss

Implement fair CRPS for empirical samples.

For a target scalar and ensemble samples, the fair CRPS estimator should be implemented over:

- batch
- channel / variable
- spatial locations

Need configurable reduction and weighting by variable.

### Training-time Sampling

To estimate CRPS during training:

- draw `K` latent samples per training example
- produce `K` forecasts from the same model
- compute fair CRPS against the target

Initial default:

- small `K` such as 2 or 4 for tractable training

### Optional Auxiliary Terms

Potential later additions:

- deterministic mean loss warm-start
- variance regularization or spread diagnostics
- temporal consistency penalty for autoregressive stability

But these should not be in the MVP unless needed for optimization stability.

## Dataset Plan

FGN is global in the paper, but the first PhysicsNeMo example does not need to reproduce the exact global setup.

Recommended staged scope:

### MVP Scope

Regional grid forecasting with the same dataset interface style as StormCast:

- state tensor
- optional background tensor
- optional invariants
- target next state

Support two prior frames:

- dataset returns `state = [x_{t-1}, x_t]`
- target is `x_{t+1}`

This is sufficient to validate:

- latent-conditioned function perturbations
- fair CRPS training
- autoregressive rollout

### Stretch Scope

Later add a global lat-lon dataset variant aligned more closely to the paper:

- ERA5 state variables
- HRES analysis / initial conditions if available
- 6-hour step size
- multi-level atmospheric fields

## Config Plan

Use Hydra configs mirroring existing weather recipes.

### `config/model/fgn.yaml`

Include:

- `model_name`
- `model_type`
- latent dimension
- latent injection type
- condition list
- channels / widths / blocks
- whether to use background and invariants

### `config/training/default.yaml`

Include:

- optimizer
- scheduler
- batch size
- gradient accumulation
- mixed precision
- `num_crps_samples`
- checkpointing and logging
- distributed/domain-parallel settings if needed later

### `config/inference/fgn.yaml`

Include:

- checkpoint path or checkpoint list
- number of trajectories
- number of steps
- latent samples per step
- output backend/path

## Inference Plan

Implement two inference modes.

### 1. Single-model stochastic inference

For one checkpoint:

- initialize from input frames
- sample latent `z_t` each step
- autoregressively predict future states
- output an ensemble of trajectories

### 2. Deep ensemble inference

For multiple checkpoints:

- load `N` independently trained models
- assign trajectories across models
- keep one model fixed for each trajectory
- resample latent `z_t` at each step

This is the direct paper-inspired inference path.

## PhysicsNeMo APIs To Use

The implementation should prefer existing PhysicsNeMo APIs and conventions:

- `physicsnemo.core.Module` for checkpoint save/load
- `physicsnemo.distributed.DistributedManager`
- existing optimizer/scheduler patterns from weather examples
- existing trainer patterns where reusable
- standard weather dataset interfaces already used in `examples/weather/stormcast`

Avoid introducing a custom framework around the example when a PhysicsNeMo utility already exists.

## Testing Plan

Add a small synthetic test suite similar to StormCast.

Minimum tests:

1. dataset mock returns the expected two-frame input format
2. model forward shape is correct with and without background/invariants
3. latent conditioning changes outputs when `z` changes
4. fair CRPS implementation matches simple reference cases
5. single-step training smoke test runs
6. autoregressive inference smoke test runs and writes outputs

## Milestones

### Milestone 1: Skeleton Recipe

- create `examples/weather/fgn/` structure
- add README/config scaffolding
- add mock dataset and smoke tests

### Milestone 2: FGN Backbone

- implement latent-conditioned U-Net
- implement two-frame dataset interface
- deterministic single-step training smoke test

### Milestone 3: CRPS Training

- implement fair CRPS loss
- support `K` stochastic samples per example
- add validation metrics and plots

### Milestone 4: Autoregressive Inference

- implement rollout
- implement trajectory ensemble generation
- write Zarr or xarray outputs similar to other weather examples

### Milestone 5: Deep Ensemble Support

- support multi-checkpoint inference
- document how to train multiple seeds and combine them

### Milestone 6: Real Weather Dataset

- port to a real dataset under `examples/weather/dataset_download/` or a dedicated curation path
- add a reference training/inference config

## Risks / Open Questions

1. Exact FGN perturbation parameterization.
   The paper motivates parameter-space uncertainty, but for an MVP we need a concrete and stable implementation choice. AdaGN/AdaLN-style modulation is the most practical first version.

2. Computational cost of CRPS sample training.
   Multi-sample forward passes increase cost linearly in `K`. The recipe should expose `num_crps_samples` explicitly and support small values for tests.

3. Global versus regional scope.
   The paper is global and 6-hourly. The initial recipe should prioritize a PhysicsNeMo-consistent implementation over exact reproduction.

4. Joint structure evaluation.
   The paper evaluates more than marginals. MVP should likely stop at marginal scores and simple spread/rank diagnostics before adding correlation-structure metrics.

## Recommended First Implementation Target

The best first target is:

- a regional grid-based FGN example
- two-frame autoregressive next-step prediction
- latent-conditioned U-Net
- fair CRPS training
- stochastic rollout inference
- deep ensemble through multiple checkpoints

This gets the core idea into the codebase quickly, uses the existing PhysicsNeMo weather recipe patterns, and leaves room for a later global-paper-faithful version.

## Proposed Next Steps

1. Create `examples/weather/fgn/` scaffold from the StormCast recipe structure.
2. Implement a minimal mock dataset with two-frame inputs.
3. Implement latent-conditioned U-Net in `utils/nn.py`.
4. Implement fair CRPS in `utils/loss.py`.
5. Add smoke-test configs and tests.
6. Add a first real-data adapter after the core recipe is stable.
