# Configuration layout

The existing bumper and body-in-white one-shot/time-conditional experiments are
unchanged. The DeFormer comparison has three autoregressive entry points:

| Config | Purpose |
|---|---|
| `crash_geoflare_autoregressive` | GeoTransolver + original point-space FLARE baseline |
| `crash_deformer_autoregressive` | Structural mesh processing + FLARE, no contact |
| `crash_deformer_contact_autoregressive` | DeFormer + predictive node-to-face contact and reference-geodesic exclusions |

DeFormer inherits the GeoFLARE data/training settings; the contact recipe inherits
DeFormer. The two model definitions live in `model/`. No teacher-forcing probes,
FLARE++ experiments, or historical memory/contact ablation presets are included.

```bash
python train.py --config-name=crash_deformer_contact_autoregressive \
  training.raw_data_dir=/data/crash/train \
  training.raw_data_dir_validation=/data/crash/validation

python inference.py --config-name=crash_deformer_contact_autoregressive \
  inference.raw_data_dir_test=/data/crash/test
```

Inference also requires the matching checkpoint and saved training statistics;
see the [recipe README](../README.md#inference). The reference defaults are 127
training cases, 8 validation cases, 26 frames at 5 ms spacing, and 500 epochs.
Override counts, paths, and `model.dt` for another dataset. Never point different
experiments at the same output/checkpoint directory when running concurrently.

## Ablations and execution options

Use overrides on the contact entry point rather than adding another preset:

| Override | Effect |
|---|---|
| `datapipe.contact_surface_exclusion=incidence` | Omit the optional geodesic filter |
| `datapipe.contact_surface_exclusion=one_ring` | Use material one-ring exclusions |
| `datapipe.contact_geodesic_gap_min=0.0` | Remove the reference recipe's explicit 5 mm gap floor |
| `model.checkpoint_offloading=false` | Disable host-memory activation offloading |
| `model.enable_contact=false` | Disable messages while preserving contact parameters |

These overrides do not change the learning-rate schedule, training budget, or
BPTT window. The contact recipe retains its explicit deterministic sampling and
contact-specific initialization settings; the no-contact entry point preserves
its original recipe. A true no-contact architecture uses
`crash_deformer_autoregressive`, not just the message-disable override.

Print an entry point without starting training:

```bash
python train.py --config-name=crash_deformer_contact_autoregressive --cfg job
```

## Shared components

| Directory | Purpose |
|---|---|
| `model/` | Architecture and rollout defaults |
| `datapipe/` | Graph and point-cloud dataset defaults |
| `reader/` | VTP and Zarr readers |
| `training/` | Generic optimization and checkpoint settings |
| `inference/` | Generic evaluation settings |
