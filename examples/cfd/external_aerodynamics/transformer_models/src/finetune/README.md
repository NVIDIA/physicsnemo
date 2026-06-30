# LoRA fine-tuning (GeoTransolver)

Parameter-efficient fine-tuning of a **pretrained GeoTransolver** (trained with
`src/train.py`, or the NIM / multi-dataset checkpoint) on a small custom
external-aerodynamics dataset, using `physicsnemo.experimental.peft`.

This recipe lives in its own `src/finetune/` folder, separate from the main
training/inference scripts in `src/`. It is a companion to `src/train.py`: same
model, same data pipeline, same `src/conf/` groups — only the entry points
(`src/finetune/finetune.py`, `src/finetune/deploy.py`) and config
(`src/conf/finetune_lora.yaml`) are new. `train.py` is unchanged.

## Why LoRA

- Small adapters (~hundreds of KB) vs full checkpoints (~tens of MB).
- Lower memory (frozen base layers drop saved activations).
- Less overfitting / forgetting in the small-data regime (α=0 = the base exactly).
- One base + N swappable adapters at serve time.

## Workflow

```text
            src/finetune/finetune.py              src/finetune/deploy.py
 base.mdlus ───────────────────▶ adapter.lora ────────────────────▶ serve (swap)
(pretrained)  apply_lora + train  (~hundreds KB)   load_adapter        or merge_lora
              only the adapters                                      → merged .mdlus
```

1. **Fine-tune** (run from the example root, same as `train.py`):

   ```bash
   python src/finetune/finetune.py init_from=/path/to/base_geotransolver.mdlus
   # multi-GPU (single node):
   torchrun --nproc_per_node=8 src/finetune/finetune.py init_from=/path/to/base.mdlus
   ```

2. **Deploy** — adapter-swap, or merge for zero overhead:

   ```bash
   python src/finetune/deploy.py init_from=/path/to/base.mdlus            # adapter-swap
   python src/finetune/deploy.py init_from=/path/to/base.mdlus merge=true # fold in → *_merged.mdlus
   ```

## Config (`src/conf/finetune_lora.yaml`)

- `init_from` (**required**): the pretrained base `.mdlus`. The `model:` block
  **must match its architecture** — `load_adapter`/`deploy.py` enforce a base
  fingerprint and refuse a mismatched base.
- `peft.target_pattern`: which layers get adapters (default = GALE attention
  projections). `peft.wrap_mlp: true` also adapts the feed-forward MLP.
- `peft.rank` / `peft.alpha`: adapter capacity / scaling. `peft.init` optionally
  customizes the `lora_A` initialization (a name or a callable).
- Point the `data` group at your small dataset (see `src/conf/data/{core,surface}.yaml`).

## How it differs from `train.py`

- Only LoRA (+`extras_trainable`) params train; the base is frozen.
- Those params go to **AdamW**, never Muon (Newton-Schulz is degenerate on
  rank-`r` factors) — via `split_params_for_optimizer`.
- DDP uses `find_unused_parameters=True` (frozen base params get no grad).
- Multi-GPU shards the dataset per rank via `DistributedSampler` + `set_indices`
  (same as `train.py`); launch with `torchrun --nproc_per_node=<N>`.
- **float32 only**: the minimal recipe does not wire the mixed/fp8 path (autocast,
  fp8 padding, GradScaler); it errors if `precision != float32`. Use `train.py`
  for fp8.
- `finetune.py` keeps a minimal MSE loop for readability; reuse
  `train.forward_pass` if you want the full metrics/normalization path.
- Deploy `merge=true` only writes a merged `.mdlus` if all adapters are
  mergeable; a fused `te.LayerNormMLP` residual (from `wrap_mlp` under TE) is
  left in place and you deploy via `load_adapter` instead.

The PEFT API used here is covered by `test/experimental/peft/`. A full
data-driven run needs a base checkpoint, a dataset, and the PhysicsNeMo container.
