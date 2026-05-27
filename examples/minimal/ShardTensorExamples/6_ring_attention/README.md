# Ring Attention Benchmark

Benchmarks `scaled_dot_product_attention` on a single GPU vs. distributed ring
attention via ShardTensor.

## Quick start

**Single GPU:**

```bash
python benchmark_sharded_attention.py \
    --seq_len 4096 --num_heads 16 --head_dim 64
```

**Distributed (ring attention):**

```bash
torchrun --nproc-per-node 4 benchmark_sharded_attention.py \
    --seq_len 4096 --num_heads 16 --head_dim 64
```

## Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--seq_len` | 4096 | Sequence length (world-size-divisible; chunk multiple of 32) |
| `--num_heads` | 16 | Number of attention heads |
| `--head_dim` | 64 | Dimension per head |
| `--batch_size` | 1 | Batch size |
| `--mode` | `inference` | `inference` (forward only) or `train` (forward + backward) |
| `--dtype` | `float32` | `float32`, `float16`, or `bfloat16` |
| `--num_warmup` | 5 | Warmup iterations |
| `--num_iterations` | 10 | Timed iterations |
| `--results_dir` | `<script_dir>/results/` | Directory for the auto-named JSON output |
| `--print-only` | off | Skip writing JSON; print to stdout only |

By default the benchmark writes a JSON file to `results/` whose name
encodes the run configuration, e.g.
`single_gpu_inference_float32_seq4096.json` or
`distributed_4gpu_train_bfloat16_seq8192.json`. Pass `--print-only` to
disable this.

## Plotting results

After collecting JSON results in `results/`, generate scaling plots:

> Note!  To run the scaling plot, you will need matplotlib installed.  `pip install matplotlib`.

```bash
python plot_scaling_results.py
```

This reads all `results/*.json` files and writes per-mode latency plots
(e.g. `ring_attention_shard_tensor_inference.png`).

The module also exposes helpers for custom analysis:

```python
import plot_scaling_results as psr

df = psr.load_results()                          # DataFrame, one row per JSON file
train = psr.filter(df, mode="train", gpus=4)     # filter by mode / GPUs / seq_len
df = psr.add_efficiency(df)                      # adds speedup & parallel_efficiency columns
print(psr.summary_table(df))                     # pivot table of mean latency (ms)
```
