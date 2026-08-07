# Domain-Parallel Training Loop

This example shows how to adapt a single-device or DDP training loop to use
domain parallelism, for both training and inference, and how to combine it
with `torch.compile`.

The data is synthetically generated image-like data in 2D or 3D.  The training
script can benchmark the model over a variety of image sizes, with a choice
of architecture (`--model`):

- **`vit`** (default) — a hybrid Vision Transformer: convolutional patch
  embedding + Transformer blocks.  Sequence-sharded attention dispatches to
  **ring attention**.
- **`conv`** — a residual ConvNet.  Spatially-sharded convolutions use
  **halo exchanges** (kernel 3 / stride 1 / padding 1, the supported
  envelope; the patch-reduction stem uses the kernel == stride exception).
- **`natten`** — a ViT variant using **neighborhood attention** (`na2d` /
  `na3d`) on channels-last spatial tokens, with halo exchanges.  Requires
  the optional `natten` package
  (`pip install nvidia-physicsnemo[cu12,natten-cu12]`); odd kernels only,
  and each rank's local patch-grid extent must be >= the kernel size.

All three support 2D and 3D via `--dimension`.  In every case the model is a
completely vanilla `nn.Module` — see below for bringing your own.

The default ViT is a convolutional embedding followed by 16 layers of
Transformer blocks.

```python
HybridViT(
  (patch_embed): PatchEmbedding2d(
    (conv): Conv2d(3, 768, kernel_size=(8, 8), stride=(8, 8))
    (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (stages): ModuleList(
    (0-15): 16 x TransformerBlock(
      (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (attn): MultiHeadAttention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (proj): Linear(in_features=768, out_features=768, bias=True)
      )
      (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): MLP(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
      )
    )
  )
  (head): Linear(in_features=768, out_features=1000, bias=True)
)
Number of parameters: 126907624
```

## Parallelism model

`ShardTensor` inherits directly from `torch.Tensor`.  The model is a completely
vanilla `nn.Module`: when a plain model weight meets a domain-sharded activation
in the forward pass, `ShardTensor` auto-promotes it to a replicated distributed
tensor, and in the backward pass the weight's gradient is reduced over the
domain group before it reaches the plain parameter.  This autograd reduction
applies to paths involving `ShardTensor`; it is separate from startup state
synchronization.  No `distribute_module`, no model surgery.

The configurations map to standard PyTorch tools:

- **Data parallel only** (`--ddp_size N`): all parameters are plain tensors, so
  the model is wrapped in ordinary `DistributedDataParallel`.
- **Domain parallel** (`--domain_size N`): inputs are scattered across the
  domain mesh with `scatter_tensor` (sharded along the image height).  Every
  parameter -- including the positional embedding, which is added to
  sequence-sharded activations -- stays a plain, replicated tensor;
  ShardTensor's auto-promotion handles the mixing in the forward pass, and the
  replicated plain parameters and buffers are broadcast once from the domain
  source rank at startup
  with `physicsnemo.domain_parallel.sync_module_over_mesh` (neither DDP nor
  FSDP2 syncs the domain axis).
- **Both** (`--ddp_size N --domain_size M`): because all parameters are still
  plain tensors, ordinary DDP over the `ddp` mesh axis works here too -- no
  special combined-parallelism wrapper is required.
- **Parameter sharding** (`--fsdp`): opt in to sharding the weights over the
  `ddp` mesh axis with FSDP2 (`fully_shard`), reducing per-GPU parameter and
  optimizer-state memory.  On this path the parameters become distributed
  tensors anyway, so the positional embedding is additionally *split* -- not
  replicated -- across the domain mesh, matching the sequence-sharded
  activations it is added to.  It is sharded with plain `DTensor`
  (`distribute_tensor(..., [Shard(1)])`): parameters are statically shaped, so
  DTensor's even chunking is exactly right, while `ShardTensor` handles the
  (potentially unevenly sharded) activations.  Gradients are already reduced
  over the domain axis by ShardTensor, so FSDP2 only sees the data-parallel
  mesh.  With `--ddp_size 1` the shard is degenerate (size-1, no
  communication), but the parameters uniformly become distributed tensors on
  every `--fsdp` configuration.

## Synchronization responsibilities

At training startup, DDP typically performance a broadcast of parameters
from rank 0 to all ranks.  With domain parallelism, DDP (and FSDP2) will still
do this, but note that they will only do it on *their* axis of the mesh.

To cover the domain axis, this example calls `sync_module_over_mesh` before
wrapping the model. It broadcasts the plain parameters and buffers within each
domain group, but leaves `ShardTensor` and `DTensor` values alone. Those are
already synchronized when created with `scatter_tensor` or `distribute_tensor`;
`from_local` is the exception, since it assumes each rank already has the right
local shard.  If you have a plain model, `sync_module_over_mesh` ensures proper
start up coordination.

ShardTensor autograd performs the domain reduction for plain weights used in
ShardTensor operations.

The startup broadcast is not repeated during training. If a plain buffer changes
independently on each domain rank—for example, BatchNorm running statistics—you
must synchronize it where your training loop needs the ranks to agree:

```python
group = domain_mesh.get_group()
src = torch.distributed.get_global_rank(group, 0)
torch.distributed.broadcast(model.norm.running_mean, src=src, group=group)
torch.distributed.broadcast(model.norm.running_var, src=src, group=group)
```

The same rule applies when only one rank loads a plain checkpoint: broadcast
that state afterward. Distributed checkpoints should restore each rank's local
shards directly.

## Bring your own model

The models live in `model/` and are registered in `model/__init__.py` via a
small `ModelSpec`.  To benchmark your own architecture:

1. Write a plain `nn.Module` — **zero ShardTensor or distributed imports** —
   that takes `(B, in_channels, *spatial)` and returns `(B, num_classes)`
   logits.
2. Add one `MODEL_REGISTRY` entry with three hooks: the constructor, which
   submodules to compile when the domain is sharded (anything doing
   halo/ring communication stays eager), and which statically-shaped spatial
   parameters (positional embeddings) to shard on the FSDP2 path.

That registry entry is the *entire* integration surface.  The parallelism —
input scattering, weight sync, DDP/FSDP2 wrapping — lives in the training
script and is identical for every model.

## torch.compile

Pass `--compile` to enable `torch.compile` (inductor backend):

- With `--domain_size 1`, the entire model is compiled.
- With `--domain_size > 1`, sharded attention (ring SDPA for `vit`,
  neighborhood attention for `natten`) cannot live inside a compiled region.
  Each model's `ModelSpec` compiles the model *regionally* instead --
  patch embedding, per-block norms and MLPs, and the head are compiled;
  attention stays eager.  The conv model compiles its residual stages whole.

Compilation happens during the warmup iterations, so the reported timings
measure steady-state performance.

## Usage

```bash
usage: training_script.py [-h] [--batch_size BATCH_SIZE] [--dimension {2,3}]
[--image_size_start IMAGE_SIZE_START] [--image_size_stop IMAGE_SIZE_STOP]
[--image_size_step IMAGE_SIZE_STEP]
[--ddp_size DDP_SIZE] [--domain_size DOMAIN_SIZE] [--use_mixed_precision]

Benchmark HybridViT model performance

options:
  -h, --help            show this help message and exit
  --model {vit,conv,natten}
                        Model architecture (default: vit); see model/__init__.py
                        to register your own
  --batch_size BATCH_SIZE
                        Global Batch size for training (default: 1)
  --dimension {2,3}     Dimension of the model: 2D or 3D (default: 2)
  --image_size_start IMAGE_SIZE_START
                        Starting image size (default: 256)
  --image_size_stop IMAGE_SIZE_STOP
                        Ending image size (default: 2048)
  --image_size_step IMAGE_SIZE_STEP
                        Step size for image size progression (default: 128)
  --ddp_size DDP_SIZE   DDP world size (default: 1)
  --domain_size DOMAIN_SIZE
                        Domain parallel size (default: 1)
  --use_mixed_precision
                        Enable mixed precision training (default: False)
  --inference_only      Run inference benchmarks only, skip training (default: False)
  --fsdp                Shard model parameters over the ddp mesh axis with FSDP2
                        instead of replicating them (default: False)
  --compile             Enable torch.compile; regional compile when domain_size > 1
                        (default: False)
```

The model code is identical in all use cases: only the input data changes, and
whether the model is wrapped in DDP or FSDP2.  Output will include a
table of performance results.

## Testing

A smoke-test suite covers every parallelism / compile configuration by
launching the script through `torchrun` with small, fast settings.  It is
sized for a 4-GPU machine (configurations needing more GPUs than available
are skipped):

```bash
uv run pytest test_training_script.py -v
```
