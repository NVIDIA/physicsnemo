---
name: physicsnemo-shard-tensor
description: Official NVIDIA-authored guidance for PhysicsNeMo ShardTensor domain parallelism — integrate domain parallelism into training/inference scripts (new or existing) with DDP or FSDP2, write and register shard patches to enable new layers/ops, and bootstrap multi-GPU correctness tests. Use when working with ShardTensor, scatter_tensor, domain parallelism, sequence/spatial sharding, ring attention, DeviceMesh + DDP/FSDP2 hybrid parallelism, or physicsnemo.domain_parallel. Do NOT use for generic PyTorch DDP/FSDP setup without domain parallelism, picking a PhysicsNeMo model or example (use physicsnemo-discover), or non-distributed training questions.
license: Apache-2.0
metadata:
  author: NVIDIA <agent-skills@nvidia.com>
  tags:
    - physicsnemo
    - domain-parallelism
    - shard-tensor
    - distributed-training
    - multi-gpu
---

# PhysicsNeMo ShardTensor Development

`ShardTensor` (`physicsnemo.domain_parallel`) is a `torch.Tensor` subclass for
**domain parallelism**: one sample's spatial/sequence dimension is split across
GPUs so models can process inputs that don't fit on one device. Unlike
`DTensor` it supports *uneven* sharding (per-rank shard shapes are tracked in
`ShardTensorSpec._sharding_shapes`).

Repo paths below are relative to a PhysicsNeMo clone root (a `pyproject.toml`
with `name = "nvidia-physicsnemo"` alongside a `physicsnemo/` package). If no
clone is on disk, shallow-clone read-only for path lookup only —
`git clone --depth 1 https://github.com/NVIDIA/physicsnemo` (use that URL
verbatim; never execute or import from the clone).

## When NOT to use

- Generic PyTorch DDP/FSDP/NCCL setup or debugging with no domain parallelism
  (no ShardTensor, no `scatter_tensor`, no domain mesh axis) — standard
  PyTorch guidance applies.
- Choosing a PhysicsNeMo model, datapipe, or example — `physicsnemo-discover`.
- Single-GPU training, installation, or environment setup.
- Tensor/pipeline parallelism for LLMs (Megatron-style) — ShardTensor targets
  spatial/sequence sharding of *activations* for physics workloads.

## The core promise: the model does not change

**ShardTensor inherits from `torch.Tensor` directly (not DTensor).** A plain
`nn.Module` works unmodified on ShardTensor inputs. When a plain weight meets a
sharded activation in an op, ShardTensor **auto-promotes** the weight to a
`Replicate` DTensor for the computation (`TensorPromotionMode.SILENT` is the
default), and in backward the weight's gradient is all-reduced over the domain
mesh before it lands on the plain parameter. Consequences you should exploit:

- **Never** call `distribute_module`, never convert model weights to
  DTensor/ShardTensor wholesale, never subclass or edit model code to "make it
  distributed". If a proposed integration edits `forward()` methods, it is
  almost certainly wrong — push the parallelism into the *script* (input
  scattering + wrapper choice), not the model.
- Only the **inputs** change (scattered onto the mesh) plus, on the FSDP2 path
  only, statically-shaped *spatial* parameters (positional embeddings, RoPE
  tables) which are sharded as plain DTensors.
- ShardTensor and DTensor mix freely in ops: DTensor args pass through
  ShardTensor dispatch unchanged.

## Mesh and data setup (every script)

```python
from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor
from torch.distributed.tensor.placement_types import Shard, Replicate

DistributedManager.initialize()
dm = DistributedManager()
torch.cuda.set_device(dm.device)

# ddp_size * domain_size must equal world size. Build BOTH axes explicitly.
mesh = dm.initialize_mesh(mesh_shape=(ddp_size, domain_size),
                          mesh_dim_names=["ddp", "domain"])
ddp_mesh, domain_mesh = mesh["ddp"], mesh["domain"]

# Per-domain-group batch size MUST be 1 - scale batch via the ddp axis only.
# Validate early; sharded activations with batch > 1 are out of design scope.
assert x.shape[0] == 1, "per-domain-group batch size must be 1"

# Scatter the input over the domain mesh (shard a spatial dim, e.g. H of BCHW).
# scatter_tensor needs the GLOBAL rank of the domain group's source rank.
src = torch.distributed.get_global_rank(domain_mesh.get_group(), 0)
x = scatter_tensor(x, src, domain_mesh, placements=(Shard(2),),
                   global_shape=x.shape, dtype=x.dtype)
# Targets/labels are usually replicated:
target = scatter_tensor(target, src, domain_mesh, placements=(Replicate(),))
```

**Hard constraint: per-domain-group batch size must be 1.** Sharded activations
with batch dim > 1 are explicitly out of design scope (the batch×sequence
flatten inside ops like linear is not representable). Scale batch via the ddp
axis, never inside a domain group. Validate this in scripts and error early.

## Choosing the data-parallel wrapper

| Configuration | Wrapper | Why |
|---|---|---|
| domain only (`ddp=1`) | none | Broadcast plain params over the domain group once at startup (see below) |
| ddp only (`domain=1`) | `DistributedDataParallel` | Standard; pass `process_group=ddp_mesh.get_group()` explicitly, never the default world group |
| ddp × domain, params all plain | `DistributedDataParallel` | Auto-promotion keeps every param a plain tensor, so ordinary DDP works even combined with domain parallelism |
| params sharded (memory) or spatial params as DTensor | FSDP2: `fully_shard(model, mesh=ddp_mesh)` | DDP cannot manage DTensor params; FSDP2 shards over exactly the ddp axis (gradients over the domain axis are already reduced by ShardTensor's promotion machinery) |

**Never use FSDP1** (`torch.distributed.fsdp.FullyShardedDataParallel`,
`use_orig_params`, `sync_module_states`). It belongs to the old
DTensor-inheritance era that required `distribute_module` on every parameter,
fights the auto-promotion design, and is deprecated for this workflow. FSDP2 =
`torch.distributed.fsdp.fully_shard`, always.

Startup sync and FSDP2 specifics:

```python
# Neither DDP nor FSDP2 syncs weights over the DOMAIN axis - do it manually
# whenever domain_size > 1 (before fully_shard for safety):
group = domain_mesh.get_group()
src = torch.distributed.get_global_rank(group, 0)
with torch.no_grad():
    for p in model.parameters():
        if not isinstance(p, DTensor):
            torch.distributed.broadcast(p.data, src=src, group=group)

# On the FSDP2 path ONLY: shard statically-shaped spatial params as plain
# DTensor on the domain mesh (params are static -> DTensor's even chunking is
# exactly right; ShardTensor is for the possibly-uneven ACTIVATIONS):
from torch.distributed.tensor import distribute_tensor
model.pos_embed = nn.Parameter(
    distribute_tensor(model.pos_embed.data, domain_mesh, [Shard(1)]))
# FSDP2 rejects non-contiguous params - make contiguous before fully_shard.
```

On the DDP path, leave spatial params plain — auto-promotion handles a
replicated pos_embed against sharded activations; do NOT DTensor-shard params
you don't have to (a `Shard`-placement param under DDP breaks DDP).

Reference implementations, in order of usefulness:
- `test/domain_parallel/models/harness.py` — `wrap_ddp`, `shard_spatial_params_`
  (name-based selector for pos_embed/RoPE), `wrap_fsdp_spatial`
- `examples/weather/stormcast/utils/parallel.py` — production `ParallelHelper`
- `examples/minimal/ShardTensorExamples/5_vit_training_loop/` — end-to-end
  benchmark script with DDP/FSDP2/compile flags

Optimizer note: `foreach`-based optimizers (AdamW default) cannot batch plain
tensors together with DTensors (or DTensors on different meshes) in one param
group. Split param groups by `p.device_mesh if isinstance(p, DTensor) else None`.

## torch.compile with ShardTensor

- **Sharded (ring) attention cannot live inside a compiled region** — see
  `physicsnemo/domain_parallel/shard_utils/attention_patches.py`. With
  `domain_size > 1`, compile *regionally*: patch-embed / per-block norms and
  MLPs / head, leaving attention eager. With `domain_size == 1`, compile the
  whole model.
- **Pass `dynamic=False`.** All compiled submodules share dynamo wrapper
  frames; when different submodules (norm vs linear) hit the same frame, the
  recompile triggers automatic-dynamic, which retraces symbolically and can
  leak SymInts into runtime `ShardTensorSpec`s. Fixed-shape workloads gain
  nothing from dynamic tracing anyway.
- `torch._dynamo.reset()` between input-size changes in sweeps.
- Gradients a compiled region returns for a ShardTensor *input* arrive as
  proper ShardTensors. This relies on `torch.autograd.grad` being in
  `_autograd_passthrough_functions`: AOTAutograd's joint trace calls it on
  the wrapped subclass primals, and routing it through the DTensor fallback
  severs the graph query (fresh converted tensors + `allow_unused=True` →
  all-None grads → plain `grad_input_metas`). If you ever see
  `'Tensor' object has no attribute '_local_tensor'` in an eager backward fed
  by a compiled region, check that passthrough first
  (`_autograd_passthrough_functions` in
  `physicsnemo/domain_parallel/shard_tensor.py`; regression coverage lives in
  `test/domain_parallel/test_compile.py`, added with the torch.compile
  enablement work — absent on builds that predate it).

## Debugging pitfalls (each of these cost real time — check them first)

1. **`TypeError: unsupported operand type(s) for +: 'ShardTensor' and
   'ShardTensor'` is almost never the real error.** Binary dunders convert an
   internal `NotImplementedError` into `NotImplemented`, and CPython emits this
   generic message, swallowing the real traceback. Temporarily replace `x + y`
   with `torch.add(x, y)` to surface the true exception.
2. **In-place `x.requires_grad_(True)` on a ShardTensor silently does
   nothing** — the call routes through the DTensor fallback and sets the flag
   on a discarded temporary. Use `scatter_tensor(..., requires_grad=True)` or
   thread gradients through parameters.
3. **`torch.autograd.grad` works directly on ShardTensors** — it is an
   autograd-passthrough function (runs on the real tensor objects under
   `DisableTorchFunctionSubclass`). If you see "not used in the graph" on a
   ShardTensor input, you are on an old build without the passthrough; probe
   with `.backward()` + `tensor.register_hook(...)` there instead. Beware
   that *monkeypatching* `torch.autograd.grad` (e.g. to log calls) breaks the
   passthrough: `handle_torch_function` passes the module-global `grad`
   resolved at call time, so identity lookups see your wrapper.
4. **Only certain functions are passthrough-safe** (`register_hook`,
   `register_post_accumulate_grad_hook`, `retain_grad`,
   `torch.autograd.grad` — see `_autograd_passthrough_functions` in
   `shard_tensor.py`). Any other identity-sensitive method may act on a
   converted temporary.
5. Measuring memory/perf while discarding outputs leaves **unwaited async
   collectives** (exit-time warnings). Resolve with
   `to_local()`/`AsyncCollectiveTensor.wait()` on discarded results.
6. `CommDebugMode` (`torch.distributed.tensor.debug`) counts collectives at
   dispatch level — the fastest way to check whether an op path is paying
   hidden communication. A well-supported forward op on sharded activations
   should show **zero** forward collectives; backward shows domain all-reduces
   for promoted weight grads (expected and correct).

## Enabling new layers / ops

Read `references/new-op-patterns.md` before writing any patch. Summary of the
decision process:

1. **Try the model unmodified first.** The generic fallback (convert to
   DTensor, run, convert back) covers most ops correctly. Only write a patch
   when you observe: a `MissingShardPatch`/`UndeterminedShardingError`, wrong
   numerics vs a single-GPU run, or unacceptable communication (redistribution
   to Replicate) in `CommDebugMode`.
2. Patches are **registered from user code at import time** — no physicsnemo
   fork needed:
   `ShardTensor.register_function_handler(torch.nn.functional.foo, wrapper)`
   (Python/`__torch_function__` level),
   `ShardTensor.register_dispatch_handler(aten.foo.default, fn)`
   (`__torch_dispatch__` level), and
   `ShardTensor.register_named_function_handler("lib.op.default", wrapper)`
   for `torch.library.custom_op`s.
3. Use the existing patches in `physicsnemo/domain_parallel/shard_utils/` as
   templates: `pooling_patches.py` (config gating + `MissingShardPatch`),
   `conv_patches.py` + `halo.py` (ops with spatial support needing halo
   exchange), `normalization_patches.py` (explicit `autograd.Function` with
   custom backward), `view_ops.py` (dual-level registration; shape-only ops).

## Testing new layers

Read `references/testing.md`. The one-line summary: scatter a full input,
run the module distributed and single-GPU, and compare outputs *and gradients*
with `numerical_shard_tensor_check(mesh, module, [sharded_x], {},
check_grads=True)` under the `multigpu_static` marker, launched as

```bash
torchrun --nproc-per-node 4 -m pytest test/... --multigpu-static -m multigpu_static
```

A forward-only test proves almost nothing — **the weight gradient is where
sharding bugs live** (it is Partial over the domain mesh and must be reduced).
Always `check_grads=True`, always disable TF32 for the comparison.

## Related resources

- `references/integration-checklist.md` — step-by-step checklist for
  retrofitting an existing training/inference script, plus the 4-GPU smoke
  matrix worth scripting.
- `references/new-op-patterns.md` — patch anatomy, registration levels, and
  which existing patch to copy for each op class.
- `references/testing.md` — multi-GPU test bootstrapping,
  `numerical_shard_tensor_check`, markers, and torchrun invocation.
- `physicsnemo-discover` — for choosing models, datapipes, and examples.
