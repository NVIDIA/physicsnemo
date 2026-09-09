# Converting a training/inference script to domain parallelism

A checklist for retrofitting an existing single-GPU or DDP script. The model
file should end this process **unchanged** — every step below lives in the
script. If you find yourself editing `forward()`, stop and re-read the
auto-promotion section of SKILL.md.

## Training

1. **Init + mesh.** `DistributedManager.initialize()`; validate
   `ddp_size * domain_size == world_size` (infer one from the other; error
   with a clear message otherwise). Build the 2-D mesh with named axes
   `["ddp", "domain"]` whenever `world_size > 1`, even if one axis is size 1 —
   explicit axes prevent the classic bug of DDP reducing over the world group.
2. **Batch semantics.** Global batch is divided over the *ddp* axis only.
   Enforce `batch % ddp == 0` and per-domain-group batch == 1. In the
   dataloader, each domain group reads ONE sample (in a simple port: rank 0 of
   each domain group reads, then scatters; in a real app each rank reads its
   own shard of the sample).
3. **Scatter inputs.** `scatter_tensor(x, src_global_rank, domain_mesh,
   (Shard(spatial_dim),))` for inputs; `(Replicate(),)` for
   labels/targets — loss gradients scatter correctly through the sharded
   output on their own.
4. **Sync weights over the domain axis** (broadcast plain params from domain
   rank 0). DDP handles the ddp axis at construction; FSDP2 handles neither
   axis's initial sync — do the domain broadcast in all cases, before
   `fully_shard`.
5. **Wrap** per the table in SKILL.md (DDP by default; FSDP2 = `fully_shard`
   only when you want parameter sharding or have DTensor spatial params;
   FSDP1 never). Keep a `--fsdp`-style opt-in flag rather than hardcoding.
6. **Optimizer.** Build after wrapping. Split param groups by tensor
   type/mesh if any params are DTensors (foreach kernels can't mix).
7. **Loss.** Plain `nn.CrossEntropyLoss()` etc. on the (sharded or partial)
   output and replicated target works — no changes. `.backward()` works.
   Don't call `loss.item()` inside anything that will be traced.
8. **Logging/checkpointing.** `shard.full_tensor()` /
   `redistribute(placements=[Replicate()])` gathers when a full tensor is
   needed on every rank. Log scalars from rank 0. For checkpoints on the FSDP2
   path use the standard FSDP2 `get_model_state_dict` flow; on the DDP path
   params are plain — nothing changes.
9. **Verification before declaring success.** Fixed seed, few steps, compare
   the loss trajectory of `domain_size=N` against the single-GPU baseline —
   they should track to ~1e-3 in fp32 (differences are reduction order only).
   A run that "doesn't crash" is not a verified port; wrong gradient
   reductions train quietly and badly.

## Inference

Same steps 1–3; no wrapper, no optimizer, no domain broadcast *if* weights are
loaded from a checkpoint (loading replicates them by construction — but a
broadcast is cheap insurance). Under `torch.no_grad()`, be careful when
discarding outputs (benchmarks, warmups): async collectives are only waited
when a result is used — call `to_local()` / `.wait()` on
`AsyncCollectiveTensor`s or PyTorch prints unwaited-collective warnings at
exit.

## Adding torch.compile (optional, last)

Only after eager parallel runs are verified. `dynamic=False`; regional compile
excluding sharded attention when `domain_size > 1`; `torch._dynamo.reset()`
between shape changes; re-verify numerics compiled-vs-eager. See SKILL.md for
current known issues at the compiled/eager gradient boundary.

## Smoke matrix worth scripting (fits 4 GPUs)

single-GPU baseline; `ddp=4`; `domain=4` (no wrapper + broadcast path);
`ddp=2,domain=2` with plain DDP; the same with FSDP2; FSDP2 degenerate
`ddp=1` (params become DTensors, no ddp comms — valid and worth testing);
each ± compile; one `inference_only`; plus the expected-failure cases
(invalid mesh factorization, per-domain batch > 1). Script it as a small
pytest harness that launches each configuration of
`examples/minimal/ShardTensorExamples/5_vit_training_loop/training_script.py`
through torchrun subprocesses and asserts on exit codes.
