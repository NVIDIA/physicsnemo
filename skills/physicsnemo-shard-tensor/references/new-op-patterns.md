# Writing and registering ShardTensor op patches

## When a patch is needed (and when it is not)

ShardTensor intercepts ops at two layers:

- **`__torch_function__`** (Python API level: `F.linear`, `Tensor.view`,
  `F.pad`, ...). Checks `_function_registry` and `_named_function_registry`;
  otherwise falls back via `_torch_function_fallback_via_dtensor`, which
  converts ShardTensors to DTensors **with autograd bridges**
  (`_ShardTensorToDTensor` / `_DTensorToShardTensor` autograd Functions) and
  auto-promotes plain tensors to `Replicate` DTensors on the reference mesh.
- **`__torch_dispatch__`** (ATen level: `aten.foo.default`). Checks
  `_dispatch_registry` (by op object) and `_dispatch_registry_by_name`;
  otherwise falls back via `_dispatch_fallback_via_dtensor` — *pure* data
  conversion, because native autograd wraps above this level.

The fallback is correct for most ops. Write a patch only when one of these is
observed:

1. `MissingShardPatch` / `UndeterminedShardingError` raised.
2. Wrong numerics vs a single-GPU reference (always verify with the test
   harness before assuming).
3. `CommDebugMode` shows the fallback redistributing (all-gather to Replicate)
   where a local computation + spec propagation would do — i.e. a performance
   patch.
4. The op has spatial support crossing shard boundaries (convolution, pooling
   with overlap, padding, interpolation) — needs halo exchange.
5. The op's DTensor decomposition is unrepresentable for the sharding (e.g.
   anything that flattens the sharded dim together with other dims).

## Registration (from user code, at import time)

```python
from physicsnemo.domain_parallel import ShardTensor
from torch.ops import aten

# Python-level (methods and functional API):
ShardTensor.register_function_handler(torch.nn.functional.my_op, my_op_wrapper)
ShardTensor.register_function_handler(torch.Tensor.my_method, my_op_wrapper)

# ATen ops can ALSO arrive at __torch_function__ (PyTorch 2.6+ internal
# codepaths call aten ops directly on subclasses) - register both when the op
# appears inside decompositions of higher-level ops (see view_ops.py):
ShardTensor.register_function_handler(aten.my_op.default, my_op_aten_wrapper)
ShardTensor.register_dispatch_handler(aten.my_op.default, my_op_dispatch)

# torch.library.custom_op-defined ops match by NAME:
ShardTensor.register_named_function_handler("mylib.my_op.default", wrapper)
```

Put registrations at module scope in a file your application imports early
(mirroring how `physicsnemo.domain_parallel.shard_utils.__init__` imports each
patch module for its side effects).

## Handler signatures and autograd rules

**Function-level handler** — `wrapper(func, types, args, kwargs) -> result`.
This runs *above* autograd, so if the op is differentiable you must provide the
autograd connection yourself: either build the result through differentiable
building blocks (ops that themselves dispatch through ShardTensor), or write an
explicit `torch.autograd.Function` (see `normalization_patches.py`,
`custom_ops/_reductions.py`). If the op consumes plain weights, promote them
through the differentiable path (`DTensor.from_local` with `Replicate`) or
all-reduce their gradient over the domain group in your backward — **the
weight gradient is Partial across domain ranks** (each rank saw only its slice
of the data); forgetting the reduction produces silently wrong training.

**Dispatch-level handler** — `handler(*args, **kwargs) -> result`, called with
the raw ATen op arguments. Autograd wraps *above* `__torch_dispatch__`, so a
pure function is sufficient — no `autograd.Function` (see
`_sharded_view_dispatch` in `view_ops.py`).

## Anatomy of a patch (template)

Modeled on `pooling_patches.py` / `conv_patches.py`:

```python
from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.domain_parallel.shard_utils.patch_core import (
    MissingShardPatch, promote_to_iterable,
)

def generic_my_op_wrapper(func, types, args, kwargs):
    # 1. Normalize arguments (scalars -> per-dim tuples, defaults, etc.)
    x, weight, stride = repackage_my_op_args(*args, **kwargs)

    # 2. Gate on supported configurations. Raise MissingShardPatch (a
    #    NotImplementedError subclass) for configs you have not implemented -
    #    NEVER silently compute something wrong. NOTE: inside a binary dunder
    #    this surfaces as a bare TypeError; that is expected.
    if not config_is_supported(stride, x.placements):
        raise MissingShardPatch(f"my_op: unsupported config {stride=}")

    # 3. If the op has spatial support, exchange halos first
    #    (physicsnemo.domain_parallel.shard_utils.halo).

    # 4. Compute LOCALLY on the local shard.
    local_out = func(x._local_tensor, weight, stride)

    # 5. Propagate the sharding: output placements + per-rank shard shapes.
    #    For shape-preserving ops, reuse the input spec. For shape-changing
    #    ops, compute output shard shapes per rank from the input's
    #    spec.sharding_shapes() - pure arithmetic, no collectives.
    return ShardTensor.from_local(
        local_out, x.device_mesh, x.placements,
        sharding_shapes=computed_shapes_dict,   # or "chunk" + global_shape
    )

ShardTensor.register_function_handler(torch.nn.functional.my_op,
                                      generic_my_op_wrapper)
```

Result-construction options, cheapest first:

1. Reuse/edit the input spec and `ShardTensor.__new__(local, spec,
   requires_grad=...)` — zero communication; use when you can derive the
   output spec exactly (this is what `custom_ops/_reductions.py`
   `build_reduction_result` does).
2. `ShardTensor.from_local(..., sharding_shapes=<dict>)` — explicit per-rank
   shapes, no communication.
3. `from_local(..., sharding_shapes="chunk", global_shape=...)` — chunk
   semantics, no communication, only correct for evenly-chunkable results.
4. `from_local(..., sharding_shapes="infer")` — collective communication to
   learn neighbor shapes. Avoid in hot paths and NEVER in code that must be
   AOT-traceable (blocking collectives don't trace).

## Spec hygiene (compile compatibility)

- Store `_sharding_shapes` as plain `tuple[int, ...]` entries, never
  `torch.Size` (Size is symint-special-cased under fakeification — see the
  `ShardTensorSpec` docstring).
- Never persist trace-time (SymInt-bearing) shapes into runtime specs; if you
  build specs in code that runs under tracing, derive shapes from values that
  are concrete at runtime (`__tensor_unflatten__` shows the pattern).
- Ops that produce `Partial` placements (reductions over the sharded dim) are
  valid, including at compiled-region boundaries: the tangent-coercion hooks
  handle the Partial→Replicate flip, and grads for ShardTensor inputs of
  compiled regions arrive as ShardTensors (backed by the
  `torch.autograd.grad` autograd-passthrough).

## Existing patches to use as guides

| Pattern you need | Template file (`physicsnemo/domain_parallel/shard_utils/`) |
|---|---|
| Config gating, arg normalization | `pooling_patches.py` |
| Halo exchange (spatial support) | `conv_patches.py`, `halo.py` |
| Custom autograd.Function backward | `normalization_patches.py`, `../custom_ops/_reductions.py` |
| Shape-only ops, dual-level registration | `view_ops.py` |
| Padding semantics across shards | `padding.py` |
| Attention (ring), compile exclusion notes | `attention_patches.py` |
| Index/gather ops | `index_ops.py` |
