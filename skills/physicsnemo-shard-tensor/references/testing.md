# Testing ShardTensor ops and models

The gold standard is **numerical equivalence against a single-GPU run**:
same module, same (seeded) weights, full input vs scattered input — outputs
AND gradients must match at ~1e-5 in fp32. Forward-only comparisons prove
almost nothing; sharding bugs live in gradients (weight grads are `Partial`
over the domain mesh and must be reduced).

## How physicsnemo runs multi-GPU tests

Two markers, gated by mutually-exclusive pytest flags (`test/conftest.py`):

- `@pytest.mark.multigpu_static` — the whole pytest session runs *inside*
  torchrun with a fixed world; `DistributedManager` is initialized at pytest
  configure time and mesh fixtures come from
  `test/plugins/distributed_fixtures` (`distributed_mesh` = 1-D over all
  ranks, `distributed_mesh_2d` = 2-D). This is the mode for ShardTensor tests.

```bash
torchrun --nproc-per-node 4 -m pytest test/domain_parallel/ \
    --multigpu-static -m multigpu_static
```

- `@pytest.mark.multigpu_dynamic` — the test itself spawns processes; not used
  for ShardTensor op tests.

Without either flag, all multigpu tests are skipped — so these tests are
invisible to a plain `pytest` run; don't be fooled into thinking they passed.

Precision: comparisons at 1e-4/1e-5 require **TF32 disabled** — the
domain_parallel `conftest.py` has an autouse fixture forcing full fp32
(`allow_tf32 = False`, `fp32_precision = "ieee"`). Replicate it in user repos.

## Op-level test template

From `test/domain_parallel/ops/test_padding.py` — the whole test is
declarative:

```python
import pytest, torch
from torch.distributed.tensor.placement_types import Shard
from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor

@pytest.mark.multigpu_static
@pytest.mark.parametrize("backward", [False, True])
def test_my_op_1dmesh(distributed_mesh, backward):
    dm = DistributedManager()
    image = torch.randn(1, 16, 128, 128).to(dm.device)   # batch 1!

    sharded = scatter_tensor(image, 0, distributed_mesh, (Shard(2),),
                             requires_grad=backward)

    module = MyNewLayer(...)
    numerical_shard_tensor_check(distributed_mesh, module, [sharded], {},
                                 check_grads=backward)
```

`numerical_shard_tensor_check` (`test/domain_parallel/ops/utils.py`) runs the
module on the distributed input and on the gathered full input, compares
forward outputs and (with `check_grads=True`) every parameter gradient. Useful
knobs: `atol/rtol`, `amp=True` for autocast testing, `distribute_fn=` to test
under a DDP/FSDP2 wrapper instead of replication, `output_check_fn=` to assert
output placements.

Cover at minimum: 1-D mesh `Shard(spatial_dim)`, 2-D mesh
(`distributed_mesh_2d`, e.g. `(Shard(2), Shard(3))`), forward and backward
parametrized, and every config branch of your patch — including one test
asserting `MissingShardPatch` is raised for unsupported configs:

```python
with pytest.raises(NotImplementedError):   # MissingShardPatch subclasses it
    module(sharded_bad_config)
```

## Model-level test template

`test/domain_parallel/models/harness.py` makes a model test a few declarative
lines: describe how to build the model, the full inputs, and how to shard
them, then hand a `DomainParallelModelCase` to
`run_domain_parallel_model_check`. Test both strategies:

- `strategy="ddp"` — all params plain, `wrap_ddp` over the mesh group.
- `strategy="fsdp_spatial"` — spatial params (pos_embed / RoPE, selected by
  name) DTensor-sharded on the domain mesh, `fully_shard` (FSDP2) over the ddp
  mesh. This mirrors production (`ParallelHelper.distribute_model`).

Never add `strategy="distribute_module"` for new work — it exists only as a
legacy comparison point.

## Testing in user code (outside physicsnemo): vendorable template

The physicsnemo `test/` tree is not an installed package — vendor these two
small files into the user repo instead of importing physicsnemo test
internals.

`tests/conftest.py`:

```python
import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--multigpu", action="store_true",
                     help="run distributed tests (launch via torchrun)")


def pytest_configure(config):
    config.addinivalue_line("markers", "multigpu: needs torchrun + --multigpu")
    if config.getoption("--multigpu"):
        from physicsnemo.distributed import DistributedManager
        DistributedManager.initialize()


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--multigpu"):
        skip = pytest.mark.skip(reason="needs torchrun + --multigpu")
        for item in items:
            if "multigpu" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def domain_mesh():
    from physicsnemo.distributed import DistributedManager
    dm = DistributedManager()
    torch.cuda.set_device(dm.device)
    # 1-D mesh over all ranks; add a (ddp, domain) 2-D variant for wrapper
    # tests: dm.initialize_mesh((world // 2, 2), ("ddp", "domain"))["domain"]
    return dm.initialize_mesh((dm.world_size,), mesh_dim_names=("domain",))


@pytest.fixture(autouse=True)
def deterministic():
    # Identical per-rank weight init is what makes distributed-vs-local
    # comparison valid; TF32 (~1e-3 error) would swamp the 1e-4 tolerance.
    torch.manual_seed(42)
    prev = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
```

`tests/shard_check.py` — the distributed-vs-local comparison:

```python
import copy
import torch
from torch.distributed.tensor import DTensor
from physicsnemo.domain_parallel import ShardTensor


def check_distributed_vs_local(module, sharded_args, *, check_grads=True,
                               atol=1e-4, rtol=1e-4):
    """Run `module` on sharded and on full inputs; outputs and grads must match.

    `module` must NOT be wrapped (no DDP/FSDP2) - the deepcopy below is the
    single-device reference and must share the distributed module's weights,
    which the seed fixture guarantees at construction time.
    """
    local_module = copy.deepcopy(module)

    full_args = []
    for a in sharded_args:
        if isinstance(a, ShardTensor):
            fa = a.full_tensor().detach().requires_grad_(a.requires_grad)
        else:
            fa = a
        full_args.append(fa)

    out_dist = module(*sharded_args)
    out_local = local_module(*full_args)

    out_dist_full = (out_dist.full_tensor()
                     if isinstance(out_dist, (ShardTensor, DTensor)) else out_dist)
    torch.testing.assert_close(out_dist_full, out_local, atol=atol, rtol=rtol)

    if check_grads:
        # Same scalar loss on both sides. sum() of a sharded output routes
        # through the ShardTensor reduction ops - that is part of the test.
        out_dist.sum().backward()
        out_local.sum().backward()
        for (name, p_d), (_, p_l) in zip(module.named_parameters(),
                                         local_module.named_parameters()):
            assert (p_d.grad is None) == (p_l.grad is None), name
            if p_d.grad is None:
                continue
            g = p_d.grad
            if isinstance(g, (ShardTensor, DTensor)):
                g = g.full_tensor()
            torch.testing.assert_close(g, p_l.grad, atol=atol, rtol=rtol,
                                       msg=f"grad mismatch: {name}")
        # Input grads too, when requested via requires_grad at scatter time.
        for sa, fa in zip(sharded_args, full_args):
            if isinstance(sa, ShardTensor) and sa.requires_grad:
                torch.testing.assert_close(sa.grad.full_tensor(), fa.grad,
                                           atol=atol, rtol=rtol)
```

A test then looks identical to the physicsnemo op tests:

```python
@pytest.mark.multigpu
def test_my_layer(domain_mesh):
    x = torch.randn(1, 16, 128, 128, device="cuda")          # batch 1!
    sharded = scatter_tensor(x, 0, domain_mesh, (Shard(2),),
                             requires_grad=True)
    check_distributed_vs_local(MyNewLayer(16), [sharded])
```

Launch (CI and locally):

```bash
torchrun --standalone --nproc-per-node 2 -m pytest tests/ --multigpu -m multigpu
```

2 GPUs catch most sharding bugs; keep at least one 4-GPU test on a 2×2
`(ddp, domain)` mesh — mesh-axis mixups (reducing over the wrong axis) are
invisible on a 1-D mesh. Caveats that make this template valid: per-domain
batch of 1; every rank constructs the module under the same seed *before* any
wrapper; `full_tensor()` is differentiable, so gathering for comparison does
not break the grad checks; hooks/`requires_grad` must be set at
`scatter_tensor(...)` time, not after (see SKILL.md pitfalls).

## Regression tests for compile support

When a patch must work under `torch.compile`, add a compile variant (see
`test/domain_parallel/test_compile.py`, added with the torch.compile
enablement work — absent on builds that predate it): compile the module
(`backend="inductor"`, `dynamic=False`; the suite historically uses
`aot_eager` as a fallback backend when inductor is flaky), run forward AND
backward twice (the second iteration catches guard/recompile issues), and
compare against eager. Compiled tests must also assert that outputs remain
usable by *eager* ops afterward (hash the spec, add two results) — the
eager/compiled boundary is where subclass metadata gets lost.
