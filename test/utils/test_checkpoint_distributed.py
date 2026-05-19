# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Distributed checkpoint tests.

Run with::

    torchrun --nproc-per-node 4 -m pytest --multigpu-static \
        test/utils/test_checkpoint_distributed.py -x
"""

import shutil
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import (
    ShardingStrategy,
)
from torch.distributed.tensor import DTensor, distribute_module, distribute_tensor
from torch.distributed.tensor.placement_types import Shard

from physicsnemo import Module
from physicsnemo.core.version_check import check_version_spec
from physicsnemo.distributed import DistributedManager
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.utils import load_checkpoint, load_model_weights, save_checkpoint

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def shared_tmp_dir():
    """Broadcast a temp directory from rank 0 so all ranks use the same path."""
    dm = DistributedManager()
    d = tempfile.mkdtemp() if dm.rank == 0 else ""
    obj = [d]
    dist.broadcast_object_list(obj, src=0)
    path = obj[0]
    yield path
    dist.barrier()
    if dm.rank == 0:
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Plain FSDP (1-D mesh, no domain sharding)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@pytest.mark.parametrize("sync_module_states", [True, False])
@pytest.mark.parametrize("use_orig_params", [True, False])
@pytest.mark.parametrize(
    "sharding_strategy",
    [ShardingStrategy.NO_SHARD, ShardingStrategy.FULL_SHARD],
)
def test_fsdp_checkpoint_roundtrip(
    shared_tmp_dir, use_orig_params, sharding_strategy, sync_module_states
):
    """Save and load a plain FSDP model through the checkpoint utilities."""
    dm = DistributedManager()
    if dm.world_size < 2:
        pytest.skip("Need at least 2 ranks")

    device = dm.device
    mesh = init_device_mesh("cuda", (dm.world_size,), mesh_dim_names=("world",))

    model = FullyConnected(
        in_features=16, out_features=16, num_layers=2, layer_size=32
    ).to(device)
    fsdp_model = FSDP(
        model,
        device_mesh=mesh["world"],
        sharding_strategy=sharding_strategy,
        use_orig_params=use_orig_params,
        sync_module_states=sync_module_states,
    )
    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

    x = torch.randn(4, 16, device=device)
    for _ in range(3):
        loss = fsdp_model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    with torch.no_grad():
        ref_output = fsdp_model(x).clone()

    save_checkpoint(
        shared_tmp_dir,
        models=fsdp_model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        metadata={"test": True},
        optimizer_model=fsdp_model,
    )
    dist.barrier()

    # Fresh model + FSDP + optimizer, then load
    model2 = FullyConnected(
        in_features=16, out_features=16, num_layers=2, layer_size=32
    ).to(device)
    fsdp_model2 = FSDP(
        model2,
        device_mesh=mesh["world"],
        sharding_strategy=sharding_strategy,
        use_orig_params=use_orig_params,
        sync_module_states=sync_module_states,
    )
    optimizer2 = torch.optim.Adam(fsdp_model2.parameters(), lr=1e-3)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=5)

    meta: dict = {}
    epoch = load_checkpoint(
        shared_tmp_dir,
        models=fsdp_model2,
        optimizer=optimizer2,
        scheduler=scheduler2,
        metadata_dict=meta,
        optimizer_model=fsdp_model2,
    )

    assert epoch == 3
    assert meta.get("test") is True

    with torch.no_grad():
        loaded_output = fsdp_model2(x)
    assert torch.allclose(ref_output, loaded_output, rtol=1e-5, atol=1e-5), (
        "Model outputs differ after FSDP checkpoint round-trip"
    )
    assert scheduler2.last_epoch == scheduler.last_epoch


# ---------------------------------------------------------------------------
# Plain FSDP + channels_last  (regression for cross-rank layout mismatch)
# ---------------------------------------------------------------------------


class _ConvNet(nn.Module):
    """Tiny conv net so the parameter set includes a 4-D weight."""

    def __init__(self, in_ch: int = 4, out_ch: int = 8, k: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k // 2, bias=True)
        self.gn = nn.GroupNorm(num_groups=4, num_channels=out_ch)

    def forward(self, x):
        return self.gn(self.conv(x))


def _all_ranks_bit_exact(t: torch.Tensor) -> bool:
    """True iff every rank holds element-wise bit-identical values for *t*."""
    t_min = t.detach().clone().float()
    t_max = t.detach().clone().float()
    dist.all_reduce(t_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
    return torch.equal(t_min, t_max)


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@pytest.mark.parametrize("use_orig_params", [True, False])
@pytest.mark.parametrize(
    "sharding_strategy",
    [ShardingStrategy.NO_SHARD],
)
def test_fsdp_checkpoint_channels_last_roundtrip(
    shared_tmp_dir, use_orig_params, sharding_strategy
):
    """Round-trip an FSDP+channels_last conv model and assert per-rank parity.

    Regression for a layout-mismatch bug in DCP's broadcast_from_rank0 path:
    ``dist.broadcast`` accepts a channels_last sender (``is_contiguous`` check
    passes for that format) but transfers bytes in storage order, while
    receivers allocate ``torch.empty(shape, dtype, device)`` (standard NCHW),
    so 4-D conv weights were silently permuted on non-rank-0. The fix
    (``_force_standard_contiguous`` on rank 0 before ``set_model_state_dict``)
    keeps sender and receiver layouts consistent.

    Asserts bit-exact agreement across ranks on the live FlatParameter (for
    ``use_orig_params=False``) / each original parameter (for True). Output
    equivalence isn't sufficient — a permuted conv weight preserves abs-sum
    and the model can stagger toward similar outputs over noise — so we
    check the parameter values directly.

    The optimizer state is intentionally *not* asserted here. The optim load
    path is layout-correct (verified by the standalone smoketest and by
    running this test in isolation), but suite-level state pollution (NCCL
    allreduce ordering across many prior tests) accumulates FP noise in the
    pre-load training step, which then survives the load and makes a tight
    cross-rank check flaky. The existing ``test_fsdp_checkpoint_roundtrip``
    already covers the optim path with a tolerance-based output comparison
    that's robust to that noise.
    """
    dm = DistributedManager()
    if dm.world_size < 2:
        pytest.skip("Need at least 2 ranks")

    device = dm.device
    mesh = init_device_mesh("cuda", (dm.world_size,), mesh_dim_names=("world",))

    # Build, move to channels_last (only conv weights are affected), wrap.
    torch.manual_seed(0)
    model = _ConvNet().to(device=device, memory_format=torch.channels_last)
    fsdp_model = FSDP(
        model,
        device_mesh=mesh["world"],
        sharding_strategy=sharding_strategy,
        use_orig_params=use_orig_params,
        sync_module_states=True,
    )
    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=1e-3)

    # Same x on every rank (we want the source-of-truth state to be identical
    # across ranks pre-save, so any post-load divergence is checkpoint-induced).
    x = torch.randn(2, 4, 8, 8, device=device).contiguous(
        memory_format=torch.channels_last
    )
    for _ in range(2):
        fsdp_model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()

    save_checkpoint(
        shared_tmp_dir,
        models=fsdp_model,
        optimizer=optimizer,
        epoch=2,
        optimizer_model=fsdp_model,
    )
    dist.barrier()

    # Build a *differently-seeded* fresh model so sync_module_states alone can't
    # mask the bug by leaving rank 0's pre-load values on every rank.
    torch.manual_seed(dm.rank + 1234)
    model2 = _ConvNet().to(device=device, memory_format=torch.channels_last)
    fsdp_model2 = FSDP(
        model2,
        device_mesh=mesh["world"],
        sharding_strategy=sharding_strategy,
        use_orig_params=use_orig_params,
        sync_module_states=True,
    )
    optimizer2 = torch.optim.Adam(fsdp_model2.parameters(), lr=1e-3)
    # Step once so optimizer state is shaped before the load.
    fsdp_model2(x).sum().backward()
    optimizer2.step()
    optimizer2.zero_grad()

    epoch = load_checkpoint(
        shared_tmp_dir,
        models=fsdp_model2,
        optimizer=optimizer2,
        optimizer_model=fsdp_model2,
    )
    assert epoch == 2

    # --- Cross-rank parity checks ------------------------------------------
    # FlatParameter (use_orig_params=False) or each original param (True).
    if use_orig_params:
        for name, p in fsdp_model2.named_parameters():
            assert _all_ranks_bit_exact(p), (
                f"Parameter '{name}' (shape={tuple(p.shape)}) differs across "
                f"ranks after channels_last+FSDP load"
            )
    else:
        flat_param = fsdp_model2._flat_param
        assert _all_ranks_bit_exact(flat_param), (
            "FlatParameter differs across ranks after channels_last+FSDP load"
        )

    # Optimizer state cross-rank check intentionally omitted -- see docstring.


# ---------------------------------------------------------------------------
# Cross-mode load: 1-proc non-distributed save → N-proc FSDP load (with CL)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
def test_cross_mode_channels_last_model_load(shared_tmp_dir):
    """Save from a single (rank-0-only) non-FSDP CL model; load model state
    into N-proc FSDP.

    Realistic "trained on multi-rank, fine-tuned/inspected on a single GPU,
    resumed multi-rank" round-trip with channels_last. Confirms that the
    on-disk model state produced by the non-distributed save path is loadable
    by the distributed FSDP load path on every rank without layout-induced
    corruption.

    Model side asserts:
      * every rank's post-load FlatParameter is bit-exact identical, AND
      * rank 0's logical values match what was saved.

    Cross-rank parity alone can be satisfied by "everyone got the same wrong
    values" (e.g. silent drop), so we also check against the saved snapshot.

    Optimizer cross-mode load is *not* tested here. The non-distributed save
    path writes int-keyed (param-id) optim state via ``optimizer.state_dict()``,
    while the distributed FSDP load path expects FQN-keyed input -- DCP's
    ``_split_optim_state_dict`` early-returns for int keys without converting,
    and the downstream ``_rekey_sharded_optim_state_dict`` then crashes on
    ``int.unflat_param_names``. That's a separate, pre-existing limitation
    of cross-mode optim restore; same-mode optim restore is exercised by
    ``test_fsdp_checkpoint_channels_last_roundtrip`` and is what the
    channels_last fix is concerned with.
    """
    dm = DistributedManager()
    if dm.world_size < 2:
        pytest.skip("Need at least 2 ranks")

    device = dm.device

    # ===== Phase A: 1-proc save on rank 0 only =====
    saved_params: dict[str, torch.Tensor] = {}
    if dm.rank == 0:
        torch.manual_seed(0)
        model_save = _ConvNet().to(device=device, memory_format=torch.channels_last)
        optimizer_save = torch.optim.Adam(model_save.parameters(), lr=1e-3)

        x = torch.randn(2, 4, 8, 8, device=device).contiguous(
            memory_format=torch.channels_last
        )
        # Two steps so the saved weights have actually moved off init.
        for _ in range(2):
            model_save(x).sum().backward()
            optimizer_save.step()
            optimizer_save.zero_grad()

        # Snapshot. ``contiguous()`` pins a canonical layout for comparison;
        # the values are what matter.
        for name, p in model_save.named_parameters():
            saved_params[name] = p.detach().clone().contiguous().cpu()

        # We deliberately save the optimizer state too, mirroring real-world
        # usage, but the load side will not consume it (see docstring).
        save_checkpoint(
            shared_tmp_dir,
            models=model_save,
            optimizer=optimizer_save,
            epoch=2,
        )
    dist.barrier()

    # ===== Phase B: N-proc FSDP-only load (model only) =====
    mesh = init_device_mesh("cuda", (dm.world_size,), mesh_dim_names=("world",))

    # Different per-rank seed so sync_module_states alone can't mask anything.
    torch.manual_seed(dm.rank + 4242)
    model_load = _ConvNet().to(device=device, memory_format=torch.channels_last)
    fsdp_load = FSDP(
        model_load,
        device_mesh=mesh["world"],
        sharding_strategy=ShardingStrategy.NO_SHARD,
        use_orig_params=False,
        sync_module_states=True,
    )

    # Pass optimizer=None: cross-mode optim load is a separate, pre-existing
    # PyTorch DCP limitation (see docstring). We're testing the model path.
    epoch = load_checkpoint(
        shared_tmp_dir,
        models=fsdp_load,
    )
    assert epoch == 2

    # ===== Phase C.1: per-rank parity =====
    flat_param = fsdp_load._flat_param
    assert _all_ranks_bit_exact(flat_param), (
        "FlatParameter differs across ranks after cross-mode model load"
    )

    # ===== Phase C.2: loaded values match saved values (rank 0) =====
    # Collective: gather the full model state dict on every rank.
    full_loaded_model = get_model_state_dict(
        fsdp_load, options=StateDictOptions(full_state_dict=True)
    )
    if dm.rank == 0:
        for name, expected in saved_params.items():
            assert name in full_loaded_model, f"Loaded model state missing '{name}'"
            actual = full_loaded_model[name].detach().contiguous().cpu()
            assert torch.equal(actual, expected), (
                f"Logical model values for '{name}' differ between save and "
                f"load (cross-mode)"
            )


# ---------------------------------------------------------------------------
# load_model_weights — plain FSDP
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@pytest.mark.parametrize("sync_module_states", [True, False])
@pytest.mark.parametrize("model_type", ["physicsnemo", "pytorch"])
@pytest.mark.parametrize("use_orig_params", [True, False])
@pytest.mark.parametrize(
    "sharding_strategy",
    [ShardingStrategy.NO_SHARD, ShardingStrategy.FULL_SHARD],
)
def test_load_model_weights_fsdp(
    shared_tmp_dir, use_orig_params, sharding_strategy, model_type, sync_module_states
):
    """load_model_weights loads a .mdlus or .pt file into an FSDP-wrapped model."""
    dm = DistributedManager()
    if dm.world_size < 2:
        pytest.skip("Need at least 2 ranks")

    device = dm.device
    mesh = init_device_mesh("cuda", (dm.world_size,), mesh_dim_names=("world",))

    if model_type == "physicsnemo":
        model = FullyConnected(
            in_features=16, out_features=16, num_layers=2, layer_size=32
        ).to(device)
        weights_file = f"{shared_tmp_dir}/trained.mdlus"
    else:
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16)).to(
            device
        )
        weights_file = f"{shared_tmp_dir}/trained.pt"

    # Train a few steps so weights diverge from init
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(4, 16, device=device)
    for _ in range(3):
        model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()

    # Save trained weights (rank 0 only)
    if dm.rank == 0:
        if model_type == "physicsnemo":
            model.save(weights_file)
        else:
            torch.save(model.state_dict(), weights_file)
    dist.barrier()

    with torch.no_grad():
        ref_output = model(x).clone()

    # Build a fresh FSDP-wrapped model and load the weights
    if model_type == "physicsnemo":
        model2 = FullyConnected(
            in_features=16, out_features=16, num_layers=2, layer_size=32
        ).to(device)
    else:
        model2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16)).to(
            device
        )
    fsdp_model2 = FSDP(
        model2,
        device_mesh=mesh["world"],
        sharding_strategy=sharding_strategy,
        use_orig_params=use_orig_params,
        sync_module_states=sync_module_states,
    )

    load_model_weights(fsdp_model2, weights_file)

    with torch.no_grad():
        loaded_output = fsdp_model2(x)
    assert torch.allclose(ref_output, loaded_output, rtol=1e-5, atol=1e-5), (
        "Model outputs differ after load_model_weights into FSDP model"
    )


# ---------------------------------------------------------------------------
# FSDP + ShardTensor on 2-D mesh  (ddp × domain)
# ---------------------------------------------------------------------------

_HAS_TORCH_26 = check_version_spec("torch", "2.6.0", hard_fail=False)


class _PosEmbedModel(Module):
    """Tiny model with a positional-embedding parameter that is selectively sharded."""

    def __init__(self, embed_tokens: int = 24, embed_dim: int = 8, hidden: int = 16):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, embed_tokens, embed_dim))
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) where T is sharded across the domain mesh.
        # Reduce the sharded token dim before the linear layers because
        # nn.Linear flattens leading dims, which DTensor cannot do across
        # a sharded dimension.
        out = x + self.pos_embed
        out = out.mean(dim=1)  # (B, D)
        return self.fc2(torch.relu(self.fc1(out)))


def _partition_pos_embed(
    name: str,
    submodule: nn.Module,
    device_mesh: torch.distributed.device_mesh.DeviceMesh,
):
    """Shard ``pos_embed`` along dim 1 across *device_mesh*."""
    for key, param in submodule._parameters.items():
        if param is None:
            continue
        if "pos_embed" in key:
            sharded = distribute_tensor(
                param, device_mesh=device_mesh, placements=[Shard(1)]
            )
            submodule.register_parameter(key, nn.Parameter(sharded))


@pytest.mark.timeout(60)
@pytest.mark.multigpu_static
@pytest.mark.skipif(not _HAS_TORCH_26, reason="ShardTensor requires torch >= 2.6")
@pytest.mark.parametrize("sync_module_states", [True, False])
@pytest.mark.parametrize("use_orig_params", [True, False])
def test_fsdp_shard_tensor_checkpoint_roundtrip(
    shared_tmp_dir, use_orig_params, sync_module_states
):
    """Checkpoint round-trip with a 2-D mesh: FSDP(NO_SHARD) on ddp, ShardTensor on domain."""
    if use_orig_params:
        pytest.skip(
            "use_orig_params=True + ShardTensor under FSDP NO_SHARD is unsupported: "
            "FSDP writeback fails when local parameter shape changes"
        )
    torch.manual_seed(0)

    dm = DistributedManager()
    if dm.world_size < 4 or dm.world_size % 2 != 0:
        pytest.skip("Need at least 4 ranks (divisible by 2) for 2-D mesh test")

    device = dm.device
    domain_size = 2
    dp_size = dm.world_size // domain_size
    mesh = init_device_mesh(
        "cuda", (dp_size, domain_size), mesh_dim_names=("ddp", "domain")
    )

    embed_tokens = 24  # divisible by domain_size=2

    def _build_distributed_model():
        m = _PosEmbedModel(embed_tokens=embed_tokens, embed_dim=8, hidden=16).to(device)
        m = distribute_module(
            m, device_mesh=mesh["domain"], partition_fn=_partition_pos_embed
        )
        m = FSDP(
            m,
            device_mesh=mesh["ddp"],
            sharding_strategy=ShardingStrategy.NO_SHARD,
            use_orig_params=use_orig_params,
            sync_module_states=sync_module_states,
        )
        return m

    fsdp_model = _build_distributed_model()
    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=1e-3)

    # Create and shard input tensor
    x = torch.randn(2, embed_tokens, 8, device=device)
    x = distribute_tensor(x, device_mesh=mesh["domain"], placements=[Shard(1)])

    for _ in range(3):
        loss = fsdp_model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Capture full model state on rank 0 for reference
    full_options = StateDictOptions(full_state_dict=True)
    ref_params = get_model_state_dict(fsdp_model, options=full_options)

    save_checkpoint(
        shared_tmp_dir,
        models=fsdp_model,
        optimizer=optimizer,
        epoch=3,
        optimizer_model=fsdp_model,
    )
    dist.barrier()

    # Build a fresh distributed model and load
    fsdp_model2 = _build_distributed_model()
    optimizer2 = torch.optim.Adam(fsdp_model2.parameters(), lr=1e-3)

    epoch = load_checkpoint(
        shared_tmp_dir,
        models=fsdp_model2,
        optimizer=optimizer2,
        optimizer_model=fsdp_model2,
    )
    assert epoch == 3

    # Verify full model state matches reference
    loaded_params = get_model_state_dict(fsdp_model2, options=full_options)
    if dm.rank == 0:
        for key in ref_params:
            assert torch.allclose(
                ref_params[key], loaded_params[key], rtol=1e-5, atol=1e-5
            ), f"Parameter {key} differs after checkpoint round-trip"

    # Verify pos_embed is actually sharded (local shapes differ across domain ranks)
    inner = fsdp_model2.module  # unwrap FSDP
    local_pos_embed = inner.pos_embed
    assert isinstance(local_pos_embed, DTensor), (
        "pos_embed should be a DTensor after load"
    )

    local_shape = local_pos_embed.to_local().shape
    assert local_shape[1] == embed_tokens // domain_size, (
        f"Expected pos_embed local tokens={embed_tokens // domain_size}, got {local_shape[1]}"
    )

    # Verify forward pass matches
    with torch.no_grad():
        out1 = fsdp_model(x).full_tensor()
        out2 = fsdp_model2(x).full_tensor()
    assert torch.allclose(out1, out2, rtol=1e-5, atol=1e-5), (
        "Model outputs differ after 2-D mesh checkpoint round-trip"
    )


# ---------------------------------------------------------------------------
# load_model_weights — FSDP + ShardTensor on 2-D mesh
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
@pytest.mark.multigpu_static
@pytest.mark.skipif(not _HAS_TORCH_26, reason="ShardTensor requires torch >= 2.6")
@pytest.mark.parametrize("sync_module_states", [True, False])
@pytest.mark.parametrize("file_format", ["mdlus", "pt"])
@pytest.mark.parametrize("use_orig_params", [True, False])
def test_load_model_weights_fsdp_shard_tensor(
    shared_tmp_dir, use_orig_params, file_format, sync_module_states
):
    """load_model_weights loads a .mdlus or .pt file into an FSDP+ShardTensor model."""
    if use_orig_params:
        pytest.skip(
            "use_orig_params=True + ShardTensor under FSDP NO_SHARD is unsupported: "
            "FSDP writeback fails when local parameter shape changes"
        )
    torch.manual_seed(0)

    dm = DistributedManager()
    if dm.world_size < 4 or dm.world_size % 2 != 0:
        pytest.skip("Need at least 4 ranks (divisible by 2) for 2-D mesh test")

    device = dm.device
    domain_size = 2
    dp_size = dm.world_size // domain_size
    mesh = init_device_mesh(
        "cuda", (dp_size, domain_size), mesh_dim_names=("ddp", "domain")
    )

    embed_tokens = 24

    # Train a plain model to get non-trivial weights
    model = _PosEmbedModel(embed_tokens=embed_tokens, embed_dim=8, hidden=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x_full = torch.randn(2, embed_tokens, 8, device=device)
    for _ in range(3):
        model(x_full).sum().backward()
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        ref_output = model(x_full).clone()

    # Save trained weights (rank 0 only) using the requested format
    weights_file = f"{shared_tmp_dir}/trained_shard.{file_format}"
    if dm.rank == 0:
        if file_format == "mdlus":
            model.save(weights_file)
        else:
            torch.save(model.state_dict(), weights_file)

    dist.barrier()

    # Build a fresh distributed model and load from the single file
    def _build_distributed_model():
        m = _PosEmbedModel(embed_tokens=embed_tokens, embed_dim=8, hidden=16).to(device)
        m = distribute_module(
            m, device_mesh=mesh["domain"], partition_fn=_partition_pos_embed
        )
        m = FSDP(
            m,
            device_mesh=mesh["ddp"],
            sharding_strategy=ShardingStrategy.NO_SHARD,
            use_orig_params=use_orig_params,
            sync_module_states=sync_module_states,
        )
        return m

    fsdp_model2 = _build_distributed_model()
    load_model_weights(fsdp_model2, weights_file)

    # Verify full model state matches reference
    full_options = StateDictOptions(full_state_dict=True)
    loaded_params = get_model_state_dict(fsdp_model2, options=full_options)
    ref_params = model.state_dict()
    if dm.rank == 0:
        for key in ref_params:
            assert torch.allclose(
                ref_params[key].cpu(), loaded_params[key].cpu(), rtol=1e-5, atol=1e-5
            ), f"Parameter {key} differs after load_model_weights"

    # Verify pos_embed is still sharded
    inner = fsdp_model2.module
    assert isinstance(inner.pos_embed, DTensor), (
        "pos_embed should be a DTensor after load"
    )
    assert inner.pos_embed.to_local().shape[1] == embed_tokens // domain_size

    # Verify forward pass matches
    x_sharded = distribute_tensor(
        x_full, device_mesh=mesh["domain"], placements=[Shard(1)]
    )
    with torch.no_grad():
        loaded_output = fsdp_model2(x_sharded).full_tensor()
    assert torch.allclose(ref_output, loaded_output, rtol=1e-4, atol=1e-4), (
        "Model outputs differ after load_model_weights into 2-D mesh model"
    )


# ---------------------------------------------------------------------------
# Non-distributed models still work correctly in a multi-rank environment
# ---------------------------------------------------------------------------


@pytest.mark.timeout(15)
@pytest.mark.multigpu_static
def test_non_distributed_fallback(shared_tmp_dir):
    """Checkpoint utilities fall back to single-rank behaviour for non-FSDP models."""
    dm = DistributedManager()
    device = dm.device

    model = FullyConnected(
        in_features=8, out_features=8, num_layers=2, layer_size=16
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(2, 8, device=device)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        ref = model(x).clone()

    # Only rank 0 saves (non-distributed path)
    if dm.rank == 0:
        save_checkpoint(shared_tmp_dir, models=model, optimizer=optimizer, epoch=1)
    dist.barrier()

    # All ranks load independently (non-distributed path)
    model2 = FullyConnected(
        in_features=8, out_features=8, num_layers=2, layer_size=16
    ).to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    epoch = load_checkpoint(
        shared_tmp_dir, models=model2, optimizer=optimizer2, device=device
    )
    assert epoch == 1

    with torch.no_grad():
        loaded = model2(x)
    assert torch.allclose(ref, loaded, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Missing checkpoint directory — distributed path
# ---------------------------------------------------------------------------


@pytest.mark.timeout(15)
@pytest.mark.multigpu_static
def test_distributed_missing_directory_returns_zero(shared_tmp_dir):
    """load_checkpoint returns 0 for all ranks when checkpoint dir is missing."""
    dm = DistributedManager()
    if dm.world_size < 2:
        pytest.skip("Need at least 2 ranks")

    device = dm.device
    mesh = init_device_mesh("cuda", (dm.world_size,), mesh_dim_names=("world",))

    model = FullyConnected(
        in_features=8, out_features=8, num_layers=2, layer_size=16
    ).to(device)
    fsdp_model = FSDP(
        model, device_mesh=mesh["world"], sharding_strategy=ShardingStrategy.NO_SHARD
    )
    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=1e-3)

    epoch = load_checkpoint(
        shared_tmp_dir + "/nonexistent",
        models=fsdp_model,
        optimizer=optimizer,
        optimizer_model=fsdp_model,
    )
    assert epoch == 0


# ---------------------------------------------------------------------------
# Multiple FSDP-wrapped models in a single checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@pytest.mark.parametrize("sync_module_states", [True, False])
def test_fsdp_multiple_models_checkpoint(shared_tmp_dir, sync_module_states):
    """Checkpoint round-trip with two separate FSDP-wrapped models."""
    dm = DistributedManager()
    if dm.world_size < 2:
        pytest.skip("Need at least 2 ranks")

    device = dm.device
    mesh = init_device_mesh("cuda", (dm.world_size,), mesh_dim_names=("world",))

    model_a = FullyConnected(
        in_features=8, out_features=8, num_layers=2, layer_size=16
    ).to(device)
    model_b = FullyConnected(
        in_features=4, out_features=4, num_layers=2, layer_size=16
    ).to(device)
    fsdp_a = FSDP(
        model_a,
        device_mesh=mesh["world"],
        sharding_strategy=ShardingStrategy.NO_SHARD,
        sync_module_states=sync_module_states,
    )
    fsdp_b = FSDP(
        model_b,
        device_mesh=mesh["world"],
        sharding_strategy=ShardingStrategy.NO_SHARD,
        sync_module_states=sync_module_states,
    )

    x_a = torch.randn(2, 8, device=device)
    x_b = torch.randn(2, 4, device=device)

    opt_a = torch.optim.Adam(fsdp_a.parameters(), lr=1e-3)
    opt_b = torch.optim.Adam(fsdp_b.parameters(), lr=1e-3)
    for _ in range(3):
        fsdp_a(x_a).sum().backward()
        opt_a.step()
        opt_a.zero_grad()
        fsdp_b(x_b).sum().backward()
        opt_b.step()
        opt_b.zero_grad()

    with torch.no_grad():
        ref_a = fsdp_a(x_a).clone()
        ref_b = fsdp_b(x_b).clone()

    save_checkpoint(shared_tmp_dir, models=[fsdp_a, fsdp_b], epoch=1)
    dist.barrier()

    model_a2 = FullyConnected(
        in_features=8, out_features=8, num_layers=2, layer_size=16
    ).to(device)
    model_b2 = FullyConnected(
        in_features=4, out_features=4, num_layers=2, layer_size=16
    ).to(device)
    fsdp_a2 = FSDP(
        model_a2,
        device_mesh=mesh["world"],
        sharding_strategy=ShardingStrategy.NO_SHARD,
        sync_module_states=sync_module_states,
    )
    fsdp_b2 = FSDP(
        model_b2,
        device_mesh=mesh["world"],
        sharding_strategy=ShardingStrategy.NO_SHARD,
        sync_module_states=sync_module_states,
    )

    epoch = load_checkpoint(shared_tmp_dir, models=[fsdp_a2, fsdp_b2])
    assert epoch == 1

    with torch.no_grad():
        loaded_a = fsdp_a2(x_a)
        loaded_b = fsdp_b2(x_b)
    assert torch.allclose(ref_a, loaded_a, rtol=1e-5, atol=1e-5), (
        "Model A outputs differ after multi-model FSDP checkpoint round-trip"
    )
    assert torch.allclose(ref_b, loaded_b, rtol=1e-5, atol=1e-5), (
        "Model B outputs differ after multi-model FSDP checkpoint round-trip"
    )


# ---------------------------------------------------------------------------
# Plain nn.Module (not physicsnemo.Module) with FSDP
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@pytest.mark.parametrize("sync_module_states", [True, False])
@pytest.mark.parametrize("use_orig_params", [True, False])
def test_fsdp_pytorch_module_checkpoint_roundtrip(
    shared_tmp_dir, use_orig_params, sync_module_states
):
    """Checkpoint round-trip for a plain nn.Module (not physicsnemo.Module) under FSDP."""
    dm = DistributedManager()
    if dm.world_size < 2:
        pytest.skip("Need at least 2 ranks")

    device = dm.device
    mesh = init_device_mesh("cuda", (dm.world_size,), mesh_dim_names=("world",))

    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16)).to(device)
    fsdp_model = FSDP(
        model,
        device_mesh=mesh["world"],
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=use_orig_params,
        sync_module_states=sync_module_states,
    )
    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=1e-3)

    x = torch.randn(4, 16, device=device)
    for _ in range(3):
        fsdp_model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        ref_output = fsdp_model(x).clone()

    save_checkpoint(
        shared_tmp_dir,
        models=fsdp_model,
        optimizer=optimizer,
        epoch=2,
        optimizer_model=fsdp_model,
    )
    dist.barrier()

    model2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16)).to(device)
    fsdp_model2 = FSDP(
        model2,
        device_mesh=mesh["world"],
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=use_orig_params,
        sync_module_states=sync_module_states,
    )
    optimizer2 = torch.optim.Adam(fsdp_model2.parameters(), lr=1e-3)

    epoch = load_checkpoint(
        shared_tmp_dir,
        models=fsdp_model2,
        optimizer=optimizer2,
        optimizer_model=fsdp_model2,
    )
    assert epoch == 2

    with torch.no_grad():
        loaded_output = fsdp_model2(x)
    assert torch.allclose(ref_output, loaded_output, rtol=1e-5, atol=1e-5), (
        "Model outputs differ after nn.Module FSDP checkpoint round-trip"
    )


# ---------------------------------------------------------------------------
# GradScaler state preservation under FSDP
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@pytest.mark.parametrize("sync_module_states", [True, False])
def test_fsdp_grad_scaler_checkpoint(shared_tmp_dir, sync_module_states):
    """Checkpoint round-trip preserves GradScaler state under FSDP."""
    dm = DistributedManager()
    if dm.world_size < 2:
        pytest.skip("Need at least 2 ranks")

    device = dm.device
    mesh = init_device_mesh("cuda", (dm.world_size,), mesh_dim_names=("world",))

    model = FullyConnected(
        in_features=16, out_features=16, num_layers=2, layer_size=32
    ).to(device)
    fsdp_model = FSDP(
        model,
        device_mesh=mesh["world"],
        sharding_strategy=ShardingStrategy.NO_SHARD,
        sync_module_states=sync_module_states,
    )
    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")

    x = torch.randn(4, 16, device=device)
    for _ in range(3):
        with torch.amp.autocast("cuda"):
            loss = fsdp_model(x).sum()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    ref_scale = scaler.get_scale()

    save_checkpoint(
        shared_tmp_dir,
        models=fsdp_model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=3,
        optimizer_model=fsdp_model,
    )
    dist.barrier()

    model2 = FullyConnected(
        in_features=16, out_features=16, num_layers=2, layer_size=32
    ).to(device)
    fsdp_model2 = FSDP(
        model2,
        device_mesh=mesh["world"],
        sharding_strategy=ShardingStrategy.NO_SHARD,
        sync_module_states=sync_module_states,
    )
    optimizer2 = torch.optim.Adam(fsdp_model2.parameters(), lr=1e-3)
    scaler2 = torch.amp.GradScaler("cuda")

    epoch = load_checkpoint(
        shared_tmp_dir,
        models=fsdp_model2,
        optimizer=optimizer2,
        scaler=scaler2,
        optimizer_model=fsdp_model2,
    )
    assert epoch == 3
    assert scaler2.get_scale() == ref_scale

    with torch.no_grad():
        ref_output = fsdp_model(x)
        loaded_output = fsdp_model2(x)
    assert torch.allclose(ref_output, loaded_output, rtol=1e-5, atol=1e-5), (
        "Model outputs differ after FSDP+GradScaler checkpoint round-trip"
    )


# ---------------------------------------------------------------------------
# channels_last optimizer state survives a checkpoint round-trip
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
@pytest.mark.multigpu_static
@pytest.mark.parametrize("use_orig_params", [False, True])
@pytest.mark.parametrize(
    "memory_format",
    [
        pytest.param(torch.channels_last, id="channels_last"),
        pytest.param(torch.contiguous_format, id="contiguous"),
    ],
)
def test_fsdp_channels_last_optim_roundtrip(
    shared_tmp_dir, memory_format, use_orig_params
):
    """Optimizer state for channels_last Conv2d weights survives a checkpoint round-trip.

    With ``use_orig_params=False`` PyTorch FSDP packs the FlatParameter
    using ``as_strided((numel,), (1,))`` for non-truly-contiguous params
    (storage byte order) but unpacks loaded optimizer state with
    ``torch.flatten`` (logical NCHW order).  For channels_last Conv2d
    weights those orders differ, which silently corrupts ``exp_avg`` /
    ``exp_avg_sq`` after a save-load cycle.  ``checkpoint.py`` works
    around the asymmetry via ``_remap_channels_last_optim_sd``.

    Parametrized to also pin the negative cases:

    * ``contiguous_format``: same code path with no remap needed.
    * ``use_orig_params=True``: optim state goes per-original-param,
      not via the FlatParameter, so the asymmetry doesn't exist and the
      remap must *not* fire (firing would itself scramble the state).
    """
    torch.manual_seed(0)
    dm = DistributedManager()
    device = dm.device
    mesh = init_device_mesh("cuda", (dm.world_size,), mesh_dim_names=("world",))

    def _build_model():
        # Conv2d (4-D weight, exercises channels_last) followed by a Linear
        # (2-D weight) so we cover both contiguous and non-contiguous params
        # in the same FlatParameter.
        m = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(8 * 4 * 4, 8),
        ).to(device, memory_format=memory_format)
        return FSDP(
            m,
            device_mesh=mesh["world"],
            sharding_strategy=ShardingStrategy.NO_SHARD,
            use_orig_params=use_orig_params,
            sync_module_states=True,
        )

    fsdp_model = _build_model()
    optimizer = torch.optim.Adam(fsdp_model.parameters(), lr=1e-3)

    x = torch.randn(2, 3, 4, 4, device=device).to(memory_format=memory_format)
    for _ in range(3):
        loss = fsdp_model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    full_options = StateDictOptions(full_state_dict=True)
    ref_optim_sd = get_optimizer_state_dict(fsdp_model, optimizer, options=full_options)

    save_checkpoint(
        shared_tmp_dir,
        models=fsdp_model,
        optimizer=optimizer,
        epoch=3,
        optimizer_model=fsdp_model,
    )
    dist.barrier()

    fsdp_model2 = _build_model()
    optimizer2 = torch.optim.Adam(fsdp_model2.parameters(), lr=1e-3)
    epoch = load_checkpoint(
        shared_tmp_dir,
        models=fsdp_model2,
        optimizer=optimizer2,
        optimizer_model=fsdp_model2,
    )
    assert epoch == 3

    loaded_optim_sd = get_optimizer_state_dict(
        fsdp_model2, optimizer2, options=full_options
    )

    if dm.rank == 0:
        assert ref_optim_sd["state"].keys() == loaded_optim_sd["state"].keys()
        for pname, pstate in ref_optim_sd["state"].items():
            for k, ref_v in pstate.items():
                loaded_v = loaded_optim_sd["state"][pname][k]
                if not isinstance(ref_v, torch.Tensor):
                    assert ref_v == loaded_v, (
                        f"Optimizer state {pname}.{k} differs: {ref_v} vs {loaded_v}"
                    )
                    continue
                assert torch.equal(ref_v.cpu(), loaded_v.cpu()), (
                    f"Optimizer state {pname}.{k} differs after round-trip"
                )
