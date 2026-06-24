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

"""Domain-parallel tests for the DiT 2D-RoPE + NaN-mask-token features.

These exercise the exact distribution configuration used by the StormCast
recipe: a 2-D ``(ddp, domain)`` device mesh where the model is sharded along the
height (``Shard(0)``) of its spatial buffers via ``distribute_module`` on the
domain mesh and then ``fully_shard`` (FSDP2) on the ddp mesh, with the spatial
input sharded along H on the domain mesh.

The key property under test is the "shard-along-height" design: building the
RoPE cos/sin tables at the global spatial size and sharding them along height
gives each rank globally-correct rows with no explicit ``h_offset`` arithmetic,
so the distributed forward is numerically equivalent to a single-GPU forward on
the full input (NATTEN halo exchange handles the window boundaries). The
invalid-region mask is supplied dynamically to ``forward`` as a height-sharded
``ShardTensor`` (like the spatial input) and pooled to token granularity inside
the model, so it lines up with the sequence-sharded tokens with no buffer.

Run with, e.g.::

    torchrun --nproc-per-node 4 -m pytest --multigpu-static \
        test/domain_parallel/models/test_dit.py -x
"""

import pytest
import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import distribute_module, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import scatter_tensor
from physicsnemo.models.dit import DiT
from physicsnemo.nn.module.rope import build_axial_rope_cos_sin_2d
from test.conftest import requires_module

# Latent grid is 4x4; with a domain mesh of size 2 each rank owns 2 rows.
INPUT_SIZE = (16, 16)
PATCH_SIZE = 4
HIDDEN_SIZE = 64
NUM_HEADS = 4  # head_dim = 16 (divisible by 4, required for axial 2D RoPE)
DEPTH = 2
ATTN_KERNEL = 3
IN_CHANNELS = 4


def _shard_dim_for(name: str) -> int | None:
    """Mirror of StormCast's ``shard_dim_selector`` for the buffers under test."""
    if any(s in name for s in ("pos_embed", "pos_embd", "spatial_emb")):
        return 1
    if any(s in name for s in ("rope_cos", "rope_sin")):
        return 0
    return None


def _partition_dit(name, submodule, device_mesh):
    """Local copy of StormCast's ``partition_model_selective``.

    Shards the spatial buffers along height and replicates everything else,
    handling both parameters and buffers explicitly.
    """
    for key, param in submodule._parameters.items():
        if param is None:
            continue
        shard_dim = _shard_dim_for(key)
        placement = Shard(shard_dim) if shard_dim is not None else Replicate()
        dt = distribute_tensor(param, device_mesh=device_mesh, placements=[placement])
        submodule.register_parameter(
            key, nn.Parameter(dt, requires_grad=param.requires_grad)
        )
    for key, buffer in submodule._buffers.items():
        if buffer is None:
            continue
        shard_dim = _shard_dim_for(key)
        placement = Shard(shard_dim) if shard_dim is not None else Replicate()
        dt = distribute_tensor(buffer, device_mesh=device_mesh, placements=[placement])
        persistent = key not in submodule._non_persistent_buffers_set
        submodule.register_buffer(key, dt, persistent=persistent)


def _build_dit(device, attention_backend: str, use_nan_mask_tokens: bool) -> DiT:
    """Construct a small NATTEN DiT with identical init across ranks (seeded).

    ``attention_backend`` selects the vanilla ``"natten2d"`` or the RoPE
    ``"natten2d_rope"`` variant; the RoPE variant has the DiT inject ``latent_hw``
    into the attention constructor automatically.
    """
    torch.manual_seed(0)
    model = DiT(
        input_size=INPUT_SIZE,
        in_channels=IN_CHANNELS,
        out_channels=IN_CHANNELS,
        patch_size=PATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        attention_backend=attention_backend,
        attn_kwargs={"attn_kernel": ATTN_KERNEL},
        use_nan_mask_tokens=use_nan_mask_tokens,
    )
    return model.to(device).eval()


def _make_pixel_mask(B, device) -> torch.Tensor:
    """A per-sample invalid pixel mask of shape ``(B, 1, H, W)``.

    The invalid region (whole top-left patch + one interior pixel) is shared
    across the batch here purely for a simple reference; the forward path treats
    it as a fully dynamic, potentially per-sample mask.
    """
    pixel_mask = torch.zeros(B, 1, *INPUT_SIZE, dtype=torch.bool, device=device)
    pixel_mask[:, :, 0:PATCH_SIZE, 0:PATCH_SIZE] = True  # -> patch (0, 0)
    pixel_mask[:, :, 9, 9] = True  # -> patch (2, 2)
    return pixel_mask


@requires_module(["natten"])
@pytest.mark.multigpu_static
@pytest.mark.parametrize("attention_backend", ["natten2d", "natten2d_rope"])
@pytest.mark.parametrize("use_nan_mask_tokens", [False, True])
def test_dit_natten_distributed_matches_single_gpu(
    distributed_mesh_2d, attention_backend, use_nan_mask_tokens
):
    """Distributed (ddp x domain, FSDP2) forward equals the single-GPU forward.

    Covers both the vanilla ``natten2d`` and the RoPE ``natten2d_rope`` backends
    so that any failure shared by both (e.g. in the QKV projection on a
    sequence-sharded ``ShardTensor``) is distinguishable from a RoPE-specific
    one. For ``natten2d_rope`` it also validates that sharding the RoPE tables
    along height delivers globally-correct rows per rank with no ``h_offset``;
    with ``use_nan_mask_tokens`` it validates the mask-token splice is
    sharding-safe.
    """
    dm = DistributedManager()
    if dm.world_size != 4:
        pytest.skip("Requires exactly 4 ranks for the (ddp=2, domain=2) mesh")

    ddp_mesh = distributed_mesh_2d["axis1"]
    domain_mesh = distributed_mesh_2d["axis2"]

    _run_dit_distributed_check(
        dm, ddp_mesh, domain_mesh, dm.device, attention_backend, use_nan_mask_tokens
    )


def _run_dit_distributed_check(
    dm, ddp_mesh, domain_mesh, device, attention_backend, use_nan_mask_tokens
):
    # Local (per-rank) batch size 1: this matches every domain-parallel config
    # the StormCast recipe runs. A 3-D F.linear on a sequence-sharded ShardTensor
    # folds the leading dims; with local batch 1 the size-1 batch dim is dropped
    # (no flatten of the sharded sequence dim), whereas local batch >= 2 would
    # flatten (B, N) with N sharded and hit DTensor's strict_view restriction --
    # a base ShardTensor/F.linear limitation independent of RoPE / NaN-masking.
    B = 1
    # Identical inputs on every rank so the gathered output can be compared to a
    # single reference computed from the full input.
    torch.manual_seed(123)
    x_full = torch.randn(B, IN_CHANNELS, *INPUT_SIZE, device=device)
    t = torch.rand(B, device=device)

    model = _build_dit(device, attention_backend, use_nan_mask_tokens)
    # Per-sample invalid mask, supplied dynamically to forward (no model buffer).
    pixel_mask_full = _make_pixel_mask(B, device) if use_nan_mask_tokens else None
    if use_nan_mask_tokens:
        # Give the learned mask tokens a non-trivial value so masking actually
        # changes the result (init is zero); seeded for cross-rank consistency.
        torch.manual_seed(7)
        for m in model.modules():
            if getattr(m, "mask_token", None) is not None:
                nn.init.normal_(m.mask_token)

    # --- single-GPU reference on the full input (before distribution) ---
    with torch.no_grad():
        ref_out = model(x_full, t, invalid_mask=pixel_mask_full).detach().clone()

    # --- distribute exactly as the StormCast recipe does ---
    model = distribute_module(
        model, device_mesh=domain_mesh, partition_fn=_partition_dit
    )
    with torch.no_grad():
        for p in model.parameters():
            if not p.is_contiguous():
                p.data = p.data.contiguous()
    fully_shard(model, mesh=ddp_mesh)

    # ------------------------------------------------------------------
    # Run ALL collective operations first, BEFORE any assertion. A failing
    # assert on a subset of ranks would otherwise leave the others blocked on
    # the next collective (manifesting as a SIGTERM hang), so every collective
    # below is executed unconditionally and symmetrically on all ranks.
    # ------------------------------------------------------------------
    # Source must be a rank within the domain submesh: use this group's local
    # rank-0 global rank (rank 0 for the {0,1} group, rank 2 for {2,3}). x_full
    # is identical on every rank, so both groups receive the same shards.
    domain_src = torch.distributed.get_global_rank(domain_mesh.get_group(), 0)
    x_sharded = scatter_tensor(
        x_full, domain_src, domain_mesh, (Shard(2),), requires_grad=False
    )
    # The invalid mask is a forward input, sharded along height exactly like x,
    # so the pooled token mask lines up with the sequence-sharded tokens.
    mask_sharded = (
        scatter_tensor(
            pixel_mask_full, domain_src, domain_mesh, (Shard(2),), requires_grad=False
        )
        if use_nan_mask_tokens
        else None
    )
    # Scalars/conditions must be replicated DTensors on the domain mesh so they
    # compose with the now-DTensor model buffers (e.g. the timestep embedder's
    # `freqs`); this mirrors StormCast's ParallelHelper.replicate_tensor.
    t_dt = distribute_tensor(t, domain_mesh, [Replicate()])
    with torch.no_grad():
        out = model(x_sharded, t_dt, invalid_mask=mask_sharded)
    gathered = out.full_tensor()  # collective: gather H-shards over domain mesh

    # Reassemble the sharded RoPE table globally (collectives), so correctness of
    # the per-rank row bands can be checked symmetrically.
    rope_cos_global = None
    rope_cos_local_shape = None
    for m in model.modules():
        if hasattr(m, "rope_cos") and hasattr(m.rope_cos, "full_tensor"):
            rope_cos_local_shape = tuple(m.rope_cos.to_local().shape)
            rope_cos_global = m.rope_cos.full_tensor()
            break

    # ------------------------------------------------------------------
    # Assertions (no further collectives): these evaluate identically on every
    # rank, so a failure fails the whole job cleanly rather than hanging.
    # ------------------------------------------------------------------
    latent_h = INPUT_SIZE[0] // PATCH_SIZE
    latent_w = INPUT_SIZE[1] // PATCH_SIZE
    head_dim = HIDDEN_SIZE // NUM_HEADS
    h_local = latent_h // domain_mesh.size()

    assert out.shape == (B, IN_CHANNELS, *INPUT_SIZE)
    # Output stays sharded along H on the domain mesh.
    assert any(p.is_shard() for p in out._spec.placements)

    # Distributed forward must match the single-GPU reference: this is the
    # end-to-end proof that the sharded RoPE rows and the dynamically-supplied,
    # height-sharded invalid mask are globally correct.
    torch.testing.assert_close(gathered, ref_out, atol=1e-4, rtol=1e-4)

    # For the RoPE backend, the table is sharded (each rank owns h_local rows)
    # and reassembles to the canonical global table.
    if attention_backend == "natten2d_rope":
        assert rope_cos_global is not None, "no RoPE attention module found"
        assert rope_cos_local_shape == (h_local, latent_w, head_dim)
        full_cos, _ = build_axial_rope_cos_sin_2d(latent_h, latent_w, head_dim)
        torch.testing.assert_close(
            rope_cos_global, full_cos.to(device), atol=1e-5, rtol=1e-5
        )
    else:
        assert rope_cos_global is None
