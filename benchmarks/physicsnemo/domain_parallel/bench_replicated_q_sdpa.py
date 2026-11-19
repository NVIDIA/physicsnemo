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

"""Micro-benchmark: replicated-Q SDPA over sharded K/V (FLARE context parallel).

Usage::

    torchrun --nproc-per-node 2 bench_replicated_q_sdpa.py
    torchrun --nproc-per-node 1 bench_replicated_q_sdpa.py

Compares ``PHYSICSNEMO_DP_SDPA_BACKEND`` = efficient vs flash. With one rank it
times plain ``F.scaled_dot_product_attention`` (flash) as the reference.
Prints a markdown table on rank 0.
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.placement_types import Replicate, Shard

from physicsnemo.domain_parallel import ShardTensor

B, H, SQ, D = 1, 8, 64, 16
N_LOCALS = (25_000, 100_000, 200_000)
WARMUP, ITERS = 10, 50


def _time(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / ITERS


def main():
    r"""Time replicated-Q SDPA fwd+bwd per backend and print a table on rank 0."""
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    mesh = init_device_mesh("cuda", (world,))
    dt = torch.float16
    backends = ["flash"] if world == 1 else ["efficient", "flash"]
    rows = []
    for n_local in N_LOCALS:
        for be in backends:
            os.environ["PHYSICSNEMO_DP_SDPA_BACKEND"] = be
            torch.manual_seed(rank)
            q = torch.randn(B, H, SQ, D, device=dev, dtype=dt)
            dist.broadcast(q, 0)
            k = torch.randn(B, H, n_local, D, device=dev, dtype=dt)
            v = torch.randn(B, H, n_local, D, device=dev, dtype=dt)
            if world == 1:
                q, k, v = (t.requires_grad_() for t in (q, k, v))
            else:
                q = ShardTensor.from_local(q.requires_grad_(), mesh, (Replicate(),))
                k = ShardTensor.from_local(k.requires_grad_(), mesh, (Shard(2),))
                v = ShardTensor.from_local(v.requires_grad_(), mesh, (Shard(2),))

            def fwd():
                with torch.no_grad():
                    return F.scaled_dot_product_attention(q, k, v)

            def fwdbwd():
                out = F.scaled_dot_product_attention(q, k, v)
                out.sum().backward()

            f = _time(fwd)
            fb = _time(fwdbwd)
            rows.append((world, n_local, be if world > 1 else "flash (F.sdpa)", f, fb))
    if rank == 0:
        print("\n| ranks | N_local/rank | backend | fwd ms | fwd+bwd ms |")
        print("|---|---|---|---|---|")
        for w, n, be, f, fb in rows:
            print(f"| {w} | {n} | {be} | {f:.3f} | {fb:.3f} |")
        sys.stdout.flush()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
