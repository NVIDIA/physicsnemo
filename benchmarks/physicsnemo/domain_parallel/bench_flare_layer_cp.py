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

"""Benchmark a single FLARE attention layer under domain parallelism.

Run per world size k (writes JSON rows), then build the table::

    torchrun --standalone --nproc-per-node K bench_flare_layer_cp.py \
        --sizes 200000 800000 --out res_k.json
    python bench_flare_layer_cp.py --report res_1.json res_2.json ...

k=1 uses a plain tensor (baseline). Run k=1 with sizes N/k to get the ideal
per-GPU work. Timing: fwd+bwd, 10 warmup + 50 iters, CUDA events, fp16 autocast.
"""

import argparse
import json
import sys

import torch
import torch.distributed as dist

WARMUP, ITERS = 10, 50
HIDDEN, HEADS, LATENTS = 128, 8, 64


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


def run(sizes, out):
    r"""Time eager and compiled fwd+bwd of one FLARE layer per size.

    Each rank holds ``N / k`` tokens (``Shard(1)``) for ``k > 1``. Writes the
    rows (``n``, ``k``, ``mode``, rank-0 and max-over-ranks ms, ``None`` on
    failure / OOM) to ``out`` as JSON on rank 0.
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.placement_types import Shard

    from physicsnemo.distributed import DistributedManager
    from physicsnemo.domain_parallel import scatter_tensor
    from physicsnemo.nn.module.flare_attention import FLARE

    DistributedManager.initialize()
    rank, k = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    mesh = init_device_mesh("cuda", (k,)) if k > 1 else None
    rows = []
    for n in sizes:
        for mode in ("eager", "compiled"):
            torch.manual_seed(0)
            layer = FLARE(
                HIDDEN, heads=HEADS, dim_head=HIDDEN // HEADS, n_global_queries=LATENTS
            ).to(dev)
            for p in layer.parameters():
                dist.broadcast(p.data, 0)
            mod = torch.compile(layer) if mode == "compiled" else layer
            ms = None
            try:
                x = torch.randn(1, n, HIDDEN, device=dev, dtype=torch.float16)
                if k > 1:
                    x = scatter_tensor(x, 0, mesh, (Shard(1),))
                x.requires_grad_(True)

                def step(mod=mod, x=x):
                    with torch.autocast("cuda", dtype=torch.float16):
                        y = mod(x)
                    y.float().square().mean().backward()

                ms = _time(step)
            except torch.cuda.OutOfMemoryError:
                ms = None
            except Exception as ex:  # record, keep sweeping
                print(f"[rank {rank}] N={n} {mode} failed: {ex!r}", file=sys.stderr)
                ms = None
            t = torch.tensor([ms if ms is not None else -1.0], device=dev)
            allt = [torch.zeros_like(t) for _ in range(k)]
            dist.all_gather(allt, t)
            allt = [a.item() for a in allt]
            ok = all(a > 0 for a in allt)
            rows.append(
                dict(
                    n=n,
                    k=k,
                    mode=mode,
                    r0=allt[0] if ok else None,
                    mx=max(allt) if ok else None,
                )
            )
            x = mod = layer = None
            torch.cuda.empty_cache()
            if rank == 0:
                print(rows[-1], flush=True)
    if rank == 0:
        json.dump(rows, open(out, "w"))
    dist.destroy_process_group()


def report(files):
    r"""Print markdown scaling tables from the JSON files written by :func:`run`."""
    rows = [r for f in files for r in json.load(open(f))]
    get = lambda n, k, m: next(  # noqa: E731
        (r for r in rows if r["n"] == n and r["k"] == k and r["mode"] == m), None
    )
    fmt = lambda v: "OOM" if v is None else f"{v:.2f}"  # noqa: E731
    for mode in ("eager", "compiled"):
        print(f"\n### {mode}\n")
        print(
            "| N_global | k | tokens/GPU | ms r0 | ms max | speedup t1/tk | "
            "eff | ideal 1-GPU ms @N/k | CP overhead (max/ideal) |"
        )
        print("|---|---|---|---|---|---|---|---|---|")
        for n in sorted({r["n"] for r in rows if r["k"] > 1 or True}):
            base = get(n, 1, mode)
            for k in sorted({r["k"] for r in rows}):
                r = get(n, k, mode)
                if r is None:
                    continue
                ideal = get(n // k, 1, mode) if k > 1 else None
                sp = ef = ov = None
                if r["mx"] and base and base["r0"]:
                    sp = base["r0"] / r["mx"]
                    ef = sp / k
                if ideal and ideal["r0"] and r["mx"]:
                    ov = r["mx"] / ideal["r0"]
                f2 = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
                print(
                    f"| {n} | {k} | {n // k} | {fmt(r['r0'])} | {fmt(r['mx'])} | "
                    f"{f2(sp)} | {f2(ef)} | {fmt(ideal['r0']) if ideal else '-'} "
                    f"| {f2(ov)} |"
                )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sizes", type=int, nargs="+", default=[200_000, 800_000, 1_600_000, 6_400_000]
    )
    ap.add_argument("--out", default="flare_layer_cp.json")
    ap.add_argument("--report", nargs="+")
    a = ap.parse_args()
    if a.report:
        report(a.report)
    else:
        run(a.sizes, a.out)
