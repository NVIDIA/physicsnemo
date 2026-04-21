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

# Benchmark for scaled_dot_product_attention: single GPU vs ShardTensor (ring attention).
#
# Single GPU:
#   python benchmark_sharded_attention.py --seq_len 4096 --num_heads 16 --head_dim 64
#
# Distributed (ring attention via ShardTensor):
#   torchrun --nproc-per-node 8 benchmark_sharded_attention.py --seq_len 4096 --num_heads 16 --head_dim 64

import argparse
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.utils import Profiler


def parse_args():
    """Parse command-line arguments for the attention benchmark.

    Returns:
        argparse.Namespace: Parsed arguments including seq_len, num_heads,
            head_dim, batch_size, warmup/iteration counts, dtype, benchmark
            mode (inference or train), and an optional output file path.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark scaled_dot_product_attention: single GPU vs ShardTensor"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=4096,
        help="Total sequence length (same for Q, K, V)",
    )
    parser.add_argument(
        "--num_heads", type=int, default=16, help="Number of attention heads"
    )
    parser.add_argument(
        "--head_dim", type=int, default=64, help="Dimension per attention head"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num_warmup", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of timed benchmark iterations",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for Q, K, V tensors",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="inference",
        choices=["inference", "train"],
        help="Benchmark mode: 'inference' (forward only) or 'train' (forward + backward)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to write JSON results file. If not set, results are only printed.",
    )
    return parser.parse_args()


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def benchmark_attention(q, k, v, num_warmup, num_iterations, mode="inference"):
    """Benchmark scaled_dot_product_attention with CUDA event timing.

    Args:
        mode: "inference" for forward-only, "train" for forward + backward.

    Returns (times, mean_time_s, std_time_s) where times is the list of
    per-iteration durations in seconds.
    """
    is_train = mode == "train"

    # Warmup
    for _ in range(num_warmup):
        if is_train:
            out = F.scaled_dot_product_attention(q, k, v)
            loss = out.sum()
            loss.backward()
            q.grad = None
            k.grad = None
            v.grad = None
        else:
            with torch.no_grad():
                _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    # Profiler().enable("torch")
    # Profiler().initialize()

    with Profiler():
        # Timed iterations
        times = []
        for _ in range(num_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            if is_train:
                out = F.scaled_dot_product_attention(q, k, v)
                loss = out.sum()
                loss.backward()
                q.grad = None
                k.grad = None
                v.grad = None
            else:
                with torch.no_grad():
                    _ = F.scaled_dot_product_attention(q, k, v)
            end.record()

            torch.cuda.synchronize()
            # elapsed_time returns milliseconds
            times.append(start.elapsed_time(end) / 1000.0)
    # Profiler().finalize()

    return times, float(np.mean(times)), float(np.std(times))


def main():
    """Run the scaled_dot_product_attention benchmark.

    Sets up single-GPU or distributed (ring attention via ShardTensor)
    execution based on the runtime environment, allocates Q/K/V tensors,
    benchmarks attention in either inference or train mode, collects
    timing and memory statistics, and optionally writes results to JSON.
    """
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]

    DistributedManager.initialize()
    dm = DistributedManager()
    device = dm.device
    distributed = dm.distributed

    B = args.batch_size
    H = args.num_heads
    S = args.seq_len
    D = args.head_dim

    is_train = args.mode == "train"

    if distributed:
        world_size = dm.world_size
        assert S % world_size == 0, (
            f"seq_len ({S}) must be divisible by world_size ({world_size})"
        )
        local_S = S // world_size
        # The efficient attention kernel pads log_sumexp to a multiple of 32,
        # which causes a shape mismatch in ring attention if the local sequence
        # length is not already aligned.
        assert local_S % 32 == 0, (
            f"seq_len / world_size ({local_S}) must be a multiple of 32"
        )

        print(f"Local size is {local_S}")

        mesh = dm.initialize_mesh(mesh_shape=[-1], mesh_dim_names=["domain"])

        # Each rank generates its own local chunk of Q, K, V
        # Shape per rank: (B, H, local_S, D)
        q_local = torch.randn(
            B, H, local_S, D, device=device, dtype=dtype, requires_grad=is_train
        )
        k_local = torch.randn(
            B, H, local_S, D, device=device, dtype=dtype, requires_grad=is_train
        )
        v_local = torch.randn(
            B, H, local_S, D, device=device, dtype=dtype, requires_grad=is_train
        )

        # Shard(2) indicates sharding along the sequence dimension (axis 2 of B,H,S,D)
        placements = (Shard(2),)
        q = ShardTensor.from_local(q_local, mesh, placements)
        k = ShardTensor.from_local(k_local, mesh, placements)
        v = ShardTensor.from_local(v_local, mesh, placements)
    else:
        # Single GPU: full tensors
        q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=is_train)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=is_train)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=is_train)

    # --- Memory accounting: analytical input size ---
    element_size = torch.finfo(dtype).bits // 8
    local_seq = local_S if distributed else S
    # 3 tensors (Q, K, V) each of shape (B, H, local_seq, D)
    input_tensors_bytes = 3 * B * H * local_seq * D * element_size

    # Reset peak stats so they reflect only the benchmark, not tensor allocation
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    pre_benchmark_allocated = torch.cuda.memory_allocated(device)

    iter_times, mean_time, std_time = benchmark_attention(
        q, k, v, args.num_warmup, args.num_iterations, mode=args.mode
    )

    # Capture memory stats immediately after the benchmark
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    post_benchmark_allocated = torch.cuda.memory_allocated(device)
    allocated_delta = peak_allocated - pre_benchmark_allocated

    # In distributed mode, find the max peak across all ranks
    if distributed:
        peak_tensor = torch.tensor([peak_allocated], device=device, dtype=torch.long)
        torch.distributed.all_reduce(peak_tensor, op=torch.distributed.ReduceOp.MAX)
        max_peak_allocated_across_ranks = int(peak_tensor.item())
    else:
        max_peak_allocated_across_ranks = peak_allocated

    if dm.rank == 0:
        mode = (
            f"ShardTensor (ring attention, {dm.world_size} GPUs)"
            if distributed
            else "Single GPU"
        )

        results = {
            "timestamp": datetime.now().isoformat(),
            "parallelism": mode,
            "benchmark_mode": args.mode,
            "distributed": distributed,
            "world_size": dm.world_size,
            "config": {
                "seq_len": S,
                "num_heads": H,
                "head_dim": D,
                "batch_size": B,
                "dtype": args.dtype,
            },
            "benchmark": {
                "num_warmup": args.num_warmup,
                "num_iterations": args.num_iterations,
                "mean_time_s": mean_time,
                "std_time_s": std_time,
                "min_time_s": float(np.min(iter_times)),
                "max_time_s": float(np.max(iter_times)),
                "median_time_s": float(np.median(iter_times)),
                "per_iteration_times_s": iter_times,
            },
            "memory": {
                "input_tensors_bytes": input_tensors_bytes,
                "pre_benchmark_allocated_bytes": pre_benchmark_allocated,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "post_benchmark_allocated_bytes": post_benchmark_allocated,
                "allocated_delta_bytes": allocated_delta,
                "max_peak_allocated_across_ranks_bytes": max_peak_allocated_across_ranks,
            },
        }

        print(f"Parallelism:     {mode}")
        print(f"Benchmark mode:  {args.mode}")
        print(f"Sequence length: {S}")
        print(f"Num heads:       {H}")
        print(f"Head dim:        {D}")
        print(f"Batch size:      {B}")
        print(f"Dtype:           {args.dtype}")
        print(f"Num warmup:      {args.num_warmup}")
        print(f"Num iterations:  {args.num_iterations}")
        print(f"Mean time:       {mean_time:.6f} s")
        print(f"Std time:        {std_time:.6f} s")
        print(f"Min time:        {results['benchmark']['min_time_s']:.6f} s")
        print(f"Max time:        {results['benchmark']['max_time_s']:.6f} s")
        print(f"Median time:     {results['benchmark']['median_time_s']:.6f} s")

        mb = 1024 * 1024
        print(f"--- Memory ---")
        print(f"Input tensors (Q+K+V, per rank): {input_tensors_bytes / mb:.2f} MB")
        print(f"Pre-benchmark allocated: {pre_benchmark_allocated / mb:.2f} MB")
        print(f"Peak allocated:          {peak_allocated / mb:.2f} MB")
        print(f"Peak reserved:           {peak_reserved / mb:.2f} MB")
        print(f"Post-benchmark allocated:{post_benchmark_allocated / mb:.2f} MB")
        print(f"Allocated delta (peak - pre): {allocated_delta / mb:.2f} MB")
        if distributed:
            print(
                f"Max peak allocated (across {dm.world_size} ranks): {max_peak_allocated_across_ranks / mb:.2f} MB"
            )

        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
