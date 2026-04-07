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

import csv
import os

from tabulate import tabulate


HEADERS = [
    "Size\n(px)",
    "Global\nBS",
    "Local\nBS",
    "Params\n",
    "Fwd\n(s)",
    "Train\n(s)",
    "Inf.\nMem (GB)",
    "Inf.\n(samp/s)",
    "Inf.\n(samp/s/gpu)",
    "Train\nMem (GB)",
    "Train\n(samp/s)",
    "Train\n(samp/s/gpu)",
]


def _result_to_row(result, args, world_size):
    """Convert a single result dict to a table row."""
    if result["forward_time"] != float("inf"):
        # Training fields are None when running in inference-only mode
        training_time = result["training_time"]
        training_memory = result["training_memory"]

        if training_time is not None:
            train_time_str = f"{training_time:.5f}"
            train_mem_str = f"{training_memory:.3f}"
            train_throughput_str = f"{args.batch_size / training_time:.3f}"
            train_throughput_per_gpu_str = (
                f"{args.batch_size / training_time / world_size:.3f}"
            )
        else:
            train_time_str = "N/A"
            train_mem_str = "N/A"
            train_throughput_str = "N/A"
            train_throughput_per_gpu_str = "N/A"

        return [
            result["image_size"],
            args.batch_size,
            args.batch_size,
            f"{result['params']}",
            f"{result['forward_time']:.5f}",
            train_time_str,
            f"{result['inference_memory']:.3f}",
            f"{args.batch_size / result['forward_time']:.3f}",
            f"{args.batch_size / result['forward_time'] / world_size:.3f}",
            train_mem_str,
            train_throughput_str,
            train_throughput_per_gpu_str,
        ]
    else:
        return [
            result["image_size"],
            args.batch_size,
            args.batch_size,
            f"{result['params']}",
            "OOM",
            "OOM",
            "OOM",
            "OOM",
            "OOM",
        ]


def get_csv_filename(args, precision_mode):
    """Generate the CSV filename for this benchmark run."""
    os.makedirs("results", exist_ok=True)
    return (
        f"results/benchmark_results_{args.batch_size}bs_{args.dimension}d"
        f"_{precision_mode}_{args.domain_size}dp_{args.ddp_size}ddp"
        f"_{args.image_size_start}-{args.image_size_stop}px.csv"
    )


def save_result_incremental(filename, result, args, world_size):
    """Append a single result row to the CSV, writing headers if the file is new."""
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    row = _result_to_row(result, args, world_size)
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([h.replace("\n", " ") for h in HEADERS])
        writer.writerow(row)


def print_and_save_results(results, args, precision_mode, world_size):
    """Print a formatted benchmark summary table and save the results to a CSV file.

    Args:
        results: List of benchmark result dicts to format and persist.
        args: Parsed CLI arguments (used for titles, filenames, and row formatting).
        precision_mode: Label for the precision config (e.g. "fp16", "bf16").
        world_size: Total number of distributed ranks.
    """
    table_data = [_result_to_row(r, args, world_size) for r in results]

    # Print summary table
    print("\n" + "=" * 80)
    print(
        f"BENCHMARK SUMMARY - Hybrid ViT Base in {args.dimension}D ({precision_mode})"
    )
    print("=" * 80)
    print(tabulate(table_data, headers=HEADERS, tablefmt="grid"))

    filename = get_csv_filename(args, precision_mode)

    # Write full CSV (overwrites any incremental file with the complete results)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([h.replace("\n", " ") for h in HEADERS])
        writer.writerows(table_data)

    print(f"\nResults saved to {filename}")
