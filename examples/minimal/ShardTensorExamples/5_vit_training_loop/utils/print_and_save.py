# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

def print_and_save_results(results, args, precision_mode, world_size):
    
    # Prepare table data with units as first row
    headers = [
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
        "Train\n(samp/s/gpu)"
    ]
    table_data = []
    
    for result in results:
        if result['forward_time'] != float('inf'):
            # Successful run
            row = [
                result['image_size'],
                args.batch_size,  # Global batch size
                args.batch_size,  # Local batch size (same as global for single GPU)
                f"{result['params']}",
                f"{result['forward_time']:.5f}",
                f"{result['training_time']:.5f}",
                f"{result['inference_memory']:.3f}",
                f"{args.batch_size / result['forward_time']:.3f}",
                f"{args.batch_size / result['forward_time'] / world_size:.3f}",
                f"{result['training_memory']:.3f}",
                f"{args.batch_size / result['training_time']:.3f}",
                f"{args.batch_size / result['training_time'] / world_size:.3f}",
            ]
        else:
            # Out of memory
            row = [
                result['image_size'],
                args.batch_size,  # Global batch size
                args.batch_size,  # Local batch size (same as global for single GPU)
                f"{result['params']}",
                "OOM", "OOM", "OOM", "OOM", "OOM"
            ]
        table_data.append(row)
    
    
    # Print summary table
    print("\n" + "="*80)
    print(f"BENCHMARK SUMMARY - Hybrid ViT Base in {args.dimension}D ({precision_mode})")
    print("="*80)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Save the data to csv:

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Generate filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'results/benchmark_results_{args.batch_size}bs_{args.dimension}d_{precision_mode}_{args.domain_size}dp_{args.ddp_size}ddp.csv'
    
    # Write to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([h.replace('\n', ' ') for h in headers])
        writer.writerows(table_data)
    
    print(f"\nResults saved to {filename}")