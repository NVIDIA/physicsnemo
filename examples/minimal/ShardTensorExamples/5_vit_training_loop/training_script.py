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
 
import torch

from model import HybridViT

# from baseline_model import HybridViT
from utils import parse_args, print_and_save_results, end_to_end_benchmark

from physicsnemo.distributed import DistributedManager

# Add DDP import
from torch.nn.parallel import DistributedDataParallel as DDP

# Imports for Domain Parallelism
from physicsnemo.distributed import DistributedManager, scatter_tensor
from torch.distributed.tensor import distribute_module, distribute_tensor

# FSDP instead of DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.placement_types import (  # noqa: E402
    Replicate,
    Shard,
)

def partition_model(name, submodule, device_mesh):
    
    for key, param in submodule._parameters.items():
        if "pos_embed" in key:
            # Replace the pos_embed with a scattered ShardTensor
            # Global source is the global rank of local rank 0:
            scattered_pos_emd = distribute_tensor(
                submodule.pos_embed,
                device_mesh=device_mesh,
                placements=[
                    Shard(1),
                ],
            )
            submodule.register_parameter(key, torch.nn.Parameter(scattered_pos_emd))



def main():
    """Main benchmarking script."""
    # Configuration
    
    args = parse_args()

    image_sizes = list(range(args.image_size_start, args.image_size_stop + 1, args.image_size_step))
    device = torch.device('cuda')
    
    # Generate image sizes based on start, stop, and step
    if args.dimension == 2:
        image_sizes = list(range(args.image_size_start, args.image_size_stop + 1, args.image_size_step))
    elif args.dimension == 3:
        image_sizes = list(range(args.image_size_start, min(args.image_size_stop + 1, 513), args.image_size_step))
    
    # Initialize distributed manager first
    DistributedManager.initialize()
    dm = DistributedManager()
    
    # Set device based on local rank
    device = dm.device
    torch.cuda.set_device(device)
    
    if args.domain_size > 1:
        # NEW FOR SHARDING:
        mesh = dm.initialize_mesh(
            mesh_shape=(args.ddp_size, args.domain_size,), # -1 works the same way as reshaping
            mesh_dim_names = ["ddp","domain"]
        )
        ddp_mesh = mesh["ddp"]
        domain_mesh = mesh["domain"]
    
    
    num_classes = 1000
    precision_mode = "FP16" if args.use_mixed_precision and torch.cuda.is_available() else "FP32"
    
    if dm.rank == 0:
        print(f"Device: {device}")
        print(f"Batch size: {args.batch_size}")
        print(f"Domain size: {args.domain_size}")
        print(f"DDP size: {args.ddp_size}")
        print(f"Number of classes: {num_classes}")
        print(f"Precision: {precision_mode}")
        print("-" * 80)
    
    results = []

    ddp_size = args.ddp_size
    domain_size = args.domain_size



    
    for img_size in image_sizes:
        if dm.rank == 0:
            print(f"\nTesting image size: {img_size}x{img_size}")
        
        if args.dimension == 2:
            full_img_size = (img_size, img_size)
        elif args.dimension == 3:
            full_img_size = (img_size, img_size, img_size)
        
        if args.batch_size // ddp_size == 0:
            raise ValueError(f"Batch size {args.batch_size} is not divisible by DDP size {ddp_size}")
        
        # Create synthetic data - scale the batch size down by DDP size.
        x = torch.randn(args.batch_size // ddp_size, 3, * full_img_size, device=device)
        target = torch.randint(0, num_classes, (args.batch_size // ddp_size,), device=device)
        
        
        # Domain Parallel NOTE: we're generating data once per GPU but only keeping the data once per domain.
        # In a real application, you'd do this properly - each GPU would read it's own shard of the data.
        
        if args.domain_size > 1:
            
            # When scattering the data, we need to know the global rank of the source
            # But by definition, we use the domain_rank == 0 as the source.  Convert:
            global_rank_of_source = torch.distributed.get_global_rank(domain_mesh.get_group(), 0)
    
            # Scatter the input data across the domain:
            x = scatter_tensor(
                x, 
                global_rank_of_source, 
                domain_mesh, 
                placements=(Shard(2),), # Shard along the 2nd dimension (B C **H** W) which is the Height
                global_shape = x.shape, # This will be inferred if not provided!
                dtype = x.dtype, # This will be inferred if not provided!
            )

            target = scatter_tensor(
                target, 
                global_rank_of_source, 
                domain_mesh, 
                placements=(Replicate(),),  # REPLICATE the target - gradients will still be scattered properly.
                global_shape = target.shape, # This will be inferred if not provided!
                dtype = target.dtype, # This will be inferred if not provided!
            )
         
        # Test base model
        model = HybridViT(img_size = full_img_size, in_channels=3, num_classes=num_classes)
        model = model.to(device)

        if args.ddp_size > 1 and args.domain_size == 1:
            # Wrap model with DDP
            model = DDP(model, device_ids=[dm.local_rank], output_device=dm.local_rank)
        if args.domain_size > 1:

            # This step syncs across the domain only
            model = distribute_module(
                model,
                device_mesh=domain_mesh,
                partition_fn = partition_model,
            )
            if args.ddp_size > 1:
                # This step goes in the other axis on the mesh: every rank "i" of
                # each domain will sync up here.
                model = FSDP(model, device_mesh=ddp_mesh, use_orig_params=False)
                
                
        results.append(end_to_end_benchmark(args, model, (x, target), full_img_size, device, num_classes))
    
        if dm.rank == 0:
            print(f"Completed image size: {img_size}x{img_size}")
    
    if dm.rank == 0:
        print_and_save_results(results, args, precision_mode, dm.world_size)


if __name__ == "__main__":
    main()
