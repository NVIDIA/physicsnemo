import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import contextlib
import time
import numpy as np
from tabulate import tabulate
from baseline_model import HybridViT

from physicsnemo.distributed import DistributedManager, scatter_tensor
from torch.distributed.tensor import distribute_module, distribute_tensor

# FSDP instead of DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.placement_types import (  # noqa: E402
    Replicate,
    Shard,
)

def benchmark_model(model, x, target, optimizer, num_warmup=5, num_iterations=10, use_mixed_precision=False):
    """Benchmark forward pass and training step performance.
    
    Args:
        model: The model to benchmark
        x: Input tensor
        target: Target tensor for loss computation
        optimizer: Optimizer for training step
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        use_mixed_precision: Whether to use mixed precision training
        
    Returns:
        Tuple of (forward_time, training_time) in seconds
    """
    

    if use_mixed_precision:
        context = autocast("cuda")
    else:
        context = contextlib.nullcontext()

    # You would use a grad scalar to do stable mixed precision in real training!

    
    # Warmup runs
    for _ in range(num_warmup):
        with torch.no_grad():
            with context:
                _ = model(x)
        
        # Training warmup step
        optimizer.zero_grad()
        with context:
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
    
    
    # Benchmark forward pass
    torch.cuda.synchronize()
    forward_times = []
    
    for _ in range(num_iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            with context:
                _ = model(x)
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds
        forward_times.append(elapsed_time)
        
    
    avg_forward_time = np.mean(forward_times)
    
    # Benchmark training step
    torch.cuda.synchronize()
    training_times = []
    
    for _ in range(num_iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        optimizer.zero_grad()
        with context:
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        end_event.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert ms to seconds
        training_times.append(elapsed_time)
    
    
    avg_training_time = np.mean(training_times)
    
    return avg_forward_time, avg_training_time


def get_model_memory_usage(model, x, target=None, optimizer=None, mode='inference', use_mixed_precision=False):
    """Estimate model memory usage for inference or training.
    
    Args:
        model: The model to measure
        x: Input tensor
        target: Target tensor (required for training mode)
        optimizer: Optimizer (required for training mode)
        mode: 'inference' or 'training'
        use_mixed_precision: Whether to use mixed precision
        
    Returns:
        Peak memory usage in GB
    """
    
    
    if use_mixed_precision:
        context = autocast("cuda")
    else:
        context = contextlib.nullcontext()
        
    torch.cuda.reset_peak_memory_stats()
    
    if mode == 'inference':
        with torch.no_grad():
            with context:
                _ = model(x)
                
    elif mode == 'training':
        if target is None or optimizer is None:
            raise ValueError("target and optimizer must be provided for training mode")
        
        optimizer.zero_grad()
        
        with context:
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
    return torch.cuda.max_memory_allocated() / 1024**3  # GB

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
    batch_size = 2
    domain_size = 2
    num_classes = 1000
    dimension = 3
    if dimension == 2:
        image_sizes = [256, 320, 384, 448, 512, 576, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]  # Progressive image sizes
    elif dimension == 3:
        image_sizes = [32, 64, 128, 256]
        # image_sizes = [32, 64, 128, 256, 384, 512]
    use_mixed_precision = False  # Set to True to enable mixed precision training
    
    # Initialize distributed manager first
    DistributedManager.initialize()
    dm = DistributedManager()
    
    # Set device based on local rank
    device = dm.device
    torch.cuda.set_device(device)
    
    # Only print from rank 0 to avoid duplicate output
    if dm.rank == 0:
        print(f"Device: {device}")
        print(f"World size: {dm.world_size}")
        print(f"Batch size per GPU: {batch_size}")
        print(f"Total batch size: {batch_size * dm.world_size}")
        print(f"Number of classes: {num_classes}")
        print(f"Mixed precision: {use_mixed_precision and torch.cuda.is_available()}")
        print("-" * 80)
    
    results = []

    # NEW FOR SHARDING:
    mesh = dm.initialize_mesh(
        mesh_shape=(-1, domain_size,), mesh_dim_names = ["ddp","domain"]
    )
    ddp_mesh = mesh["ddp"]
    domain_mesh = mesh["domain"]
    
    
    # With 2D paralellism, we need to keep track of both global 
    # and local ranks and sizes.
    global_rank = dm.rank
    domain_rank = torch.distributed.get_rank(domain_mesh.get_group())
    global_size = dm.world_size
    
    # When scattering the data, we need to know the global rank of the source
    # But by definition, we use the domain_rank == 0 as the source.  Convert:
    global_rank_of_source = torch.distributed.get_global_rank(domain_mesh.get_group(), 0)
    
    
    for img_size in image_sizes:
        if dm.rank == 0:
            print(f"\nTesting image size: {img_size}x{img_size}")
        
        if dimension == 2:
            full_img_size = (img_size, img_size)
        elif dimension == 3:
            full_img_size = (img_size, img_size, img_size)
        
        ddp_size = ddp_mesh.size()
        
        # Create synthetic data (Batch size above is GLOBAL)
        
        # NOTE: we're doing this once per GPU but only keeping the data once per domain.
        # In a real application, you'd do this properly.
        x = torch.randn(batch_size // ddp_size, 3, * full_img_size, device=device)
        target = torch.randint(0, num_classes, (batch_size // ddp_size,), device=device)
        
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
            placements=(Replicate(),),
            global_shape = target.shape, # This will be inferred if not provided!
            dtype = target.dtype, # This will be inferred if not provided!
        )
        
        # Test base model
        if dm.rank == 0:
            print("  Initializing HybridViT...")
        model = HybridViT(img_size = full_img_size, in_channels=3, num_classes=num_classes, depth=1)
        model = model.to(device)
        
        # Wrap model with DDP
        # Distribute the model across the domain, then wrap in FSDP to scale out over the batch.
        model = distribute_module(
            model,
            device_mesh=domain_mesh,
            # partition_fn = partition_model,
        )
        model = FSDP(model, device_mesh=ddp_mesh, use_orig_params=False)
        
        if dm.rank == 0:
            print(f"  Model param device: {next(model.parameters()).device}")

        # Count parameters (only count on base model to avoid duplication)
        num_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        
        # Create optimizer
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
        
        try:
            # Benchmark model
            forward_time, training_time = benchmark_model(
                model, x, target, optimizer, use_mixed_precision=use_mixed_precision
            )
            
            # Memory usage - measure both inference and training
            inference_memory = get_model_memory_usage(
                model, x, mode='inference', use_mixed_precision=use_mixed_precision
            )
            training_memory = get_model_memory_usage(
                model, x, target, optimizer, mode='training', use_mixed_precision=use_mixed_precision
            )
            
            # Only print results from rank 0
            if dm.rank == 0:
                precision_str = " (FP16)" if use_mixed_precision and torch.cuda.is_available() else " (FP32)"
                print(f"    Parameters: {num_params:,}")
                print(f"    Forward time{precision_str}: {forward_time:.4f}s")
                print(f"    Training time{precision_str}: {training_time:.4f}s")
                print(f"    Inference memory{precision_str}: {inference_memory:.2f}GB")
                print(f"    Training memory{precision_str}: {training_memory:.2f}GB")
                print(f"    Throughput{precision_str}: {batch_size/training_time:.2f} samples/sec per GPU")
                print(f"    Total throughput{precision_str}: {(batch_size * dm.world_size)/training_time:.2f} samples/sec")
            
            # Store results (only on rank 0 to avoid duplication)
            if dm.rank == 0:
                results.append({
                    'image_size': img_size,
                    'params': num_params,
                    'forward_time': forward_time,
                    'training_time': training_time,
                    'inference_memory': inference_memory,
                    'training_memory': training_memory,
                    'throughput_per_gpu': batch_size/training_time,
                    'total_throughput': (batch_size * dm.world_size)/training_time,
                    'mixed_precision': use_mixed_precision and torch.cuda.is_available()
                })
            
        except RuntimeError as e:
            if dm.rank == 0:
                print(f"    Error: {e}")
                # Store failed result
                results.append({
                    'image_size': img_size,
                    'params': num_params,
                    'forward_time': float('inf'),
                    'training_time': float('inf'),
                    'inference_memory': float('inf'),
                    'training_memory': float('inf'),
                    'throughput_per_gpu': 0,
                    'total_throughput': 0,
                    'mixed_precision': use_mixed_precision and torch.cuda.is_available()
                })
    
        torch.cuda.synchronize()
        # Clear cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        del model, optimizer
    
    # Print summary table (only from rank 0)
    if dm.rank == 0:
        print("\n" + "="*80)
        precision_mode = "FP16" if use_mixed_precision and torch.cuda.is_available() else "FP32"
        print(f"BENCHMARK SUMMARY - Hybrid ViT Base in {dimension}D ({precision_mode}) - DDP with {dm.world_size} GPUs")
        print("="*80)
        
        # Prepare table data with units as first row
        headers = ["Size", "Params", "Forward", "Training", "Inf Mem", "Train Mem", "Per GPU", "Total"]
        units = ["(px)", "(M)", "(s)", "(s)", "(GB)", "(GB)", "(samp/s)", "(samp/s)"]
        
        table_data = [units]  # Start with units row
        
        for result in results:
            if result['forward_time'] != float('inf'):
                # Successful run
                row = [
                    result['image_size'],
                    f"{result['params'] / 1e6:.1f}",
                    f"{result['forward_time']:.4f}",
                    f"{result['training_time']:.4f}",
                    f"{result['inference_memory']:.2f}",
                    f"{result['training_memory']:.2f}",
                    f"{result['throughput_per_gpu']:.2f}",
                    f"{result['total_throughput']:.2f}"
                ]
            else:
                # Out of memory
                row = [
                    result['image_size'],
                    f"{result['params'] / 1e6:.1f}",
                    "OOM", "OOM", "OOM", "OOM", "OOM", "OOM"
                ]
            table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
