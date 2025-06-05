import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import contextlib
import time
import numpy as np
from tabulate import tabulate
from baseline_model import HybridViT


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


def main():
    """Main benchmarking script."""
    # Configuration
    batch_size = 1
    num_classes = 1000
    dimension = 3
    if dimension == 2:
        image_sizes = [256, 320, 384, 448, 512, 576, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048]  # Progressive image sizes
    elif dimension == 3:
        image_sizes = [32, 64, 128, 256, 384, 512]
    use_mixed_precision = False  # Set to True to enable mixed precision training
    device = torch.device('cuda')
    
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Mixed precision: {use_mixed_precision and torch.cuda.is_available()}")
    print("-" * 80)
    
    results = []
    
    for img_size in image_sizes:
        print(f"\nTesting image size: {img_size}x{img_size}")
        
        if dimension == 2:
            full_img_size = (img_size, img_size)
        elif dimension == 3:
            full_img_size = (img_size, img_size, img_size)
        
        # Create synthetic data
        x = torch.randn(batch_size, 3, * full_img_size, device=device)
        target = torch.randint(0, num_classes, (batch_size,), device=device)
        
        # Test base model
        print("  Initializing HybridViT...")
        model = HybridViT(img_size = full_img_size, in_channels=3, num_classes=num_classes)
        model = model.to(device)
        print(f"  Model param device: {next(model.parameters()).device}")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
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
            
            precision_str = " (FP16)" if use_mixed_precision and torch.cuda.is_available() else " (FP32)"
            print(f"    Parameters: {num_params:,}")
            print(f"    Forward time{precision_str}: {forward_time:.4f}s")
            print(f"    Training time{precision_str}: {training_time:.4f}s")
            print(f"    Inference memory{precision_str}: {inference_memory:.2f}GB")
            print(f"    Training memory{precision_str}: {training_memory:.2f}GB")
            print(f"    Throughput{precision_str}: {batch_size/training_time:.2f} samples/sec")
            
            # Store results
            results.append({
                'image_size': img_size,
                'params': num_params,
                'forward_time': forward_time,
                'training_time': training_time,
                'inference_memory': inference_memory,
                'training_memory': training_memory,
                'throughput': batch_size/training_time,
                'mixed_precision': use_mixed_precision and torch.cuda.is_available()
            })
            
        except RuntimeError as e:
            print(f"    Error: {e}")
            # Store failed result
            results.append({
                'image_size': img_size,
                'params': num_params,
                'forward_time': float('inf'),
                'training_time': float('inf'),
                'inference_memory': float('inf'),
                'training_memory': float('inf'),
                'throughput': 0,
                'mixed_precision': use_mixed_precision and torch.cuda.is_available()
            })
        
        # Clear cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        del model, optimizer
    
    # Print summary table
    print("\n" + "="*80)
    precision_mode = "FP16" if use_mixed_precision and torch.cuda.is_available() else "FP32"
    print(f"BENCHMARK SUMMARY - Hybrid ViT Base in {dimension}D ({precision_mode})")
    print("="*80)
    
    # Prepare table data with units as first row
    headers = ["Size", "Params", "Forward", "Training", "Inf Mem", "Train Mem", "Throughput"]
    units = ["(px)", "(M)", "(s)", "(s)", "(GB)", "(GB)", "(samp/s)"]
    
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
                f"{result['throughput']:.2f}"
            ]
        else:
            # Out of memory
            row = [
                result['image_size'],
                f"{result['params'] / 1e6:.1f}",
                "OOM", "OOM", "OOM", "OOM", "OOM"
            ]
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
