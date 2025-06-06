import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark HybridViT model performance')
    
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Global Batch size for training (default: 1)')
    parser.add_argument('--dimension', type=int, default=2, choices=[2, 3],
                       help='Dimension of the model: 2D or 3D (default: 2)')
    parser.add_argument('--image_size_start', type=int, default=256,
                       help='Starting image size (default: 256)')
    parser.add_argument('--image_size_stop', type=int, default=512,
                       help='Ending image size (default: 2048)')
    parser.add_argument('--image_size_step', type=int, default=128,
                       help='Step size for image size progression (default: 128)')
    parser.add_argument('--ddp_size', type=int, default=1,
                       help='DDP world size (default: 1)')
    parser.add_argument('--domain_size', type=int, default=1,
                       help='Domain parallel size (default: 1)')
    
    parser.add_argument('--use_mixed_precision', action='store_true',
                       help='Enable mixed precision training (default: False)')

    
    args = parser.parse_args()

    return args