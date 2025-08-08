# ViT Training

This example shows how to adapt a single-device or DDP training loop to use
domain parallelism, for both training and inference.

The data is synthetically generated image-like data in 2D or 3D.  The training
script can benchmark the model over a variety
of image sizes.

The model is a convolutional embedding followed by 15 layers of Transformer
blocks.

```python
HybridViT(
  (patch_embed): PatchEmbedding2d(
    (conv): Conv2d(3, 768, kernel_size=(8, 8), stride=(8, 8))
    (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (stages): ModuleList(
    (0-15): 16 x TransformerBlock(
      (norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (attn): MultiHeadAttention(
        (qkv): Linear(in_features=768, out_features=2304, bias=True)
        (proj): Linear(in_features=768, out_features=768, bias=True)
      )
      (norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): MLP(
        (fc1): Linear(in_features=768, out_features=3072, bias=True)
        (act): GELU(approximate='none')
        (fc2): Linear(in_features=3072, out_features=768, bias=True)
      )
    )
  )
  (head): Linear(in_features=768, out_features=1000, bias=True)
)
Number of parameters: 126907624
```

The script allows the user to control how large each parallelism axis is:

```bash
usage: training_script.py [-h] [--batch_size BATCH_SIZE] [--dimension {2,3}]
[--image_size_start IMAGE_SIZE_START] [--image_size_stop IMAGE_SIZE_STOP]
[--image_size_step IMAGE_SIZE_STEP]
[--ddp_size DDP_SIZE] [--domain_size DOMAIN_SIZE] [--use_mixed_precision]

Benchmark HybridViT model performance

options:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Global Batch size for training (default: 1)
  --dimension {2,3}     Dimension of the model: 2D or 3D (default: 2)
  --image_size_start IMAGE_SIZE_START
                        Starting image size (default: 256)
  --image_size_stop IMAGE_SIZE_STOP
                        Ending image size (default: 2048)
  --image_size_step IMAGE_SIZE_STEP
                        Step size for image size progression (default: 128)
  --ddp_size DDP_SIZE   DDP world size (default: 1)
  --domain_size DOMAIN_SIZE
                        Domain parallel size (default: 1)
  --use_mixed_precision
                        Enable mixed precision training (default: False)
```

The model code is identical in all use cases: only the input data changes, and
whether the model is wrapped in DDP or FSDP.  Output will include a
table of performance results.
