# Convolution

This examples shows how to use `ShardTensor` with convolution layers, including
the backwards pass.

We check numerical agreement between the outputs as well as the gradients.

Note: This example also shows a utility in PhysicsNeMo for setting up model weights
with auto-promotion with ShardTensor.  When using tools like `DDP`, ShardTensor
can assume any `torch.Tensor` input it gets is replicated across all devices.
Meaning, in plain words, that ShardTensor assumes these `torch.Tensors` are model
weights and identical across devices.

If you're familiar with `DDP`, you know that `DDP` will synchronize the weights
across `DDP`'s set of GPUs.  With `ShardTensor` + `DDP`, actually, this is 2D
parallelism and you'll have a whole axis of `DDP` groups to coordinate.

To help this, in PhysicsNeMo we provide `sync_module_over_mesh` which initializes
weights over a designated mesh.  We aren't using `DDP` here, explicitly (see
the next example), but this is the first time we are coordinating weights over
a mesh at startup.

Run the example:

```bash
torchrun --nproc-per-node 8 sharded_conv.py
```

We also see, by manually printing out the gradient of the input tensor, that the
activations are distributed.
