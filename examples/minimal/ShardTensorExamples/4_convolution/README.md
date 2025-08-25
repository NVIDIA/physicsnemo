# Convolution

This examples shows how to use `ShardTensor` with convolution layers, including
the backwards pass.

We check numerical agreement between the outputs as well as the gradients.

Run the example:

```bash
torchrun --nproc-per-node 8 sharded_conv.py
```

We also see, by manually printing out the gradient of the input tensor, that the
activations are distributed.
