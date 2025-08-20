# `ShardTensor` Examples

This repository contains several examples and tutorials that showcase usage of
PhysicsNeMo's `ShardTensor` utility.

> These examples are the ones from the online tutorials.  You can run them
> here, but for a more detailed explanation of what's happening, see the
> [PhysicsNeMo documentation](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/tutorials/domain_parallelism_entry_point.html)

The contents of the repository are:

1. Vector Addition - See how to use `ShardTensor` for basic domain parallelism,
in an operation that requires no collectives.

2. Vector Dot Product - See how to extend an operation with a collective
reduction to compute a doct product over distributed tensors.

3. kNN - Parallelize a more complicated and challenging operation with a ring
passing scheme.

4. Convolution - See how to apply a loss function and backward pass for domain
parallel operations, and validate numerical accuracy and gradient placements.

5. ViT - Learn how to implement a fully training loop with domain parallelism,
and benchmark computational speed and memory usage.  Shows the differences in
the training script for a single-GPU, 1D (DDP) and 2D (ShardTensor + FSDP)
parallelism.

## Resources

Learn more about the tools used in these examples:

- [PyTorch Distributed API and Guide](https://docs.pytorch.org/docs/stable/distributed.html)
- [DTensor](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
- [PhysicsNeMo Domain Parallelism Guide](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/tutorials/domain_parallelism_entry_point.html)
- [ShardTensor API](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/api/physicsnemo.distributed.shardtensor.html)
