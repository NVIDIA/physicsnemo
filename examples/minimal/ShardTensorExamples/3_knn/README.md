# KNN

This examples shows how to parallelize non-trivial operations, such as a kNN.

The first example, the brute-foce kNN, will use a single GPU:

```bash
python knn_brute_force_baseline.py
```

The second example uses sharded tensors, leveraging DTensor automatic fallback
paths in a sub-optimal way:

```bash
torchrun --nproc-per-node 8 knn_brute_force_sharded.py
```

However, the automatic path for parallelization is not the most efficient.
See how to implement a better parallel operation manually with the last script.

```bash
torchrun --nproc-per-node 8 knn_brute_force_ring_sharded.py
```
