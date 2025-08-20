# Vector Addition

This example shows basic creation of `ShardTensors`.  You can also see how
distributed computation can acclerate basic operations.

Run the baseline example, which is not sharded:

```bash
python vector_add_baseline.py
```

Then, run the sharded example (using `torchrun` as a launcher, for example)

```bash
torchrun --nproc-per-node 8 vector_add_sharded.py
```

You should see an improved performance measurement for the sharded runs.
