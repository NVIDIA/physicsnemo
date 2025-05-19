import torch
import time

from physicsnemo.distributed import DistributedManager, scatter_tensor
from torch.distributed.tensor.placement_types import Shard

# Another really big tensor:
N = 1_000_000_000

DistributedManager.initialize()
dm = DistributedManager()

device = dm.device

a = torch.randn(N, device=device)
b = torch.randn(N, device=device)

def f(x, y):
    return x + y

# Get the baseline result
c_baseline = f(a,b)

mesh = dm.initialize_mesh([-1,], ["domain"])
placements = (Shard(0),)
a_sharded = scatter_tensor(a, 0, mesh, placements)
b_sharded = scatter_tensor(b, 0, mesh, placements)

c_sharded = f(a_sharded,b_sharded)

# Comparison requires that we coalesce the results:
c_sharded = c_sharded.full_tensor()

# Now, performance measurement:
# Warm up:
for i in range(5):
    c = f(a_sharded,b_sharded)

# Measure execution time
torch.cuda.synchronize()
start_time = time.time()
for i in range(10):
    c = f(a_sharded,b_sharded)
torch.cuda.synchronize()
end_time = time.time()
elapsed_time = end_time - start_time

if dm.rank == 0:
    print(f"Rank {dm.rank}, Tensor agreement? {torch.allclose(c_baseline, c_sharded)}")
    print(f"Execution time for 10 runs: {elapsed_time:.4f} seconds")
