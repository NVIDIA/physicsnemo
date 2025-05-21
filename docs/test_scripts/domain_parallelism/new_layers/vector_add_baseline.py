import torch
import time

# Make a really big tensor:
N = 1_000_000_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a = torch.randn(N, device=device)
b = torch.randn(N, device=device)

def f(a, b):
    # This is a truly local operation: no communication is needed.
    return a + b

# run a couple times to warmup:
for i in range(5):
    c = f(a,b)

# Optional: Benchmark it if you like:

# Measure execution time
torch.cuda.synchronize()
start_time = time.time()
for i in range(10):
    c = f(a,b)
torch.cuda.synchronize()
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Execution time for 10 runs: {elapsed_time:.4f} seconds")