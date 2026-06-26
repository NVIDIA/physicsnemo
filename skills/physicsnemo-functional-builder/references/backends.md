# Backends — torch reference, Warp kernels, cuML & SciPy

Every op ships a **torch reference** (the `baseline`). Accelerated backends are
optional and **must be wrapped in `torch.library.custom_op`** so they compose
with autograd and `torch.compile`. Study `farthest_point_sampling/` (Warp+torch)
and `neighbors/knn/` (cuML+SciPy+torch) before writing.

## 1. torch reference (`_torch_impl.py`) — always present

- Pure PyTorch, device-agnostic (runs on CPU and CUDA).
- The correctness oracle every other backend is tested against, and the
  `baseline=True` benchmark reference.
- No `custom_op` wrapper needed (it's already plain torch).
- Prioritize clarity over speed — this is the spec, not the fast path.

## 2. Warp backend (`kernels.py` + `_warp_impl.py`)

**`kernels.py` — pure Warp, no torch import:**

```python
import warp as wp

@wp.kernel
def fps_fused(
    points: wp.array3d(dtype=wp.float32),   # (B, N, D)
    selected: wp.array2d(dtype=wp.int32),   # (B, K) output
    num_points: wp.int32,
    num_samples: wp.int32,
    dim: wp.int32,
):
    b, t = wp.tid()
    ...  # the contributor's algorithm; use wp.tile_* for block reductions
```

**`_warp_impl.py` — wrap the launch in a `custom_op`:**

```python
import torch
import warp as wp
from physicsnemo.core.function_spec import FunctionSpec
from .kernels import fps_fused
from .utils import validate_inputs

wp.init()  # at module load

@torch.library.custom_op("physicsnemo::farthest_point_sampling_warp", mutates_args=())
def farthest_point_sampling(points: torch.Tensor, num_samples: int,
                            random_start: bool = False) -> torch.Tensor:
    points_b, was_unbatched = validate_inputs(points, num_samples)
    if points_b.device.type != "cuda":
        raise ValueError("The Warp farthest_point_sampling backend requires CUDA tensors.")
    points_f = points_b.detach().to(torch.float32).contiguous()
    selected = torch.empty((batch, num_samples), dtype=torch.int32, device=points_f.device)

    wp_device, wp_stream = FunctionSpec.warp_launch_context(points_f)   # device/stream
    with wp.ScopedStream(wp_stream):
        wp.launch(
            fps_fused,
            dim=(batch, block_size),
            block_dim=block_size,
            inputs=[
                wp.from_torch(points_f, return_ctype=True),            # zero-copy
                wp.from_torch(selected, return_ctype=True),
                num_points, num_samples, point_dim,
            ],
            device=wp_device, stream=wp_stream,
        )
    selected = selected.to(torch.int64)
    return selected.squeeze(0) if was_unbatched else selected

@farthest_point_sampling.register_fake
def _(points, num_samples, random_start=False):
    # shape inference for torch.compile / opcheck — no compute
    if points.ndim == 2:
        return torch.empty((num_samples,), dtype=torch.int64, device=points.device)
    return torch.empty((points.shape[0], num_samples), dtype=torch.int64, device=points.device)
```

Warp essentials:
- `wp.init()` once at module load.
- `wp.from_torch(t, return_ctype=True)` — zero-copy torch→warp; tensors must be
  `.contiguous()` and a Warp-supported dtype (cast bf16→fp32 first).
- `FunctionSpec.warp_launch_context(tensor)` → `(device, stream)`; launch under
  `wp.ScopedStream(stream)` so it respects the active CUDA stream.
- Warp kernels here are **CUDA-only** — raise a clear `ValueError` on CPU tensors
  (the torch baseline covers CPU via `dispatch`).
- `@<op>.register_fake` is **mandatory** — without it `torch.compile`/`opcheck`
  can't infer output shapes.

## 3. cuML / SciPy backends (`_cuml_impl.py` / `_scipy_impl.py`) — dep-gated

Never `import cuml` at top level. Gate the whole impl on availability and
register a clear-error stub otherwise:

```python
import importlib
import torch
from physicsnemo.core.version_check import check_version_spec   # confirm the exact path

CUML_AVAILABLE = check_version_spec("cuml", "26.2.0", hard_fail=False)
CUPY_AVAILABLE = check_version_spec("cupy", "13.6.0", hard_fail=False)

if CUML_AVAILABLE and CUPY_AVAILABLE:
    cuml = importlib.import_module("cuml")
    cp = importlib.import_module("cupy")

    @torch.library.custom_op("physicsnemo::knn_cuml", mutates_args=())
    def knn_impl(points: torch.Tensor, queries: torch.Tensor, k: int = 3
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        if points.device.type != "cuda":                 # device check INSIDE the impl
            raise ValueError(f"`knn` cuml does not support CPU, got {points.device=}")
        restore = points.dtype
        if restore == torch.bfloat16:                    # cuFFT/cuML want fp32
            points, queries = points.float(), queries.float()
        points = cp.from_dlpack(points)                  # zero-copy via DLPack
        queries = cp.from_dlpack(queries)
        nn = cuml.neighbors.NearestNeighbors(n_neighbors=k); nn.fit(points)
        distance, indices = nn.kneighbors(queries)
        indices = torch.from_dlpack(indices)
        distance = torch.from_dlpack(distance)
        if restore == torch.bfloat16:
            distance = distance.to(restore)
        return indices, distance

    @knn_impl.register_fake
    def _(points, queries, k=3):
        return (torch.empty(queries.shape[0], k, device=queries.device, dtype=torch.int64),
                torch.empty(queries.shape[0], k, device=queries.device, dtype=queries.dtype))
else:
    def knn_impl(*args, **kwargs):                        # stub: clear ImportError
        raise ImportError("physicsnemo kNN: cuml/cupy not installed.")
```

SciPy mirrors this for **CPU**: gate on `check_version_spec("scipy", ...)`, check
`points.device.type != "cpu"` inside, move data via `.detach().numpy()`, build
the structure (e.g. `scipy.spatial.KDTree`), and convert results back with
`torch.from_numpy(...)`. Cast bf16→fp32 the same way.

Optional-dep rules:
- `check_version_spec(pkg, ver, hard_fail=False)` is the availability gate
  (verify the import path in the live repo).
- Convert zero-copy where possible: **DLPack** for GPU (cuML/cuPy), numpy for CPU
  (SciPy).
- Keep the **hard device check inside the impl** — `dispatch` chooses by device,
  but a directly-called backend must reject the wrong device itself.
- Match the registered `required_imports` versions to what the impl actually gates
  on.

## Backend ↔ registration mapping

The `_*_impl.py` functions are what the `@FunctionSpec.register(name=...)` methods
in `<op_name>.py` call. Keep the registered `name`, `required_imports`, and `rank`
consistent with the impl they wrap (e.g. `name="cuml"`,
`required_imports=("cuml>=26.2.0","cupy>=13.6.0")`, `rank=0`).
