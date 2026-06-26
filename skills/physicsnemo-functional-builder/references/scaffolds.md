# Scaffolds — copy-paste skeletons

Adapt names/shapes; **verify the `FunctionSpec` API and `check_version_spec`
import path against the live repo** before trusting these verbatim. All files
start with the SPDX Apache-2.0 header. Replace `myop` / `MyOp` / `<category>`.

## Package layout

```
physicsnemo/nn/functional/<category>/myop/
  __init__.py  <op>.py  _torch_impl.py  utils.py
  kernels.py  _warp_impl.py            # if Warp backend
  _cuml_impl.py  _scipy_impl.py        # if those backends
```

## `utils.py` — shared validation

```python
from __future__ import annotations
import torch

def validate_inputs(points: torch.Tensor, num_samples: int) -> tuple[torch.Tensor, bool]:
    """Validate and canonicalize inputs (skipped under torch.compile)."""
    if not torch.compiler.is_compiling():
        if points.ndim not in (2, 3):
            raise ValueError(f"points must be rank 2 or 3, got {points.ndim}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
    was_unbatched = points.ndim == 2
    points_b = points.unsqueeze(0) if was_unbatched else points
    return points_b, was_unbatched
```

## `_torch_impl.py` — the baseline

```python
from __future__ import annotations
import torch
from jaxtyping import Float, Int
from .utils import validate_inputs

def myop_torch(
    points: Float[torch.Tensor, "*batch num_points dim"],
    num_samples: int,
    random_start: bool = False,
) -> Int[torch.Tensor, "*batch num_samples"]:
    points_b, was_unbatched = validate_inputs(points, num_samples)
    # ... pure-torch reference algorithm (device-agnostic) ...
    out = ...
    return out.squeeze(0) if was_unbatched else out
```

## `kernels.py` — pure Warp (no torch import)

```python
import warp as wp

@wp.kernel
def myop_kernel(
    points: wp.array3d(dtype=wp.float32),
    out: wp.array2d(dtype=wp.int32),
    num_points: wp.int32,
    num_samples: wp.int32,
    dim: wp.int32,
):
    b, t = wp.tid()
    ...  # the algorithm; wp.tile_max / wp.tile_min for block reductions
```

## `_warp_impl.py` — Warp wrapped in `custom_op`

```python
from __future__ import annotations
import torch
import warp as wp
from physicsnemo.core.function_spec import FunctionSpec
from .kernels import myop_kernel
from .utils import validate_inputs

wp.init()

@torch.library.custom_op("physicsnemo::myop_warp", mutates_args=())
def myop(points: torch.Tensor, num_samples: int, random_start: bool = False) -> torch.Tensor:
    points_b, was_unbatched = validate_inputs(points, num_samples)
    if points_b.device.type != "cuda":
        raise ValueError("The Warp myop backend requires CUDA tensors.")
    points_f = points_b.detach().to(torch.float32).contiguous()
    batch, num_points, dim = points_f.shape
    out = torch.empty((batch, num_samples), dtype=torch.int32, device=points_f.device)
    block = min(256, max(1, num_points))

    wp_device, wp_stream = FunctionSpec.warp_launch_context(points_f)
    with wp.ScopedStream(wp_stream):
        wp.launch(
            myop_kernel,
            dim=(batch, block), block_dim=block,
            inputs=[wp.from_torch(points_f, return_ctype=True),
                    wp.from_torch(out, return_ctype=True),
                    num_points, num_samples, dim],
            device=wp_device, stream=wp_stream,
        )
    out = out.to(torch.int64)
    return out.squeeze(0) if was_unbatched else out

@myop.register_fake
def _(points, num_samples, random_start=False):
    if points.ndim == 2:
        return torch.empty((num_samples,), dtype=torch.int64, device=points.device)
    return torch.empty((points.shape[0], num_samples), dtype=torch.int64, device=points.device)
```

## `_cuml_impl.py` / `_scipy_impl.py` — dep-gated

```python
from __future__ import annotations
import importlib
import torch
from physicsnemo.core.version_check import check_version_spec   # confirm path

SCIPY_AVAILABLE = check_version_spec("scipy", "1.7.0", hard_fail=False)

if SCIPY_AVAILABLE:
    KDTree = importlib.import_module("scipy.spatial").KDTree

    @torch.library.custom_op("physicsnemo::myop_scipy", mutates_args=())
    def myop(points: torch.Tensor, queries: torch.Tensor, k: int = 3
             ) -> tuple[torch.Tensor, torch.Tensor]:
        if points.device.type != "cpu":
            raise ValueError(f"`myop` scipy does not support CUDA, got {points.device=}")
        restore = points.dtype
        if restore == torch.bfloat16:
            points, queries = points.float(), queries.float()
        tree = KDTree(points.detach().numpy())
        distance, indices = tree.query(queries.detach().numpy(), k=k)
        indices = torch.from_numpy(indices).reshape(queries.shape[0], k)
        distance = torch.from_numpy(distance).reshape(queries.shape[0], k)
        return indices, distance.to(restore) if restore == torch.bfloat16 else distance

    @myop.register_fake
    def _(points, queries, k=3):
        return (torch.empty(queries.shape[0], k, device=queries.device, dtype=torch.int64),
                torch.empty(queries.shape[0], k, device=queries.device, dtype=queries.dtype))
else:
    def myop(*args, **kwargs):
        raise ImportError("physicsnemo myop: scipy is not installed.")
```

(cuML mirrors this: gate on `cuml`+`cupy`, check `device.type != "cuda"`, move via
`cp.from_dlpack` / `torch.from_dlpack`.)

## `<op>.py` — the `FunctionSpec`

```python
from __future__ import annotations
import torch
from jaxtyping import Float, Int
from physicsnemo.core.function_spec import FunctionSpec
from ._torch_impl import myop_torch
from ._warp_impl import myop as myop_warp        # if Warp

class MyOp(FunctionSpec):
    r"""One-line contract.

    Parameters
    ----------
    points : Float[torch.Tensor, "*batch num_points dim"]
        ...
    Returns
    -------
    Int[torch.Tensor, "*batch num_samples"]
        ...
    Raises
    ------
    ValueError
        If ...
    """

    @FunctionSpec.register(name="warp", required_imports=("warp>=0.6.0",), rank=0)
    def warp_forward(points, num_samples, random_start=False):
        return myop_warp(points, num_samples, random_start)

    @FunctionSpec.register(name="torch", rank=1, baseline=True)
    def torch_forward(points, num_samples, random_start=False):
        return myop_torch(points, num_samples, random_start)

    @classmethod
    def dispatch(cls, points, num_samples, random_start=False, implementation=None):
        impls = cls._get_impls()
        cls._check_impl(implementation, impls)
        if implementation is not None:
            impl = impls[implementation]
            if not impl.available:
                raise ImportError(f"Implementation '{implementation}' is not available.")
            return impl.func(points, num_samples, random_start)
        warp_impl = impls.get("warp")
        if points.is_cuda and warp_impl is not None and warp_impl.available:
            return warp_impl.func(points, num_samples, random_start)
        return impls["torch"].func(points, num_samples, random_start)

    @classmethod
    def make_inputs_forward(cls, device="cpu"):
        device = torch.device(device)
        for label, n, d, k in (("small", 256, 3, 16), ("large", 4096, 3, 256)):
            yield (label, (torch.rand(n, d, device=device), k), {})

    @classmethod
    def compare_forward(cls, output, reference):
        torch.testing.assert_close(output.sort(dim=-1).values, reference.sort(dim=-1).values)

myop = MyOp.make_function("myop")
```

## `__init__.py` (op package) + re-exports

```python
# physicsnemo/nn/functional/<category>/myop/__init__.py
from .myop import MyOp, myop
__all__ = ["MyOp", "myop"]
```

Then add to `<category>/__init__.py` and `physicsnemo/nn/functional/__init__.py`
(re-export `myop` + `MyOp`, extend `__all__`) — match the existing style there.

## Test module

```python
# test/nn/functional/<category>/test_myop.py
import pytest, torch
from physicsnemo.nn.functional import myop
from physicsnemo.nn.functional.<category>.myop.myop import MyOp
from physicsnemo.core.version_check import check_version_spec

@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_myop_known_answer(device, implementation):
    if implementation == "warp" and "cpu" in device:
        pytest.skip("warp backend is CUDA-only")
    points = ...  # deterministic input with a known result
    out = myop(points, 3, implementation=implementation)
    assert out.tolist() == [...]

def test_myop_backend_parity(device):
    if "cpu" in device:
        pytest.skip("warp backend is CUDA-only")
    points = ...  # tie-free, well-separated
    MyOp.compare_forward(
        myop(points, 40, implementation="warp"),
        myop(points, 40, implementation="torch"),
    )

def test_myop_opcheck(device):
    if "cpu" in device:
        pytest.skip("warp backend is CUDA-only")
    from physicsnemo.nn.functional.<category>.myop._warp_impl import myop as myop_warp_op
    torch.library.opcheck(myop_warp_op, args=(..., 8), kwargs={"random_start": False})
```
