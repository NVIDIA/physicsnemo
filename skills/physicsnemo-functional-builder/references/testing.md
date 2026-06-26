# Testing — cross-backend equivalence

Functionals have **no checkpoint round-trip** to test. The job instead is: every
backend produces the right answer, agrees with the torch baseline, and skips
cleanly when its device/dependency is absent. Tests mirror the source path:
`test/nn/functional/<category>/test_<op_name>.py`.

## The four test kinds

### 1. Known-answer (backend-independent truth)

Construct an input whose result is analytically known, and assert it per backend.

```python
@pytest.mark.parametrize("implementation", ["torch", "warp"])
def test_fps_known_answer_collinear(device, implementation):
    if implementation == "warp" and "cpu" in device:
        pytest.skip("warp FPS backend is CUDA-only")
    m = 9
    xs = torch.arange(m, dtype=torch.float32, device=device).reshape(m, 1)
    points = torch.cat([xs, torch.zeros(m, 2, device=device)], dim=1)
    idx = farthest_point_sampling(points, 3, implementation=implementation)
    assert idx.tolist() == [0, m - 1, (m - 1) // 2]
```

### 2. Output-contract checks

Shape, dtype, value ranges, and ordering invariants — run on the torch backend
across dtypes/`k`.

```python
def _assert_knn_outputs(points, queries, indices, distances, k):
    assert indices.shape == (queries.shape[0], k)
    assert (indices >= 0).all() and (indices < points.shape[0]).all()
    assert (distances >= 0).all()
    assert torch.all(distances[:, 1:] >= distances[:, :-1])   # sorted
```

### 3. Backend parity (the important one) — use `compare_forward`

Run the accelerated backend and the torch baseline on the same input and compare
via the op's `compare_forward` hook, which encodes **tie-invariance**:

```python
def test_knn_backend_forward_parity(device):
    points = torch.randn(53, 3, device=device)
    queries = torch.randn(21, 3, device=device)
    k = 5
    if "cuda" in device:
        if not check_version_spec("cuml", "26.2.0", hard_fail=False):
            pytest.skip("cuml not available")
        out_a = knn(points, queries, k, implementation="cuml")
    else:
        if not check_version_spec("scipy", "1.7.0", hard_fail=False):
            pytest.skip("scipy not available")
        out_a = knn(points, queries, k, implementation="scipy")
    out_b = knn(points, queries, k, implementation="torch")
    KNN.compare_forward(out_a, out_b)
```

> **The classic trap:** equal-distance neighbors are ordered differently across
> backends, so **never compare neighbor indices directly** — compare **sorted
> distances**. That logic belongs in `compare_forward` so every parity test
> inherits it:
> ```python
> @classmethod
> def compare_forward(cls, output, reference):
>     _, distances = output
>     _, ref_distances = reference
>     torch.testing.assert_close(
>         torch.sort(distances, dim=1)[0], torch.sort(ref_distances, dim=1)[0],
>         atol=1e-5, rtol=1e-5)
> ```
> For sampling ops (FPS), sort the selected indices before comparing
> (set-equality, order-invariant).

### 4. `opcheck` for each `custom_op` backend

Validates the custom-op schema, `register_fake`, and autograd plumbing:

```python
def test_fps_opcheck(device):
    if "cpu" in device:
        pytest.skip("warp FPS backend is CUDA-only")
    from physicsnemo.nn.functional.geometry.farthest_point_sampling._warp_impl import (
        farthest_point_sampling as fps_warp_op,
    )
    points = _well_separated_cloud(device, n=40)
    torch.library.opcheck(fps_warp_op, args=(points, 8), kwargs={"random_start": False})
```

## Skips — device and dependency

- **Device:** Warp and cuML are CUDA-only → `if "cpu" in device: pytest.skip(...)`.
  SciPy is CPU-only → skip on CUDA.
- **Optional dep:** `if not check_version_spec(pkg, ver, hard_fail=False): pytest.skip(...)`.
- Use the repo's `device` fixture (it parametrizes/serves cpu+cuda and auto-skips
  CUDA when unavailable) rather than hard-coding a device.

## Determinism

Build inputs deterministically (fixed grids, seeded `torch.rand`, well-separated
clouds so there are no ties) — parity tests must not be flaky. Tie-free inputs
also let you compare indices directly when you want a stricter check.

## Run

```
pytest test/nn/functional/<category>/test_<op_name>.py -q
# CPU-only host: Warp/cuML tests self-skip; SciPy/torch still exercise parity.
```
