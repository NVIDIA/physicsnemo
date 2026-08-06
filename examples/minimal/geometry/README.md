# PhysicsNeMo Geometry Examples

The standalone `physicsnemo.geometry` adapters accept
`physicsnemo.mesh.Mesh` and, where documented, `DomainMesh` objects. The
examples in this directory use `Mesh`. Tensor-only kernels remain available
from `physicsnemo.nn.functional`.

From the repository root, install PhysicsNeMo and run the example:

```bash
pip install -e .
python examples/minimal/geometry/deformation_energy_optimization.py
```

## Differentiable Deformation Energy Optimization

Run `deformation_energy_optimization.py` for a compact shape-optimization
example. It preserves a prescribed radial-basis handle displacement while
penalizing strain, total-area change, and element inversion. The script uses
Warp on CUDA when available and falls back to Torch on CPU.
