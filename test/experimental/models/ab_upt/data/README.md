# AB-UPT regression fixtures

These fixtures were generated from this independent PhysicsNeMo implementation,
not the research repository. They check serialization and numerical regressions;
they do not establish agreement with research weights or benchmark accuracy.

The two configurations in `test_ab_upt.py` use seeds 1729 and 1730, respectively.
Each `.mdlus` contains the untrained model; `_output.pth` contains its forward
outputs for the standard model-test helper; `_methods.pth` contains inputs and
reference geometry encodings, anchor contexts, query outputs, and vorticity.
Regenerate from the repository root with:

```bash
python -m test.experimental.models.ab_upt.generate_data
```

The generator records distinct supernode/surface-anchor sets, 9 surface queries,
11 volume queries, and a vorticity step of 0.01. These fixtures were produced on
CPU with PyTorch 2.13.0+cu130 in float32.
