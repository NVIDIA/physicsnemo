# Third-Party Notices

PhysicsNeMo is licensed under the Apache License 2.0 (see
[LICENSE.txt](LICENSE.txt)). Some files in this repository contain code copied
or adapted from third-party open-source projects. Those files keep the upstream
copyright and license notice in their file header, and that notice governs the
third-party portions. This file lists them in one place.

The upstream version or commit each file was taken from was not recorded when
the code was added, so it is not listed here.

## Code distributed in the `physicsnemo` package

### Transolver

- Upstream: <https://github.com/thuml/Transolver>
- Copyright (c) 2024 THUML @ Tsinghua University
- License: MIT
- Files:
  - `physicsnemo/models/transolver/__init__.py`
  - `physicsnemo/models/transolver/transolver.py`
  - `physicsnemo/nn/module/physics_attention.py`

### pix2pixHD and pytorch-CycleGAN-and-pix2pix

- Upstream: <https://github.com/NVIDIA/pix2pixHD> and
  <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix>
- Copyright (C) 2019 NVIDIA Corporation. Ting-Chun Wang, Ming-Yu Liu,
  Jun-Yan Zhu.
- Copyright (c) 2017, Jun-Yan Zhu and Taesung Park
- License: BSD (full text in the file headers)
- Files:
  - `physicsnemo/models/pix2pix/pix2pix.py`
  - `physicsnemo/models/pix2pix/pix2pixunet.py`

### BSMS-GNN

- Upstream: <https://github.com/Eydcao/BSMS-GNN>
- License: Apache-2.0 (full text in the file headers)
- Files:
  - `physicsnemo/nn/module/gnn_layers/bsms.py`
  - `physicsnemo/datapipes/gnn/bsms.py`

### Virtual Foundry Graphnet (VFGN)

- Copyright 2023 HP Development Company, L.P.
- License: Apache-2.0
- Files:
  - `physicsnemo/models/vfgn/__init__.py`
  - `physicsnemo/models/vfgn/graph_network_modules.py`

### GraphCast

- Upstream: <https://github.com/google-deepmind/graphcast>
- Copyright 2023 DeepMind Technologies Limited.
- License: Apache-2.0
- Files:
  - `physicsnemo/models/graphcast/utils/icosahedral_mesh.py`

### Solar zenith angle (climt)

- Upstream: climt, as named in the file headers
- Copyright (c) 2016, Rodrigo Caballero
- License: BSD-3-Clause (full text in the file headers)
- Files:
  - `physicsnemo/utils/zenith_angle.py`
  - `physicsnemo/datapipes/climate/utils/zenith_angle.py`
  - `test/utils/test_zenith_angle.py` (not in the package)

### a-PyTorch-Tutorial-to-Super-Resolution

- Upstream: <https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution>
- Copyright (c) 2020 Sagar Vinodababu
- License: MIT
- Files:
  - `physicsnemo/models/srrn/super_res_net.py`

### ConvGRU-pytorch

- Upstream: <https://github.com/happyjin/ConvGRU-pytorch>
- Copyright (c) 2019 Jin Li
- License: MIT
- Files:
  - `physicsnemo/models/dlwp_healpix/layers/healpix_blocks.py` (`ConvGRUBlock`)

### fairchem

- Upstream: <https://github.com/facebookresearch/fairchem>
- Copyright (c) Meta, Inc. and its affiliates.
- License: MIT (full text in
  [`physicsnemo/experimental/nn/symmetry/THIRD_PARTY.txt`](physicsnemo/experimental/nn/symmetry/THIRD_PARTY.txt))
- Files: the files under `physicsnemo/experimental/nn/symmetry/` and
  `test/experimental/nn/symmetry/` that carry the fairchem notice.

## Code in `examples/` only

These files are not part of the `physicsnemo` package.

### Transolver (examples)

- Upstream: <https://github.com/thuml/Transolver>
- Copyright (c) 2024 THUML @ Tsinghua University
- License: MIT
- Files:
  - `examples/cfd/darcy_transolver/utils/testloss.py`

### Virtual Foundry Graphnet (examples)

- Copyright 2023 HP Development Company, L.P.
- License: Apache-2.0
- Files: the files under `examples/additive_manufacturing/sintering_physics/`
  that carry the HP notice.

### Stanford University

- Copyright 2023 Stanford University
- License: Apache-2.0
- Files:
  - the files under `examples/healthcare/bloodflow_1d_mgn/` that carry the
    Stanford notice
  - `examples/cfd/stokes_mgn/raw_dataset/download_dataset.sh`
  - `examples/healthcare/brain_anomaly_detection/conf/config_FNO_launch.yaml`

### DeepMind Research

- Upstream: <https://github.com/google-deepmind/deepmind-research>
- Copyright 2020 Deepmind Technologies Limited.
- License: Apache-2.0
- Files:
  - `examples/cfd/lagrangian_mgn/raw_datasets/download_dataset.sh`

### torch-harmonics

- Copyright (c) 2022 The torch-harmonics Authors.
- License: BSD-3-Clause
- Files:
  - `examples/cfd/swe_distributed_gnn/shallow_water_pde_dataset.py`
  - `examples/cfd/swe_distributed_gnn/shallow_water_solver.py`
  - `examples/cfd/swe_distributed_gnn/train.py`

### Reservoir simulation utilities

- Copyright 2025 Tsubasa Onishi
- License: Apache-2.0
- Files: the files under `examples/reservoir_simulation/sim_utils/` that carry
  this notice.

## NVIDIA files under the MIT license

These NVIDIA-authored example files are released under the MIT license instead
of Apache-2.0. Each carries the MIT text in its header.

- `examples/cfd/darcy_transolver/config.yaml`
- `examples/cfd/darcy_transolver/config_fix.yaml`
- `examples/cfd/darcy_transolver/convert_mat_to_npz.py`
- `examples/cfd/external_aerodynamics/transformer_models/deprecated/datapipe.py`
- `examples/cfd/external_aerodynamics/transformer_models/src/compute_normalizations.py`

## Dependencies

PhysicsNeMo's runtime and optional dependencies are not included in this
repository or in its packages. They are installed separately and remain under
their own licenses. They are listed in [pyproject.toml](pyproject.toml), and
the resolved versions are in `uv.lock`.
