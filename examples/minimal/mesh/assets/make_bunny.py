# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Make the Stanford bunny mesh for the tutorials."""

from pathlib import Path

import pyvista as pv
import torch

from physicsnemo.mesh.io.io_pyvista import from_pyvista
from physicsnemo.mesh.remeshing._remeshing import remesh

mesh = from_pyvista(pv.examples.download_bunny_coarse())
mesh = remesh(mesh.clean().subdivide(levels=3, filter="linear"), 400)
mesh = mesh.rotate(axis="x", angle=torch.pi / 2).rotate(axis="z", angle=torch.pi / 2)

torch.save(mesh, Path(__file__).parent / "bunny.pt")
