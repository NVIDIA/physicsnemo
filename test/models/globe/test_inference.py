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

import pytest
import torch

from physicsnemo.experimental.models.globe.model import GLOBE
from physicsnemo.mesh.primitives.procedural import lumpy_sphere

# Number of prediction points to evaluate at
N_PREDICTION_POINTS = 5


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_globe_inference(device: str) -> None:
    """Instantiate `GLOBE` and run inference on a lumpy-sphere boundary mesh."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ### Create model
    model = GLOBE(
        n_spatial_dims=3,
        output_field_ranks={"pressure": 0, "velocity": 1},
        boundary_source_data_ranks={"no_slip": {}},
        reference_length_names=["test_length"],
        reference_area=1.0,
        hidden_layer_sizes=[8],
    ).to(device)
    model.eval()

    ### Create a nontrivial boundary mesh (lumpy sphere, 1 subdivision -> 80 triangles)
    mesh = lumpy_sphere.load(subdivisions=1, device=device)

    ### Prediction points scattered near the surface
    generator = torch.Generator(device=device).manual_seed(0)
    prediction_points = torch.randn(
        N_PREDICTION_POINTS, 3, generator=generator, device=device
    )
    reference_lengths = {
        "test_length": torch.tensor(1.0, dtype=torch.float32, device=device)
    }

    ### Run inference
    with torch.no_grad():
        output_mesh = model(
            prediction_points=prediction_points,
            boundary_meshes={"no_slip": mesh},
            reference_lengths=reference_lengths,
            chunk_size=None,
        )

    ### Validate Mesh structure
    from physicsnemo.mesh import Mesh

    assert isinstance(output_mesh, Mesh)
    assert output_mesh.points.shape == (N_PREDICTION_POINTS, 3)

    ### Validate output fields and shapes
    fields = output_mesh.point_data
    assert set(fields.keys()) == {"pressure", "velocity"}
    assert fields["pressure"].shape == (N_PREDICTION_POINTS,)
    assert fields["velocity"].shape == (N_PREDICTION_POINTS, 3)
    assert fields["pressure"].device.type == device
    assert fields["velocity"].device.type == device

    ### Validate outputs are finite (no NaN or Inf from the forward pass)
    assert torch.all(torch.isfinite(fields["pressure"]))
    assert torch.all(torch.isfinite(fields["velocity"]))
