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
from tensordict import TensorDict

from physicsnemo.experimental.models.globe.model import GLOBE
from physicsnemo.mesh.primitives.procedural import lumpy_sphere

# Number of prediction points to evaluate at
N_PREDICTION_POINTS = 5


def test_communication_meshes_have_independent_cache_containers(monkeypatch) -> None:
    """Communication updates preserve caches without aliasing their containers."""
    model = GLOBE(
        n_spatial_dims=3,
        output_field_ranks={"pressure": 0},
        boundary_source_data_ranks={"no_slip": {}},
        reference_length_names=["test_length"],
        reference_area=1.0,
        hidden_layer_sizes=[8],
    )

    mesh = lumpy_sphere.load(subdivisions=0)
    mesh = mesh.with_data(
        point_data={"discarded": torch.ones(mesh.n_points)},
        cell_data={
            "physical": TensorDict(
                {"input": torch.ones(mesh.n_cells)},
                batch_size=[mesh.n_cells],
            )
        },
        global_data={"discarded": torch.tensor(1.0)},
    )
    cached_centroids = mesh.cell_centroids

    def _fake_evaluate_hyperlayer(**kwargs) -> TensorDict:
        target_points = kwargs["target_points"]
        return TensorDict(
            {"latent": torch.ones(target_points.shape[0])},
            batch_size=[target_points.shape[0]],
        )

    monkeypatch.setattr(model, "_evaluate_hyperlayer", _fake_evaluate_hyperlayer)
    result = model._evaluate_communication_hyperlayer(
        layer_idx=0,
        boundary_meshes={"no_slip": mesh},
        reference_lengths={},
        global_data=None,
        cluster_trees={"no_slip": None},
        comm_plans={"no_slip": {}},
        source_areas={},
    )["no_slip"]

    # The historical reconstruction intentionally retained only cell data.
    assert not list(result.point_data.keys())
    assert not list(result.global_data.keys())
    assert set(result.cell_data.keys()) == {"physical", "latent"}

    # Geometry caches remain reusable, while structural mutations on the
    # communication result cannot leak back into its input mesh.
    assert result._cache is not mesh._cache
    assert result._cache["cell"] is not mesh._cache["cell"]
    assert result._cache["cell", "centroids"] is cached_centroids
    result._cache["cell", "probe"] = torch.ones(mesh.n_cells)
    assert "probe" not in mesh._cache["cell"]


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
        )

    ### Validate Mesh structure
    from physicsnemo.mesh import Mesh

    assert isinstance(output_mesh, Mesh[0, 3])
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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_globe_inference_multi_bc(device: str) -> None:
    """Inference with two BC types exercises cross-BC interaction plans."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ### Create model with two BC types
    model = GLOBE(
        n_spatial_dims=3,
        output_field_ranks={"pressure": 0, "velocity": 1},
        boundary_source_data_ranks={"no_slip": {}, "freestream": {}},
        reference_length_names=["test_length"],
        reference_area=1.0,
        hidden_layer_sizes=[8],
    ).to(device)
    model.eval()

    ### Use meshes with different face counts to stress-test cross-BC
    ### interaction plans (same face count would mask index-range bugs).
    mesh_no_slip = lumpy_sphere.load(subdivisions=1, device=device)  # 80 faces
    mesh_freestream = lumpy_sphere.load(subdivisions=0, device=device)  # 20 faces
    assert mesh_no_slip.n_cells != mesh_freestream.n_cells

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
            boundary_meshes={
                "no_slip": mesh_no_slip,
                "freestream": mesh_freestream,
            },
            reference_lengths=reference_lengths,
        )

    ### Validate structure and outputs
    fields = output_mesh.point_data
    assert set(fields.keys()) == {"pressure", "velocity"}
    assert fields["pressure"].shape == (N_PREDICTION_POINTS,)
    assert fields["velocity"].shape == (N_PREDICTION_POINTS, 3)
    assert torch.all(torch.isfinite(fields["pressure"]))
    assert torch.all(torch.isfinite(fields["velocity"]))
