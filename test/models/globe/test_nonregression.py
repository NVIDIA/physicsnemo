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

"""Non-regression tests for the GLOBE model.

Locks in numerical output of the full GLOBE forward pass against saved
reference data.  Any change to the indexing, gather/scatter, or feature
engineering pipeline that alters output values (even slightly) will be
caught here.

Reference data is generated on first run and saved to
``test/models/globe/data/``.  Subsequent runs compare against the saved
reference.  If an intentional change alters outputs, delete the
``.pth`` file and re-run to regenerate.
"""

from pathlib import Path

import pytest
import torch

from physicsnemo.experimental.models.globe.model import GLOBE
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.procedural import lumpy_sphere

DATA_DIR = Path(__file__).parent / "data"
SEED = 42
ATOL = 1e-5
RTOL = 1e-5


def _make_globe_and_inputs(
    device: torch.device,
) -> tuple[GLOBE, dict]:
    """Construct a GLOBE model and inputs matching the DrivAerML config.

    Uses small mesh sizes for fast test execution while exercising all
    interaction phases (near-field, far-field, cross-BC communication).

    Returns the model and a dict of forward-pass keyword arguments.
    """
    torch.manual_seed(SEED)

    model = GLOBE(
        n_spatial_dims=3,
        output_field_ranks={"C_p": 0, "C_f": 1},
        boundary_source_data_ranks={
            "vehicle": {},
            "floor": {},
        },
        reference_length_names=["L_ref"],
        reference_area=1.0,
        n_communication_hyperlayers=2,
        hidden_layer_sizes=(32, 32),
        n_latent_scalars=4,
        n_latent_vectors=2,
        n_spherical_harmonics=4,
        theta=2.0,
        leaf_size=4,
    ).to(device)
    model.eval()

    ### Boundary meshes: lumpy sphere for vehicle, smaller for floor
    mesh_vehicle = lumpy_sphere.load(subdivisions=1, device=device)
    mesh_floor = lumpy_sphere.load(subdivisions=0, device=device)

    ### Prediction points
    generator = torch.Generator(device=device).manual_seed(SEED)
    prediction_points = torch.randn(50, 3, generator=generator, device=device)

    reference_lengths = {
        "L_ref": torch.tensor(1.0, dtype=torch.float32, device=device),
    }

    forward_kwargs = {
        "prediction_points": prediction_points,
        "boundary_meshes": {
            "vehicle": mesh_vehicle,
            "floor": mesh_floor,
        },
        "reference_lengths": reference_lengths,
    }

    return model, forward_kwargs


def _save_reference(output_mesh: Mesh[0, 3], path: Path) -> None:
    """Save output mesh point_data as a flat dict of tensors."""
    data = {k: v.detach().cpu() for k, v in output_mesh.point_data.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def _load_reference(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load saved reference tensors."""
    data = torch.load(path, map_location=device, weights_only=True)
    return data


@pytest.mark.parametrize("device", ["cpu"])
def test_globe_nonregression(device: str) -> None:
    """Full GLOBE forward pass must reproduce saved reference output.

    On first run, generates the reference file.  On subsequent runs,
    compares output against the saved reference with tight tolerances.
    Any numerical change from refactoring will cause this test to fail.
    """
    ref_path = DATA_DIR / "globe_nonregression_output.pth"
    device_obj = torch.device(device)

    model, forward_kwargs = _make_globe_and_inputs(device_obj)

    with torch.no_grad():
        output_mesh = model(**forward_kwargs)

    ### Sanity: outputs are finite
    for k, v in output_mesh.point_data.items():
        assert torch.all(torch.isfinite(v)), f"Non-finite values in {k}"

    if not ref_path.exists():
        _save_reference(output_mesh, ref_path)
        pytest.fail(
            f"Reference file {ref_path} did not exist and has been created. "
            f"Re-run the test to verify reproducibility."
        )

    reference = _load_reference(ref_path, device_obj)

    for key in reference:
        torch.testing.assert_close(
            output_mesh.point_data[key],
            reference[key],
            atol=ATOL,
            rtol=RTOL,
            msg=f"Output field '{key}' differs from reference",
        )


@pytest.mark.parametrize("device", ["cpu"])
def test_globe_gradient_flow(device: str) -> None:
    """Verify gradients flow through the full GLOBE forward pass.

    Ensures that the backward pass through all interaction phases,
    gather/scatter operations, and the MLP produces non-zero gradients
    on all model parameters.
    """
    device_obj = torch.device(device)
    model, forward_kwargs = _make_globe_and_inputs(device_obj)
    model.train()

    output_mesh = model(**forward_kwargs)
    loss = sum(v.sum() for v in output_mesh.point_data.values())
    loss.backward()

    n_params_with_grad = 0
    n_params_total = 0
    for name, p in model.named_parameters():
        n_params_total += 1
        if p.grad is not None and p.grad.abs().max() > 0:
            n_params_with_grad += 1

    assert n_params_with_grad == n_params_total, (
        f"Only {n_params_with_grad}/{n_params_total} parameters received "
        f"non-zero gradients"
    )
