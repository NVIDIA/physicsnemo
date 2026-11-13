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

import importlib
import re
import sys
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.core import Module
from physicsnemo.core.warnings import LegacyFeatureWarning
from physicsnemo.mesh import Mesh
from physicsnemo.mesh.primitives.procedural import lumpy_sphere
from physicsnemo.models.globe import GLOBE
from test.models.globe._helpers import assert_globe_attributes

DATA_DIR = Path(__file__).parent / "data"
SEED = 42
### The C_f outputs are about 1e-5, so an ATOL of 1e-5 hid row and column
### mix-ups.  Unchanged code matches the reference to about 1e-11.
ATOL = 1e-9
RTOL = 1e-5

### Reference file of each configuration, with its extra GLOBE arguments.
### The second reaches the far-target centroids and ``_compress_latents``.
### Its scale is 1e-2, not the examples' 100.0, which changes the outputs
### of this small model by only about 1e-12.
CONFIGS = {
    "globe_nonregression_output.pth": {},
    "globe_nonregression_output_no_far_expansion_compression_1e-2.pth": {
        "expand_far_targets": False,
        "latent_compression_scale": 1e-2,
    },
}


def _make_globe_and_inputs(
    device: torch.device,
    **globe_kwargs,
) -> tuple[GLOBE, dict]:
    """Construct a GLOBE model and inputs matching the DrivAerML config.

    Uses small mesh sizes for fast test execution while exercising all
    interaction phases (near-field, far-field, cross-BC communication).

    ``globe_kwargs`` are passed on to :class:`GLOBE`.

    Returns the model and a dict of forward-pass keyword arguments.
    """
    torch.manual_seed(SEED)

    model = GLOBE(
        n_spatial_dims=3,
        output_schema={"C_p": {"rank": 0}, "C_f": {"rank": 1}},
        boundary_source_schemas={
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
        **globe_kwargs,
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
@pytest.mark.parametrize(
    ("ref_name", "globe_kwargs"), CONFIGS.items(), ids=list(CONFIGS)
)
def test_globe_nonregression(device: str, ref_name: str, globe_kwargs: dict) -> None:
    """Full GLOBE forward pass must reproduce saved reference output.

    On first run, generates the reference file.  On subsequent runs,
    compares output against the saved reference with tight tolerances.
    Any numerical change from refactoring will cause this test to fail.
    """
    ref_path = DATA_DIR / ref_name
    device_obj = torch.device(device)

    model, forward_kwargs = _make_globe_and_inputs(device_obj, **globe_kwargs)

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


### A committed checkpoint, made with the code of its version, must keep
### loading through the base class and giving the same outputs.
CHECKPOINT_PATH = DATA_DIR / "globe_v1.mdlus"
CHECKPOINT_OUTPUT_PATH = DATA_DIR / "globe_checkpoint_output.pth"
CHECKPOINT_KWARGS = {
    "n_spatial_dims": 3,
    "output_schema": {"C_p": {"rank": 0}, "C_f": {"rank": 1}},
    "boundary_source_schemas": {"vehicle": {"alpha": {"rank": 0}}, "floor": {}},
    "reference_length_names": ["L_ref"],
    "reference_area": 1.0,
    "global_schema": {"U_inf": {"rank": 1}},
    "n_communication_hyperlayers": 1,
    "n_latent_scalars": 2,
    "n_latent_vectors": 1,
    "hidden_layer_sizes": [8],
    "leaf_size": 4,
}


def _make_checkpoint_inputs(device: torch.device) -> dict:
    """Forward-pass keyword arguments for the checkpointed model.

    Built on CPU and then moved, since ``lumpy_sphere`` draws its noise from
    a generator on the requested device, which would change the geometry.
    """
    mesh_vehicle = lumpy_sphere.load(subdivisions=1)
    mesh_vehicle.cell_data["alpha"] = mesh_vehicle.cell_centroids[:, 0]
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    return {
        "prediction_points": torch.randn(20, 3, generator=generator).to(device),
        "boundary_meshes": {
            "vehicle": mesh_vehicle.to(device),
            "floor": lumpy_sphere.load(radius=3.0, subdivisions=0).to(device),
        },
        "reference_lengths": {"L_ref": torch.tensor(1.0).to(device)},
        "global_data": TensorDict({"U_inf": torch.tensor([1.0, 0.0, 0.0])}).to(device),
    }


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_globe_from_checkpoint(device: str) -> None:
    """``Module.from_checkpoint`` rebuilds GLOBE from the committed checkpoint,
    with its constructor arguments, and reproduces the reference outputs.

    The checkpoint stores the pre-move class path
    ``physicsnemo.experimental.models.globe.model``, so this also tests that
    old checkpoints load as the promoted class."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not CHECKPOINT_PATH.exists():
        torch.manual_seed(SEED)
        model = GLOBE(**CHECKPOINT_KWARGS).eval()
        model.save(CHECKPOINT_PATH)
        with torch.no_grad():
            output = model(**_make_checkpoint_inputs(torch.device("cpu")))
        _save_reference(output, CHECKPOINT_OUTPUT_PATH)
        pytest.fail(
            f"Reference checkpoint {CHECKPOINT_PATH} and outputs "
            f"{CHECKPOINT_OUTPUT_PATH} did not exist and have been created. "
            "Re-run the test to verify reproducibility."
        )

    model = Module.from_checkpoint(CHECKPOINT_PATH).to(device).eval()

    assert type(model) is GLOBE
    assert_globe_attributes(model, CHECKPOINT_KWARGS)

    with torch.no_grad():
        output = model(**_make_checkpoint_inputs(torch.device(device)))
    reference = _load_reference(CHECKPOINT_OUTPUT_PATH, torch.device(device))
    assert set(output.point_data.keys()) == set(reference)
    ### The reference was made on CPU; CUDA rounds differently.
    tolerances = (
        {"atol": ATOL, "rtol": RTOL}
        if device == "cpu"
        else {"atol": 1e-6, "rtol": 1e-4}
    )
    for key in reference:
        torch.testing.assert_close(
            output.point_data[key],
            reference[key],
            **tolerances,
            msg=f"Output field '{key}' differs from reference",
        )


def test_globe_legacy_checkpoint_class_path() -> None:
    """The class paths stored by pre-move checkpoints resolve to GLOBE."""
    from physicsnemo.experimental.models.globe import GLOBE as LegacyPackageGLOBE
    from physicsnemo.experimental.models.globe.field_kernel import (
        BarnesHutKernel as LegacyBarnesHutKernel,
    )
    from physicsnemo.experimental.models.globe.model import (
        GLOBE as LegacyModuleGLOBE,
    )
    from physicsnemo.models.globe.field_kernel import BarnesHutKernel

    assert LegacyPackageGLOBE is GLOBE
    assert LegacyModuleGLOBE is GLOBE
    assert LegacyBarnesHutKernel is BarnesHutKernel


def test_globe_legacy_import_paths() -> None:
    """The pre-move package warns and re-exports the promoted objects."""
    from physicsnemo.models.globe import field_kernel

    ### Drop the cached legacy modules so the shim warning fires again.
    for module_name in list(sys.modules):
        if module_name.startswith("physicsnemo.experimental.models.globe"):
            del sys.modules[module_name]

    with pytest.warns(
        LegacyFeatureWarning, match=re.escape("physicsnemo.models.globe")
    ):
        legacy_pkg = importlib.import_module("physicsnemo.experimental.models.globe")

    assert legacy_pkg.GLOBE is GLOBE
    for name in ["Kernel", "BarnesHutKernel", "MultiscaleKernel"]:
        assert getattr(legacy_pkg, name) is getattr(field_kernel, name), name
