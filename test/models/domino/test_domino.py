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

r"""
Tests for the DoMINO model.

This module contains tests for:
- MOD-008a: Constructor/attributes tests
- MOD-008b: Non-regression tests with reference data
- MOD-008c: Checkpoint loading tests
"""

from pathlib import Path

import pytest
import torch

from physicsnemo.models.domino.config import DEFAULT_MODEL_PARAMS as model_params
from test.common.fwdaccuracy import save_output
from test.common.utils import compare_output
from test.conftest import requires_module


def validate_domino(
    model,
    input_dict,
    file_name,
    device,
    rtol=1e-3,
    atol=1e-3,
):
    """Validate DoMINO model output against reference data."""
    # Perform a forward pass of the model
    output = model.forward(input_dict)

    assert not torch.isnan(output[0]).any()
    assert not torch.isnan(output[1]).any()

    if file_name is None:
        file_name = model.meta.name + "_output.pth"
    file_name = (
        Path(__file__).parents[0].resolve() / Path("data") / Path(file_name.lower())
    )
    # If file does not exist, we will create it then error
    # Model should then reproduce it on next pytest run
    if not file_name.exists():
        save_output(output, file_name)
        raise IOError(
            f"Output check file {str(file_name)} wasn't found so one was created. "
            f"Please re-run the test."
        )
    # Load tensor dictionary and check
    else:
        tensor_dict = torch.load(str(file_name))
        output_target = tuple([value.to(device) for value in tensor_dict.values()])
        return compare_output(output, output_target, rtol, atol)


def create_test_input_dict(device, params):
    """Create a test input dictionary for DoMINO model."""
    bsize = 1
    nx, ny, nz = params.interp_res
    num_neigh = params.num_neighbors_surface

    pos_normals_closest_vol = torch.randn(bsize, 100, 3).to(device)
    pos_normals_com_vol = torch.randn(bsize, 100, 3).to(device)
    pos_normals_com_surface = torch.randn(bsize, 100, 3).to(device)
    geom_centers = torch.randn(bsize, 100, 3).to(device)
    grid = torch.randn(bsize, nx, ny, nz, 3).to(device)
    surf_grid = torch.randn(bsize, nx, ny, nz, 3).to(device)
    sdf_grid = torch.randn(bsize, nx, ny, nz).to(device)
    sdf_surf_grid = torch.randn(bsize, nx, ny, nz).to(device)
    sdf_nodes = torch.randn(bsize, 100, 1).to(device)
    surface_coordinates = torch.randn(bsize, 100, 3).to(device)
    surface_neighbors = torch.randn(bsize, 100, num_neigh, 3).to(device)
    surface_normals = torch.randn(bsize, 100, 3).to(device)
    surface_neighbors_normals = torch.randn(bsize, 100, num_neigh, 3).to(device)
    surface_sizes = torch.rand(bsize, 100).to(device) + 1e-6
    surface_neighbors_sizes = torch.rand(bsize, 100, num_neigh).to(device) + 1e-6
    volume_coordinates = torch.randn(bsize, 100, 3).to(device)
    vol_grid_max_min = torch.randn(bsize, 2, 3).to(device)
    surf_grid_max_min = torch.randn(bsize, 2, 3).to(device)
    global_params_values = torch.randn(bsize, 2, 1).to(device)
    global_params_reference = torch.randn(bsize, 2, 1).to(device)

    return {
        "pos_volume_closest": pos_normals_closest_vol,
        "pos_volume_center_of_mass": pos_normals_com_vol,
        "pos_surface_center_of_mass": pos_normals_com_surface,
        "geometry_coordinates": geom_centers,
        "grid": grid,
        "surf_grid": surf_grid,
        "sdf_grid": sdf_grid,
        "sdf_surf_grid": sdf_surf_grid,
        "sdf_nodes": sdf_nodes,
        "surface_mesh_centers": surface_coordinates,
        "surface_mesh_neighbors": surface_neighbors,
        "surface_normals": surface_normals,
        "surface_neighbors_normals": surface_neighbors_normals,
        "surface_areas": surface_sizes,
        "surface_neighbors_areas": surface_neighbors_sizes,
        "volume_mesh_centers": volume_coordinates,
        "volume_min_max": vol_grid_max_min,
        "surface_min_max": surf_grid_max_min,
        "global_params_values": global_params_values,
        "global_params_reference": global_params_reference,
    }


# =============================================================================
# MOD-008a: Constructor/attributes tests
# =============================================================================


@requires_module("warp")
@pytest.mark.parametrize(
    "config",
    ["default", "custom"],
    ids=["with_defaults", "with_custom_args"],
)
def test_domino_constructor(device, config, pytestconfig):
    """Test DoMINO model constructor and attributes (MOD-008a).

    This test verifies:
    1. Model can be instantiated with default arguments
    2. Model can be instantiated with custom arguments
    3. All public attributes have expected values
    """
    from physicsnemo.models.domino.model import DoMINO

    torch.manual_seed(0)

    params = model_params

    if config == "default":
        # Test with minimal required arguments (uses defaults for optional params)
        model = DoMINO(
            input_features=3,
            output_features_vol=4,
            output_features_surf=5,
            model_parameters=params,
        ).to(device)

        # Verify default values
        assert model.global_features == 2
        assert model.output_features_vol == 4
        assert model.output_features_surf == 5
        assert model.num_variables_vol == 4
        assert model.num_variables_surf == 5
        assert model.grid_resolution == params.interp_res
        assert model.use_surface_normals == params.use_surface_normals
        assert model.use_surface_area == params.use_surface_area
        assert model.encode_parameters == params.encode_parameters
        assert model.geo_encoding_type == params.geometry_encoding_type

    else:
        # Test with custom arguments
        custom_global_features = 4
        custom_output_vol = 6
        custom_output_surf = 8

        model = DoMINO(
            input_features=3,
            output_features_vol=custom_output_vol,
            output_features_surf=custom_output_surf,
            global_features=custom_global_features,
            model_parameters=params,
        ).to(device)

        # Verify custom values
        assert model.global_features == custom_global_features
        assert model.output_features_vol == custom_output_vol
        assert model.output_features_surf == custom_output_surf
        assert model.num_variables_vol == custom_output_vol
        assert model.num_variables_surf == custom_output_surf

    # Common assertions for both configs
    assert model.meta.name == "DoMINO"
    assert hasattr(model, "geo_rep_volume")
    assert hasattr(model, "geo_rep_surface")
    assert hasattr(model, "surface_local_geo_encodings")
    assert hasattr(model, "volume_local_geo_encodings")
    assert hasattr(model, "solution_calculator_surf")
    assert hasattr(model, "solution_calculator_vol")


@requires_module("warp")
def test_domino_constructor_volume_only(device, pytestconfig):
    """Test DoMINO model in volume-only mode."""
    from physicsnemo.models.domino.model import DoMINO

    torch.manual_seed(0)

    params = model_params

    model = DoMINO(
        input_features=3,
        output_features_vol=4,
        output_features_surf=None,
        model_parameters=params,
    ).to(device)

    assert model.output_features_vol == 4
    assert model.output_features_surf is None
    assert hasattr(model, "solution_calculator_vol")
    assert not hasattr(model, "solution_calculator_surf")


@requires_module("warp")
def test_domino_constructor_surface_only(device, pytestconfig):
    """Test DoMINO model in surface-only mode."""
    from physicsnemo.models.domino.model import DoMINO

    torch.manual_seed(0)

    params = model_params

    model = DoMINO(
        input_features=3,
        output_features_vol=None,
        output_features_surf=5,
        model_parameters=params,
    ).to(device)

    assert model.output_features_vol is None
    assert model.output_features_surf == 5
    assert not hasattr(model, "solution_calculator_vol")
    assert hasattr(model, "solution_calculator_surf")


@requires_module("warp")
def test_domino_constructor_invalid(device, pytestconfig):
    """Test DoMINO model raises error when both outputs are None."""
    from physicsnemo.models.domino.model import DoMINO

    params = model_params

    with pytest.raises(ValueError, match="At least one of"):
        DoMINO(
            input_features=3,
            output_features_vol=None,
            output_features_surf=None,
            model_parameters=params,
        )


# =============================================================================
# MOD-008b: Non-regression tests with reference data
# =============================================================================


@requires_module("warp")
@pytest.mark.parametrize("processor_type", ["unet", "conv"])
def test_domino_forward(device, processor_type, pytestconfig):
    """Test DoMINO forward pass against reference output (MOD-008b)."""
    from physicsnemo.models.domino.model import DoMINO

    torch.manual_seed(0)

    params = model_params
    params.geometry_rep.geo_processor.processor_type = processor_type

    model = DoMINO(
        input_features=3,
        output_features_vol=4,
        output_features_surf=5,
        global_features=2,
        model_parameters=params,
    ).to(device)

    input_dict = create_test_input_dict(device, params)

    assert validate_domino(
        model,
        input_dict,
        file_name=f"domino_output-{processor_type}.pth",
        device=device,
    )


@requires_module("warp")
def test_domino_forward_output_shapes(device, pytestconfig):
    """Test DoMINO forward pass output shapes."""
    from physicsnemo.models.domino.model import DoMINO

    torch.manual_seed(0)

    params = model_params
    output_vol = 4
    output_surf = 5
    num_points = 100

    model = DoMINO(
        input_features=3,
        output_features_vol=output_vol,
        output_features_surf=output_surf,
        model_parameters=params,
    ).to(device)

    input_dict = create_test_input_dict(device, params)

    output = model(input_dict)

    # Check output tuple structure
    assert isinstance(output, tuple)
    assert len(output) == 2

    # Check volume output shape
    vol_output, surf_output = output
    assert vol_output is not None
    assert vol_output.shape == (1, num_points, output_vol)

    # Check surface output shape
    assert surf_output is not None
    assert surf_output.shape == (1, num_points, output_surf)


@requires_module("warp")
def test_domino_forward_input_validation(device, pytestconfig):
    """Test DoMINO forward pass input validation."""
    from physicsnemo.models.domino.model import DoMINO

    torch.manual_seed(0)

    params = model_params

    model = DoMINO(
        input_features=3,
        output_features_vol=4,
        output_features_surf=5,
        model_parameters=params,
    ).to(device)

    # Test with missing required key
    incomplete_dict = {"geometry_coordinates": torch.randn(1, 100, 3).to(device)}

    with pytest.raises(ValueError, match="Missing required keys"):
        model(incomplete_dict)


# =============================================================================
# MOD-008c: Checkpoint loading tests
# =============================================================================


@requires_module("warp")
def test_domino_checkpoint_save_load(device, tmp_path, pytestconfig):
    """Test DoMINO model checkpoint save and load (MOD-008c).

    This test verifies:
    1. Model can be saved to checkpoint
    2. Model can be loaded from checkpoint
    3. Loaded model produces same output as original
    """
    from physicsnemo import Module
    from physicsnemo.models.domino.model import DoMINO

    torch.manual_seed(0)

    params = model_params

    # Create and configure original model
    model_original = DoMINO(
        input_features=3,
        output_features_vol=4,
        output_features_surf=5,
        global_features=2,
        model_parameters=params,
    ).to(device)
    model_original.eval()

    # Create test input
    input_dict = create_test_input_dict(device, params)

    # Get original output
    with torch.no_grad():
        output_original = model_original(input_dict)

    # Save checkpoint
    checkpoint_path = tmp_path / "domino_test.mdlus"
    model_original.save(str(checkpoint_path))

    # Verify checkpoint file exists
    assert checkpoint_path.exists()

    # Load model from checkpoint
    model_loaded = Module.from_checkpoint(str(checkpoint_path)).to(device)
    model_loaded.eval()

    # Verify loaded model attributes
    assert model_loaded.output_features_vol == model_original.output_features_vol
    assert model_loaded.output_features_surf == model_original.output_features_surf
    assert model_loaded.global_features == model_original.global_features
    assert model_loaded.grid_resolution == model_original.grid_resolution

    # Get loaded model output
    with torch.no_grad():
        output_loaded = model_loaded(input_dict)

    # Compare outputs
    assert torch.allclose(output_loaded[0], output_original[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(output_loaded[1], output_original[1], atol=1e-5, rtol=1e-5)


@requires_module("warp")
def test_domino_model_import(pytestconfig):
    """Test that DoMINO can be imported from physicsnemo.models."""
    from physicsnemo.models import DoMINO

    assert DoMINO is not None
    assert hasattr(DoMINO, "forward")
    assert hasattr(DoMINO, "__init__")
