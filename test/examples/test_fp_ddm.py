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

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint

EXAMPLE_ROOT = Path(__file__).resolve().parents[2] / "examples" / "cfd" / "fp_ddm"
sys.path.insert(0, str(EXAMPLE_ROOT))

from fpddm.data import ThermalDataset  # noqa: E402
from fpddm.domain import Domain, Fields, MaskKind  # noqa: E402
from fpddm.interfaces import ExchangeInterfaceHandler  # noqa: E402
from fpddm.model import ThermalPINO, thermal_residual  # noqa: E402
from fpddm.pipeline import run_fpddm  # noqa: E402
from fpddm.schwarz import PhysicsInformedAdapter  # noqa: E402
from fpddm.solvers import NeuralDomainSolver  # noqa: E402
from fpddm.thermal import Heat2DSolver  # noqa: E402


@pytest.fixture
def model_config():
    return {
        "in_channels": 5,
        "out_channels": 1,
        "modes": 2,
        "width": 8,
        "layers": 1,
        "decoder_layers": 1,
        "decoder_width": 8,
        "padding": 0,
        "normalization": {
            "length": 1.0,
            "conductivity": 200.0,
            "temperature_reference": 350.0,
            "temperature_scale": 100.0,
            "heat_source": 1.0,
        },
    }


def _two_patch_domain() -> Domain:
    domain = Domain(rows=1, columns=2, width=8, height=8, overlap=2)
    shape = (domain.total_height, domain.total_width)
    temperature_bc = torch.zeros(shape)
    temperature_bc[0] = 300.0
    temperature_bc[-1] = 400.0
    temperature_bc[:, 0] = 350.0
    temperature_bc[:, -1] = 325.0
    domain.fields = {
        Fields.TEMPERATURE: temperature_bc.clone(),
        Fields.CONDUCTIVITY: torch.ones(shape),
        Fields.HEAT_SOURCE: torch.zeros(shape),
        Fields.TEMPERATURE_BC: temperature_bc,
    }
    mask = torch.ones(shape, dtype=torch.bool)
    mask[1:-1, 1:-1] = False
    domain.masks = {Fields.TEMPERATURE_BC: {MaskKind.DIRICHLET: mask}}
    return domain


def test_thermal_dataset_coordinates_and_shape():
    dataset = ThermalDataset(
        {
            "n_samples": 2,
            "grid_size": 8,
            "k_generation_method": "uniform",
            "boundary_generation_method": "pixelwise",
            "use_on_the_fly": True,
        }
    )
    sample = dataset[0]
    assert sample.shape == (5, 8, 8)
    assert torch.isfinite(sample).all()
    expected_axis = torch.linspace(0.0, 1.0, 8)
    assert torch.allclose(sample[0, 0], expected_axis)
    assert torch.allclose(sample[0, :, 0], torch.zeros(8))
    assert torch.allclose(sample[1, :, 0], expected_axis)
    assert torch.allclose(sample[1, 0], torch.zeros(8))
    assert torch.unique(sample[2]).numel() == 1
    assert torch.count_nonzero(sample[3, 1:-1, 1:-1]) == 0


def test_thermal_residual_includes_heat_source():
    temperature = torch.zeros(1, 1, 8, 8)
    inputs = torch.zeros(1, 5, 8, 8)
    inputs[:, 2] = 1.0
    assert torch.count_nonzero(thermal_residual(temperature, inputs)) == 0

    inputs[:, 4] = 2.0
    residual = thermal_residual(temperature, inputs, source_scale=0.25)
    assert torch.allclose(residual, torch.full_like(residual, 0.5))


def test_physicsnemo_fno_shape_and_boundary(model_config):
    model = ThermalPINO(model_config)
    model.set_boundary_enforcement(True)
    inputs = torch.randn(2, 5, 8, 8)
    prediction = model(inputs)
    assert prediction.shape == (2, 1, 8, 8)
    assert torch.equal(prediction[:, :, 0], inputs[:, 3:4, 0])
    assert torch.equal(prediction[:, :, -1], inputs[:, 3:4, -1])


def test_heat_solver_handles_mixed_batch_convergence():
    solver = Heat2DSolver(max_iter=50, tolerance=1.0e-7, device="cpu")
    conductivity = torch.ones(2, 8, 8)
    heat_source = torch.zeros(2, 8, 8)
    temperature_bc = torch.zeros(2, 8, 8)
    temperature_bc[1, 0] = 1.0
    mask = torch.ones(2, 8, 8, dtype=torch.bool)
    mask[:, 1:-1, 1:-1] = False
    temperature = solver.solve(
        conductivity,
        heat_source,
        temperature_bc,
        mask,
        torch.zeros_like(mask),
    )
    assert temperature.shape == (2, 8, 8)
    assert torch.isfinite(temperature).all()
    assert torch.count_nonzero(temperature[0]) == 0


def test_exchange_interface_updates_neighbor_boundaries():
    grid = _two_patch_domain().build_subdomains()
    left, right = grid[0]
    left.fields[Fields.TEMPERATURE].fill_(1.0)
    right.fields[Fields.TEMPERATURE].fill_(3.0)

    handler = ExchangeInterfaceHandler(alpha=1.0)
    handler([left, right])

    assert torch.all(left.fields[Fields.TEMPERATURE_BC][:, -1] == 3.0)
    assert torch.all(right.fields[Fields.TEMPERATURE_BC][:, 0] == 1.0)
    assert torch.count_nonzero(left.fields[Fields.TEMPERATURE_BC][1:-1, 1:-1]) == 0


def test_physicsnemo_checkpoint_round_trip(tmp_path, model_config):
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    model = ThermalPINO(model_config)
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    checkpoint_dir = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint_dir,
        models=model,
        epoch=3,
        metadata={"validation_loss": 0.125},
    )

    restored = ThermalPINO(model_config)
    metadata = {}
    epoch = load_checkpoint(
        checkpoint_dir,
        models=restored,
        metadata_dict=metadata,
        device="cpu",
    )
    assert epoch == 3
    assert metadata == {"validation_loss": 0.125}
    for name, value in restored.state_dict().items():
        assert torch.equal(value, expected[name])


def test_test_time_adaptation_restores_inference_modes(model_config):
    solver = NeuralDomainSolver(model_config, batch_size=2, device="cpu")
    subdomains = [
        item for row in _two_patch_domain().build_subdomains() for item in row
    ]
    adapter = PhysicsInformedAdapter(
        solver,
        steps=1,
        learning_rate=1.0e-5,
        batch_size=2,
    )
    adapter(subdomains)
    solver.solve_batch(subdomains)

    assert solver.model.use_dimensional_input
    assert solver.model.use_dimensional_output
    assert solver.model.enforce_boundary
    assert not solver.model.training
    assert torch.isfinite(torch.tensor(solver.loss))
    assert solver.loss > 0.0


def test_fem_fpddm_end_to_end(tmp_path, model_config):
    output_dir = tmp_path / "run"
    config = {
        "model": model_config,
        "dataset": {
            "grid_size": 8,
            "k_min": 3.0,
            "k_max": 20.0,
            "T_min": 300.0,
            "T_max": 400.0,
            "q_min": 0.0,
            "q_max": 0.0,
        },
        "domain": {
            "rows": 2,
            "columns": 2,
            "width": 8,
            "height": 8,
            "overlap": 2,
            "coarse_initialization": False,
            "outer_boundary": {
                "top": {"type": "dirichlet", "value": 330.0},
                "bottom": {"type": "dirichlet", "value": 390.0},
                "left": {"type": "dirichlet", "value": 350.0},
                "right": {"type": "dirichlet", "value": 310.0},
            },
        },
        "fem": {"batch_size": 4, "max_iter": 100, "tolerance": 1.0e-7},
        "run": {
            "solver": "fem",
            "output_dir": str(output_dir),
            "handler": "parallel",
            "handler_learning_rate": 0.01,
            "max_iterations": 2,
            "tolerance": -1.0,
            "patience": 100,
            "visualize": False,
            "ground_truth": True,
        },
        "visualization": {},
    }
    result = run_fpddm(config, device="cpu")

    assert result.reference is not None
    assert result.reference.shape == (14, 14)
    assert len(result.metrics) == 2
    assert all("loss_overlap" in row for row in result.metrics)
    assert all("loss_true_NRMAE" in row for row in result.metrics)
    assert (output_dir / "metrics.csv").is_file()
    with (output_dir / "summary.json").open(encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    assert summary["iterations"] == 2
