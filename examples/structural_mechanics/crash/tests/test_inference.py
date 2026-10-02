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

import sys
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference import save_vtp_sequence  # noqa: E402


@pytest.mark.parametrize("prediction_dtype", [torch.float16, torch.bfloat16])
def test_save_vtp_sequence_casts_amp_outputs_to_float32(tmp_path, prediction_dtype):
    """Verify save VTP sequence casts AMP outputs to float32."""
    frames_dir = tmp_path / "frames"
    pred_dir = tmp_path / "predicted"
    exact_dir = tmp_path / "exact"
    frames_dir.mkdir()

    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([3, 0, 1, 2])
    pv.PolyData(points, faces).save(frames_dir / "frame_000.vtp")

    predicted_positions = torch.tensor(
        [[[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.1, 0.0]]],
        dtype=prediction_dtype,
    )
    exact_positions = torch.tensor([points], dtype=torch.float32)
    predicted_stress = [torch.tensor([[0.1], [0.2], [0.3]], dtype=prediction_dtype)]
    exact_stress = [torch.tensor([[0.0], [0.1], [0.2]], dtype=torch.float32)]

    save_vtp_sequence(
        preds=predicted_positions,
        exacts=exact_positions,
        vtp_frames_dir=frames_dir,
        out_pred_dir=pred_dir,
        out_exact_dir=exact_dir,
        extra_fields={"stress_vm": predicted_stress},
        exact_extra_fields={"stress_vm": exact_stress},
    )

    predicted_mesh = pv.read(pred_dir / "frame_000_pred.vtp")
    exact_mesh = pv.read(exact_dir / "frame_000_exact.vtp")

    assert predicted_mesh.point_data["pred_stress_vm"].dtype == np.float32
    assert exact_mesh.point_data["pred_stress_vm"].dtype == np.float32
    np.testing.assert_allclose(
        exact_mesh.point_data["diff_stress_vm"],
        np.array([0.1, 0.1, 0.1], dtype=np.float32),
        atol=1e-3,
    )
