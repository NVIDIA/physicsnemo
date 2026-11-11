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

"""Predict the surface fields and compressor outputs of one Rotor37 case."""

import argparse
from pathlib import Path

import numpy as np
import torch
from dataset import CaseEncoder
from models import load_model, predict
from objectives import FieldDecoder
from omegaconf import OmegaConf
from prepare_data import FIELD_NAMES, GLOBAL_NAMES
from train import strict_float32


class Rotor37Predictor:
    """Run a trained model on blade geometries and operating conditions.

    Everything the prediction needs is read from the run directory, so no
    prepared dataset is required.
    """

    def __init__(self, run_dir, device="cpu"):
        run_dir = Path(run_dir)
        run_cfg = OmegaConf.load(run_dir / "config.yaml")
        self.device = torch.device(device)
        self.encoder = CaseEncoder(
            run_dir / "basis.npz",
            run_dir / "stats.json",
            run_cfg.data.geometry_points,
        )
        self.decoder = FieldDecoder(self.encoder.basis, self.encoder.stats)
        self.decoder = self.decoder.to(self.device)
        self.model = load_model(run_dir, self.device)

    @torch.no_grad()
    def __call__(self, points, normals, conditions):
        """Predict one case.

        ``points`` and ``normals`` have shape ``[vertices, 3]`` in the vertex
        order of the training mesh, and ``conditions`` holds the rotation
        speed ``Omega`` and the pressure ``P``. Returns density, pressure and
        temperature at every vertex and the mass flow, compression ratio and
        efficiency in source units.
        """
        inputs = self.encoder.encode(
            np.asarray(points), np.asarray(normals), np.asarray(conditions)
        )
        batch = {key: value[None].to(self.device) for key, value in inputs.items()}
        with strict_float32():
            prediction = predict(self.model, batch, self.decoder)
            fields = self.decoder.physical(prediction.coefficients)[0]
        globals_ = prediction.globals[0].cpu().numpy().astype(np.float64)
        return {
            "fields": fields.cpu().numpy(),
            "globals": self.encoder.denormalize_globals(globals_),
        }


def main():
    """Predict a case stored as an ``.npz`` file of points, normals and conditions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/geotransolver"))
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction.npz"))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    predictor = Rotor37Predictor(args.run_dir, args.device)
    with np.load(args.case) as case:
        points = case["points"]
        prediction = predictor(points, case["normals"], case["conditions"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points=points,
        quads=predictor.encoder.basis["quads"],
        field_names=np.array(FIELD_NAMES),
        global_names=np.array(GLOBAL_NAMES),
        **prediction,
    )
    for name, value in zip(GLOBAL_NAMES, prediction["globals"]):
        print(f"{name}: {value:.6g}")
    print(f"Saved the surface fields to {args.output}")


if __name__ == "__main__":
    main()
