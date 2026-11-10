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

"""Optional mesh descriptors -> density scores -> application-owned policy.

Run: python examples/minimal/uq/geometry_scoring.py
Synthetic spheres illustrate composition and persistence, not OOD accuracy.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch

from physicsnemo.experimental.uq.ood import DensityScorer
from physicsnemo.experimental.uq.ood.geometry import FeatureSchema, extract_features
from physicsnemo.mesh.primitives import surfaces


def features(radii):
    """Replace primitive generation with application-owned mesh loading as needed."""
    return torch.stack(
        [
            extract_features(
                surfaces.sphere_uv.load(
                    radius=float(radius), theta_resolution=12, phi_resolution=12
                ),
                return_tensor=True,
            )
            for radius in radii
        ]
    )


def main():
    """Fit, save, restore, validate feature identity, and apply the same policy."""
    reference = features(torch.linspace(0.8, 1.2, 24))
    calibration = features(torch.linspace(0.805, 1.195, 24))
    query = features([1.0, 2.0])
    # These values belong to the application, not to a generic density model.
    metadata = {
        "names": list(FeatureSchema.names),
        "version": FeatureSchema.version,
        "hash": FeatureSchema.hash,
        "length_unit": "m",
        "preprocessing": "raw descriptor vector",
    }
    # Illustrative ranks with distinct attainable WARN/REJECT tiers for 24 rows.
    policy = {"warn_percentile": 90.0, "reject_percentile": 97.0}
    scorer = DensityScorer(method="pce", pce_components=3)
    scorer.fit(reference)

    scorer.calibrate(calibration)

    # One weights-only artifact holds state and application metadata.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "density.pt"
        torch.save(
            {"scorer": scorer.state_dict(), "features": metadata, "policy": policy},
            path,
        )
        saved = torch.load(path, weights_only=True, map_location="cpu")
        if saved["features"] != metadata:
            raise ValueError(
                "Reference features use different schema, units, or preprocessing"
            )
        restored = DensityScorer()
        restored.load_state_dict(saved["scorer"])
        policy = saved["policy"]
    torch.testing.assert_close(restored.score(query), scorer.score(query))
    scores = restored.score(query)
    percentiles = restored.percentiles(scores)
    statuses = [
        "REJECT"
        if p >= policy["reject_percentile"]
        else "WARN"
        if p >= policy["warn_percentile"]
        else "OK"
        for p in percentiles
    ]
    np.testing.assert_allclose(percentiles, scorer.percentiles(scorer.score(query)))
    print("percentiles:", percentiles, "application decisions:", statuses)


if __name__ == "__main__":
    main()
