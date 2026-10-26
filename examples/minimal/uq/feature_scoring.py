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

"""GeoTransolver features -> reusable scorer -> application-owned decision.

Run: python examples/minimal/uq/feature_scoring.py
Random inputs and an untrained model illustrate composition only.
"""

import torch

from physicsnemo.experimental.uq.ood import DensityScorer, KNNScorer
from physicsnemo.models.geotransolver import GeoTransolver


@torch.no_grad()
def main():
    """Use detached geometry tokens without installing model hooks or a guard."""
    torch.manual_seed(41)
    model = GeoTransolver(
        functional_dim=3,
        geometry_dim=3,
        global_dim=2,
        out_dim=1,
        n_layers=1,
        n_hidden=16,
        n_head=4,
        slice_num=4,
        use_te=False,
    ).eval()

    def features(batch):
        inputs = torch.randn(batch, 11, 3)
        geometry = torch.randn(batch, 13, 3)
        global_inputs = torch.randn(batch, 1, 2)
        _, tokens = model(
            inputs,
            geometry=geometry,
            global_embedding=global_inputs,
            return_geometry_features=True,
        )
        # Pool over heads and slice tokens.
        # Persist this pooling choice with application/checkpoint metadata.
        return tokens.mean(dim=(1, 2)), global_inputs.squeeze(1)

    (reference, reference_inputs), (validation, _), (query, query_inputs) = (
        features(60),
        features(20),
        features(3),
    )
    # Optional input-range policy belongs to the recipe, separately from OOD scores.
    lower, upper = reference_inputs.amin(0), reference_inputs.amax(0)
    outside_bounds = ((query_inputs < lower) | (query_inputs > upper)).any(dim=-1)
    for scorer in (KNNScorer(k=5), DensityScorer(method="gmm")):
        scorer.fit(reference)
        threshold = torch.quantile(scorer.score(validation), 0.95)
        scores = scorer.score(query)
        print(
            type(scorer).__name__,
            scores,
            "flagged:",
            (scores > threshold) | outside_bounds,
        )


if __name__ == "__main__":
    main()
