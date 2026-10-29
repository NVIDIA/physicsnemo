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

"""Descriptor/scorer/policy composition replaces the geometry wrapper lifecycle."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("method", ["gmm", "pce"])
def test_geometry_recipe_round_trip_and_metadata(tmp_path, device, method):
    pv = pytest.importorskip("pyvista")
    path = (
        Path(__file__).parents[2] / "examples/minimal/guardrails/geometry_validation.py"
    )
    spec = importlib.util.spec_from_file_location("geometry_recipe", path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    train, query = tmp_path / "train", tmp_path / "query"
    train.mkdir()
    query.mkdir()
    for index, radius in enumerate(np.linspace(0.8, 1.2, 12)):
        mesh = pv.Sphere(radius=radius, theta_resolution=12, phi_resolution=12)
        mesh.scale((1, 1.5, 2), inplace=True)
        mesh.save(train / f"sample_{index:02d}.stl")
    pv.Sphere(radius=2.0, theta_resolution=12, phi_resolution=12).save(
        query / "query.stl"
    )
    (query / "bad.stl").write_text("not an STL file")
    artifact = tmp_path / "density.npz"
    with pytest.warns(UserWarning, match="Skipping invalid mesh"):
        result = recipe.fit_and_score(train, query, method, "test", artifact, device)
    assert result["total"] == 1
    saved = json.loads(artifact.with_suffix(".json").read_text())
    restored, policy = recipe.load_scorer(artifact, saved["features"], device)
    with pytest.warns(UserWarning, match="Skipping invalid mesh"):
        names, features = recipe.directory_features(query)
    score = restored.score(features)
    percentile = restored.percentiles(score)[0]
    row = result["results"][0]
    assert row["name"] == names[0]
    assert row["score"] == pytest.approx(score.item())
    assert row["percentile"] == pytest.approx(percentile)
    assert row["status"] == (
        "REJECT"
        if percentile >= policy["reject_pct"]
        else "WARN"
        if percentile >= policy["warn_pct"]
        else "OK"
    )
    with pytest.raises(ValueError, match="schema, units, or preprocessing"):
        recipe.load_scorer(artifact, saved["features"] | {"length_unit": "mm"}, device)
    # A v1.0 artifact used the same values but incorrectly named its PCA columns.
    legacy_names = saved["features"]["names"].copy()
    legacy_names[3:9] = [f"pca_axis{axis}_{c}" for axis in (1, 2) for c in "xyz"]
    from physicsnemo.experimental.uq.ood.geometry import feature_hash

    legacy_features = saved["features"] | {
        "names": legacy_names,
        "version": "v1.0",
        "hash": feature_hash(legacy_names),
    }
    artifact.with_suffix(".json").write_text(
        json.dumps(saved | {"features": legacy_features})
    )
    with pytest.raises(ValueError, match="schema, units, or preprocessing"):
        recipe.load_scorer(artifact, saved["features"], device)
    # Correcting metadata alone restores the same scores; do not permute weights.
    artifact.with_suffix(".json").write_text(json.dumps(saved))
    corrected, _ = recipe.load_scorer(artifact, saved["features"], device)
    np.testing.assert_array_equal(
        corrected.score(features).cpu().numpy(), score.cpu().numpy()
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="No valid STL"):
        recipe.directory_features(empty)
