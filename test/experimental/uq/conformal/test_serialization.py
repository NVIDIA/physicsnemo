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

"""Tests for portable fitted-predictor artifacts."""

import pickle
import re

import pytest
import torch

from physicsnemo.experimental.uq.conformal import (
    AuxDifficulty,
    ConformalPredictor,
    NormalizedErrorScore,
    QuantileRegressionScore,
)
from physicsnemo.experimental.uq.conformal._utils import points_fingerprint
from test.experimental.uq.conformal._helpers import (
    TIERS,
    fit,
    fitted_risk,
    make_predictor,
)

_MESH_POINTS = torch.arange(12, dtype=torch.float32).reshape(6, 2)
_MESH = points_fingerprint(_MESH_POINTS)


@pytest.mark.parametrize("tier", TIERS)
def test_save_load_round_trip(tier, tmp_path):
    generator = torch.Generator().manual_seed(51)
    predictor, points = fit(tier, generator=generator, n_samples=10, shape=(6, 3))
    if tier == "functional":  # a fresh point set at deployment is allowed
        points = torch.rand(6, 2, generator=generator)
    prediction = torch.zeros(6, 3)
    lo_ref, hi_ref = predictor.predict_interval(prediction, points=points)

    path = tmp_path / f"{tier}.pt"
    predictor.save(path)
    loaded = ConformalPredictor.load(path)
    assert type(loaded) is ConformalPredictor
    assert loaded.tier == tier
    assert loaded.alpha == predictor.alpha
    assert loaded.n_cal == predictor.n_cal
    assert loaded._mesh_fingerprint == predictor._mesh_fingerprint
    lo, hi = loaded.predict_interval(prediction, points=points)
    torch.testing.assert_close(lo, lo_ref)
    torch.testing.assert_close(hi, hi_ref)


def test_artifact_has_one_exact_schema_and_cellwise_load_requires_the_mesh(tmp_path):
    provenance = {"dataset": "drivaer", "epoch": 300}
    path = tmp_path / "artifact.pt"
    make_predictor(
        tier="cellwise", thresholds=torch.ones(6, 3), points=_MESH_POINTS
    ).save(path, provenance=provenance)
    payload = torch.load(path, weights_only=True)

    assert set(payload) == {
        "format",
        "version",
        "tier",
        "score",
        "alpha",
        "n_cal",
        "thresholds",
        "difficulty",
        "mesh_fingerprint",
        "provenance",
    }
    assert payload["format"] == "physicsnemo.uq.conformal"
    assert payload["version"] == 1
    assert payload["tier"] == "cellwise"
    assert payload["score"] == {"kind": "absolute_error", "kwargs": {}}
    assert payload["difficulty"] is None
    assert payload["mesh_fingerprint"] == _MESH
    assert payload["provenance"] == provenance
    assert set(payload["thresholds"]) == {"__tensor__"}

    loaded = ConformalPredictor.load(path)
    prediction = torch.zeros(6, 3)
    with pytest.raises(TypeError, match="points must be a torch.Tensor"):
        loaded.predict_interval(prediction)
    for changed in (_MESH_POINTS.flip(0), _MESH_POINTS.to(torch.float64)):
        with pytest.raises(ValueError, match="exact calibration mesh"):
            loaded.predict_interval(prediction, points=changed)
    lo, hi = loaded.predict_interval(prediction, points=_MESH_POINTS.clone())
    assert lo.shape == hi.shape == prediction.shape


def test_constructor_and_threshold_access_snapshot_storage(tmp_path):
    thresholds = torch.ones(6, 3)
    predictor = make_predictor(
        tier="cellwise", thresholds=thresholds, points=_MESH_POINTS
    )
    path = tmp_path / "artifact.pt"
    predictor.save(path)
    loaded = ConformalPredictor.load(path)
    for tensor in (thresholds, predictor.thresholds, loaded.thresholds):
        tensor.fill_(float("nan"))
    assert torch.equal(predictor.thresholds, torch.ones(6, 3))
    assert torch.equal(loaded.thresholds, torch.ones(6, 3))


def test_map_location_places_loaded_thresholds(device, tmp_path):
    path = tmp_path / "artifact.pt"
    predictor, points = fit("cellwise", n_samples=5, shape=(6, 3))
    predictor.save(path)
    loaded = ConformalPredictor.load(path, map_location=device)
    lo, _ = loaded.predict_interval(
        torch.zeros(6, 3, device=device), points=points.to(device)
    )
    assert lo.device.type == torch.device(device).type


def test_provenance_is_read_only_snapshotted_and_survives_resave(tmp_path):
    provenance = {
        "identity": {"dataset": "drivaer", "checkpoint": "/runs/x"},
        "audit": [True, None, 300, 0.25],
    }
    predictor = fitted_risk()
    with pytest.raises(AttributeError, match="no setter"):
        predictor.provenance = provenance

    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    predictor.save(first, provenance=provenance)
    provenance["identity"]["dataset"] = "mutated"
    assert predictor.provenance == {}

    loaded = ConformalPredictor.load(first)
    view = loaded.provenance
    view["identity"]["dataset"] = "view-mutated"
    assert loaded.provenance["identity"]["dataset"] == "drivaer"
    loaded.save(second)
    assert ConformalPredictor.load(second).provenance == loaded.provenance


def test_non_json_provenance_fails_closed_without_replacing_artifact(tmp_path):
    predictor = fitted_risk()
    path = tmp_path / "artifact.pt"
    predictor.save(path, provenance={"note": "valid"})
    original_bytes = path.read_bytes()

    with pytest.raises(TypeError, match="provenance must be a mapping"):
        predictor.save(path, provenance=[1, 2, 3])
    assert path.read_bytes() == original_bytes
    assert ConformalPredictor.load(path).provenance == {"note": "valid"}


# fmt: off
STANDALONE_STRATEGIES = [  # (id, make_predictor overrides, aux, spec section, expected spec)
    ("absolute_error", {"tier": "functional"}, None, "score", {"kind": "absolute_error", "kwargs": {}}),
    ("normalized_error", {"tier": "functional", "score": NormalizedErrorScore(eps=1e-4)}, {"sigma": torch.full((3,), 0.7)}, "score", {"kind": "normalized_error", "kwargs": {"eps": 1e-4}}),
    ("quantile_regression", {"tier": "functional", "score": QuantileRegressionScore()}, {"lo": -torch.ones(3), "hi": torch.ones(3)}, "score", {"kind": "quantile_regression", "kwargs": {}}),
    ("aux_difficulty", {"difficulty": AuxDifficulty(key="spread", eps=1e-4)}, {"spread": torch.tensor([0.5, 0.7, 0.9])}, "difficulty", {"kind": "aux", "kwargs": {"key": "spread", "eps": 1e-4}}),
]
# fmt: on


@pytest.mark.parametrize(
    "overrides,aux,spec_name,expected_spec",
    [pytest.param(*row[1:], id=row[0]) for row in STANDALONE_STRATEGIES],
)
def test_all_builtin_strategies_are_standalone_artifacts(
    overrides, aux, spec_name, expected_spec, tmp_path
):
    predictor = make_predictor(**overrides)
    prediction = torch.zeros(3)
    lo_ref, hi_ref = predictor.predict_interval(prediction, aux=aux)
    path = tmp_path / "artifact.pt"
    predictor.save(path)
    assert torch.load(path, weights_only=True)[spec_name] == expected_spec
    lo, hi = ConformalPredictor.load(path).predict_interval(prediction, aux=aux)
    torch.testing.assert_close(lo, lo_ref)
    torch.testing.assert_close(hi, hi_ref)


BASES = {
    "risk": fitted_risk,
    "normalized": lambda: make_predictor(
        tier="functional", score=NormalizedErrorScore()
    ),
    "aux": lambda: make_predictor(difficulty=AuxDifficulty(key="spread", eps=1e-4)),
    "cellwise": lambda: make_predictor(
        tier="cellwise", thresholds=torch.ones(6, 3), points=_MESH_POINTS
    ),
}


def _set(key, value):
    return lambda payload: payload.__setitem__(key, value)


def _set_in(section, key, value):
    return lambda payload: payload[section].__setitem__(key, value)


def _delete(key):
    return lambda payload: payload.__delitem__(key)


_CTOR = "Invalid constructor arguments"
# fmt: off
# name: (mutator, load-error regex[, base predictor]); base defaults to "risk".
# A mutator may return a replacement payload, otherwise it edits in place.
CORRUPTIONS = {
    "not_an_artifact": (lambda p: {"weights": torch.ones(3)}, "[Nn]ot a conformal predictor artifact"),
    "unknown_version": (_set("version", 999), "version 999"),
    "infeasible_alpha": (_set("alpha", 0.01), "calibration samples"),
    "boolean_n_cal": (_set("n_cal", True), "n_cal must be an integer"),
    "negative_radius": (_set_in("thresholds", "__tensor__", torch.tensor(-1.0, dtype=torch.float64)), "negative threshold"),
    "empty_thresholds": (_set("thresholds", {}), "at least one field"),
    "named_key_beside_sentinel": (_set_in("thresholds", "pressure", torch.tensor(1.0)), "reserved field name"),
    "non_string_threshold_key": (_set_in("thresholds", 7, torch.tensor(1.0)), "keys must be strings"),
    "missing_difficulty": (_delete("difficulty"), "schema is invalid: missing"),
    "extra_top_level_key": (_set("duplicate_state", {}), "schema is invalid: unexpected"),
    "unknown_score_kind": (_set_in("score", "kind", "bogus"), "Unknown built-in score kind"),
    "extra_score_key": (_set_in("score", "class", "AbsoluteErrorScore"), "score spec schema is invalid"),
    "unknown_difficulty_kind": (_set("difficulty", {"kind": "bogus", "kwargs": {}}), "Unknown built-in difficulty kind"),
    "extra_difficulty_key": (_set("difficulty", {"kind": "aux", "kwargs": {"key": "s", "eps": 1e-8}, "class": "X"}), "difficulty spec schema is invalid"),
    "non_mapping_difficulty": (_set("difficulty", "aux"), "difficulty spec schema is invalid"),
    "non_json_provenance": (_set("provenance", {"tags": ("x",)}), "strict-JSON"),
    # Strategy kwargs: exact key sets and exact wire types.
    "aux_missing_key": (_set_in("difficulty", "kwargs", {"eps": 1e-4}), "kwargs schema is invalid", "aux"),
    "aux_string_eps": (_set_in("difficulty", "kwargs", {"key": "spread", "eps": "0.1"}), "exact type", "aux"),
    # Wire-valid (exact float) but semantically invalid eps is still rejected at load.
    "normalized_negative_eps": (_set_in("score", "kwargs", {"eps": -1.0}), _CTOR, "normalized"),
    "aux_negative_eps": (_set_in("difficulty", "kwargs", {"key": "spread", "eps": -1.0}), _CTOR, "aux"),
    # Cellwise artifacts need a well-formed mesh fingerprint.
    "cellwise_malformed_mesh": (_set("mesh_fingerprint", "not-a-digest"), "mesh_fingerprint must be", "cellwise"),
    "cellwise_uppercase_mesh": (_set("mesh_fingerprint", "float32-3x1-" + "A" * 32), "mesh_fingerprint must be", "cellwise"),
    "cellwise_sha256_mesh": (_set("mesh_fingerprint", "a" * 64), "mesh_fingerprint must be", "cellwise"),
}
# fmt: on


@pytest.mark.parametrize("corruption", sorted(CORRUPTIONS))
def test_corrupted_artifacts_fail_to_load_with_a_named_reason(tmp_path, corruption):
    mutate, match, *base = CORRUPTIONS[corruption]
    path = tmp_path / "artifact.pt"
    BASES[base[0] if base else "risk"]().save(path)
    payload = torch.load(path, weights_only=True)
    payload = mutate(payload) or payload
    torch.save(payload, path)
    with pytest.raises(ValueError, match=match):
        ConformalPredictor.load(path)


_PICKLE_SIDE_EFFECTS = []


class _Exploit:
    def __reduce__(self):
        return (_PICKLE_SIDE_EFFECTS.append, ("executed",))


def test_load_refuses_pickled_code(tmp_path):
    path = tmp_path / "artifact.pt"
    torch.save({"format": "physicsnemo.uq.conformal", "payload": _Exploit()}, path)
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        ConformalPredictor.load(path)
    assert _PICKLE_SIDE_EFFECTS == []


def test_atomic_save_preserves_previous_artifact(tmp_path, monkeypatch):
    predictor = fitted_risk()
    path = tmp_path / "artifact.pt"
    predictor.save(path, provenance={"epoch": 1})
    original_bytes = path.read_bytes()

    def exploding_save(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", exploding_save)
    with pytest.raises(RuntimeError, match="disk full"):
        predictor.save(path, provenance={"epoch": 2})
    monkeypatch.undo()
    assert path.read_bytes() == original_bytes
    assert ConformalPredictor.load(path).provenance == {"epoch": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_save_semantically_validates_before_replace(tmp_path):
    path = tmp_path / "artifact.pt"
    fitted_risk().save(path, provenance={"epoch": 1})
    original_bytes = path.read_bytes()

    invalid = fitted_risk(torch.Generator().manual_seed(19))
    invalid._thresholds_by_key["__tensor__"].fill_(float("inf"))
    with pytest.raises(ValueError, match=f"at {re.escape(str(path))}: .*non-finite"):
        invalid.save(path, provenance={"epoch": 2})
    assert path.read_bytes() == original_bytes
    assert ConformalPredictor.load(path).provenance == {"epoch": 1}
