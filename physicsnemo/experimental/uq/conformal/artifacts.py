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

r"""Encode and decode conformal predictor state as ``weights_only``-safe files."""

import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

import torch
from tensordict import TensorDict
from torch import Tensor

from ._utils import pack_fields, validate_provenance
from .scores import (
    _SCALE_REGISTRY,
    _SCORE_REGISTRY,
    _check_config_value,
    _strategy_kind,
)

_ARTIFACT_FORMAT = "physicsnemo.uq.conformal"
_ARTIFACT_VERSION = 1
_SCHEMA_KEYS = {
    "format",
    "version",
    "tier",
    "score",
    "alpha",
    "n_cal",
    "thresholds",
    "scale",
    "mesh_fingerprint",
    "weighted",
    "provenance",
}


def _check_exact_keys(value: object, expected: set[str], where: str) -> Mapping:
    """Require a mapping with exactly the ``expected`` keys."""
    keys = set(value)
    missing = expected - keys
    unexpected = keys - expected
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)!r}")
        if unexpected:
            details.append(f"unexpected {sorted(map(repr, unexpected))!r}")
        raise ValueError(f"{where} schema is invalid: {', '.join(details)}.")
    return value


def _strategy_spec(strategy: object, registry: Mapping) -> dict:
    """Encode a registered score or scale as ``{"kind", "kwargs"}``."""
    return {
        "kind": _strategy_kind(strategy, registry),
        "kwargs": {
            name: getattr(strategy, name) for name in type(strategy).config_fields
        },
    }


def _resolve_strategy(spec: object, registry: Mapping, what: str):
    """Rebuild a registered score or scale from its saved spec."""
    spec = _check_exact_keys(spec, {"kind", "kwargs"}, f"Artifact {what} spec")
    kind = spec["kind"]
    if type(kind) is not str or kind not in registry:
        raise ValueError(
            f"Unknown {what} kind {kind!r}; registered kinds are "
            f"{sorted(registry)!r}. If it is a custom {what}, import the module "
            "that registers it before loading."
        )
    cls = registry[kind]
    kwargs = _check_exact_keys(
        spec["kwargs"], set(cls.config_fields), f"Artifact {what} {kind!r} kwargs"
    )
    for name, value in kwargs.items():
        _check_config_value(f"Artifact {what} {kind!r}", name, value)
    try:
        strategy = cls(**dict(kwargs))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid constructor arguments for {what} {kind!r}: {exc}"
        ) from exc
    # The rebuilt object must hold exactly the saved values, so a corrupted
    # file cannot load as a differently configured strategy.
    for name, value in kwargs.items():
        rebuilt = getattr(strategy, name)
        if type(rebuilt) is not type(value) or rebuilt != value:
            raise ValueError(
                f"Artifact {what} {kind!r} kwarg {name!r} is {value!r}, but the "
                f"constructor stores {rebuilt!r}."
            )
    return strategy


def _wire_thresholds(value: object) -> Tensor | TensorDict:
    """Check the saved threshold mapping and repack it."""
    if not isinstance(value, Mapping):
        raise ValueError("Artifact thresholds must be a mapping.")
    for key, threshold in value.items():
        if type(key) is not str:
            raise TypeError(f"Artifact threshold keys must be strings, got {key!r}.")
        if not isinstance(threshold, Tensor):
            raise TypeError(
                f"Artifact threshold {key!r} must be a torch.Tensor, got "
                f"{type(threshold).__name__}."
            )
    return pack_fields(dict(value))


def _parse_artifact(payload: object) -> dict:
    """Check a loaded payload against the artifact schema; return predictor state."""
    marker = payload.get("format") if isinstance(payload, Mapping) else type(payload)
    if type(marker) is not str or marker != _ARTIFACT_FORMAT:
        raise ValueError(
            "Not a conformal predictor artifact (format marker missing or "
            f"unknown: {marker!r})."
        )
    version = payload.get("version")
    if type(version) is not int or version != _ARTIFACT_VERSION:
        raise ValueError(
            f"Artifact version {version!r} was written by an incompatible "
            f"physicsnemo version (this one reads version {_ARTIFACT_VERSION}). "
            "Recalibrate and save again."
        )
    payload = _check_exact_keys(payload, _SCHEMA_KEYS, "Conformal artifact")
    score = _resolve_strategy(payload["score"], _SCORE_REGISTRY, "score")
    scale = (
        None
        if payload["scale"] is None
        else _resolve_strategy(
            payload["scale"], _SCALE_REGISTRY, "scale"
        )
    )
    if type(payload["weighted"]) is not bool:
        raise TypeError(
            f"Artifact weighted must be a bool, got {payload['weighted']!r}."
        )
    return {
        "tier": payload["tier"],
        "score": score,
        "alpha": payload["alpha"],
        "n_cal": payload["n_cal"],
        "thresholds": _wire_thresholds(payload["thresholds"]),
        "scale": scale,
        "mesh_fingerprint": payload["mesh_fingerprint"],
        "weighted": payload["weighted"],
        "provenance": payload["provenance"],
    }


def _artifact_payload(state: Mapping) -> dict:
    """Encode predictor state, thresholds keyed by field, for ``torch.save``."""
    scale = state["scale"]
    return {
        "format": _ARTIFACT_FORMAT,
        "version": _ARTIFACT_VERSION,
        "tier": state["tier"],
        "score": _strategy_spec(state["score"], _SCORE_REGISTRY),
        "alpha": state["alpha"],
        "n_cal": state["n_cal"],
        "thresholds": {
            key: value.detach().cpu() for key, value in state["thresholds"].items()
        },
        "scale": (
            None
            if scale is None
            else _strategy_spec(scale, _SCALE_REGISTRY)
        ),
        "mesh_fingerprint": state["mesh_fingerprint"],
        "weighted": state["weighted"],
        "provenance": validate_provenance(state["provenance"]),
    }


def _save_artifact(
    state: Mapping, path: Path | str, verify: Callable[[str], object]
) -> None:
    """Write the artifact atomically after ``verify`` accepts the written file."""
    payload = _artifact_payload(state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
        # Free the payload first so saving keeps at most one extra threshold copy.
        del payload
        verify(temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
