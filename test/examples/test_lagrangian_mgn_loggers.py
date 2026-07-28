# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _load_loggers_module(monkeypatch):
    pytest.importorskip("termcolor")
    if importlib.util.find_spec("wandb") is None:
        monkeypatch.setitem(sys.modules, "wandb", types.ModuleType("wandb"))
    module_path = (
        Path(__file__).parents[2] / "examples" / "cfd" / "lagrangian_mgn" / "loggers.py"
    )
    spec = importlib.util.spec_from_file_location("lagrangian_mgn_loggers", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_summary_redacts_credentials(monkeypatch):
    loggers = _load_loggers_module(monkeypatch)
    config = OmegaConf.create(
        {
            "model": {"name": "meshgraphnet"},
            "logging": {
                "api_key": "not-for-logs",
                "nested": [{"access_token": "also-not-for-logs"}],
            },
        }
    )

    summary = loggers.config_summary(config)

    assert "meshgraphnet" in summary
    assert "not-for-logs" not in summary
    assert "also-not-for-logs" not in summary
    assert summary.count("<redacted>") == 2


def test_wandb_logger_preserves_configured_key_and_run_id(monkeypatch, tmp_path):
    loggers = _load_loggers_module(monkeypatch)
    login_calls = []
    init_calls = []

    monkeypatch.setattr(
        loggers,
        "DistributedManager",
        lambda: types.SimpleNamespace(rank=0),
    )
    monkeypatch.setattr(
        loggers.wandb,
        "login",
        lambda **kwargs: login_calls.append(kwargs),
        raising=False,
    )
    monkeypatch.setattr(
        loggers.wandb,
        "init",
        lambda **kwargs: init_calls.append(kwargs),
        raising=False,
    )

    loggers.WandBLogger(
        wandb_key="configured-key",
        id="existing-run",
        dir=str(tmp_path),
        project="test-project",
    )

    assert login_calls == [{"key": "configured-key"}]
    assert init_calls[0]["id"] == "existing-run"
