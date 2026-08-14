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

"""Configuration tests for AeroJEPA's optional Transformer Engine backend."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

_CONF_DIR = Path(__file__).parents[1] / "conf"


def _component_use_te_values(cfg) -> tuple[bool, bool, bool, bool]:
    return (
        bool(cfg.model.trunk.context_encoder.use_te),
        bool(cfg.model.trunk.target_encoder.use_te),
        bool(cfg.model.trunk.decoder.use_te),
        bool(cfg.model.predictor.use_te),
    )


@pytest.mark.parametrize("config_name", ["config", "config_paper", "config_inference"])
def test_transformer_engine_is_disabled_by_default(config_name):
    """Every supported entry point defaults all model components to PyTorch."""
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        cfg = compose(config_name=config_name)

    assert _component_use_te_values(cfg) == (False, False, False, False)


@pytest.mark.parametrize("config_name", ["config", "config_paper", "config_inference"])
def test_transformer_engine_override_propagates(config_name):
    """One top-level override enables TE consistently across the whole model."""
    with initialize_config_dir(version_base=None, config_dir=str(_CONF_DIR)):
        cfg = compose(config_name=config_name, overrides=["use_te=true"])

    assert _component_use_te_values(cfg) == (True, True, True, True)
