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

"""pytest configuration of the ISLA tests.

The tests marked ``slow`` (mostly ``torch.compile`` runs, tens of seconds each on a
CPU) are skipped unless ``ISLA_RUN_SLOW_TESTS=1`` is set or the run selects them with
``-m slow``. Elsewhere in the repository a ``slow`` test runs unless it is
deselected; here it is the other way round, because the repository's CI runs every
test it collects and these take about as long as the rest of the directory together.
Run every ISLA test with::

    ISLA_RUN_SLOW_TESTS=1 pytest test/experimental/models/isla

The CUDA tests carry the ``cuda`` marker besides their skip (``_marks.py``'s
``requires_cuda`` or ``requires_cuda_triton``): the marker alone skips nothing.
``test/conftest.py`` registers both markers for the whole repository; they are
registered here too so that the ISLA directory stays self-contained (run from its own
root, or copied with the model)."""

import os
from pathlib import Path

import pytest

_HERE = Path(__file__).parent


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running (for optional exclusion)"
    )
    config.addinivalue_line(
        "markers", "cuda: mark test as requiring CUDA (skipped if unavailable)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip this directory's ``slow`` tests unless ``ISLA_RUN_SLOW_TESTS=1`` is set or
    the run's ``-m`` expression selects ``slow``."""
    if os.environ.get("ISLA_RUN_SLOW_TESTS") == "1" or "slow" in (
        config.getoption("-m") or ""
    ).replace("not slow", ""):
        return
    skip = pytest.mark.skip(
        reason="slow ISLA test: set ISLA_RUN_SLOW_TESTS=1 or pass -m slow to run it"
    )
    for item in items:
        if "slow" in item.keywords and _HERE in Path(item.fspath).parents:
            item.add_marker(skip)
