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

"""Timeout configuration and real NCCL initialization for DistributedManager."""

import os
import socket
from datetime import timedelta

import numpy as np
import pytest
import torch
import torch.distributed as dist

from physicsnemo.distributed import DistributedManager

_TIMEOUT_ENV = "PHYSICSNEMO_DIST_TIMEOUT_S"
_LAUNCHER_ENV = {
    "ENV": {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"},
    "SLURM": {
        "SLURM_PROCID": "0",
        "SLURM_NPROCS": "2",
        "SLURM_LOCALID": "0",
        "SLURM_LAUNCH_NODE_IPADDR": "localhost",
    },
    "OPENMPI": {
        "OMPI_COMM_WORLD_RANK": "0",
        "OMPI_COMM_WORLD_SIZE": "2",
        "OMPI_COMM_WORLD_LOCAL_RANK": "0",
    },
}


@pytest.fixture(autouse=True)
def _isolated_manager_environment(monkeypatch):
    """Isolate launcher variables, singleton state, and initialization side effects."""
    keys = {key for values in _LAUNCHER_ENV.values() for key in values} | {
        _TIMEOUT_ENV,
        "MASTER_ADDR",
        "MASTER_PORT",
        "PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD",
        "NCCL_ASYNC_ERROR_HANDLING",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "TORCHINDUCTOR_CACHE_DIR",
    }
    for key in keys:
        # Track originally absent variables too, since setup writes some of them.
        monkeypatch.setenv(key, os.environ.get(key, ""))
        monkeypatch.delenv(key)
    monkeypatch.setattr(DistributedManager, "_shared_state", {})
    random_state = np.random.get_state()
    yield
    np.random.set_state(random_state)


@pytest.fixture
def mock_process_group(monkeypatch):
    """Record initialization without requiring a GPU, network, or process group."""
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(
        DistributedManager, "_isolate_torch_compile_cache", lambda *args: None
    )
    monkeypatch.setattr(
        dist, "init_process_group", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    return calls


def _configure_launcher(monkeypatch, launcher, forced=False):
    """Select one launcher through either auto-detection or its explicit override."""
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "12399")
    for key, value in _LAUNCHER_ENV[launcher].items():
        monkeypatch.setenv(key, value)
    if forced:
        monkeypatch.setenv("PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD", launcher)


@pytest.mark.parametrize("launcher", ["ENV", "SLURM", "OPENMPI"])
@pytest.mark.parametrize("forced", [False, True], ids=["detected", "forced"])
@pytest.mark.parametrize("legacy_signature", [False, True], ids=["current", "legacy"])
@pytest.mark.parametrize(
    ("argument", "environment", "expected"),
    [
        (None, None, None),
        (None, "", None),
        (None, "90.25", timedelta(seconds=90.25)),
        (60.5, None, timedelta(seconds=60.5)),
        (60, "invalid-but-overridden", timedelta(seconds=60)),
        (timedelta(seconds=15.5), "90", timedelta(seconds=15.5)),
    ],
)
def test_launcher_forwards_effective_timeout(
    monkeypatch,
    mock_process_group,
    launcher,
    forced,
    legacy_signature,
    argument,
    environment,
    expected,
):
    """All launchers and the old-PyTorch retry retain the chosen timeout."""
    _configure_launcher(monkeypatch, launcher, forced)
    if environment is not None:
        monkeypatch.setenv(_TIMEOUT_ENV, environment)
    if legacy_signature:

        def legacy_init(*args, **kwargs):
            """Model PyTorch releases that lack the device_id keyword."""
            mock_process_group.append((args, kwargs))
            if "device_id" in kwargs:
                raise TypeError("unexpected keyword argument 'device_id'")

        monkeypatch.setattr(dist, "init_process_group", legacy_init)

    DistributedManager.initialize(timeout=argument)

    assert DistributedManager.is_initialized()
    assert len(mock_process_group) == (2 if legacy_signature else 1)
    for args, kwargs in mock_process_group:
        assert args == ("gloo",)
        assert kwargs["timeout"] == expected
        assert kwargs["rank"] == 0
        assert kwargs["world_size"] == 2
    if legacy_signature:
        assert "device_id" not in mock_process_group[-1][1]


@pytest.mark.parametrize(
    ("argument", "environment", "expected"),
    [
        (None, "90.25", timedelta(seconds=90.25)),
        (30, "invalid-but-overridden", timedelta(seconds=30)),
        (timedelta(seconds=45), None, timedelta(seconds=45)),
        (None, None, None),
    ],
)
def test_direct_setup_accepts_timeout(
    monkeypatch, mock_process_group, argument, environment, expected
):
    """Direct setup preserves the same explicit, environment, and default choices."""
    if environment is not None:
        monkeypatch.setenv(_TIMEOUT_ENV, environment)
    DistributedManager.setup(
        rank=0, world_size=2, local_rank=0, backend="gloo", timeout=argument
    )
    assert mock_process_group[0][1]["timeout"] == expected


@pytest.mark.parametrize("entrypoint", ["initialize", "setup"])
@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        "60",
        object(),
        0,
        -1,
        float("nan"),
        float("inf"),
        -float("inf"),
        1e20,
        10**400,
        1e-9,
        timedelta(0),
        timedelta(seconds=-1),
    ],
)
def test_invalid_api_timeout_leaves_manager_uninitialized(
    monkeypatch, mock_process_group, entrypoint, value
):
    """Invalid public values fail before launcher fallback or singleton mutation."""
    _configure_launcher(monkeypatch, "ENV")
    method = getattr(DistributedManager, entrypoint)
    with pytest.raises((TypeError, ValueError), match="timeout"):
        method(timeout=value)
    assert not DistributedManager.is_initialized()
    assert not mock_process_group

    if entrypoint == "setup":
        method(local_rank=0, backend="gloo", timeout=30)
    else:
        method(timeout=30)
    assert DistributedManager.is_initialized()
    assert mock_process_group[0][1]["timeout"] == timedelta(seconds=30)


@pytest.mark.parametrize("with_launcher", [False, True], ids=["serial", "distributed"])
@pytest.mark.parametrize("value", ["bad", " ", "0", "-1", "nan", "inf", "1e20", "1e-9"])
def test_invalid_environment_can_be_corrected_and_retried(
    monkeypatch, mock_process_group, with_launcher, value
):
    """Invalid environment values cannot become successful serial fallback."""
    if with_launcher:
        _configure_launcher(monkeypatch, "ENV")
    monkeypatch.setenv(_TIMEOUT_ENV, value)
    with pytest.raises(ValueError, match="PHYSICSNEMO_DIST_TIMEOUT_S"):
        DistributedManager.initialize()
    assert not DistributedManager.is_initialized()
    assert not mock_process_group

    monkeypatch.setenv(_TIMEOUT_ENV, "30.25")
    if with_launcher:
        DistributedManager.initialize()
        assert mock_process_group[0][1]["timeout"] == timedelta(seconds=30.25)
    else:
        with pytest.warns(UserWarning, match="single process"):
            DistributedManager.initialize()
        assert not mock_process_group
    assert DistributedManager.is_initialized()


def _nccl_timeout_worker(rank, port, source):
    """Initialize two real CUDA ranks and exercise their configured process group."""
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD="ENV",
    )
    DistributedManager._shared_state = {}
    os.environ.pop(_TIMEOUT_ENV, None)
    argument = None
    if source in ("environment", "precedence"):
        os.environ[_TIMEOUT_ENV] = "120.25" if source == "environment" else "180"
    if source in ("argument", "precedence"):
        argument = 120.25

    real_init = dist.init_process_group
    observed = []

    def recording_init(*args, **kwargs):
        """Observe the timeout while preserving real NCCL initialization."""
        observed.append(kwargs.get("timeout"))
        return real_init(*args, **kwargs)

    dist.init_process_group = recording_init
    try:
        DistributedManager.initialize(timeout=argument)
        manager = DistributedManager()
        assert observed[-1] == timedelta(seconds=120.25)
        assert dist.get_backend() == "nccl"
        assert manager.world_size == 2
        assert manager.rank == rank
        value = torch.tensor(float(rank + 1), device=manager.device)
        dist.all_reduce(value)
        assert value.item() == 3.0
    finally:
        dist.init_process_group = real_init
        if dist.is_initialized():
            DistributedManager.cleanup()


@pytest.mark.multigpu_dynamic
@pytest.mark.parametrize("source", ["argument", "environment", "precedence"])
def test_timeout_initializes_real_nccl_process_group(source):
    """The configured timeout reaches a working two-GPU NCCL process group."""
    assert torch.cuda.device_count() >= 2, "Two GPUs are required for this test"
    assert dist.is_nccl_available(), "NCCL is required for this test"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    torch.multiprocessing.spawn(
        _nccl_timeout_worker, args=(port, source), nprocs=2, join=True
    )
