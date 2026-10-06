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

from pathlib import Path

import pytest

from test.conftest import requires_module


@pytest.fixture
def offline_mlflow(tmp_path, monkeypatch):
    """Run in an empty directory with no file-store opt-out, and reset the
    launch logger's MLflow state afterwards."""
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.utils.logging import LaunchLogger

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    DistributedManager.initialize()
    yield tmp_path
    LaunchLogger.mlflow_run = None
    LaunchLogger.mlflow_client = None
    LaunchLogger.mlflow_backend = False


@requires_module("mlflow")
def test_offline_default_uses_sqlite_store(offline_mlflow):
    """The default offline location must work with MLflow versions that refuse
    the file-system tracking store, keeping everything under ./mlruns."""
    import mlflow

    from physicsnemo.utils.logging.mlflow import initialize_mlflow

    client, run = initialize_mlflow(experiment_name="test", mode="offline")

    mlruns = offline_mlflow / "mlruns"
    assert mlflow.get_tracking_uri() == f"sqlite:///{mlruns}/mlflow.db"
    assert (mlruns / "mlflow.db").is_file()

    # metrics land in the database and artifacts under ./mlruns/artifacts
    client.log_metric(run.info.run_id, "loss", 0.5)
    artifact = offline_mlflow / "note.txt"
    artifact.write_text("hello")
    client.log_artifact(run.info.run_id, str(artifact))
    assert client.get_run(run.info.run_id).data.metrics["loss"] == 0.5
    assert Path(run.info.artifact_uri.removeprefix("file://")).is_relative_to(
        mlruns / "artifacts"
    )
    assert list((mlruns / "artifacts").rglob("note.txt"))


@requires_module("mlflow")
def test_offline_reuses_existing_experiment(offline_mlflow):
    """A second run in the same directory must find the experiment in the
    database rather than create a duplicate."""
    from physicsnemo.utils.logging.mlflow import initialize_mlflow

    _, first = initialize_mlflow(experiment_name="test", mode="offline")
    client, second = initialize_mlflow(experiment_name="test", mode="offline")

    assert first.info.experiment_id == second.info.experiment_id
    assert len(client.search_experiments(filter_string="name = 'test'")) == 1


@requires_module("mlflow")
def test_offline_custom_directory(offline_mlflow):
    """A plain directory as ``tracking_location`` holds the database and
    artifacts, and is created if missing."""
    import mlflow

    from physicsnemo.utils.logging.mlflow import initialize_mlflow

    location = offline_mlflow / "nested" / "tracking"
    initialize_mlflow(
        experiment_name="test", mode="offline", tracking_location=str(location)
    )

    assert mlflow.get_tracking_uri() == f"sqlite:///{location}/mlflow.db"
    assert (location / "mlflow.db").is_file()


@requires_module("mlflow")
@pytest.mark.parametrize("scheme", ["sqlite", "file"])
def test_offline_explicit_uri_is_respected(offline_mlflow, monkeypatch, scheme):
    """An explicit ``sqlite:`` or ``file://`` URI is passed through unchanged."""
    import mlflow

    from physicsnemo.utils.logging.mlflow import initialize_mlflow

    if scheme == "sqlite":
        uri = f"sqlite:///{offline_mlflow}/explicit.db"
    else:
        # choosing the file store explicitly also means opting in to it
        monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
        uri = f"file://{offline_mlflow}/explicit_store"
    initialize_mlflow(experiment_name="test", mode="offline", tracking_location=uri)

    assert mlflow.get_tracking_uri() == uri


@requires_module("mlflow")
def test_launch_logger_logs_epoch_metrics(offline_mlflow):
    """End to end as the examples use it: LaunchLogger epoch metrics reach the
    MLflow run."""
    from physicsnemo.utils.logging import LaunchLogger
    from physicsnemo.utils.logging.mlflow import initialize_mlflow

    client, run = initialize_mlflow(experiment_name="test", mode="offline")
    LaunchLogger.initialize(use_mlflow=True)
    with LaunchLogger("train", epoch=1) as logger:
        logger.log_epoch({"loss": 0.25})

    history = client.get_metric_history(run.info.run_id, "train/loss")
    assert [m.value for m in history] == [0.25]
