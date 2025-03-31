# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import os
from datetime import datetime
from pathlib import Path
from typing import Literal, Tuple

import torch

try:
    import mlflow  # noqa: F401 for docs
    from mlflow.entities.run import Run
    from mlflow.tracking import MlflowClient
except ImportError:
    raise ImportError(
        "These utilities require the MLFlow library. Install MLFlow using `pip install mlflow`. "
        + "For more info, refer: https://www.mlflow.org/docs/2.5.0/quickstart.html#install-mlflow"
    )

from physicsnemo.distributed import DistributedManager

from .console import PythonLogger
from .launch import LaunchLogger

logger = PythonLogger("mlflow")


def initialize_mlflow(
    experiment_name: str,
    experiment_desc: str = None,
    run_name: str = None,
    run_desc: str = None,
    user_name: str = None,
    mode: Literal["offline", "online", "ngc"] = "offline",
    tracking_location: str = None,
    artifact_location: str = None,
) -> Tuple[MlflowClient, Run]:
    """Initializes MLFlow logging client and run.

    Parameters
    ----------
    experiment_name : str
        Experiment name
    experiment_desc : str, optional
        Experiment description, by default None
    run_name : str, optional
        Run name, by default None
    run_desc : str, optional
        Run description, by default None
    user_name : str, optional
        User name, by default None
    mode : str, optional
        MLFlow mode. Supports "offline", "online" and "ngc". Offline mode records logs to
        local file system. Online mode is for remote tracking servers. NGC is specific
        standardized setup for NGC runs, default "offline"
    tracking_location : str, optional
        Tracking location for MLFlow. For offline this would be an absolute folder directory.
        For online mode this would be a http URI or databricks. For NGC, this option is
        ignored, by default "/<run directory>/mlruns"
    artifact_location : str, optional
        Optional separate artifact location, by default None

    Note
    ----
    For NGC mode, one needs to mount a NGC workspace / folder system with a metric folder
    at `/mlflow/mlflow_metrics/` and a artifact folder at `/mlflow/mlflow_artifacts/`.

    Note
    ----
    This will set up PhysicsNeMo Launch logger for MLFlow logging. Only one MLFlow logging
    client is supported with the PhysicsNeMo Launch logger.

    Returns
    -------
    Tuple[MlflowClient, Run]
        Returns MLFlow logging client and active run object
    """
    dist = DistributedManager()
    if dist.rank != 0:  # only root process should be logging to mlflow
        return

    start_time = datetime.now().astimezone()
    time_string = start_time.strftime("%m/%d/%y_%H-%M-%S")
    group_name = f"{run_name}_{time_string}"

    # Set default value here for Hydra
    if tracking_location is None:
        tracking_location = str(Path("./mlruns").absolute())

    # Set up URI (remote or local)
    if mode == "online":
        tracking_uri = tracking_location
    elif mode == "offline":
        if not tracking_location.startswith("file://"):
            tracking_location = "file://" + tracking_location
        tracking_uri = tracking_location
    elif mode == "ngc":
        if not Path("/mlflow/mlflow_metrics").is_dir():
            raise IOError(
                "NGC MLFlow config select but metrics folder '/mlflow/mlflow_metrics'"
                + " not found. Aborting MLFlow setup."
            )
            return

        if not Path("/mlflow/mlflow_artifacts").is_dir():
            raise IOError(
                "NGC MLFlow config select but artifact folder '/mlflow/mlflow_artifacts'"
                + " not found. Aborting MLFlow setup."
            )
            return
        tracking_uri = "file:///mlflow/mlflow_metrics"
        artifact_location = "file:///mlflow/mlflow_artifacts"
    else:
        logger.warning(f"Unsupported MLFlow mode '{mode}' provided")
        tracking_uri = "file://" + str(Path("./mlruns").absolute())

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    check_mlflow_logged_in(client)

    experiment = client.get_experiment_by_name(experiment_name)
    # If experiment does not exist create one
    if experiment is None:
        logger.info(f"No {experiment_name} experiment found, creating...")
        experiment_id = client.create_experiment(
            experiment_name, artifact_location=artifact_location
        )
        client.set_experiment_tag(experiment_id, "mlflow.note.content", experiment_desc)
    else:
        logger.success(f"Existing {experiment_name} experiment found")
        experiment_id = experiment.experiment_id

    # Create an run and set its tags
    run = client.create_run(
        experiment_id, tags={"mlflow.user": user_name}, run_name=run_name
    )
    client.set_tag(run.info.run_id, "mlflow.note.content", run_desc)

    start_time = datetime.now().astimezone()
    time_string = start_time.strftime("%m/%d/%y %H:%M:%S")
    client.set_tag(run.info.run_id, "date", time_string)
    client.set_tag(run.info.run_id, "host", os.uname()[1])
    if torch.cuda.is_available():
        client.set_tag(run.info.run_id, "gpu", torch.cuda.get_device_name(dist.device))
    client.set_tag(run.info.run_id, "group", group_name)

    run = client.get_run(run.info.run_id)

    # Set run instance in PhysicsNeMo logger
    LaunchLogger.mlflow_run = run
    LaunchLogger.mlflow_client = client

    return client, run


def check_mlflow_logged_in(client: MlflowClient):
    """Checks to see if MLFlow URI is functioning

    This isn't the best solution right now and overrides http timeout. Can update if MLFlow
    use is increased.
    """

    logger.warning(
        "Checking MLFlow logging location is working (if this hangs it's not)"
    )
    t0 = os.environ.get("MLFLOW_HTTP_REQUEST_TIMEOUT", None)
    try:
        # Adjust http timeout to 5 seconds
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(max(int(t0), 5)) if t0 else "5"
        experiment = client.create_experiment("test")
        client.delete_experiment(experiment)

    except Exception as e:
        logger.error("Failed to validate MLFlow logging location works")
        raise e
    finally:
        # Restore http request
        if t0:
            os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = t0
        else:
            del os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"]

    logger.success("MLFlow logging location is working")
