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

import importlib
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Tuple

import torch

from physicsnemo.core.version_check import check_version_spec
from physicsnemo.distributed import DistributedManager

from .console import PythonLogger
from .launch import LaunchLogger

MLFLOW_AVAILABLE = check_version_spec("mlflow", "2.5.0", hard_fail=False)


logger = PythonLogger("mlflow")


def _offline_tracking(
    tracking_location: str, artifact_location: str | None
) -> Tuple[str, str | None]:
    """Tracking URI and artifact location for offline (local) MLflow logging.

    A folder becomes a SQLite database ``<folder>/mlflow.db`` with artifacts in
    ``<folder>/artifacts``; recent MLflow versions refuse the file-system
    tracking store by default. Explicit ``sqlite:`` and ``file://`` URIs are
    returned unchanged.

    Parameters
    ----------
    tracking_location : str
        Folder or URI to track runs in
    artifact_location : str | None
        Artifact location requested by the caller, if any

    Returns
    -------
    Tuple[str, str | None]
        Tracking URI and artifact location
    """
    if tracking_location.startswith(("sqlite:", "file://")):
        return tracking_location, artifact_location
    folder = Path(tracking_location).absolute()
    folder.mkdir(parents=True, exist_ok=True)
    if artifact_location is None:
        artifact_location = (folder / "artifacts").as_uri()
    return f"sqlite:///{folder}/mlflow.db", artifact_location


if MLFLOW_AVAILABLE:
    mlflow = importlib.import_module("mlflow")
    Run = importlib.import_module("mlflow.entities.run").Run
    MlflowClient = importlib.import_module("mlflow.tracking").MlflowClient

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
            a local SQLite database. Online mode is for remote tracking servers. NGC is specific
            standardized setup for NGC runs, default "offline"
        tracking_location : str, optional
            Tracking location for MLFlow. For offline this is a folder that holds the
            ``mlflow.db`` SQLite database and an ``artifacts`` folder, or an explicit
            ``sqlite:`` or ``file://`` URI, which is used as given. For online mode this
            would be a http URI or databricks. For NGC, this option is ignored, by default
            "/<run directory>/mlruns"
        artifact_location : str, optional
            Optional separate artifact location, by default None. For an offline folder
            location this defaults to ``<tracking_location>/artifacts``.

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
            tracking_uri, artifact_location = _offline_tracking(
                tracking_location, artifact_location
            )
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
            tracking_uri, artifact_location = _offline_tracking(
                str(Path("./mlruns").absolute()), artifact_location
            )

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
            client.set_experiment_tag(
                experiment_id, "mlflow.note.content", experiment_desc
            )
        else:
            logger.success(f"Existing {experiment_name} experiment found")
            experiment_id = experiment.experiment_id

        # Create an run and set its tags
        # MLflow rejects a None user
        tags = {"mlflow.user": user_name} if user_name is not None else {}
        run = client.create_run(experiment_id, tags=tags, run_name=run_name)
        client.set_tag(run.info.run_id, "mlflow.note.content", run_desc)

        start_time = datetime.now().astimezone()
        time_string = start_time.strftime("%m/%d/%y %H:%M:%S")
        client.set_tag(run.info.run_id, "date", time_string)
        client.set_tag(run.info.run_id, "host", os.uname()[1])
        if torch.cuda.is_available():
            client.set_tag(
                run.info.run_id, "gpu", torch.cuda.get_device_name(dist.device)
            )
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
            os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = (
                str(max(int(t0), 5)) if t0 else "5"
            )
            # unique name: deleted experiments keep their name reserved, so a
            # timestamp collides when two runs start within the same second
            experiment = client.create_experiment(f"test-{uuid.uuid4().hex}")
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

else:

    def initialize_mlflow(
        *args,
        **kwargs,
    ):
        """Stand-in for ``initialize_mlflow`` when MLflow is not installed.

        Raises
        ------
        ImportError
            Always; install MLflow to use this utility
        """
        raise ImportError(
            "These utilities require the MLFlow library. Install MLFlow using `pip install mlflow`. "
            + "For more info, refer: https://www.mlflow.org/docs/2.5.0/quickstart.html#install-mlflow"
        )

    def check_mlflow_logged_in(*args, **kwargs):
        """Stand-in for ``check_mlflow_logged_in`` when MLflow is not installed.

        Raises
        ------
        ImportError
            Always; install MLflow to use this utility
        """
        raise ImportError(
            "These utilities require the MLFlow library. Install MLFlow using `pip install mlflow`. "
            + "For more info, refer: https://www.mlflow.org/docs/2.5.0/quickstart.html#install-mlflow"
        )
