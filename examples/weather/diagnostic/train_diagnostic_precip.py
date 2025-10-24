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

import hydra
from omegaconf import OmegaConf

from physicsnemo.launch.logging import LaunchLogger
from physicsnemo.launch.logging.mlflow import initialize_mlflow

from diagnostic import data, distribute, loss, models, precip, train


@hydra.main(
    version_base=None, config_path="config", config_name="diagnostic_precip.yaml"
)
def main(cfg):
    train_diagnostic(**OmegaConf.to_container(cfg))


def train_diagnostic(**cfg):
    """Top-level training function: setup everything and train model."""

    # setup model
    model = models.setup_model(**cfg["model"])
    (model, dist_manager) = distribute.distribute_model(model)

    # setup datapipes
    (train_specs, valid_specs) = data.data_source_specs(
        cfg["sources"]["state_params"], cfg["sources"]["diag_params"]
    )
    (train_datapipe, valid_datapipe) = data.setup_datapipes(
        train_specs,
        valid_specs,
        **cfg["datapipe"],
        dist_manager=dist_manager,
    )

    # setup MLFlow logging
    mlflow_cfg = cfg.get("logging", {}).get("mlflow", {})
    if mlflow_cfg.pop("use_mlflow", False):
        initialize_mlflow(**mlflow_cfg)
        LaunchLogger.initialize(use_mlflow=True)

    # setup loss
    loss_func = loss.GeometricL2Loss(
        lat_indices_used=train_datapipe.crop_window[0]
    )  # TODO: this should be configurable
    loss_func.to(device=dist_manager.device)

    # conversion from datapipe format to (input, target) tuples
    batch_conv = data.batch_converter(
        *train_specs, train_datapipe, diag_norm=precip.PrecipNorm()
    )

    # setup training loop
    trainer = train.Trainer(
        model,
        dist_manager=dist_manager,
        loss=loss_func,
        train_datapipe=train_datapipe,
        valid_datapipe=valid_datapipe,
        input_output_from_batch_data=batch_conv,
        **cfg["training"],
    )

    # train model
    trainer.fit()


if __name__ == "__main__":
    main()
