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

from physicsnemo.launch.utils import load_checkpoint

from diagnostic import data, distribute, export, models


@hydra.main(
    version_base=None, config_path="config", config_name="diagnostic_precip.yaml"
)
def main(cfg):
    export_diagnostic(**OmegaConf.to_container(cfg))


def export_diagnostic(
    out_dir="exports",
    model_name=None,
    epoch=None,  # None loads latest checkpoint (default)
    **cfg
):
    # setup model
    model = models.setup_model(**cfg["model"])
    (model, dist_manager) = distribute.distribute_model(model)

    # setup datapipes
    (train_specs, valid_specs) = data.data_source_specs(
        cfg["sources"]["state_params"], cfg["sources"]["diag_params"]
    )
    (train_datapipe, _) = data.setup_datapipes(
        train_specs,
        valid_specs,
        **cfg["datapipe"],
        dist_manager=dist_manager,
    )

    load_checkpoint(cfg["training"]["checkpoint_dir"], models=model, epoch=epoch)

    export.export_diagnostic_e2mip(
        out_dir=out_dir,
        model_name=model_name,
        model=model,
        datapipe=train_datapipe,
        in_source=train_datapipe.sources[0],
        out_source=train_datapipe.sources[1],
    )


if __name__ == "__main__":
    main()
