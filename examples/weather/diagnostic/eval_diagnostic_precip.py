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

import os

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from diagnostic import data, distribute, loss, models, precip, train


@hydra.main(
    version_base=None, config_path="config", config_name="diagnostic_precip.yaml"
)
def main(cfg):
    test_diagnostic(**OmegaConf.to_container(cfg))


def test_diagnostic(**cfg):
    # setup model
    model = models.setup_model(**cfg["model"])
    (model, dist_manager) = distribute.distribute_model(model)

    # setup datapipes
    (train_specs, valid_specs) = data.data_source_specs(
        cfg["sources"]["state_params"],
        cfg["sources"]["diag_params"],
        valid_dir="out_of_sample",
    )
    cfg["datapipe"]["num_samples_per_year_valid"] = cfg["datapipe"][
        "num_samples_per_year_train"
    ]  # validate on entire year
    (train_datapipe, valid_datapipe) = data.setup_datapipes(
        train_specs,
        valid_specs,
        **cfg["datapipe"],
        dist_manager=dist_manager,
    )

    # create callback for tracking error
    mean = valid_specs[1].mu
    std = valid_specs[1].sd
    rmse_callback = RMSECallback(device=dist_manager.device, mean=mean, std=std)

    # setup loss
    loss_func = loss.GeometricL2Loss(
        lat_indices_used=train_datapipe.crop_window[0]
    )  # TODO: this should be configurable
    loss_func = loss_func.to(device=dist_manager.device)

    # conversion from datapipe format to (input, target) tuples
    batch_conv = data.batch_converter(
        *train_specs, train_datapipe, diag_norm=precip.PrecipNorm()
    )

    # setup trainer to produce test samples
    trainer = train.Trainer(
        model,
        dist_manager=dist_manager,
        loss=loss_func,
        train_datapipe=train_datapipe,
        valid_datapipe=valid_datapipe,
        input_output_from_batch_data=batch_conv,
        validation_callbacks=[rmse_callback],
        **cfg["training"],
    )

    # evaluate model
    trainer.validate_on_epoch()

    # save results
    rmse = rmse_callback.value().cpu().numpy()

    os.makedirs("./results", exist_ok=True)
    np.save("./results/rmse.npy", rmse)  # TODO: should be configurable


class RMSECallback:
    """Callable that keeps track of RMS error.
    Can be used in `Trainer.validation_callbacks`.
    """

    def __init__(self, device, mean=None, std=None):
        self.mse = None
        self.n_samples = 0
        self.mean = None if mean is None else torch.from_numpy(mean).to(device=device)
        self.std = None if std is None else torch.from_numpy(std).to(device=device)

    def __call__(self, outvar_true, outvar_pred, **kwargs):
        # reverse normalization
        if self.mean is not None:
            outvar_true = outvar_true * self.std + self.mean
            outvar_pred = outvar_pred * self.std + self.mean

        # compute squared difference
        sqr_diff = torch.square(outvar_true - outvar_pred)
        batch_size = sqr_diff.shape[0]
        avg_axes = tuple(range(sqr_diff.ndim - 2))
        sqr_diff = torch.mean(sqr_diff, axis=avg_axes)

        # accumulate MSE
        if self.mse is None:
            self.mse = sqr_diff
        else:
            old_weight = self.n_samples / (self.n_samples + batch_size)
            new_weight = 1 - old_weight
            self.mse = old_weight * self.mse + new_weight * sqr_diff
        self.n_samples += batch_size

    def value(self):
        return torch.sqrt(self.mse)


if __name__ == "__main__":
    main()
