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
import json
from omegaconf import OmegaConf
import numpy as np

from physicsnemo import Module


def save_inference_model_package(
    model,
    cfg,
    predicted_variable_normalizer,
    unpredicted_variable_normalizer,
    latitude,
    longitude,
    save_path,
    readme=None,
):
    """
    Save inference model package for global weather models.

    The model package is a directory with the following files:
    - model.mdlus: The model file.
    - metadata.json: A json file with cfg parameters.
    - predicted_variable_means.npy: A numpy file with the predicted variable means.
    - predicted_variable_stds.npy: A numpy file with the predicted variable standard deviations.
    - unpredicted_variable_means.npy: A numpy file with the unpredicted variable means.
    - unpredicted_variable_stds.npy: A numpy file with the unpredicted variable standard deviations.
    - latitude.npy: A numpy file with the latitude values.
    - longitude.npy: A numpy file with the longitude values.
    - README.md: A readme file with the readme text.
    These files help transfer trained models to inference usecases.

    TODO: Add better support for Mean and Std saving.

    Parameters
    ----------
    model : physicsnemo.Module
        Model to save model card for.
    cfg : DictConfig
        DictConfig with model and data parameters.
    predicted_variable_normalizer : nn.BatchNorm2d
        Normalizer for predicted variables.
    unpredicted_variable_normalizer : nn.BatchNorm2d
        Normalizer for unpredicted variables.
    save_path : str
        Path to save model card to.
    readme : str
        readme text for model card.
    """

    # DDP fix
    if not isinstance(model, Module) and hasattr(model, "module"):
        model = model.module

    # Create model card directory
    os.makedirs(save_path, exist_ok=True)

    # Save model mdlus file
    model.save(
        os.path.join(save_path, "model.mdlus"),
    )

    # Save json files with cfg parameters
    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=4)

    # Get numpy arrays from normalizers mean and std and save them as npy files
    predicted_variable_means = predicted_variable_normalizer.running_mean.cpu().numpy()
    predicted_variable_stds = predicted_variable_normalizer.running_var.cpu().numpy()
    with open(os.path.join(save_path, "predicted_variable_means.npy"), "wb") as f:
        np.save(f, predicted_variable_means)
    with open(os.path.join(save_path, "predicted_variable_stds.npy"), "wb") as f:
        np.save(f, predicted_variable_stds)
    unpredicted_variable_means = (
        unpredicted_variable_normalizer.running_mean.cpu().numpy()
    )
    unpredicted_variable_stds = (
        unpredicted_variable_normalizer.running_var.cpu().numpy()
    )
    with open(os.path.join(save_path, "unpredicted_variable_means.npy"), "wb") as f:
        np.save(f, unpredicted_variable_means)
    with open(os.path.join(save_path, "unpredicted_variable_stds.npy"), "wb") as f:
        np.save(f, unpredicted_variable_stds)

    # Save latitude and longitude
    with open(os.path.join(save_path, "latitude.npy"), "wb") as f:
        np.save(f, latitude)
    with open(os.path.join(save_path, "longitude.npy"), "wb") as f:
        np.save(f, longitude)

    # Save readme
    if readme is not None:
        with open(os.path.join(save_path, "README.md"), "w") as f:
            f.write(readme)
