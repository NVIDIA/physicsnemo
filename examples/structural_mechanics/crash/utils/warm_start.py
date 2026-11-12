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

"""Initialise a model from another run's ``.mdlus`` or state-dict file."""

import os

import torch


def _is_mdlus_archive(path: str) -> bool:
    """A physicsnemo ``.mdlus`` archive: a zip, or the tar of older releases, with ``model.pt``."""
    import tarfile
    import zipfile

    # Tar first: a tar archive holding a zip member passes the zip sniff as
    # well. A torch.save file is a zip too, without model.pt.
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            return "model.pt" in archive.getnames()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return "model.pt" in archive.namelist()
    return False


def load_warm_start(model: torch.nn.Module, path) -> dict[str, list[str]]:
    """Initialise ``model`` from another run's ``.mdlus`` (or state-dict ``.pt``) file.

    A ``.mdlus`` archive (zip, or the tar format of releases before 2.2.0) is
    loaded with :func:`physicsnemo.utils.load_model_weights`; a ``torch.save``
    file may hold a state dict or a checkpoint dict with ``model_state_dict``.
    Every tensor of the file must exist in the model and every tensor of the
    model must be in the file. Returns the loaded key list. Used to initialise
    a run from a finished one (``training.warm_start_model``), by the evaluators
    and by the corrector to load its frozen teacher.
    """
    from physicsnemo.utils import load_model_weights

    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"warm start file {path} does not exist")
    if _is_mdlus_archive(path):
        try:
            load_model_weights(model, path)
        except RuntimeError as error:  # the strict load names the mismatched keys
            raise ValueError(
                f"warm start {path} does not match the model: {error}"
            ) from error
        return {"loaded": sorted(model.state_dict().keys())}
    # a torch.save file: a state dict or a checkpoint dict
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(
            f"warm start {path} has unexpected tensors not in the model: "
            f"{sorted(result.unexpected_keys)[:8]}"
        )
    if result.missing_keys:
        raise ValueError(
            f"warm start {path} is missing tensors of the model: "
            f"{sorted(result.missing_keys)[:8]}"
        )
    return {"loaded": sorted(state.keys())}
