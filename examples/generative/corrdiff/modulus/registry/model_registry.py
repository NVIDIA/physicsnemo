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

from importlib.metadata import EntryPoint, entry_points
from typing import List, Union

# This import is required for compatibility with doctests.
import importlib_metadata

import modulus


# This model registry follows conventions similar to fsspec,
# https://github.com/fsspec/filesystem_spec/blob/master/fsspec/registry.py#L62C2-L62C2
# Tutorial on entrypoints: https://amir.rachum.com/blog/2017/07/28/python-entry-points/
# Borg singleton pattern: https://stackoverflow.com/questions/1318406/why-is-the-borg-pattern-better-than-the-singleton-pattern-in-python
class ModelRegistry:
    _shared_state = {"_model_registry": None}

    def __new__(cls, *args, **kwargs):
        obj = super(ModelRegistry, cls).__new__(cls)
        obj.__dict__ = cls._shared_state
        if cls._shared_state["_model_registry"] is None:
            cls._shared_state["_model_registry"] = cls._construct_registry()
        return obj

    @staticmethod
    def _construct_registry() -> dict:
        registry = {}
        entrypoints = entry_points(group="modulus.models")
        for entry_point in entrypoints:
            registry[entry_point.name] = entry_point
        return registry

    def register(self, model: "modulus.Module", name: Union[str, None] = None) -> None:
        """
        Registers a modulus model in the model registry under the provided name. If no name
        is provided, the model's name (from its `__name__` attribute) is used. If the
        name is already in use, raises a ValueError.

        Parameters
        ----------
        model : modulus.Module
            The model to be registered. Can be an instance of any class.
        name : str, optional
            The name to register the model under. If None, the model's name is used.

        Raises
        ------
        ValueError
            If the provided name is already in use in the registry.
        """

        # Check if model is a modulus model
        if not issubclass(model, modulus.Module):
            raise ValueError(
                f"Only subclasses of modulus.Module can be registered. "
                f"Provided model is of type {type(model)}"
            )

        # If no name provided, use the model's name
        if name is None:
            name = model.__name__

        # Check if name already in use
        if name in self._model_registry:
            raise ValueError(f"Name {name} already in use")

        # Add this class to the dict of model registry
        self._model_registry[name] = model

    def factory(self, name: str) -> "modulus.Module":
        """
        Returns a registered model given its name.

        Parameters
        ----------
        name : str
            The name of the registered model.

        Returns
        -------
        model : modulus.Module
            The registered model.

        Raises
        ------
        KeyError
            If no model is registered under the provided name.
        """

        model = self._model_registry.get(name)
        if model is not None:
            if isinstance(model, (EntryPoint, importlib_metadata.EntryPoint)):
                model = model.load()
            return model

        raise KeyError(f"No model is registered under the name {name}")

    def list_models(self) -> List[str]:
        """
        Returns a list of the names of all models currently registered in the registry.

        Returns
        -------
        List[str]
            A list of the names of all registered models. The order of the names is not
            guaranteed to be consistent.
        """
        return list(self._model_registry.keys())

    def __clear_registry__(self):
        # NOTE: This is only used for testing purposes
        self._model_registry = {}

    def __restore_registry__(self):
        # NOTE: This is only used for testing purposes
        self._model_registry = self._construct_registry()
