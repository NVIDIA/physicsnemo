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

from __future__ import annotations

import warnings
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, Dict, List, Union

# Handle both stdlib and backport EntryPoint classes for isinstance checks.
# Some packages (e.g., mlflow, opentelemetry) import importlib_metadata which
# can cause entry points to be instances of importlib_metadata.EntryPoint
# instead of importlib.metadata.EntryPoint.

# For future maintainers who think, "I could clean this up ..."
# As much as I WANT to remove this, it is not worth it. If there is any
# usage of importlib_metadata in other packages it could break.  It certainly
# breaks docstring testing in CI.  And it was very difficult to debug,
# since the breaking import that changes the EntryPoint definition typically
# shows up OUTSIDE of the physicsnemo code base, you'll go crazy trying to debug.
try:
    import importlib_metadata

    _ENTRYPOINT_TYPES: tuple = (EntryPoint, importlib_metadata.EntryPoint)
except ImportError:
    _ENTRYPOINT_TYPES = (EntryPoint,)

if TYPE_CHECKING:
    from physicsnemo.core.module import Module


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
    def _construct_registry() -> Dict[str, type["Module"] | EntryPoint]:
        registry: Dict[str, type["Module"] | EntryPoint] = {}
        entrypoints = entry_points(group="physicsnemo.models")
        for entry_point in entrypoints:
            registry[entry_point.name] = entry_point

        # Pull in any modulus models for backwards compatibility
        entrypoints = entry_points(group="modulus.models")
        for entry_point in entrypoints:
            if entry_point.name not in registry:
                # Add depricated warning
                warnings.warn(
                    f"Model {entry_point.name} is being loaded from the 'modulus.models' group. "
                    f"This probably means it is being exposed from a package that has not yet been "
                    f"updated to use the 'physicsnemo.models' group. This group may be removed in a "
                    f"future release. Please contact the package maintainer to update the entry point.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                registry[entry_point.name] = entry_point

        return registry

    def register(self, model: type["Module"], name: Union[str, None] = None) -> None:
        """
        Registers a physicsnemo model class in the model registry under the provided name. If no name
        is provided, the model's name (from its `__name__` attribute) is used. If the
        name is already in use, raises a ValueError.

        Parameters
        ----------
        model : physicsnemo.core.Module
            The model class to be registered.
        name : str, optional
            The name to register the model under. If None, the model class name
            is used.

        Raises
        ------
        ValueError
            If the provided name is already in use in the registry.

        Examples
        --------
        Example 1: Register a model class using its default name (from ``__name__``):

        >>> from physicsnemo.core import Module, ModelRegistry
        >>> # Define a custom model class
        >>> class MyCustomModel(Module):
        ...     def __init__(self, hidden_size):
        ...         super().__init__()
        ...         self.hidden_size = hidden_size
        ...
        ...     def forward(self, x):
        ...         return x
        >>> # Get the registry instance
        >>> registry = ModelRegistry()
        >>> # Register the model without specifying a name
        >>> # The class name 'MyCustomModel' will be used automatically
        >>> registry.register(MyCustomModel)
        >>> # Retrieve the model class from the registry
        >>> ModelClass = registry.factory('MyCustomModel')
        >>> # Instantiate the model
        >>> model = ModelClass(hidden_size=128)


        """

        # If no name provided, use the model class name
        if name is None:
            name = model.__name__

        # Check if name already in use
        if name in self._model_registry:
            raise ValueError(
                f"Name {name} already in use.\n"
                f"Current registered models are: {sorted(self.list_models())}"
            )

        # Add this class to the dict of model registry
        self._model_registry[name] = model

    def factory(self, name: str) -> type["Module"]:
        """
        Returns a registered model class given its name.

        Parameters
        ----------
        name : str
            The name of the registered model.

        Returns
        -------
        model : physicsnemo.core.Module
            The registered model.

        Raises
        ------
        KeyError
            If no model is registered under the provided name.
        """

        model = self._model_registry.get(name)
        if model is not None:
            if isinstance(model, _ENTRYPOINT_TYPES):
                model = model.load()
                # Update the registry with the loaded object:
                self._model_registry[name] = model
            return model

        raise KeyError(
            f"No model is registered under the name {name}. "
            f"Current registered models are: {sorted(self.list_models())}"
        )

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
