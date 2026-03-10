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

import importlib
import inspect
from typing import Any, Callable
from warnings import warn

from physicsnemo.active_learning.protocols import ActiveLearningProtocol

__all__ = ["registry"]


class ActiveLearningRegistry:
    """
    Registry for active learning protocols.

    This class provides a centralized registry for user-defined active learning
    protocols that implement the :class:`~physicsnemo.active_learning.protocols.ActiveLearningProtocol`. It enables string-based
    lookups for checkpointing and provides argument validation when constructing
    protocol instances.

    The registry supports two primary modes of interaction:
    1. Registration via decorator: ``@registry.register("my_strategy")``
    2. Construction with validation: ``registry.construct("my_strategy", **kwargs)``

    Attributes
    ----------
    _registry : dict
        Internal dictionary mapping protocol names to their class types.

    Methods
    -------
    register(cls_name: str) -> Callable
        Decorator to register a protocol class with a given name.
    construct(cls_name: str, **kwargs) -> :class:`~physicsnemo.active_learning.protocols.ActiveLearningProtocol`
        Construct an instance of a registered protocol with argument validation.
    is_registered(cls_name: str) -> bool
        Check if a protocol name is registered.

    Properties
    ----------
    registered_names : list
        A list of all registered protocol names, sorted alphabetically.

    See Also
    --------
    ActiveLearningProtocol : Base protocol for active learning strategies
    QueryStrategy : Query strategy protocol
    LabelStrategy : Label strategy protocol
    MetrologyStrategy : Metrology strategy protocol

    Examples
    --------
    Register a custom strategy:

    >>> from physicsnemo.active_learning._registry import registry
    >>> @registry.register("my_custom_strategy")
    ... class MyCustomStrategy:
    ...     def __init__(self, param1: int, param2: str):
    ...         self.param1 = param1
    ...         self.param2 = param2

    Construct an instance with validation:

    >>> strategy = registry.construct("my_custom_strategy", param1=42, param2="test")
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._registry: dict[str, type[ActiveLearningProtocol]] = {}

    def register(
        self, cls_name: str
    ) -> Callable[[type[ActiveLearningProtocol]], type[ActiveLearningProtocol]]:
        """
        Decorator to register an active learning protocol class.

        This decorator registers a class implementing the :class:`~physicsnemo.active_learning.protocols.ActiveLearningProtocol`
        under the given name, allowing it to be retrieved and constructed later
        using the :meth:`construct` method.

        Parameters
        ----------
        cls_name : str
            The name to register the protocol under. This will be used as the
            key for later retrieval.

        Returns
        -------
        Callable
            A decorator function that registers the class and returns it unchanged.

        Raises
        ------
        ValueError
            If a protocol with the same name is already registered.

        Examples
        --------
        >>> @registry.register("my_new_strategy")
        ... class MyStrategy:
        ...     def __init__(self, param: int):
        ...         self.param = param
        """

        def decorator(
            cls: type[ActiveLearningProtocol],
        ) -> type[ActiveLearningProtocol]:
            """
            Method for decorating a class to registry it with the registry.
            """
            if cls_name in self._registry:
                raise ValueError(
                    f"Protocol '{cls_name}' is already registered. "
                    f"Existing class: {self._registry[cls_name].__name__}"
                )
            self._registry[cls_name] = cls
            return cls

        return decorator

    def construct(
        self, cls_name: str, module_path: str | None = None, **kwargs: Any
    ) -> ActiveLearningProtocol:
        """
        Construct an instance of a registered protocol with argument validation.

        This method retrieves a registered protocol class by name, validates that
        the provided keyword arguments match the class's constructor signature,
        and returns a new instance of the class.

        Parameters
        ----------
        cls_name : str
            The name of the registered protocol to construct.
        module_path: str or None
            The path to the module to get the class from.
        **kwargs : Any
            Keyword arguments to pass to the protocol's constructor.

        Returns
        -------
        :class:`~physicsnemo.active_learning.protocols.ActiveLearningProtocol`
            A new instance of the requested protocol class.

        Raises
        ------
        KeyError
            If the protocol name is not registered.
        TypeError
            If the provided keyword arguments do not match the constructor signature.
            This includes missing required parameters or unexpected parameters.

        Examples
        --------
        >>> from physicsnemo.active_learning._registry import registry
        >>> @registry.register("my_latest_strategy")
        ... class MyStrategy:
        ...     def __init__(self, param: int):
        ...         self.param = param
        >>> strategy = registry.construct("my_latest_strategy", param=42)
        """
        cls = self.get_class(cls_name, module_path)

        # Validate arguments against the class signature
        try:
            sig = inspect.signature(cls.__init__)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Could not inspect signature of {cls.__name__}.__init__: {e}"
            )

        # Get parameters, excluding 'self'
        params = {
            name: param for name, param in sig.parameters.items() if name != "self"
        }

        # Check if the signature accepts **kwargs
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        # Check for missing required parameters
        missing = []
        for name, param in params.items():
            if (
                param.kind
                not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
                and param.default is inspect.Parameter.empty
                and name not in kwargs
            ):
                missing.append(name)

        if missing:
            raise TypeError(
                f"Missing required arguments for {cls.__name__}: {', '.join(missing)}"
            )

        # Check for unexpected parameters (unless **kwargs is present)
        if not has_var_keyword:
            param_names = {
                name
                for name, param in params.items()
                if param.kind
                not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
            }
            unexpected = [name for name in kwargs if name not in param_names]

            if unexpected:
                warn(
                    f"Unexpected arguments for {cls.__name__}: {', '.join(unexpected)}. "
                    f"Valid parameters: {', '.join(sorted(param_names))}"
                )
        return cls(**kwargs)

    def __getitem__(self, cls_name: str) -> type[ActiveLearningProtocol]:
        """
        Retrieve a registered protocol class by name using dict-like access.

        This method allows accessing registered protocol classes using square
        bracket notation, e.g., `registry['my_strategy']`.

        Parameters
        ----------
        cls_name : str
            The name of the registered protocol to retrieve.

        Returns
        -------
        type[ActiveLearningProtocol]
            The class type of the registered protocol.

        Raises
        ------
        KeyError
            If the protocol name is not registered.

        Examples
        --------
        >>> from physicsnemo.active_learning._registry import registry
        >>> @registry.register("my_strategy")
        ... class MyStrategy:
        ...     def __init__(self, param: int):
        ...         self.param = param
        >>> RetrievedClass = registry['my_strategy']
        >>> instance = RetrievedClass(param=42)
        """
        if cls_name not in self._registry:
            available = ", ".join(self._registry.keys()) if self._registry else "none"
            raise KeyError(
                f"Protocol '{cls_name}' is not registered. "
                f"Available protocols: {available}"
            )
        return self._registry[cls_name]

    def is_registered(self, cls_name: str) -> bool:
        """
        Check if a protocol name is registered.

        Parameters
        ----------
        cls_name : str
            The name of the protocol to check.

        Returns
        -------
        bool
            True if the protocol is registered, False otherwise.
        """
        return cls_name in self._registry

    @property
    def registered_names(self) -> list[str]:
        """
        A list of all registered protocol names, sorted alphabetically.

        Returns
        -------
        list[str]
            A list of all registered protocol names, sorted alphabetically.
        """
        return sorted(self._registry.keys())

    def get_class(self, cls_name: str, module_path: str | None = None) -> type:
        """
        Get a class by name from the registry or from a module path.

        Parameters
        ----------
        cls_name: str
            The name of the class to get.
        module_path: str | None
            The path to the module to get the class from.

        Returns
        -------
        type
            The class.

        Raises
        ------
        NameError: If the class is not found in the registry or module.
        ModuleNotFoundError: If the module is not found with the specified module path.
        """
        if cls_name in self.registered_names:
            return self._registry[cls_name]
        else:
            if module_path:
                module = importlib.import_module(module_path)
                cls = getattr(module, cls_name, None)
                if not cls:
                    raise NameError(
                        f"Class {cls_name} not found in module {module_path}"
                    )
                return cls
            else:
                raise NameError(
                    f"Class {cls_name} not found in registry, and no module path was provided."
                )


# Module-level registry instance for global access
registry = ActiveLearningRegistry()
