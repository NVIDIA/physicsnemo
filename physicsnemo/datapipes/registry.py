# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""
Registry for datapipe components.

Provides registries for transforms and readers, enabling:

- Short aliases in Hydra configuration
- Component discovery and introspection
- Consistent instantiation patterns

Examples
--------
>>> from physicsnemo.datapipes.registry import COMPONENT_REGISTRY
>>> from physicsnemo.datapipes.transforms import Transform
>>>
>>> @COMPONENT_REGISTRY.register()
... class MyTransform(Transform):
...     pass
>>>
>>> # Get registered component by name
>>> cls = COMPONENT_REGISTRY.get("MyTransform")
>>>
>>> # List all registered components
>>> print(COMPONENT_REGISTRY.list()) # doctest: +SKIP

OmegaConf Resolver for Hydra configs
------------------------------------
After calling ``register_resolvers()``, you can use short names in YAML configs:

>>> from physicsnemo.datapipes.registry import register_resolvers
>>> register_resolvers()

Then in your YAML config:

.. code-block:: yaml

    # Instead of:
    _target_: physicsnemo.datapipes.transforms.CenterOfMass

    # You can write:
    _target_: ${dp:CenterOfMass}

    # Or for readers:
    _target_: ${dp:ZarrReader}
"""

from __future__ import annotations

from typing import Callable, Type, TypeVar

from omegaconf import OmegaConf

T = TypeVar("T")


class ComponentRegistry:
    """
    Registry for datapipe components with short aliases.

    A registry allows components (transforms, readers) to be registered
    with a name and later retrieved by that name. This enables:

    - Hydra configuration with short names instead of full import paths
    - Runtime discovery of available components
    - Validation that a component exists

    Parameters
    ----------
    name : str
        Human-readable name for this registry (e.g., "transforms").

    Examples
    --------
    >>> from physicsnemo.datapipes.registry import ComponentRegistry
    >>> from physicsnemo.datapipes.transforms import Transform
    >>> registry = ComponentRegistry("transforms")
    >>>
    >>> @registry.register()
    ... class Normalize(Transform):
    ...     pass
    >>>
    >>> @registry.register("norm")  # Custom alias
    ... class Normalize(Transform):
    ...     pass
    >>>
    >>> # Retrieve by name
    >>> Normalize = registry.get("Normalize")
    >>> Normalize = registry.get("norm")
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the registry.

        Parameters
        ----------
        name : str
            Human-readable name for this registry (e.g., "transforms").
        """
        self.name = name
        self._registry: dict[str, Type] = {}

    def register(self, name: str | None = None) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a component class.

        Parameters
        ----------
        name : str, optional
            Name to register under. If None, uses the class name.

        Returns
        -------
        Callable[[Type[T]], Type[T]]
            Decorator function that registers the class.

        Raises
        ------
        ValueError
            If the name is already registered.

        Examples
        --------
        >>> from physicsnemo.datapipes.registry import registry
        >>> from physicsnemo.datapipes.transforms import Transform
        >>> @registry.register()
        ... class MyTransform(Transform):
        ...     pass
        >>>
        >>> @registry.register("custom_name")
        ... class AnotherTransform(Transform):
        ...     pass
        """

        def decorator(cls: Type[T]) -> Type[T]:
            key = name if name is not None else cls.__name__
            if key in self._registry:
                raise ValueError(
                    f"Component '{key}' is already registered in {self.name} registry. "
                    f"Existing: {self._registry[key]}, New: {cls}"
                )
            self._registry[key] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type:
        """
        Get a registered component by name.

        Parameters
        ----------
        name : str
            The registered name of the component.

        Returns
        -------
        Type
            The registered class.

        Raises
        ------
        KeyError
            If the name is not registered.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"Component '{name}' not found in {self.name} registry. "
                f"Available: {available or '(none)'}"
            )
        return self._registry[name]

    def list(self) -> list[str]:
        """
        List all registered component names.

        Returns
        -------
        list[str]
            Sorted list of registered names.
        """
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """
        Check if a name is registered.

        Parameters
        ----------
        name : str
            Name to check.

        Returns
        -------
        bool
            True if registered, False otherwise.
        """
        return name in self._registry

    def __len__(self) -> int:
        """
        Return the number of registered components.

        Returns
        -------
        int
            Number of registered components.
        """
        return len(self._registry)

    def __repr__(self) -> str:
        """
        Return string representation.

        Returns
        -------
        str
            String representation of the registry.
        """
        return f"ComponentRegistry({self.name!r}, count={len(self)})"


# Global component registry for all datapipe components (transforms, readers, etc.)
COMPONENT_REGISTRY = ComponentRegistry("components")


def register(name: str | None = None) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a datapipe component class.

    Registered components can be referenced by short name in Hydra configs
    using the ``${dp:ComponentName}`` syntax after calling ``register_resolvers()``.

    Parameters
    ----------
    name : str, optional
        Name to register under. If None, uses the class name.

    Returns
    -------
    Callable[[Type[T]], Type[T]]
        Decorator function that registers the class.

    Examples
    --------
    >>> from physicsnemo.datapipes.registry import register
    >>> from physicsnemo.datapipes.transforms import Transform
    >>> @register()
    ... class ATransform(Transform):
    ...     pass
    >>>
    >>> @register("custom_name")
    ... class AnotherTransform(Transform):
    ...     pass
    """
    return COMPONENT_REGISTRY.register(name)


def _resolve_component(name: str) -> str:
    """
    Resolve a short component name to its full module path.

    Parameters
    ----------
    name : str
        Short name of the component (e.g., "CenterOfMass", "ZarrReader").

    Returns
    -------
    str
        Full module path for use in Hydra's ``_target_`` field.

    Raises
    ------
    KeyError
        If the name is not found in the registry.
    """
    if name in COMPONENT_REGISTRY:
        cls = COMPONENT_REGISTRY.get(name)
        return f"{cls.__module__}.{cls.__name__}"

    # Not found - build helpful error message
    available = COMPONENT_REGISTRY.list()
    raise KeyError(f"Component '{name}' not found in registry. Available: {available}.")


def register_resolvers() -> None:
    """
    Register OmegaConf resolvers for datapipe components.

    This enables short names in Hydra YAML configs using the ``${dp:...}`` syntax.
    Call this function before using Hydra's ``instantiate()`` or loading configs.

    The resolver looks up components in COMPONENT_REGISTRY dynamically at resolve
    time, so custom components registered after this function is called will still
    be available.

    Examples
    --------
    >>> from physicsnemo.datapipes.registry import register_resolvers
    >>> register_resolvers()

    Then in YAML:

    .. code-block:: yaml

        # Use short names with ${dp:...}
        - _target_: ${dp:CenterOfMass}
          coords_key: stl_centers
          areas_key: stl_areas
          output_key: center_of_mass

        - _target_: ${dp:SubsamplePoints}
          input_keys:
            - surface_mesh_centers
          n_points: 50000

    Notes
    -----
    This function can be called multiple times safely. The resolver dynamically
    looks up components from COMPONENT_REGISTRY, so custom transforms registered
    after this call will still be resolvable.
    """
    OmegaConf.register_new_resolver("dp", _resolve_component, replace=True)
