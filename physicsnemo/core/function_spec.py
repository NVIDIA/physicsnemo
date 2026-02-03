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

from __future__ import annotations

import importlib
import importlib.util
import re
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Sequence, Tuple

import torch
from packaging.requirements import Requirement

from physicsnemo.core.version_check import check_version_spec


@dataclass(frozen=True)
class Implementation:
    """Stores data for a functional implementation.

    Attributes
    ----------
    name : str
        Implementation name used for registration and dispatch.
    func : Callable
        Callable that executes the backend implementation.
    required_imports : Tuple[str, ...]
        Optional dependency requirements for the implementation.
    rank : int
        Lower rank is preferred during default dispatch.
    baseline : bool
        Marks the reference implementation for benchmarking.
    available : bool, optional
        Whether required imports are satisfied, by default True.
    """

    name: str
    func: Callable
    required_imports: Tuple[str, ...]
    rank: int
    baseline: bool
    available: bool = True

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class FunctionSpec:
    """Base class for PhysicsNeMo function wrappers.

    ``FunctionSpec`` ties together multiple backend implementations of the same
    operation (Warp, PyTorch, cuML, SciPy, ...) while providing a consistent
    surface for benchmarking and correctness comparisons. It gives a single
    place to register implementations and to describe how they are selected at
    runtime.

    Overview
    --------
    ``FunctionSpec`` provides a small registry and dispatch layer for functions
    that have multiple backend implementations. Implementations are registered
    on the subclass using :meth:`FunctionSpec.register` and selected by
    :meth:`FunctionSpec.dispatch`.

    The default dispatch path selects the *lowest-rank* available implementation
    (rank is an integer; lower is preferred). Users can override selection with
    ``implementation="name"``.

    Implementing a FunctionSpec
    ---------------------------
    1. Subclass ``FunctionSpec``.
    2. Register one or more backend implementations with the decorator
       ``@FunctionSpec.register``. Provide a ``name`` and a ``rank`` (lower
       wins). Optionally set ``baseline=True`` for the reference implementation
       used in benchmarking. The decorator must be used inside the class body.
    3. Implement :meth:`make_inputs` and :meth:`compare` for benchmarking and
       correctness checks. These are optional for basic usage, but they are
       highly encouraged and required for benchmarking and validation
       workflows. ``make_inputs`` should be a generator that yields
       representative input tuples for the function, and ``compare`` should
       validate that outputs from two implementations match.
    4. Expose a functional entry point with :meth:`make_function`.

    Dispatch rules
    --------------
    The ``required_imports`` field on registrations accepts requirement strings
    like ``"warp>=0.6.0"``. Dispatch skips implementations whose requirements
    are not satisfied, then selects the available implementation with the
    lowest rank. If a lower-rank implementation is unavailable and a higher-rank
    fallback is used, a one-time warning is emitted describing the fallback.
    Users can override selection with ``implementation="name"``.

    Examples
    --------
    A minimal identity function with both Warp and PyTorch implementations
    (modeled after ``sdf.py``):

    .. code-block:: python

        import importlib
        import torch

        from physicsnemo.core.function_spec import FunctionSpec
        from physicsnemo.core.version_check import check_version_spec

        WARP_AVAILABLE = check_version_spec("warp", "0.6.0", hard_fail=False)

        if WARP_AVAILABLE:
            wp = importlib.import_module("warp")
            wp.init()
            wp.config.quiet = True

            @wp.kernel
            def _identity_kernel(
                x: wp.array(dtype=wp.float32),
                y: wp.array(dtype=wp.float32),
            ):
                i = wp.tid()
                y[i] = x[i]

            @torch.library.custom_op("physicsnemo::identity_warp", mutates_args=())
            def identity_impl(x: torch.Tensor) -> torch.Tensor:
                out = torch.empty_like(x)
                device, stream = FunctionSpec.warp_launch_context(x)
                wp_x = wp.from_torch(x, dtype=wp.float32, return_ctype=True)
                wp_y = wp.from_torch(out, dtype=wp.float32, return_ctype=True)
                with wp.ScopedStream(stream):
                    wp.launch(
                        kernel=_identity_kernel,
                        dim=x.numel(),
                        inputs=[wp_x, wp_y],
                        device=device,
                        stream=stream,
                    )
                return out

            @identity_impl.register_fake
            def identity_impl_fake(x: torch.Tensor) -> torch.Tensor:
                return torch.empty_like(x)
        else:

            def identity_impl(*args, **kwargs) -> torch.Tensor:
                raise ImportError(
                    "warp>=0.6.0 is required for the Warp identity implementation"
                )

        def identity_torch(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        class Identity(FunctionSpec):
            \"\"\"Identity function with Warp and PyTorch backends.\"\"\"

            @FunctionSpec.register(
                name="warp",
                required_imports=("warp>=0.6.0",),
                rank=0,
            )
            def warp_forward(x: torch.Tensor) -> torch.Tensor:
                return identity_impl(x)

            @FunctionSpec.register(name="torch", rank=1, baseline=True)
            def torch_forward(x: torch.Tensor) -> torch.Tensor:
                return identity_torch(x)

            @classmethod
            def make_inputs(cls, device: torch.device | str = "cpu"):
                device = torch.device(device)
                for size in (1024, 4096):
                    yield (torch.randn(size, device=device),)

            @classmethod
            def compare(
                cls, output: torch.Tensor, reference: torch.Tensor
            ) -> None:
                torch.testing.assert_close(output, reference)

        identity = Identity.make_function("identity")

        x = torch.arange(8, device="cuda")
        y = identity(x)

    Notes
    -----
    - Only one implementation may be marked as ``baseline=True``; this is the
      reference used when benchmarking.
    - The function returned by
      :meth:`~physicsnemo.core.function_spec.FunctionSpec.make_function` copies
      the class ``__doc__``. Keep this docstring up to date so the public API
      documentation for the function wrapper stays accurate.


    """

    _impl_registry: Dict[str, Dict[str, Implementation]] = {}
    _fallback_warned: set[str] = set()

    @classmethod
    def register(
        cls,
        name: str,
        required_imports: Sequence[str] | None = None,
        rank: int = 0,
        baseline: bool = False,
    ):
        """Decorator to register an implementation on a subclass.

        Parameters
        ----------
        name : str
            Implementation name.
        required_imports : Sequence[str] | None, optional
            Optional import requirements, by default None.
        rank : int, optional
            Rank for selection, by default 0.
        baseline : bool, optional
            Whether this is the baseline implementation, by default False.

        Returns
        -------
        Callable
            Decorator that registers the implementation immediately.
            The decorator returns a ``staticmethod`` wrapper so the implementation
            can be called directly on the class.
        """

        def decorator(func: Callable):
            # Unwrap staticmethod/classmethod to the underlying function before registering.
            # This is a safeguard if users add @staticmethod or @classmethod decorators
            # to the implementation function.
            if isinstance(func, (staticmethod, classmethod)):
                target = func.__func__
            else:
                target = func

            # infer the class key from the function's qualname
            # This requires the implementation decorator to
            # be called inside the class definition.
            qualname = getattr(target, "__qualname__", "")
            if "." not in qualname:
                raise ValueError(
                    "FunctionSpec.register must be used inside a class body. "
                    "Use it to decorate methods defined on the FunctionSpec subclass."
                )
            owner = qualname.rsplit(".", 1)[0]
            class_key = f"{target.__module__}.{owner}"

            # Register the implementation
            imports = tuple(required_imports or ())
            available = cls._check_imports(imports)
            impl = Implementation(
                name=name,
                func=target,
                required_imports=imports,
                rank=rank,
                baseline=baseline,
                available=available,
            )
            cls._register_impl(impl=impl, class_key=class_key)

            # Return the function as a staticmethod
            # (makes it callable without an instance)
            # Not necessary but keeping for now
            return staticmethod(target)

        return decorator

    @classmethod
    def make_inputs(cls, device: torch.device) -> Iterable[Tuple[torch.Tensor, ...]]:
        """Generator for example inputs to the function.
        This is used for benchmarking and testing.
        Generated inputs should be representative of the function's
        expected input and reasonable for code coverage as well
        as benchmarking.

        Parameters
        ----------
        device : torch.device
            Device for generated tensors.

        Returns
        -------
        Iterable[Tuple[torch.Tensor, ...]]
            Iterable of input tuples.
        """
        raise NotImplementedError(f"{cls.__name__}.make_inputs must be implemented")

    @classmethod
    def compare(cls, output: object, reference: object) -> None:
        """Compare implementation outputs for validation.
        This is used to validate different implementations of the same function
        against a baseline implementation.

        Parameters
        ----------
        output : object
            Output from the implementation to compare.
        reference : object
            Reference output to compare against.
        """
        raise NotImplementedError(f"{cls.__name__}.compare must be implemented")

    def __call__(self, *args, **kwargs):
        """Dispatch to the selected implementation.

        Parameters
        ----------
        *args, **kwargs
            Arguments forwarded to the implementation.

        Returns
        -------
        object
            The implementation result.
        """
        return self.dispatch(*args, **kwargs)

    @classmethod
    def make_function(cls, name: str | None = None):
        """Create a functional wrapper around the class dispatch.
        The function created this way will be whats exposed to the user.

        Parameters
        ----------
        name : str | None, optional
            Function name override, by default None.

        Returns
        -------
        Callable
            Callable that forwards to ``dispatch``.
        """

        # Define the function
        def _function(*args, **kwargs):
            return cls.dispatch(*args, **kwargs)

        # Set the function attributes
        # This keeps things like docstrings for API documentation.
        _function.__name__ = name or cls.__name__
        _function.__qualname__ = _function.__name__
        _function.__module__ = cls.__module__
        _function.__doc__ = cls.__doc__
        return _function

    @classmethod
    def dispatch(cls, *args, **kwargs):
        """Dispatch to the chosen implementation.

        Parameters
        ----------
        *args, **kwargs
            Arguments forwarded to the implementation.

        Returns
        -------
        object
            Implementation output.
        """

        # Resolve explicit implementation selection (implementation in kwargs).
        implementation = kwargs.pop("implementation", None)

        # Lookup the implementation registry for this FunctionSpec.
        impls = cls._get_impls()

        # Check if the implementation is registered
        cls._check_impl(implementation, impls)

        # If a specific implementation is requested, validate and call it.
        if implementation is not None:
            # Get the implementation
            impl = impls[implementation]

            # Check if the implementation's required imports are available
            if not impl.available:
                raise ImportError(
                    f"Implementation '{implementation}' is not available for {cls.__name__}"
                )

            # Execute the implementation
            return impl.func(*args, **kwargs)

        # Otherwise, find all available implementations and select the lowest-rank one.
        available = [impl for impl in impls.values() if impl.available]
        if not available:
            raise ImportError(f"No available implementations found for {cls.__name__}")

        # Select the lowest-rank implementation
        selected = min(available, key=lambda impl: impl.rank)

        # Get the preferred implementation
        preferred = min(impls.values(), key=lambda impl: impl.rank)

        # Emit a one-time warning if we had to fall back from the preferred impl.
        cls._warn_fallback(preferred, selected)

        # Execute the selected implementation.
        return selected.func(*args, **kwargs)

    @classmethod
    def _get_impls(cls) -> Dict[str, Implementation]:
        """Return the implementation registry for the class.

        Returns
        -------
        Dict[str, Implementation]
            Mapping of implementation names to Implementation objects.
        """
        return cls._impl_registry.get(cls._class_key(), {})

    @classmethod
    def _check_impl(
        cls, implementation: str | None, impls: Dict[str, Implementation] | None = None
    ) -> None:
        """Validate that the implementation name is registered.

        Parameters
        ----------
        implementation : str | None
            Implementation name to validate. ``None`` is a no-op.
        impls : Dict[str, Implementation] | None, optional
            Registry mapping to validate against, by default None.

        Raises
        ------
        KeyError
            If the implementation is not registered.
        """
        if impls is None:
            impls = cls._get_impls()
        if implementation is None:
            return
        if implementation not in impls:
            raise KeyError(
                f"No implementation named '{implementation}' for {cls.__name__}"
            )

    @classmethod
    def _warn_fallback(
        cls, preferred: Implementation | None, selected: Implementation
    ) -> None:
        """Emit a one-time warning if we fall back to a lower-priority implementation.

        Parameters
        ----------
        preferred : Implementation | None
            Preferred implementation (may be None if not registered).
        selected : Implementation
            Selected implementation after availability checks.
        """
        if preferred is None:
            return
        if selected.rank == preferred.rank:
            return
        key = cls._class_key()
        if key in cls._fallback_warned:
            return
        cls._fallback_warned.add(key)
        warnings.warn(
            f"{cls.__name__} falling back to implementation '{selected.name}' "
            f"(rank {selected.rank}); preferred is '{preferred.name}' "
            f"(rank {preferred.rank}) but is unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )

    @classmethod
    def _class_key(cls) -> str:
        """Return the registry key for the class.
        This is used to make sure implementations with the same name
        but different FunctionSpecs are not overridden.

        Returns
        -------
        str
            Fully qualified class name.
        """
        return f"{cls.__module__}.{cls.__qualname__}"

    @classmethod
    def _register_impl(
        cls,
        impl: Implementation,
        class_key: str | None = None,
    ) -> None:
        """Register a new implementation for the class.

        Parameters
        ----------
        impl : Implementation
            Implementation to register.
        class_key : str | None
            Optional class key override.
        """

        # Get the class key
        key = class_key or cls._class_key()

        # Set default implementation registry for the class key
        impls = cls._impl_registry.setdefault(key, {})

        # Check if we can register the implementation
        for existing in impls.values():
            if existing.rank == impl.rank:
                raise ValueError(
                    f"{cls.__name__}: duplicate rank {impl.rank} for '{impl.name}'"
                )
            if impl.baseline and existing.baseline:
                raise ValueError(
                    f"{cls.__name__}: baseline already set to '{existing.name}'"
                )
        if impl.name in impls:
            raise ValueError(
                f"{cls.__name__}: implementation '{impl.name}' already registered"
            )

        # Create and register the implementation
        impls[impl.name] = impl

    @classmethod
    def _check_imports(cls, required_imports: Sequence[str]) -> bool:
        """Check whether all required imports are available.

        Parameters
        ----------
        required_imports : Sequence[str]
            Import requirement strings.

        Returns
        -------
        bool
            True if all requirements are satisfied.
        """
        for requirement in required_imports:
            req = Requirement(requirement)
            module_name = req.name
            spec = str(req.specifier) or None
            if spec:
                normalized = spec.split(",")[0].strip()
                normalized = re.sub(r"^[<>=!~]+", "", normalized)
                if not normalized:
                    return False
                if not check_version_spec(module_name, normalized, hard_fail=False):
                    return False
            else:
                if importlib.util.find_spec(module_name) is None:
                    return False
        return True

    @classmethod
    def implementations(cls) -> Tuple[str, ...]:
        """Return all registered implementation names for this function.
        This is used for introspection and debugging.

        Returns
        -------
        Tuple[str, ...]
            Implementation names ordered by rank then name.
        """
        impls = cls._get_impls()
        ordered = sorted(impls.values(), key=lambda impl: (impl.rank, impl.name))
        return tuple(impl.name for impl in ordered)

    @classmethod
    def available_implementations(cls) -> Tuple[str, ...]:
        """Return implementation names whose required imports are satisfied.
        This is used for introspection and debugging.

        Returns
        -------
        Tuple[str, ...]
            Available implementation names ordered by rank then name.
        """
        impls = cls._get_impls()
        available = [impl for impl in impls.values() if impl.available]
        ordered = sorted(available, key=lambda impl: (impl.rank, impl.name))
        return tuple(impl.name for impl in ordered)

    ############################################################
    # Helper functions for converting between different backends
    ############################################################

    @staticmethod
    def warp_launch_context(tensor: torch.Tensor):
        """Helper for getting Warp device and stream for a torch tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor used to infer device/stream.

        Returns
        -------
        tuple[str | None, object | None]
            Warp device and stream.
        """
        try:
            wp = importlib.import_module("warp")
        except ImportError as exc:
            raise ImportError("warp is not available") from exc
        if tensor.device.type == "cuda":
            stream = wp.stream_from_torch(torch.cuda.current_stream(tensor.device))
            device = None
        else:
            stream = None
            device = "cpu"
        return device, stream
