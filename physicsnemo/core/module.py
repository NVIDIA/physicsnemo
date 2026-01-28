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
import inspect
import io
import json
import keyword
import logging
import os
import re
import tarfile
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

import torch

from physicsnemo.core.filesystem import _download_cached, _get_fs
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.registry import ModelRegistry

# Used for saving checkpoints of nested modules
_BASE_CKPT_PREFIX = "__physicsnemo.Module__"


def _load_state_dict_with_logging(
    module: torch.nn.Module, state_dict: Dict[str, Any], *args, **kwargs
):
    """Load state dictionary and log missing and unexpected keys

    Parameters
    ----------
    module : torch.nn.Module
        Module to load state dictionary into
    state_dict : Dict[str, Any]
        State dictionary to load
    *args, **kwargs
        Additional arguments to pass to load_state_dict
    """
    missing_keys, unexpected_keys = module.load_state_dict(state_dict, *args, **kwargs)
    if missing_keys:
        logging.warning(
            f"Missing keys when loading {module.__class__.__name__}: {missing_keys}"
        )
    if unexpected_keys:
        logging.warning(
            f"Unexpected keys when loading {module.__class__.__name__}: {unexpected_keys}"
        )
    return missing_keys, unexpected_keys


def _ignore_device_buffer_keys(module, incompatible_keys):
    """Post-hook to ignore device_buffer in missing/unexpected keys.

    The device_buffer is a non-persistent buffer used for device tracking.
    Old checkpoints (before v2.0.0) may have it saved (when it was persistent), and new
    checkpoints won't have it. This hook ensures backward compatibility.
    """
    incompatible_keys.missing_keys[:] = [
        k for k in incompatible_keys.missing_keys if not k.endswith("device_buffer")
    ]
    incompatible_keys.unexpected_keys[:] = [
        k for k in incompatible_keys.unexpected_keys if not k.endswith("device_buffer")
    ]


class Module(torch.nn.Module):
    """The base class for all network models in PhysicsNeMo.

    This should be used as a direct replacement for torch.nn.module and provides
    additional functionality for saving and loading models, as well as
    handling file system abstractions.

    There is one important requirement for all models in PhysicsNeMo. They must
    have json serializable arguments in their ``__init__`` function. This is
    required for saving and loading models and allow models to be instantiated
    from a checkpoint. The only one exception to this rule is when the argument
    passed to the ``__init__`` function is itself a ``physicsnemo.Module`` instance.
    In this case, it is possible to construct, save and load nested Modules,
    with multiple levels of nesting and/or multiple ``physicsnemo.Module``
    instances at each level.
    To be able to pass a ``torch.nn.Module`` instance as an argument to the
    ``__init__`` function, it is necessary to first use the ``Module.from_torch`` method
    to convert the ``torch.nn.Module`` subclass to a ``physicsnemo.Module`` subclass
    To pass nested ``torch.nn.Module`` instances as arguments to the
    ``__init__`` function, it is necessary to convert **all** nested ``torch.nn.Module``
    instances to ``physicsnemo.Module`` instances using the
    ``Module.from_torch`` method. See the examples below for more details.

    Parameters
    ----------
    meta : ModelMetaData, optional
        Meta data class for storing info regarding model, by default None

    Examples
    --------
    To construct nested ``physicsnemo.Module`` instances with multiple levels of nesting and/or
    multiple ``physicsnemo.Module`` instances at each level:

    .. code-block:: python

        class InnerModel(physicsnemo.Module):
            def __init__(self, hidden_size):
                super().__init__(meta=ModelMetaData())
                self.hidden_size = hidden_size

        class OuterModel(physicsnemo.Module):
            def __init__(self, inner_model):
                super().__init__(meta=ModelMetaData())
                self.inner_model = inner_model

        # Create and save nested model
        model = OuterModel(inner_model=InnerModel(128))
        model.save("checkpoint.mdlus")
        loaded = physicsnemo.Module.from_checkpoint("checkpoint.mdlus")

    Applying this to a ``torch.nn.Module`` instance is also possible, as long
    as all nested ``torch.nn.Module`` instances are converted to ``physicsnemo.Module``
    instances using the ``Module.from_torch`` method:

    .. code-block:: python

        class TorchInnerModel(torch.nn.Module):
            def __init__(self, size):
                super().__init__()
                self.size = size

        class TorchMyModel(torch.nn.Module):
            def __init__(self, inner_model):
                super().__init__()
                self.inner_model = inner_model

        # Convert both torch.nn.Module to physicsnemo.Module
        PNMInnerModel = physicsnemo.Module.from_torch(
            TorchInnerModel, meta=ModelMetaData()
        )
        PNMMyModel = physicsnemo.Module.from_torch(
            TorchMyModel, meta=ModelMetaData()
        )

        # Create nested model with converted torch modules
        model = PNMMyModel(inner_model=PNMInnerModel(size=128))

    When subclassing ``Module``, you can pass ``register=True`` as a class argument
    to automatically register the class in the model registry. This allows the class to be retrieved later by name using
    the :class:`~physicsnemo.core.registry.ModelRegistry`:

    >>> from physicsnemo.core import Module, ModelMetaData, ModelRegistry
    >>> class MyCustomModelA(Module, register=True):
    ...     def __init__(self, hidden_dim=64):
    ...         super().__init__(meta=ModelMetaData())
    ...         self.hidden_dim = hidden_dim
    >>> # The class is now registered and can be retrieved by name
    >>> registry = ModelRegistry()
    >>> ModelClass = registry.factory('MyCustomModelA')
    >>> model = ModelClass(hidden_dim=128)
    >>> model.hidden_dim
    128

    """

    _file_extension = ".mdlus"  # Set file extension for saving and loading
    __model_checkpoint_version__ = (
        "0.1.0"  # Used for file versioning and is not the same as physicsnemo version
    )
    __supported_model_checkpoint_version__ = {}  # Dict of supported model checkpoints and corresponding warnings messages

    # __init__ arguments that can be overridden. By default all arguments are
    # protected. Subclasses can override this to allow for overriding of specific
    # __init__'s arguments with the ``from_checkpoint`` method.
    _overridable_args: Set[str] = set()

    def __new__(cls, *args, **kwargs):
        out = super().__new__(cls)

        # Get signature of __init__ function
        sig = inspect.signature(cls.__init__)

        # Bind args and kwargs to signature
        bound_args = sig.bind_partial(
            *([None] + list(args)), **kwargs
        )  # Add None to account for self
        bound_args.apply_defaults()

        # Get args and kwargs (excluding self and unroll kwargs)
        instantiate_args = {}
        for param, (k, v) in zip(sig.parameters.values(), bound_args.arguments.items()):
            # Skip self
            if k == "self":
                continue

            # Add args and kwargs to instantiate_args
            if param.kind == param.VAR_KEYWORD:
                instantiate_args.update(v)
            else:
                instantiate_args[k] = v

        # Store args needed for instantiation
        out._args = {
            "__name__": cls.__name__,
            "__module__": cls.__module__,
            "__args__": instantiate_args,
        }
        return out

    def __init__(self, meta: Union[ModelMetaData, None] = None):
        super().__init__()
        self.meta = meta
        self.register_buffer("device_buffer", torch.empty(0), persistent=False)
        self._setup_logger()
        # Register hook to handle device_buffer compatibility with old checkpoints
        self.register_load_state_dict_post_hook(_ignore_device_buffer_keys)

    def __init_subclass__(cls, *, register=False, **kwargs):
        """
        Register the subclass of Module in the model registry if register is
        True.

        Parameters
        ----------
        register : bool, optional, default=False
            For internal use only. Whether to register the subclass in the
            model registry.
        """
        super().__init_subclass__()
        if register:
            registry = ModelRegistry()
            registry.register(cls, cls.__name__)

    def _setup_logger(self):
        self.logger = logging.getLogger("core.module")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s - %(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.WARNING)

    @staticmethod
    def _safe_members(tar, local_path):
        for member in tar.getmembers():
            if (
                ".." in member.name
                or os.path.isabs(member.name)
                or os.path.realpath(os.path.join(local_path, member.name)).startswith(
                    os.path.realpath(local_path)
                )
            ):
                yield member
            else:
                print(f"Skipping potentially malicious file: {member.name}")

    @classmethod
    def _backward_compat_arg_mapper(
        cls, version: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map arguments from older versions to current version format.

        This base implementation does nothing. Child classes should override this method
        to handle version-specific argument mappings.

        Parameters
        ----------
        version : str
            Version of the checkpoint being loaded
        args : Dict[str, Any]
            Arguments dictionary from the checkpoint

        Returns
        -------
        Dict[str, Any]
            Updated arguments dictionary compatible with current version
        """
        return args

    @classmethod
    def _override_args(
        cls, args: Dict[str, Any], override_args: Dict[str, Any]
    ) -> None:
        """Safely override ``__init__`` arguments stored in a checkpoint.

        This updates ``args`` *in-place* with the values provided in
        ``override_args``. Only keys defined in ``cls._overridable_args`` are
        allowed to be modified. Attempting to override any other key will raise
        a ``ValueError``.

        Parameters
        ----------
        args : Dict[str, Any]
            Keyword arguments that will be forwarded to the model
            constructor (e.g. ``args["__args__"]`` from a checkpoint).
        override_args : Dict[str, Any]
            Dictionary containing the desired argument overrides.
        """

        for key, value in override_args.items():
            if key not in cls._overridable_args:
                raise ValueError(
                    f"Argument '{key}' cannot be overridden for {cls.__name__}."
                )
            # In this case we are not overriding, but we are adding a new arg
            if key not in args:
                warnings.warn(f"New argument '{key}' added for {cls.__name__}.")
            args[key] = value

    @classmethod
    def _get_class_from_args(cls, arg_dict: Dict[str, Any]) -> type:
        """Get the class from a dictionary of arguments.

        Parameters
        ----------
        arg_dict : Dict[str, Any]
            Dictionary of arguments containing '__name__' and '__module__' keys.

        Returns
        -------
        type
            The class to instantiate.

        Raises
        ------
        AttributeError
            If the class cannot be found.
        """

        _cls_name = arg_dict["__name__"]
        registry = ModelRegistry()

        if cls.__name__ == arg_dict["__name__"]:  # If cls is the class
            _cls = cls
        elif _cls_name in registry.list_models():  # Built in registry
            _cls = registry.factory(_cls_name)
        else:
            try:
                # Check if module is using modulus import and change it to physicsnemo instead
                if arg_dict["__module__"].split(".")[0] == "modulus":
                    warnings.warn(
                        "Using modulus import in model checkpoint. This is deprecated and will be removed in future versions. Please use physicsnemo instead."
                    )
                    arg_module = (
                        "physicsnemo" + arg_dict["__module__"][len("modulus") :]
                    )
                else:
                    arg_module = arg_dict["__module__"]

                # Otherwise, try to import the class
                _mod = importlib.import_module(arg_module)
                _cls = getattr(_mod, arg_dict["__name__"])
            except (AttributeError, ModuleNotFoundError):
                # Cross fingers and hope for the best (maybe the class name changed)
                _cls = cls

        # This works with the importlib.metadata.EntryPoint
        # if isinstance(_cls, importlib.metadata.EntryPoint):
        if "EntryPoint" in str(type(_cls)):
            # I hate myself for this.  Somehow, we've got crossvoer pollution from
            # importlib_metadata.EntryPoint.
            _cls = _cls.load()

        return _cls

    @classmethod
    def instantiate(cls, arg_dict: Dict[str, Any]) -> "Module":
        """
        Instantiate a model from a dictionary of arguments. This method is
        reserved for advanced and internal use cases. For most use cases, it
        is recommended to instantiate the model using standard instantiation
        mechanisms.

        Parameters
        ----------
        arg_dict : Dict[str, Any]
            Dictionary of arguments to instantiate model with. This should be
            have three keys: '__name__', '__module__', and '__args__'. The first two
            are used to import the class and the last is used to instantiate
            the class. The '__args__' key should be a dictionary of arguments
            to pass to the class's __init__ function.

        Returns
        -------
        Module

        Examples
        --------
        >>> from physicsnemo.core.module import Module
        >>> # Define the argument dictionary with the three required keys
        >>> arg_dict = {
        ...     '__name__': 'FullyConnected',
        ...     '__module__': 'physicsnemo.models.mlp.fully_connected',
        ...     '__args__': {'in_features': 10, 'out_features': 5}
        ... }
        >>> # Instantiate the model using the class method
        >>> model = Module.instantiate(arg_dict)
        >>> # Verify the model was created with the correct parameters
        >>> x = torch.randn(100, 10)
        >>> output = model(x)
        >>> output.shape
        torch.Size([100, 5])
        """
        _cls = cls._get_class_from_args(arg_dict)
        return _cls(**arg_dict["__args__"])

    def debug(self):
        """Turn on debug logging"""
        self.logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"[%(asctime)s - %(levelname)s - {type(self).__name__}] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        # TODO: set up debug log
        # fh = logging.FileHandler(f'physicsnemo-core-{type(self).__name__}.log')

    def save(
        self,
        file_name: Union[str, None] = None,
        verbose: bool = False,
        legacy_format: bool = False,
    ) -> None:
        """
        Utility method for saving a ``Module`` instance to a '.mdlus' checkpoint file.

        Parameters
        ----------
        file_name : Union[str,None], optional, default=None
            File name to save the model checkpoint to. When ``None`` is provided it will default to
            the model's class name.
        verbose : bool, optional, default=False
            Whether to save the model in verbose mode which will include git hash, etc.
        legacy_format : bool, optional, default=False
            Whether to save the model in legacy tar format. If True, saves as tar archive.
            If False (default), saves as zip archive.

        Raises
        ------
        ValueError
            If file_name does not end with .mdlus extension

        Examples
        --------
        >>> from physicsnemo.models.mlp import FullyConnected
        >>> model = FullyConnected(in_features=32, out_features=64)
        >>> # Save a checkpoint with the default file name 'FullyConnected.mdlus' (using the class name).
        >>> model.save()
        >>> # Save a checkpoint to a specified file name 'my_model.mdlus'
        >>> model.save("my_model.mdlus")
        """

        # Define some helper functions
        def _save_process(module, args, metadata, mod_prefix="") -> None:
            """Recursively serialize nested physicsnemo.Module instances for checkpoint saving.

            Performs a depth-first search through the module's ``__init__`` arguments. When
            an argument is a ``physicsnemo.Module`` instance, it is replaced with a
            placeholder string (prefixed with ``_BASE_CKPT_PREFIX``) and the nested module's
            information (``__name__``, ``__module__``, ``__args__``) is stored at the root level
            of the ``args`` dictionary. The nested module metadata (e.g.,
            ``__model_checkpoint_version__``) is also added at the root level
            of ``metadata`` dictionary, with keys prefixed with
            ``_BASE_CKPT_PREFIX``.

            This allows for reconstruction of arbitrarily nested
            module hierarchies during checkpoint loading.

            Parameters
            ----------
            module : physicsnemo.Module
                The module being processed
            args : Dict[str, Any]
                Dictionary to populate with serialized module arguments. Modified in-place.
                Keys prefixed with ``_BASE_CKPT_PREFIX`` store nested module metadata.
            metadata : Dict[str, Any]
                Dictionary to populate with module metadata (e.g., version info).
                Modified in-place.
            mod_prefix : str, optional
                Current module's prefix in the nested hierarchy, by default "". Root module
                uses empty string; nested modules use format ``_BASE_CKPT_PREFIX.arg_name``.

            Raises
            ------
            TypeError
                If an argument is a ``torch.nn.Module`` instance that has not been converted
                to a ``physicsnemo.Module`` using
                :meth:`~physicsnemo.core.module.Module.from_torch`.
            """

            # Pointer to args["__args__"] for submodules
            if mod_prefix == "":
                args_ptr = args["__args__"].copy()
            else:
                args_ptr = args[mod_prefix]["__args__"].copy()

            for arg_name, arg_value in args_ptr.items():
                if isinstance(arg_value, Module):
                    next_mod_prefix = (
                        f"{mod_prefix if mod_prefix else _BASE_CKPT_PREFIX}.{arg_name}"
                    )
                    args[next_mod_prefix] = arg_value._args.copy()
                    args_ptr[arg_name] = next_mod_prefix
                    metadata[f"{next_mod_prefix}.mdlus_file_version"] = (
                        arg_value.__model_checkpoint_version__
                    )
                    _save_process(arg_value, args, metadata, next_mod_prefix)
                elif isinstance(arg_value, torch.nn.Module):
                    raise TypeError(
                        f"Submodule {arg_name} of module {module.__class__.__name__} is"
                        f" a PyTorch module, which is not supported by 'Module.save'. Please "
                        f"first convert it to a PhysicsNeMo module using 'Module.from_torch'."
                    )

            if mod_prefix == "":
                args["__args__"] = args_ptr
            else:
                args[mod_prefix]["__args__"] = args_ptr

            return

        if file_name is not None and not file_name.endswith(self._file_extension):
            raise ValueError(
                f"File name must end with {self._file_extension} extension"
            )

        # Strip out torch dynamo wrapper
        if isinstance(self, torch._dynamo.eval_frame.OptimizedModule):
            self._orig_mod.save(file_name, verbose)
            return

        # Save the physicsnemo version and git hash (if available)
        metadata_info = {
            "physicsnemo_version": importlib.metadata.version("nvidia-physicsnemo"),
            "mdlus_file_version": self.__model_checkpoint_version__,
        }

        if verbose:
            import git

            try:
                repo = git.Repo(search_parent_directories=True)
                metadata_info["git_hash"] = repo.head.object.hexsha
            except git.InvalidGitRepositoryError:
                metadata_info["git_hash"] = None

        # Copy self._args to avoid side effects
        _args = self._args.copy()

        # Recursively populate _args and metadata_info with submodules
        # information
        _save_process(self, _args, metadata_info)

        # If file_name is not provided, use the model's class name
        if file_name is None:
            file_name = f"{type(self).__name__}.mdlus"

        # Write checkpoint file
        fs = _get_fs(file_name)

        if not legacy_format:
            # Save in zip format (default)
            try:
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                    tmp_path = tmp.name

                with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED) as archive:
                    # Save model state dict
                    state_dict_buffer = io.BytesIO()
                    torch.save(self.state_dict(), state_dict_buffer)
                    archive.writestr("model.pt", state_dict_buffer.getvalue())

                    # Save args
                    args_str = json.dumps(_args)
                    archive.writestr("args.json", args_str)

                    # Save metadata
                    metadata_str = json.dumps(metadata_info)
                    archive.writestr("metadata.json", metadata_str)

                # Upload to final destination
                fs.put(tmp_path, file_name)
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            # Save in legacy tar format
            with tempfile.TemporaryDirectory() as temp_dir:
                local_path = Path(temp_dir)

                # Save model state dict
                torch.save(self.state_dict(), local_path / "model.pt")

                # Save args
                with open(local_path / "args.json", "w") as f:
                    json.dump(_args, f)

                # Save metadata
                with open(local_path / "metadata.json", "w") as f:
                    json.dump(metadata_info, f)

                # Create tar archive
                with tarfile.open(local_path / "model.tar", "w") as tar:
                    for file in local_path.iterdir():
                        tar.add(str(file), arcname=file.name)

                # Upload to final destination
                fs.put(local_path / "model.tar", file_name)

    @staticmethod
    def _detect_checkpoint_format(file_path: str) -> str:
        """Detect whether checkpoint is zip or tar format

        Parameters
        ----------
        file_path : str
            Path to checkpoint file

        Returns
        -------
        str
            Either 'zip' or 'tar'

        Raises
        ------
        IOError
            If file format cannot be determined
        """
        try:
            # NOTE: the check for tarfile MUST come first, as older checkpoints
            # will be both zip and tar archives, but newer checkpoints will
            # only be zip.
            if tarfile.is_tarfile(file_path):
                return "tar"
            elif zipfile.is_zipfile(file_path):
                return "zip"
            else:
                raise IOError(
                    f"Checkpoint file {file_path} is neither a valid zip "
                    f"nor tar archive"
                )
        except Exception as e:
            raise IOError(
                f"Could not determine checkpoint format for {file_path}: {e}"
            ) from e

    @staticmethod
    def _check_checkpoint(local_path: Path | str) -> None:
        local_path = Path(local_path)
        expected_files = ["args.json", "metadata.json", "model.pt"]
        for file in expected_files:
            if not (local_path / file).exists():
                raise IOError(f"File '{file}' not found in checkpoint")

    def load(
        self,
        file_name: str,
        map_location: Union[None, str, torch.device] = None,
        strict: bool = True,
    ) -> None:
        """
        Utility method for loading the model weights from a '.mdlus'
        checkpoint file. Unlike
        :meth:`~physicsnemo.core.module.Module.from_checkpoint`, this method
        *does not* instantiate the model, but rather loads the ``state_dict`` for an
        already instantiated model.

        Parameters
        ----------
        file_name : str
            Checkpoint file name. Must be a valid '.mdlus' checkpoint file.
        map_location : Union[None, str, torch.device], optional, default=None
            Map location for loading the model weights, ``None`` will use the model's device.
        strict: bool, optional, default=True
            Whether to strictly enforce that the keys in ``state_dict`` match.

        Raises
        ------
        IOError
            If ``file_name`` provided does not exist or is not a valid checkpoint

        Examples
        --------
        Basic example loading the model weights (state_dict) from a checkpoint:

        .. code-block:: python

            from physicsnemo.models.mlp import FullyConnected

            # Create a model with the same architecture as the saved one
            model = FullyConnected(in_features=32, out_features=64)

            # Load the weights from checkpoint
            model.load("FullyConnected.mdlus")

        Loading with specific device mapping:

        .. code-block:: python

            import torch
            from physicsnemo.models.mlp import FullyConnected

            model = FullyConnected(in_features=32, out_features=64)

            # Load checkpoint to CPU even if it was saved on GPU
            model.load("FullyConnected.mdlus", map_location="cpu")

            # Or load to a specific GPU
            model.load("FullyConnected.mdlus", map_location=torch.device("cuda:0"))
        """

        # Download and cache the checkpoint file if needed
        cached_file_name = _download_cached(file_name)

        # Detect checkpoint format
        checkpoint_format = Module._detect_checkpoint_format(cached_file_name)

        device = map_location if map_location is not None else self.device

        if checkpoint_format == "zip":
            # Load directly from zip file (no extraction needed)
            with zipfile.ZipFile(cached_file_name, "r") as archive:
                # Check if all expected files are present
                expected_files = ["args.json", "metadata.json", "model.pt"]
                archive_files = archive.namelist()
                for expected_file in expected_files:
                    if expected_file not in archive_files:
                        raise IOError(f"File '{expected_file}' not found in checkpoint")

                # Read into memory
                model_bytes = archive.read("model.pt")

            # Load state dict after closing archive
            model_dict = torch.load(io.BytesIO(model_bytes), map_location=device)

            # Load state_dict into the model
            _load_state_dict_with_logging(self, model_dict, strict=strict)

        else:  # tar format (backward compatibility)
            # Use a temporary directory to extract the tar file
            with tempfile.TemporaryDirectory() as temp_dir:
                local_path = Path(temp_dir)

                # Open tar file and extract contents to temporary directory
                with tarfile.open(cached_file_name, "r") as tar:
                    # Safely extract while supporting Python < 3.12
                    extract_kwargs = dict(
                        path=local_path,
                        members=list(Module._safe_members(tar, local_path)),
                    )
                    if "filter" in tar.extractall.__code__.co_varnames:
                        extract_kwargs["filter"] = "data"
                    tar.extractall(**extract_kwargs)  # noqa: S202

                # Check if the checkpoint is valid
                Module._check_checkpoint(local_path)

                # Load the model weights
                model_dict = torch.load(
                    local_path.joinpath("model.pt"), map_location=device
                )

            # Load state dict into the model
            _load_state_dict_with_logging(self, model_dict, strict=strict)

    @classmethod
    def from_checkpoint(
        cls,
        file_name: str,
        override_args: Optional[Dict[str, Any]] = None,
        strict: bool = True,
    ) -> "Module":
        """
        Utility class method for instantiating and loading a ``Module``
        instance from a '.mdlus' checkpoint file.

        Parameters
        ----------
        file_name : str
            Checkpoint file name. Must be a valid '.mdlus' checkpoint file.
        override_args : Optional[Dict[str, Any]], optional, default=None
            Dictionary of arguments to override the ``__init__`` method's
            arguments saved in the checkpoint. The override of arguments occurs
            *before* the model is instantiated, which allows for *ad-hoc*
            modifications to the model's initialization. Argument overrides are
            however applied *before* the state-dict is loaded, which means that
            for parameters or buffers saved in the state-dict, the values
            contained in the state-dict will take precedence over the override.
            This might also result in unexpected behavior if the model is
            instantiated with different arguments than the ones saved in the
            checkpoint, and some mismatching keys are saved in the state-dict.

            *Note*: Only arguments defined in ``cls._overridable_args`` can be
            overridden. ``Module``'s subclasses by default disable this
            functionality, unless they explicity define an ``_overridable_args``
            class attribute. Attempting to override any other argument will raise
            a ``ValueError``. This API should be used with caution and only if
            you fully understand the implications of the override.
        strict : bool, optional
            Whether to strictly enforce that the keys in state_dict match, by default True

        Returns
        -------
        Module

        Raises
        ------
        IOError
            If file_name provided does not exist or is not a valid checkpoint

        Examples
        --------
        Simple argument override:

        .. code-block:: python

            class MyModel(Module):
                _overridable_args = set(["a", "b"])
                def __init__(self, a, b=2.0):
                    super().__init__()
                    # ... model implementation ...
            model = MyModel(1.0, b=2.0)
            model.save("checkpoint.mdlus")
            model_loaded = MyModel.from_checkpoint("checkpoint.mdlus", override_args={"a": 5.0})

        For nested module, override is possible with dot-separated syntax:

        .. code-block:: python

            class SubModule(Module):
                _overridable_args = set(["a"])
                def __init__(self, a):
                    super().__init__()
                    # ... submodule implementation ...
            class MyModel(Module):
                def __init__(self, submodule):
                    super().__init__()
                    self.submodule = submodule
                    # ... model implementation ...
            submodule = SubModule(1.0)
            model = MyModel(submodule)
            model.save("checkpoint.mdlus")
            model = MyModel.from_checkpoint("checkpoint.mdlus", override_args={"submodule.a": 2.0})
        """

        # Validate the format of override_args keys
        override_args = override_args or {}
        for k in override_args.keys():
            if not isinstance(k, str):
                raise ValueError(
                    f"All keys in override_args must be strings, got {type(k)} for key {k}"
                )
            if not all(
                p and p.isidentifier() and not keyword.iskeyword(p)
                for p in k.split(".")
            ):
                raise ValueError(
                    f"Key {k} in override_args does not match the expected format "
                    f"arg_name1.arg_name2..."
                )

        # Define some helper functions
        def _from_checkpoint_process(
            cls_in: type[Module],
            args: Dict[str, Any],
            metadata: Dict[str, Any],
            override_args: Dict[str, Any],
            strict: bool,
            mod_prefix: str = "",
        ):
            """Recursively deserialize and instantiate nested physicsnemo.Module instances.

            Performs a depth-first reconstruction of the module hierarchy from a checkpoint.
            When an argument value is a placeholder string (prefixed with ``_BASE_CKPT_PREFIX``),
            it is replaced with a recursively instantiated ``physicsnemo.Module`` instance.
            This is the reciprocal operation of ``_save_process``, reconstructing the original
            nested module structure from the serialized checkpoint data.

            Parameters
            ----------
            cls_in : type
                The class of the module to instantiate at the current recursion level
            args : Dict[str, Any]
                Dictionary containing serialized module arguments from the checkpoint.
                Keys prefixed with ``_BASE_CKPT_PREFIX`` contain nested module metadata.
                Modified in-place as nested modules are processed and removed.
            metadata : Dict[str, Any]
                Dictionary containing module metadata (e.g., version info) from the checkpoint.
                Modified in-place as nested modules are processed and removed.
            override_args : Dict[str, Any]
                Dictionary of arguments to override in the module's ``__init__`` method.
                Supports dot-separated syntax for nested module arguments.
            strict : bool
                Whether to strictly enforce that state_dict keys match when loading weights
            mod_prefix : str, optional
                Current module's prefix in the nested hierarchy, by default "". Root module
                uses empty string; nested modules use format ``_BASE_CKPT_PREFIX.arg_name``.

            Returns
            -------
            physicsnemo.Module
                The instantiated module with all nested submodules recursively constructed

            Raises
            ------
            IOError
                If the checkpoint version is incompatible with the current model version
            ValueError
                If argument names or prefixes don't match the expected format
            """

            # Pointer to args (for submodules)
            if mod_prefix == "":
                args_ptr = {
                    k: v for k, v in args.items() if not k.startswith(_BASE_CKPT_PREFIX)
                }
                override_args_ptr = {
                    k: v
                    for k, v in override_args.items()
                    if k.isidentifier() and not keyword.iskeyword(k)
                }
            else:
                args_ptr = args[mod_prefix]
                prefix = mod_prefix[len(_BASE_CKPT_PREFIX) + 1 :]
                override_args_ptr = {}
                for k, v in override_args.items():
                    if k.startswith(f"{prefix}."):
                        suffix = k[len(prefix) + 1 :]  # +1 for the dot
                        if suffix.isidentifier() and not keyword.iskeyword(suffix):
                            override_args_ptr[suffix] = v

            # Get the checkpoint version
            version = metadata.get(
                f"{mod_prefix}{'.' if mod_prefix else ''}mdlus_file_version",
                cls_in.__model_checkpoint_version__,
            )

            # Get the class from args
            _cls = cls_in._get_class_from_args(args_ptr)

            # Check if the checkpoint version is compatible with the current version
            # If not, apply backward compatibility mapping if method exists
            if version != _cls.__model_checkpoint_version__:
                if version in _cls.__supported_model_checkpoint_version__:
                    warnings.warn(_cls.__supported_model_checkpoint_version__[version])
                    args_ptr["__args__"] = _cls._backward_compat_arg_mapper(
                        version, args_ptr["__args__"]
                    )
                else:
                    raise IOError(
                        f"Model checkpoint version {version} is not compatible with "
                        f"current version {_cls.__model_checkpoint_version__} of class "
                        f"{_cls.__name__}"
                    )

            # Process all args and recursively instantiate those that are
            # submodules
            for arg_name, arg_value in args_ptr["__args__"].items():
                if not isinstance(arg_value, str):
                    continue
                is_module = re.match(rf"{_BASE_CKPT_PREFIX}(.*)", arg_value)
                if is_module:
                    suffix = is_module.group(1)
                    args_split = re.match(r"^(.*\.)*([^\.]+)$", suffix)
                    if args_split:
                        _arg_name = args_split.group(2)
                        # Make sure that arg_value has the expected format
                        if _arg_name != arg_name:
                            raise ValueError(
                                f"Argument name '{_arg_name}' does not match the "
                                f"expected '{arg_name}' for module {_cls.__name__}"
                            )
                        # Instantiate the submodule
                        next_mod_prefix = arg_value
                        args_ptr["__args__"][arg_name] = _from_checkpoint_process(
                            Module._get_class_from_args(args[next_mod_prefix]),
                            args,
                            metadata,
                            override_args,
                            strict,
                            mod_prefix=next_mod_prefix,
                        )
                        # Cleanup args and metadata by removing the items
                        # related to the submodule
                        args.pop(next_mod_prefix, None)
                        metadata.pop(f"{next_mod_prefix}.mdlus_file_version", None)
                    else:
                        # Make sure that arg_value has the expected format
                        raise ValueError(
                            f"Argument value '{arg_value}' for argument '{arg_name}' "
                            f"of module {_cls.__name__} does not match the expected format "
                            f"{_BASE_CKPT_PREFIX}.arg_name1.arg_name2..."
                        )

            # Override args_ptr["__args__"] with override_args
            if override_args is not None:
                _cls._override_args(args_ptr["__args__"], override_args_ptr)

            # Instantiate the module
            model = cls_in.instantiate(args_ptr)
            return model

        # Download and cache the checkpoint file if needed
        cached_file_name = _download_cached(file_name)

        # Detect checkpoint format
        checkpoint_format = Module._detect_checkpoint_format(cached_file_name)

        if checkpoint_format == "zip":
            # Load directly from zip file (no extraction needed)
            with zipfile.ZipFile(cached_file_name, "r") as archive:
                # Check if all expected files are present
                expected_files = ["args.json", "metadata.json", "model.pt"]
                archive_files = archive.namelist()
                for expected_file in expected_files:
                    if expected_file not in archive_files:
                        raise IOError(f"File '{expected_file}' not found in checkpoint")

                # Load model arguments and instantiate the model
                with archive.open("args.json") as f:
                    args = json.loads(f.read().decode("utf-8"))

                # Load metadata to get version
                with archive.open("metadata.json") as f:
                    metadata = json.loads(f.read().decode("utf-8"))

                model = _from_checkpoint_process(
                    cls,
                    args,
                    metadata,
                    override_args,
                    strict,
                )

                # Read into memory
                model_bytes = archive.read("model.pt")

            # Load state dict after closing archive
            model_dict = torch.load(io.BytesIO(model_bytes), map_location=model.device)

            # Load state_dict into the model
            _load_state_dict_with_logging(model, model_dict, strict=strict)

        else:  # tar format (backward compatibility)
            # Use a temporary directory to extract the tar file
            with tempfile.TemporaryDirectory() as temp_dir:
                local_path = Path(temp_dir)

                # Open tar file and extract contents to temporary directory
                with tarfile.open(cached_file_name, "r") as tar:
                    # Safely extract while supporting Python < 3.12
                    extract_kwargs = dict(
                        path=local_path,
                        members=list(Module._safe_members(tar, local_path)),
                    )
                    if "filter" in tar.extractall.__code__.co_varnames:
                        extract_kwargs["filter"] = "data"
                    tar.extractall(**extract_kwargs)  # noqa: S202

                # Check if the checkpoint is valid
                Module._check_checkpoint(local_path)

                # Load model arguments and instantiate the model
                with open(local_path.joinpath("args.json"), "r") as f:
                    args = json.load(f)

                # Load metadata to get version
                with open(local_path.joinpath("metadata.json"), "r") as f:
                    metadata = json.load(f)

                model = _from_checkpoint_process(
                    cls,
                    args,
                    metadata,
                    override_args,
                    strict,
                )

                # Load the model weights
                model_dict = torch.load(
                    local_path.joinpath("model.pt"), map_location=model.device
                )

            # Load state_dict into the model
            _load_state_dict_with_logging(model, model_dict, strict=strict)

        return model

    @staticmethod
    def from_torch(
        torch_model_class: type[torch.nn.Module],
        meta: ModelMetaData | None = None,
        name: str | None = None,
        register: bool = False,
    ) -> type[Module]:
        """
        Construct a PhysicsNeMo module from a PyTorch module. The resulting
        class is a PhysicsNeMo Module class. Any instance of this class will be
        a PhysicsNeMo Module instance with an attribute ``inner_model`` that is an
        instance of the PyTorch model class.

        Parameters
        ----------
        torch_model_class : torch.nn.Module
            PyTorch module class
        meta : ModelMetaData, optional, default=None
            Meta data for the model.
        name : str, optional, default=None
            Name of the PhysicsNeMo model class. Used for registering the class in the
            model registry. If None, the name of the PyTorch model class is
            used.
        register : bool, optional, default=False
            Whether to register the class in the model registry. If True, the
            class will be registered and can be retrieved later using
            ``ModelRegistry().factory(name)``.

            .. important::

                To be able to later load the model with
                :meth:`~physicsnemo.core.module.Module.from_checkpoint`, it is
                necessary to register the class in the model registry by setting
                ``register=True``. A class created via ``from_torch`` that is not
                registered will not be able to be loaded with ``from_checkpoint``.

        Returns
        -------
        Module

        Examples
        --------
        Example 1: Convert a PyTorch model to PhysicsNeMo without specifying a name:

        >>> import torch
        >>> import torch.nn as nn
        >>> from physicsnemo.core import Module, ModelMetaData, ModelRegistry
        >>> # Define a simple MLP in PyTorch
        >>> class SimpleMLP(nn.Module):
        ...     def __init__(self, input_size, hidden_size, output_size):
        ...         super().__init__()
        ...         self.input_size = input_size
        ...         self.hidden_size = hidden_size
        ...         self.output_size = output_size
        ...         self.fc1 = nn.Linear(input_size, hidden_size)
        ...         self.relu = nn.ReLU()
        ...         self.fc2 = nn.Linear(hidden_size, output_size)
        ...
        ...     def forward(self, x):
        ...         x = self.fc1(x)
        ...         x = self.relu(x)
        ...         x = self.fc2(x)
        ...         return x
        >>> # Convert PyTorch model to PhysicsNeMo Module without registering
        >>> PNMSimpleMLP = Module.from_torch(SimpleMLP, meta=ModelMetaData())
        >>> # The physicsnemo class name is the same as the PyTorch class name
        >>> PNMSimpleMLP.__name__
        'SimpleMLP'
        >>> # Instantiate the PhysicsNeMo model
        >>> model = PNMSimpleMLP(input_size=10, hidden_size=64, output_size=5)
        >>> # Access the inner PyTorch model
        >>> assert model.inner_model.input_size == 10
        >>> assert model.inner_model.hidden_size == 64
        >>> assert model.inner_model.output_size == 5
        >>> # Use the model for inference
        >>> x = torch.randn(32, 10)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 5])
        >>> # Cannot retrieve the model class from the registry because it is not registered

        Example 2: Convert a PyTorch model with a custom name:

        >>> import torch
        >>> import torch.nn as nn
        >>> from physicsnemo.core import Module, ModelMetaData, ModelRegistry
        >>> # Define a simple MLP in PyTorch
        >>> class SimpleMLP(nn.Module):
        ...     def __init__(self, input_size, hidden_size, output_size):
        ...         super().__init__()
        ...         self.input_size = input_size
        ...         self.hidden_size = hidden_size
        ...         self.output_size = output_size
        ...         self.fc1 = nn.Linear(input_size, hidden_size)
        ...         self.relu = nn.ReLU()
        ...         self.fc2 = nn.Linear(hidden_size, output_size)
        ...
        ...     def forward(self, x):
        ...         x = self.fc1(x)
        ...         x = self.relu(x)
        ...         x = self.fc2(x)
        ...         return x
        >>> # Convert with a custom name
        >>> PNMSimpleMLP = Module.from_torch(
        ...     SimpleMLP,
        ...     meta=ModelMetaData(),
        ...     name='CustomSimpleMLP'
        ... )
        >>> # The physicsnemo class name is defined by the name parameter
        >>> PNMSimpleMLP.__name__
        'CustomSimpleMLP'
        >>> # Instantiate the PhysicsNeMo model
        >>> model = PNMSimpleMLP(input_size=10, hidden_size=64, output_size=5)
        >>> # Access the inner PyTorch model
        >>> assert model.inner_model.input_size == 10
        >>> assert model.inner_model.hidden_size == 64
        >>> assert model.inner_model.output_size == 5
        >>> # Cannot retrieve the model class from the registry because it is not registered


        >>> import torch
        >>> import torch.nn as nn
        >>> from physicsnemo.core import Module, ModelMetaData, ModelRegistry
        >>> # Define a simple MLP in PyTorch
        >>> class SimpleMLP(nn.Module):
        ...     def __init__(self, input_size, hidden_size, output_size):
        ...         super().__init__()
        ...         self.input_size = input_size
        ...         self.hidden_size = hidden_size
        ...         self.output_size = output_size
        ...         self.fc1 = nn.Linear(input_size, hidden_size)
        ...         self.relu = nn.ReLU()
        ...         self.fc2 = nn.Linear(hidden_size, output_size)
        ...
        ...     def forward(self, x):
        ...         x = self.fc1(x)
        ...         x = self.relu(x)
        ...         x = self.fc2(x)
        ...         return x
        >>> # Convert with register=True to add to the model registry
        >>> PNMSimpleMLP = Module.from_torch(
        ...     SimpleMLP,
        ...     meta=ModelMetaData(),
        ...     name='RegisteredMLP',
        ...     register=True
        ... )
        >>> # The class is now registered and can be retrieved by name
        >>> registry = ModelRegistry()
        >>> ModelClass = registry.factory('RegisteredMLP')
        >>> isinstance(ModelClass, type) and issubclass(ModelClass, Module)
        True
        >>> # The class name is defined by the name parameter
        >>> ModelClass.__name__
        'RegisteredMLP'
        >>> # Instantiate the model from the registry
        >>> model = ModelClass(input_size=10, hidden_size=64, output_size=5)
        >>> model.inner_model.input_size
        10

        """

        # Define an internal class as before
        class PhysicsNeMoModel(Module):
            def __init__(self, *args, **kwargs):
                super().__init__(meta=meta)
                self.inner_model = torch_model_class(*args, **kwargs)

            def forward(self, x):
                return self.inner_model(x)

        # Get the argument names and default values of the PyTorch model's init
        # method
        init_argspec = inspect.getfullargspec(torch_model_class.__init__)
        model_argnames = init_argspec.args[1:]  # Exclude 'self'
        model_defaults = init_argspec.defaults or []
        defaults_dict = dict(
            zip(model_argnames[-len(model_defaults) :], model_defaults)
        )

        # Define the signature of new init
        params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        params += [
            inspect.Parameter(
                argname,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=defaults_dict.get(argname, inspect.Parameter.empty),
            )
            for argname in model_argnames
        ]
        init_signature = inspect.Signature(params)

        # Replace PhysicsNeMoModel.__init__ signature with new init signature
        PhysicsNeMoModel.__init__.__signature__ = init_signature

        # Generate a unique name for the created class
        new_class_name = f"{torch_model_class.__name__}" if name is None else name
        PhysicsNeMoModel.__name__ = new_class_name

        # Register the class in the model registry if requested
        if register:
            registry = ModelRegistry()
            registry.register(PhysicsNeMoModel, new_class_name)

        return PhysicsNeMoModel

    @property
    def device(self) -> torch.device:
        """Get device model is on

        Returns
        -------
        torch.device
            PyTorch device
        """
        return self.device_buffer.device

    def num_parameters(self) -> int:
        """Gets the number of learnable parameters"""
        count = 0
        for name, param in self.named_parameters():
            count += param.numel()
        return count
