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

import copy
import importlib
import inspect
import json
import logging
import os
import tarfile
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

import torch

import physicsnemo
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.util_compatibility import convert_ckp_apex
from physicsnemo.registry import ModelRegistry
from physicsnemo.utils.filesystem import _download_cached, _get_fs


class Module(torch.nn.Module):
    """The base class for all network models in PhysicsNeMo.

    This should be used as a direct replacement for torch.nn.module and provides
    additional functionality for saving and loading models, as well as
    handling file system abstractions.

    There is one important requirement for all models in PhysicsNeMo. They must
    have json serializable arguments in their __init__ function. This is
    required for saving and loading models and allow models to be instantiated
    from a checkpoint.

    Parameters
    ----------
    meta : ModelMetaData, optional
        Meta data class for storing info regarding model, by default None
    """

    _file_extension = ".mdlus"  # Set file extension for saving and loading
    __model_checkpoint_version__ = (
        "0.1.0"  # Used for file versioning and is not the same as physicsnemo version
    )
    __supported_model_checkpoint_version__ = (
        {}
    )  # Dict of supported model checkpoints and corresponding warnings messages

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
        self.register_buffer("device_buffer", torch.empty(0))
        self._setup_logger()

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
                    f"Argument '{key}' cannot be overridden for " f"{cls.__name__}."
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
            return cls
        elif _cls_name in registry.list_models():  # Built in registry
            return registry.factory(_cls_name)
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
            except AttributeError:
                # Cross fingers and hope for the best (maybe the class name changed)
                _cls = cls

        # This works with the importlib.metadata.EntryPoint
        if isinstance(_cls, importlib.metadata.EntryPoint):
            _cls = _cls.load()

        return _cls

    @classmethod
    def instantiate(cls, arg_dict: Dict[str, Any]) -> "Module":
        """Instantiate a model from a dictionary of arguments

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
        >>> from physicsnemo.models import Module
        >>> from physicsnemo.registry import ModelRegistry
        >>> registry = ModelRegistry()
        >>> model_entry = registry.factory('FullyConnected')
        >>> fcn = model_entry(**{'in_features': 10})
        >>> fcn
        FullyConnected(
          (layers): ModuleList(
            (0): FCLayer(
              (activation_fn): SiLU()
              (linear): Linear(in_features=10, out_features=512, bias=True)
            )
            (1-5): 5 x FCLayer(
              (activation_fn): SiLU()
              (linear): Linear(in_features=512, out_features=512, bias=True)
            )
          )
          (final_layer): FCLayer(
            (activation_fn): Identity()
            (linear): Linear(in_features=512, out_features=512, bias=True)
          )
        )
        """
        _cls = cls._get_class_from_args(arg_dict)
        return _cls(**arg_dict["__args__"])

    def debug(self):
        """Turn on debug logging"""
        self.logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"[%(asctime)s - %(levelname)s - {self.meta.name}] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        # TODO: set up debug log
        # fh = logging.FileHandler(f'physicsnemo-core-{self.meta.name}.log')

    def save(self, file_name: Union[str, None] = None, verbose: bool = False) -> None:
        """Simple utility for saving just the model

        Parameters
        ----------
        file_name : Union[str,None], optional
            File name to save model weight to. When none is provide it will default to
            the model's name set in the meta data, by default None
        verbose : bool, optional
            Whether to save the model in verbose mode which will include git hash, etc, by default False

        Raises
        ------
        ValueError
            If file_name does not end with .mdlus extension
        """

        if file_name is not None and not file_name.endswith(self._file_extension):
            raise ValueError(
                f"File name must end with {self._file_extension} extension"
            )

        # Strip out torch dynamo wrapper
        if isinstance(self, torch._dynamo.eval_frame.OptimizedModule):
            self._orig_mod.save(file_name, verbose)
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)

            torch.save(self.state_dict(), local_path / "model.pt")

            with open(local_path / "args.json", "w") as f:
                json.dump(self._args, f)

            # Save the physicsnemo version and git hash (if available)
            metadata_info = {
                "physicsnemo_version": physicsnemo.__version__,
                "mdlus_file_version": self.__model_checkpoint_version__,
            }

            if verbose:
                import git

                try:
                    repo = git.Repo(search_parent_directories=True)
                    metadata_info["git_hash"] = repo.head.object.hexsha
                except git.InvalidGitRepositoryError:
                    metadata_info["git_hash"] = None

            with open(local_path / "metadata.json", "w") as f:
                json.dump(metadata_info, f)

            # Once all files are saved, package them into a tar file
            with tarfile.open(local_path / "model.tar", "w") as tar:
                for file in local_path.iterdir():
                    tar.add(str(file), arcname=file.name)

            if file_name is None:
                file_name = self.meta.name + ".mdlus"

            # Save files to remote destination
            fs = _get_fs(file_name)
            fs.put(str(local_path / "model.tar"), file_name)

    @staticmethod
    def _check_checkpoint(local_path: str) -> bool:
        if not local_path.joinpath("args.json").exists():
            raise IOError("File 'args.json' not found in checkpoint")

        if not local_path.joinpath("metadata.json").exists():
            raise IOError("File 'metadata.json' not found in checkpoint")

        if not local_path.joinpath("model.pt").exists():
            raise IOError("Model weights 'model.pt' not found in checkpoint")

        if not local_path.joinpath("metadata.json").exists():
            raise IOError("Metadata 'metadata.json' not found in checkpoint")

    def load(
        self,
        file_name: str,
        map_location: Union[None, str, torch.device] = None,
        strict: bool = True,
    ) -> None:
        """Simple utility for loading the model weights from checkpoint

        Parameters
        ----------
        file_name : str
            Checkpoint file name
        map_location : Union[None, str, torch.device], optional
            Map location for loading the model weights, by default None will use model's device
        strict: bool, optional
            whether to strictly enforce that the keys in state_dict match, by default True

        Raises
        ------
        IOError
            If file_name provided does not exist or is not a valid checkpoint
        """

        # Download and cache the checkpoint file if needed
        cached_file_name = _download_cached(file_name)

        # Use a temporary directory to extract the tar file
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)

            # Open the tar file and extract its contents to the temporary directory
            with tarfile.open(cached_file_name, "r") as tar:
                tar.extractall(
                    path=local_path, members=list(Module._safe_members(tar, local_path))
                )

            # Check if the checkpoint is valid
            Module._check_checkpoint(local_path)

            # Load the model weights
            device = map_location if map_location is not None else self.device
            model_dict = torch.load(
                local_path.joinpath("model.pt"), map_location=device
            )
            self.load_state_dict(model_dict, strict=strict)

    @classmethod
    def from_checkpoint(
        cls, file_name: str, override_args: Optional[Dict[str, Any]] = None
    ) -> "Module":
        """Simple utility for constructing a model from a checkpoint

        Parameters
        ----------
        file_name : str
            Checkpoint file name
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

        Returns
        -------
        Module

        Raises
        ------
        IOError
            If file_name provided does not exist or is not a valid checkpoint
        """

        # Download and cache the checkpoint file if needed
        cached_file_name = _download_cached(file_name)

        # Use a temporary directory to extract the tar file
        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)

            # Open the tar file and extract its contents to the temporary directory
            with tarfile.open(cached_file_name, "r") as tar:
                tar.extractall(
                    path=local_path, members=list(Module._safe_members(tar, local_path))
                )

            # Check if the checkpoint is valid
            Module._check_checkpoint(local_path)

            # Load model arguments and instantiate the model
            with open(local_path.joinpath("args.json"), "r") as f:
                args = json.load(f)

            ckp_args = copy.deepcopy(args)

            # Load metadata to get version
            with open(local_path.joinpath("metadata.json"), "r") as f:
                metadata = json.load(f)
                version = metadata.get(
                    "mdlus_file_version", cls.__model_checkpoint_version__
                )

            # Get class from args
            _cls = Module._get_class_from_args(args)

            # Check if the checkpoint version is compatible with the current version
            # If not, apply backward compatibility mapping if method exists
            if version != _cls.__model_checkpoint_version__:
                if version in _cls.__supported_model_checkpoint_version__:
                    warnings.warn(_cls.__supported_model_checkpoint_version__[version])
                    args["__args__"] = _cls._backward_compat_arg_mapper(
                        version, args["__args__"]
                    )
                else:
                    raise IOError(
                        f"Model checkpoint version {version} is not compatible with current version {_cls.__model_checkpoint_version__}"
                    )

            # Override args["__args__"] with override_args
            if override_args is not None:
                _cls._override_args(args["__args__"], override_args)

            # Instantiate the model
            model = Module.instantiate(args)

            # Load the model weights
            model_dict = torch.load(
                local_path.joinpath("model.pt"), map_location=model.device
            )

            model_dict = convert_ckp_apex(ckp_args, override_args, model_dict)
            model.load_state_dict(model_dict, strict=False)
        return model

    @staticmethod
    def from_torch(
        torch_model_class: torch.nn.Module, meta: ModelMetaData = None
    ) -> "Module":
        """Construct a PhysicsNeMo module from a PyTorch module

        Parameters
        ----------
        torch_model_class : torch.nn.Module
            PyTorch module class
        meta : ModelMetaData, optional
            Meta data for the model, by default None

        Returns
        -------
        Module
        """

        # Define an internal class as before
        class PhysicsNeMoModel(Module):
            def __init__(self, *args, **kwargs):
                super().__init__(meta=meta)
                self.inner_model = torch_model_class(*args, **kwargs)

            def forward(self, x):
                return self.inner_model(x)

        # Get the argument names and default values of the PyTorch model's init method
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
        new_class_name = f"{torch_model_class.__name__}PhysicsNeMoModel"
        PhysicsNeMoModel.__name__ = new_class_name

        # Add this class to the dict of models classes
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
