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

import importlib
import inspect
import json
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Union, Optional


import torch

import modulus
from modulus.models.meta import ModelMetaData
from modulus.registry import ModelRegistry
from modulus.utils.filesystem import _download_cached, _get_fs


class Module(torch.nn.Module):
    """The base class for all network models in Modulus.

    This should be used as a direct replacement for torch.nn.module and provides
    additional functionality for saving and loading models, as well as
    handling file system abstractions.

    There is one important requirement for all models in Modulus. They must
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
        "0.1.0"  # Used for file versioning and is not the same as modulus version
    )

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
        >>> from modulus.models import Module
        >>> fcn = Module.instantiate({'__name__': 'FullyConnected', '__module__': 'modulus.models.mlp', '__args__': {'in_features': 10}})
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

        _cls_name = arg_dict["__name__"]
        registry = ModelRegistry()
        if cls.__name__ == arg_dict["__name__"]:  # If cls is the class
            _cls = cls
        elif _cls_name in registry.list_models():  # Built in registry
            _cls = registry.factory(_cls_name)
        else:
            try:
                # Otherwise, try to import the class
                _mod = importlib.import_module(arg_dict["__module__"])
                _cls = getattr(_mod, arg_dict["__name__"])
            except AttributeError:
                # Cross fingers and hope for the best (maybe the class name changed)
                _cls = cls
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
        # fh = logging.FileHandler(f'modulus-core-{self.meta.name}.log')

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

        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir)

            torch.save(self.state_dict(), local_path / "model.pt")

            with open(local_path / "args.json", "w") as f:
                json.dump(self._args, f)

            # Save the modulus version and git hash (if available)
            metadata_info = {
                "modulus_version": modulus.__version__,
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

        # Check if the checkpoint version is compatible with the current version
        with open(local_path.joinpath("metadata.json"), "r") as f:
            metadata_info = json.load(f)
            if (
                metadata_info["mdlus_file_version"]
                != Module.__model_checkpoint_version__
            ):
                raise IOError(
                    f"Model checkpoint version {metadata_info['mdlus_file_version']} is not compatible with current version {Module.__version__}"
                )

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
    def from_checkpoint(cls, file_name: str, model_args: Optional[Dict] = None) -> "Module":
        """Simple utility for constructing a model from a checkpoint

        Parameters
        ----------
        file_name : str
            Checkpoint file name

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
                    path=local_path, members=list(cls._safe_members(tar, local_path))
                )

            # Check if the checkpoint is valid
            Module._check_checkpoint(local_path)

            # Load model arguments and instantiate the model
            with open(local_path.joinpath("args.json"), "r") as f:
                args = json.load(f)
            
            # Merge model_args (adding new keys and updating existing ones)
            args["__args__"].update(model_args)
            model = cls.instantiate(args)
            # Load the model weights
            model_dict = torch.load(
                local_path.joinpath("model.pt"), map_location=model.device
            )
            if "use_apex_gn" in args["__args__"].keys() and args["__args__"]["use_apex_gn"]:
                filtered_state_dict = {}
                for key, value in model_dict.items():
                    filtered_state_dict[key] = value  # Keep the original key
                    # Duplicate weight/bias for Apex GroupNorm (without removing the original)
                    for norm_layer in ["norm0", "norm1", "norm2", "aux_norm"]:
                        if f"{norm_layer}.weight" in key:
                            new_key = key.replace(f"{norm_layer}.weight", f"{norm_layer}.gn.weight")
                            filtered_state_dict[new_key] = value  # Duplicate weight
                        elif f"{norm_layer}.bias" in key:
                            new_key = key.replace(f"{norm_layer}.bias", f"{norm_layer}.gn.bias")
                            filtered_state_dict[new_key] = value  # Duplicate bias
                model.load_state_dict(filtered_state_dict,strict=False)
            else:
                model.load_state_dict(model_dict,strict=False)

        return model

    @staticmethod
    def from_torch(
        torch_model_class: torch.nn.Module, meta: ModelMetaData = None
    ) -> "Module":
        """Construct a Modulus module from a PyTorch module

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
        class ModulusModel(Module):
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

        # Replace ModulusModel.__init__ signature with new init signature
        ModulusModel.__init__.__signature__ = init_signature

        # Generate a unique name for the created class
        new_class_name = f"{torch_model_class.__name__}ModulusModel"
        ModulusModel.__name__ = new_class_name

        # Add this class to the dict of models classes
        registry = ModelRegistry()
        registry.register(ModulusModel, new_class_name)

        return ModulusModel

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
