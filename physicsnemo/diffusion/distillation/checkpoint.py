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

r"""Named training-state bundles built from existing PhysicsNeMo checkpoints."""

import random
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import fsspec
import numpy as np
import torch

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import checkpoint as _checkpoint


class _Stateful(Protocol):
    r"""Objects that expose restart state independently of their configuration."""

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class CheckpointStateBundle:
    r"""Save and restore named models and their training state together.
    A bundle groups your student, teacher, optimizers, and loss statistics.

    Use a bundle to resume distillation with the associated training state.
    For model weights alone, use the model's checkpoint API. Save after an
    optimizer update, then recreate the same models, noise schedules, and
    loss settings before loading. Pending gradients are not part of the
    checkpoint.

    Parameters
    ----------
    models : mapping
        Models to save, keyed by names such as ``student`` and ``fake_score``.
        Use the same names when resuming training.
    optimizers : mapping, optional
        Optimizers to save, keyed by the name of the model they update.
    optimizer_models : mapping, optional
        Map optimizer names to model names when they differ, for example
        ``{"student_head": "student"}``.
    schedulers, scalers : mapping, optional
        Learning-rate schedulers and gradient scalers keyed by optimizer name.
    stateful : mapping, optional
        Other objects to save, such as the loss or a data-loader cursor.
        Each needs ``state_dict`` and ``load_state_dict`` methods.

    Examples
    --------
    Register the student under a stable name and save its state at a chosen epoch.
    Loading that epoch restores the registered module and returns its progress data.

    >>> import tempfile
    >>> model = torch.nn.Linear(3, 2)
    >>> # The key must match when reconstructing a bundle for loading.
    >>> bundle = CheckpointStateBundle({"student": model})
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     bundle.save_checkpoint(directory, epoch=0, progress={"step": 0})
    ...     progress = bundle.load_checkpoint(directory, epoch=0)
    >>> progress
    {'step': 0}

    Notes
    -----
    Call save and load on every rank when training across devices. Restart
    with the same world size and parallel layout. The bundle uses PhysicsNeMo's
    checkpoint utilities and saves each rank's buffers and optional random state.
    Each optimizer saves a copy of its model's weights, even when optimizers
    share a model.
    """

    def __init__(
        self,
        models: Mapping[str, torch.nn.Module],
        *,
        optimizers: Mapping[str, torch.optim.Optimizer] | None = None,
        optimizer_models: Mapping[str, str] | None = None,
        schedulers: Mapping[str, Any] | None = None,
        scalers: Mapping[str, Any] | None = None,
        stateful: Mapping[str, _Stateful] | None = None,
    ) -> None:
        self.models = dict(models)
        self.optimizers = dict(optimizers or {})
        self.optimizer_models = dict(optimizer_models or {})
        self.schedulers = dict(schedulers or {})
        self.scalers = dict(scalers or {})
        self.stateful = dict(stateful or {})
        if not self.models:
            raise ValueError("A checkpoint bundle requires named models.")
        for collection in (
            self.models,
            self.optimizers,
            self.schedulers,
            self.scalers,
            self.stateful,
        ):
            if any(
                not isinstance(name, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None
                for name in collection
            ):
                raise ValueError(
                    "Checkpoint names must contain letters, digits, hyphens, or underscores."
                )
        if set(self.optimizer_models) - set(self.optimizers):
            raise ValueError("optimizer_models contains unknown optimizer names.")
        if (set(self.schedulers) | set(self.scalers)) - set(self.optimizers):
            raise ValueError("Schedulers and scalers must name configured optimizers.")
        for name, optimizer in self.optimizers.items():
            role = self.optimizer_models.setdefault(name, name)
            if role not in self.models:
                raise ValueError(f"Optimizer {name!r} needs a known model role.")
            parameters = set(self.models[role].parameters())
            if any(
                parameter not in parameters
                for group in optimizer.param_groups
                for parameter in group["params"]
            ):
                raise ValueError(
                    f"Optimizer {name!r} contains parameters outside model role {role!r}."
                )

    def _schema(self) -> dict[str, Any]:
        r"""Describe role associations independently of mapping insertion order."""
        return {
            "schema_version": 1,
            "models": sorted(self.models),
            "optimizer_models": dict(self.optimizer_models),
            "schedulers": sorted(self.schedulers),
            "scalers": sorted(self.scalers),
            "stateful": sorted(self.stateful),
            "world_size": torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1,
        }

    @staticmethod
    def _rank() -> int:
        r"""Initialize the same manager used by the underlying checkpoint utilities."""
        if not DistributedManager.is_initialized():
            DistributedManager.initialize()
        return DistributedManager().rank

    @staticmethod
    def _barrier() -> None:
        r"""Finish collective snapshots before publishing per-rank bundle metadata."""
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _entries(self, root: str, rank: int):
        r"""Yield standard checkpoint calls with explicit optimizer associations."""
        entries = [
            (f"models/{role}", model, None)
            for role, model in sorted(self.models.items())
        ]
        entries += [
            (f"optimizers/{name}", self.models[self.optimizer_models[name]], name)
            for name in sorted(self.optimizers)
        ]
        for name, model, optimizer_name in entries:
            named = _checkpoint._unique_model_names([model])
            distributed = any(
                _checkpoint._is_distributed_model(value) for value in named.values()
            )
            suffix = "distributed" if distributed else f"rank-{rank}"
            kwargs = {"models": model}
            if optimizer_name is not None:
                kwargs.update(
                    optimizer=self.optimizers[optimizer_name],
                    optimizer_model=model,
                    scheduler=self.schedulers.get(optimizer_name),
                    scaler=self.scalers.get(optimizer_name),
                )
            yield f"{root}/{name}/{suffix}", kwargs, named, distributed

    def save_checkpoint(
        self,
        path: Path | str,
        *,
        epoch: int,
        progress: Mapping[str, Any] | None = None,
        include_rng: bool = False,
    ) -> None:
        r"""Save the bundle's models and training state at one checkpoint index.

        Call after a completed optimizer update on every rank. Choose a new
        index for each save, including retries after an incomplete save.

        Parameters
        ----------
        path : Path or str
            Root directory or filesystem URI accepted by PhysicsNeMo checkpoints.
        epoch : int
            Nonnegative checkpoint index. Choose a new index for each save.
        progress : mapping, optional
            Counters to restore, such as ``{"step": 1000}``.
        include_rng : bool, optional, default=False
            Capture Python, NumPy, CPU, and available CUDA random states per rank.
        """
        if epoch < 0:
            raise ValueError("epoch must be nonnegative.")
        rank = self._rank()
        root = f"{str(path).rstrip('/')}/step-{epoch}"
        fs = fsspec.filesystem(fsspec.utils.get_protocol(root))
        exists = fs.exists(root)
        occupied = [exists]
        if torch.distributed.is_initialized():
            occupied = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(occupied, exists)
        if any(occupied):
            raise FileExistsError(f"Checkpoint index already exists: {root}.")
        self._barrier()
        metadata = {
            "bundle": self._schema(),
            "progress": dict(progress or {}),
            "stateful": {
                name: value.state_dict() for name, value in self.stateful.items()
            },
        }
        if include_rng:
            numpy_state = np.random.get_state()
            metadata["rng"] = {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
            }
        for directory, kwargs, _, _ in self._entries(root, rank):
            _checkpoint.save_checkpoint(
                directory, epoch=epoch, metadata={"bundle_epoch": epoch}, **kwargs
            )
        self._barrier()
        _checkpoint.save_checkpoint(
            f"{root}/state/rank-{rank}", epoch=epoch, metadata=metadata
        )
        self._barrier()

    def load_checkpoint(
        self,
        path: Path | str,
        *,
        epoch: int,
        device: str | torch.device = "cpu",
        restore_rng: bool = False,
    ) -> dict[str, Any]:
        r"""Load saved weights and training state into the objects in this bundle.

        To resume training, recreate the same objects and call on every rank
        with the saved world size and parallel layout. If loading fails,
        recreate the objects before retrying because some state may have loaded.

        Parameters
        ----------
        path : Path or str
            Root supplied to :meth:`save_checkpoint`.
        epoch : int
            Checkpoint index used when saving. All models load from this index.
        device : str or torch.device, optional, default="cpu"
            Device for checkpoint tensor loading.
        restore_rng : bool, optional, default=False
            Restore captured random states after all model and state loading.

        Returns
        -------
        dict
            Progress dictionary passed to :meth:`save_checkpoint`.
        """
        if epoch < 0:
            raise ValueError("epoch must be nonnegative.")
        rank = self._rank()
        root = f"{str(path).rstrip('/')}/step-{epoch}"
        state_path = f"{root}/state/rank-{rank}"
        fs = fsspec.filesystem(fsspec.utils.get_protocol(root))
        entries = list(self._entries(root, rank))
        error = None
        metadata = {}
        try:
            state_file, _ = _checkpoint._resolve_checkpoint_files(
                state_path, fs, {}, epoch
            )
            if state_file is None:
                raise FileNotFoundError(
                    f"Missing complete bundle state at {state_path}."
                )
            _checkpoint.load_checkpoint(
                state_path, epoch=epoch, metadata_dict=metadata, device=device
            )
            if metadata.get("bundle", {}).get("schema_version") != 1:
                raise ValueError("Unsupported checkpoint bundle schema version.")
            if metadata["bundle"] != self._schema():
                raise ValueError(
                    "Checkpoint bundle roles, associations, or world size do not match."
                )
            if restore_rng and "rng" not in metadata:
                raise ValueError("This checkpoint does not contain random states.")
            for directory, _, named, distributed in entries:
                state_file, model_files = _checkpoint._resolve_checkpoint_files(
                    directory, fs, named, epoch, distributed=distributed
                )
                if state_file is None:
                    raise FileNotFoundError(f"Missing role checkpoint at {directory}.")
                _checkpoint._validate_checkpoint_files(state_file, model_files)
        except Exception as exc:
            # Deserialization failures must reach every rank before collectives.
            error = (type(exc).__name__, str(exc))
        errors = [error]
        if torch.distributed.is_initialized():
            errors = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(errors, error)
        if any(errors):
            exception = (
                FileNotFoundError
                if all(
                    value is None or value[0] == "FileNotFoundError" for value in errors
                )
                else ValueError
            )
            raise exception(f"Invalid checkpoint bundle: {errors}")
        for directory, kwargs, _, _ in entries:
            _checkpoint.load_checkpoint(directory, epoch=epoch, device=device, **kwargs)
        for name, value in self.stateful.items():
            value.load_state_dict(metadata["stateful"][name])
        if restore_rng:
            rng = metadata["rng"]
            random.setstate(rng["python"])
            state = rng["numpy"]
            np.random.set_state(
                (state[0], np.asarray(state[1], dtype=np.uint32), *state[2:])
            )
            torch.set_rng_state(rng["torch"].cpu())
            if rng["cuda"]:
                torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
        return metadata["progress"]
