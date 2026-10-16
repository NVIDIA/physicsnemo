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

r"""Optional diffusion orchestration through the general Alchemi training loop."""

import importlib
import math
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

from .checkpoint import CheckpointStateBundle

_backend = importlib.import_module("nvalchemi.training.strategy")
_TrainingStrategy = _backend.TrainingStrategy
_UpdateHook = _backend.TrainingUpdateHook
Field, PrivateAttr = _backend.Field, _backend.PrivateAttr
field_validator, model_validator = _backend.field_validator, _backend.model_validator

_COUNTERS = (
    "step_count",
    "global_step_count",
    "batch_count",
    "epoch_count",
    "epoch_step_count",
)


def _batch_arguments(
    models: Mapping[str, torch.nn.Module], batch: TensorDict
) -> TensorDict:
    r"""Pass loss arguments without evaluating any model before the loss."""
    return batch


class _Accumulation(_UpdateHook):
    r"""Keep one optimization phase across a complete accumulation window."""

    priority = 10

    def __call__(self, ctx: Any, stage: Any, will_skip: bool):
        r"""Scale the backward loss and veto intermediate updates."""
        strategy = ctx.workflow
        if stage == _backend.TrainingStage.BEFORE_BATCH:
            return strategy._micro_step == 0, ctx.loss
        if stage == _backend.TrainingStage.DO_BACKWARD:
            return True, ctx.loss / strategy.accumulation_steps
        if stage == _backend.TrainingStage.DO_OPTIMIZER_STEP:
            return strategy._micro_step + 1 == strategy.accumulation_steps, ctx.loss
        return True, ctx.loss


class _JointScaledUpdate(_UpdateHook):
    r"""Skip the entire phase if scaling detects a nonfinite active gradient."""

    priority = 25

    def __call__(self, ctx: Any, stage: Any, will_skip: bool):
        r"""Coordinate fake-score and discriminator updates after unscaling."""
        if (
            stage == _backend.TrainingStage.DO_OPTIMIZER_STEP
            and not will_skip
            and ctx.grad_scaler is not None
        ):
            found_inf = [
                value
                for optimizer in ctx.optimizers
                for value in ctx.grad_scaler._found_inf_per_device(optimizer).values()
            ]
            if any(bool(value.item()) for value in found_inf):
                ctx.grad_scaler.update()
                return False, ctx.loss
        return True, ctx.loss


class DistillationStrategy(_TrainingStrategy):
    r"""Train diffusion models with a loss and named optimizers.

    The strategy manages alternating phases, gradient accumulation, and
    checkpoints. Use it to run a distillation loss without writing an optimizer
    loop, or call the same loss directly when your recipe needs a custom loop.
    Install ``nvidia-physicsnemo[diffusion-strategy]``, supply your models and
    optimizer settings, and pass batches to ``run``. Custom losses do not
    require FastGen.

    Parameters
    ----------
    models : mapping
        Models keyed by name, including ``student`` and optional
        ``teacher``, ``fake_score``, ``discriminator``, or EMA targets.
    loss_fn : callable
        Loss accepting batch fields, ``phase``, ``step``, and ``update_state=True``.
        Return a dictionary containing ``total_loss``. The strategy computes gradients
        through its mean, divided by ``accumulation_steps``.
    optimizer_configs : mapping
        Per-role lists of dictionaries with ``optimizer_cls`` and
        ``optimizer_kwargs``, plus optional scheduler settings.
    num_steps, num_epochs : int, optional
        Supply exactly one training duration. Steps count completed updates.
    phase_fn : callable, optional
        Map completed update count to a phase name. The default trains only the
        student, or one student update followed by four auxiliary updates.
    phase_roles : mapping, optional
        Optimizer roles to update in each phase, for example
        ``{"student": ("student",), "auxiliary": ("fake_score",)}``.
        This controls optimizer and scheduler steps, not gradient computation.
    accumulation_steps : int, optional, default=1
        Batches per update. Phase selection stays fixed during accumulation.
    bind_models_fn : callable, optional
        Callback ``fn(loss_fn, models)`` that updates model references in a
        custom loss if a setup hook wraps or replaces a model.
        Built-in losses handle this automatically.
    precision : torch.dtype, optional
        Autocast precision. ``torch.float16`` also enables gradient scaling and
        skips all active optimizers if any active gradient is infinite or NaN.
    ema_models : mapping, optional
        Models to average and their target names, such as
        ``{"student": "ema"}``. Initialize the targets and include them in ``models``.
    ema_decay : float, optional, default=0.999
        Decay applied only after a successful update of the source role.
    hooks, devices : sequence, optional
        Training hooks and devices following the underlying training strategy.

    Examples
    --------
    This strategy trains one student model for a single batch with Adam.
    The batch supplies clean samples, and the final check confirms one optimizer step.

    >>> import torch
    >>> from tensordict import TensorDict
    >>> model = torch.nn.Linear(3, 3)
    >>> def loss_fn(x0, **kwargs):
    ...     return {"total_loss": model(x0).square().mean()}
    >>> # Use the same model key for the optimizer configuration.
    >>> strategy = DistillationStrategy(
    ...     models={"student": model}, loss_fn=loss_fn, num_steps=1,
    ...     optimizer_configs={"student": [{"optimizer_cls": torch.optim.Adam,
    ...                                      "optimizer_kwargs": {"lr": 1e-3}}]},
    ... )
    >>> strategy.train_batch(TensorDict({"x0": torch.randn(2, 3)}, [2]))
    >>> strategy.step_count
    1

    Notes
    -----
    Put ``x0`` and optional loss arguments in each ``TensorDict`` batch.
    For explicit MeanFlow times, store ``t`` and ``r`` as separate batch
    fields. Use ``training_fn(models, batch)`` to return a dictionary of loss
    arguments with ``t=(batch["t"], DiffusionExtraTimes(r=batch["r"],
    batch_size=batch.batch_size))``. Do not store the tuple inside a TensorDict.
    Reserve ``phase``, ``step``, ``update_state``, and ``return_outputs`` for the
    strategy. Custom losses control gradient flow through their model calls;
    the strategy preserves existing frozen parameters and training modes.
    Keep inference and validation in your recipe.

    Compile and wrap models before constructing the loss and strategy.
    With ``num_epochs``, the total batch budget must divide evenly by
    ``accumulation_steps``. Save only at completed accumulation boundaries.
    Recreate matching settings, hooks, and phase callbacks before loading.
    Checkpoint your data-loader cursor separately for exact batch resumption.
    """

    models: dict[str, torch.nn.Module]
    loss_fn: Any
    training_fn: Any = _batch_arguments
    phase_fn: Callable[[int], str] | None = None
    phase_roles: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    accumulation_steps: int = Field(default=1, ge=1)
    bind_models_fn: Callable[..., None] | None = None
    precision: torch.dtype | None = None
    ema_models: dict[str, str] = Field(default_factory=dict)
    ema_decay: float = Field(default=0.999, ge=0, le=1)
    _micro_step: int = PrivateAttr(default=0)
    _phase: str = PrivateAttr(default="student")
    _bound_models: dict[str, torch.nn.Module] = PrivateAttr(default_factory=dict)
    _model_modes: dict[torch.nn.Module, bool] = PrivateAttr(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __exit__(self, exc_type, exc, tb) -> None:
        r"""Close autocast contexts while keeping scaler state across updates."""
        scalers = []
        for hook in self.hooks:
            children = (
                hook._hooks
                if isinstance(hook, _backend.TrainingUpdateOrchestrator)
                else [hook]
            )
            scalers.extend(
                (child, child._scaler)
                for child in children
                if isinstance(child, _backend.MixedPrecisionHook)
            )
        try:
            super().__exit__(exc_type, exc, tb)
        finally:
            for hook, scaler in scalers:
                hook._scaler = scaler

    @model_validator(mode="before")
    @classmethod
    def _normalize_inputs(cls, data: Any) -> Any:
        r"""Accept native modules without atomic-model or target requirements."""
        if not isinstance(data, dict):
            return data
        values = dict(data)
        values["single_model_input"] = False
        hooks = list(values.get("hooks", ()))
        hooks.append(_Accumulation())
        if values.get("precision") is not None:
            hooks.append(_backend.MixedPrecisionHook(precision=values["precision"]))
        hooks.append(_JointScaledUpdate())
        values["hooks"] = hooks
        return values

    @field_validator("loss_fn", mode="before")
    @classmethod
    def _normalize_loss_fn(cls, value: Any) -> Any:
        r"""Accept structural diffusion losses without an atomic loss wrapper."""
        if not callable(value):
            raise ValueError("loss_fn must be callable.")
        return value

    @model_validator(mode="after")
    def _validate_strategy(self):
        r"""Validate diffusion roles while retaining backend hook ownership checks."""
        if (self.num_steps is None) == (self.num_epochs is None):
            raise ValueError("Supply exactly one of num_steps or num_epochs.")
        if "student" not in self.models or not self.optimizer_configs:
            raise ValueError("Configure a student model and named optimizers.")
        if set(self.optimizer_configs) - set(self.models) or any(
            not value for value in self.optimizer_configs.values()
        ):
            raise ValueError(
                "Each optimizer configuration must name a model and contain a nonempty list."
            )
        if not self.devices or len(self.devices) not in (1, len(self.models)):
            raise ValueError("devices must contain one device or one per model.")
        if self.validation_config is not None:
            raise ValueError(
                "Diffusion validation is caller-owned; omit validation_config."
            )
        if not self.phase_roles:
            self.phase_roles = {"student": ("student",)}
            auxiliary = tuple(
                role
                for role in ("fake_score", "discriminator")
                if role in self.optimizer_configs
            )
            if auxiliary:
                self.phase_roles["auxiliary"] = auxiliary
        for roles in self.phase_roles.values():
            if not roles or set(roles) - set(self.optimizer_configs):
                raise ValueError("Every phase must name configured optimizer roles.")
        for source, target in self.ema_models.items():
            if (
                source not in self.optimizer_configs
                or target not in self.models
                or target in self.optimizer_configs
            ):
                raise ValueError(
                    "EMA requires an optimized source and a non-optimized target role."
                )
        if len({id(hook) for hook in self.hooks}) != len(self.hooks):
            raise ValueError("Hook instances must be distinct.")
        _backend._validate_single_do_claimants(self.hooks)
        _backend._validate_hook_dependencies(self.hooks)
        return self

    def model_post_init(self, __context: Any) -> None:
        r"""Initialize the common loop without atomic target assembly."""
        self._init_hooks(list(self.hooks))
        self._refresh_hook_claim_flags()
        self._last_batch = self._last_losses = self._last_loss = None
        self._optimizers, self._lr_schedulers = [], []
        self._runtime_optimizers = []
        self._target_keys = ()
        self._bound_models = dict(self.models)

    def _run_setup_hooks(self, dataloader: Any = None) -> Any:
        r"""Bind final execution wrappers before building optimizers or computing loss."""
        dataloader = super()._run_setup_hooks(dataloader)
        changed = any(
            self._bound_models.get(role) is not model
            for role, model in self.models.items()
        )
        bind = self.bind_models_fn or getattr(
            self.loss_fn, "_bind_execution_models", None
        )
        if self.bind_models_fn is not None:
            bind(self.loss_fn, self.models)
        elif bind is not None:
            bind(self.models)
        elif changed:
            raise ValueError(
                "Setup changed model wrappers; supply bind_models_fn for this custom loss."
            )
        if changed and self._runtime_optimizers:
            roles = [
                role
                for role, configs in self.optimizer_configs.items()
                for _ in configs
            ]
            for role, optimizer in zip(roles, self._optimizers, strict=True):
                parameters = set(self.models[role].parameters())
                if any(
                    value not in parameters
                    for group in optimizer.param_groups
                    for value in group["params"]
                ):
                    raise ValueError(
                        "Setup replaced parameters after optimizer construction; create a fresh strategy."
                    )
        self._bound_models = dict(self.models)
        return dataloader

    @contextmanager
    def _preserve_modes(self):
        r"""Preserve caller-selected mixed child modes around backend contexts."""
        self._model_modes = {
            module: module.training
            for model in self.models.values()
            for module in model.modules()
        }
        try:
            yield
        finally:
            for module, training in self._model_modes.items():
                module.training = training
            self._model_modes = {}

    def train_batch(self, batch: TensorDict) -> None:
        r"""Compute one batch loss and accumulate its gradients.

        Use this method when your recipe controls batch iteration. Pass loss
        arguments in a ``TensorDict``. An optimizer update occurs only after
        ``accumulation_steps`` batches, with the same phase throughout.
        """
        with self._preserve_modes():
            super().train_batch(batch)

    def run(self, dataloader: Any) -> None:
        r"""Run batch iteration until the configured update budget completes.

        Pass an iterable of ``TensorDict`` batches. Set ``num_steps`` for a
        completed update count, or ``num_epochs`` with a data loader that has
        a length. Skipped updates can require extra epochs.
        """
        self._resume_optimizer_state = bool(self._runtime_optimizers)
        with self._preserve_modes():
            super().run(dataloader)

    def _resolve_target_step_count(self, batches_per_epoch: int | None) -> int:
        r"""Convert epoch-sized batch budgets to completed accumulation windows."""
        target = super()._resolve_target_step_count(batches_per_epoch)
        if self.num_steps is not None:
            return target
        if target % self.accumulation_steps:
            raise ValueError(
                "The epoch batch budget must divide evenly into accumulation windows; use num_steps otherwise."
            )
        return math.ceil(target / self.accumulation_steps)

    def _train_batch_with_optimizers(
        self, batch: TensorDict, flat_opts: list, flat_scheds: list
    ) -> None:
        r"""Select only active optimizers without changing the underlying step loop."""
        if self._micro_step == 0:
            self._phase = (
                self.phase_fn(self.step_count)
                if self.phase_fn is not None
                else (
                    "auxiliary"
                    if "auxiliary" in self.phase_roles and self.step_count % 5
                    else "student"
                )
            )
        if self._phase not in self.phase_roles:
            raise ValueError(f"Unknown phase {self._phase!r}.")
        roles = [
            role for role, configs in self.optimizer_configs.items() for _ in configs
        ]
        active = [
            i for i, role in enumerate(roles) if role in self.phase_roles[self._phase]
        ]
        for module, training in self._model_modes.items():
            module.training = training
        try:
            super()._train_batch_with_optimizers(
                batch, [flat_opts[i] for i in active], [flat_scheds[i] for i in active]
            )
        finally:
            self._optimizers, self._lr_schedulers = flat_opts, flat_scheds

    def _run_optimizer_step_phase(
        self, batch: TensorDict, flat_opts: list, flat_scheds: list
    ) -> bool:
        r"""Complete EMA and accumulation bookkeeping before after-batch hooks."""
        updated = super()._run_optimizer_step_phase(batch, flat_opts, flat_scheds)
        self._micro_step = (self._micro_step + 1) % self.accumulation_steps
        if updated:
            with torch.no_grad():
                for source, target in self.ema_models.items():
                    if source not in self.phase_roles[self._phase]:
                        continue
                    source_values = (
                        *self.models[source].parameters(),
                        *self.models[source].buffers(),
                    )
                    target_values = (
                        *self.models[target].parameters(),
                        *self.models[target].buffers(),
                    )
                    for value, average in zip(
                        source_values, target_values, strict=True
                    ):
                        if average.is_floating_point():
                            average.lerp_(value, 1 - self.ema_decay)
                        else:
                            average.copy_(value)
        return updated

    def _compute_losses(
        self,
        predictions: Mapping[str, Any],
        batch: TensorDict,
        *,
        step: int,
        epoch: int,
    ) -> dict:
        r"""Evaluate the loss once and translate only its reporting structure."""
        arguments = dict(predictions.items())
        if set(arguments) & {"phase", "step", "update_state", "return_outputs"}:
            raise ValueError(
                "The strategy owns phase, step, update_state, and return_outputs."
            )
        losses = self.loss_fn(
            **arguments, phase=self._phase, step=step, update_state=True
        )
        return {
            "total_loss": losses["total_loss"].mean(),
            "per_component_unweighted": dict(losses),
            "per_component_sample": dict(losses),
            "per_component_weight": {key: 1.0 for key in losses},
            "per_component_raw_weight": {key: 1.0 for key in losses},
        }

    def state_dict(self) -> dict[str, Any]:
        r"""Return counters, phase settings, and hook and gradient scaler state.

        Call only at a completed accumulation boundary. Use
        ``checkpoint_bundle`` or ``save_checkpoint`` to include models,
        optimizers, schedulers, and loss state in the same checkpoint.
        """
        if self._micro_step:
            raise ValueError("Save only at completed accumulation boundaries.")
        hooks = []
        for hook in self.hooks:
            hooks.extend(
                hook._hooks
                if isinstance(hook, _backend.TrainingUpdateOrchestrator)
                else [hook]
            )
        state = {
            "schema_version": 1,
            "counters": {name: getattr(self, name) for name in _COUNTERS},
        }
        state["hook_types"] = [
            f"{type(hook).__module__}.{type(hook).__qualname__}" for hook in hooks
        ]
        state["phase_roles"] = self.phase_roles
        state["accumulation_steps"] = self.accumulation_steps
        state["hooks"] = {
            str(i): hook.state_dict()
            for i, hook in enumerate(hooks)
            if callable(getattr(hook, "state_dict", None))
            and callable(getattr(hook, "load_state_dict", None))
        }
        state["scalers"] = {
            str(i): hook._scaler.state_dict()
            for i, hook in enumerate(hooks)
            if isinstance(hook, _backend.MixedPrecisionHook)
            and hook._scaler is not None
        }
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        r"""Restore counters and hook and gradient scaler state.

        Recreate matching hooks, phase roles, and accumulation settings before
        passing a dictionary returned by ``state_dict``. Recreate the phase
        callback separately because this state does not include it.
        """
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported strategy state schema.")
        hooks = []
        for hook in self.hooks:
            hooks.extend(
                hook._hooks
                if isinstance(hook, _backend.TrainingUpdateOrchestrator)
                else [hook]
            )
        if (
            state["hook_types"]
            != [f"{type(hook).__module__}.{type(hook).__qualname__}" for hook in hooks]
            or state["phase_roles"] != self.phase_roles
            or state["accumulation_steps"] != self.accumulation_steps
        ):
            raise ValueError(
                "Reconstruct matching hooks, phase roles, and accumulation settings before loading."
            )
        for name in _COUNTERS:
            setattr(self, name, int(state["counters"][name]))
        for index, value in state["hooks"].items():
            hooks[int(index)].load_state_dict(value)
        for index, value in state["scalers"].items():
            hook = hooks[int(index)]
            hook._scaler = torch.amp.GradScaler(self.devices[0].type)
            hook._scaler.load_state_dict(value)
        self._micro_step = 0
        self._resume_optimizer_state = True

    def checkpoint_bundle(self) -> CheckpointStateBundle:
        r"""Collect models, optimizers, schedulers, and training state.

        Use the returned ``CheckpointStateBundle`` with a custom checkpoint
        workflow. It includes this strategy and any loss exposing
        ``state_dict``, and names optimizers by role and index.
        """
        self._setup_runtime_optimizers()
        names = [
            (f"{role}-{i}", role)
            for role, configs in self.optimizer_configs.items()
            for i, _ in enumerate(configs)
        ]
        stateful = {"strategy": self}
        if callable(getattr(self.loss_fn, "state_dict", None)):
            stateful["loss"] = self.loss_fn
        return CheckpointStateBundle(
            self.models,
            optimizers={
                name: record.optimizer
                for (name, _), record in zip(
                    names, self._runtime_optimizers, strict=True
                )
            },
            optimizer_models=dict(names),
            schedulers={
                name: record.scheduler
                for (name, _), record in zip(
                    names, self._runtime_optimizers, strict=True
                )
                if record.scheduler is not None
            },
            stateful=stateful,
        )

    def save_checkpoint(
        self, path: Path | str, *, epoch: int, include_rng: bool = False
    ) -> None:
        r"""Save models and training state to a checkpoint.

        Call at a completed accumulation boundary with the checkpoint ``path``
        and ``epoch`` identifier. Set ``include_rng=True`` to save random number
        generator state. Data-loader position requires a separate checkpoint.
        """
        self.checkpoint_bundle().save_checkpoint(
            path, epoch=epoch, include_rng=include_rng
        )

    def load_checkpoint(
        self, path: Path | str, *, epoch: int, restore_rng: bool = False
    ) -> None:
        r"""Restore models and training state from a checkpoint.

        Recreate matching models, loss, settings, hooks, and phase callbacks,
        then pass the saved ``path`` and ``epoch`` identifier. Set
        ``restore_rng=True`` to restore saved random number generator state.
        """
        with self._preserve_modes(), self:
            self._prepare_setup_hooks()
            self._validate_runtime_devices()
            self.models = _backend.move_to_devices(self.models, self.devices)
            self._run_setup_hooks()
            self.checkpoint_bundle().load_checkpoint(
                path, epoch=epoch, device=self.devices[0], restore_rng=restore_rng
            )

    def to_spec_dict(self) -> dict[str, Any]:
        r"""Raise ``NotImplementedError`` for configuration serialization.

        Reconstruct diffusion components explicitly and use ``load_checkpoint``
        to restore training state.
        """
        raise NotImplementedError(
            "Reconstruct diffusion components explicitly, then use load_checkpoint."
        )
