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

r"""DMD2 distribution matching with separate student and auxiliary objectives."""

import importlib
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from jaxtyping import Float, Shaped
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch import Tensor

from physicsnemo.core import Module
from physicsnemo.core.version_check import check_version_spec
from physicsnemo.diffusion._fastgen import _FastGenNetwork
from physicsnemo.diffusion.base import (
    DiffusionExtraTimes,
)
from physicsnemo.diffusion.noise_schedulers import (
    DomainParallelNoiseScheduler,
    LinearGaussianNoiseScheduler,
)
from physicsnemo.diffusion.noise_schedulers._fastgen import _FastGenNoiseScheduler

from ..._fastgen import _configure_options, _FastGenBinding

_AVAILABLE = check_version_spec("fastgen", "0.1.0", hard_fail=False)
if _AVAILABLE:
    _backend = importlib.import_module("fastgen.methods.distribution_matching.dmd2")
    _DMD2Base = _backend.DMD2Model
else:
    _DMD2Base = Module


@contextmanager
def _fixed_parameters(*models: torch.nn.Module | None):
    r"""Freeze parameters temporarily while preserving gradients through inputs."""
    parameters = {
        p: p.requires_grad
        for model in models
        if model is not None
        for p in model.parameters()
    }
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in parameters.items():
            parameter.requires_grad_(requires_grad)


class _Discriminator(torch.nn.Module):
    r"""Pass teacher features to a discriminator as a named TensorDict."""

    def __init__(self, model: torch.nn.Module, names: tuple[str, ...]) -> None:
        super().__init__()
        self.model = model
        self.names = names
        self.feature_indices = set(range(len(names)))

    def forward(self, features: list[Tensor]) -> Float[Tensor, " B logits"]:
        r"""Return discriminator logits from the named teacher features."""
        values = TensorDict(
            dict(zip(self.names, features)), batch_size=[features[0].shape[0]]
        )
        logits = self.model(values)
        if not isinstance(logits, Tensor) or logits.ndim != 2:
            raise ValueError("Discriminator output must have shape [B, logits].")
        return logits


class _DMD2Loss(_FastGenBinding, _DMD2Base):
    r"""Match a student's generated distribution to a fixed diffusion teacher.

    Use separate student and auxiliary phases to train the student and fake-score
    model. An optional discriminator adds adversarial training; each phase
    preserves the supplied models' training modes and frozen parameters.
    """

    def __init__(
        self,
        student: torch.nn.Module,
        noise_scheduler: LinearGaussianNoiseScheduler | DomainParallelNoiseScheduler,
        *,
        teacher: torch.nn.Module | None,
        auxiliary_models: Mapping[str, torch.nn.Module],
        method_options: Mapping[str, Any],
        model_options: Mapping[str, Mapping[str, Any]],
    ) -> None:
        torch.nn.Module.__init__(self)
        if teacher is None or "fake_score" not in auxiliary_models:
            raise ValueError(
                "DMD2 requires teacher and auxiliary_models['fake_score']."
            )
        unknown = set(auxiliary_models) - {"fake_score", "discriminator"}
        if unknown:
            raise ValueError(f"Unknown auxiliary roles: {sorted(unknown)}.")
        unknown = set(model_options) - {"student", "teacher", "fake_score"}
        if unknown:
            raise ValueError(f"Unknown model roles: {sorted(unknown)}.")
        self._bind_models(
            student,
            noise_scheduler,
            teacher,
            {key: value for key, value in model_options.items() if key != "fake_score"},
        )
        options = dict(model_options.get("fake_score", {}))
        schedule = _FastGenNoiseScheduler(
            options.pop("noise_scheduler", noise_scheduler)
        )
        schedule._configure_training(**options.pop("scheduler_options", {}))
        options.setdefault("prediction_type", "x0")
        self.fake_score = _FastGenNetwork(
            auxiliary_models["fake_score"], schedule, **options
        )
        for role in (self.teacher, self.fake_score):
            if (
                role.noise_scheduler._scheduler
                is not self.net.noise_scheduler._scheduler
            ):
                role._input_noise_scheduler = self.net.noise_scheduler
        discriminator = auxiliary_models.get("discriminator")
        self.discriminator = None
        if discriminator is not None:
            names = tuple(model_options.get("teacher", {}).get("feature_names", ()))
            if not names or model_options.get("teacher", {}).get("feature_fn") is None:
                raise ValueError(
                    "Configure teacher feature_names and feature_fn for the discriminator."
                )
            self.discriminator = _Discriminator(discriminator, names)
        roles = [student, teacher, auxiliary_models["fake_score"]]
        if discriminator is not None:
            roles.append(discriminator)
        owned = [set(model.parameters()) for model in roles]
        if any(
            left & right for i, left in enumerate(owned) for right in owned[i + 1 :]
        ):
            raise ValueError("Training roles must not share parameters.")
        self.config = _configure_options(
            {
                "gan_loss_weight_gen": 0.001 if discriminator is not None else 0.0,
                "gan_use_same_t_noise": False,
                "gan_r1_reg_weight": 0.0,
                "gan_r1_reg_alpha": 0.1,
                "guidance_scale": None,
                "skip_layers": None,
                "fake_score_pred_type": None,
                "student_sample_steps": 1,
                "sampling": {},
                "fake_score_sampling": None,
            },
            method_options,
        )
        if self.config.gan_loss_weight_gen < 0 or self.config.gan_r1_reg_weight < 0:
            raise ValueError("GAN loss weights must be nonnegative.")
        if self.config.gan_loss_weight_gen > 0 and discriminator is None:
            raise ValueError("GAN objectives require an explicit discriminator.")
        if self.config.student_sample_steps < 1:
            raise ValueError("student_sample_steps must be positive.")
        self.config.sample_t_cfg = OmegaConf.create(
            {"time_dist_type": "physicsnemo", **self.config.sampling}
        )
        self.config.fake_score_pred_type = (
            self.config.fake_score_pred_type or self.fake_score.net_pred_type
        )
        self._phase = None
        self._adversarial_fn = None
        self._update_state = True

    def evaluate(
        self,
        x0: Float[Tensor, " B *dims"],
        t: Float[Tensor, " B"]
        | tuple[Float[Tensor, " B"], Shaped[DiffusionExtraTimes, " B"]]
        | None,
        condition: Tensor | TensorDict | None,
        *,
        phase: str | None,
        step: int,
        return_outputs: bool,
        update_state: bool,
        weight: Float[Tensor, " #B *#dims"] | None = None,
        neg_condition: Tensor | TensorDict | None = None,
        sample_t_fn: Callable[
            ...,
            Float[Tensor, " B"]
            | tuple[Float[Tensor, " B"], Shaped[DiffusionExtraTimes, " B"]],
        ]
        | None = None,
        input_student: Float[Tensor, " B *dims"] | None = None,
        t_student: Float[Tensor, " B"] | None = None,
        noise: Float[Tensor, " B *dims"] | None = None,
        adversarial_fn: Callable[..., tuple[Tensor, Tensor]] | None = None,
    ) -> Any:
        r"""Compute the selected phase's losses for the supplied training step."""
        if phase not in ("student", "auxiliary"):
            raise ValueError("Select phase='student' or phase='auxiliary' explicitly.")
        if step < 0:
            raise ValueError("step must be nonnegative.")
        if t is not None and sample_t_fn is not None:
            raise ValueError("Supply explicit times or sample_t_fn, not both.")
        if (input_student is None) != (t_student is None):
            raise ValueError("Supply input_student and t_student together.")
        if x0.ndim < 2 or x0.shape[0] == 0:
            raise ValueError("Expected nonempty states with shape [B, ...].")
        if weight is not None:
            weight = torch.broadcast_to(weight.to(x0.device), x0.shape)
            if not torch.isfinite(weight).all() or torch.any(weight < 0):
                raise ValueError("Weights must be finite and nonnegative.")
            if self.config.gan_loss_weight_gen > 0 and adversarial_fn is None:
                raise ValueError(
                    "Weighted GAN objectives require adversarial_fn to map valid support to logits."
                )
        self.device = x0.device
        batch = x0.shape[0]
        source = self.net.noise_scheduler
        if sample_t_fn is not None:
            t = sample_t_fn(batch, device=x0.device, dtype=source.t_precision)
        if isinstance(t, tuple):
            raise ValueError(
                "Distribution matching requires a single diffusion time tensor."
            )
        if t is None:
            sampling = self.config.fake_score_sampling if phase == "auxiliary" else None
            t = source.sample_t(
                batch, device=x0.device, **(sampling or self.config.sampling)
            )
        if input_student is None:
            initial_noise = torch.randn_like(x0)
            if self.config.student_sample_steps == 1:
                t_student = torch.full_like(t, source.max_t)
                input_student = source.latents(noise=initial_noise)
            else:
                t_student = source.sample_from_t_list(
                    batch,
                    sample_steps=self.config.student_sample_steps,
                    device=x0.device,
                )
                input_student = source.forward_process(x0, initial_noise, t_student)
        noise = torch.randn_like(x0) if noise is None else noise
        previous = (
            self._phase,
            self._call_weight,
            self._adversarial_fn,
            self._update_state,
        )
        self._phase, self._call_weight, self._adversarial_fn, self._update_state = (
            phase,
            weight,
            adversarial_fn,
            update_state,
        )
        try:
            fixed = (
                (self.teacher, self.fake_score, self.discriminator)
                if phase == "student"
                else (self.teacher, self.net)
            )
            with _fixed_parameters(*fixed):
                losses, outputs = self.single_train_step(
                    {
                        "real": x0,
                        "t": t,
                        "input_student": input_student,
                        "t_student": t_student,
                        "eps": noise,
                        "condition": condition,
                        "neg_condition": neg_condition,
                    },
                    step,
                )
        finally:
            self._phase, self._call_weight, self._adversarial_fn, self._update_state = (
                previous
            )
        losses = {
            key: value if key == "total_loss" else value.detach()
            for key, value in losses.items()
        }
        return (
            (losses, {key: value.detach() for key, value in outputs.items()})
            if return_outputs
            else losses
        )

    def single_train_step(
        self, data: Mapping[str, Any], iteration: int
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        r"""Evaluate the selected student or auxiliary loss."""
        args = (data["input_student"], data["t_student"], data["t"], data["eps"])
        return self._dispatch_training_phase(
            self._phase,
            *args,
            data,
            data["real"],
            condition=data["condition"],
            neg_condition=data["neg_condition"],
        )

    def _vsd_loss(
        self,
        generated: Tensor,
        teacher: Tensor,
        fake: Tensor,
        scale: Tensor | None = None,
    ) -> Tensor:
        r"""Compute weighted score distillation with normalization over valid entries."""
        return _backend.variational_score_distillation_loss(
            generated,
            teacher,
            fake,
            additional_scale=scale,
            loss_weight=self._call_weight,
        )

    def _logits(
        self, x: Tensor, t: Tensor, condition: Any
    ) -> tuple[Tensor, Tensor | None]:
        r"""Return discriminator logits and optional weights from the mask callback."""
        if self._adversarial_fn is not None:
            logits, weight = self._adversarial_fn(x, t, condition, self._call_weight)
            weight = torch.broadcast_to(weight.to(logits.device), logits.shape)
            if (
                logits.ndim != 2
                or not torch.isfinite(weight).all()
                or torch.any(weight < 0)
            ):
                raise ValueError(
                    "Adversarial logits require rank two and finite nonnegative weights."
                )
            return logits, weight
        features = self.teacher(
            x,
            t,
            condition=condition,
            return_features_early=True,
            feature_indices=self.discriminator.feature_indices,
        )
        return self.discriminator(features), None

    @staticmethod
    def _logit_mean(value: Tensor, weight: Tensor | None) -> Tensor:
        r"""Average weighted logits over all entries, with zero weights excluded."""
        return (
            value.mean()
            if weight is None
            else (torch.where(weight > 0, value, 0) * weight).mean()
        )

    def _student_update_step(
        self,
        input_student: Tensor,
        t_student: Tensor,
        t: Tensor,
        eps: Tensor,
        data: Mapping[str, Any],
        condition: Any = None,
        neg_condition: Any = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        r"""Compute student score-matching and optional adversarial losses."""
        generated = self.gen_data_from_net(
            input_student, t_student, condition=condition
        )
        perturbed = self.net.noise_scheduler.forward_process(generated, eps, t)
        with torch.no_grad():
            fake = self.fake_score(
                perturbed, t, condition=condition, fwd_pred_type="x0"
            )
        gan = generated.new_zeros(())
        scale = None
        if self.config.gan_loss_weight_gen > 0 and self._adversarial_fn is None:
            teacher, features = self.teacher(
                perturbed,
                t,
                condition=condition,
                fwd_pred_type="x0",
                feature_indices=self.discriminator.feature_indices,
            )
            logits, weight = self.discriminator(features), None
            teacher = teacher.detach()
        else:
            with torch.no_grad():
                teacher = self.teacher(
                    perturbed, t, condition=condition, fwd_pred_type="x0"
                )
            if self.config.gan_loss_weight_gen > 0:
                logits, weight = self._logits(perturbed, t, condition)
        if self.config.guidance_scale is not None:
            teacher = self._apply_classifier_free_guidance(
                perturbed, t, teacher, neg_condition=neg_condition
            )
        if self.config.gan_loss_weight_gen > 0:
            gan = _backend.gan_loss_generator(logits, loss_weight=weight)
            scale = self._ratio_weight(logits, t, weight)
        vsd = self._vsd_loss(generated, teacher, fake, scale)
        return {
            "total_loss": vsd + self.config.gan_loss_weight_gen * gan,
            "vsd_loss": vsd,
            "gan_loss_gen": gan,
        }, self._get_outputs(generated, input_student)

    def _ratio_weight(
        self, logits: Tensor, t: Tensor, weight: Tensor | None
    ) -> Tensor | None:
        r"""DMD2 does not reweight scores with a density ratio."""
        return None

    def _fake_score_discriminator_update_step(
        self,
        input_student: Tensor,
        t_student: Tensor,
        t: Tensor,
        eps: Tensor,
        real_data: Tensor,
        condition: Any = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        r"""Compute auxiliary losses with student samples held fixed."""
        source = self.net.noise_scheduler
        with torch.no_grad():
            generated = self.gen_data_from_net(
                input_student, t_student, condition=condition
            )
            perturbed = source.forward_process(generated, eps, t)
        prediction_type = self.config.fake_score_pred_type
        prediction = self.fake_score(
            perturbed, t, condition=condition, fwd_pred_type=prediction_type
        )
        score_loss = _backend.denoising_score_matching_loss(
            prediction_type,
            prediction,
            x0=generated,
            eps=eps,
            noise_scheduler=source,
            t=t,
            loss_weight=self._call_weight,
            flow_target=source.cond_velocity(generated, eps, t)
            if prediction_type == "flow"
            else None,
        )
        gan = score_loss * 0
        r1 = score_loss * 0
        if self.config.gan_loss_weight_gen > 0:
            fake_logits, fake_weight = self._logits(perturbed, t, condition)
            if self.config.gan_use_same_t_noise:
                t_real, eps_real = t, eps
            else:
                t_real = source.sample_t(
                    real_data.shape[0], device=real_data.device, **self.config.sampling
                )
                eps_real = torch.randn_like(real_data)
            real_perturbed = source.forward_process(real_data, eps_real, t_real)
            real_logits, real_weight = self._logits(real_perturbed, t_real, condition)
            gan = _backend.gan_loss_discriminator(
                real_logits,
                fake_logits,
                real_weight=real_weight,
                fake_weight=fake_weight,
            )
            if self.config.gan_r1_reg_weight > 0:
                jittered = real_data + self.config.gan_r1_reg_alpha * torch.randn_like(
                    real_data
                )
                jitter_logits, jitter_weight = self._logits(jittered, t_real, condition)
                support = real_weight
                if jitter_weight is not None:
                    support = (
                        jitter_weight
                        if support is None
                        else torch.where(jitter_weight > 0, support, 0)
                    )
                delta = real_logits - jitter_logits
                if support is not None:
                    delta = torch.where(support > 0, delta, 0)
                r1 = self._logit_mean(delta.square(), support)
        return {
            "total_loss": score_loss + gan + self.config.gan_r1_reg_weight * r1,
            "fake_score_loss": score_loss,
            "gan_loss_disc": gan,
            "gan_loss_ar1": r1,
        }, self._get_outputs(generated, input_student)

    def _get_outputs(
        self, gen_data: Tensor, input_student: Tensor, condition: Any = None
    ) -> dict[str, Tensor]:
        r"""Return generated samples and their student inputs."""
        return {"prediction": gen_data, "input_student": input_student}
