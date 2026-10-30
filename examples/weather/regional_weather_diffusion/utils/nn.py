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

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from tensordict import TensorDict
import torch

import importlib

from physicsnemo.core import Module
from physicsnemo.models.diffusion_unets import StormCastUNet
from physicsnemo.diffusion.preconditioners import EDMPreconditioner
from physicsnemo.diffusion.samplers import sample as diffusion_sample
from physicsnemo.diffusion.utils import ConcatConditionWrapper
from physicsnemo.models.dit import DiT
from physicsnemo.diffusion.noise_schedulers import NoiseScheduler

import utils.apex  # do not remove, enables Apex LayerNorm with ShardTensor


class TimeScaledModel(Module):
    r"""Scale flow-matching times before passing them to the backbone.

    Flow-matching times live in :math:`[0, 1]`, while the sinusoidal timestep
    embedders of the DiT and U-Net resolve inputs spanning roughly
    :math:`[0, 1000]`. The scale is part of the saved model, so training and
    sampling always agree. Unlike the ``EDMPreconditioner``, this wrapper
    does not scale the inputs and outputs: the network output is the
    prediction itself (``training.loss.prediction_type``), with any noise
    schedule.

    Parameters
    ----------
    model : Module
        Backbone called as ``model(x, t, condition=condition)``.
    time_scale : float, optional
        Factor applied to the time before calling the backbone, by default 999.

    Forward
    -------
    x : torch.Tensor
        Noisy state of shape :math:`(B, C, H, W)`.
    t : torch.Tensor
        Flow-matching times of shape :math:`(B,)`.
    condition : torch.Tensor | TensorDict | None, optional
        Condition passed to the backbone.
    **model_kwargs : Any
        Other keyword arguments passed to the backbone.

    Outputs
    -------
    torch.Tensor
        Backbone output of shape :math:`(B, C, H, W)`.
    """

    def __init__(self, model: Module, time_scale: float = 999.0):
        super().__init__()
        self.model = model
        self.time_scale = time_scale

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | TensorDict | None = None,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        if not torch.compiler.is_compiling():
            if x.ndim != 4:
                raise ValueError(
                    f"Expected tensor of shape (B, C, H, W) but got tensor of shape {tuple(x.shape)}"
                )
            if t.shape != (x.shape[0],):
                raise ValueError(
                    f"Expected tensor of shape ({x.shape[0]},) but got tensor of shape {tuple(t.shape)}"
                )
        # Scale in float32: a reduced precision t would quantize the scaled
        # time to a coarser grid than in training
        return self.model(
            x, t.float() * self.time_scale, condition=condition, **model_kwargs
        )


def wrap_diffusion_network(
    net: torch.nn.Module,
    preconditioner: Literal["edm", "time_scaled"],
    time_scale: float = 999.0,
) -> EDMPreconditioner | TimeScaledModel:
    """Wrap a network with the configured preconditioner (``model.preconditioner``).

    Args:
        net: the backbone, called as ``net(x, t, condition=condition)``
        preconditioner: "edm" applies the EDM input, output, and noise-level
            scalings, so the wrapped network predicts the clean state x0;
            "time_scaled" only multiplies the times by ``time_scale``, so the
            wrapped network predicts ``training.loss.prediction_type``
        time_scale: factor applied to the times by the "time_scaled" preconditioner
    Returns:
        the wrapped network
    """
    wrapped = ConcatConditionWrapper(net)
    if preconditioner == "edm":
        return EDMPreconditioner(model=wrapped)
    elif preconditioner == "time_scaled":
        return TimeScaledModel(wrapped, time_scale=time_scale)
    raise ValueError("preconditioner must be 'edm' or 'time_scaled'")


def get_preconditioned_unet(
    name: str,
    target_channels: int,
    conditional_channels: int = 0,
    img_resolution: tuple = (512, 640),
    model_type: str | None = None,
    lead_time_steps: int = 0,
    lead_time_channels: int = 4,
    amp_mode: bool = False,
    use_apex_gn: bool = False,
    preconditioner: Literal["edm", "time_scaled"] = "edm",
    time_scale: float = 999.0,
    **model_kwargs,
) -> EDMPreconditioner | TimeScaledModel | StormCastUNet:
    """
    Create a preconditioner-wrapped SongUNet network.

    Args:
        name: 'regression' or 'diffusion' to select between either model type
        target_channels: The number of channels in the target
        conditional_channels: The number of channels in the conditioning
        img_resolution: resolution of the data (U-Net inputs/outputs)
        model_type: the model class to use, or None to select it automatically
        lead_time_steps: the number of possible lead time steps, if 0 lead time embedding will be disabled
        lead_time_channels: the number of channels to use for each lead time embedding
        amp_mode: whether to use automatic mixed precision
        use_apex_gn: whether to use Apex GroupNorm
        preconditioner: "edm" or "time_scaled", see wrap_diffusion_network
        time_scale: factor applied to the times by the "time_scaled" preconditioner
    Returns:
        EDMPreconditioner, TimeScaledModel or StormCastUNet: a wrapped torch
        module net(x_t, t, condition) -> prediction
    """

    if model_type is None:
        model_type = "SongUNetPosLtEmbd" if lead_time_steps else "SongUNet"

    model_params = {
        "img_resolution": img_resolution,
        "img_out_channels": target_channels,
        "model_type": model_type,
        "amp_mode": amp_mode,
        "use_apex_gn": use_apex_gn,
    }
    model_params.update(model_kwargs)

    if lead_time_steps:
        model_params["N_grid_channels"] = 0
        model_params["lead_time_channels"] = lead_time_channels
        model_params["lead_time_steps"] = lead_time_steps
    else:
        lead_time_channels = 0

    if name == "diffusion":
        unet_module = importlib.import_module("physicsnemo.models.diffusion_unets")
        model_class = getattr(unet_module, model_params.pop("model_type"))
        out_channels = model_params.pop("img_out_channels")
        unet = model_class(
            in_channels=target_channels + conditional_channels + lead_time_channels,
            out_channels=out_channels,
            **model_params,
        )
        return wrap_diffusion_network(unet, preconditioner, time_scale)

    elif name == "regression":
        return StormCastUNet(
            img_in_channels=conditional_channels + lead_time_channels,
            embedding_type="zero",
            **model_params,
        )


def get_preconditioned_natten_dit(
    target_channels: int,
    conditional_channels: int = 0,
    scalar_condition_channels: int = 0,
    img_resolution: tuple = (512, 640),
    hidden_size: int = 768,
    depth: int = 16,
    num_heads: int = 16,
    patch_size: int = 4,
    attn_kernel_size: int = 31,
    lead_time_steps: int = 0,
    attention_backend: Literal["natten2d", "natten2d_rope"] = "natten2d",
    layernorm_backend: Literal["torch", "apex"] = "torch",
    conditioning_embedder: Literal["dit", "edm", "zero"] = "dit",
    attn_kwargs: dict[str, Any] = {},
    preconditioner: Literal["edm", "time_scaled"] = "edm",
    time_scale: float = 999.0,
    **model_kwargs,
) -> EDMPreconditioner | TimeScaledModel:
    """
    Create a preconditioner-wrapped Diffusion Transformer (DiT) network.

    Args:
        target_channels: The number of channels in the target
        conditional_channels: The number of channels in the conditioning
        scalar_condition_channels: The number of scalar condition channels
        img_resolution: Resolution of the data (DiT inputs/outputs)
        hidden_size: The number of channels in the internal layers of the DiT
        depth: The number of transformer blocks in the DiT
        num_heads: number of heads in multi-head attention
        patch_size: the patch size used by the DiT embedder
        attn_kernel_size: the attention neighborhood size
        lead_time_steps: the number of possible lead time steps, if 0 lead time embedding will be disabled
        attention_backend: neighborhood attention without ("natten2d") or with
            2D rotary position embeddings ("natten2d_rope")
        layernorm_backend: layer normalization implementation of the blocks
        conditioning_embedder: the DiT conditioning embedder
        attn_kwargs: parameters to be passed to the DiT attention backend, for
            example {"qk_norm": True, "norm_layer": "apex"} to normalize the
            attention queries and keys
        preconditioner: "edm" or "time_scaled", see wrap_diffusion_network
        time_scale: factor applied to the times by the "time_scaled" preconditioner
        **model_kwargs: any additional parameters to the model
    Returns:
        EDMPreconditioner or TimeScaledModel: a wrapped torch module
        net(x_t, t, condition) -> prediction
    """

    condition_dim = scalar_condition_channels + lead_time_steps
    attn_kw = {"attn_kernel": attn_kernel_size}
    attn_kw.update(attn_kwargs)
    dit = DiT(
        input_size=img_resolution,
        in_channels=target_channels + conditional_channels,
        out_channels=target_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size,
        attention_backend=attention_backend,
        layernorm_backend=layernorm_backend,
        attn_kwargs=attn_kw,
        condition_dim=condition_dim,
        conditioning_embedder=conditioning_embedder,
        **model_kwargs,
    )
    return wrap_diffusion_network(dit, preconditioner, time_scale)


def build_network_condition_and_target(
    background: torch.Tensor,
    target: torch.Tensor,
    invariant_tensor: torch.Tensor | None,
    scalar_conditions: torch.Tensor | None = None,
    lead_time_label: torch.Tensor | None = None,
    regression_net: Module | None = None,
    condition_list: Iterable[str] = ("background",),
    regression_condition_list: Iterable[str] = ("background",),
) -> tuple[torch.Tensor | TensorDict, torch.Tensor, torch.Tensor | None]:
    """Build the condition and target tensors for the network.

    Args:
        background: background tensor -- all per-sample conditioning, including
            any past state the dataset folds in as extra channels
        target: training target (the "state" tensor from the batch)
        invariant_tensor: invariant tensor or None if no invariant is used
        lead_time_label: lead time label or None if lead time embedding is not used
        regression_net: regression model, can be None if 'regression' is not in condition_list
        condition_list: list of conditions to include, may include 'background', 'regression' and 'invariant'
        regression_condition_list: list of conditions for the regression network, may include 'background' and 'invariant'
            This is only used if regression_net is set.
    Returns:
        A tuple of tensors: (
            condition: model condition concatenated from conditions specified in condition_list,
            target: training target (residual of the regression estimate if 'regression' is in condition_list),
            regression: regression model output
        ). The regression model output will be None if 'regression' is not in condition_list.
    """
    if ("regression" in condition_list) and (regression_net is None):
        raise ValueError(
            "regression_net must be provided if 'regression' is in condition_list"
        )

    condition_tensors = {
        "background": background,
        "invariant": invariant_tensor,
        "regression": None,
    }

    with torch.no_grad():
        if "regression" in condition_list:
            # Inference regression model
            condition_tensors["regression"] = regression_model_forward(
                regression_net,
                background,
                invariant_tensor,
                lead_time_label=lead_time_label,
                condition_list=regression_condition_list,
            )
            target = target - condition_tensors["regression"]

        condition = [
            y for c in condition_list if (y := condition_tensors[c]) is not None
        ]
        condition = torch.cat(condition, dim=1) if condition else None

    if scalar_conditions is not None:
        condition = TensorDict(
            (
                {"cond_concat": condition, "cond_vec": scalar_conditions}
                if condition is not None
                else {"cond_vec": scalar_conditions}
            ),
            device=target.device,
        ).to(dtype=target.dtype)

    return (condition, target, condition_tensors["regression"])


def unpack_batch(
    batch: dict[str, Any],
    device: torch.device | str,
    memory_format: torch.memory_format = torch.preserve_format,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack a data batch into background, state and lead time label with the correct
    device and data types.
    """
    if isinstance(batch["state"], (list, tuple)):
        raise ValueError(
            "batch['state'] must be a single tensor (the training target). "
            "The two-element [input, target] convention has been removed: "
            "concatenate any past state your model needs as input onto the "
            "'background' tensor instead. See the 'Adding custom datasets' "
            "section of the README for details."
        )

    (background, state, mask) = nested_to(
        (batch.get("background"), batch["state"], batch.get("mask")),
        device=device,
        dtype=torch.float32,
        non_blocking=True,
        memory_format=memory_format,
    )

    lead_time_label = batch.get("lead_time_label")
    if lead_time_label is not None:
        lead_time_label = lead_time_label.to(
            device=device, dtype=torch.int64, non_blocking=True
        )
    scalar_conditions = batch.get("scalar_conditions")
    if scalar_conditions is not None:
        scalar_conditions = scalar_conditions.to(
            device=device, dtype=torch.float32, non_blocking=True
        )

    return (background, state, mask, lead_time_label, scalar_conditions)


def flow_conversions(scheduler: NoiseScheduler) -> dict[str, Any]:
    """Scheduler functions ``f(prediction, x_t, t)`` that convert each
    prediction type to the flow (velocity) of the rectified-flow ODE."""
    # The domain-parallel wrapper only implements the sampling and loss
    # methods; the element-wise conversions come from the wrapped scheduler
    scheduler = getattr(scheduler, "inner_scheduler", scheduler)
    return {
        "x0": scheduler.x0_to_flow,
        "score": scheduler.score_to_flow,
        "epsilon": lambda eps, x_t, t: scheduler.x0_to_flow(
            scheduler.epsilon_to_x0(eps, x_t, t), x_t, t
        ),
        "flow": lambda flow, x_t, t: flow,
    }


def diffusion_model_forward(
    model: Module,
    condition: torch.Tensor,
    shape: Iterable[int],
    scheduler: NoiseScheduler,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    lead_time_label: torch.Tensor | None = None,
    sampler_args: dict[str, Any] = {},
    invalid_mask: torch.Tensor | None = None,
    formulation: Literal["edm", "flow"] = "edm",
    prediction_type: Literal["x0", "epsilon", "score", "flow"] = "x0",
) -> torch.Tensor:
    """Run diffusion model sampling using the ``physicsnemo.diffusion`` API.

    Uses the provided noise scheduler for timestep generation and
    :func:`~physicsnemo.diffusion.samplers.sample` for the reverse ODE
    integration. With ``formulation="edm"`` the scheduler's denoiser factory
    turns the prediction into the ODE right-hand side. For rectified flow
    (``formulation="flow"``) the right-hand side is the flow itself, converted
    from the prediction with the scheduler.

    For domain-parallel inference, pass a scheduler that has already been
    wrapped with
    :class:`~physicsnemo.diffusion.DomainParallelNoiseScheduler` (e.g.
    via :meth:`~utils.parallel.ParallelHelper.make_domain_parallel_scheduler`).

    Parameters
    ----------
    model : Module
        Preconditioned diffusion model (``EDMPreconditioner``).
    condition : torch.Tensor
        Conditioning tensor for the model.
    shape : Iterable[int]
        Shape of the output tensor, e.g. ``(B, C, H, W)``.
    scheduler : NoiseScheduler
        Noise scheduler (e.g.
        :class:`~physicsnemo.diffusion.noise_schedulers.EDMNoiseScheduler`).
    dtype : torch.dtype, optional
        Precision for ODE integration. Defaults to the condition tensor's dtype.
    device : torch.device, optional
        Device for latent generation. Defaults to the condition tensor's device.
    lead_time_label : torch.Tensor | None
        Lead-time labels forwarded to the model.
    sampler_args : dict
        Sampler configuration. Supported keys: ``num_steps``,
        ``solver`` (``"heun"`` or ``"euler"``),
        ``S_churn``, ``S_min``, ``S_max``, ``S_noise`` (stochastic sampling,
        see the note in the code on rectified flow), and
        ``time_shift`` (rectified flow only): the time shift :math:`a` warps
        the sampling times as :math:`t \to a t / (1 + (a - 1) t)` to spend
        more steps at high noise levels, which helps when sampling a larger
        domain than used in training (https://arxiv.org/abs/2403.03206).
    invalid_mask : torch.Tensor | None
        Per-sample invalid-region mask ``(B, 1, H, W)`` forwarded to the model's
        NaN-mask-token path at every denoising step. ``None`` disables masking.
    formulation : {"edm", "flow"}
        Noise schedule family of ``scheduler``: ``"edm"`` for
        ``training.loss.type: edm``, ``"flow"`` for ``flow_matching``.
    prediction_type : {"x0", "epsilon", "score", "flow"}
        Quantity predicted by ``model`` (``training.loss.prediction_type``).
    """
    if isinstance(condition, TensorDict):
        ref_tensor = condition.get("cond_concat", condition.get("cond_vec"))
        if ref_tensor is None:
            raise ValueError(
                "condition TensorDict must contain 'cond_concat' or 'cond_vec'"
                "if a TensorDict is passed as condition."
            )
    else:
        ref_tensor = condition
    if dtype is None:
        dtype = ref_tensor.dtype
    if device is None:
        device = ref_tensor.device

    sa = sampler_args
    num_steps = sa.get("num_steps", 18)
    solver_name = sa.get("solver", "heun")
    S_churn = float(sa.get("S_churn", 0))
    S_min = float(sa.get("S_min", 0))
    S_max = float(sa.get("S_max", float("inf")))
    S_noise = float(sa.get("S_noise", 1))
    time_shift = float(sa.get("time_shift", 1.0))

    B = shape[0]
    t_steps = scheduler.timesteps(num_steps, device=device, dtype=dtype)
    if formulation == "flow":
        t_steps = time_shift * t_steps / (1.0 + (time_shift - 1.0) * t_steps)
    tN = t_steps[0].expand(B)
    xN = scheduler.init_latents(tuple(shape[1:]), tN, device=device, dtype=dtype)

    extra_kwargs: dict[str, Any] = {}
    if lead_time_label is not None:
        extra_kwargs["lead_time_label"] = lead_time_label
    if invalid_mask is not None:
        extra_kwargs["invalid_mask"] = invalid_mask

    def predictor(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return model(x, t, condition=condition, **extra_kwargs)

    if formulation == "edm":
        denoiser = scheduler.get_denoiser(**{f"{prediction_type}_predictor": predictor})
    elif formulation == "flow":
        # The flow is the right-hand side of the rectified-flow ODE; converting
        # through scheduler.get_denoiser would divide by alpha(t) = 1 - t,
        # which is zero at the pure-noise start t = 1
        to_flow = flow_conversions(scheduler)[prediction_type]

        def denoiser(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return to_flow(predictor(x, t).float(), x, t)

    else:
        raise ValueError("formulation must be 'edm' or 'flow'")
    # Stochastic sampling (S_churn > 0) works with both formulations. The churn
    # of the physicsnemo solvers adds noise without rescaling the signal, which
    # is exact for the EDM schedule (alpha = 1) and an approximation for
    # rectified flow (alpha = 1 - t); sigma(t) = t for both schedulers, so no
    # sigma callbacks are needed.
    solver_options: dict[str, Any] | None = None
    if S_churn > 0:
        solver_name = (
            "edm_stochastic_heun" if solver_name == "heun" else "edm_stochastic_euler"
        )
        solver_options = {
            "S_churn": S_churn,
            "S_min": S_min,
            "S_max": S_max,
            "S_noise": S_noise,
            "num_steps": num_steps,
        }

    return diffusion_sample(
        denoiser,
        xN,
        scheduler,
        num_steps,
        solver=solver_name,
        solver_options=solver_options,
        time_steps=t_steps,
    )


def regression_model_forward(
    model: Module,
    background: torch.Tensor,
    invariant_tensor: torch.Tensor,
    lead_time_label: torch.Tensor | None = None,
    condition_list: Iterable[str] = ("background",),
) -> torch.Tensor:
    """Helper function to run regression model forward pass in inference"""

    (x, _, _) = build_network_condition_and_target(
        background,
        None,
        invariant_tensor,
        lead_time_label=lead_time_label,
        condition_list=condition_list,
    )

    labels = {} if lead_time_label is None else {"lead_time_label": lead_time_label}
    return model(x, **labels)


def nested_to(
    x: torch.Tensor | Mapping | list | tuple | Any, **kwargs
) -> torch.Tensor | dict | list | Any:
    """Move tensors in nested structures to a device/dtype.

    Parameters
    ----------
    x : torch.Tensor or Mapping or list or tuple
        Input structure.
    **kwargs
        Keyword arguments forwarded to ``Tensor.to``.

    Returns
    -------
    torch.Tensor or dict or list
        Structure with tensors moved.
    """
    if isinstance(x, Mapping):
        return {k: nested_to(v, **kwargs) for (k, v) in x.items()}
    elif isinstance(x, (list, tuple)):
        return [nested_to(v, **kwargs) for v in x]
    else:
        if not isinstance(x, torch.Tensor):
            return x
        return x.to(**kwargs)
