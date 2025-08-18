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


from typing import Callable, Optional

import torch
from torch import Tensor

from physicsnemo.utils.patching import GridPatching2D


# NOTE: use two wrappers for apply, to avoid recompilation when input shape changes
@torch.compile()
def _apply_wrapper_Cin_channels(patching, input, additional_input=None):
    """
    Apply the patching operation to the input tensor with :math:`C_{in}` channels.
    """
    return patching.apply(input=input, additional_input=additional_input)


@torch.compile()
def _apply_wrapper_Cout_channels_no_grad(patching, input, additional_input=None):
    """
    Apply the patching operation to an input tensor with :math:`C_{out}`
    channels that does not require gradients.
    """
    return patching.apply(input=input, additional_input=additional_input)


@torch.compile()
def _apply_wrapper_Cout_channels_grad(patching, input, additional_input=None):
    """
    Apply the patching operation to an input tensor with :math:`C_{out}`
    channels that requires gradients.
    """
    return patching.apply(input=input, additional_input=additional_input)


@torch.compile()
def _fuse_wrapper(patching, input, batch_size):
    return patching.fuse(input=input, batch_size=batch_size)


def _apply_wrapper_select(
    input: torch.Tensor, patching: GridPatching2D | None
) -> Callable:
    """
    Select the correct patching wrapper based on the input tensor's requires_grad attribute.
    If patching is None, return the identity function.
    If patching is not None, return the appropriate patching wrapper.
    If input.requires_grad is True, return _apply_wrapper_Cout_channels_grad.
    If input.requires_grad is False, return _apply_wrapper_Cout_channels_no_grad.
    """
    if patching:
        if input.requires_grad:
            return _apply_wrapper_Cout_channels_grad
        else:
            return _apply_wrapper_Cout_channels_no_grad
    else:
        return lambda patching, input, additional_input=None: input


def stochastic_sampler(
    net: torch.nn.Module,
    latents: Tensor,
    img_lr: Tensor,
    class_labels: Optional[Tensor] = None,
    randn_like: Callable[[Tensor], Tensor] = torch.randn_like,
    patching: Optional[GridPatching2D] = None,
    mean_hr: Optional[Tensor] = None,
    lead_time_label: Optional[Tensor] = None,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 800,
    rho: float = 7,
    S_churn: float = 0,
    S_min: float = 0,
    S_max: float = float("inf"),
    S_noise: float = 1,
) -> Tensor:
    r"""
    Proposed EDM sampler (Algorithm 2) with minor changes to enable
    super-resolution and patch-based diffusion.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model that generates denoised images from noisy
        inputs.
        Expected signature: ``net(x, x_lr, t_hat, class_labels,
        lead_time_label=lead_time_label,
        embedding_selector=embedding_selector)``.

        Inputs:
            - **x** (*torch.Tensor*): Noisy input of shape :math:`(B, C_{out}, H, W)`
            - **x_lr** (*torch.Tensor*): Conditioning input of shape :math:`(B, C_{cond}, H, W)`
            - **t_hat** (*torch.Tensor*): Noise level of shape :math:`(B, 1, 1, 1)` or scalar
            - **class_labels** (*torch.Tensor, optional*): Optional class labels
            - **lead_time_label** (*torch.Tensor, optional*): Optional lead time labels
            - **embedding_selector** (*callable, optional*): Function to select
              positional embeddings. Used for patch-based diffusion.

        Output:
            - **denoised** (*torch.Tensor*): Denoised prediction of shape :math:`(B, C_{out}, H, W)`

        Required attributes:
            - **sigma_min** (*float*): Minimum supported noise level for the model
            - **sigma_max** (*float*): Maximum supported noise level for the model
            - **round_sigma** (*callable*): Method to convert sigma values to
              tensor representation

    latents : Tensor
        The latent variables (e.g., noise) used as the initial input for the
        sampler. Has shape :math:`(B, C_{out}, H, W)`.
    img_lr : Tensor
        Low-resolution input image for conditioning the super-resolution
        process. Must have shape :math:`(B, C_{lr}, H, W)`.
    class_labels : Optional[Tensor], optional
        Class labels for conditional generation, if required by the model. By
        default ``None``.
    randn_like : Callable[[Tensor], Tensor]
        Function to generate random noise with the same shape as the input
        tensor.
        By default ``torch.randn_like``.
    patching : Optional[GridPatching2D], default=None
        A patching utility for patch-based diffusion. Implements methods to
        extract patches from an image and batch the patches along dim=0.
        Should also implement a ``fuse`` method to reconstruct the original
        image from a batch of patches. See
        :class:`~physicsnemo.utils.patching.GridPatching2D` for details. By
        default ``None``, in which case non-patched diffusion is used.
    mean_hr : Optional[Tensor], optional
        Optional tensor containing mean high-resolution images for
        conditioning. Must have same height and width as ``img_lr``, with shape
        :math:`(B_{hr}, C_{hr}, H, W)`  where the batch dimension
        :math:`B_{hr}` can be either 1, either equal to batch_size, or can be omitted. If
        :math:`B_{hr} = 1` or is omitted, ``mean_hr`` will be expanded to match the shape
        of ``img_lr``. By default ``None``.
    lead_time_label : Optional[Tensor], optional
        Optional lead time labels. By default ``None``.
    num_steps : int
        Number of time steps for the sampler. By default 18.
    sigma_min : float
        Minimum noise level. By default 0.002.
    sigma_max : float
        Maximum noise level. By default 800.
    rho : float
        Exponent used in the time step discretization. By default 7.
    S_churn : float
        Churn parameter controlling the level of noise added in each step. By
        default 0.
    S_min : float
        Minimum time step for applying churn. By default 0.
    S_max : float
        Maximum time step for applying churn. By default ``float("inf")``.
    S_noise : float
        Noise scaling factor applied during the churn step. By default 1.

    Returns
    -------
    Tensor
        The final denoised image produced by the sampler. Same shape as
        ``latents``: :math:`(B, C_{out}, H, W)`.

    See Also
    --------
    :class:`~physicsnemo.models.diffusion.preconditioning.EDMPrecondSuperResolution`: A model
        wrapper that provides preconditioning for super-resolution diffusion
        models and implements the required interface for this sampler.
    """

    # Adjust noise levels based on what's supported by the network.
    # Proposed EDM sampler (Algorithm 2) with minor changes to enable
    # super-resolution/
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Safety check on type of patching
    if patching is not None and not isinstance(patching, GridPatching2D):
        raise ValueError("patching must be an instance of GridPatching2D.")

    # Safety check: if patching is used then img_lr and latents must have same
    # height and width, otherwise there is mismatch in the number
    # of patches extracted to form the final batch_size.
    if patching:
        if img_lr.shape[-2:] != latents.shape[-2:]:
            raise ValueError(
                f"img_lr and latents must have the same height and width, "
                f"but found {img_lr.shape[-2:]} vs {latents.shape[-2:]}. "
            )
    # img_lr and latents must also have the same batch_size, otherwise mismatch
    # when processed by the network
    if img_lr.shape[0] != latents.shape[0]:
        raise ValueError(
            f"img_lr and latents must have the same batch size, but found "
            f"{img_lr.shape[0]} vs {latents.shape[0]}."
        )

    # Time step discretization.
    step_indices = torch.arange(num_steps, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    batch_size = img_lr.shape[0]

    # conditioning = [mean_hr, img_lr, global_lr, pos_embd]
    x_lr = img_lr
    if mean_hr is not None:
        if mean_hr.shape[-2:] != img_lr.shape[-2:]:
            raise ValueError(
                f"mean_hr and img_lr must have the same height and width, "
                f"but found {mean_hr.shape[-2:]} vs {img_lr.shape[-2:]}."
            )
        x_lr = torch.cat((mean_hr.expand(x_lr.shape[0], -1, -1, -1), x_lr), dim=1)

    # input and position padding + patching
    if patching:
        # Patched conditioning [x_lr, mean_hr]
        # (batch_size * patch_num, C_in + C_out, patch_shape_y, patch_shape_x)
        x_lr = _apply_wrapper_Cin_channels(
            patching=patching, input=x_lr, additional_input=img_lr
        )

        # Function to select the correct positional embedding for each patch
        def patch_embedding_selector(emb):
            # emb: (N_pe, image_shape_y, image_shape_x)
            # return: (batch_size * patch_num, N_pe, patch_shape_y, patch_shape_x)
            return patching.apply(emb.expand(batch_size, -1, -1, -1))

    else:
        patch_embedding_selector = None

    optional_args = {}
    if lead_time_label is not None:
        optional_args["lead_time_label"] = lead_time_label

    # Main sampling loop.
    x_next = latents * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        # Increase noise temporarily.
        gamma = S_churn / num_steps if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)

        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step. Perform patching operation on score tensor if patch-based
        # generation is used denoised = net(x_hat, t_hat,
        # class_labels,lead_time_label=lead_time_label).to(torch.float64)
        x_hat_batch = _apply_wrapper_select(input=x_hat, patching=patching)(
            patching=patching, input=x_hat
        ).to(latents.device)

        x_lr = x_lr.to(latents.device)

        denoised = net(
            x_hat_batch,
            x_lr,
            t_hat,
            class_labels,
            embedding_selector=patch_embedding_selector,
            **optional_args,
        )

        if patching:
            # Un-patch the denoised image
            # (batch_size, C_out, img_shape_y, img_shape_x)
            denoised = _fuse_wrapper(
                patching=patching, input=denoised, batch_size=batch_size
            )

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            # Patched input
            # (batch_size * patch_num, C_out, patch_shape_y, patch_shape_x)
            x_next_batch = _apply_wrapper_select(input=x_next, patching=patching)(
                patching=patching, input=x_next
            ).to(latents.device)

            denoised = net(
                x_next_batch,
                x_lr,
                t_next,
                class_labels,
                embedding_selector=patch_embedding_selector,
                **optional_args,
            )

            if patching:
                # Un-patch the denoised image
                # (batch_size, C_out, img_shape_y, img_shape_x)
                denoised = _fuse_wrapper(
                    patching=patching, input=denoised, batch_size=batch_size
                )

            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next
