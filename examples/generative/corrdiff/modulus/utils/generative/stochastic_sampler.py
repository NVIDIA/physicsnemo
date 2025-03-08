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


import math
from typing import Any, Callable, Optional

import torch
from torch import Tensor


def image_batching(
    input: Tensor,
    img_shape_y: int,
    img_shape_x: int,
    patch_shape_y: int,
    patch_shape_x: int,
    batch_size: int,
    overlap_pix: int,
    boundary_pix: int,
    input_interp: Optional[Tensor] = None,
) -> Tensor:
    """
    Splits a full image into a batch of patched images.

    This function takes a full image and splits it into patches, adding padding where necessary.
    It can also concatenate additional interpolated data to each patch if provided.

    Parameters
    ----------
    input : Tensor
        The input tensor representing the full image with shape (batch_size, channels, img_shape_x, img_shape_y).
    img_shape_x : int
        The width (x-dimension) of the original full image.
    img_shape_y : int
        The height (y-dimension) of the original full image.
    patch_shape_x : int
        The width (x-dimension) of each image patch.
    patch_shape_y : int
        The height (y-dimension) of each image patch.
    batch_size : int
        The original batch size before patching.
    overlap_pix : int
        The number of overlapping pixels between adjacent patches.
    boundary_pix : int
        The number of pixels to crop as a boundary from each patch.
    input_interp : Optional[Tensor], optional
        Optional additional data to concatenate to each patch with shape (batch_size, interp_channels, patch_shape_x, patch_shape_y).
        By default None.

    Returns
    -------
    Tensor
        A tensor containing the image patches, with shape (total_patches * batch_size, channels [+ interp_channels], patch_shape_x, patch_shape_y).
    """
    patch_num_x = math.ceil(img_shape_x / (patch_shape_x - overlap_pix - boundary_pix))
    patch_num_y = math.ceil(img_shape_y / (patch_shape_y - overlap_pix - boundary_pix))
    padded_shape_x = (
        (patch_shape_x - overlap_pix - boundary_pix) * (patch_num_x - 1)
        + patch_shape_x
        + boundary_pix
    )
    padded_shape_y = (
        (patch_shape_y - overlap_pix - boundary_pix) * (patch_num_y - 1)
        + patch_shape_y
        + boundary_pix
    )
    pad_x_right = padded_shape_x - img_shape_x - boundary_pix
    pad_y_right = padded_shape_y - img_shape_y - boundary_pix
    input_padded = torch.zeros(
        input.shape[0], input.shape[1], padded_shape_y, padded_shape_x
    ).to(input.device)
    image_padding = torch.nn.ReflectionPad2d(
        (boundary_pix, pad_x_right, boundary_pix, pad_y_right)
    ).to(
        input.device
    )  # (padding_left,padding_right,padding_top,padding_bottom)
    input_padded = image_padding(input)
    patch_num = patch_num_x * patch_num_y
    if input_interp is not None:
        output = torch.zeros(
            patch_num * batch_size,
            input.shape[1] + input_interp.shape[1],
            patch_shape_y,
            patch_shape_x,
        ).to(input.device)
    else:
        output = torch.zeros(
            patch_num * batch_size, input.shape[1], patch_shape_y, patch_shape_x
        ).to(input.device)
    for x_index in range(patch_num_x):
        for y_index in range(patch_num_y):
            x_start = x_index * (patch_shape_x - overlap_pix - boundary_pix)
            y_start = y_index * (patch_shape_y - overlap_pix - boundary_pix)
            if input_interp is not None:
                output[
                    (x_index * patch_num_y + y_index)
                    * batch_size : (x_index * patch_num_y + y_index + 1)
                    * batch_size,
                ] = torch.cat(
                    (
                        input_padded[
                            :,
                            :,
                            y_start : y_start + patch_shape_y,
                            x_start : x_start + patch_shape_x,
                        ],
                        input_interp,
                    ),
                    dim=1,
                )
            else:
                output[
                    (x_index * patch_num_y + y_index)
                    * batch_size : (x_index * patch_num_y + y_index + 1)
                    * batch_size,
                ] = input_padded[
                    :,
                    :,
                    y_start : y_start + patch_shape_y,
                    x_start : x_start + patch_shape_x,
                ]
    return output


def image_fuse(
    input: Tensor,
    img_shape_y: int,
    img_shape_x: int,
    patch_shape_y: int,
    patch_shape_x: int,
    batch_size: int,
    overlap_pix: int,
    boundary_pix: int,
) -> Tensor:
    """
    Reconstructs a full image from a batch of patched images.

    This function takes a batch of image patches and reconstructs the full image
    by stitching the patches together. The function accounts for overlapping and
    boundary pixels, ensuring that overlapping areas are averaged.

    Parameters
    ----------
    input : Tensor
        The input tensor containing the image patches with shape (total_patches * batch_size, channels, patch_shape_x, patch_shape_y).
    img_shape_x : int
        The width (x-dimension) of the original full image.
    img_shape_y : int
        The height (y-dimension) of the original full image.
    patch_shape_x : int
        The width (x-dimension) of each image patch.
    patch_shape_y : int
        The height (y-dimension) of each image patch.
    batch_size : int
        The original batch size before patching.
    overlap_pix : int
        The number of overlapping pixels between adjacent patches.
    boundary_pix : int
        The number of pixels to crop as a boundary from each patch.

    Returns
    -------
    Tensor
        The reconstructed full image tensor with shape (batch_size, channels, img_shape_x, img_shape_y).

    """
    patch_num_x = math.ceil(img_shape_x / (patch_shape_x - overlap_pix - boundary_pix))
    patch_num_y = math.ceil(img_shape_y / (patch_shape_y - overlap_pix - boundary_pix))
    padded_shape_x = (
        (patch_shape_x - overlap_pix - boundary_pix) * (patch_num_x - 1)
        + patch_shape_x
        + boundary_pix
    )
    padded_shape_y = (
        (patch_shape_y - overlap_pix - boundary_pix) * (patch_num_y - 1)
        + patch_shape_y
        + boundary_pix
    )
    pad_x_right = padded_shape_x - img_shape_x - boundary_pix
    pad_y_right = padded_shape_y - img_shape_y - boundary_pix
    residual_x = patch_shape_x - pad_x_right  # residual pixels in the last patch
    residual_y = patch_shape_y - pad_y_right  # residual pixels in the last patch
    output = torch.zeros(
        batch_size, input.shape[1], img_shape_y, img_shape_x, device=input.device
    )
    one_map = torch.ones(1, 1, input.shape[2], input.shape[3], device=input.device)
    count_map = torch.zeros(
        1, 1, img_shape_y, img_shape_x, device=input.device
    )  # to count the overlapping times
    for x_index in range(patch_num_x):
        for y_index in range(patch_num_y):
            x_start = x_index * (patch_shape_x - overlap_pix - boundary_pix)
            y_start = y_index * (patch_shape_y - overlap_pix - boundary_pix)
            if (x_index == patch_num_x - 1) and (y_index != patch_num_y - 1):
                output[
                    :, :, y_start : y_start + patch_shape_y - 2 * boundary_pix, x_start:
                ] += input[
                    (x_index * patch_num_y + y_index)
                    * batch_size : (x_index * patch_num_y + y_index + 1)
                    * batch_size,
                    :,
                    boundary_pix : patch_shape_y - boundary_pix,
                    boundary_pix : residual_x + boundary_pix,
                ]
                count_map[
                    :, :, y_start : y_start + patch_shape_y - 2 * boundary_pix, x_start:
                ] += one_map[
                    :,
                    :,
                    boundary_pix : patch_shape_y - boundary_pix,
                    boundary_pix : residual_x + boundary_pix,
                ]
            elif (y_index == patch_num_y - 1) and ((x_index != patch_num_x - 1)):
                output[
                    :, :, y_start:, x_start : x_start + patch_shape_x - 2 * boundary_pix
                ] += input[
                    (x_index * patch_num_y + y_index)
                    * batch_size : (x_index * patch_num_y + y_index + 1)
                    * batch_size,
                    :,
                    boundary_pix : residual_y + boundary_pix,
                    boundary_pix : patch_shape_x - boundary_pix,
                ]
                count_map[
                    :, :, y_start:, x_start : x_start + patch_shape_x - 2 * boundary_pix
                ] += one_map[
                    :,
                    :,
                    boundary_pix : residual_y + boundary_pix,
                    boundary_pix : patch_shape_x - boundary_pix,
                ]
            elif x_index == patch_num_x - 1 and y_index == patch_num_y - 1:
                output[:, :, y_start:, x_start:] += input[
                    (x_index * patch_num_y + y_index)
                    * batch_size : (x_index * patch_num_y + y_index + 1)
                    * batch_size,
                    :,
                    boundary_pix : residual_y + boundary_pix,
                    boundary_pix : residual_x + boundary_pix,
                ]
                count_map[:, :, y_start:, x_start:] += one_map[
                    :,
                    :,
                    boundary_pix : residual_y + boundary_pix,
                    boundary_pix : residual_x + boundary_pix,
                ]
            else:
                output[
                    :,
                    :,
                    y_start : y_start + patch_shape_y - 2 * boundary_pix,
                    x_start : x_start + patch_shape_x - 2 * boundary_pix,
                ] += input[
                    (x_index * patch_num_y + y_index)
                    * batch_size : (x_index * patch_num_y + y_index + 1)
                    * batch_size,
                    :,
                    boundary_pix : patch_shape_y - boundary_pix,
                    boundary_pix : patch_shape_x - boundary_pix,
                ]
                count_map[
                    :,
                    :,
                    y_start : y_start + patch_shape_y - 2 * boundary_pix,
                    x_start : x_start + patch_shape_x - 2 * boundary_pix,
                ] += one_map[
                    :,
                    :,
                    boundary_pix : patch_shape_y - boundary_pix,
                    boundary_pix : patch_shape_x - boundary_pix,
                ]
    return output / count_map


def stochastic_sampler(
    net: Any,
    latents: Tensor,
    img_lr: Tensor,
    class_labels: Optional[Tensor] = None,
    randn_like: Callable[[Tensor], Tensor] = torch.randn_like,
    img_shape: int = 448,
    patch_shape: int = 448,
    overlap_pix: int = 4,
    boundary_pix: int = 2,
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
    """
    Proposed EDM sampler (Algorithm 2) with minor changes to enable super-resolution and patch-based diffusion.

    Parameters
    ----------
    net : Any
        The neural network model that generates denoised images from noisy inputs.
    latents : Tensor
        The latent variables (e.g., noise) used as the initial input for the sampler.
    img_lr : Tensor
        Low-resolution input image for conditioning the super-resolution process.
    class_labels : Optional[Tensor], optional
        Class labels for conditional generation, if required by the model. By default None.
    randn_like : Callable[[Tensor], Tensor]
        Function to generate random noise with the same shape as the input tensor.
        By default torch.randn_like.
    img_shape : int
        The height and width of the full image (assumed to be square). By default 448.
    patch_shape : int
        The height and width of each patch (assumed to be square). By default 448.
    overlap_pix : int
        Number of overlapping pixels between adjacent patches. By default 4.
    boundary_pix : int
        Number of pixels to be cropped as a boundary from each patch. By default 2.
    mean_hr : Optional[Tensor], optional
        Optional tensor containing mean high-resolution images for conditioning. By default None.
    num_steps : int
        Number of time steps for the sampler. By default 18.
    sigma_min : float
        Minimum noise level. By default 0.002.
    sigma_max : float
        Maximum noise level. By default 800.
    rho : float
        Exponent used in the time step discretization. By default 7.
    S_churn : float
        Churn parameter controlling the level of noise added in each step. By default 0.
    S_min : float
        Minimum time step for applying churn. By default 0.
    S_max : float
        Maximum time step for applying churn. By default float("inf").
    S_noise : float
        Noise scaling factor applied during the churn step. By default 1.

    Returns
    -------
    Tensor
        The final denoised image produced by the sampler.
    """

    # Adjust noise levels based on what's supported by the network.
    "Proposed EDM sampler (Algorithm 2) with minor changes to enable super-resolution."
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    if isinstance(img_shape, tuple):
        img_shape_y, img_shape_x = img_shape
    else:
        img_shape_x = img_shape_y = img_shape
    if patch_shape > img_shape_x or patch_shape > img_shape_y:
        patch_shape = min(img_shape_x, img_shape_y)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    b = latents.shape[0]
    Nx = torch.arange(img_shape_x)
    Ny = torch.arange(img_shape_y)
    grid = torch.stack(torch.meshgrid(Ny, Nx, indexing="ij"), dim=0)[
        None,
    ].expand(b, -1, -1, -1)

    # conditioning = [mean_hr, img_lr, global_lr, pos_embd]
    batch_size = img_lr.shape[0]
    x_lr = img_lr
    if mean_hr is not None:
        x_lr = torch.cat((mean_hr.expand(x_lr.shape[0], -1, -1, -1), x_lr), dim=1)
    global_index = None

    # input and position padding + patching
    if patch_shape != img_shape_x or patch_shape != img_shape_y:
        input_interp = torch.nn.functional.interpolate(
            img_lr, (patch_shape, patch_shape), mode="bilinear"
        )
        x_lr = image_batching(
            x_lr,
            img_shape_y,
            img_shape_x,
            patch_shape,
            patch_shape,
            batch_size,
            overlap_pix,
            boundary_pix,
            input_interp,
        )
        global_index = image_batching(
            grid.float(),
            img_shape_y,
            img_shape_x,
            patch_shape,
            patch_shape,
            batch_size,
            overlap_pix,
            boundary_pix,
        ).int()

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        # Increase noise temporarily.
        gamma = S_churn / num_steps if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)

        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step. Perform patching operation on score tensor if patch-based generation is used
        # denoised = net(x_hat, t_hat, class_labels,lead_time_label=lead_time_label).to(torch.float64)    #x_lr

        if patch_shape != img_shape_x or patch_shape != img_shape_y:
            x_hat_batch = image_batching(
                x_hat,
                img_shape_y,
                img_shape_x,
                patch_shape,
                patch_shape,
                batch_size,
                overlap_pix,
                boundary_pix,
            )
        else:
            x_hat_batch = x_hat
        x_hat_batch = x_hat_batch.to(latents.device)
        x_lr = x_lr.to(latents.device)
        if global_index is not None:
            global_index = global_index.to(latents.device)

        if lead_time_label is not None:
            denoised = net(
                x_hat_batch,
                x_lr,
                t_hat,
                class_labels,
                lead_time_label=lead_time_label,
                global_index=global_index,
            ).to(torch.float64)
        else:
            denoised = net(
                x_hat_batch,
                x_lr,
                t_hat,
                class_labels,
                global_index=global_index,
            ).to(torch.float64)
        if patch_shape != img_shape_x or patch_shape != img_shape_y:

            denoised = image_fuse(
                denoised,
                img_shape_y,
                img_shape_x,
                patch_shape,
                patch_shape,
                batch_size,
                overlap_pix,
                boundary_pix,
            )
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            if patch_shape != img_shape_x or patch_shape != img_shape_y:
                x_next_batch = image_batching(
                    x_next,
                    img_shape_y,
                    img_shape_x,
                    patch_shape,
                    patch_shape,
                    batch_size,
                    overlap_pix,
                    boundary_pix,
                )
            else:
                x_next_batch = x_next
            # ask about this fix
            x_next_batch = x_next_batch.to(latents.device)
            if lead_time_label is not None:
                denoised = net(
                    x_next_batch,
                    x_lr,
                    t_next,
                    class_labels,
                    lead_time_label=lead_time_label,
                    global_index=global_index,
                ).to(torch.float64)
            else:
                denoised = net(
                    x_next_batch,
                    x_lr,
                    t_next,
                    class_labels,
                    global_index=global_index,
                ).to(torch.float64)
            if patch_shape != img_shape_x or patch_shape != img_shape_y:
                denoised = image_fuse(
                    denoised,
                    img_shape_y,
                    img_shape_x,
                    patch_shape,
                    patch_shape,
                    batch_size,
                    overlap_pix,
                    boundary_pix,
                )
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next
