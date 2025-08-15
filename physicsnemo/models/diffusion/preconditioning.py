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

"""
Preconditioning schemes used in the paper"Elucidating the Design Space of
Diffusion-Based Generative Models".
"""

import importlib
import warnings
from dataclasses import dataclass
from typing import List, Literal, Tuple, Union

import numpy as np
import torch

from physicsnemo.models.diffusion.utils import _wrapped_property
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

network_module = importlib.import_module("physicsnemo.models.diffusion")


@dataclass
class VPPrecondMetaData(ModelMetaData):
    """VPPrecond meta data"""

    name: str = "VPPrecond"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class VPPrecond(Module):
    """
    Preconditioning corresponding to the variance preserving (VP) formulation.

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    beta_d : float
        Extent of the noise level schedule, by default 19.9.
    beta_min : float
        Initial slope of the noise level schedule, by default 0.1.
    M : int
        Original number of timesteps in the DDPM formulation, by default 1000.
    epsilon_t : float
        Minimum t-value used during training, by default 1e-5.
    model_type :str
        Class name of the underlying model, by default "SongUNet".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference: Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and
    Poole, B., 2020. Score-based generative modeling through stochastic differential
    equations. arXiv preprint arXiv:2011.13456.
    """

    def __init__(
        self,
        img_resolution: int,
        img_channels: int,
        label_dim: int = 0,
        use_fp16: bool = False,
        beta_d: float = 19.9,
        beta_min: float = 0.1,
        M: int = 1000,
        epsilon_t: float = 1e-5,
        model_type: str = "SongUNet",
        **model_kwargs: dict,
    ):
        super().__init__(meta=VPPrecondMetaData)
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.M = M
        self.epsilon_t = epsilon_t
        self.sigma_min = float(self.sigma(epsilon_t))
        self.sigma_max = float(self.sigma(1))
        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )  # TODO needs better handling

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def sigma(self, t: Union[float, torch.Tensor]):
        """
        Compute the sigma(t) value for a given t based on the VP formulation.

        The function calculates the noise level schedule for the diffusion process based
        on the given parameters `beta_d` and `beta_min`.

        Parameters
        ----------
        t : Union[float, torch.Tensor]
            The timestep or set of timesteps for which to compute sigma(t).

        Returns
        -------
        torch.Tensor
            The computed sigma(t) value(s).
        """
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma: Union[float, torch.Tensor]):
        """
        Compute the inverse of the sigma function for a given sigma.

        This function effectively calculates t from a given sigma(t) based on the
        parameters `beta_d` and `beta_min`.

        Parameters
        ----------
        sigma : Union[float, torch.Tensor]
            The sigma(t) value or set of sigma(t) values for which to compute the
            inverse.

        Returns
        -------
        torch.Tensor
            The computed t value(s) corresponding to the provided sigma(t).
        """
        sigma = torch.as_tensor(sigma)
        return (
            (self.beta_min**2 + 2 * self.beta_d * (1 + sigma**2).log()).sqrt()
            - self.beta_min
        ) / self.beta_d

    def round_sigma(self, sigma: Union[float, List, torch.Tensor]):
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float list, torch.Tensor]
            The sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            The tensor representation of the provided sigma value(s).
        """
        return torch.as_tensor(sigma)


@dataclass
class VEPrecondMetaData(ModelMetaData):
    """VEPrecond meta data"""

    name: str = "VEPrecond"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class VEPrecond(Module):
    """
    Preconditioning corresponding to the variance exploding (VE) formulation.

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min : float
        Minimum supported noise level, by default 0.02.
    sigma_max : float
        Maximum supported noise level, by default 100.0.
    model_type :str
        Class name of the underlying model, by default "SongUNet".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference: Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and
    Poole, B., 2020. Score-based generative modeling through stochastic differential
    equations. arXiv preprint arXiv:2011.13456.
    """

    def __init__(
        self,
        img_resolution: int,
        img_channels: int,
        label_dim: int = 0,
        use_fp16: bool = False,
        sigma_min: float = 0.02,
        sigma_max: float = 100.0,
        model_type: str = "SongUNet",
        **model_kwargs: dict,
    ):
        super().__init__(meta=VEPrecondMetaData)
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )  # TODO needs better handling

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = 1
        c_out = sigma
        c_in = 1
        c_noise = (0.5 * sigma).log()

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma: Union[float, List, torch.Tensor]):
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float list, torch.Tensor]
            The sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            The tensor representation of the provided sigma value(s).
        """
        return torch.as_tensor(sigma)


@dataclass
class iDDPMPrecondMetaData(ModelMetaData):
    """iDDPMPrecond meta data"""

    name: str = "iDDPMPrecond"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class iDDPMPrecond(Module):
    """
    Preconditioning corresponding to the improved DDPM (iDDPM) formulation.

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    C_1 : float
        Timestep adjustment at low noise levels., by default 0.001.
    C_2 : float
        Timestep adjustment at high noise levels., by default 0.008.
    M: int
        Original number of timesteps in the DDPM formulation, by default 1000.
    model_type :str
        Class name of the underlying model, by default "DhariwalUNet".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference: Nichol, A.Q. and Dhariwal, P., 2021, July. Improved denoising diffusion
    probabilistic models. In International Conference on Machine Learning
    (pp. 8162-8171). PMLR.
    """

    def __init__(
        self,
        img_resolution,
        img_channels,
        label_dim=0,
        use_fp16=False,
        C_1=0.001,
        C_2=0.008,
        M=1000,
        model_type="DhariwalUNet",
        **model_kwargs,
    ):
        super().__init__(meta=iDDPMPrecondMetaData)
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.C_1 = C_1
        self.C_2 = C_2
        self.M = M
        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels * 2,
            label_dim=label_dim,
            **model_kwargs,
        )  # TODO needs better handling

        u = torch.zeros(M + 1)
        for j in range(M, 0, -1):  # M, ..., 1
            u[j - 1] = (
                (u[j] ** 2 + 1)
                / (self.alpha_bar(j - 1) / self.alpha_bar(j)).clip(min=C_1)
                - 1
            ).sqrt()
        self.register_buffer("u", u)
        self.sigma_min = float(u[M - 1])
        self.sigma_max = float(u[0])

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = 1
        c_out = -sigma
        c_in = 1 / (sigma**2 + 1).sqrt()
        c_noise = (
            self.M - 1 - self.round_sigma(sigma, return_index=True).to(torch.float32)
        )

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        D_x = c_skip * x + c_out * F_x[:, : self.img_channels].to(torch.float32)
        return D_x

    def alpha_bar(self, j):
        """
        Compute the alpha_bar(j) value for a given j based on the iDDPM formulation.

        Parameters
        ----------
        j : Union[int, torch.Tensor]
            The timestep or set of timesteps for which to compute alpha_bar(j).

        Returns
        -------
        torch.Tensor
            The computed alpha_bar(j) value(s).
        """
        j = torch.as_tensor(j)
        return (0.5 * np.pi * j / self.M / (self.C_2 + 1)).sin() ** 2

    def round_sigma(self, sigma, return_index=False):
        """
        Round the provided sigma value(s) to the nearest value(s) in a
        pre-defined set `u`.

        Parameters
        ----------
        sigma : Union[float, list, torch.Tensor]
            The sigma value(s) to round.
        return_index : bool, optional
            Whether to return the index/indices of the rounded value(s) in `u` instead
            of the rounded value(s) themselves, by default False.

        Returns
        -------
        torch.Tensor
            The rounded sigma value(s) or their index/indices in `u`, depending on the
            value of `return_index`.
        """
        sigma = torch.as_tensor(sigma)
        index = torch.cdist(
            sigma.to(self.u.device).to(torch.float32).reshape(1, -1, 1),
            self.u.reshape(1, -1, 1),
        ).argmin(2)
        result = index if return_index else self.u[index.flatten()].to(sigma.dtype)
        return result.reshape(sigma.shape).to(sigma.device)


@dataclass
class EDMPrecondMetaData(ModelMetaData):
    """EDMPrecond meta data"""

    name: str = "EDMPrecond"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class EDMPrecond(Module):
    """
    Improved preconditioning proposed in the paper "Elucidating the Design Space of
    Diffusion-Based Generative Models" (EDM)

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels (for both input and output). If your model
        requires a different number of input or output chanels,
        override this by passing either of the optional
        img_in_channels or img_out_channels args
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min : float
        Minimum supported noise level, by default 0.0.
    sigma_max : float
        Maximum supported noise level, by default inf.
    sigma_data : float
        Expected standard deviation of the training data, by default 0.5.
    model_type :str
        Class name of the underlying model, by default "DhariwalUNet".
    img_in_channels: int
        Optional setting for when number of input channels =/= number of output
        channels. If set, will override img_channels for the input
        This is useful in the case of additional (conditional) channels
    img_out_channels: int
        Optional setting for when number of input channels =/= number of output
        channels. If set, will override img_channels for the output
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference: Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    """

    def __init__(
        self,
        img_resolution,
        img_channels,
        label_dim=0,
        use_fp16=False,
        sigma_min=0.0,
        sigma_max=float("inf"),
        sigma_data=0.5,
        model_type="DhariwalUNet",
        img_in_channels=None,
        img_out_channels=None,
        **model_kwargs,
    ):
        super().__init__(meta=EDMPrecondMetaData)
        self.img_resolution = img_resolution
        if img_in_channels is not None:
            img_in_channels = img_in_channels
        else:
            img_in_channels = img_channels
        if img_out_channels is not None:
            img_out_channels = img_out_channels
        else:
            img_out_channels = img_channels

        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_in_channels,
            out_channels=img_out_channels,
            label_dim=label_dim,
            **model_kwargs,
        )  # TODO needs better handling

    def forward(
        self,
        x,
        sigma,
        condition=None,
        class_labels=None,
        force_fp32=False,
        **model_kwargs,
    ):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        arg = c_in * x

        if condition is not None:
            arg = torch.cat([arg, condition], dim=1)

        F_x = self.model(
            arg.to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    @staticmethod
    def round_sigma(sigma: Union[float, List, torch.Tensor]):
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float list, torch.Tensor]
            The sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            The tensor representation of the provided sigma value(s).
        """
        return torch.as_tensor(sigma)


@dataclass
class EDMPrecondSuperResolutionMetaData(ModelMetaData):
    """EDMPrecondSuperResolution meta data"""

    name: str = "EDMPrecondSuperResolution"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class EDMPrecondSuperResolution(Module):
    """
    Improved preconditioning proposed in the paper "Elucidating the Design Space of
    Diffusion-Based Generative Models" (EDM).

    This is a variant of `EDMPrecond` that is specifically designed for super-resolution
    tasks. It wraps a neural network that predicts the denoised high-resolution image
    given a noisy high-resolution image, and additional conditioning that includes a
    low-resolution image, and a noise level.

    Parameters
    ----------
    img_resolution : Union[int, Tuple[int, int]]
        Spatial resolution :math:`(H, W)` of the image. If a single int is provided,
        the image is assumed to be square.
    img_in_channels : int
        Number of input channels in the low-resolution input image.
    img_out_channels : int
        Number of output channels in the high-resolution output image.
    use_fp16 : bool, optional
        Whether to use half-precision floating point (FP16) for model execution,
        by default False.
    model_type : str, optional
        Class name of the underlying model. Must be one of the following:
        'SongUNet', 'SongUNetPosEmbd', 'SongUNetPosLtEmbd', 'DhariwalUNet'.
        Defaults to 'SongUNetPosEmbd'.
    sigma_data : float, optional
        Expected standard deviation of the training data, by default 0.5.
    sigma_min : float, optional
        Minimum supported noise level, by default 0.0.
    sigma_max : float, optional
        Maximum supported noise level, by default inf.
    **model_kwargs : dict
        Keyword arguments passed to the underlying model `__init__` method.

    See Also
    --------
    For information on model types and their usage:
    :class:`~physicsnemo.models.diffusion.SongUNet`: Basic U-Net for diffusion models
    :class:`~physicsnemo.models.diffusion.SongUNetPosEmbd`: U-Net with positional embeddings
    :class:`~physicsnemo.models.diffusion.SongUNetPosLtEmbd`: U-Net with positional and lead-time embeddings

    Please refer to the documentation of these classes for details on how to call
    and use these models directly.

    Note
    ----
    References:
    - Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    - Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    # Classes that can be wrapped by this UNet class.
    _wrapped_classes = {
        "SongUNetPosEmbd",
        "SongUNetPosLtEmbd",
        "SongUNet",
        "DhariwalUNet",
    }

    # Arguments of the __init__ method that can be overridden with the
    # ``Module.from_checkpoint`` method. Here, since we use splatted arguments
    # for the wrapped model instance, we allow overriding of any overridable
    # argument of the wrapped classes.
    _overridable_args = set.union(
        *(
            getattr(getattr(network_module, cls_name), "_overridable_args", set())
            for cls_name in _wrapped_classes
        )
    )

    def __init__(
        self,
        img_resolution: Union[int, Tuple[int, int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        model_type: Literal[
            "SongUNetPosEmbd", "SongUNetPosLtEmbd", "SongUNet", "DhariwalUNet"
        ] = "SongUNetPosEmbd",
        sigma_data: float = 0.5,
        sigma_min=0.0,
        sigma_max=float("inf"),
        **model_kwargs: dict,
    ):
        super().__init__(meta=EDMPrecondSuperResolutionMetaData)

        # Validation
        if model_type not in self._wrapped_classes:
            raise ValueError(
                f"Model type '{model_type}' is not supported. "
                f"Must be one of: {', '.join(self._wrapped_classes)}"
            )

        self.img_resolution = img_resolution
        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_in_channels + img_out_channels,
            out_channels=img_out_channels,
            **model_kwargs,
        )  # TODO needs better handling
        self.scaling_fn = self._scaling_fn
        self.use_fp16 = use_fp16

    @property
    def use_fp16(self):
        """
        bool: Whether the model uses float16 precision.

        Returns
        -------
        bool
            True if the model is in float16 mode, False otherwise.
        """
        return self._use_fp16

    @use_fp16.setter
    def use_fp16(self, value: bool):
        """
        Set whether the model should use float16 precision.

        Parameters
        ----------
        value : bool
            If True, moves the model to torch.float16. If False, moves to torch.float32.

        Raises
        ------
        ValueError
            If `value` is not a boolean.
        """
        # NOTE: allow 0/1 values for older checkpoints
        if not (isinstance(value, bool) or value in [0, 1]):
            raise ValueError(
                f"`use_fp16` must be a boolean, but got {type(value).__name__}."
            )
        self._use_fp16 = value
        if value:
            self.to(torch.float16)
        else:
            self.to(torch.float32)

    @staticmethod
    def _scaling_fn(
        x: torch.Tensor, img_lr: torch.Tensor, c_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Scale input tensors by first scaling the high-resolution tensor and then
        concatenating with the low-resolution tensor.

        Parameters
        ----------
        x : torch.Tensor
            Noisy high-resolution image of shape (B, C_hr, H, W).
        img_lr : torch.Tensor
            Low-resolution image of shape (B, C_lr, H, W).
        c_in : torch.Tensor
            Scaling factor of shape (B, 1, 1, 1).

        Returns
        -------
        torch.Tensor
            Scaled and concatenated tensor of shape (B, C_in+C_out, H, W).
        """
        return torch.cat([c_in * x, img_lr.to(x.dtype)], dim=1)

    # Properties delegated to the wrapped model
    amp_mode = _wrapped_property(
        "amp_mode",
        "model",
        "Set to ``True`` when using automatic mixed precision.",
    )
    profile_mode = _wrapped_property(
        "profile_mode",
        "model",
        "Set to ``True`` to enable profiling of the wrapped model.",
    )

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        sigma: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs: dict,
    ) -> torch.Tensor:
        """
        Forward pass of the EDMPrecondSuperResolution model wrapper.

        This method applies the EDM preconditioning to compute the denoised image
        from a noisy high-resolution image and low-resolution conditioning image.

        Parameters
        ----------
        x : torch.Tensor
            Noisy high-resolution image of shape (B, C_hr, H, W). The number of
            channels `C_hr` should be equal to `img_out_channels`.
        img_lr : torch.Tensor
            Low-resolution conditioning image of shape (B, C_lr, H, W). The number
            of channels `C_lr` should be equal to `img_in_channels`.
        sigma : torch.Tensor
            Noise level of shape (B) or (B, 1) or (B, 1, 1, 1).
        force_fp32 : bool, optional
            Whether to force FP32 precision regardless of the `use_fp16` attribute,
            by default False.
        **model_kwargs : dict
            Additional keyword arguments to pass to the underlying model
            `self.model` forward method.

        Returns
        -------
        torch.Tensor
            Denoised high-resolution image of shape (B, C_hr, H, W).

        Raises
        ------
        ValueError
            If the model output dtype doesn't match the expected dtype.
        """
        # Concatenate input channels
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        if img_lr is None:
            arg = c_in * x
        else:
            arg = self.scaling_fn(x, img_lr, c_in)
        arg = arg.to(dtype)

        F_x = self.model(
            arg,
            c_noise.flatten(),
            class_labels=None,
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    @staticmethod
    def round_sigma(sigma: Union[float, List, torch.Tensor]) -> torch.Tensor:
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float, List, torch.Tensor]
            Sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            Tensor representation of sigma values.

        See Also
        --------
        EDMPrecond.round_sigma
        """
        return EDMPrecond.round_sigma(sigma)


# NOTE: This is a deprecated version of the EDMPrecondSuperResolution model.
# This was used to maintain backwards compatibility and allow loading old models.
@dataclass
class EDMPrecondSRMetaData(ModelMetaData):
    """EDMPrecondSR meta data"""

    name: str = "EDMPrecondSR"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class EDMPrecondSR(EDMPrecondSuperResolution):
    """
    NOTE: This is a deprecated version of the EDMPrecondSuperResolution model.
    This was used to maintain backwards compatibility and allow loading old models.
    Please use the EDMPrecondSuperResolution model instead.

    Improved preconditioning proposed in the paper "Elucidating the Design Space of
    Diffusion-Based Generative Models" (EDM) for super-resolution tasks

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels (deprecated, not used).
    img_in_channels : int
        Number of input color channels.
    img_out_channels : int
        Number of output color channels.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min : float
        Minimum supported noise level, by default 0.0.
    sigma_max : float
        Maximum supported noise level, by default inf.
    sigma_data : float
        Expected standard deviation of the training data, by default 0.5.
    model_type :str
        Class name of the underlying model, by default "SongUNetPosEmbd".
    scale_cond_input : bool
        Whether to scale the conditional input (deprecated), by default True.
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    References:
    - Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    - Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self,
        img_resolution,
        img_channels,  # deprecated
        img_in_channels,
        img_out_channels,
        use_fp16=False,
        sigma_min=0.0,
        sigma_max=float("inf"),
        sigma_data=0.5,
        model_type="SongUNetPosEmbd",
        scale_cond_input=True,  # deprecated
        **model_kwargs,
    ):
        warnings.warn(
            "EDMPrecondSR is deprecated and will be removed in a future version. "
            "Please use EDMPrecondSuperResolution instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        super().__init__(
            img_resolution=img_resolution,
            img_in_channels=img_in_channels,
            img_out_channels=img_out_channels,
            use_fp16=use_fp16,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            model_type=model_type,
            **model_kwargs,
        )

        if scale_cond_input:
            warnings.warn(
                "The `scale_cond_input=True` option does not properly scale the conditional input "
                "and is deprecated. It is highly recommended to set `scale_cond_input=False`. "
                "However, for loading a checkpoint previously trained with `scale_cond_input=True`, "
                "this flag must be set to `True` to ensure compatibility. "
                "For more details, see https://github.com/NVIDIA/modulus/issues/229.",
                DeprecationWarning,
            )
            self.scaling_fn = self._legacy_scaling_fn

        # Store deprecated parameters for backward compatibility
        self.img_channels = img_channels
        self.scale_cond_input = scale_cond_input

    @staticmethod
    def _legacy_scaling_fn(
        x: torch.Tensor, img_lr: torch.Tensor, c_in: torch.Tensor
    ) -> torch.Tensor:
        """
        This function does not properly scale the conditional input
        (see https://github.com/NVIDIA/modulus/issues/229)
        and will be deprecated.

        Concatenate and scale the high-resolution and low-resolution tensors.

        Parameters
        ----------
        x : torch.Tensor
            Noisy high-resolution image of shape (B, C_hr, H, W).
        img_lr : torch.Tensor
            Low-resolution image of shape (B, C_lr, H, W).
        c_in : torch.Tensor
            Scaling factor of shape (B, 1, 1, 1).

        Returns
        -------
        torch.Tensor
            Scaled and concatenated tensor of shape (B, C_in+C_out, H, W).
        """
        return c_in * torch.cat([x, img_lr.to(x.dtype)], dim=1)

    def forward(
        self,
        x,
        img_lr,
        sigma,
        force_fp32=False,
        **model_kwargs,
    ):
        """
        Forward pass of the EDMPrecondSR model wrapper.

        Parameters
        ----------
        x : torch.Tensor
            Noisy high-resolution image of shape (B, C_hr, H, W).
        img_lr : torch.Tensor
            Low-resolution conditioning image of shape (B, C_lr, H, W).
        sigma : torch.Tensor
            Noise level of shape (B) or (B, 1) or (B, 1, 1, 1).
        force_fp32 : bool, optional
            Whether to force FP32 precision regardless of the `use_fp16` attribute,
            by default False.
        **model_kwargs : dict
            Additional keyword arguments to pass to the underlying model.

        Returns
        -------
        torch.Tensor
            Denoised high-resolution image of shape (B, C_hr, H, W).
        """
        return super().forward(
            x=x, img_lr=img_lr, sigma=sigma, force_fp32=force_fp32, **model_kwargs
        )


class VEPrecond_dfsr(torch.nn.Module):
    """
    Preconditioning for dfsr model, modified from class VEPrecond, where the input
    argument 'sigma' in forward propagation function is used to receive the timestep
    of the backward diffusion process.

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min : float
        Minimum supported noise level, by default 0.02.
    sigma_max : float
        Maximum supported noise level, by default 100.0.
    model_type :str
        Class name of the underlying model, by default "SongUNet".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference: Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models.
    Advances in neural information processing systems. 2020;33:6840-51.
    """

    def __init__(
        self,
        img_resolution: int,
        img_channels: int,
        label_dim: int = 0,
        use_fp16: bool = False,
        sigma_min: float = 0.02,
        sigma_max: float = 100.0,
        dataset_mean: float = 5.85e-05,
        dataset_scale: float = 4.79,
        model_type: str = "SongUNet",
        **model_kwargs: dict,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=self.img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )  # TODO needs better handling

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        # print("sigma: ", sigma)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_in = 1
        c_noise = sigma  # Change the definitation of c_noise to avoid -inf values for zero sigma

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )

        if F_x.dtype != dtype:
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        return F_x


class VEPrecond_dfsr_cond(torch.nn.Module):
    """
    Preconditioning for dfsr model with physics-informed conditioning input, modified
    from class VEPrecond, where the input argument 'sigma' in forward propagation function
    is used to receive the timestep of the backward diffusion process. The gradient of PDE
    residual with respect to the vorticity in the governing Navier-Stokes equation is computed
    as the physics-informed conditioning variable and is combined with the backward diffusion
    timestep before being sent to the underlying model for noise prediction.

    Parameters
    ----------
    img_resolution : int
        Image resolution.
    img_channels : int
        Number of color channels.
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min : float
        Minimum supported noise level, by default 0.02.
    sigma_max : float
        Maximum supported noise level, by default 100.0.
    model_type :str
        Class name of the underlying model, by default "SongUNet".
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    Note
    ----
    Reference:
    [1] Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and
    Poole, B., 2020. Score-based generative modeling through stochastic differential
    equations. arXiv preprint arXiv:2011.13456.
    [2] Shu D, Li Z, Farimani AB. A physics-informed diffusion model for high-fidelity
    flow field reconstruction. Journal of Computational Physics. 2023 Apr 1;478:111972.
    """

    def __init__(
        self,
        img_resolution: int,
        img_channels: int,
        label_dim: int = 0,
        use_fp16: bool = False,
        sigma_min: float = 0.02,
        sigma_max: float = 100.0,
        dataset_mean: float = 5.85e-05,
        dataset_scale: float = 4.79,
        model_type: str = "SongUNet",
        **model_kwargs: dict,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=model_kwargs["model_channels"] * 2,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )  # TODO needs better handling

        # modules to embed residual loss
        self.conv_in = torch.nn.Conv2d(
            img_channels,
            model_kwargs["model_channels"],
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="circular",
        )
        self.emb_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                img_channels,
                model_kwargs["model_channels"],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            torch.nn.GELU(),
            torch.nn.Conv2d(
                model_kwargs["model_channels"],
                model_kwargs["model_channels"],
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="circular",
            ),
        )
        self.dataset_mean = dataset_mean
        self.dataset_scale = dataset_scale

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_in = 1
        c_noise = sigma

        # Compute physics-informed conditioning information using vorticity residual
        dx = (
            self.voriticity_residual((x * self.dataset_scale + self.dataset_mean))
            / self.dataset_scale
        )
        x = self.conv_in(x)
        cond_emb = self.emb_conv(dx)
        x = torch.cat((x, cond_emb), dim=1)

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )

        if F_x.dtype != dtype:
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )
        return F_x

    def voriticity_residual(self, w, re=1000.0, dt=1 / 32):
        """
        Compute the gradient of PDE residual with respect to a given vorticity w using the
        spectrum method.

        Parameters
        ----------
        w: torch.Tensor
            The fluid flow data sample (vorticity).
        re: float
            The value of Reynolds number used in the governing Navier-Stokes equation.
        dt: float
            Time step used to compute the time-derivative of vorticity included in the governing
            Navier-Stokes equation.

        Returns
        -------
        torch.Tensor
            The computed vorticity gradient.
        """

        # w [b t h w]
        w = w.clone()
        w.requires_grad_(True)
        nx = w.size(2)
        device = w.device

        w_h = torch.fft.fft2(w[:, 1:-1], dim=[2, 3])
        # Wavenumbers in y-direction
        k_max = nx // 2
        N = nx
        k_x = (
            torch.cat(
                (
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ),
                0,
            )
            .reshape(N, 1)
            .repeat(1, N)
            .reshape(1, 1, N, N)
        )
        k_y = (
            torch.cat(
                (
                    torch.arange(start=0, end=k_max, step=1, device=device),
                    torch.arange(start=-k_max, end=0, step=1, device=device),
                ),
                0,
            )
            .reshape(1, N)
            .repeat(N, 1)
            .reshape(1, 1, N, N)
        )
        # Negative Laplacian in Fourier space
        lap = k_x**2 + k_y**2
        lap[..., 0, 0] = 1.0
        psi_h = w_h / lap

        u_h = 1j * k_y * psi_h
        v_h = -1j * k_x * psi_h
        wx_h = 1j * k_x * w_h
        wy_h = 1j * k_y * w_h
        wlap_h = -lap * w_h

        u = torch.fft.irfft2(u_h[..., :, : k_max + 1], dim=[2, 3])
        v = torch.fft.irfft2(v_h[..., :, : k_max + 1], dim=[2, 3])
        wx = torch.fft.irfft2(wx_h[..., :, : k_max + 1], dim=[2, 3])
        wy = torch.fft.irfft2(wy_h[..., :, : k_max + 1], dim=[2, 3])
        wlap = torch.fft.irfft2(wlap_h[..., :, : k_max + 1], dim=[2, 3])
        advection = u * wx + v * wy

        wt = (w[:, 2:, :, :] - w[:, :-2, :, :]) / (2 * dt)

        # establish forcing term
        x = torch.linspace(0, 2 * np.pi, nx + 1, device=device)
        x = x[0:-1]
        X, Y = torch.meshgrid(x, x)
        f = -4 * torch.cos(4 * Y)

        residual = wt + (advection - (1.0 / re) * wlap + 0.1 * w[:, 1:-1]) - f
        residual_loss = (residual**2).mean()
        dw = torch.autograd.grad(residual_loss, w)[0]

        return dw
