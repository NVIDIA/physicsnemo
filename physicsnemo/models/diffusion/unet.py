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
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Set, Tuple, Union

import torch

from physicsnemo.models.diffusion.utils import _safe_setattr
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

network_module = importlib.import_module("physicsnemo.models.diffusion")


@dataclass
class MetaData(ModelMetaData):
    name: str = "UNet"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = True
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class UNet(Module):  # TODO a lot of redundancy, need to clean up
    r"""
    This interface provides a U-Net wrapper for CorrDiff deterministic
    regression model (and other deterministic downsampling models).
    It supports the following architectures:

    - :class:`~physicsnemo.models.diffusion.song_unet.SongUNet`

    - :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosEmbd`

    - :class:`~physicsnemo.models.diffusion.song_unet.SongUNetPosLtEmbd`

    - :class:`~physicsnemo.models.diffusion.dhariwal_unet.DhariwalUNet`

    It shares the same architeture as a conditional diffusion model. It does so
    by concatenating a conditioning image to a zero-filled latent state, and by
    setting the noise level and the class labels to zero.

    Parameters
    -----------
    img_resolution : Union[int, Tuple[int, int]]
        The resolution of the input/output image. If a single int is provided,
        then the image is assumed to be square.
    img_in_channels : int
        Number of channels in the input image.
    img_out_channels : int
        Number of channels in the output image.
    use_fp16: bool, optional, default=False
        Execute the underlying model at FP16 precision.
    model_type: Literal['SongUNet', 'SongUNetPosEmbd', 'SongUNetPosLtEmbd',
    'DhariwalUNet'], default='SongUNetPosEmbd'
        Class name of the underlying architecture. Must be one of the following:
        'SongUNet', 'SongUNetPosEmbd', 'SongUNetPosLtEmbd', 'DhariwalUNet'.
    **model_kwargs : dict
        Keyword arguments passed to the underlying architecture `__init__` method.

    Please refer to the documentation of these classes for details on how to call
    and use these models directly.

    Forward
    -------
    x : torch.Tensor
        The input tensor, typically zero-filled, of shape :math:`(B, C_{in}, H_{in}, W_{in})`.
    img_lr : torch.Tensor
        Conditioning image of shape :math:`(B, C_{lr}, H_{in}, W_{in})`.
    **model_kwargs : dict
        Additional keyword arguments to pass to the underlying architecture
        forward method.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, H_{in}, W_{in})` (same
        spatial dimensions as the input).

    """

    __model_checkpoint_version__ = "0.2.0"
    __supported_model_checkpoint_version__ = {
        "0.1.0": "Loading UNet checkpoint from older version 0.1.0 (current version is 0.2.0). This version is still supported, but consider re-saving the model to upgrade to version 0.2.0 and remove this warning."
    }

    # Classes that can be wrapped by this UNet class.
    _wrapped_classes: Set[str] = {
        "SongUNetPosEmbd",
        "SongUNetPosLtEmbd",
        "SongUNet",
        "DhariwalUNet",
    }

    # Arguments of the __init__ method that can be overridden with the
    # ``Module.from_checkpoint`` method. Here, since we use splatted arguments
    # for the wrapped model instance, we allow overriding of any overridable
    # argument of the wrapped classes.
    _overridable_args: Set[str] = set.union(
        *(
            getattr(getattr(network_module, cls_name), "_overridable_args", set())
            for cls_name in _wrapped_classes
        )
    )

    @classmethod
    def _backward_compat_arg_mapper(
        cls, version: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map arguments from older versions to current version format.

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
        # Call parent class method first
        args = super()._backward_compat_arg_mapper(version, args)

        if version == "0.1.0":
            # In version 0.1.0, img_channels was unused
            if "img_channels" in args:
                _ = args.pop("img_channels")

            # Sigma parameters are also unused
            if "sigma_min" in args:
                _ = args.pop("sigma_min")
            if "sigma_max" in args:
                _ = args.pop("sigma_max")
            if "sigma_data" in args:
                _ = args.pop("sigma_data")

        return args

    def __init__(
        self,
        img_resolution: Union[int, Tuple[int, int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        model_type: Literal[
            "SongUNetPosEmbd", "SongUNetPosLtEmbd", "SongUNet", "DhariwalUNet"
        ] = "SongUNetPosEmbd",
        **model_kwargs: dict,
    ):
        super().__init__(meta=MetaData)

        # Validation
        if model_type not in self._wrapped_classes:
            raise ValueError(
                f"Model type '{model_type}' is not supported. "
                f"Must be one of: {', '.join(self._wrapped_classes)}"
            )

        # for compatibility with older versions that took only 1 dimension
        if isinstance(img_resolution, int):
            self.img_shape_x = self.img_shape_y = img_resolution
        else:
            self.img_shape_y = img_resolution[0]
            self.img_shape_x = img_resolution[1]

        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels

        self.use_fp16 = use_fp16
        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_in_channels + img_out_channels,
            out_channels=img_out_channels,
            **model_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs: dict,
    ) -> torch.Tensor:

        # SR: concatenate input channels
        if img_lr is not None:
            x = torch.cat((x, img_lr), dim=1)

        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        F_x = self.model(
            x.to(dtype),  # (c_in * x).to(dtype),
            torch.zeros(x.shape[0], dtype=dtype, device=x.device),  # c_noise.flatten()
            class_labels=None,
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, " f"but got {F_x.dtype} instead."
            )

        # skip connection
        D_x = F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma: Union[float, List, torch.Tensor]) -> torch.Tensor:
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float, List, torch.Tensor]
            The sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            The tensor representation of the provided sigma value(s).
        """
        return torch.as_tensor(sigma)

    @property
    def amp_mode(self):
        """
        Property that controls the automatic mixed precision mode of the
        underlying architecture. ``True`` means that the underlying architecture
        will automatically cast tensors to the appropriate
        precision. ``False`` means that the underlying architecture will not
        automatically cast the input and output tensors to the appropriate
        precision.
        """
        return getattr(self.model, "amp_mode", None)

    @amp_mode.setter
    def amp_mode(self, value: bool):
        """
        Update ``amp_mode`` on the wrapped architecture and its sub-modules.
        """
        if not isinstance(value, bool):
            raise TypeError("amp_mode must be a boolean value.")
        self.model.apply(lambda m: _safe_setattr(m, "amp_mode", value))

    @property
    def profile_mode(self):
        """
        Property that controls the profiling mode of the underlying architecture.
        ``True`` means that the underlying architecture will be profiled.
        ``False`` means that the underlying architecture will not be profiled.
        """
        return getattr(self.model, "profile_mode", None)

    @profile_mode.setter
    def profile_mode(self, value: bool):
        """
        Update ``profile_mode`` on the wrapped architecture and its sub-modules.
        """
        if not isinstance(value, bool):
            raise TypeError("profile_mode must be a boolean value.")
        self.model.apply(lambda m: _safe_setattr(m, "profile_mode", value))


# TODO: implement amp_mode and profile_mode properties for StormCastUNet (same
# as UNet)
class StormCastUNet(Module):
    """
    U-Net wrapper for StormCast; used so the same Song U-Net network can be re-used for this model.

    Parameters
    -----------
    img_resolution : int or List[int]
        The resolution of the input/output image.
    img_channels : int
         Number of color channels.
    img_in_channels : int
        Number of input color channels.
    img_out_channels : int
        Number of output color channels.
    use_fp16: bool, optional
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min: float, optional
        Minimum supported noise level, by default 0.
    sigma_max: float, optional
        Maximum supported noise level, by default float('inf').
    sigma_data: float, optional
        Expected standard deviation of the training data, by default 0.5.
    model_type: str, optional
        Class name of the underlying model, by default 'SongUNet'.
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    """

    def __init__(
        self,
        img_resolution,
        img_in_channels,
        img_out_channels,
        use_fp16=False,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=0.5,
        model_type="SongUNet",
        **model_kwargs,
    ):
        super().__init__(meta=MetaData("StormCastUNet"))

        if isinstance(img_resolution, int):
            self.img_shape_x = self.img_shape_y = img_resolution
        else:
            self.img_shape_x = img_resolution[0]
            self.img_shape_y = img_resolution[1]

        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels

        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_in_channels,
            out_channels=img_out_channels,
            **model_kwargs,
        )

    def forward(self, x, force_fp32=False, **model_kwargs):
        """Run a forward pass of the StormCast regression U-Net.

        Args:
            x (torch.Tensor): input to the U-Net
            force_fp32 (bool, optional): force casting to fp_32 if True. Defaults to False.

        Raises:
            ValueError: If input data type is a mismatch with provided options

        Returns:
            D_x (torch.Tensor): Output (prediction) of the U-Net
        """

        x = x.to(torch.float32)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        F_x = self.model(
            x.to(dtype),
            torch.zeros(x.shape[0], dtype=x.dtype, device=x.device),
            class_labels=None,
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        D_x = F_x.to(torch.float32)
        return D_x
