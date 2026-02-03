# ignore_header_test
# ruff: noqa: E402
"""
Pix2PixUnet model. This code was modified from, https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

The following license is provided from their source,

Copyright (C) 2019 NVIDIA Corporation. Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu.
BSD License. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


--------------------------- LICENSE FOR pytorch-CycleGAN-and-pix2pix ----------------
Copyright (c) 2017, Jun-Yan Zhu and Taesung Park
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import functools
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from jaxtyping import Float
from torch.nn import init

import physicsnemo  # noqa: F401 for docs
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module

Tensor = torch.Tensor


def init_weights(
    net: nn.Module, init_type: str = "normal", init_gain: float = 0.02
) -> None:
    r"""Initialize network weights.

    Parameters
    ----------
    net : nn.Module
        Network to be initialized.
    init_type : str, optional, default="normal"
        The name of an initialization method: ``'normal'``, ``'xavier'``,
        ``'kaiming'``, or ``'orthogonal'``.
    init_gain : float, optional, default=0.02
        Scaling factor for normal, xavier and orthogonal initialization.

    Returns
    -------
    None
        Modifies the network weights in-place.

    Note
    ----
    We use ``'normal'`` in the original pix2pix and CycleGAN paper. But xavier
    and kaiming might work better for some applications.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


@dataclass
class MetaData(ModelMetaData):
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = False  # Reflect padding not supported in bfloat16
    amp_gpu: bool = True
    # Inference
    onnx: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = True
    auto_grad: bool = True


class Pix2PixUnet(Module):
    r"""Convolutional encoder-decoder based on pix2pix generator models using UNet.

    Note
    ----
    The pix2pix with UNet architecture only supports 2D fields.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    n_downsampling : int
        Number of downsampling layers in UNet.
    filter_size : int, optional, default=64
        Number of filters in the first convolution layer.
    norm_layer : type, optional, default=nn.BatchNorm2d
        Normalization layer class.
    use_dropout : bool, optional, default=False
        Whether to use dropout layers.
    gpu_ids : List[int], optional, default=[]
        List of GPU IDs for DataParallel.

        .. deprecated::
            The ``gpu_ids`` parameter is deprecated and will be removed in a future
            release. Use standard PyTorch device placement instead by calling
            ``model.to(device)`` after instantiation.

    Forward
    -------
    input : torch.Tensor
        Input tensor of shape :math:`(N, C_{in}, H, W)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(N, C_{out}, H, W)`.

    Examples
    --------
    >>> import torch
    >>> import physicsnemo
    >>> model = physicsnemo.models.pix2pix.Pix2PixUnet(
    ...     in_channels=3,
    ...     out_channels=3,
    ...     n_downsampling=8,
    ...     filter_size=64)
    >>> input = torch.randn(1, 3, 256, 256)  # (N, C, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([1, 3, 256, 256])

    Note
    ----
    Reference: Isola, Phillip, et al. "Image-To-Image translation with conditional
    adversarial networks" Conference on Computer Vision and Pattern Recognition, 2017.
    https://arxiv.org/abs/1611.07004

    Reference: Wang, Ting-Chun, et al. "High-Resolution image synthesis and semantic
    manipulation with conditional GANs" Conference on Computer Vision and Pattern
    Recognition, 2018. https://arxiv.org/abs/1711.11585

    Note
    ----
    Based on the implementation: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_downsampling: int,
        filter_size: int = 64,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
        gpu_ids: Optional[List[int]] = None,
    ):
        if not (filter_size > 0 and n_downsampling >= 0):
            raise ValueError("Invalid arch params")
        super().__init__(meta=MetaData())

        # Handle deprecated gpu_ids parameter
        if gpu_ids is None:
            gpu_ids = []
        if gpu_ids:
            warnings.warn(
                "The 'gpu_ids' parameter is deprecated and will be removed in a "
                "future release. Use standard PyTorch device placement instead by "
                "calling 'model.to(device)' after instantiation.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Store constructor parameters as attributes for introspection
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_downsampling = n_downsampling
        self.filter_size = filter_size
        self.norm_layer = norm_layer
        self.use_dropout = use_dropout

        # device (deprecated functionality, kept for backward compatibility)
        self.gpu_ids = gpu_ids
        self.model_device = (
            torch.device("cuda:{}".format(self.gpu_ids[0]))
            if self.gpu_ids
            else torch.device("cpu")
        )

        # generate Unet model recursively
        net = UnetGenerator(
            in_channels,
            out_channels,
            n_downsampling,
            filter_size,
            norm_layer,
            use_dropout,
        )
        if len(gpu_ids) > 0:
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
        # init_weights(net)

        self.netG = net

    def __patch_instance_norm_state_dict(
        self,
        state_dict: Dict[str, Any],
        module: nn.Module,
        keys: List[str],
        i: int = 0,
    ) -> None:
        r"""Patch InstanceNorm state dict for compatibility with older checkpoints.

        Removes ``running_mean``, ``running_var``, and ``num_batches_tracked`` keys
        from InstanceNorm layers if the module doesn't track running stats.

        Parameters
        ----------
        state_dict : Dict[str, Any]
            The state dictionary to patch (modified in-place).
        module : nn.Module
            The current module being checked.
        keys : List[str]
            List of keys representing the path to the current parameter.
        i : int, optional, default=0
            Current index in the keys list.

        Returns
        -------
        None
            Modifies ``state_dict`` in-place.
        """
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "running_mean" or key == "running_var"
            ):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "num_batches_tracked"
            ):
                state_dict.pop(".".join(keys))
        else:
            self.__patch_instance_norm_state_dict(
                state_dict, getattr(module, key), keys, i + 1
            )

    def load_networks(self, model_path: str) -> None:
        r"""Load network weights from a checkpoint file.

        This method handles loading weights from pix2pix-style checkpoint files,
        including compatibility patches for older InstanceNorm checkpoints.

        Parameters
        ----------
        model_path : str
            Path to the checkpoint file containing model weights.

        Returns
        -------
        None
            Loads weights into ``self.netG`` in-place.

        Note
        ----
        This method is provided for compatibility with pix2pix-style checkpoints.
        For PhysicsNeMo checkpoints, use :meth:`physicsnemo.Module.from_checkpoint`
        instead.
        """
        net = self.netG
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        state_dict = torch.load(model_path, map_location=str(self.model_device))
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(
            state_dict.keys()
        ):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, net, key.split("."))
        net.load_state_dict(state_dict)

    def test(
        self, input: Float[Tensor, "batch in_channels height width"]
    ) -> Float[Tensor, "batch out_channels height width"]:
        r"""Run inference without gradient computation.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape :math:`(N, C_{in}, H, W)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(N, C_{out}, H, W)`.
        """
        with torch.no_grad():
            return self.forward(input)

    def forward(
        self, input: Float[Tensor, "batch in_channels height width"]
    ) -> Float[Tensor, "batch out_channels height width"]:
        r"""Forward pass through the UNet generator.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape :math:`(N, C_{in}, H, W)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(N, C_{out}, H, W)`.
        """
        # Input validation
        if not torch.compiler.is_compiling():
            if input.ndim != 4:
                raise ValueError(
                    f"Expected 4D input tensor (N, C, H, W), "
                    f"got {input.ndim}D tensor with shape {tuple(input.shape)}"
                )

        return self.netG(input)


class UnetGenerator(Module):
    r"""Create a UNet-based generator.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    n_downsampling : int
        Number of downsampling layers in UNet.
    filter_size : int, optional, default=64
        Number of filters in the first convolution layer.
    norm_layer : type, optional, default=nn.BatchNorm2d
        Normalization layer class.
    use_dropout : bool, optional, default=False
        Whether to use dropout layers.

    Forward
    -------
    input : torch.Tensor
        Input tensor of shape :math:`(N, C_{in}, H, W)`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(N, C_{out}, H, W)`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_downsampling: int,
        filter_size: int = 64,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
    ):
        super().__init__(meta=None)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            filter_size * 8,
            filter_size * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )  # add the innermost layer
        for i in range(
            n_downsampling - 5
        ):  # add intermediate layers with filter_size * 8 filters
            unet_block = UnetSkipConnectionBlock(
                filter_size * 8,
                filter_size * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )

        # gradually reduce the number of filters from filter_size * 8 to filter_size
        unet_block = UnetSkipConnectionBlock(
            filter_size * 4,
            filter_size * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
        )
        unet_block = UnetSkipConnectionBlock(
            filter_size * 2,
            filter_size * 4,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
        )
        unet_block = UnetSkipConnectionBlock(
            filter_size,
            filter_size * 2,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
        )

        self.model = UnetSkipConnectionBlock(
            out_channels,
            filter_size,
            input_nc=in_channels,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )  # add the outermost layer

    def forward(
        self, input: Float[Tensor, "batch in_channels height width"]
    ) -> Float[Tensor, "batch out_channels height width"]:
        r"""Forward pass through the UNet generator.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape :math:`(N, C_{in}, H, W)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape :math:`(N, C_{out}, H, W)`.
        """
        return self.model(input)


class UnetSkipConnectionBlock(Module):
    r"""A UNet submodule with skip connections.

    Parameters
    ----------
    outer_nc : int
        Number of filters in the outer conv layer.
    inner_nc : int
        Number of filters in the inner conv layer.
    input_nc : int, optional, default=None
        Number of channels in input images/features. If ``None``, uses ``outer_nc``.
    submodule : nn.Module, optional, default=None
        Previously defined submodule to nest inside this block.
    outermost : bool, optional, default=False
        If ``True``, this module is the outermost layer.
    innermost : bool, optional, default=False
        If ``True``, this module is the innermost layer.
    norm_layer : type, optional, default=nn.BatchNorm2d
        Normalization layer class.
    use_dropout : bool, optional, default=False
        Whether to use dropout layers.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(N, C_{in}, H, W)`.

    Outputs
    -------
    torch.Tensor
        Output tensor. For outermost block: :math:`(N, C_{out}, H', W')`.
        For inner blocks: :math:`(N, C_{in} + C_{inner}, H, W)` due to skip connection.
    """

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int = None,
        submodule: nn.Module = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
    ):
        super().__init__(meta=None)
        self.outermost = outermost
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(
        self, x: Float[Tensor, "batch channels height width"]
    ) -> Float[Tensor, "batch out_channels height width"]:
        r"""Forward pass with skip connection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape :math:`(N, C_{in}, H, W)`.

        Returns
        -------
        torch.Tensor
            Output tensor with skip connection concatenated for inner blocks.
        """
        if self.outermost:
            # Outermost block: no skip connection
            return self.model(x)
        else:
            # Inner blocks: concatenate input with output for skip connection
            return torch.cat([x, self.model(x)], 1)  # (N, C_in + C_out, H, W)
