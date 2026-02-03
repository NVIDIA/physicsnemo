# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Reusable HEALPix tensor utilities and padding layers."""

import importlib

import torch
from jaxtyping import Float

from physicsnemo.core.version_check import check_version_spec

HEALPIXPAD_AVAILABLE = check_version_spec("earth2grid", "0.1.0", hard_fail=False)

if HEALPIXPAD_AVAILABLE:
    hpx_pad = importlib.import_module("earth2grid.healpix._padding").pad
else:

    def hpx_pad(*args, **kwargs):
        """Dummy symbol for missing earth2grid backend."""
        raise ImportError(
            "earth2grid is not installed, cannot use it as a backend for HEALPix "
            "padding. Install earth2grid from https://github.com/NVlabs/earth2grid.git "
            "to enable the accelerated path."
        )


def _raise_shape_error(name: str, tensor: torch.Tensor, message: str) -> None:
    raise ValueError(
        f"{name}: expected {message}, received tensor with shape {tuple(tensor.shape)}"
    )


class HEALPixFoldFaces(torch.nn.Module):
    r"""
    Fold the face dimension of a HEALPix tensor into the batch dimension.

    Parameters
    ----------
    enable_nhwc : bool, optional
        If ``True``, store the folded tensor in channels-last memory format.

    Forward
    -------
    tensor : torch.Tensor
        Input tensor of shape :math:`(B, F, C, H, W)` where :math:`F=12`.

    Outputs
    -------
    torch.Tensor
        Folded tensor of shape :math:`(B \cdot F, C, H, W)`.

    Examples
    --------
    >>> fold = HEALPixFoldFaces()
    >>> x = torch.randn(2, 12, 3, 4, 4)
    >>> fold(x).shape
    torch.Size([24, 3, 4, 4])
    """

    def __init__(self, enable_nhwc: bool = False) -> None:
        super().__init__()
        self.enable_nhwc = enable_nhwc

    def forward(
        self, tensor: Float[torch.Tensor, "batch faces channels height width"]
    ) -> Float[torch.Tensor, "batch_faces channels height width"]:
        r"""
        Fold the face dimension into the batch dimension.

        Parameters
        ----------
        tensor : torch.Tensor
            HEALPix data of shape :math:`(B, F, C, H, W)`.

        Returns
        -------
        torch.Tensor
            Folded tensor of shape :math:`(B \cdot F, C, H, W)`.
        """
        if not torch.compiler.is_compiling() and tensor.ndim != 5:
            _raise_shape_error("HEALPixFoldFaces.forward", tensor, "a 5D tensor")

        batch, faces, channels, height, width = tensor.shape
        folded = torch.reshape(tensor, shape=(batch * faces, channels, height, width))

        if self.enable_nhwc:
            folded = folded.to(memory_format=torch.channels_last)

        return folded


class HEALPixUnfoldFaces(torch.nn.Module):
    r"""
    Unfold a folded HEALPix tensor back to a face-major representation.

    Parameters
    ----------
    num_faces : int, optional
        Number of faces in the mesh (defaults to 12 for HEALPix).
    enable_nhwc : bool, optional
        If ``True``, expect channels-last tensors during unfolding.

    Forward
    -------
    tensor : torch.Tensor
        Input tensor of shape :math:`(B \cdot F, C, H, W)`.

    Outputs
    -------
    torch.Tensor
        Unfolded tensor of shape :math:`(B, F, C, H, W)`.

    Examples
    --------
    >>> unfold = HEALPixUnfoldFaces()
    >>> x = torch.randn(24, 3, 4, 4)
    >>> unfold(x).shape
    torch.Size([2, 12, 3, 4, 4])
    """

    def __init__(self, num_faces: int = 12, enable_nhwc: bool = False) -> None:
        super().__init__()
        self.num_faces = num_faces
        self.enable_nhwc = enable_nhwc

    def forward(
        self, tensor: Float[torch.Tensor, "batch_faces channels height width"]
    ) -> Float[torch.Tensor, "batch faces channels height width"]:
        r"""
        Unfold a tensor by restoring its explicit face dimension.

        Parameters
        ----------
        tensor : torch.Tensor
            Folded tensor of shape :math:`(B \cdot F, C, H, W)`.

        Returns
        -------
        torch.Tensor
            Unfolded tensor of shape :math:`(B, F, C, H, W)`.
        """
        if not torch.compiler.is_compiling():
            if tensor.ndim != 4:
                _raise_shape_error("HEALPixUnfoldFaces.forward", tensor, "a 4D tensor")
            if tensor.shape[0] % self.num_faces != 0:
                _raise_shape_error(
                    "HEALPixUnfoldFaces.forward",
                    tensor,
                    f"batch dimension divisible by num_faces={self.num_faces}",
                )

        batch_faces, channels, height, width = tensor.shape
        batch = batch_faces // self.num_faces
        return torch.reshape(
            tensor, shape=(batch, self.num_faces, channels, height, width)
        )


class HEALPixPadding(torch.nn.Module):
    r"""
    Reference padding layer for data on a HEALPix sphere.

    The input tensor must be folded with shape :math:`(B \cdot F, C, H, W)`.

    Parameters
    ----------
    padding : int
        Padding size to apply to each face.
    enable_nhwc : bool, optional
        If ``True``, operate on channels-last tensors.

    Forward
    -------
    data : torch.Tensor
        Folded input tensor of shape :math:`(B \cdot F, C, H, W)`.

    Outputs
    -------
    torch.Tensor
        Padded tensor of shape :math:`(B \cdot F, C, H + 2p, W + 2p)`.

    Examples
    --------
    >>> pad = HEALPixPadding(padding=1)
    >>> x = torch.randn(24, 3, 8, 8)
    >>> pad(x).shape
    torch.Size([24, 3, 10, 10])
    """

    def __init__(self, padding: int, enable_nhwc: bool = False) -> None:
        super().__init__()
        if not isinstance(padding, int) or padding < 1:
            raise ValueError(
                f"invalid value for 'padding', expected int > 0 but got {padding}"
            )

        self.p = padding
        self.d = (-2, -1)
        self.enable_nhwc = enable_nhwc
        self.fold = HEALPixFoldFaces(enable_nhwc=self.enable_nhwc)
        self.unfold = HEALPixUnfoldFaces(num_faces=12, enable_nhwc=self.enable_nhwc)

    def forward(
        self, data: Float[torch.Tensor, "batch_faces channels height width"]
    ) -> Float[torch.Tensor, "batch_faces channels padded_height padded_width"]:
        r"""Pad each face consistently with its neighbors on the HEALPix grid."""
        if not torch.compiler.is_compiling():
            if data.ndim != 4:
                _raise_shape_error("HEALPixPadding.forward", data, "a 4D tensor")

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push("HEALPixPadding:forward")

        data = self.unfold(data)

        f00, f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11 = [
            torch.squeeze(x, dim=1)
            for x in torch.split(tensor=data, split_size_or_sections=1, dim=1)
        ]

        p00 = self.pn(
            c=f00, t=f01, tl=f02, lft=f03, bl=f03, b=f04, br=f08, rgt=f05, tr=f01
        )
        p01 = self.pn(
            c=f01, t=f02, tl=f03, lft=f00, bl=f00, b=f05, br=f09, rgt=f06, tr=f02
        )
        p02 = self.pn(
            c=f02, t=f03, tl=f00, lft=f01, bl=f01, b=f06, br=f10, rgt=f07, tr=f03
        )
        p03 = self.pn(
            c=f03, t=f00, tl=f01, lft=f02, bl=f02, b=f07, br=f11, rgt=f04, tr=f00
        )

        p04 = self.pe(
            c=f04,
            t=f00,
            tl=self.tl(f00, f03),
            lft=f03,
            bl=f07,
            b=f11,
            br=self.br(f11, f08),
            rgt=f08,
            tr=f05,
        )
        p05 = self.pe(
            c=f05,
            t=f01,
            tl=self.tl(f01, f00),
            lft=f00,
            bl=f04,
            b=f08,
            br=self.br(f08, f09),
            rgt=f09,
            tr=f06,
        )
        p06 = self.pe(
            c=f06,
            t=f02,
            tl=self.tl(f02, f01),
            lft=f01,
            bl=f05,
            b=f09,
            br=self.br(f09, f10),
            rgt=f10,
            tr=f07,
        )
        p07 = self.pe(
            c=f07,
            t=f03,
            tl=self.tl(f03, f02),
            lft=f02,
            bl=f06,
            b=f10,
            br=self.br(f10, f11),
            rgt=f11,
            tr=f04,
        )

        p08 = self.ps(
            c=f08, t=f05, tl=f00, lft=f04, bl=f11, b=f11, br=f10, rgt=f09, tr=f09
        )
        p09 = self.ps(
            c=f09, t=f06, tl=f01, lft=f05, bl=f08, b=f08, br=f11, rgt=f10, tr=f10
        )
        p10 = self.ps(
            c=f10, t=f07, tl=f02, lft=f06, bl=f09, b=f09, br=f08, rgt=f11, tr=f11
        )
        p11 = self.ps(
            c=f11, t=f04, tl=f03, lft=f07, bl=f10, b=f10, br=f09, rgt=f08, tr=f08
        )

        res = torch.stack(
            (p00, p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11), dim=1
        )

        res = self.fold(res)

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

        return res

    def pn(
        self,
        c: torch.Tensor,
        t: torch.Tensor,
        tl: torch.Tensor,
        lft: torch.Tensor,
        bl: torch.Tensor,
        b: torch.Tensor,
        br: torch.Tensor,
        rgt: torch.Tensor,
        tr: torch.Tensor,
    ) -> torch.Tensor:
        r"""Pad a northern hemisphere face with its neighbors."""
        p = self.p
        d = self.d

        c = torch.cat((t.rot90(1, d)[..., -p:, :], c, b[..., :p, :]), dim=-2)

        left = torch.cat(
            (
                tl.rot90(2, d)[..., -p:, -p:],
                lft.rot90(-1, d)[..., -p:],
                bl[..., :p, -p:],
            ),
            dim=-2,
        )
        right = torch.cat((tr[..., -p:, :p], rgt[..., :p], br[..., :p, :p]), dim=-2)

        return torch.cat((left, c, right), dim=-1)

    def pe(
        self,
        c: torch.Tensor,
        t: torch.Tensor,
        tl: torch.Tensor,
        lft: torch.Tensor,
        bl: torch.Tensor,
        b: torch.Tensor,
        br: torch.Tensor,
        rgt: torch.Tensor,
        tr: torch.Tensor,
    ) -> torch.Tensor:
        r"""Pad an equatorial face with its neighbors."""
        p = self.p

        c = torch.cat((t[..., -p:, :], c, b[..., :p, :]), dim=-2)

        left = torch.cat((tl[..., -p:, -p:], lft[..., -p:], bl[..., :p, -p:]), dim=-2)
        right = torch.cat((tr[..., -p:, :p], rgt[..., :p], br[..., :p, :p]), dim=-2)

        return torch.cat((left, c, right), dim=-1)

    def ps(
        self,
        c: torch.Tensor,
        t: torch.Tensor,
        tl: torch.Tensor,
        lft: torch.Tensor,
        bl: torch.Tensor,
        b: torch.Tensor,
        br: torch.Tensor,
        rgt: torch.Tensor,
        tr: torch.Tensor,
    ) -> torch.Tensor:
        r"""Pad a southern hemisphere face with its neighbors."""
        p = self.p
        d = self.d

        c = torch.cat((t[..., -p:, :], c, b.rot90(1, d)[..., :p, :]), dim=-2)

        left = torch.cat((tl[..., -p:, -p:], lft[..., -p:], bl[..., :p, -p:]), dim=-2)
        right = torch.cat(
            (tr[..., -p:, :p], rgt.rot90(-1, d)[..., :p], br.rot90(2, d)[..., :p, :p]),
            dim=-2,
        )

        return torch.cat((left, c, right), dim=-1)

    def tl(self, top: torch.Tensor, lft: torch.Tensor) -> torch.Tensor:
        r"""Assemble the missing top-left corner patch."""
        ret = torch.zeros_like(top)[..., : self.p, : self.p]

        ret[..., -1, -1] = 0.5 * top[..., -1, 0] + 0.5 * lft[..., 0, -1]

        for i in range(1, self.p):
            ret[..., -i - 1, -i:] = top[..., -i - 1, :i]
            ret[..., -i:, -i - 1] = lft[..., :i, -i - 1]
            ret[..., -i - 1, -i - 1] = (
                0.5 * top[..., -i - 1, 0] + 0.5 * lft[..., 0, -i - 1]
            )

        return ret

    def br(self, b: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        r"""Assemble the missing bottom-right corner patch."""
        ret = torch.zeros_like(b)[..., : self.p, : self.p]

        ret[..., 0, 0] = 0.5 * b[..., 0, -1] + 0.5 * r[..., -1, 0]

        for i in range(1, self.p):
            ret[..., :i, i] = r[..., -i:, i]
            ret[..., i, :i] = b[..., i, -i:]
            ret[..., i, i] = 0.5 * b[..., i, -1] + 0.5 * r[..., -1, i]

        return ret


class HEALPixPaddingv2(torch.nn.Module):
    r"""
    Accelerated padding layer for HEALPix data using the optional ``earth2grid`` backend.

    Parameters
    ----------
    padding : int
        Padding size to apply to each face.

    Forward
    -------
    x : torch.Tensor
        Folded tensor of shape :math:`(B \cdot F, C, H, W)`.

    Outputs
    -------
    torch.Tensor
        Padded tensor of shape :math:`(B \cdot F, C, H + 2p, W + 2p)`.

    Examples
    --------
    >>> pad = HEALPixPaddingv2(padding=1)
    >>> x = torch.randn(24, 3, 8, 8)
    >>> y = pad(x)
    >>> y.shape
    torch.Size([24, 3, 10, 10])
    """

    def __init__(self, padding: int) -> None:  # pragma: no cover
        super().__init__()
        self.unfold = HEALPixUnfoldFaces(num_faces=12)
        self.fold = HEALPixFoldFaces()
        self.padding = padding

    def forward(
        self, x: Float[torch.Tensor, "batch_faces channels height width"]
    ) -> Float[torch.Tensor, "batch_faces channels padded_height padded_width"]:
        r"""Apply HEALPix-aware padding using the ``earth2grid`` CUDA backend."""
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push("HEALPixPaddingv2:forward")

        unfolded = self.unfold(x)
        padded = hpx_pad(unfolded, self.padding)
        result = self.fold(padded)

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

        return result


class HEALPixLayer(torch.nn.Module):
    r"""
    Wrapper that applies an arbitrary 2D layer to each HEALPix face independently.

    Parameters
    ----------
    layer : torch.nn.Module
        A callable layer class such as ``torch.nn.Conv2d``.
    **kwargs
        Parameters forwarded to ``layer``. Recognized extras:

        - ``enable_nhwc``: bool, optional
            If ``True``, use channels-last memory format.
        - ``enable_healpixpad``: bool, optional
            If ``True`` and CUDA ``earth2grid`` is available, use accelerated padding.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(..., F=12, H, W)`.

    Outputs
    -------
    torch.Tensor
        Tensor with the same leading dimensions and transformed spatial dimensions.

    Examples
    --------
    >>> conv = HEALPixLayer(torch.nn.Conv2d, in_channels=3, out_channels=8, kernel_size=3)
    >>> x = torch.randn(24, 3, 16, 16)
    >>> conv(x).shape
    torch.Size([24, 8, 16, 16])
    """

    def __init__(self, layer, **kwargs) -> None:
        super().__init__()
        layers = []

        enable_nhwc = kwargs.pop("enable_nhwc", False)
        enable_healpixpad = kwargs.pop("enable_healpixpad", False)

        if (
            layer.__bases__[0] is torch.nn.modules.conv._ConvNd
            and kwargs.get("kernel_size", 3) > 1
        ):
            kwargs["padding"] = 0
            kernel_size = kwargs.get("kernel_size", 3)
            dilation = kwargs.get("dilation", 1)
            padding = ((kernel_size - 1) // 2) * dilation
            if (
                enable_healpixpad
                and HEALPIXPAD_AVAILABLE
                and torch.cuda.is_available()
                and not enable_nhwc
            ):  # pragma: no cover
                layers.append(HEALPixPaddingv2(padding=padding))
            else:
                layers.append(HEALPixPadding(padding=padding, enable_nhwc=enable_nhwc))

        layers.append(layer(**kwargs))
        self.layers = torch.nn.Sequential(*layers)

        if enable_nhwc:
            self.layers = self.layers.to(memory_format=torch.channels_last)

    def forward(
        self, x: Float[torch.Tensor, "batch_faces channels height width"]
    ) -> Float[torch.Tensor, "batch_faces channels height width"]:
        r"""Forward pass for the HEALPix layer wrapper."""
        return self.layers(x)


# TODO: the below are not actually geometry-aware; update or remove
# Since the `HEALPixLayer` only applies padding if the wrapped layer is a convolution,
# these do not actually apply healpix padding before pooling


class HEALPixMaxPool(torch.nn.Module):
    r"""
    HEALPix-aware max pooling wrapper around ``torch.nn.MaxPool2d``.

    Parameters
    ----------
    geometry_layer : torch.nn.Module, optional
        Wrapper that applies pooling per HEALPix face. Defaults to :class:`HEALPixLayer`.
    pooling : int, optional
        Kernel size and stride for pooling. Defaults to ``2``.
    enable_nhwc : bool, optional
        If ``True``, operate on channels-last tensors.
    enable_healpixpad : bool, optional
        Enable HEALPix padding when available.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B \cdot F, C, H, W)`.

    Outputs
    -------
    torch.Tensor
        Pooled tensor with spatial dimensions downsampled by ``pooling``.
    """

    def __init__(
        self,
        geometry_layer: torch.nn.Module = HEALPixLayer,
        pooling: int = 2,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ) -> None:
        super().__init__()
        self.maxpool = geometry_layer(
            layer=torch.nn.MaxPool2d,
            kernel_size=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
        )

    def forward(
        self, x: Float[torch.Tensor, "batch_faces channels height width"]
    ) -> Float[torch.Tensor, "batch_faces channels pooled_height pooled_width"]:
        return self.maxpool(x)


class HEALPixAvgPool(torch.nn.Module):
    r"""
    HEALPix-aware average pooling wrapper around ``torch.nn.AvgPool2d``.

    Parameters
    ----------
    geometry_layer : torch.nn.Module, optional
        Wrapper that applies pooling per HEALPix face. Defaults to :class:`HEALPixLayer`.
    pooling : int, optional
        Kernel size and stride for pooling. Defaults to ``2``.
    enable_nhwc : bool, optional
        If ``True``, operate on channels-last tensors.
    enable_healpixpad : bool, optional
        Enable HEALPix padding when available.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B \cdot F, C, H, W)`.

    Outputs
    -------
    torch.Tensor
        Pooled tensor with spatial dimensions downsampled by ``pooling``.
    """

    def __init__(
        self,
        geometry_layer: torch.nn.Module = HEALPixLayer,
        pooling: int = 2,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ) -> None:
        super().__init__()
        self.avgpool = geometry_layer(
            layer=torch.nn.AvgPool2d,
            kernel_size=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
        )

    def forward(
        self, x: Float[torch.Tensor, "batch_faces channels height width"]
    ) -> Float[torch.Tensor, "batch_faces channels pooled_height pooled_width"]:
        return self.avgpool(x)
