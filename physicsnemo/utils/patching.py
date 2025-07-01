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
import random
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from einops import rearrange
from torch import Tensor

"""
This module defines utilities, including classes and functions, for domain
decomposition.
"""


class BasePatching2D(ABC):
    """
    Abstract base class for 2D image patching operations.

    This class provides a foundation for implementing various image patching
    strategies.
    It handles basic parameter validation and provides default methods for
    patching and fusing.

    It is designed to be extensible to support different patching strategies.
    Any new patching strategy for 2D images should inherit from this class and
    implement the abstract methods.

    Parameters
    ----------
    img_shape : Tuple[int, int]
        The height and width of the full input images :math:`(H, W)`.
    patch_shape : Tuple[int, int]
        The height and width of the patches to extract :math:`(H_p, W_p)`.
    """

    def __init__(
        self, img_shape: Tuple[int, int], patch_shape: Tuple[int, int]
    ) -> None:
        # Check that img_shape and patch_shape are 2D
        if len(img_shape) != 2:
            raise ValueError(f"img_shape must be 2D, got {len(img_shape)}D")
        if len(patch_shape) != 2:
            raise ValueError(f"patch_shape must be 2D, got {len(patch_shape)}D")

        # Make sure patches fit within the image
        if any(p > i for p, i in zip(patch_shape, img_shape)):
            warnings.warn(
                f"Patch shape {patch_shape} is larger than "
                f"image shape {img_shape}. "
                f"Patches will be cropped to fit within the image."
            )
        self.img_shape = img_shape
        self.patch_shape = tuple(min(p, i) for p, i in zip(patch_shape, img_shape))

    @abstractmethod
    def apply(self, input: Tensor, **kwargs) -> Tensor:
        """
        Apply the patching operation to a batch of full images.

        Parameters
        ----------
        input : Tensor
            Batch of full input images of shape :math:`(B, C, H, W)`.
        **kwargs : dict
            Additional keyword arguments specific to the patching
            implementation.

        Returns
        -------
        Tensor
            Patched tensor, shape depends on specific implementation.
        """
        pass

    def fuse(self, input: Tensor, **kwargs) -> Tensor:
        """
        Fuse patches back into a complete image.

        Parameters
        ----------
        input : Tensor
            Input tensor containing patches. Shape depends on specific implementation.
        **kwargs : dict
            Additional keyword arguments specific to the fusion implementation.

        Returns
        -------
        Tensor
            Fused tensor. Shape depends on specific implementation.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("'fuse' method must be implemented in subclasses.")

    def global_index(
        self, batch_size: int, device: Union[torch.device, str] = "cpu"
    ) -> Tensor:
        """
        Returns a tensor containing the global indices for each patch.

        Global indices correspond to :math:`(y, x)` global grid coordinates of each
        element within the original image (before patching). It is typically
        used to keep track of the original position of each patch in the
        original image.

        Parameters
        ----------
        batch_size : int
            The size :math:`B` of the batch of images to patch.
        device : Union[torch.device, str], default="cpu"
            Proper device to initialize ``global_index`` on.

        Returns
        -------
        Tensor
            A tensor of shape :math:`(P, 2, H_p, W_p)`, where :math:`P` is the
            number of patches to extract (corresponds to ``self.patch_num`` for
            classes that implement this attribute).
            The y-coordinate is stored in ``global_index[:, 0, :, :]`` and the
            x-coordinate is stored in ``global_index[:, 1, :, :]``.
        """
        Ny = torch.arange(self.img_shape[0], device=device).int()
        Nx = torch.arange(self.img_shape[1], device=device).int()
        grid = torch.stack(torch.meshgrid(Ny, Nx, indexing="ij"), dim=0).unsqueeze(0)
        global_index = self.apply(grid).long()
        return global_index


class RandomPatching2D(BasePatching2D):
    """
    Class for randomly extracting patches from 2D images.

    This class provides utilities to randomly extract patches from a batch of full
    images represented as 4D tensors. It maintains a list of random patch indices
    that can be reset as needed.

    Parameters
    ----------
    img_shape : Tuple[int, int]
        The height and width :math:`(H, W)` of the full input images.
    patch_shape : Tuple[int, int]
        The height and width :math:`(H_p, W_p)` of the patches to extract.
    patch_num : int
        The number of patches :math:`P` to extract.

    Attributes
    ----------
    patch_indices : List[Tuple[int, int]]
        The indices of the patches to extract from the images. These indices
        correspond to the :math:`(y, x)` coordinates of the upper left corner of
        each patch.

    See Also
    --------
    :class:`physicsnemo.utils.patching.BasePatching2D`
        The base class providing the patching interface.
    :class:`physicsnemo.utils.patching.GridPatching2D`
        Alternative patching strategy using deterministic patch locations.
    """

    def __init__(
        self, img_shape: Tuple[int, int], patch_shape: Tuple[int, int], patch_num: int
    ) -> None:

        super().__init__(img_shape, patch_shape)
        self._patch_num = patch_num
        # Generate the indices of the patches to extract
        self.reset_patch_indices()

    @property
    def patch_num(self) -> int:
        """
        Get the number of patches to extract.

        Returns
        -------
        int
            The number of patches :math:`P` to extract.
        """
        return self._patch_num

    def set_patch_num(self, value: int) -> None:
        """
        Set the number of patches to extract and reset patch indices.
        This is the only way to modify the ``patch_num`` attribute.

        Parameters
        ----------
        value : int
            The new number of patches :math:`P` to extract.
        """
        self._patch_num = value
        self.reset_patch_indices()

    def reset_patch_indices(self) -> None:
        """
        Generate new random indices for the patches to extract. These are the
        starting indices of the patches to extract (upper left corner).
        """
        self.patch_indices = [
            (
                random.randint(0, self.img_shape[0] - self.patch_shape[0]),
                random.randint(0, self.img_shape[1] - self.patch_shape[1]),
            )
            for _ in range(self.patch_num)
        ]
        return

    def get_patch_indices(self) -> List[Tuple[int, int]]:
        """
        Get the current list of patch starting indices.

        These are the upper-left coordinates of each extracted patch
        from the full image.

        Returns
        -------
        List[Tuple[int, int]]
            A list of (row, column) tuples representing patch starting positions.
        """
        return self.patch_indices

    def apply(
        self,
        input: Tensor,
        additional_input: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Applies the patching operation by extracting patches specified by
        ``self.patch_indices`` from the ``input`` Tensor. Extracted patches are
        batched along the first dimension of the output. The layout of the
        output assumes that for any patch index ``i``, ``out[B * i: B * (i + 1)]``
        corresponds to the *same patch* extracted from each batch element of
        ``input``.

        Parameters
        ----------
        input : Tensor
            The input tensor representing the full image with shape :math:`(B, C, H, W)`.
        additional_input : Optional[Tensor], optional
            Its shape should be :math:`(B, C_{add}, H_{add}, W_{add})`.
            Must have same batch size as ``input``. Bilinear interpolation is
            used to interpolate ``additional_input`` onto a 2D grid of shape
            :math:`(H_p, W_p)`. It is then channel-wise concatenated to the
            extracted patches.
            *Note: ``additional_input`` is not patched or decomposed.*

        Returns
        -------
        Tensor
            A tensor of shape :math:`(P \times B, C [+ C_{add}], H_p, W_p)`.
            If ``additional_input`` is provided, it is channel-wise concatenated
            to the extracted patches.

        See Also
        --------
        :func:`physicsnemo.utils.patching.image_batching`
            The underlying function used to perform the patching operation.
        """
        B = input.shape[0]
        out = torch.zeros(
            B * self.patch_num,
            (
                input.shape[1]
                + (additional_input.shape[1] if additional_input is not None else 0)
            ),
            self.patch_shape[0],
            self.patch_shape[1],
            device=input.device,
        )
        out = out.to(
            memory_format=torch.channels_last
            if input.is_contiguous(memory_format=torch.channels_last)
            else torch.contiguous_format
        )
        if additional_input is not None:
            add_input_interp = torch.nn.functional.interpolate(
                input=additional_input, size=self.patch_shape, mode="bilinear"
            )

        for i, (py, px) in enumerate(self.patch_indices):
            if additional_input is not None:
                out[B * i : B * (i + 1),] = torch.cat(
                    (
                        input[
                            :,
                            :,
                            py : py + self.patch_shape[0],
                            px : px + self.patch_shape[1],
                        ],
                        add_input_interp,
                    ),
                    dim=1,
                )
            else:
                out[B * i : B * (i + 1),] = input[
                    :,
                    :,
                    py : py + self.patch_shape[0],
                    px : px + self.patch_shape[1],
                ]
        return out


class GridPatching2D(BasePatching2D):
    """
    Class for deterministically extracting patches from 2D images in a grid pattern.

    This class provides utilities to extract patches from images in a
    deterministic manner, with configurable overlap and boundary pixels.
    The patches are extracted in a grid-like pattern covering the entire image.

    Parameters
    ----------
    img_shape : Tuple[int, int]
        The height and width of the full input images :math:`(H, W)`.
    patch_shape : Tuple[int, int]
        The height and width of the patches to extract :math:`(H_p, W_p)`.
    overlap_pix : int, optional, default=0
        Number of pixels to overlap between adjacent patches.
    boundary_pix : int, optional, default=0
        Number of pixels to crop as boundary from each patch.

    Attributes
    ----------
    patch_num : int
        Total number of patches :math:`P` that will be extracted from the image,
        calculated as :math:`P = P_x * P_y`.

    See Also
    --------
    :class:`physicsnemo.utils.patching.BasePatching2D`
        The base class providing the patching interface.
    :class:`physicsnemo.utils.patching.RandomPatching2D`
        Alternative patching strategy using random patch locations.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        patch_shape: Tuple[int, int],
        overlap_pix: int = 0,
        boundary_pix: int = 0,
    ):
        super().__init__(img_shape, patch_shape)
        self.overlap_pix = overlap_pix
        self.boundary_pix = boundary_pix
        patch_num_x = math.ceil(
            img_shape[1] / (patch_shape[1] - overlap_pix - boundary_pix)
        )
        patch_num_y = math.ceil(
            img_shape[0] / (patch_shape[0] - overlap_pix - boundary_pix)
        )
        self.patch_num = patch_num_x * patch_num_y

    def apply(
        self,
        input: Tensor,
        additional_input: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Apply deterministic patching to the input tensor.

        Splits the input tensor into patches in a grid-like pattern. Can
        optionally concatenate additional interpolated data to each patch.
        Extracted patches are batched along the first dimension of the output.
        The layout of the output assumes that for any patch index ``i``,
        ``out[B * i: B * (i + 1)]`` corresponds to the *same patch* extracted
        from each batch element of ``input``.

        Parameters
        ----------
        input : Tensor
            Batch of full input images of shape :math:`(B, C, H, W)`.
        additional_input : Optional[Tensor], optional, default=None
            Additional data to concatenate to each patch. Shape must be
            :math:`(B, C_{add}, H_{add}, W_{add})`. Will be interpolated
            to match patch dimensions :math:`(H_p, W_p)`
            *Note: ``additional_input`` is not patched or decomposed.*

        Returns
        -------
        Tensor
            Tensor containing patches with shape :math:`(P \times B, C [+ C_{add}], H_p, W_p)`.
            If ``additional_input`` is provided, it is channel-wise concatenated
            to the extracted patches.

        See Also
        --------
        :func:`physicsnemo.utils.patching.image_batching`
            The underlying function used to perform the patching operation.
        """
        if additional_input is not None:
            add_input_interp = torch.nn.functional.interpolate(
                input=additional_input, size=self.patch_shape, mode="bilinear"
            )
        else:
            add_input_interp = None
        out = image_batching(
            input=input,
            patch_shape_y=self.patch_shape[0],
            patch_shape_x=self.patch_shape[1],
            overlap_pix=self.overlap_pix,
            boundary_pix=self.boundary_pix,
            input_interp=add_input_interp,
        )
        return out

    def fuse(self, input: Tensor, batch_size: int) -> Tensor:
        r"""
        Fuse patches back into a complete image.

        Reconstructs the original image by stitching together patches,
        accounting for overlapping regions and boundary pixels. In overlapping
        regions, values are averaged.

        Parameters
        ----------
        input : Tensor
            Input tensor containing patches with shape :math:`(P \times B, C, H_p, W_p)`.
            *Note: the patch layout along the batch dimension should be the same
            as the one returned by the method
            :meth:`~physicsnemo.utils.patching.GridPatching2D.apply`.*
        batch_size : int
            The original batch size :math:`B` before patching.

        Returns
        -------
        Tensor
            Reconstructed image tensor with shape :math:`(B, C, H, W)`.

        See Also
        --------
        :func:`physicsnemo.utils.patching.image_fuse`
            The underlying function used to perform the fusion operation.
        """
        out = image_fuse(
            input=input,
            img_shape_y=self.img_shape[0],
            img_shape_x=self.img_shape[1],
            batch_size=batch_size,
            overlap_pix=self.overlap_pix,
            boundary_pix=self.boundary_pix,
        )
        return out


def image_batching(
    input: Tensor,
    patch_shape_y: int,
    patch_shape_x: int,
    overlap_pix: int,
    boundary_pix: int,
    input_interp: Optional[Tensor] = None,
) -> Tensor:
    r"""
    Splits a full image into a batch of patched images.

    This function takes a full image and splits it into patches, adding padding
    where necessary. It can also concatenate additional interpolated data to
    each patch if provided.

    Parameters
    ----------
    input : Tensor
        The input tensor representing a batch of full image with shape :math:`(B, C, H, W)`.
    patch_shape_y : int
        The height :math:`H_p` of each image patch.
    patch_shape_x : int
        The width :math:`W_p` of each image patch.
    overlap_pix : int
        The number of overlapping pixels between adjacent patches.
    boundary_pix : int
        The number of pixels to crop as a boundary from each patch.
    input_interp : Optional[Tensor], optional
        Optional additional data to concatenate to each patch with shape
        :math:`(B, C_{add}, H_{add}, W_{add})`.
        *Note: ``additional_input`` is not patched or decomposed.*

    Returns
    -------
    Tensor
        A tensor containing the image patches, with shape :math:`(P \times B, C [+ C_{add}], H_p, W_p)`.
        If ``additional_input`` is provided, it is channel-wise concatenated
        to the extracted patches.
    """
    # Infer sizes from input image
    batch_size, _, img_shape_y, img_shape_x = input.shape

    # Safety check: make sure patch_shapes are large enough to accommodate
    # overlaps and boundaries pixels
    if (patch_shape_x - overlap_pix - boundary_pix) < 1:
        raise ValueError(
            f"patch_shape_x must verify patch_shape_x ({patch_shape_x}) >= "
            f"1 + overlap_pix ({overlap_pix}) + boundary_pix ({boundary_pix})"
        )
    if (patch_shape_y - overlap_pix - boundary_pix) < 1:
        raise ValueError(
            f"patch_shape_y must verify patch_shape_y ({patch_shape_y}) >= "
            f"1 + overlap_pix ({overlap_pix}) + boundary_pix ({boundary_pix})"
        )
    # Safety check: validate input_interp dimensions if provided
    if input_interp is not None:
        if input_interp.shape[0] != batch_size:
            raise ValueError(
                f"input_interp batch size ({input_interp.shape[0]}) must match "
                f"input batch size ({batch_size})"
            )
        if (input_interp.shape[2] != patch_shape_y) or (
            input_interp.shape[3] != patch_shape_x
        ):
            raise ValueError(
                f"input_interp patch shape ({input_interp.shape[2]}, {input_interp.shape[3]}) "
                f"must match specified patch shape ({patch_shape_y}, {patch_shape_x})"
            )

    # Safety check: make sure patch_shape is large enough in comparison to
    # overlap_pix and boundary_pix. Otherwise, number of patches extracted by
    # unfold differs from the expected number of patches.
    if patch_shape_x <= overlap_pix + 2 * boundary_pix:
        raise ValueError(
            f"patch_shape_x ({patch_shape_x}) must verify "
            f"patch_shape_x ({patch_shape_x}) > "
            f"overlap_pix ({overlap_pix}) + 2 * boundary_pix ({boundary_pix})"
        )
    if patch_shape_y <= overlap_pix + 2 * boundary_pix:
        raise ValueError(
            f"patch_shape_y ({patch_shape_y}) must verify "
            f"patch_shape_y ({patch_shape_y}) > "
            f"overlap_pix ({overlap_pix}) + 2 * boundary_pix ({boundary_pix})"
        )

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
    image_padding = torch.nn.ReflectionPad2d(
        (boundary_pix, pad_x_right, boundary_pix, pad_y_right)
    ).to(
        input.device
    )  # (padding_left,padding_right,padding_top,padding_bottom)
    input_padded = image_padding(input)
    patch_num = patch_num_x * patch_num_y

    # Cast to float for unfold
    if input.dtype == torch.int32:
        input_padded = input_padded.view(torch.float32)
    elif input.dtype == torch.int64:
        input_padded = input_padded.view(torch.float64)

    x_unfold = torch.nn.functional.unfold(
        input=input_padded,
        kernel_size=(patch_shape_y, patch_shape_x),
        stride=(
            patch_shape_y - overlap_pix - boundary_pix,
            patch_shape_x - overlap_pix - boundary_pix,
        ),
    )

    # Cast back to original dtype
    if input.dtype in [torch.int32, torch.int64]:
        x_unfold = x_unfold.view(input.dtype)

    x_unfold = rearrange(
        x_unfold,
        "b (c p_h p_w) (nb_p_h nb_p_w) -> (nb_p_w nb_p_h b) c p_h p_w",
        p_h=patch_shape_y,
        p_w=patch_shape_x,
        nb_p_h=patch_num_y,
        nb_p_w=patch_num_x,
    )
    if input_interp is not None:
        input_interp_repeated = input_interp.repeat(patch_num, 1, 1, 1)
        return torch.cat((x_unfold, input_interp_repeated), dim=1)
    else:
        return x_unfold


def image_fuse(
    input: Tensor,
    img_shape_y: int,
    img_shape_x: int,
    batch_size: int,
    overlap_pix: int,
    boundary_pix: int,
) -> Tensor:
    r"""
    Reconstructs a full image from a batch of patched images. Reverts the patching
    operation performed by :func:`~physicsnemo.utils.patching.image_batching`.

    It assumes that the patches are extracted in a grid-like pattern, and that
    their layout along the batch dimension is the same as the one returned by
    :func:`~physicsnemo.utils.patching.image_batching`.

    This function takes a batch of image patches and reconstructs the full
    image by stitching the patches together. The function accounts for
    overlapping and boundary pixels, ensuring that overlapping areas are
    averaged.
    *Note: a simple unweighted average between overlapping patches is used to
    fuse the patches.*

    Parameters
    ----------
    input : Tensor
        The input tensor containing the image patches with shape :math:`(P \times B, C, H_p, W_p)`.
    img_shape_y : int
        The height :math:`H` of the original full image.
    img_shape_x : int
        The width :math:`W` of the original full image.
    batch_size : int
        The original batch size :math:`B` before patching.
    overlap_pix : int
        The number of overlapping pixels between adjacent patches.
    boundary_pix : int
        The number of pixels to crop as a boundary from each patch.

    Returns
    -------
    Tensor
        The reconstructed full image tensor with shape :math:`(B, C, H, W)`.
    """

    # Infer sizes from input image shape
    patch_shape_y, patch_shape_x = input.shape[2], input.shape[3]

    # Calculate the number of patches in each dimension
    patch_num_x = math.ceil(img_shape_x / (patch_shape_x - overlap_pix - boundary_pix))
    patch_num_y = math.ceil(img_shape_y / (patch_shape_y - overlap_pix - boundary_pix))

    # Calculate the shape of the input after padding
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
    # Calculate the shape of the padding to add to input
    pad_x_right = padded_shape_x - img_shape_x - boundary_pix
    pad_y_right = padded_shape_y - img_shape_y - boundary_pix
    pad = (boundary_pix, pad_x_right, boundary_pix, pad_y_right)

    # Count local overlaps between patches
    input_ones = torch.ones(
        (batch_size, input.shape[1], padded_shape_y, padded_shape_x),
        device=input.device,
    )
    overlap_count = torch.nn.functional.unfold(
        input=input_ones,
        kernel_size=(patch_shape_y, patch_shape_x),
        stride=(
            patch_shape_y - overlap_pix - boundary_pix,
            patch_shape_x - overlap_pix - boundary_pix,
        ),
    )
    overlap_count = torch.nn.functional.fold(
        input=overlap_count,
        output_size=(padded_shape_y, padded_shape_x),
        kernel_size=(patch_shape_y, patch_shape_x),
        stride=(
            patch_shape_y - overlap_pix - boundary_pix,
            patch_shape_x - overlap_pix - boundary_pix,
        ),
    )

    # Reshape input to make it 3D to apply fold
    x = rearrange(
        input,
        "(nb_p_w nb_p_h b) c p_h p_w -> b (c p_h p_w) (nb_p_h nb_p_w)",
        p_h=patch_shape_y,
        p_w=patch_shape_x,
        nb_p_h=patch_num_y,
        nb_p_w=patch_num_x,
    )

    # Cast to float for fold
    if input.dtype == torch.int32:
        x = x.view(torch.float32)
    elif input.dtype == torch.int64:
        x = x.view(torch.float64)

    # Stitch patches together (by summing over overlapping patches)
    x_folded = torch.nn.functional.fold(
        input=x,
        output_size=(padded_shape_y, padded_shape_x),
        kernel_size=(patch_shape_y, patch_shape_x),
        stride=(
            patch_shape_y - overlap_pix - boundary_pix,
            patch_shape_x - overlap_pix - boundary_pix,
        ),
    )

    # Cast back to original dtype
    if input.dtype in [torch.int32, torch.int64]:
        x_folded = x_folded.view(input.dtype)

    # Remove padding
    x_no_padding = x_folded[
        ..., pad[2] : pad[2] + img_shape_y, pad[0] : pad[0] + img_shape_x
    ]
    overlap_count_no_padding = overlap_count[
        ..., pad[2] : pad[2] + img_shape_y, pad[0] : pad[0] + img_shape_x
    ]

    # Normalize by overlap count
    return x_no_padding / overlap_count_no_padding
