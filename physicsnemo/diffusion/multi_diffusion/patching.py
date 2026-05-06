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

"""Utilities for multi-diffusion (patching and fusion)."""

import math
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor


class BasePatching2D(torch.nn.Module, ABC):
    r"""Abstract base class for 2D image patching operations.

    Provides the common interface and validation logic for patching
    strategies that decompose a batch of 2D images into smaller spatial
    tiles (patches). Concrete subclasses must implement
    :meth:`forward` to define the actual patching logic.

    Parameters
    ----------
    img_shape : Tuple[int, int]
        Height and width of the full input images :math:`(H, W)`.
    patch_shape : Tuple[int, int]
        Height and width of the patches to extract :math:`(H_p, W_p)`.
    """

    def __init__(
        self, img_shape: Tuple[int, int], patch_shape: Tuple[int, int]
    ) -> None:
        super().__init__()
        if len(img_shape) != 2:
            raise ValueError(f"img_shape must be 2D, got {len(img_shape)}D")
        if len(patch_shape) != 2:
            raise ValueError(f"patch_shape must be 2D, got {len(patch_shape)}D")

        if any(p > i for p, i in zip(patch_shape, img_shape)):
            warnings.warn(
                f"Patch shape {patch_shape} is larger than "
                f"image shape {img_shape}. "
                f"Patches will be cropped to fit within the image."
            )
        self.img_shape = img_shape
        self.patch_shape = tuple(min(p, i) for p, i in zip(patch_shape, img_shape))

    def forward(
        self, input: Float[Tensor, "B C H W"], **kwargs
    ) -> Float[Tensor, "P_times_B C Hp Wp"]:
        r"""Forward pass. Delegates to :meth:`apply` by default."""
        return self.apply(input, **kwargs)

    @abstractmethod
    def apply(
        self,
        input: Float[Tensor, "B C H W"],
        *args,
        **kwargs,
    ) -> Float[Tensor, "P_times_B C Hp Wp"]:
        r"""Apply the patching operation to a batch of full images.

        Subclasses **must** override this method.

        Parameters
        ----------
        input : Tensor
            Batch of full images of shape :math:`(B, C, H, W)`.
        *args : tuple
            Additional positional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        Tensor
            Patched tensor of shape :math:`(P \times B, C, H_p, W_p)`.
        """
        pass

    def fuse(
        self, input: Float[Tensor, "P_times_B C Hp Wp"], **kwargs
    ) -> Float[Tensor, "B C H W"]:
        r"""Fuse patches back into a complete image.

        Parameters
        ----------
        input : Tensor
            Patched tensor. Shape depends on the subclass.
        **kwargs : dict
            Additional keyword arguments specific to the subclass.

        Returns
        -------
        Tensor
            Fused image tensor.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("'fuse' method must be implemented in subclasses.")

    def global_index(
        self, batch_size: int = 1, device: Union[torch.device, str] = "cpu"
    ) -> Int[Tensor, "P 2 Hp Wp"]:
        r"""Return the global :math:`(y, x)` grid coordinates for each patch.

        Returns a **new tensor** (clone) each time, so the caller owns the
        result and it will not be mutated by subsequent calls to
        :meth:`~RandomPatching2D.reset_patch_indices`. For zero-copy access
        to the underlying buffer, use ``self._global_index`` directly.

        Parameters
        ----------
        batch_size : int, default=1
            Kept for backward compatibility. Ignored.
        device : Union[torch.device, str], default="cpu"
            Kept for backward compatibility. The buffer follows the module
            device (use ``.to(device)`` to move the module).

        Returns
        -------
        Tensor
            Integer tensor of shape :math:`(P, 2, H_p, W_p)`.
            Channel 0 holds y-coordinates, channel 1 holds x-coordinates.
        """
        if hasattr(self, "_global_index") and self._global_index is not None:
            return self._global_index.clone()
        return self._compute_global_index()

    def _compute_global_index(self) -> Int[Tensor, "P 2 Hp Wp"]:
        r"""Compute the global-index tensor from current patch positions.

        The ``global_index`` tensor is created and computed on the same device
        as ``patch_indices`` (if the buffer exists) to avoid cross-device
        indexing errors.

        Returns
        -------
        Tensor
            Integer tensor of shape :math:`(P, 2, H_p, W_p)`.
        """
        device = None
        if hasattr(self, "patch_indices") and isinstance(self.patch_indices, Tensor):
            device = self.patch_indices.device
        Ny = torch.arange(self.img_shape[0], device=device).int()
        Nx = torch.arange(self.img_shape[1], device=device).int()
        grid = torch.stack(torch.meshgrid(Ny, Nx, indexing="ij"), dim=0).unsqueeze(
            0
        )  # (1, 2, H, W)
        return self(grid).long()


class RandomPatching2D(BasePatching2D):
    r"""Randomly extract patches from 2D images.

    Maintains a set of :math:`P` random upper-left corner positions and
    extracts the corresponding patches from every batch element.  Positions
    are drawn at construction time and can be refreshed by calling
    :meth:`reset_patch_indices`.

    Parameters
    ----------
    img_shape : Tuple[int, int]
        Height and width :math:`(H, W)` of the full input images.
    patch_shape : Tuple[int, int]
        Height and width :math:`(H_p, W_p)` of the patches to extract.
    patch_num : int
        Number of patches :math:`P` to extract per image.

    Attributes
    ----------
    patch_indices : Tensor
        Buffer of shape :math:`(P, 2)` with the :math:`(y, x)` upper-left
        corner of each patch.

    See Also
    --------
    :class:`~physicsnemo.diffusion.multi_diffusion.GridPatching2D` :
        Deterministic grid-based patching strategy.

    Examples
    --------
    Extract patches, re-draw random positions, then extract again:

    >>> import torch
    >>> from physicsnemo.diffusion.multi_diffusion import RandomPatching2D
    >>> rp = RandomPatching2D(img_shape=(16, 16), patch_shape=(8, 8), patch_num=6)
    >>> x = torch.randn(2, 3, 16, 16)
    >>> # First extraction: 6 patches per sample, batch of 2
    >>> rp.apply(x).shape
    torch.Size([12, 3, 8, 8])
    >>> # Re-draw random positions (e.g. between training steps)
    >>> rp.reset_patch_indices()
    >>> rp.apply(x).shape
    torch.Size([12, 3, 8, 8])

    Retrieve the global (y, x) coordinates for each patch:

    >>> gi = rp.global_index()
    >>> gi.shape  # (P, 2, Hp, Wp) — channel 0 = y, channel 1 = x
    torch.Size([6, 2, 8, 8])
    """

    def __init__(
        self, img_shape: Tuple[int, int], patch_shape: Tuple[int, int], patch_num: int
    ) -> None:
        super().__init__(img_shape, patch_shape)
        self._patch_num = patch_num
        self.reset_patch_indices()

    @property
    def patch_num(self) -> int:
        r"""Number of patches :math:`P` to extract."""
        return self._patch_num

    def set_patch_num(self, value: int) -> None:
        r"""Set the number of patches and re-draw positions.

        Parameters
        ----------
        value : int
            New number of patches :math:`P`.
        """
        self._patch_num = value
        self.reset_patch_indices()

    def reset_patch_indices(
        self,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        r"""Re-draw random upper-left corner positions for all patches.

        The cached ``_global_index`` buffer is invalidated and will be
        lazily recomputed on the next call to :meth:`global_index`.

        Parameters
        ----------
        generator : torch.Generator, optional
            Pseudo-random number generator for reproducible sampling.
        """
        has_buffer = hasattr(self, "patch_indices") and isinstance(
            self.patch_indices, Tensor
        )
        device = self.patch_indices.device if has_buffer else None

        max_y = self.img_shape[0] - self.patch_shape[0]
        max_x = self.img_shape[1] - self.patch_shape[1]

        py = torch.randint(
            0,
            max_y + 1,
            (self.patch_num,),
            dtype=torch.long,
            device=device,
            generator=generator,
        )
        px = torch.randint(
            0,
            max_x + 1,
            (self.patch_num,),
            dtype=torch.long,
            device=device,
            generator=generator,
        )
        new_indices = torch.stack([py, px], dim=1)

        if has_buffer and new_indices.shape == self.patch_indices.shape:
            self.patch_indices.copy_(new_indices)
        else:
            self.register_buffer("patch_indices", new_indices, persistent=False)

        self._global_index_needs_update = True

    def global_index(
        self, batch_size: int = 1, device: Union[torch.device, str] = "cpu"
    ) -> Int[Tensor, "P 2 Hp Wp"]:
        r"""Return global :math:`(y, x)` grid coordinates for each patch.
        Recomputes lazily if patch positions have changed since the last call.

        Parameters
        ----------
        batch_size : int, default=1
            Kept for backward compatibility. Ignored.
        device : Union[torch.device, str], default="cpu"
            Kept for backward compatibility. The buffer follows the module
            device (use ``.to(device)`` to move the module).

        Returns
        -------
        Tensor
            Integer tensor of shape :math:`(P, 2, H_p, W_p)`.
        """
        if getattr(self, "_global_index_needs_update", True):
            new_global_index = self._compute_global_index()
            if (
                hasattr(self, "_global_index")
                and isinstance(self._global_index, Tensor)
                and new_global_index.shape == self._global_index.shape
            ):
                self._global_index.copy_(new_global_index)
            else:
                self.register_buffer(
                    "_global_index", new_global_index, persistent=False
                )
            self._global_index_needs_update = False
        return self._global_index.clone()

    def forward(
        self,
        input: Float[Tensor, "B C H W"],
        additional_input: Optional[Float[Tensor, "B C_add H_add W_add"]] = None,
    ) -> Float[Tensor, "P_times_B C_out Hp Wp"]:
        r"""Extract random patches from the input tensor."""
        B, C, H, W = input.shape
        Hp, Wp = self.patch_shape
        P = self.patch_num
        K = Hp * Wp

        patch_indices = self.patch_indices.to(input.device)
        py = patch_indices[:, 0]  # (P,)
        px = patch_indices[:, 1]  # (P,)

        dy = torch.arange(Hp, device=input.device)
        dx = torch.arange(Wp, device=input.device)
        base = (py * W + px).reshape(P, 1, 1)  # (P, 1, 1)
        rel = (dy[:, None] * W + dx[None, :]).reshape(1, 1, K)  # (1, 1, K)
        idx = (base + rel).expand(P, B, K)  # (P, B, Hp*Wp)

        x_flat = input.reshape(B, C, H * W)  # (B, C, HW)
        gathered = torch.gather(
            x_flat.unsqueeze(0).expand(P, B, C, H * W),
            dim=3,
            index=idx.unsqueeze(2).expand(P, B, C, K),
        )  # (P, B, C, Hp*Wp)

        out = gathered.reshape(P * B, C, Hp, Wp)

        if input.is_contiguous(memory_format=torch.channels_last):
            out = out.to(memory_format=torch.channels_last)

        if additional_input is not None:
            add_input_interp = torch.nn.functional.interpolate(
                input=additional_input, size=self.patch_shape, mode="bilinear"
            )  # (B, C_add, Hp, Wp)
            out = torch.cat(
                (out, add_input_interp.repeat(P, 1, 1, 1)), dim=1
            )  # (P*B, C+C_add, Hp, Wp)

        return out

    def apply(
        self,
        input: Float[Tensor, "B C H W"],
        additional_input: Optional[Float[Tensor, "B C_add H_add W_add"]] = None,
    ) -> Float[Tensor, "P_times_B C_out Hp Wp"]:
        r"""Extract random patches.

        Parameters
        ----------
        input : Tensor
            Full images of shape :math:`(B, C, H, W)`.
        additional_input : Tensor, optional
            Interpolated to :math:`(H_p, W_p)` and concatenated
            channel-wise to each patch. Should have shape :math:`(B, C_add,
            H_add, W_add)`.

        Returns
        -------
        Tensor
            Shape :math:`(P \times B, C [+ C_{add}], H_p, W_p)`.

        """
        if isinstance(input, Tensor):
            return self(input, additional_input=additional_input)
        return super().apply(input)


class GridPatching2D(BasePatching2D):
    r"""Deterministically extract patches from 2D images in a grid pattern.

    Tiles the image with a regular grid of :math:`P = P_y \times P_x`
    patches, with configurable overlap and boundary padding. Supports
    reconstructing the full image from patches via :meth:`fuse`.

    Parameters
    ----------
    img_shape : Tuple[int, int]
        Height and width of the full input images :math:`(H, W)`.
    patch_shape : Tuple[int, int]
        Height and width of the patches to extract :math:`(H_p, W_p)`.
    overlap_pix : int, optional, default=0
        Number of overlapping pixels between adjacent patches.
    boundary_pix : int, optional, default=0
        Number of boundary pixels to pad on each side.

    Attributes
    ----------
    patch_num : int
        Total number of patches :math:`P = P_y \times P_x`.

    See Also
    --------
    :class:`~physicsnemo.diffusion.multi_diffusion.RandomPatching2D` :
        Random patching strategy.

    Examples
    --------
    Patch an image and fuse it back (roundtrip):

    >>> import torch
    >>> from physicsnemo.diffusion.multi_diffusion import GridPatching2D
    >>> gp = GridPatching2D(img_shape=(16, 16), patch_shape=(8, 8))
    >>> x = torch.randn(2, 3, 16, 16)
    >>> patches = gp.apply(x)
    >>> patches.shape  # (P*B, C, Hp, Wp)
    torch.Size([8, 3, 8, 8])
    >>> # Fuse is the inverse of apply
    >>> reconstructed = gp.fuse(patches, batch_size=2)
    >>> reconstructed.shape
    torch.Size([2, 3, 16, 16])
    >>> torch.allclose(x, reconstructed)
    True

    Retrieve the global (y, x) coordinates for each patch:

    >>> gi = gp.global_index()
    >>> gi.shape  # (P, 2, Hp, Wp)
    torch.Size([4, 2, 8, 8])
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
        self.register_buffer(
            "_overlap_count",
            self.get_overlap_count(
                self.patch_shape, self.img_shape, self.overlap_pix, self.boundary_pix
            ),
            persistent=False,
        )
        self.register_buffer(
            "_global_index", self._compute_global_index(), persistent=False
        )

    def forward(
        self,
        input: Float[Tensor, "B C H W"],
        additional_input: Optional[Float[Tensor, "B C_add H_add W_add"]] = None,
    ) -> Float[Tensor, "P_times_B C_out Hp Wp"]:
        r"""Extract grid patches from the input tensor."""
        if additional_input is not None:
            add_input_interp = torch.nn.functional.interpolate(
                input=additional_input, size=self.patch_shape, mode="bilinear"
            )
        else:
            add_input_interp = None
        return image_batching(
            input=input,
            patch_shape_y=self.patch_shape[0],
            patch_shape_x=self.patch_shape[1],
            overlap_pix=self.overlap_pix,
            boundary_pix=self.boundary_pix,
            input_interp=add_input_interp,
        )

    def apply(
        self,
        input: Float[Tensor, "B C H W"],
        additional_input: Optional[Float[Tensor, "B C_add H_add W_add"]] = None,
    ) -> Float[Tensor, "P_times_B C_out Hp Wp"]:
        r"""Apply deterministic grid patching.

        Splits the input tensor into patches in a grid-like pattern.
        Extracted patches are batched along the first dimension.  For any
        patch index ``i``, ``out[B * i : B * (i + 1)]`` corresponds to the
        *same patch* extracted from every batch element.

        Parameters
        ----------
        input : Tensor
            Full images of shape :math:`(B, C, H, W)`.
        additional_input : Tensor, optional, default=None
            Additional data of shape :math:`(B, C_{add}, H', W')`.
            Interpolated to :math:`(H_p, W_p)` and channel-wise
            concatenated to each patch. Not decomposed.

        Returns
        -------
        Tensor
            Shape :math:`(P \times B, C [+ C_{add}], H_p, W_p)`.

        See Also
        --------
        :func:`~physicsnemo.diffusion.multi_diffusion.image_batching` :
            Low-level function used internally.
        """
        if isinstance(input, Tensor):
            return self(input, additional_input=additional_input)
        return super().apply(input)

    def fuse(
        self,
        input: Float[Tensor, "P_times_B C Hp Wp"],
        batch_size: int,
    ) -> Float[Tensor, "B C H W"]:
        r"""Fuse patches back into a complete image.

        Reconstructs the original image by stitching together patches.
        Overlapping regions are averaged with a uniform weight.

        Parameters
        ----------
        input : Tensor
            Patches of shape :math:`(P \times B, C, H_p, W_p)`, with the
            same batch layout as returned by :meth:`apply`.
        batch_size : int
            Original batch size :math:`B` before patching.

        Returns
        -------
        Tensor
            Reconstructed image of shape :math:`(B, C, H, W)`.

        See Also
        --------
        :func:`~physicsnemo.diffusion.multi_diffusion.image_fuse` :
            Low-level function used internally.
        """
        return image_fuse(
            input=input,
            img_shape_y=self.img_shape[0],
            img_shape_x=self.img_shape[1],
            batch_size=batch_size,
            overlap_pix=self.overlap_pix,
            boundary_pix=self.boundary_pix,
            overlap_count=self._overlap_count,
        )

    @staticmethod
    def get_overlap_count(
        patch_shape: tuple[int, int],
        img_shape: tuple[int, int],
        overlap_pix: int,
        boundary_pix: int,
    ) -> Float[Tensor, "1 1 Hpad Wpad"]:
        r"""Compute per-pixel overlap count for patch reconstruction.

        Calculates how many patches cover each pixel in the padded image.
        Used to normalise the reconstructed image after folding.

        Parameters
        ----------
        patch_shape : tuple[int, int]
            Patch dimensions :math:`(H_p, W_p)`.
        img_shape : tuple[int, int]
            Full image dimensions :math:`(H, W)`.
        overlap_pix : int
            Overlap between adjacent patches in pixels.
        boundary_pix : int
            Boundary padding in pixels.

        Returns
        -------
        Tensor
            Overlap count of shape :math:`(1, 1, H_{pad}, W_{pad})`.
        """
        patch_shape_y, patch_shape_x = patch_shape
        img_shape_y, img_shape_x = img_shape

        patch_num_x = math.ceil(
            img_shape_x / (patch_shape_x - overlap_pix - boundary_pix)
        )
        patch_num_y = math.ceil(
            img_shape_y / (patch_shape_y - overlap_pix - boundary_pix)
        )

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

        stride = (
            patch_shape_y - overlap_pix - boundary_pix,
            patch_shape_x - overlap_pix - boundary_pix,
        )
        kernel = (patch_shape_y, patch_shape_x)
        ones = torch.ones(1, 1, padded_shape_y, padded_shape_x)
        overlap_count = torch.nn.functional.fold(
            input=torch.nn.functional.unfold(
                input=ones, kernel_size=kernel, stride=stride
            ),
            output_size=(padded_shape_y, padded_shape_x),
            kernel_size=kernel,
            stride=stride,
        )
        return overlap_count


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------


def image_batching(
    input: Float[Tensor, "B C H W"],
    patch_shape_y: int,
    patch_shape_x: int,
    overlap_pix: int,
    boundary_pix: int,
    input_interp: Optional[Float[Tensor, "B C_add Hp Wp"]] = None,
) -> Float[Tensor, "P_times_B C_out Hp Wp"]:
    r"""Split a batch of images into a batch of patches.

    Adds reflection padding where necessary and extracts patches in a
    regular grid using ``torch.nn.functional.unfold``.

    Parameters
    ----------
    input : Tensor
        Batch of full images of shape :math:`(B, C, H, W)`.
    patch_shape_y : int
        Patch height :math:`H_p`.
    patch_shape_x : int
        Patch width :math:`W_p`.
    overlap_pix : int
        Overlap between adjacent patches in pixels.
    boundary_pix : int
        Boundary padding in pixels.
    input_interp : Tensor, optional
        Pre-interpolated additional data of shape
        :math:`(B, C_{add}, H_p, W_p)`.  Channel-wise concatenated to
        every extracted patch (not decomposed).

    Returns
    -------
    Tensor
        Patches of shape :math:`(P \times B, C [+ C_{add}], H_p, W_p)`.
    """
    batch_size, _, img_shape_y, img_shape_x = input.shape

    # Validate patch / overlap / boundary compatibility
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

    # Grid layout
    stride_x = patch_shape_x - overlap_pix - boundary_pix
    stride_y = patch_shape_y - overlap_pix - boundary_pix
    patch_num_x = math.ceil(img_shape_x / stride_x)
    patch_num_y = math.ceil(img_shape_y / stride_y)
    padded_shape_x = stride_x * (patch_num_x - 1) + patch_shape_x + boundary_pix
    padded_shape_y = stride_y * (patch_num_y - 1) + patch_shape_y + boundary_pix
    patch_num = patch_num_x * patch_num_y

    # Reflection-pad to fit the grid. Use the functional form (not
    # ``torch.nn.ReflectionPad2d(...)(input)``) to avoid instantiating a fresh
    # nn.Module on every call, which is much less friendly to ``torch.compile``
    # / AOT autograd tracing.
    pad_x_right = padded_shape_x - img_shape_x - boundary_pix
    pad_y_right = padded_shape_y - img_shape_y - boundary_pix
    input_padded = torch.nn.functional.pad(
        input,
        (boundary_pix, pad_x_right, boundary_pix, pad_y_right),
        mode="reflect",
    )

    # Integer dtypes are not supported by unfold — cast temporarily
    if input.dtype == torch.int32:
        input_padded = input_padded.view(torch.float32)
    elif input.dtype == torch.int64:
        input_padded = input_padded.view(torch.float64)

    # Extract patches via unfold
    x_unfold = torch.nn.functional.unfold(
        input=input_padded,
        kernel_size=(patch_shape_y, patch_shape_x),
        stride=(stride_y, stride_x),
    )

    if input.dtype in [torch.int32, torch.int64]:
        x_unfold = x_unfold.view(input.dtype)

    # Rearrange to patch-major batch layout: (P*B, C, Hp, Wp)
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
    return x_unfold


def image_fuse(
    input: Float[Tensor, "P_times_B C Hp Wp"],
    img_shape_y: int,
    img_shape_x: int,
    batch_size: int,
    overlap_pix: int,
    boundary_pix: int,
    overlap_count: Optional[Float[Tensor, "1 1 Hpad Wpad"]] = None,
) -> Float[Tensor, "B C H W"]:
    r"""Reconstruct a full image from a batch of grid patches.

    Reverts the operation performed by
    :func:`~physicsnemo.diffusion.multi_diffusion.image_batching`.
    Overlapping regions are averaged with a uniform weight.

    Parameters
    ----------
    input : Tensor
        Patches of shape :math:`(P \times B, C, H_p, W_p)`.
    img_shape_y : int
        Height :math:`H` of the original full image.
    img_shape_x : int
        Width :math:`W` of the original full image.
    batch_size : int
        Original batch size :math:`B`.
    overlap_pix : int
        Overlap between adjacent patches in pixels.
    boundary_pix : int
        Boundary padding in pixels.
    overlap_count : Tensor, optional
        Pre-computed overlap count of shape :math:`(1, 1, H_{pad}, W_{pad})`.
        Computed internally if not provided.

    Returns
    -------
    Tensor
        Reconstructed image of shape :math:`(B, C, H, W)`.
    """
    patch_shape_y, patch_shape_x = input.shape[2], input.shape[3]

    stride_x = patch_shape_x - overlap_pix - boundary_pix
    stride_y = patch_shape_y - overlap_pix - boundary_pix
    patch_num_x = math.ceil(img_shape_x / stride_x)
    patch_num_y = math.ceil(img_shape_y / stride_y)

    padded_shape_x = stride_x * (patch_num_x - 1) + patch_shape_x + boundary_pix
    padded_shape_y = stride_y * (patch_num_y - 1) + patch_shape_y + boundary_pix

    pad_x_right = padded_shape_x - img_shape_x - boundary_pix
    pad_y_right = padded_shape_y - img_shape_y - boundary_pix
    pad = (boundary_pix, pad_x_right, boundary_pix, pad_y_right)

    if overlap_count is None:
        overlap_count = GridPatching2D.get_overlap_count(
            (patch_shape_y, patch_shape_x),
            (img_shape_y, img_shape_x),
            overlap_pix,
            boundary_pix,
        )

    if overlap_count.device != input.device:
        overlap_count = overlap_count.to(input.device)

    # Rearrange patches back to fold-compatible layout
    x = rearrange(
        input,
        "(nb_p_w nb_p_h b) c p_h p_w -> b (c p_h p_w) (nb_p_h nb_p_w)",
        p_h=patch_shape_y,
        p_w=patch_shape_x,
        nb_p_h=patch_num_y,
        nb_p_w=patch_num_x,
    )

    # Integer dtypes are not supported by fold — cast temporarily
    if input.dtype == torch.int32:
        x = x.view(torch.float32)
    elif input.dtype == torch.int64:
        x = x.view(torch.float64)

    # Stitch patches by summing over overlapping regions
    x_folded = torch.nn.functional.fold(
        input=x,
        output_size=(padded_shape_y, padded_shape_x),
        kernel_size=(patch_shape_y, patch_shape_x),
        stride=(stride_y, stride_x),
    )

    if input.dtype in [torch.int32, torch.int64]:
        x_folded = x_folded.view(input.dtype)

    # Crop padding and normalise by overlap count
    x_no_padding = x_folded[
        ..., pad[2] : pad[2] + img_shape_y, pad[0] : pad[0] + img_shape_x
    ]
    overlap_count_no_padding = overlap_count[
        ..., pad[2] : pad[2] + img_shape_y, pad[0] : pad[0] + img_shape_x
    ]
    return x_no_padding / overlap_count_no_padding
