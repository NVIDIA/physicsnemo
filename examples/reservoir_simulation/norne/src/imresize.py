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

"""
@Modified by: clement etienam
"""

import numpy as np
from math import ceil


def derive_size_from_scale(img_shape, scale):
    """Compute output image dimensions from input shape and scale factors.

    Parameters
    ----------
    img_shape : tuple of int
        Spatial dimensions (height, width) of the input image.
    scale : list of float
        Scale factors for each spatial dimension.

    Returns
    -------
    list of int
        Output spatial dimensions after applying the scale factors.
    """
    return [ceil(scale[k] * img_shape[k]) for k in range(2)]


def derive_scale_from_size(img_shape_in, img_shape_out):
    """Compute per-dimension scale factors from input and output image sizes.

    Parameters
    ----------
    img_shape_in : tuple of int
        Spatial dimensions (height, width) of the source image.
    img_shape_out : tuple of int
        Desired spatial dimensions (height, width) of the output image.

    Returns
    -------
    list of float
        Scale factor for each spatial dimension (output / input).
    """
    return [1.0 * img_shape_out[k] / img_shape_in[k] for k in range(2)]


def triangle(x):
    """Evaluate the piecewise-linear (bilinear) interpolation kernel.

    Parameters
    ----------
    x : array-like of float
        Sample points at which to evaluate the kernel.

    Returns
    -------
    numpy.ndarray
        Kernel values at each sample point; zero outside [-1, 1].
    """
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x >= -1), x < 0)
    greaterthanzero = np.logical_and((x <= 1), x >= 0)
    return np.multiply((x + 1), lessthanzero) + np.multiply((1 - x), greaterthanzero)


def cubic(x):
    """Evaluate the piecewise-cubic (bicubic) interpolation kernel.

    Parameters
    ----------
    x : array-like of float
        Sample points at which to evaluate the kernel.

    Returns
    -------
    numpy.ndarray
        Kernel values at each sample point; zero outside [-2, 2].
    """
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    return np.multiply(1.5 * absx3 - 2.5 * absx2 + 1, absx <= 1) + np.multiply(
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2, (absx > 1) & (absx <= 2)
    )


def contributions(in_length, out_length, scale, kernel, k_width):
    """Compute interpolation weights and source indices along one image dimension.

    Parameters
    ----------
    in_length : int
        Number of pixels in the source dimension.
    out_length : int
        Number of pixels in the output dimension.
    scale : float
        Ratio of output size to input size along this dimension.
    kernel : callable
        Interpolation kernel function (e.g. ``cubic`` or ``triangle``).
    k_width : float
        Support width of the kernel in input-pixel units.

    Returns
    -------
    weights : numpy.ndarray
        Normalised kernel weights, shape (out_length, n_support).
    indices : numpy.ndarray
        Source pixel indices corresponding to each weight, shape (out_length, n_support).
    """
    if scale < 1:

        def h(x):
            """Scale-adjusted kernel: applies anti-aliasing down-scaling factor.

            Parameters
            ----------
            x : float or np.ndarray
                Distance(s) from the sampling position.

            Returns
            -------
            float or np.ndarray
                Kernel response scaled by ``scale``.
            """
            return scale * kernel(scale * x)

        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length + 1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = ceil(kernel_width) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1  # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1)  # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate(
        (np.arange(in_length), np.arange(in_length - 1, -1, step=-1))
    ).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def imresizemex(inimg, weights, indices, dim):
    """Resize an image along one dimension using a loop-based weighted sum.

    Parameters
    ----------
    inimg : numpy.ndarray
        Source image array of shape (H, W, C) or (H, W).
    weights : numpy.ndarray
        Interpolation weights from ``contributions``, shape (out_len, n_support).
    indices : numpy.ndarray
        Source pixel indices from ``contributions``, shape (out_len, n_support).
    dim : int
        Dimension along which to apply resampling (0 for height, 1 for width).

    Returns
    -------
    numpy.ndarray
        Resampled image with the target dimension replaced by out_len pixels.
    """
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w, i_img] = np.sum(
                    np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0
                )
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img, i_w] = np.sum(
                    np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0
                )
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    return outimg


def imresizevec(inimg, weights, indices, dim):
    """Resize an image along one dimension using a vectorized weighted sum.

    Parameters
    ----------
    inimg : numpy.ndarray
        Source image array of shape (H, W, C) or (H, W).
    weights : numpy.ndarray
        Interpolation weights from ``contributions``, shape (out_len, n_support).
    indices : numpy.ndarray
        Source pixel indices from ``contributions``, shape (out_len, n_support).
    dim : int
        Dimension along which to apply resampling (0 for height, 1 for width).

    Returns
    -------
    numpy.ndarray
        Resampled image with the target dimension replaced by out_len pixels.
    """
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg = np.sum(
            weights * ((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1
        )
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg = np.sum(
            weights * ((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2
        )
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    return outimg


def resize_along_dim(A, dim, weights, indices, mode="vec"):
    """Apply precomputed interpolation weights along one image dimension.

    Dispatches to the loop-based (``imresizemex``) or vectorized
    (``imresizevec``) backend depending on ``mode``.

    Parameters
    ----------
    A : numpy.ndarray
        Source image array of shape (H, W, C) or (H, W).
    dim : int
        Dimension along which to resample (0=height, 1=width).
    weights : numpy.ndarray
        Interpolation weights from ``contributions``.
    indices : numpy.ndarray
        Source pixel indices from ``contributions``.
    mode : str, optional
        ``'org'`` for loop-based, any other value for vectorized (default ``'vec'``).

    Returns
    -------
    numpy.ndarray
        Resampled image array.
    """
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out


def imresize(
    image_array, scalar_scale=None, method="bicubic", output_shape=None, mode="vec"
):
    """Resize an image using bicubic or bilinear interpolation.

    Exactly one of ``scalar_scale`` or ``output_shape`` must be provided.

    Parameters
    ----------
    image_array : numpy.ndarray
        Source image of shape (H, W) or (H, W, C).
    scalar_scale : float, optional
        Uniform scale factor applied to both spatial dimensions.
    method : str, optional
        Interpolation method: ``'bicubic'`` (default) or ``'bilinear'``.
    output_shape : tuple of int, optional
        Desired output spatial dimensions (height, width).
    mode : str, optional
        Resampling backend: ``'vec'`` for vectorized (default) or ``'org'`` for loop.

    Returns
    -------
    numpy.ndarray
        Resized image array, same dtype-compatible format as input.
    """
    if method == "bicubic":
        kernel = cubic
    elif method == "bilinear":
        kernel = triangle
    else:
        print("Error: Unidentified method supplied")

    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = derive_size_from_scale(image_array.shape, scale)
    elif output_shape is not None:
        scale = derive_scale_from_size(image_array.shape, output_shape)
        output_size = list(output_shape)
    else:
        print("Error: scalar_scale OR output_shape should be defined!")
        return None
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(
            image_array.shape[k], output_size[k], scale[k], kernel, kernel_width
        )
        weights.append(w)
        indices.append(ind)
    B = np.copy(image_array)
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resize_along_dim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B


def convert_double_to_byte(image_array):
    """Convert a floating-point image in [0, 1] to uint8 in [0, 255].

    Parameters
    ----------
    image_array : numpy.ndarray
        Float image array with values expected in the range [0, 1].

    Returns
    -------
    numpy.ndarray
        Uint8 image array with values clipped and scaled to [0, 255].
    """
    B = np.clip(image_array, 0.0, 1.0)
    B = 255 * B
    return np.around(B).astype(np.uint8)
