"""
SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-FileCopyrightText: All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@Modified by: clement etienam
"""

# from __future__ import print_function
import numpy as np
from math import ceil


def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape


def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale


def triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x >= -1), x < 0)
    greaterthanzero = np.logical_and((x <= 1), x >= 0)
    f = np.multiply((x + 1), lessthanzero) + np.multiply((1 - x), greaterthanzero)
    return f


def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5 * absx3 - 2.5 * absx2 + 1, absx <= 1) + np.multiply(
        -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2, (1 < absx) & (absx <= 2)
    )
    return f


def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:

        def h(x):
            return scale * kernel(scale * x)

        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length + 1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
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
    else:
        return outimg


def imresizevec(inimg, weights, indices, dim):
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
    else:
        return outimg


def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out


def imresize(
    image_array, scalar_scale=None, method="bicubic", output_shape=None, mode="vec"
):
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
        output_size = deriveSizeFromScale(image_array.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(image_array.shape, output_shape)
        output_size = list(output_shape)
    else:
        print("Error: scalar_scale OR output_shape should be defined!")
        return
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
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B


def convertDouble2Byte(image_array):
    B = np.clip(image_array, 0.0, 1.0)
    B = 255 * B
    return np.around(B).astype(np.uint8)
