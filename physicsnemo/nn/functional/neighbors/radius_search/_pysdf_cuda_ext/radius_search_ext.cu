// SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// -----------------------------------------------------------------------------
// PyTorch CUDA extension for the "pysdf_cuda" radius-search backend.
//
// The QBVH build (``genericSpatialMedianBuilder``) and the point-range-query
// traversal (``QBVHPointRangeQuery``) are the software (no-OptiX) BVH from
// NVIDIA's pysdf / minigql (vendored under ``third_party/``). The device kernels
// below are adapted from pysdf ``src/prq.cu``; here they read PyTorch ``(N, 3)``
// fp32 tensors directly and select the ``max_points`` nearest neighbors per
// query in a single traversal pass (no host round-trips, no CSR, no in-kernel
// allocation). See ``third_party/NOTICE`` for provenance and licenses.
// -----------------------------------------------------------------------------

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "gequel/common.h"
#include "gequel/bvhLib/builder.h"
#include "gequel/bvhLib/pointRangeQuery.h"

using namespace gequel;
using namespace gequel::common;
using namespace gequel::bvhLib;

#define PNEMO_CUDA_CHECK(call)                                                 \
  do {                                                                        \
    cudaError_t _err = (call);                                                \
    TORCH_CHECK(_err == cudaSuccess, "pysdf_cuda CUDA error: ",              \
                cudaGetErrorString(_err));                                    \
  } while (0)

namespace {

constexpr int kBlockDim = 128;

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Build a degenerate (point) AABB for every reference point so the spatial
// median builder can construct the QBVH. Adapted from pysdf prq.cu.
__global__ void fill_build_prims_kernel(PrimState *prims, int num_points,
                                        const float3 *points) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= num_points) return;
  float3 p = points[gid];
  vec3f p3(p.x, p.y, p.z);
  prims[gid].bounds = box3f(p3);
}

// One thread per query: traverse the QBVH once and keep the ``max_points``
// closest in-radius reference points (bounded selection, closest-first not
// guaranteed in slot order; callers treat neighbor order as unspecified). The
// per-query working distances live in ``work`` (one row of ``max_points`` per
// query) so no compile-time array bound or device-side allocation is required.
__global__ void query_select_kernel(
    int num_points, const float3 *points, int num_queries,
    const float3 *queries, float radius, QBVH<float>::DevRep bvh, int max_points,
    bool return_dists, bool return_points, int *out_idx, int *out_count,
    float *out_dist, float3 *out_pts, float *work) {
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= num_queries) return;

  float3 QP = queries[q];
  vec3f QP3(QP.x, QP.y, QP.z);
  QBVHPointRangeQuery<float> range_query(bvh, QP3, radius);

  const float radius2 = radius * radius;
  const int row = q * max_points;

  int filled = 0;
  float worst = -1.0f;  // valid only once filled == max_points
  int worst_pos = 0;

  uint32_t cand;
  while (range_query.next(cand)) {
    float3 cp = points[cand];
    float dx = QP.x - cp.x;
    float dy = QP.y - cp.y;
    float dz = QP.z - cp.z;
    float d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > radius2) continue;

    if (filled < max_points) {
      out_idx[row + filled] = (int)cand;
      work[row + filled] = d2;
      filled++;
      if (filled == max_points) {
        worst = work[row];
        worst_pos = 0;
        for (int k = 1; k < max_points; k++) {
          if (work[row + k] > worst) {
            worst = work[row + k];
            worst_pos = k;
          }
        }
      }
    } else if (d2 < worst) {
      out_idx[row + worst_pos] = (int)cand;
      work[row + worst_pos] = d2;
      worst = work[row];
      worst_pos = 0;
      for (int k = 1; k < max_points; k++) {
        if (work[row + k] > worst) {
          worst = work[row + k];
          worst_pos = k;
        }
      }
    }
  }

  out_count[q] = filled;

  if (return_dists || return_points) {
    for (int i = 0; i < filled; i++) {
      int idx = out_idx[row + i];
      if (return_dists) out_dist[row + i] = sqrtf(work[row + i]);
      if (return_points) out_pts[row + i] = points[idx];
    }
  }
}

}  // namespace

// Single-batch radius search returning the (indices, points, distances, count)
// tensors for one (points, queries) pair. The Python layer handles batching,
// dtype casting, and the batch-dimension squeeze.
std::vector<torch::Tensor> radius_search_pysdf_cuda_single(
    torch::Tensor points, torch::Tensor queries, double radius,
    int64_t max_points, bool return_dists, bool return_points) {
  TORCH_CHECK(points.is_cuda() && queries.is_cuda(),
              "pysdf_cuda: points and queries must be CUDA tensors");
  TORCH_CHECK(points.scalar_type() == torch::kFloat32 &&
                  queries.scalar_type() == torch::kFloat32,
              "pysdf_cuda: points and queries must be float32");
  TORCH_CHECK(points.dim() == 2 && points.size(1) == 3,
              "pysdf_cuda: points must have shape (N, 3)");
  TORCH_CHECK(queries.dim() == 2 && queries.size(1) == 3,
              "pysdf_cuda: queries must have shape (Q, 3)");
  TORCH_CHECK(max_points > 0, "pysdf_cuda: max_points must be > 0");

  points = points.contiguous();
  queries = queries.contiguous();

  const int N = static_cast<int>(points.size(0));
  const int Q = static_cast<int>(queries.size(0));
  const int mp = static_cast<int>(max_points);

  auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(points.device());
  auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(points.device());

  torch::Tensor out_idx = torch::zeros({Q, mp}, i32);
  torch::Tensor out_count = torch::zeros({Q}, i32);
  torch::Tensor out_dist =
      return_dists ? torch::zeros({Q, mp}, f32) : torch::empty({0}, f32);
  torch::Tensor out_pts = return_points ? torch::zeros({Q, mp, 3}, f32)
                                        : torch::empty({0, mp, 3}, f32);

  if (N == 0 || Q == 0) {
    return {out_idx, out_pts, out_dist, out_count};
  }

  const float3 *d_points =
      reinterpret_cast<const float3 *>(points.data_ptr<float>());
  const float3 *d_queries =
      reinterpret_cast<const float3 *>(queries.data_ptr<float>());

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  // ---- Build the QBVH over the reference points (pysdf/minigql). ----
  PrimState *d_prims = nullptr;
  PNEMO_CUDA_CHECK(cudaMalloc(&d_prims, sizeof(PrimState) * N));
  nvtxRangePushA("pysdf_cuda: fill prims");
  fill_build_prims_kernel<<<ceil_div(N, kBlockDim), kBlockDim, 0, stream>>>(
      d_prims, N, d_points);
  PNEMO_CUDA_CHECK(cudaPeekAtLastError());
  // The builder + DeviceData allocate/launch on the default stream internally,
  // so prims must be ready before it runs.
  PNEMO_CUDA_CHECK(cudaStreamSynchronize(stream));
  nvtxRangePop();

  QBVH<float> *bvh = new QBVH<float>();
  nvtxRangePushA("pysdf_cuda: build QBVH");
  genericSpatialMedianBuilder(*bvh, static_cast<size_t>(N), d_prims, stream);
  // Ensure the BVH (built partly on the default stream) is complete before the
  // query kernel reads it on ``stream``.
  PNEMO_CUDA_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();

  // ---- Per-query range query + bounded max_points selection. ----
  torch::Tensor work = torch::empty({Q, mp}, f32);
  nvtxRangePushA("pysdf_cuda: range query");
  query_select_kernel<<<ceil_div(Q, kBlockDim), kBlockDim, 0, stream>>>(
      N, d_points, Q, d_queries, static_cast<float>(radius), bvh->get(), mp,
      return_dists, return_points, out_idx.data_ptr<int>(),
      out_count.data_ptr<int>(),
      return_dists ? out_dist.data_ptr<float>() : nullptr,
      return_points ? reinterpret_cast<float3 *>(out_pts.data_ptr<float>())
                    : nullptr,
      work.data_ptr<float>());
  PNEMO_CUDA_CHECK(cudaPeekAtLastError());
  // The query kernel reads the BVH; finish before freeing it (DeviceData frees
  // on the default stream).
  PNEMO_CUDA_CHECK(cudaStreamSynchronize(stream));
  nvtxRangePop();

  PNEMO_CUDA_CHECK(cudaFree(d_prims));
  delete bvh;

  return {out_idx, out_pts, out_dist, out_count};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("radius_search_pysdf_cuda_single", &radius_search_pysdf_cuda_single,
        "pysdf QBVH radius search, single (points, queries) pair");
}
