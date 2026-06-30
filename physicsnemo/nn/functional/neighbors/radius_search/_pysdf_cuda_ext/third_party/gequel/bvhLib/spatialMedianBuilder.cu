// ======================================================================== //
// Copyright 2020++ Ingo Wald                                               //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "gequel/bvhLib/builder.h"
#include "gequel/bvhLib/BVH.h" // bvh_vecf_t
#include "gequel/common.h"

#include <cassert>
#include <nvtx3/nvToolsExt.h>
#include <thrust/device_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/logical.h>

#include <cooperative_groups.h>
#include <algorithm>

// --- PhysicsNeMo vendoring patch: CUB compatibility ----------------------
// Newer CUDA/CUB (>= ~12.5, incl. the CUDA 12.8/13 Blackwell toolkits) removed
// the ``cub::Min()`` / ``cub::Max()`` reduction functors. Provide portable
// drop-in replacements so this vendored builder compiles across CUDA versions.
// (Local modification to the upstream minigql source; see ../../NOTICE.)
namespace pnemo_cub_compat {
struct Min {
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return b < a ? b : a;
  }
};
struct Max {
  template <typename T>
  __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return a < b ? b : a;
  }
};
}  // namespace pnemo_cub_compat
// -------------------------------------------------------------------------

//#define PERSISTENT_BUILDER

namespace gequel {
namespace bvhLib {
using namespace gequel::common;

// Avoid a ton of #ifdef's below.
__device__ void printf_bvh_vec(bvh_vecf_t vec){
  #ifdef BVH_IS_3D
    printf("(%f, %f, %f)", vec.x, vec.y, vec.z);
  #else
    printf("(%f, %f)", vec.x, vec.y);
  #endif
}

struct SplitJob {
  /*! @{ the node/sub we're splitting, and how we're splitting it */
  uint32_t srcSub : 2;
  uint32_t dim : 2;
  float pos;
  /*! this values is used to resolve cases where we ouldn't decide
    based on split plane; in this case we atomically increase
    this counter to get a unique ID for each prim in the (degen)
    leaf, then use a bit of that value to sort left/right */
  uint32_t numWritten;
  /*! @} */

  uint32_t tgtNode : 28;
  uint32_t tgtSub0 : 2;
  uint32_t tgtSub1 : 2;
};

/*! during building, we will temporarily use this node layout,
  since this allows for more easily doing atomic updates, and
  for storing temp build job data */
struct BuildNode {
  enum { UNUSED = 0, ACTIVE, DONE_LEAF, DONE_INNER } Status;

  static BuildNode kickOffNode() {
    BuildNode bn;
    for (int i = 0; i < 4; i++) {
      bn.child[i].resetBounds();
      bn.child[i].numPrims = 0;
      bn.child[i].status = (i == 0) ? BuildNode::ACTIVE : BuildNode::UNUSED;
    }
    return bn;
  }

  class Child {
  public:

    using vec3i = owl::common::vec_t<int, 3>;
    using box3i = owl::common::box_t<vec3i>;

    using vec2i = owl::common::vec_t<int, 2>;
    using box2i = owl::common::box_t<vec2i>;

    inline __device__ uint32_t extend(const bvh_vecf_t P) {
      uint32_t oldNumPrims = atomicAdd(&numPrims, 1);

      setAtomicMinBits(&centBounds_or_finalBounds.lower.x, P.x);
      setAtomicMaxBits(&centBounds_or_finalBounds.upper.x, P.x);

      setAtomicMinBits(&centBounds_or_finalBounds.lower.y, P.y);
      setAtomicMaxBits(&centBounds_or_finalBounds.upper.y, P.y);

      #ifdef BVH_IS_3D
        setAtomicMinBits(&centBounds_or_finalBounds.lower.z, P.z);
        setAtomicMaxBits(&centBounds_or_finalBounds.upper.z, P.z);
      #endif

      return oldNumPrims;
    }

    inline __device__ uint32_t extend(const bvh_boxf_t &P) {
      uint32_t oldNumPrims = atomicAdd(&numPrims, 1);

      setAtomicMinBits(&centBounds_or_finalBounds.lower.x, P.lower.x);
      setAtomicMaxBits(&centBounds_or_finalBounds.upper.x, P.upper.x);

      setAtomicMinBits(&centBounds_or_finalBounds.lower.y, P.lower.y);
      setAtomicMaxBits(&centBounds_or_finalBounds.upper.y, P.upper.y);

      #ifdef BVH_IS_3D
        setAtomicMinBits(&centBounds_or_finalBounds.lower.z, P.lower.z);
        setAtomicMaxBits(&centBounds_or_finalBounds.upper.z, P.upper.z);
      #endif

      return oldNumPrims;
    }

    inline __device__ void reduce_bounds(const bvh_vecf_t lower, const bvh_vecf_t upper) {
      setAtomicMinBits(&centBounds_or_finalBounds.lower.x, lower.x);
      setAtomicMaxBits(&centBounds_or_finalBounds.upper.x, upper.x);

      setAtomicMinBits(&centBounds_or_finalBounds.lower.y, lower.y);
      setAtomicMaxBits(&centBounds_or_finalBounds.upper.y, upper.y);

      #ifdef BVH_IS_3D
        setAtomicMinBits(&centBounds_or_finalBounds.lower.z, lower.z);
        setAtomicMaxBits(&centBounds_or_finalBounds.upper.z, upper.z);
      #endif
    }

    inline __device__ bool tryGrowBounds(const bvh_boxf_t primBounds) {
      atomicAdd(&numPrims, 1);
      return setAtomicMinBits_returnIfModified(&centBounds_or_finalBounds.lower.x,
                                           primBounds.lower.x) |
             setAtomicMinBits_returnIfModified(&centBounds_or_finalBounds.lower.y,
                                           primBounds.lower.y) |
        #ifdef BVH_IS_3D
             setAtomicMinBits_returnIfModified(&centBounds_or_finalBounds.lower.z,
                                           primBounds.lower.z) |
        #endif
             setAtomicMaxBits_returnIfModified(&centBounds_or_finalBounds.upper.x,
                                           primBounds.upper.x) |
             setAtomicMaxBits_returnIfModified(&centBounds_or_finalBounds.upper.y,
                                           primBounds.upper.y) |
        #ifdef BVH_IS_3D
             setAtomicMaxBits_returnIfModified(&centBounds_or_finalBounds.upper.z,
                                           primBounds.upper.z);
        #else
            false;
        #endif
    }

    inline __device__ void minimal_copy(const Child a) {
      this->centBounds_or_finalBounds.lower = a.centBounds_or_finalBounds.lower;
      this->centBounds_or_finalBounds.upper = a.centBounds_or_finalBounds.upper;
      this->numPrims =
          0; // in shared mem this is set to zero to account for just new prims.
      return;
    }

    __host__ __device__
    bvh_boxf_t getDecodedBoundsCopy() const {
      return decode_box(centBounds_or_finalBounds);
    }

    __host__ __device__
    void setBounds(const bvh_boxf_t &decoded) {
      centBounds_or_finalBounds = encode_box(decoded);
    }

    __host__ __device__
    void resetBounds() {
      centBounds_or_finalBounds = encode_box(bvh_boxf_t());
    }

    uint32_t numPrims;
    /*! do not need to store 'unused' : unused means numPrims == 0 */
    uint32_t offset : 30;
    uint32_t status : 2;

  private:

    __host__ __device__
    static bvh_vecf_t decode_vec(const bvh_veci_t &encoded){
      #ifdef BVH_IS_3D
        return bvh_vecf_t(decode(encoded.x), decode(encoded.y), decode(encoded.z));
      #else
        return bvh_vecf_t(decode(encoded.x), decode(encoded.y));
      #endif
    }
    __host__ __device__
    static bvh_boxf_t decode_box(const bvh_boxi_t &encoded){
      return bvh_boxf_t(decode_vec(encoded.lower), decode_vec(encoded.upper));
    }

    __host__ __device__
    static bvh_veci_t encode_vec(const bvh_vecf_t &decoded){
      #ifdef BVH_IS_3D
        return bvh_veci_t(encode(decoded.x), encode(decoded.y), encode(decoded.z));
      #else
        return bvh_veci_t(encode(decoded.x), encode(decoded.y));
      #endif
    }
    __host__ __device__
    static bvh_boxi_t encode_box(const bvh_boxf_t &decoded){
      return bvh_boxi_t(encode_vec(decoded.lower), encode_vec(decoded.upper));
    }

    /*! the _centroid_ bounds of that node, to select next split pos */
    // Conceptually, this is a float. But it sits encoded at rest as an int.
    // See common.h Atomic*Bits functions.
    bvh_boxi_t centBounds_or_finalBounds;

  } child[4];
  int parent;
  SplitJob split;
};

struct BuildStats {
  volatile uint32_t numBuildNodes;
  volatile uint32_t numActiveSplits;
  uint32_t leafListPos;
};

inline __device__ int allocateNode(BuildStats *buildStats,
                                   BuildNode *buildNodes) {
  int newNodeID = atomicAdd((unsigned *)&buildStats->numBuildNodes, 1);
  BuildNode &node = buildNodes[newNodeID];
  for (int i = 0; i < 4; i++)
    node.child[i].status = BuildNode::UNUSED;
  return newNodeID;
}

template <int VERBOSE = 0>
inline __device__ bool findSplit(BuildNode *buildNodes, uint32_t thisNodeID,
                                 BuildStats *buildStats,
                                 int maxPrimsAllowedPerLeaf) {
  BuildNode &node = buildNodes[thisNodeID];

  /* creating/finding a split consists of two stps: first, to
     find which of the usb-nodes to split; and second, to
     allocate where the split will go into (either the current
     sub plus a not-yet-used other sub, or the first two subs in
     another node */

  // step 1: find which node to split:
  int bestSubToSplit = -1;
  int bestSplitDim = -1;
  float bestSplitPos = 0.f;
  float bestSplitGain = -1; // negative value, so ANY split (even
  // zero volume) is better
  // 0.f;
  for (int sub = 0; sub < 4; sub++) {
    if (node.child[sub].status != BuildNode::ACTIVE)
      // this is not a valid sub yet
      continue;

    if (node.child[sub].numPrims <= maxPrimsAllowedPerLeaf) {
      node.child[sub].status = BuildNode::DONE_LEAF;
      continue;
    }

    const auto decodedBounds = node.child[sub].getDecodedBoundsCopy();
    float splitGain = area(decodedBounds);
    if (splitGain < bestSplitGain)
      // not a good split (for now), but node remains active
      continue;

    int dim = arg_max(decodedBounds.size());
    float lo = decodedBounds.lower[dim];
    float hi = decodedBounds.upper[dim];
    float mid = (lo + hi) / 2.f;
    if (mid == lo || mid == hi) {
      /* this is a case where a _real_ partition based on prim
         centroid being left/right of a split plane isn't
         possible, either because all prims' centroids are all
         co-located (which does, in particular, also happen in the
         'usual" single-prim-beomes-a-leaf case, because a single
         prim by default has 'all' its centroids on a single
         point!), or because the node is so ridiculously thin that
         a split plane cannot be represented with float accuracy
         any more. What to do now depends on whehter we are
         allowed to create leaves with more than one prim (then we
         just create one), or whether we have to keep splitting
         (in which case we have to split based on another
         criterion, for which we use dim==3, and then use
         'numWritten%2 to sort left/right) */

      if (node.child[sub].numPrims <= maxPrimsAllowedPerLeaf) {
        if (VERBOSE){
          const auto box = node.child[sub].getDecodedBoundsCopy();
          printf("made leaf %i:%i -> ", thisNodeID, sub);
          printf_bvh_vec(box.lower);
          printf(":");
          printf_bvh_vec(box.upper);
          printf(": %i\n", node.child[sub].numPrims);
        }
        node.child[sub].status = BuildNode::DONE_LEAF;
        continue;
      } else {
        dim = 3;
      }
    }

    // yay - we have a split
    bestSplitPos = mid;
    bestSplitDim = dim;
    bestSplitGain = splitGain;
    bestSubToSplit = sub;
  }
  if (bestSubToSplit == -1) {
    if (VERBOSE)
      printf("node %i -> no split found\n", thisNodeID);
    return false;
  }

  SplitJob &split = node.split;
  split.pos = bestSplitPos;
  split.dim = bestSplitDim;
  split.numWritten = 0; // bestSplitDim;
  split.srcSub = bestSubToSplit;
  // split.srcNode = thisNodeID;

  // step 2: allocate output:
  int firstFreeSub = 0;
  while (firstFreeSub < 4 &&
         node.child[firstFreeSub].status != BuildNode::UNUSED)
    firstFreeSub++;

  if (firstFreeSub == 4) {
    // no more free subs in this node.... finalize current node
    // as done inner node, and allocate two new subs in a new
    // node
    int newNodeID = allocateNode(buildStats, buildNodes);
    node.child[split.srcSub].status = BuildNode::DONE_INNER;
    node.child[split.srcSub].offset = newNodeID;

    split.tgtNode = newNodeID;
    split.tgtSub0 = 0;
    split.tgtSub1 = 1;
  } else {
    // there's one more sub here - overwrite the current node,
    // and use free one, which together is two that we can write
    // to.
    split.tgtNode = thisNodeID;
    split.tgtSub0 = split.srcSub;
    split.tgtSub1 = firstFreeSub;
  }

  if (VERBOSE){
    printf("splitting %i:%i (", thisNodeID, split.srcSub);
    printf_bvh_vec(node.child[split.srcSub].getDecodedBoundsCopy().lower);
    printf(":");
    printf_bvh_vec(node.child[split.srcSub].getDecodedBoundsCopy().upper);
    printf(":%i) -> %f@%i -> %i:(%i,%i)\n",
           node.child[split.srcSub].numPrims, split.pos, split.dim,
           split.tgtNode, split.tgtSub0, split.tgtSub1);
  }

  // finally, clear the two target subs so we can properly write
  // into in the next pass
  BuildNode &tgtNode = buildNodes[split.tgtNode];

  tgtNode.child[split.tgtSub0].resetBounds();
  tgtNode.child[split.tgtSub0].status = BuildNode::ACTIVE;
  tgtNode.child[split.tgtSub0].numPrims = 0;

  tgtNode.child[split.tgtSub1].resetBounds();
  tgtNode.child[split.tgtSub1].status = BuildNode::ACTIVE;
  tgtNode.child[split.tgtSub1].numPrims = 0;

  return true;
}

/*! tihs assumes that buildNode[0].child[0] was cleared, and that
  all the primStates[.].bounds have already been filled in before
  this got called */
__device__ __forceinline__ void reduce_block_wide(const bvh_vecf_t P, bvh_vecf_t &lower,
                                                  bvh_vecf_t &upper) {
  // PhysicsNeMo: reduce in float (coords are float). Upstream used
  // ``cub::BlockReduce<int, 128>``, which truncated sub-1 / negative
  // coordinates and produced wrong root bounds for general point clouds.
  // ``cub::Min()`` / ``cub::Max()`` replaced by the pnemo_cub_compat shim above
  // (those functors were removed in newer CUB).
  typedef cub::BlockReduce<float, 128> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_st1, temp_st2;

  lower.x = BlockReduce(temp_st1).Reduce(P.x, pnemo_cub_compat::Min());
  upper.x = BlockReduce(temp_st2).Reduce(P.x, pnemo_cub_compat::Max());
  __syncthreads();
  lower.y = BlockReduce(temp_st1).Reduce(P.y, pnemo_cub_compat::Min());
  upper.y = BlockReduce(temp_st2).Reduce(P.y, pnemo_cub_compat::Max());
  __syncthreads();
  #ifdef BVH_IS_3D
    lower.z = BlockReduce(temp_st1).Reduce(P.z, pnemo_cub_compat::Min());
    upper.z = BlockReduce(temp_st2).Reduce(P.z, pnemo_cub_compat::Max());
    __syncthreads();
  #endif
  return;
}

__global__ void kickOffBuild(BuildStats *buildStats, BuildNode *buildNodes,
                             PrimState *primStates, int numPoints) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= numPoints)
    return;

  // const vec3f P = points[gid];
  auto &prim = primStates[gid];
  prim.node = 0;
  prim.sub = 0;
  prim.done = false;
  bvh_vecf_t lower, upper;
  reduce_block_wide(prim.bounds.center(), lower, upper);

  auto &gnode = buildNodes[0].child[0];
  if (threadIdx.x == 0) {
    gnode.reduce_bounds(lower, upper);
  }

  if (gid == 0) {
    buildStats->numBuildNodes = 1;
    buildStats->leafListPos = 0;
    gnode.numPrims = numPoints;
  }
}

// This kernel relies on the fact that all build nodes can be put into the CTA
// shmem If they cannot the non shared version is called.
__global__ void tricklePrims_shared(BuildNode *buildNodes,
                                    bool *buildNodeActive,
                                    PrimState *primStates, const int numPrims,
                                    const int numBuildNodes) {
  extern __shared__ BuildNode::Child child_nodes[];
  for (int ll = threadIdx.x; ll < numBuildNodes * 4; ll += blockDim.x) {
    int nodeidx = (int)(ll / 4);
    int childidx = ll % 4;
    child_nodes[ll].minimal_copy(buildNodes[nodeidx].child[childidx]);
  }
  __syncthreads();

  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  bool ret_flag = false;

  if (gid < numPrims) {

    PrimState &prim = primStates[gid];
    if (!prim.done) {
      // if (gid < 10) printf("   //trickle: prim %i is alreay done...\n",gid);
      BuildNode &node = buildNodes[prim.node];

      const SplitJob split = node.split;
      if (node.child[prim.sub].status == BuildNode::DONE_LEAF) {
        // the last split has set our sub in that node to be a leaf -
        // we're now officially done.
        prim.done = true;
        ret_flag = true;
      }

      if (prim.sub != split.srcSub) {
        // we _are_ active, and we _are_ in a node that's being split
        // - just not in the sub that's being split
        ret_flag = true;
      }

      if (!ret_flag) {
        int ourSide;
        // we _are_ in a node/sub that's being split
        if (split.dim == 3) {
          // careful here: this was a degereate split, where we couldn't
          // create any valid split plane but still have to keep on
          // splitting
          int thisPrimInThisLeaf = atomicAdd(&node.split.numWritten, 1);
          ourSide = thisPrimInThisLeaf % 2;
        } else {
          // OK, this is a "regular" split where we check whether our
          // centroid is left/right of the split plane
          float mid =
              (prim.bounds.lower[split.dim] + prim.bounds.upper[split.dim]) /
              2.f;
          ourSide = mid >= split.pos;
        }
        int ourSub = (ourSide != 0) ? split.tgtSub1 : split.tgtSub0;
        int idxx = split.tgtNode * 4 + ourSub;
        // printf(" value of tgtNode %d and ourSub %d from tid
        // %d\n",split.tgtNode,ourSub,gid);
        child_nodes[idxx].extend(prim.bounds.center());

        prim.node = split.tgtNode;
        prim.sub = ourSub;
      }
    }
  }

  __syncthreads();

  for (int ll = threadIdx.x; ll < numBuildNodes * 4; ll += blockDim.x) {
    int nodeidx = (int)(ll / 4);
    int childidx = ll % 4;
    auto &curr_schild = child_nodes[ll];
    int new_cnt = curr_schild.numPrims;
    // printf("new count for childnode %d from id %d\n",new_cnt,ll);
    if (new_cnt != 0) {
      const auto decodedBounds = curr_schild.getDecodedBoundsCopy();
      auto lower = decodedBounds.lower;
      auto upper = decodedBounds.upper;
      auto &curr_gchild = buildNodes[nodeidx].child[childidx];
      curr_gchild.reduce_bounds(lower, upper);
      atomicAdd(&curr_gchild.numPrims, new_cnt);
    }
  }
}

__global__ void nodeprinter(BuildNode *buildNodes, const int numBuildNodes) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < numBuildNodes * 4) {
    int nodeidx = (int)(i / 4);
    int childidx = i % 4;
    auto &curr_gchild = buildNodes[nodeidx].child[childidx];
    const auto decodedBounds = curr_gchild.getDecodedBoundsCopy();
    auto lower = decodedBounds.lower;
    auto upper = decodedBounds.upper;
    printf("node printer childid %d and bounds ", i);
    printf_bvh_vec(lower);
    printf(" - ");
    printf_bvh_vec(upper);
    printf("\n");
  }
}

__global__ void tricklePrims(BuildNode *buildNodes, bool *buildNodeActive,
                             PrimState *primStates, int numPrims) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= numPrims)
    return;
  PrimState &prim = primStates[gid];
  if (prim.done) {
    // if (gid < 10) printf("   //trickle: prim %i is alreay done...\n",gid);
    return;
  }

  BuildNode &node = buildNodes[prim.node];

  const SplitJob split = node.split;
  if (node.child[prim.sub].status == BuildNode::DONE_LEAF) {
    // the last split has set our sub in that node to be a leaf -
    // we're now officially done.
    prim.done = true;
    return;
  }

  if (prim.sub != split.srcSub) {
    // we _are_ active, and we _are_ in a node that's being split
    // - just not in the sub that's being split
    return;
  }

  int ourSide;
  // we _are_ in a node/sub that's being split
  if (split.dim == 3) {
    // careful here: this was a degereate split, where we couldn't
    // create any valid split plane but still have to keep on
    // splitting
    int thisPrimInThisLeaf = atomicAdd(&node.split.numWritten, 1);
    ourSide = thisPrimInThisLeaf % 2;
  } else {
    // OK, this is a "regular" split where we check whether our
    // centroid is left/right of the split plane
    float mid =
        (prim.bounds.lower[split.dim] + prim.bounds.upper[split.dim]) / 2.f;
    ourSide = mid >= split.pos;
  }
  int ourSub = (ourSide != 0) ? split.tgtSub1 : split.tgtSub0;
  buildNodes[split.tgtNode].child[ourSub].extend(prim.bounds.center());
  prim.node = split.tgtNode;
  prim.sub = ourSub;
}

/*! tihs assumes that buildNode[0].child[0] was cleared, and that
  we can atomically grow this node for every point */
__global__ void createSplits(BuildStats *buildStats, BuildNode *buildNodes,
                             bool *buildNodeActive,
                             /*! num build nodes at START of this kernel -
                               (the kernel may allocate new ones when it
                               processes splits, but these shouldn't be
                               processed just yet) */
                             int numBuildNodes, int maxPrimsAllowedPerLeaf) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= numBuildNodes)
    return;

  if (!buildNodeActive[gid])
    // this (quad-)node is already fully done, and will not find
    // any new splits
    return;

  bool foundValidSplit =
      findSplit(buildNodes, gid, buildStats, maxPrimsAllowedPerLeaf);
  auto &split = buildNodes[gid].split;
  if (foundValidSplit) {
    // we did create a split - that split either has the current
    // build node split iself internally (then no changes are
    // reqrueid to the active list), or activeates a new build
    // node for the children it created (but still remains active
    // since there are other children that may still remain split)
    if (split.tgtNode == gid) {
      /* nothing to do - this is a split that splits one of the
         subs into two new subs */
    } else {
      /*! this split creates two new subs in another buildnode -
        we have t oative this - but may still have other subs to
        split, so remains alive */
      buildNodeActive[split.tgtNode] = true;
    }

    atomicAdd((unsigned *)&buildStats->numActiveSplits, 1);
  } else {
    /* no split found .. */
    buildNodeActive[gid] = false;
  }
}

/*! tihs assumes that buildNode[0].child[0] was cleared, and that
  we can atomically grow this node for every point */
__global__ void persistent_builder(BuildStats *buildStats,
                                   BuildNode *buildNodes, bool *buildNodeActive,
                                   PrimState *primStates, int numPrims,
                                   int maxPrimsAllowedPerLeaf) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto g = cooperative_groups::this_grid();

  for (int kk = 0;; kk++) {
    auto numBuildNodes = buildStats->numBuildNodes;

    if (tid == 0) {
      buildStats->numActiveSplits = 0;
    }
    g.sync();

    for (int gid = tid; gid < numBuildNodes; gid += blockDim.x * gridDim.x) {

      if (!buildNodeActive[gid])
        // this (quad-)node is already fully done, and will not find
        // any new splits
        continue;

      bool foundValidSplit =
          findSplit(buildNodes, gid, buildStats, maxPrimsAllowedPerLeaf);
      auto &split = buildNodes[gid].split;
      if (foundValidSplit) {
        // we did create a split - that split either has the current
        // build node split iself internally (then no changes are
        // reqrueid to the active list), or activeates a new build
        // node for the children it created (but still remains active
        // since there are other children that may still remain split)
        if (split.tgtNode == gid) {
          /* nothing to do - this is a split that splits one of the
             subs into two new subs */
        } else {
          /*! this split creates two new subs in another buildnode -
            we have t oative this - but may still have other subs to
            split, so remains alive */
          buildNodeActive[split.tgtNode] = true;
        }

        atomicAdd((unsigned *)&buildStats->numActiveSplits, 1);
      } else {
        /* no split found .. */
        buildNodeActive[gid] = false;
      }
    }

    g.sync();

    if (buildStats->numActiveSplits == 0)
      return;

    for (int gid = tid; gid < numPrims; gid += blockDim.x * gridDim.x) {
      PrimState &prim = primStates[gid];
      if (prim.done) {
        // if (gid < 10) printf("   //trickle: prim %i is alreay
        // done...\n",gid);
        continue;
      }

      BuildNode &node = buildNodes[prim.node];

      const SplitJob split = node.split;
      if (node.child[prim.sub].status == BuildNode::DONE_LEAF) {
        // the last split has set our sub in that node to be a leaf -
        // we're now officially done.
        prim.done = true;
        continue;
      }

      if (prim.sub != split.srcSub) {
        // we _are_ active, and we _are_ in a node that's being split
        // - just not in the sub that's being split
        continue;
      }

      int ourSide;
      // we _are_ in a node/sub that's being split
      if (split.dim == 3) {
        // careful here: this was a degereate split, where we couldn't
        // create any valid split plane but still have to keep on
        // splitting
        int thisPrimInThisLeaf = atomicAdd(&node.split.numWritten, 1);
        ourSide = thisPrimInThisLeaf % 2;
      } else {
        // OK, this is a "regular" split where we check whether our
        // centroid is left/right of the split plane
        float mid =
            (prim.bounds.lower[split.dim] + prim.bounds.upper[split.dim]) / 2.f;
        ourSide = mid >= split.pos;
      }
      int ourSub = (ourSide != 0) ? split.tgtSub1 : split.tgtSub0;
      buildNodes[split.tgtNode].child[ourSub].extend(prim.bounds.center());
      prim.node = split.tgtNode;
      prim.sub = ourSub;
    }
    g.sync();
  }
}

template <int max_nodes>
__global__ void
persistent_builder_shared(BuildStats *buildStats, BuildNode *buildNodes,
                          bool *buildNodeActive, PrimState *primStates,
                          int numPrims, int maxPrimsAllowedPerLeaf) {
  extern __shared__ BuildNode::Child child_nodes[];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto g = cooperative_groups::this_grid();

  for (int kk = 0;; kk++) {
    auto numBuildNodes = buildStats->numBuildNodes;

    if (tid == 0) {
      buildStats->numActiveSplits = 0;
    }
    g.sync();

    // This is creating splits
    for (int gid = tid; gid < numBuildNodes; gid += blockDim.x * gridDim.x) {

      if (!buildNodeActive[gid])
        // this (quad-)node is already fully done, and will not find
        // any new splits
        continue;

      bool foundValidSplit =
          findSplit(buildNodes, gid, buildStats, maxPrimsAllowedPerLeaf);
      auto &split = buildNodes[gid].split;
      if (foundValidSplit) {
        // we did create a split - that split either has the current
        // build node split iself internally (then no changes are
        // reqrueid to the active list), or activeates a new build
        // node for the children it created (but still remains active
        // since there are other children that may still remain split)
        if (split.tgtNode == gid) {
          /* nothing to do - this is a split that splits one of the
             subs into two new subs */
        } else {
          /*! this split creates two new subs in another buildnode -
            we have t oative this - but may still have other subs to
            split, so remains alive */
          buildNodeActive[split.tgtNode] = true;
        }

        atomicAdd((unsigned *)&buildStats->numActiveSplits, 1);
      } else {
        /* no split found .. */
        buildNodeActive[gid] = false;
      }
    }

    g.sync();

    numBuildNodes = buildStats->numBuildNodes;
    if (buildStats->numActiveSplits == 0 || numBuildNodes > max_nodes)
      return;

    for (int ll = threadIdx.x; ll < numBuildNodes * 4; ll += blockDim.x) {
      int nodeidx = (int)(ll / 4);
      int childidx = ll % 4;
      child_nodes[ll].minimal_copy(buildNodes[nodeidx].child[childidx]);
    }
    __syncthreads();

    for (int gid = tid; gid < numPrims; gid += blockDim.x * gridDim.x) {
      PrimState &prim = primStates[gid];
      if (prim.done) {
        // if (gid < 10) printf("   //trickle: prim %i is alreay
        // done...\n",gid);
        continue;
      }

      BuildNode &node = buildNodes[prim.node];

      const SplitJob split = node.split;
      if (node.child[prim.sub].status == BuildNode::DONE_LEAF) {
        // the last split has set our sub in that node to be a leaf -
        // we're now officially done.
        prim.done = true;
        continue;
      }

      if (prim.sub != split.srcSub) {
        // we _are_ active, and we _are_ in a node that's being split
        // - just not in the sub that's being split
        continue;
      }

      int ourSide;
      // we _are_ in a node/sub that's being split
      if (split.dim == 3) {
        // careful here: this was a degereate split, where we couldn't
        // create any valid split plane but still have to keep on
        // splitting
        int thisPrimInThisLeaf = atomicAdd(&node.split.numWritten, 1);
        ourSide = thisPrimInThisLeaf % 2;
      } else {
        // OK, this is a "regular" split where we check whether our
        // centroid is left/right of the split plane
        float mid =
            (prim.bounds.lower[split.dim] + prim.bounds.upper[split.dim]) / 2.f;
        ourSide = mid >= split.pos;
      }
      int ourSub = (ourSide != 0) ? split.tgtSub1 : split.tgtSub0;
      // buildNodes[split.tgtNode].child[ourSub].extend(prim.bounds.center());
      int idxx = split.tgtNode * 4 + ourSub;
      // if(idxx > 256*4){
      //  printf("hey am here with number %d and %d nodes tgt %d and our sub
      //  %d\n", idxx, numBuildNodes,split.tgtNode, ourSub);
      //}
      child_nodes[idxx].extend(prim.bounds.center());
      prim.node = split.tgtNode;
      prim.sub = ourSub;
    }
    __syncthreads();

    for (int ll = threadIdx.x; ll < numBuildNodes * 4; ll += blockDim.x) {
      int nodeidx = (int)(ll / 4);
      int childidx = ll % 4;
      auto &curr_schild = child_nodes[ll];
      int new_cnt = curr_schild.numPrims;
      // printf("new count for childnode %d from id %d\n",new_cnt,ll);
      if (new_cnt != 0) {
        const auto decodedBounds = curr_schild.getDecodedBoundsCopy();
        auto lower = decodedBounds.lower;
        auto upper = decodedBounds.upper;
        auto &curr_gchild = buildNodes[nodeidx].child[childidx];
        curr_gchild.reduce_bounds(lower, upper);
        atomicAdd(&curr_gchild.numPrims, new_cnt);
      }
    }
    g.sync();
  }
}

/*! in the first build state ,we only operated on _centroid_
  bounds - but final BVH needs world bounds. this clears all the
  bounds to empty, so later prim writing stage can atomically
  update them */
__global__ void clearNodeBoundsAndWriteParents(BuildStats *buildStats,
                                               BuildNode *buildNodes,
                                               int numBuildNodes) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  int nodeID = gid / 4;
  int subID = gid % 4;
  if (nodeID >= numBuildNodes)
    return;

  auto &thisSub = buildNodes[nodeID].child[subID];
  if (gid == 0) {
    buildNodes[0].parent = -1; // parents[0] = -1;
  }

  thisSub.resetBounds();
  if (thisSub.status == BuildNode::DONE_INNER) {
    buildNodes[thisSub.offset].parent = nodeID;
    // parents[thisSub.offset] = nodeID;
  } else if (thisSub.status == BuildNode::DONE_LEAF) {
    // xxxx
    // if we _do_ write leaf lists, then this can from now on be
    // used as a position in the leaf list to write to. if not
    // ,this doesn't matter - but alos doesn't hurt
    thisSub.offset = atomicAdd(&buildStats->leafListPos, thisSub.numPrims);
#if GQL_USE_ITEM_LISTS
    // CAREFUL: if we DO use item lists, then tihs value has to be
    // set to 0, because it's a "temporary" value that the builder
    // uses in the final item list writing stage; in that stage,
    // it will atomically increase this for each item it writes,
    // and therefore _requires_ this to start with 0.... but if we
    // do NOT use item lists, then this final atomic increasing
    // will never get done (since the item list will never get
    // written), so we have to write the _final_ value, which is 1.
    thisSub.numPrims = 0;
#else
    thisSub.numPrims = 1;
#endif
  } else {
    /* nothing to do - this is a invalid/unused sub in that node */
  }
}

inline __device__ int findSub(BuildNode buildNode, uint32_t nodeID) {
  for (int i = 0; i < 4; i++)
    if (buildNode.child[i].status == BuildNode::DONE_INNER &&
        buildNode.child[i].offset == nodeID)
      return i;
  return 4;
}

/*! we build inner nodes by atomically pushing all leaf nodes up,
  until said leaf node ither no longer grows the parent node, or
  it reaches the root */
__global__ void buildInnerBounds(BuildNode *buildNodes,
                                 // int        *parents,
                                 int numBuildNodes) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  int nodeID = gid / 4;
  int subID = gid % 4;
  if (nodeID >= numBuildNodes)
    return;

  auto &thisSub = buildNodes[nodeID].child[subID];
  if (thisSub.status != BuildNode::DONE_LEAF)
    // we'll only start pushing upwards from leaves ....
    return;

  const auto decodedBounds = thisSub.getDecodedBoundsCopy();
  int parent = buildNodes[nodeID].parent; // parents[nodeID];
  while (parent >= 0) {
    int parentSubID = findSub(buildNodes[parent], nodeID);
    auto &parentSub = buildNodes[parent].child[parentSubID];
    if (!parentSub.tryGrowBounds(decodedBounds))
      // adding ourselves didn't change the parent, so no need to
      // go up even further
      break;

    nodeID = parent;
    parent = buildNodes[nodeID].parent; // parents[nodeID];
  }
}

/*! we build inner nodes by atomically pushing all leaf nodes up,
  until said leaf node ither no longer grows the parent node, or
  it reaches the root */
__global__ void checkBounds(BuildNode *buildNodes, int numBuildNodes) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  int nodeID = gid / 4;
  int subID = gid % 4;
  if (nodeID >= numBuildNodes)
    return;

  auto &thisSub = buildNodes[nodeID].child[subID];
  if (thisSub.status != BuildNode::DONE_INNER)
    // we'll only start pushing upwards from leaves ....
    return;

  bvh_boxf_t childBounds;
  BuildNode childNodes = buildNodes[thisSub.offset];
  for (int i = 0; i < 4; i++){
    childBounds.extend(childNodes.child[i].getDecodedBoundsCopy());
  }

  const auto thisSubDecodedBoundsCopy = thisSub.getDecodedBoundsCopy();
  if (!(thisSubDecodedBoundsCopy.contains(childBounds.lower) &&
        thisSubDecodedBoundsCopy.contains(childBounds.upper))){
    printf("INVALID BOUNDS: \t");
    printf_bvh_vec(childBounds.lower);
    printf(":");
    printf_bvh_vec(childBounds.upper);

    printf("\n  not in \t");
    printf_bvh_vec(thisSubDecodedBoundsCopy.lower);
    printf(": ");
    printf_bvh_vec(thisSubDecodedBoundsCopy.upper);
    printf("\n");

  } // if
}

/*! we build inner nodes by atomically pushing all leaf nodes up,
  until said leaf node ither no longer grows the parent node, or
  it reaches the root */
__global__ void writeFinalNodes(QBVH<float>::QuadNode *finalNodes,
                                const BuildNode *buildNodes,
                                // int *parents,
                                int numBuildNodes) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  int nodeID = gid / 4;
  int subID = gid % 4;

  if (nodeID >= numBuildNodes)
    return;

  auto &in = buildNodes[nodeID].child[subID];
  auto &out = finalNodes[nodeID].child[subID];

  finalNodes[nodeID].child[0].inChild0_parentID =
      buildNodes[nodeID].parent; // parents[nodeID];
  finalNodes[nodeID].child[1].inChild1_numChildrenRefitted = 0;

  out.bounds = in.getDecodedBoundsCopy();
  if (in.status == BuildNode::UNUSED) {
    out.makeInvalid();
  } else if (in.status == BuildNode::DONE_LEAF) {
    // printf("dev %i:%i making leaf %i
    // %i\n",nodeID,subID,in.offset,in.numPrims);
    out.makeLeaf(in.offset, in.numPrims);
  } else {
    out.makeInner(in.offset);
    if (in.offset >= numBuildNodes)
      printf("dev %i:%i making wrong inner %i\n", nodeID, subID, in.offset);
  }

  // if (nodeID < 10 && subID == 0) {
  //   auto &mem = finalNodes[nodeID];
  //   printf("final bounds %i:%i = (%f %f)(%f
  //   %f)\n%16lx\n%16lx\n%16lx\n%16lx\n%16lx\n%16lx\n%16lx\n%16lx\n%16lx\n%16lx\n%16lx\n%16lx\n%16lx\n%16lx\n%16lx\n%16lx\n",
  //          nodeID,subID,
  //          out.bounds.lower.x,
  //          out.bounds.lower.y,
  //          out.bounds.upper.x,
  //          out.bounds.upper.y,
  //          ((size_t*)&mem)[0],
  //          ((size_t*)&mem)[1],
  //          ((size_t*)&mem)[2],
  //          ((size_t*)&mem)[3],
  //          ((size_t*)&mem)[4],
  //          ((size_t*)&mem)[5],
  //          ((size_t*)&mem)[6],
  //          ((size_t*)&mem)[7],
  //          ((size_t*)&mem)[8],
  //          ((size_t*)&mem)[9],
  //          ((size_t*)&mem)[10],
  //          ((size_t*)&mem)[11],
  //          ((size_t*)&mem)[12],
  //          ((size_t*)&mem)[13],
  //          ((size_t*)&mem)[14],
  //          ((size_t*)&mem)[15]
  //          );
  // }
}

__global__ void writePrimsAndLeafBounds(BuildNode *buildNodes,
                                        QBVH<float>::QuadNode *finalNodes,
                                        const PrimState *primStates,
                                        int numPrims,
                                        uint32_t *itemListMem = 0) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= numPrims)
    return;

  PrimState prim = primStates[gid];
  auto &sub = buildNodes[prim.node].child[prim.sub];

  if (itemListMem) {
    uint32_t leafListOffset = sub.extend(prim.bounds);
    itemListMem[sub.offset + leafListOffset] = gid;
  } else {
    sub.setBounds(prim.bounds);
    sub.offset = gid;
  }
}

template <typename T>
struct uninitialized_allocator : thrust::device_allocator<T> {
  __host__ uninitialized_allocator() {}
  __host__ uninitialized_allocator(const uninitialized_allocator &other)
      : thrust::device_allocator<T>(other) {}
  __host__ ~uninitialized_allocator() {}

#if THRUST_CPP_DIALECT >= 2011
  uninitialized_allocator &operator=(const uninitialized_allocator &) = default;
#endif

  // for correctness, you should also redefine rebind when you inherit
  // from an allocator type; this way, if the allocator is rebound somewhere,
  // it's going to be rebound to the correct type - and not to its base
  // type for U
  template <typename U> struct rebind {
    typedef uninitialized_allocator<U> other;
  };

  // note that construct is annotated as
  // a __host__ __device__ function
  __host__ __device__ void construct(T *p) {
    // no-op
  }
};

void checkNodes(DeviceData<QBVH<float>::QuadNode> &d_nodes, int numNodes) {
  std::cout << "======================================================="
            << std::endl;
  std::cout << "sanity checking nodes ...." << std::endl;
  std::cout << "======================================================="
            << std::endl;
  PRINT(numNodes);
  PRINT(d_nodes.numElements);
  for (int nodeID = 0; nodeID < numNodes; nodeID++) {
    auto &node = d_nodes.d_ptr[nodeID];
    for (int cID = 0; cID < 4; cID++) {
      auto &child = node.child[cID];
      if (child.isValid() && !child.isLeaf() &&
          child.ref32.payload >= numNodes) {
        std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
                  << std::endl;
        PING;
        PRINT(child.ref32.payload);
        PRINT(numNodes);
      }
    }
  }
  std::cout << "==> NODES OK" << std::endl;
}

void genericSpatialMedianBuilder(QBVH<float> &bvh, size_t numPrims,
                                 PrimState *d_primStates, cudaStream_t stream) {
  constexpr int blockSize = 128;

#if GQL_USE_ITEM_LISTS
  const int maxPrimsAllowedPerLeaf = GQL_ITEM_LISTS_MAX_PRIMS;

  // TODO: We need a real heuristic for how much memory we'll need, accounting for internal nodes.
  const int maxPrimsAllowedPerLeaf_memoryEstimatePurposes = 7;

  static_assert(maxPrimsAllowedPerLeaf < (1 << QBVH<float>::numLeafCounterBits),
                "too few bits to encode this many prims per leaf");
  if (numPrims == 0) {
    bvh.d_bvh = QBVH<float>::invalid();
    return;
  }
#else
  const int maxPrimsAllowedPerLeaf = 1;
  const int maxPrimsAllowedPerLeaf_memoryEstimatePurposes = maxPrimsAllowedPerLeaf;
#endif
  int dev;
  cudaGetDevice(&dev);
  int max_shmem_per_block;
  CUDA_CHECK(cudaDeviceGetAttribute(&max_shmem_per_block,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));
  max_shmem_per_block /= 2;
  int max_bvh_nodes_shared = (int)(max_shmem_per_block / (4 * sizeof(BuildNode::Child)));
  CUDA_CHECK(cudaFuncSetAttribute(tricklePrims_shared,cudaFuncAttributeMaxDynamicSharedMemorySize,max_bvh_nodes_shared*4*sizeof(BuildNode::Child)));

  BuildStats *d_buildStats;
  BuildNode *d_buildNodes;
  bool *d_buildNodeActive;
  size_t max_build_nodes = (size_t)(numPrims / maxPrimsAllowedPerLeaf_memoryEstimatePurposes);
  if(numPrims % maxPrimsAllowedPerLeaf_memoryEstimatePurposes != 0)
    max_build_nodes++;

  CUDA_CHECK(cudaMallocAsync((void **)&d_buildNodes, max_build_nodes * sizeof(BuildNode), stream));
  CUDA_CHECK(cudaMallocAsync((void **)&d_buildStats, sizeof(BuildStats), stream));
  CUDA_CHECK(cudaMallocAsync((void **)&d_buildNodeActive, sizeof(bool) * max_build_nodes, stream));

  // compute buildnode[0] bounds atomically from all point positoins
  BuildNode buildNodes = BuildNode::kickOffNode();
  bool buildNodeActive = true;
  BuildStats bs;

  CUDA_CHECK(cudaMemcpyAsync(d_buildNodeActive, &buildNodeActive, sizeof(bool),
             cudaMemcpyHostToDevice,stream));
  CUDA_CHECK(cudaMemcpyAsync(d_buildNodes, &buildNodes, sizeof(BuildNode),
             cudaMemcpyHostToDevice,stream));
  kickOffBuild<<<divRoundUp((int)numPrims, blockSize), blockSize, 0, stream>>>(
      d_buildStats, d_buildNodes, d_primStates, numPrims);

#ifdef PERSISTENT_BUILDER
  // For persistent style kernel
  constexpr int shared_builder_nodes = 128;
  int numSM;
  cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, dev);
  int blocks_per_sm, blocks_per_sm_shared;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, persistent_builder, blockSize, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm_shared, persistent_builder_shared<shared_builder_nodes>,
      blockSize, 4 * shared_builder_nodes * sizeof(BuildNode::Child));

  void *kernelArgs[] = {
      (void *)&d_buildStats,      (void *)&d_buildNodes,
      (void *)&d_buildNodeActive, (void *)&d_primStates,
      (void *)&numPrims,          (void *)&maxPrimsAllowedPerLeaf};

  cudaLaunchCooperativeKernel(
      (const void *)persistent_builder_shared<shared_builder_nodes>,
      numSM * blocks_per_sm_shared, blockSize, kernelArgs,
      4 * shared_builder_nodes * sizeof(BuildNode::Child), nullptr);
  CUDA_CHECK(cudaMemcpy(&bs, d_buildStats, sizeof(BuildStats), cudaMemcpyDeviceToHost));
  CUDA_SYNC_CHECK();
  // std::cout << "number of split " << bs.numActiveSplits << " and number of
  // build nodes  " << bs.numBuildNodes << std::endl;
  if (bs.numActiveSplits != 0) {
    tricklePrims<<<divRoundUp((int)numPrims, blockSize), blockSize>>>(
        d_buildNodes, d_buildNodeActive, d_primStates, numPrims);
    cudaLaunchCooperativeKernel((const void *)persistent_builder,
                                numSM * blocks_per_sm, blockSize, kernelArgs, 0,
                                nullptr);
    CUDA_CHECK(cudaMemcpy(&bs, d_buildStats, sizeof(BuildStats), cudaMemcpyDeviceToHost));
    CUDA_SYNC_CHECK();
  }
#else
  CUDA_CHECK(cudaMemcpyAsync(&bs, d_buildStats, sizeof(BuildStats),
                        cudaMemcpyDeviceToHost, stream));
  // std::cout << "size of child node " << sizeof(BuildNode::Child) <<std::endl;
  for (int kk = 0;; kk++) {
    // create an initial split job based on that first build node
    // bs.numActiveSplits = 0;
    // cudaMemcpy(d_buildStats, &bs, sizeof(BuildStats),
    // cudaMemcpyHostToDevice);
    // buildStats[0] = bs;
    CUDA_CHECK(cudaMemsetAsync(((char *)d_buildStats) + sizeof(int), 0,
                               sizeof(int), stream));
    createSplits<<<divRoundUp((int)bs.numBuildNodes, blockSize), blockSize, 0, stream>>>(
        d_buildStats, d_buildNodes, d_buildNodeActive, bs.numBuildNodes,
        maxPrimsAllowedPerLeaf);

    CUDA_CHECK(cudaMemcpyAsync(&bs, d_buildStats, sizeof(BuildStats),
                          cudaMemcpyDeviceToHost, stream));
    // bs = buildStats[0];
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // printf("\n\nLoop id %d and num splits %d and numnodes
    // %d\n",kk,bs.numActiveSplits,bs.numBuildNodes); printf("printing befroe the
    // kernel call\n");
    // nodeprinter<<<divRoundUp((int)bs.numBuildNodes*4,blockSize),blockSize>>>(d_buildNodes,bs.numBuildNodes);
    // CUDA_SYNC_CHECK();

    if (bs.numActiveSplits == 0)
      // no more splits generated in this pass
      break;

    // Child node size is 32. 8 BVHnodes consume 1KB. 256 BVH nodes is 32KB
    // decent.
    if(bs.numBuildNodes >= max_build_nodes) {

      const auto old_max_build_nodes = max_build_nodes;
      max_build_nodes = std::min(
        static_cast<size_t>(old_max_build_nodes * 1.5),
        numPrims
      );

      BuildNode *h_buildNodes;
      printf("Warning: Builder ran out of memory. Reallocating memory\n");
      std::cout << "max nodes ->" << old_max_build_nodes << " bs.nodes " << bs.numBuildNodes << std::endl;
      //Copy build nodes
      CUDA_CHECK(cudaMallocHost((void**)&h_buildNodes, old_max_build_nodes * sizeof(BuildNode)));
      CUDA_CHECK(cudaMemcpyAsync(h_buildNodes,d_buildNodes,old_max_build_nodes * sizeof(BuildNode),cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaFreeAsync(d_buildNodes, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaMallocAsync((void**)&d_buildNodes, max_build_nodes * sizeof(BuildNode), stream));
      CUDA_CHECK(cudaMemcpyAsync(d_buildNodes,h_buildNodes,old_max_build_nodes * sizeof(BuildNode),cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaFreeHost(h_buildNodes));
     }

    if (bs.numBuildNodes <= max_bvh_nodes_shared) {
      tricklePrims_shared<<<divRoundUp((int)numPrims, blockSize), blockSize,
                            bs.numBuildNodes * 4 * sizeof(BuildNode::Child), stream>>>(
          d_buildNodes, d_buildNodeActive, d_primStates, numPrims,
          bs.numBuildNodes);
    } else {
      tricklePrims<<<divRoundUp((int)numPrims, blockSize), blockSize, 0, stream>>>(
          d_buildNodes, d_buildNodeActive, d_primStates, numPrims);
    }
    // CUDA_SYNC_CHECK();
    // printf("printing after the kernel call\n");
    // nodeprinter<<<divRoundUp((int)bs.numBuildNodes*4,blockSize),blockSize>>>(d_buildNodes,bs.numBuildNodes);
    // CUDA_SYNC_CHECK();
  }
#endif

  bvh.d_nodes.resize(bs.numBuildNodes);
  auto &d_nodes = bvh.d_nodes;

#if GQL_USE_ITEM_LISTS
  auto &d_itemLists = bvh.d_itemLists;
  uint32_t *d_itemListMem = 0;
  bvh.d_itemLists.resize(numPrims);
  d_itemListMem = bvh.d_itemLists.get();
#endif
  clearNodeBoundsAndWriteParents<<<
      divRoundUp((int)bs.numBuildNodes * 4, blockSize), blockSize, 0, stream>>>(
      d_buildStats, d_buildNodes, bs.numBuildNodes);

  writePrimsAndLeafBounds<<<divRoundUp((int)numPrims, blockSize), blockSize, 0,stream>>>(
      d_buildNodes, d_nodes.get(), d_primStates, numPrims
#if GQL_USE_ITEM_LISTS
      ,
      d_itemListMem
#endif
  );

  buildInnerBounds<<<divRoundUp((int)bs.numBuildNodes * 4, blockSize),
                     blockSize, 0,stream>>>(d_buildNodes, bs.numBuildNodes);

  writeFinalNodes<<<divRoundUp((int)bs.numBuildNodes * 4, blockSize),
                    blockSize, 0,stream>>>(d_nodes.get(), d_buildNodes, bs.numBuildNodes);
  CUDA_SYNC_CHECK();

  bvh.d_bvh.nodes = bvh.d_nodes.get();
  bvh.d_bvh.numNodes = bvh.d_nodes.numElements;
  // bvh.d_bvh.parents   = d_parents.get();
#if GQL_USE_ITEM_LISTS
  bvh.d_bvh.itemLists = d_itemLists.get();
#endif
  // checkNodes(bvh.d_nodes,bs.numBuildNodes);

  cudaFreeAsync(d_buildStats, stream);
  cudaFreeAsync(d_buildNodes, stream);
  cudaFreeAsync(d_buildNodeActive, stream);
}

} // namespace bvhLib
} // namespace gequel
