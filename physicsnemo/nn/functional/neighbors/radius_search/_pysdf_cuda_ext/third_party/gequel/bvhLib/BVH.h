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

#pragma once

#include "gequel/common/DeviceData.h"

#define __align(a) OWL_ALIGN(a)

// Keeping these using statements outside the class.
// At the moment, do not want to template this type alias on T.
// Many other places in code currently hardcoded with floats when interacting with BVH.
// Properly templating everything on T is outside scope of 2D/3D option optimization.
#ifdef BVH_IS_3D
  template<typename T>
  using bvh_vec_t = owl::common::vec_t<T, 3>;

#else
  template<typename T>
  using bvh_vec_t = owl::common::vec_t<T, 2>;

#endif

template<typename T>
using bvh_box_t = owl::common::box_t<bvh_vec_t<T> >;

using bvh_vecf_t = bvh_vec_t<float>;
using bvh_boxf_t = bvh_box_t<float>;
using bvh_veci_t = bvh_vec_t<int>;
using bvh_boxi_t = bvh_box_t<int>;

namespace gequel {
  namespace bvhLib {

    /*! if enabled, leaves contain 'item lists', i.e., they can
        contain anything from 1 to 7 primitives (to be decided by the
        builder); this will produce fewer BVH nodes, and in theory
        fewer traversal steps - but also significantly 'bigger' leaves
        (in terms of BBox), so higher chance that certain prims will
        get encountered by a traverser even though the particular
        prim's bbox iteslf couldn't even touch the query
        box/region. If item lists are disabled each leaf will contain
        exactly one prim, and the leaf's bbox will be pretty much
        exactly the same as the bbox of that prim, meaning fewer
        'false positive' prims show up during traversal .... but at
        cost of significantly larger BVH, "probably" higher build
        time, and possibly many more traversal steps */
#ifndef GQL_USE_ITEM_LISTS
#define GQL_USE_ITEM_LISTS 0
#endif

// NOTE: relatinoship between GQL_ITEM_LISTS_MAX_PRIMS and numLeafCounterBits and numPrims member in the ref32 struct below.
#ifndef GQL_ITEM_LISTS_MAX_PRIMS
#define GQL_ITEM_LISTS_MAX_PRIMS 14 // Runs out of memory when going to 15 (spatialMedianBuilder.cu memory warning.)
#endif
    
    template<typename T>
    struct QBVH {

      enum { numLeafCounterBits = 4,
             numPayloadBits = 32-numLeafCounterBits };
      
      // fwd-def
      struct DevRep;

      QBVH() {}
    private:
      // disable copying:
      inline QBVH(const QBVH &);
      inline QBVH &operator=(const QBVH &);
    public:
      inline ~QBVH() {
      }
      /*! return a device-representatoin that clearly describes a
          'invalid' BVH (eg, for 0 elemnet inputs */
      static __both__ DevRep invalid()
      { return { nullptr,0
#if GQL_USE_ITEM_LISTS
                 ,nullptr
#endif
        }; }

      /*! one of the four 'nodes' in a multi-node */
      struct __align(16) SubNode {
        inline __device__ __host__ bool isValid() const
        { return bounds.lower.x <= bounds.upper.x; }
        
        inline __device__ __host__ bool isLeaf()  const
        { return ref32.numPrims != 0; }

        inline __device__ __host__ int getPayload()  const
        { return ref32.payload; }

        inline __device__ void makeLeaf(const uint32_t payload,
                                        const uint32_t leafCounter=1)  
        { /* iw - not sure that's required, i think box getc cleared, anyway */
          ref32.payload  = payload;
          ref32.numPrims = leafCounter;
        }

        inline __device__ void makeInner(const uint32_t childNodeID)  
        { /* iw - not sure that's required, i think box getc cleared, anyway */
          ref32.payload = childNodeID;
          ref32.numPrims = 0;
        }

        inline __device__ void makeInvalid()  
        { /* iw - not sure that's required, i think box gets cleared, anyway */
          bounds = bvh_boxf_t();
        }
        
        //0..24B:
        bvh_boxf_t bounds;

        /* 24..28: the use of these four bytes depends on which of the
           children it is ... tihs allows for storing certain
           per-_multi_node data (such as parentID, an int for
           refitting, etc) across the four children without affecting
           their alignment (the cleaner solution would be have each
           child be four bytes less in size, and explicitly add the
           required fields to the parent .... but that'd mess with the
           float4 alignment of the child nodes) */
        union {
          /*! subnodes should never use this value, it is owned by the
            parent multi-node */
          uint32_t doNotUse_ownedByParent;
          /*! in these bits in child[0], the parent stores the
            parentID (so we don't need a separate parents[] array */
          uint32_t inChild0_parentID;
          /*! in these bits in child[1], the parent keeps an int that
            the builder an use for tracking which of the children
            have already been refitted (so the last one to refit can
            keep on going to the parent) */
          uint32_t inChild1_numChildrenRefitted;
        };
        // 28..32B
        struct {
          uint32_t payload    : 28;//numPayloadBits;
          /*! number of primitives in this leaf, if a leaf, or 0 if
            inner node. For BVHes that store data directly in the
            payload, this value shuld be set to 1 for leaves */
          uint32_t numPrims   : 4; //numLeafCounterBits;
        } ref32;
      };
      
      struct __align(64) QuadNode {
        SubNode child[4];
      };

      /*! device-side representation w plain pointers etc */
      struct DevRep {
        QuadNode *nodes     = 0;
        int32_t   numNodes  = 0;
#if GQL_USE_ITEM_LISTS
        /*! array of node parent indices - one entry per (quad) node, if present */
        // int32_t  *parents   = 0;
        uint32_t *itemLists = 0;
#endif
      };
      
      /*! get device-side representation that device side kernels
        can/will operate on */
      inline DevRep get() const { return d_bvh; }

      DeviceData<QuadNode> d_nodes;
#if GQL_USE_ITEM_LISTS
      DeviceData<uint32_t> d_itemLists;
#endif      
      /*! the actual device representation - just (device-)pinter(s),
        and node count */
      DevRep d_bvh;
    };
  } // gequel::bvhLib
  using bvhLib::QBVH;
} // 

