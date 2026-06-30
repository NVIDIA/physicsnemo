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

#include "gequel/bvhLib/BVH.h"

namespace gequel {
  namespace bvhLib {

    /*! helper class that can be used to traverse a QBVH with a range
      query for a maximum (suqare) distance around a given query
      point. this query radius may be shrunk during traversal */
    template<typename T>
    struct QBVHPointRangeQuery {
      using boxtype = bvh_box_t<T>;
      using vectype = bvh_vec_t<T>;

      inline __device__
      QBVHPointRangeQuery(const typename QBVH<float>::DevRep &bvh,
                          const vectype P,
                          float newMaxRadius = infty());
      
      inline __device__ bool next(uint32_t &itemListStart,
                                  uint32_t &itemListSize,
                                  float    &leafSqrDist);

      /*! find the next leaf 'payload' in the BVH that is within the
        maximum query radius. if one can be found, sets's
        nextPayload to the value stored in the leaf node found
        during traversal, sets nextSqrDist to the (conservative)
        clamped square distance to that node, and returns true. if
        no more leaf node cna be found within the query radius, this
        returns false */
      inline __device__ bool next(uint32_t &nextPayload,
                                  float    &nextSqrDist);

      /*! same as 'next(payload,dist2)', except this silently drops the dist value */
      inline __device__ bool next(uint32_t &nextPayload);

      /*! set new maximum query radius; all nodes with (conservative)
        distance >= this radius will get culled. radius may not be
        NaN, and for correctness must be <= the previously used
        radius (ie, queries may only foreshorten that radius */
      inline __device__ void setMaxQueryRadius(float newMaxRadius);
      inline __device__ void setMaxQueryRadiusSquared(float newMaxRadius);
      
      /*! a 64-bit integer that encods in its upper 32 bits a distance
        (used for sorting), and in its lower 32bits a 'ref32' form a
        BVH node (ie, a handle/reference to a bvh quad-node). */
      struct ChildRef        { uint64_t bits; };
      
      /*! the four children of a node, in sorted order; sorting etc is
        hidden in the details/implementation seciton */
      struct SortedChildRefs { ChildRef child[4]; };
        
    private:
      /*! the BVH we are traversing */
      const typename QBVH<float>::DevRep &bvh;
      
      /*! the query point we are traversing for */
      const vectype P;

      /*! 'culling distance' - ie, distance at which nodes will no
        longer get considered for traveral - any node or subtree
        whose distance is guaranteed to be at least this value will
        get culled/skippe during traversal */
      float   cullSqrDist = infty();

      /*! @{ traversal state variables: */
      //! (quad-)node we need to traverse next
      int32_t thisNodeID = 0;
      
      //! (quad-)node we did our last traversal step in
      int32_t lastNodeID = -1;
      
      /*! _sub_ node within the (quad-)node we did our last traversal
        step in, in sorted child order. Ie, this value being 0 does
        not mean that we last asscessed node.child[0], but whichever
        of the node.child[]'s was closest to the query point */
      int32_t lastSubID  = 0;
      /*! @} */

      /*! our sorted children in the current/last step - cached between
        calls for reasons of speed; only valid if and when
        thisNodeID==lastNodeID */
      SortedChildRefs children;

#if GQL_USE_ITEM_LISTS
      uint32_t *leafListPtr = 0;
      uint32_t  leafListLen = 0;
      float     leafDist;
#endif
    };

    // ==================================================================
    // IMPLEMENTATION SECTION for float
    // ==================================================================

    namespace qbvhPointRangeQuery_impl {
      template <typename T>
      using ChildRef = typename QBVHPointRangeQuery<T>::ChildRef;
      template <typename T>
      using SortedChildRefs = typename QBVHPointRangeQuery<T>::SortedChildRefs;
      using boxtype = bvh_boxf_t;
      using vectype = bvh_vecf_t;

      enum { numPayloadBits = QBVH<float>::numPayloadBits };
      
      template<typename T>
      using vectypeT = bvh_vec_t<T>;

      template<typename T>
      inline __device__ float clampedSqrDist(const vectypeT<T> &P,
                                             const boxtype &box)
      {
        vectypeT<float> Pf(P);
        const vectype clampedP = min(max(Pf,box.lower),box.upper);
        return sqr(Pf-clampedP);
      }

      template<typename T>
      inline __device__ uint32_t getPayload(ChildRef<T> child) 
      { return child.bits & ((1<<numPayloadBits)-1); }
      
      template<typename T>
      inline __device__ uint32_t getChildID(ChildRef<T> child)
      { return child.bits; }

      template<typename T>
      inline __device__ T getSqrDist(ChildRef<T> child);

      template<>
      inline __device__ float getSqrDist(ChildRef<float> child)
      { return __int_as_float(child.bits >> 32); }

      template<typename T>
      inline __device__ uint32_t getNumPrimsInLeaf(ChildRef<T> child) 
      { return uint32_t(child.bits) >> numPayloadBits; }

      template<>
      inline __device__ int getSqrDist(ChildRef<int> child)
      { return (child.bits >> 32); }

      template<typename T>
      inline __device__ bool isLeaf(ChildRef<T> child) 
      { return uint32_t(child.bits) >> numPayloadBits; }
      // { return child.bits & (1u<<31); }

      template<typename T>
      inline __device__ bool inRange(ChildRef<T> child, T dist2) 
      { return getSqrDist<T>(child) < dist2; }
      
      template <typename T>
      inline __device__
      ChildRef<T> make_ChildRef(const QBVH<float>::SubNode &node,
                             const vectypeT<T> &P)
      {
        const float dist2
          = node.isValid()
          ? clampedSqrDist(P,node.bounds)
          : infty();
        const uint32_t lower32 = (uint32_t&)node.ref32;
        const uint32_t upper32 = __float_as_int(dist2);
#if 1
	uint64_t result;
	asm("mov.b64 %0,{%1,%2}; \n\t" : "=l"(result) : "r"(lower32), "r"(upper32));
	return { result };
#else
	return { lower32 + (uint64_t(upper32)<<32) };
#endif
      }
        
      
      template <typename T>
      inline __device__ void sort_swap(ChildRef<T> &a, ChildRef<T> &b)
      {
        const uint64_t lo = min(a.bits,b.bits);
        const uint64_t hi = max(a.bits,b.bits);
        a.bits = lo;
        b.bits = hi;
      }
      
      
      template <typename T>
      inline __device__ int findIndex(SortedChildRefs<T> &children, uint32_t nodeID)
      {
#if 1
#pragma unroll
        for (int i=0;i<3;i++)
          if (getChildID<T>(children.child[i]) == nodeID) return i;
        return 3;
#else
        for (int i=0;i<4;i++)
          if (getChildID<T>(children.child[i]) == nodeID) {
            return i;
          }
        return -1;
#endif
      }    
        
      template <typename T>
      inline __device__
      ChildRef<T> get(SortedChildRefs<T> &children, int idx)
      {
        if (idx == 0) return children.child[0];
        if (idx == 1) return children.child[1];
        if (idx == 2) return children.child[2];
        return children.child[3];
      }
        
      /*! yes, i know... bitonic sort might just shave off one or
        two more min/maxes... */
      template <typename T>
      inline __device__
      void sort(SortedChildRefs<T> &children)
      {
        sort_swap<T>(children.child[0],children.child[1]);
        sort_swap<T>(children.child[1],children.child[2]);
        sort_swap<T>(children.child[2],children.child[3]);

        sort_swap<T>(children.child[0],children.child[1]);
        sort_swap<T>(children.child[1],children.child[2]);

        sort_swap<T>(children.child[0],children.child[1]);
      }

      template <typename T>
      inline __device__
      SortedChildRefs<T> computeSortedChildRefs(const QBVH<float>::DevRep &qbvh,
                                             uint32_t thisNodeID,
                                             const vectypeT<T> P)
      {
        SortedChildRefs<T> children;
        const QBVH<float>::QuadNode &node = qbvh.nodes[thisNodeID];
        for (int i=0;i<4;i++) 
          children.child[i] = make_ChildRef(node.child[i],P);
        
        sort<T>(children);
        return children;
      }
    }
    
    template<>
    inline __device__
    QBVHPointRangeQuery<float>::QBVHPointRangeQuery(const QBVH<float>::DevRep &bvh,
                                                    const vectype P,
                                                    float newMaxRadius)
      : bvh(bvh),
        P(P),
        cullSqrDist(newMaxRadius*newMaxRadius),
        thisNodeID( bvh.nodes ? 0 : -1) {}

    /*! set new maximum query radius; all nodes with (conservative)
      distance >= this radius will get culled. radius may not be
      NaN, and for correctness must be <= the previously used
      radius (ie, queries may only foreshorten that radius */
    template<typename T>
    inline __device__ void QBVHPointRangeQuery<T>::setMaxQueryRadius(float newMaxRadius)
    { cullSqrDist = newMaxRadius*newMaxRadius; }
    
    template<typename T>
    inline __device__ void QBVHPointRangeQuery<T>::setMaxQueryRadiusSquared(float newMaxRadiusSqr)
    { cullSqrDist = newMaxRadiusSqr; }
    
    /*! find the next leaf 'payload' in the BVH that is within the
      maximum query radius. if one can be found, sets's
      nextPayload to the value stored in the leaf node found
      during traversal, sets nextSqrDist to the (conservative)
      clamped square distance to that node, and returns true. if
      no more leaf node cna be found within the query radius, this
      returns false */
    template<typename T>
    inline __device__
    bool QBVHPointRangeQuery<T>::next(uint32_t &nextPayload,
                                      float    &nextSqrDist)
    {
      using namespace qbvhPointRangeQuery_impl;

#if GQL_USE_ITEM_LISTS
      if (leafListLen) {
        nextPayload = *leafListPtr++;
        nextSqrDist = leafDist;
        leafListLen--;
        return true;
      }
#endif      
      while (thisNodeID >= 0) {
        const bool fromChild  = lastNodeID > thisNodeID;
        const bool fromParent = lastNodeID < thisNodeID;

        if (fromChild | fromParent)
          children = computeSortedChildRefs(bvh,thisNodeID,P);
        
        /* the 'sub' ID is the index in the (sorted) list of (up
           to) four child nodes of the current multi-node. I.e.,
           subID==0 means the first child in that (sorted) list. */
        const int thisSubID
          = (fromChild
             ? (/* on way back we always go to next child _after_ the
                   node that spawned us */ findIndex<T>(children,lastNodeID)+1)
             : (fromParent
                ?/*parents always go straight to first (sorted) child:*/0
                :/*siblings go to _next_ (sorted) child*/(lastSubID+1)
                )
             );
             
        lastNodeID = thisNodeID;

        if (thisSubID == 4) {
          // no more siblings to go to - go to parent
          thisNodeID = bvh.nodes[thisNodeID].child[0].inChild0_parentID;
          continue;
        }
        
        /*! get required child; if nextSubID is an invalid index
          this will autmatically return a invalid child */
        const ChildRef child = get<T>(children,thisSubID);
        if (!inRange<T>(child,cullSqrDist)) {
          /* current child is not a valid next traversal node -
             either the sub index is now out of range (ie, no more
             children in this node), or the current child's distance
             is already out of query range (in which case all other
             children in this node iwll be out of range, too)
             .... either way, there's nothing more in this node (and
             thus, not in any subtree, either) to traverse - let's
             back up to parent */
          thisNodeID = bvh.nodes[thisNodeID].child[0].inChild0_parentID;
          continue;
        } else {
          lastSubID = thisSubID;
          /* we do now have a valid child (in the sense of one of
             the four child nodes of the four-wide multi-node) -
             check if that's a leaf node or a inner node */
          if (isLeaf<T>(child)) {
            /* if the current child is a leaf, return it to the
               caller - all the current traversal state (lastSubID
               and lastNodeID) is saved in this class, so safe to
               return and lose teporaries */
#if GQL_USE_ITEM_LISTS
            if (bvh.itemLists) {
              // hardcoded path for item lists
              leafListPtr = bvh.itemLists + getPayload<T>(child);
              leafListLen = getNumPrimsInLeaf<T>(child);
              leafDist    = getSqrDist<T>(child);
              
              nextSqrDist = leafDist;
              nextPayload = *leafListPtr++;
              leafListLen--;
              return true;
            } 
#endif
            // this code works only for BVHes *witout* item lists
            nextPayload = getPayload<T>(child);
            nextSqrDist = getSqrDist<T>(child);
            return true;
          } else {
            thisNodeID = getChildID<T>(child);
            continue;
          }
        }
      }
      return false;
    }

    
    template<typename T>
    inline __device__
    bool QBVHPointRangeQuery<T>::next(uint32_t &nextPayload)
    {
      float ignoreDist;
      return next(nextPayload,ignoreDist);
    }
    
  } // ::gequel::device
} // ::gequel
