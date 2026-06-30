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
#include <nvtx3/nvToolsExt.h>

namespace gequel {
  namespace bvhLib {
    
    /*! a record that captures the state of a primitive during building */
    struct PrimState {
      bvh_boxf_t    bounds;
      uint32_t done: 1;
      uint32_t node:29;
      uint32_t sub : 2;
    };

    template<typename GeomData, typename getBoundsLambda>
    __global__ void lambdaFillBuildPrims(size_t numPrims,
                                         PrimState *d_prims,
                                         const GeomData geomData,
                                         const getBoundsLambda getBounds)
    {
      int gid = threadIdx.x + blockIdx.x*blockDim.x;
      if (gid >= numPrims) return;
      d_prims[gid].bounds = getBounds(geomData,gid);
    }
      
    void genericSpatialMedianBuilder(QBVH<float> &bvh,
                                     size_t numPrims,
                                     PrimState *primStates, cudaStream_t stream);
    
    template<typename T, typename GeomData, typename getBoundsLambda>
    QBVH<T> *spatialMedianBuilder(size_t numPrims,
                                  const GeomData &geomData,
                                  const getBoundsLambda &getBounds, cudaStream_t stream)
    {
      QBVH<T> *bvh = new QBVH<T>;
      
      nvtxRangePush("inside miniGQL builder");
      PrimState* d_prims;
      cudaMallocAsync((void**)&d_prims,sizeof(PrimState)*numPrims,stream);
      lambdaFillBuildPrims<<<divRoundUp(int(numPrims),128),128,0,stream>>>
        (numPrims,d_prims,geomData,getBounds);
      genericSpatialMedianBuilder(*bvh,numPrims,d_prims);
      cudaFreeAsync(d_prims,stream);
      nvtxRangePop();
      return bvh;
    }

  } // ::gequel::bvhLib
} // ::gequel
