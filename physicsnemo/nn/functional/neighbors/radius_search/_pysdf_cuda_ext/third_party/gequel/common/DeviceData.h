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

#include "gequel/common.h"

namespace gequel {

  using namespace gequel::common;

  template<typename T>
  struct DeviceData {
    inline DeviceData() {}
    inline DeviceData(size_t count) { resize(count); }
    inline DeviceData(T *ptr, size_t count) : d_ptr(ptr), numElements(count), owned(false) {}
    inline ~DeviceData() {free();}

    inline __both__ T *get() const { return (T*)d_ptr; }

    /*! returns number of bytes in this array */
    inline size_t numBytes() const { return numElements*sizeof(T); }

    void upload(const std::vector<T> &vec) { upload(vec.data(),vec.size()); }

    /*! uploads data from host to device; will alloc (more) memory if
        so required. if data _can_ be uploaded without reallocing, it
        will do so - so if the DeviceData currently points to a
        _non_owned memory regoin that's large enough, it willupload
        without changint his ownership */
    void upload(const T *h_data, size_t hostNumElements)
    {
      /* iw - leave this test in, it's importnat: we don't want to
         resize if we can already fit */
      if (hostNumElements > numElements)
          resize(hostNumElements);
      CUDA_CALL(Memcpy(d_ptr,h_data,numBytes(),cudaMemcpyDefault));
    }

    /*! upload() plus device synchronize */
    void upload_sync(const T *h_data, size_t hostNumElements)
    {
      upload(h_data,hostNumElements);
      CUDA_SYNC_CHECK();
    }
    
    /*! frees existing memory, and allocated aexactly as muc has
        requested. note this _will_ free and (re-)alloc even if
        existing memory would already be vailable and large enough */
    void resize(size_t newNumElements)
    {
      free();
      numElements = newNumElements;
      CUDA_CHECK(cudaMallocAsync(&d_ptr,numBytes(),0));
      owned = true;
    }
        
    void free()
    {
      if (!d_ptr) return;
        
      if (owned)
        CUDA_CHECK(cudaFreeAsync((void*)d_ptr,0));
      owned = false;
      numElements = 0;
      d_ptr = 0;
    }

    /*! pointer to device-side memory containing this data. This
        pointer must either be null, or be valid device-accessible
        memory (an be managed mem, host-pinned mem, etc but must be
        device accessible */
    T     *d_ptr       = 0;

    /*! number of elements (not bytes!) in the device-side memory. for
        non-owned data this value may be 0 even if d_ptr is non-null;
        for owned data this value must be 0 exactly if d_ptr is null,
        and if not 0, then d_ptr must have at least numElements
        device-readable elements */
    size_t numElements = 0;

    /*! if true, _we_ own this data, and will release it upon
        destruction; if false ,then we got this device-pointer assinge
        dby somebody else that owns it, and that is responsible for
        releasing it - _we_ will then not delete it. Careful: if
        non-owned, then it's the app's responsibility to ensure that
        d_ptr remains valid device-accessible memory while any kernel
        may use it */
    bool   owned       = 0;
  };
      
}
