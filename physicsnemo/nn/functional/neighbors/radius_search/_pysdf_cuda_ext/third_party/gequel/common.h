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

#include <stdexcept>
#include <cuda_runtime.h>
#include "owl/common/math/box.h"
#include <cuda_fp16.h>
#include <owl/helper/cuda.h>
#include <owl/common/math/vec.h>
#include <cuda_runtime.h>

#define GQL_NOTIMPLEMENTED throw std::runtime_error("not implemented: "+std::string(__PRETTY_FUNCTION__))

namespace gequel {
  namespace common {
    // using namespace owl::common;

    using owl::common::vec2f;
    using owl::common::vec3f;
    
    using owl::common::box2f;
    using owl::common::box3f;

    using owl::common::divRoundUp;

    using owl::common::min;
    using owl::common::max;
    using owl::common::infty;

    using owl::common::prettyDouble;
    using owl::common::prettyNumber;
    using owl::common::getCurrentTime;

    // -> this should arguably go into owl
    inline __device__ float clamp(float f, float lo=0.f, float hi=1.f)
    { return max(min(f,hi),lo); }

    // -> this should arguably go into owl
    
    
    template <typename T, int N>
    inline __both__ T dot(owl::vec_t<T, N> a, owl::vec_t<T, N> b);

    template <>
    inline __both__ float dot(owl::vec_t<float, 2> a, owl::vec_t<float, 2> b)
    { return a.x*b.x+a.y*b.y; }

    template <>
    inline __both__ int dot(owl::vec_t<int, 2> a, owl::vec_t<int, 2> b)
    { return a.x*b.x+a.y*b.y; }

    template <>
    inline __both__ float dot(owl::vec_t<float, 3> a, owl::vec_t<float, 3> b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
    template <>
    inline __both__ int dot(owl::vec_t<int, 3> a, owl::vec_t<int, 3> b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
    // -> this should arguably go into owl
    
    inline __both__ float length(vec2f a) { return sqrtf(dot(a,a)); }

    template <typename T>
    inline __device__ float sqr(owl::vec_t<T,2> v) { return dot(v,v); }

    template <typename T>
    inline __device__ float sqr(owl::vec_t<T,3> v) { return dot(v,v); }
    //    inline __device__ float infty() { return INFINITY; }
    // inline __device__ float sqr(const vec3f v) { return dot(v,v); }
    
#ifdef __CUDACC__

  // from rapids cpp/src_prims/stats/minmax.cuh
  template <class To, class From>
  inline __host__ __device__
  constexpr To bit_cast(const From& from) noexcept
  {
    To to{};
    static_assert(sizeof(To) == sizeof(From));
    memcpy(&to, &from, sizeof(To));
    return to;
  }

  template <typename T>
  struct encode_traits { };

  template <>
  struct encode_traits<float> { using E = int; };

  template <>
  struct encode_traits<double> { using E = long long; };

  inline __host__ __device__
  int encode(float val)
  {
    int i = bit_cast<int>(val);
    return i >= 0 ? i : (1 << 31) | ~i;
  }

  inline __host__ __device__
  long long encode(double val)
  {
    std::int64_t i = bit_cast<std::int64_t>(val);
    return i >= 0 ? i : (1ULL << 63) | ~i;
  }

  inline __host__ __device__
  float decode(int val)
  {
    if (val < 0) val = (1 << 31) | ~val;
    return bit_cast<float>(val);
  }

  inline __host__ __device__
  double decode(long long val)
  {
    if (val < 0) val = (1ULL << 63) | ~val;
    return bit_cast<double>(val);
  }

  template <typename T, typename E>
  inline __device__
  T atomicMaxBits(T* address, T val)
  {
    E old = atomicMax((E*)address, encode(val));
    return decode(old);
  }

  template <typename T, typename E>
  inline __device__
  T atomicMinBits(T* address, T val)
  {
    E old = atomicMin((E*)address, encode(val));
    return decode(old);
  }


    /*! perform atomic max on the memory locatoin indicated (with
        value indiceated); return true if the value was changed by
        this operation; and false if it was already set to a higher
        value by somebody else */
    inline __device__ bool setAtomicMaxBits_returnIfModified(
        typename encode_traits<float>::E *address, float val)
    {
      using E = encode_traits<float>::E;
      // Why cast? If address was a float*, then the cast here would be replaced with way more casts in spatialMedianBuilder.
        // Wanted the type of centBounds_or_finalBounds to reflect the data representation at rest. (encoded)
      float old = atomicMaxBits<float, E>(reinterpret_cast<float*>(address), val);
      return val > old;
    }
  
    /*! perform atomic max on the memory locatoin indicated (with
        value indiceated); return true if the value was changed by
        this operation; and false if it was already set to a higher
        value by somebody else */
    inline __device__ bool setAtomicMinBits_returnIfModified(
        typename encode_traits<float>::E *address, float val)
    {
      using E = encode_traits<float>::E;
      float old = atomicMinBits<float, E>(reinterpret_cast<float*>(address), val);
      return val < old;
    }

    inline __device__ void setAtomicMaxBits(
        typename encode_traits<float>::E *address, float val)
    {
      using E = encode_traits<float>::E;
      atomicMaxBits<float, E>(reinterpret_cast<float*>(address), val);
    }
    /*! atomically set memory region to the max of its value and 'val' - no return value */
    inline __device__ void setAtomicMinBits(
        typename encode_traits<float>::E *address, float val)
    {
      using E = encode_traits<float>::E;
      atomicMinBits<float, E>(reinterpret_cast<float*>(address), val);
    }

    // atomicCAS versions.
    /*! atomically set memory region to the max of its value and 'val' - no return value */
    inline __device__ void setAtomicMaxCAS(float* address, float val)
    {
      if (*address >= val) return;
#if 0
      ::atomicMax((int*)address,(int&)val);
#else
      int* address_as_i = (int*) address;
      int old = *address_as_i, assumed;
      do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __float_as_int(::fmaxf(val, __int_as_float(assumed))));
      } while (assumed != old);
#endif
    }
  
    /*! atomically set memory region to the max of its value and 'val' - no return value */
    inline __device__ void setAtomicMinCAS(float* address, float val)
    {
      if (*address <= val) return;
#if 0
      ::atomicMin((int*)address,(int&)val);
#else
      int* address_as_i = (int*) address;
      int old = *address_as_i, assumed;
      do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __float_as_int(::fminf(val, __int_as_float(assumed))));
      } while (assumed != old);
#endif
    }
#endif
  }
}

