// ======================================================================== //
// Copyright 2018-2020 Ingo Wald                                            //
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

#include "owl/common/owl-common.h"
#include "owl/common/math/constants.h"
#include <iostream>
#include <math.h>
#include <algorithm>

namespace owl {
  namespace common {

    template<typename T> struct long_type_of { typedef T type; };
    template<> struct long_type_of<int32_t>  { typedef int64_t  type; };
    template<> struct long_type_of<uint32_t> { typedef uint64_t type; };
  
    template<typename T, int N>
    struct OWL_INTERFACE vec_t { T t[N]; };


    template<typename ScalarTypeA, typename ScalarTypeB> struct BinaryOpResultType;

    // Binary Result type: scalar type with itself always returns same type
    template<typename ScalarType>
    struct BinaryOpResultType<ScalarType,ScalarType> { typedef ScalarType type; };

    template<> struct BinaryOpResultType<int,float> { typedef float type; };
    template<> struct BinaryOpResultType<float,int> { typedef float type; };
    template<> struct BinaryOpResultType<unsigned int,float> { typedef float type; };
    template<> struct BinaryOpResultType<float,unsigned int> { typedef float type; };

    template<> struct BinaryOpResultType<int,double> { typedef double type; };
    template<> struct BinaryOpResultType<double,int> { typedef double type; };
    template<> struct BinaryOpResultType<unsigned int,double> { typedef double type; };
    template<> struct BinaryOpResultType<double,unsigned int> { typedef double type; };
  
    // ------------------------------------------------------------------
    // vec1 - not really a vector, but makes a scalar look like a
    // vector, so we can use it in, say, box1f
    // ------------------------------------------------------------------
    template<typename T>
    struct OWL_INTERFACE vec_t<T,1> {
      enum { dims = 1 };
      typedef T scalar_t;
    
      inline __both__ vec_t() {}
      inline __both__ vec_t(const T &v) : v(v) {}

      /*! assignment operator */
      inline __both__ vec_t<T,1> &operator=(const vec_t<T,1> &other) {
        this->v = other.v;
        return *this;
      }
    
      /*! construct 2-vector from 2-vector of another type */
      template<typename OT>
        inline __both__ explicit vec_t(const vec_t<OT,1> &o) : v(o.v) {}
    
      inline __both__ T &operator[](size_t dim) {
        assert(dim == 0);
        return x;
        // return (&x)[dim];
      }
      inline __both__ const T &operator[](size_t dim) const
      {
        assert(dim == 0);
        return x;
        // return (&x)[dim];
      }

      union {
        T v;
        T x; //!< just to allow all vec types to use x,y,z,w,...
      };
    };
 
    // ------------------------------------------------------------------
    // vec2
    // ------------------------------------------------------------------
    template<typename T>
    struct OWL_INTERFACE vec_t<T,2> {
      enum { dims = 2 };
      typedef T scalar_t;
    
      inline __both__ vec_t() {}
      inline __both__ vec_t(const T &t) : x(t), y(t) {}
      inline __both__ vec_t(const T &x, const T &y) : x(x), y(y) {}
#ifdef __CUDACC__
      inline __both__ vec_t(const float2 v) : x(v.x), y(v.y) {}
      inline __both__ vec_t(const int2 v) : x(v.x), y(v.y) {}
      inline __both__ vec_t(const uint2 v) : x(v.x), y(v.y) {}
    
      inline __both__ operator float2() const { return make_float2(x,y); }
      inline __both__ operator int2() const { return make_int2(x,y); }
      inline __both__ operator uint2() const { return make_uint2(x,y); }

      /*! auto-convert to cuda dim3 type, so we can use vec2is as
        paramters to kernel launches */
      inline __both__ operator dim3() const { dim3 d; d.x = x; d.y = y; d.z = 1; return d; }
      inline explicit __both__ vec_t(const dim3 v) : x(v.x), y(v.y) {}
#endif

      /*! assignment operator */
      inline __both__ vec_t<T,2> &operator=(const vec_t<T,2> &other) {
        this->x = other.x;
        this->y = other.y;
        return *this;
      }
    
      /*! construct 2-vector from 2-vector of another type */
      template<typename OT>
        inline __both__ explicit vec_t(const vec_t<OT,2> &o) : x((T)o.x), y((T)o.y) {}
    
      inline __both__ T &operator[](size_t dim) { return (&x)[dim]; }
      inline __both__ const T &operator[](size_t dim) const { return (&x)[dim]; }
    
      union {
        struct { T x, y; };
        struct { T s, t; };
        struct { T u, v; };
      };
    };

    // ------------------------------------------------------------------
    // vec3
    // ------------------------------------------------------------------
    template<typename T>
    struct OWL_INTERFACE vec_t<T,3> {
      enum { dims = 3 };
      typedef T scalar_t;
    
      inline // __both__
        vec_t(const vec_t &) = default;
      inline __both__ vec_t() {}
      inline __both__ vec_t(const T &t) : x(t), y(t), z(t) {}
      inline __both__ vec_t(const T &_x, const T &_y, const T &_z) : x(_x), y(_y), z(_z) {}
#ifdef __CUDACC__
      inline __both__ vec_t(const int3 &v)   : x(v.x), y(v.y), z(v.z) {}
      inline __both__ vec_t(const uint3 &v)  : x(v.x), y(v.y), z(v.z) {}
      inline __both__ vec_t(const float3 &v) : x(v.x), y(v.y), z(v.z) {}
      /*! initialize from a float4 - use an actual copy rather than
          const-ref here to force nvcc to actually do the full float4
          load if tihs is from memory */
      inline __both__ vec_t(const float4 v)  : x(v.x), y(v.y), z(v.z) {}
      /*! initialize from a int4 - use an actual copy rather than
          const-ref here to force nvcc to actually do the full float4
          load if tihs is from memory */
      inline __both__ vec_t(const int4 v)  : x(v.x), y(v.y), z(v.z) {}
      /*! initialize from a uint4 - use an actual copy rather than
          const-ref here to force nvcc to actually do the full float4
          load if tihs is from memory */
      inline __both__ vec_t(const uint4 v)  : x(v.x), y(v.y), z(v.z) {}
      inline __both__ operator float3() const { return make_float3(x,y,z); }
      inline __both__ operator int3() const { return make_int3(x,y,z); }
      inline __both__ operator uint3() const { return make_uint3(x,y,z); }
#endif
      inline __both__ explicit vec_t(const vec_t<T,4> &v);
      /*! construct 3-vector from 3-vector of another type */
      template<typename OT>
        inline __both__ explicit vec_t(const vec_t<OT,3> &o) : x((T)o.x), y((T)o.y), z((T)o.z) {}

      /*! swizzle ... */
      inline __both__ vec_t<T,3> yzx() const { return vec_t<T,3>(y,z,x); }
    
      /*! assignment operator */
      inline __both__ vec_t<T,3> &operator=(const vec_t<T,3> &other) {
        this->x = other.x;
        this->y = other.y;
        this->z = other.z;
        return *this;
      }
    
      inline __both__ T &operator[](size_t dim) { return (&x)[dim]; }
      inline __both__ const T &operator[](size_t dim) const { return (&x)[dim]; }

      template<typename OT, typename Lambda>
        static inline __both__ vec_t<T,3> make_from(const vec_t<OT,3> &v, const Lambda &lambda)
      { return vec_t<T,3>(lambda(v.x),lambda(v.y),lambda(v.z)); }
    
      union {
        struct { T x, y, z; };
        struct { T r, s, t; };
        struct { T u, v, w; };
      };
    };

    // ------------------------------------------------------------------
    // vec3a
    // ------------------------------------------------------------------
    template<typename T>
    struct OWL_INTERFACE OWL_ALIGN(16) vec3a_t : public vec_t<T,3> {
      inline vec3a_t() {}
      inline vec3a_t(const T &t) : vec_t<T,3>(t) {}
      inline vec3a_t(const T &x, const T &y, const T &z) : vec_t<T,3>(x,y,z) {}
#ifdef __CUDACC__
      inline __both__ vec3a_t(const int3 &v) : vec_t<T,3>(v) {};
      inline __both__ vec3a_t(const uint3 &v) : vec_t<T,3>(v) {};
      inline __both__ vec3a_t(const float3 &v) : vec_t<T,3>(v) {};
      inline __both__ vec3a_t(const int4 v) : vec_t<T,3>(v) {};
      inline __both__ vec3a_t(const uint4 v) : vec_t<T,3>(v) {};
      inline __both__ vec3a_t(const float4 v) : vec_t<T,3>(v) {};
#endif

      template<typename OT>
        inline vec3a_t(const vec_t<OT,3> &v) : vec_t<T,3>(v.x,v.y,v.z) {}
    
      T a;
      // add one elemnet for 'forced' alignment
    };
  
    // ------------------------------------------------------------------
    // vec4
    // ------------------------------------------------------------------
    template<typename T>
    struct OWL_INTERFACE vec_t<T,4> {
      enum { dims = 4 };
      typedef T scalar_t;
    
      inline __both__ vec_t() {}

      inline __both__ vec_t(const T &t)
        : x(t), y(t), z(t), w(t)
      {}
      inline __both__ vec_t(const vec_t<T,3> &xyz, const T &_w)
        : x(xyz.x), y(xyz.y), z(xyz.z), w(_w)
      {}
      inline __both__ vec_t(const T &_x, const T &_y, const T &_z, const T &_w)
        : x(_x), y(_y), z(_z), w(_w)
      {}
    
#ifdef __CUDACC__
      inline __both__ vec_t(const float4 &v)
        : x(v.x), y(v.y), z(v.z), w(v.w)
        {}
      inline __both__ vec_t(const int4 &v)
        : x(v.x), y(v.y), z(v.z), w(v.w)
        {}
      inline __both__ vec_t(const uint4 &v)
        : x(v.x), y(v.y), z(v.z), w(v.w)
        {}
      inline __both__ operator float4() const { return make_float4(x,y,z,w); }
      inline __both__ operator uint4()  const { return make_uint4(x,y,z,w); }
      inline __both__ operator int4()   const { return make_int4(x,y,z,w); }
#endif
      /*! construct 3-vector from 3-vector of another type */
      template<typename OT>
        inline __both__ explicit vec_t(const vec_t<OT,4> &o)
        : x((T)o.x), y((T)o.y), z((T)o.z), w((T)o.w)
        {}
      inline __both__ vec_t(const vec_t<T,4> &o) : x(o.x), y(o.y), z(o.z), w(o.w) {}

      /*! assignment operator */
      inline __both__ vec_t<T,4> &operator=(const vec_t<T,4> &other) {
        this->x = other.x;
        this->y = other.y;
        this->z = other.z;
        this->w = other.w;
        return *this;
      }
    
      inline __both__ T &operator[](size_t dim) { return (&x)[dim]; }
      inline __both__ const T &operator[](size_t dim) const { return (&x)[dim]; }

      template<typename OT, typename Lambda>
        static inline __both__ vec_t<T,4> make_from(const vec_t<OT,4> &v,
                                                    const Lambda &lambda)
      { return vec_t<T,4>(lambda(v.x),lambda(v.y),lambda(v.z),lambda(v.w)); }
    
      T x, y, z, w;
    };

    template<typename T>
    inline __both__ vec_t<T,3>::vec_t(const vec_t<T,4> &v)
      : x(v.x), y(v.y), z(v.z)
    {}

    // =======================================================
    // default functions
    // =======================================================

    template<typename T>
    inline __both__ typename long_type_of<T>::type area(const vec_t<T,2> &v)
    { return (typename long_type_of<T>::type)(v.x)*(typename long_type_of<T>::type)(v.y); }

  
    template<typename T>
    inline __both__ typename long_type_of<T>::type volume(const vec_t<T,3> &v)
    { return
        (typename long_type_of<T>::type)(v.x)*
        (typename long_type_of<T>::type)(v.y)*
        (typename long_type_of<T>::type)(v.z);
    }

    template<typename T>
    inline __both__ typename long_type_of<T>::type volume(const vec_t<T,4> &v)
    { return
        (typename long_type_of<T>::type)(v.x)*
        (typename long_type_of<T>::type)(v.y)*
        (typename long_type_of<T>::type)(v.z)*
        (typename long_type_of<T>::type)(v.w);
    }

    template<typename T>
    inline __both__ typename long_type_of<T>::type area(const vec_t<T,3> &v)
    { return
        T(2)*((typename long_type_of<T>::type)(v.x)*v.y+
              (typename long_type_of<T>::type)(v.y)*v.z+
              (typename long_type_of<T>::type)(v.z)*v.x);
    }



    /*! vector cross product */
    template<typename T>
    inline __both__ vec_t<T,3> cross(const vec_t<T,3> &a, const vec_t<T,3> &b)
    {
      return vec_t<T,3>(a.y*b.z-b.y*a.z,
                        a.z*b.x-b.z*a.x,
                        a.x*b.y-b.x*a.y);
    }

    /*! vector cross product */
    template<typename T>
    inline __both__ T dot(const vec_t<T,3> &a, const vec_t<T,3> &b)
    {
      return a.x*b.x + a.y*b.y + a.z*b.z;
    }

    /*! vector cross product */
    template<typename T>
    inline __both__ vec_t<T,3> normalize(const vec_t<T,3> &v)
    {
      return v * owl::common::polymorphic::rsqrt(dot(v,v));
    }

    /*! vector cross product */
    template<typename T>
    inline __both__ T length(const vec_t<T,3> &v)
    {
      return owl::common::polymorphic::sqrt(dot(v,v));
    }

    template<typename T>
    inline __owl_host std::ostream &operator<<(std::ostream &o, const vec_t<T,1> &v)
    {
      o << "(" << v.x << ")";
      return o;
    }
  
    template<typename T>
    inline __owl_host std::ostream &operator<<(std::ostream &o, const vec_t<T,2> &v)
    {
      o << "(" << v.x << "," << v.y << ")";
      return o;
    }
  
    template<typename T>
    inline __owl_host std::ostream &operator<<(std::ostream &o, const vec_t<T,3> &v)
    {
      o << "(" << v.x << "," << v.y << "," << v.z << ")";
      return o;
    }

    template<typename T>
    inline __owl_host std::ostream &operator<<(std::ostream &o, const vec_t<T,4> &v)
    {
      o << "(" << v.x << "," << v.y << "," << v.z <<  "," << v.w << ")";
      return o;
    }

    // =======================================================
    // default instantiations
    // =======================================================
  
#define _define_vec_types(T,t)                  \
    using vec2##t = vec_t<T,2>;                 \
    using vec3##t = vec_t<T,3>;                 \
    using vec4##t = vec_t<T,4>;                 \
    using vec3##t##a = vec3a_t<T>;              \
  
    _define_vec_types(bool ,b);
    _define_vec_types(int8_t ,c);
    _define_vec_types(int16_t ,s);
    _define_vec_types(int32_t ,i);
    _define_vec_types(int64_t ,l);
    _define_vec_types(uint8_t ,uc);
    _define_vec_types(uint16_t,us);
    _define_vec_types(uint32_t,ui);
    _define_vec_types(uint64_t,ul);
    _define_vec_types(float,f);
    _define_vec_types(double,d);
  
#undef _define_vec_types
  
  } // ::owl::common

  using namespace owl::common;
} // ::owl

// comparison operators
#include "vec/compare.h"
#include "vec/functors.h"
#include "vec/rotate.h"

