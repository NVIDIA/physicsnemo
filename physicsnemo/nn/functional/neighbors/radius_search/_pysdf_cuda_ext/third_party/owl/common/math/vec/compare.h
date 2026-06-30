// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
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

namespace owl {
  namespace common {

    // ------------------------------------------------------------------
    // ==
    // ------------------------------------------------------------------

#if __CUDACC__
    template<typename T>
    inline __both__ bool operator==(const vec_t<T,2> &a, const vec_t<T,2> &b)
    { return (a.x==b.x) & (a.y==b.y); }
  
    template<typename T>
    inline __both__ bool operator==(const vec_t<T,3> &a, const vec_t<T,3> &b)
    { return (a.x==b.x) & (a.y==b.y) & (a.z==b.z); }
  
    template<typename T>
    inline __both__ bool operator==(const vec_t<T,4> &a, const vec_t<T,4> &b)
    { return (a.x==b.x) & (a.y==b.y) & (a.z==b.z) & (a.w==b.w); }
#else
    template<typename T>
    inline __both__ bool operator==(const vec_t<T,2> &a, const vec_t<T,2> &b)
    { return a.x==b.x && a.y==b.y; }

    template<typename T>
    inline __both__ bool operator==(const vec_t<T,3> &a, const vec_t<T,3> &b)
    { return a.x==b.x && a.y==b.y && a.z==b.z; }

    template<typename T>
    inline __both__ bool operator==(const vec_t<T,4> &a, const vec_t<T,4> &b)
    { return a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w; }
#endif
  
    // ------------------------------------------------------------------
    // !=
    // ------------------------------------------------------------------
  
    template<typename T, int N>
    inline __both__ bool operator!=(const vec_t<T,N> &a, const vec_t<T,N> &b)
    { return !(a==b); }


    // ------------------------------------------------------------------
    // comparison operators returning result _vector_
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // not (!)
    // ------------------------------------------------------------------

    template<typename T>
    inline __both__ auto nt(const vec_t<T,2> &a)
      -> vec_t<decltype(!a.x),2>
    { return { !a.x, !a.y }; }

    template<typename T>
    inline __both__ auto nt(const vec_t<T,3> &a)
      -> vec_t<decltype(!a.x),3>
    { return { !a.x, !a.y, !a.z }; }

    template<typename T>
    inline __both__ auto nt(const vec_t<T,4> &a)
      -> vec_t<decltype(!a.x),4>
    { return { !a.x, !a.y, !a.z, !a.w }; }

    // ------------------------------------------------------------------
    // eq (==)
    // ------------------------------------------------------------------

    template<typename T>
    inline __both__ auto eq(const vec_t<T,2> &a, const vec_t<T,2> &b)
      -> vec_t<decltype(a.x==b.x),2>
    { return { a.x==b.x, a.y==b.y }; }

    template<typename T>
    inline __both__ auto eq(const vec_t<T,3> &a, const vec_t<T,3> &b)
      -> vec_t<decltype(a.x==b.x),3>
    { return { a.x==b.x, a.y==b.y, a.z==b.z }; }

    template<typename T>
    inline __both__ auto eq(const vec_t<T,4> &a, const vec_t<T,4> &b)
      -> vec_t<decltype(a.x==b.x),4>
    { return { a.x==b.x, a.y==b.y, a.z==b.z, a.w==b.w }; }

    // ------------------------------------------------------------------
    // neq (!=)
    // ------------------------------------------------------------------

    template<typename T, int N>
    inline __both__ auto neq(const vec_t<T,N> &a, const vec_t<T,N> &b)
      -> decltype(nt(eq(a,b)))
    { return nt(eq(a,b)); }

    // ------------------------------------------------------------------
    // lt (<)
    // ------------------------------------------------------------------

    template<typename T>
    inline __both__ auto lt(const vec_t<T,2> &a, const vec_t<T,2> &b)
      -> vec_t<decltype(a.x<b.x),2>
    { return { a.x<b.x, a.y<b.y }; }

      template<typename T>
      inline __both__ auto lt(const vec_t<T,3> &a, const vec_t<T,3> &b)
        -> vec_t<decltype(a.x<b.x),3>
      { return { a.x<b.x, a.y<b.y, a.z<b.z }; }

        template<typename T>
        inline __both__ auto lt(const vec_t<T,4> &a, const vec_t<T,4> &b)
          -> vec_t<decltype(a.x<b.x),4>
        { return { a.x<b.x, a.y<b.y, a.z<b.z, a.w<b.w }; }

        // ------------------------------------------------------------------
        // le (<=)
        // ------------------------------------------------------------------

          template<typename T, int N>
          inline __both__ auto le(const vec_t<T,N> &a, const vec_t<T,N> &b)
            -> decltype(nt(lt(b,a)))
          { return nt(lt(b,a)); }

    // ------------------------------------------------------------------
    // gt (>)
    // ------------------------------------------------------------------

    template<typename T, int N>
    inline __both__ auto gt(const vec_t<T,N> &a, const vec_t<T,N> &b)
      -> decltype(lt(b,a))
    { return lt(b,a); }

    // ------------------------------------------------------------------
    // ge (>=)
    // ------------------------------------------------------------------

    template<typename T, int N>
    inline __both__ auto ge(const vec_t<T,N> &a, const vec_t<T,N> &b)
      -> decltype(nt(lt(a,b)))
    { return nt(lt(a,b)); }

    // ------------------------------------------------------------------
    // reduce
    // ------------------------------------------------------------------

    template<typename T, int N>
    inline __both__ bool any(const vec_t<T,N> &a)
    { for (int i=0;i<N;++i) if (a[i]) return true; return false; }

    template<typename T, int N>
    inline __both__ bool all(const vec_t<T,N> &a)
    { for (int i=0;i<N;++i) if (!a[i]) return false; return true; }

    // template<typename T>
    // inline __both__ bool any(const vec_t<T,3> &a)
    // { return a[i] | b[i] | c[i]; }

    // ------------------------------------------------------------------
    // select
    // ------------------------------------------------------------------

    template<typename T>
    inline __both__ vec_t<T,2> select(const vec_t<bool,2> &mask,
                                      const vec_t<T,2> &a,
                                      const vec_t<T,2> &b)
    { return { mask.x?a.x:b.x, mask.y?a.y:b.y }; }

    template<typename T>
    inline __both__ vec_t<T,3> select(const vec_t<bool,3> &mask,
                                      const vec_t<T,3> &a,
                                      const vec_t<T,3> &b)
    { return { mask.x?a.x:b.x, mask.y?a.y:b.y, mask.z?a.z:b.z }; }

    template<typename T>
    inline __both__ vec_t<T,4> select(const vec_t<bool,4> &mask,
                                      const vec_t<T,4> &a,
                                      const vec_t<T,4> &b)
    { return { mask.x?a.x:b.x, mask.y?a.y:b.y, mask.z?a.z:b.z }; }

    template<typename T, int N>
    inline __both__ vec_t<T,N> select(const vec_t<bool,N> &mask,
                                      const vec_t<T,N> &a,
                                      const vec_t<T,N> &b)
    {
      vec_t<T,N> res;
      for (int i=0; i<N; ++i)
        res[i] = mask[i]?a[i]:b[i];
      return res;
    }
  
  } // ::owl::common
} // ::owl
