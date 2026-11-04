// Element access into the canonical Qwen K/V cache.
//
// The FP16 plane stores one binary16 per value. The q8_0 plane stores 32
// values per block, each block a binary16 scale and 32 signed codes; a
// value is reconstructed by multiplying its code by the block's scale and
// rounding the product to binary16, which is the shared codec's decode
// contract.
//
// A KvElements names the plane and the element its indices are relative to.
// The relative index of a vector's first element and the plane's base
// together decide which block holds it.
//
// See docs/kv-codec.md.
#ifndef GUFO_FAMILIES_QWEN_KERNELS_KV_READ_HPP_
#define GUFO_FAMILIES_QWEN_KERNELS_KV_READ_HPP_

#include "src/families/qwen/hip_common/execution_policy.hpp"
#include "src/families/qwen/kernels/kv_q8.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::hip::detail {

/// The canonical K/V cache plane, addressed by element index.
template<QwenKvFormat Format>
struct KvElements {
  const void* base{nullptr};
  /// The element the relative indices start from.
  std::size_t offset{0};

  /// One element as binary32.
  [[nodiscard]] __device__ __forceinline__ float Load(
      std::size_t element) const {
    const std::size_t absolute = offset + element;
    if constexpr (Format == QwenKvFormat::kFp16) {
      return __half2float(static_cast<const __half*>(base)[absolute]);
    } else {
      const Q8KvBlock& block =
          static_cast<const Q8KvBlock*>(base)[absolute / 32];
      return Q8KvFloat(block, static_cast<std::uint32_t>(absolute % 32));
    }
  }

  /// Four consecutive elements as four binary32 values.
  ///
  /// The four elements share one q8_0 block: every Qwen attention route
  /// reads vectors of four, and their bases are multiples of four.
  [[nodiscard]] __device__ __forceinline__ float4 Load4(
      std::size_t element) const {
    const std::size_t absolute = offset + element;
    if constexpr (Format == QwenKvFormat::kFp16) {
      const auto* pairs =
          reinterpret_cast<const __half2*>(static_cast<const __half*>(base) +
                                           absolute);
      const float2 lower = __half22float2(pairs[0]);
      const float2 upper = __half22float2(pairs[1]);
      return make_float4(lower.x, lower.y, upper.x, upper.y);
    } else {
      return Q8KvLoadFloat4(static_cast<const Q8KvBlock*>(base), absolute);
    }
  }

  /// Two consecutive elements as one binary16 pair.
  [[nodiscard]] __device__ __forceinline__ __half2 LoadHalf2(
      std::size_t element) const {
    const std::size_t absolute = offset + element;
    if constexpr (Format == QwenKvFormat::kFp16) {
      return reinterpret_cast<const __half2*>(static_cast<const __half*>(base) +
                                              absolute)[0];
    } else {
      return Q8KvLoadHalf2(static_cast<const Q8KvBlock*>(base), absolute);
    }
  }

  /// Eight consecutive elements as eight binary16 values.
  ///
  /// The eight elements share one q8_0 block: the staged tiles read vectors
  /// of eight, and their bases are multiples of eight.
  [[nodiscard]] __device__ __forceinline__ uint4 LoadHalf8(
      std::size_t element) const {
    const std::size_t absolute = offset + element;
    if constexpr (Format == QwenKvFormat::kFp16) {
      return *reinterpret_cast<const uint4*>(static_cast<const __half*>(base) +
                                             absolute);
    } else {
      return Q8KvLoadHalf8(static_cast<const Q8KvBlock*>(base), absolute);
    }
  }
};

/// The canonical cache plane.
template<QwenKvFormat Format>
[[nodiscard]] __device__ __forceinline__ KvElements<Format> KvPlane(
    const void* base) {
  return {.base = base, .offset = 0};
}

/// The canonical cache plane, with the element indices starting at
/// `offset`.
template<QwenKvFormat Format>
[[nodiscard]] __device__ __forceinline__ KvElements<Format> KvPlaneAt(
    const void* base, std::size_t offset) {
  return {.base = base, .offset = offset};
}

}  // namespace gufo::hip::detail

#endif  // GUFO_FAMILIES_QWEN_KERNELS_KV_READ_HPP_
