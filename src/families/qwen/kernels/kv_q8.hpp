// Device-side q8_0 K/V accessors for the Qwen attention paths.
//
// The Qwen caches are token-major: one row per (layer, position) holding
// `kv_width` values, the KV heads side by side. The FP16 plane stores one
// binary16 per value; the q8_0 plane stores the same row as 32-element
// blocks, each a binary16 scale and 32 signed codes.
//
// Every helper here takes an absolute element index into the plane. The
// element's block is the index divided by 32 and its position inside the
// block is the index modulo 32.
//
// The numeric contract is the shared codec's; see docs/kv-codec.md and
// src/core/kv_codec/kv_q8_device.hpp.
#ifndef GUFO_FAMILIES_QWEN_KERNELS_KV_Q8_HPP_
#define GUFO_FAMILIES_QWEN_KERNELS_KV_Q8_HPP_

#include <cstddef>
#include <cstdint>

#include "src/core/kv_codec/kv_q8_device.hpp"

namespace gufo::hip::detail {

using Q8KvBlock = gufo::core::kv_codec::device::Q8KvBlock;

/// Blocks one row of `kv_width` values occupies.
[[nodiscard]] constexpr std::uint32_t Q8KvBlocksPerRow(
    std::uint32_t kv_width) noexcept {
  return gufo::core::kv_codec::device::Q8KvBlocksPerRow(kv_width);
}

/// Reconstructs one element as binary16.
///
/// The stored scale is the block's binary16 magnitude. The product is
/// computed in binary32 and rounded to binary16, which is the host codec's
/// decode contract.
[[nodiscard]] __device__ __forceinline__ __half
Q8KvHalf(const Q8KvBlock& block, std::uint32_t offset) noexcept {
  const float scale = __half2float(block.scale);
  return __float2half(static_cast<float>(block.values[offset]) * scale);
}

/// Reconstructs one element as binary32.
[[nodiscard]] __device__ __forceinline__ float Q8KvFloat(
    const Q8KvBlock& block, std::uint32_t offset) noexcept {
  return __half2float(Q8KvHalf(block, offset));
}

/// Four consecutive elements as four binary32 values.
///
/// The four elements share one block. Every Qwen attention route reads
/// vectors of four, and the vector bases are multiples of four, so the four
/// elements never straddle a block boundary.
[[nodiscard]] __device__ __forceinline__ float4
Q8KvLoadFloat4(const Q8KvBlock* base, std::size_t absolute) noexcept {
  const Q8KvBlock& block = base[absolute / 32];
  const std::uint32_t offset = static_cast<std::uint32_t>(absolute % 32);
  return make_float4(Q8KvFloat(block, offset), Q8KvFloat(block, offset + 1),
                     Q8KvFloat(block, offset + 2),
                     Q8KvFloat(block, offset + 3));
}

/// Two consecutive elements as one binary16 pair.
[[nodiscard]] __device__ __forceinline__ __half2
Q8KvLoadHalf2(const Q8KvBlock* base, std::size_t absolute) noexcept {
  const Q8KvBlock& block = base[absolute / 32];
  const std::uint32_t offset = static_cast<std::uint32_t>(absolute % 32);
  return __halves2half2(Q8KvHalf(block, offset), Q8KvHalf(block, offset + 1));
}

/// Four consecutive elements as two binary16 pairs.
///
/// The four elements share one block: every Qwen attention route reads
/// vectors of four, and their bases are multiples of four.
[[nodiscard]] __device__ __forceinline__ void Q8KvLoadHalf4(
    const Q8KvBlock* base, std::size_t absolute, __half2& lower,
    __half2& upper) {
  const Q8KvBlock& block = base[absolute / 32];
  const std::uint32_t offset = static_cast<std::uint32_t>(absolute % 32);
  lower = __halves2half2(Q8KvHalf(block, offset), Q8KvHalf(block, offset + 1));
  upper =
      __halves2half2(Q8KvHalf(block, offset + 2), Q8KvHalf(block, offset + 3));
}

/// Eight consecutive elements as eight binary16 values.
///
/// The eight elements share one block: the staged tiles read vectors of
/// eight, and their bases are multiples of eight.
[[nodiscard]] __device__ __forceinline__ uint4
Q8KvLoadHalf8(const Q8KvBlock* base, std::size_t absolute) {
  const Q8KvBlock& block = base[absolute / 32];
  const std::uint32_t offset = static_cast<std::uint32_t>(absolute % 32);
  const float scale = __half2float(block.scale);
  __half out[8];
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    out[i] = __float2half(static_cast<float>(block.values[offset + i]) * scale);
  }
  uint4 packed;
  __builtin_memcpy(&packed, out, sizeof(packed));
  return packed;
}

}  // namespace gufo::hip::detail

#endif  // GUFO_FAMILIES_QWEN_KERNELS_KV_Q8_HPP_
