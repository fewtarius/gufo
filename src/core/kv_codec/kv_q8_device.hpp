// Device-side q8_0 primitives for the attention K/V cache.
//
// The host codec in src/core/kv_codec/codec.hpp is the reference for the
// numeric contract; this header is its device half. The two agree byte for
// byte: the encoder rounds ties away from zero, divides by the unrounded
// binary32 scale, stores the scale as binary16, and the decoder rounds the
// product to binary16 before promoting it.
//
// See docs/kv-codec.md.
#ifndef GUFO_CORE_KV_CODEC_KV_Q8_DEVICE_HPP_
#define GUFO_CORE_KV_CODEC_KV_Q8_DEVICE_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>

#include <cstddef>
#include <cstdint>

namespace gufo::core::kv_codec::device {

/// One q8_0 K/V block: a binary16 scale and 32 signed codes.
struct Q8KvBlock {
  __half scale;
  std::int8_t values[32];
};
static_assert(sizeof(Q8KvBlock) == 34);
static_assert(offsetof(Q8KvBlock, values) == 2);

/// Elements per quantized block.
inline constexpr std::uint32_t kQ8BlockElements = 32;

/// Bytes one row of `elements` values occupies in q8_0.
[[nodiscard]] constexpr std::size_t Q8KvRowBytes(
    std::uint32_t elements) noexcept {
  return static_cast<std::size_t>(elements / kQ8BlockElements) *
         sizeof(Q8KvBlock);
}

/// Blocks one row of `elements` values occupies in q8_0.
[[nodiscard]] constexpr std::uint32_t Q8KvBlocksPerRow(
    std::uint32_t elements) noexcept {
  return elements / kQ8BlockElements;
}

/// Largest finite binary16 value. A scale above this is not representable.
inline constexpr float kQ8MaxHalf = 65504.0F;

/// The scale for a block whose largest magnitude is `amax`.
///
/// The host codec rejects a scale that overflows binary16. The device cannot
/// reject anything, so it saturates instead: the magnitude is capped at the
/// largest finite binary16, which keeps every reconstruction finite at the
/// cost of resolution for inputs that are absurdly large. Real K/V values are
/// orders of magnitude below that.
[[nodiscard]] __host__ __device__ inline float Q8KvScale(float amax) noexcept {
  const float capped = amax < kQ8MaxHalf ? amax : kQ8MaxHalf;
  return capped / 127.0F;
}

/// The canonical ggml q8_0 rounding primitive: ties away from zero.
///
/// `roundf` is not used because the hardware float-to-int convert rounds to
/// nearest-even, which is a different rounding and changes tie bytes.
[[nodiscard]] __host__ __device__ inline int Q8KvCode(float value,
                                                      float scale) noexcept {
  if (scale == 0.0F) {
    return 0;
  }
  const float scaled = value / scale;
  const float truncated = truncf(scaled);
  const float fraction = scaled - truncated;
  if (fraction >= 0.5F) {
    return static_cast<int>(truncated) + 1;
  }
  if (fraction <= -0.5F) {
    return static_cast<int>(truncated) - 1;
  }
  return static_cast<int>(truncated);
}

/// Clamps a code to the q8_0 range.
[[nodiscard]] __host__ __device__ inline std::int8_t Q8KvClamp(
    int code) noexcept {
  if (code > 127) {
    return 127;
  }
  if (code < -127) {
    return -127;
  }
  return static_cast<std::int8_t>(code);
}

/// Reconstructs one element of a block.
///
/// The stored scale is the block's binary16 magnitude. The product is
/// computed in binary32 and rounded to binary16, which is what the host
/// codec's decode contract says.
[[nodiscard]] __host__ __device__ inline float Q8KvDequantize(
    const Q8KvBlock& block, std::uint32_t offset) noexcept {
  const float scale = __half2float(block.scale);
  return __half2float(
      __float2half(static_cast<float>(block.values[offset]) * scale));
}

/// Reconstructs eight codes from one q8_0 block into eight halves.
///
/// `offset` is the index of the first code and must be a multiple of eight.
///
/// The scale is the block's binary16 magnitude. The product is computed in
/// binary32 and rounded to binary16, which is what the host codec's decode
/// contract says. A small integer times a binary16 scale is exactly
/// representable in binary32, so the single rounding to binary16 is the same
/// one the host performs.
[[nodiscard]] __host__ __device__ inline uint4 DequantizeQ8Group(
    const Q8KvBlock* block, std::uint32_t offset) {
  const float scale = __half2float(block->scale);
  __half out[8];
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    const float value = static_cast<float>(block->values[offset + i]) * scale;
    out[i] = __float2half(value);
  }
  uint4 packed;
  __builtin_memcpy(&packed, out, sizeof(packed));
  return packed;
}

#if defined(__HIPCC__)
/// Largest magnitude across a warp.
[[nodiscard]] __device__ inline float WarpMaxAbs(float value) noexcept {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor(value, offset));
  }
  return value;
}
#endif  // defined(__HIPCC__)

}  // namespace gufo::core::kv_codec::device

#endif  // GUFO_CORE_KV_CODEC_KV_Q8_DEVICE_HPP_
