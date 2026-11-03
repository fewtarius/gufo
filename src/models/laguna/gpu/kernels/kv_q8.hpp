// Device-side q8_0 KV storage for the Laguna attention path.
//
// The layout matches the host codec's BlockQ8_0 exactly: one binary16 scale
// followed by 32 signed codes, 34 bytes per block, no inter-block padding.
// The host codec in src/core/kv_codec/ is the reference for the numeric
// contract; this header is the device half of it.
//
// See docs/kv-codec.md.
#ifndef GUFO_MODELS_LAGUNA_GPU_KERNELS_KV_Q8_HPP_
#define GUFO_MODELS_LAGUNA_GPU_KERNELS_KV_Q8_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>

#include <cstddef>
#include <cstdint>

namespace gufo::models::laguna::gpu::rocm::detail {

/// One q8_0 KV block: a binary16 scale and 32 signed codes.
struct Q8KvBlock {
  __half scale;
  std::int8_t values[32];
};
static_assert(sizeof(Q8KvBlock) == 34);

/// Bytes one row of `elements` values occupies in q8_0.
[[nodiscard]] constexpr std::size_t Q8KvRowBytes(
    std::uint32_t elements) noexcept {
  return static_cast<std::size_t>(elements / 32) * sizeof(Q8KvBlock);
}

/// Largest finite binary16 value. A scale above this is not representable.
constexpr float kQ8MaxHalf = 65504.0F;

/// The scale for a block whose largest magnitude is `amax`.
///
/// The host codec rejects a scale that overflows binary16. The device
/// cannot reject anything, so it saturates instead: the magnitude is capped
/// at the largest finite binary16, which keeps every reconstruction finite
/// at the cost of resolution for inputs that are absurdly large. Real K/V
/// values are orders of magnitude below that.
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
  if (scale == 0.0F)
    return 0;
  const float scaled = value / scale;
  const float truncated = truncf(scaled);
  const float fraction = scaled - truncated;
  if (fraction >= 0.5F)
    return static_cast<int>(truncated) + 1;
  if (fraction <= -0.5F)
    return static_cast<int>(truncated) - 1;
  return static_cast<int>(truncated);
}

/// Clamps a code to the q8_0 range.
[[nodiscard]] __host__ __device__ inline std::int8_t Q8KvClamp(
    int code) noexcept {
  if (code > 127)
    return 127;
  if (code < -127)
    return -127;
  return static_cast<std::int8_t>(code);
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

}  // namespace gufo::models::laguna::gpu::rocm::detail

#endif  // GUFO_MODELS_LAGUNA_GPU_KERNELS_KV_Q8_HPP_
