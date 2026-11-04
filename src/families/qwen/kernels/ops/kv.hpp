#ifndef GUFO_MODELS_QWEN_HIP_OPS_KV_HPP_
#define GUFO_MODELS_QWEN_HIP_OPS_KV_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// Encodes K and V rows into the q8_0 K/V cache.
///
/// `source_k` and `source_v` hold `rows` rows of `kv_width` values each, in
/// the token-major order the cache uses. Row `r` of the source becomes row
/// `start_position + r` of `layer` in the cache.
///
/// The encoder is the device half of the host codec: one binary16 scale and
/// 32 signed codes per 32 values, ties away from zero, divided by the
/// unrounded binary32 scale. See docs/kv-codec.md.
void LaunchEncodeQ8KvRows(const float* source_k, const float* source_v,
                          void* k_cache_q8, void* v_cache_q8,
                          std::uint32_t layer_idx, std::uint32_t start_position,
                          std::uint32_t rows, std::uint32_t max_context,
                          std::uint32_t kv_width, hipStream_t stream = nullptr);

/// The same, with the start position read from device memory.
///
/// The decode path keeps its position on the device so the captured graph
/// does not depend on a host value.
void LaunchEncodeQ8KvRowsAtDevicePosition(
    const float* source_k, const float* source_v, void* k_cache_q8,
    void* v_cache_q8, std::uint32_t layer_idx, const std::uint32_t* d_position,
    std::uint32_t rows, std::uint32_t max_context, std::uint32_t kv_width,
    hipStream_t stream = nullptr);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_OPS_KV_HPP_
