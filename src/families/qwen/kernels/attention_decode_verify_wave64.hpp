#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_ATTENTION_DECODE_VERIFY_WAVE64_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_ATTENTION_DECODE_VERIFY_WAVE64_HPP_

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "src/families/qwen/hip_common/execution_policy.hpp"

namespace gufo::hip::detail {

// Split-K attention partials for verification rows > 1, packaged as
// three wave64 per pair of KV heads. Requires num_heads == 6 * num_kv_heads,
// an even num_kv_heads and head_dim 256. Writes the scratch layout of
// QwenDecodeSplitKAttentionHalfPartialsKernel<true, 16> with the same
// per-head arithmetic, so QwenDecodeSplitKAttentionReduceKernel follows.
//
// The canonical cache is staged into binary16 tiles whatever its format, so
// the per-head arithmetic is the FP16 one.
template<QwenKvFormat Format>
void LaunchVerifySplitKAttentionHalfPartialsWave64(
    const float* q, const void* k_cache, const void* v_cache, float* scratch,
    std::uint32_t layer_idx, std::uint32_t start_pos, unsigned rows,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t splits, hipStream_t stream);

}  // namespace gufo::hip::detail
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_ATTENTION_DECODE_VERIFY_WAVE64_HPP_
