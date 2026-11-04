#ifndef GUFO_MODELS_QWEN_HIP_MOE_FFN_HPP_
#define GUFO_MODELS_QWEN_HIP_MOE_FFN_HPP_

#include "src/core/model_config.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/state.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// The mixture-of-experts feed-forward for one token.
///
/// The router picks `num_experts_used` of the `num_experts` routed experts
/// on the device; their outputs are scaled by the renormalized router
/// probability and summed with the shared expert's, scaled by the sigmoid
/// of its own gate.
///
/// `x` and `out` are the single token's normalized input and the feed-
/// forward's output. `scratch` holds the router's logits, the selected
/// experts and their outputs; `row` is the scratch row this token uses.
void MoeFfnToken(const models::QwenLayerWeights& layer,
                 const core::ModelConfig& config, const QwenMoeScratch& scratch,
                 std::uint32_t row, const float* x, float* out,
                 hipStream_t stream);

/// Runs every layer's expert kernels once outside a graph capture, so the
/// matrix library's workspace pool is sized before the decode graph is
/// captured.
void WarmMoeDecode(const models::QwenModelWeights& weights,
                   const core::ModelConfig& config,
                   const QwenMoeScratch& scratch, const float* x,
                   hipStream_t stream);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_MOE_FFN_HPP_
