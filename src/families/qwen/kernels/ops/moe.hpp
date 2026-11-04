#ifndef GUFO_FAMILIES_QWEN_KERNELS_OPS_MOE_HPP_
#define GUFO_FAMILIES_QWEN_KERNELS_OPS_MOE_HPP_

#include <cstddef>
#include <cstdint>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::hip {

/// The router of a mixture-of-experts feed-forward: softmax over the routed
/// experts, the selection of the `num_used` highest-probability ones, the
/// renormalization of their weights to sum one, and the shared expert's own
/// gate.
///
/// `router_logits` holds `num_experts` scores per row and `shared_logit` one
/// per row. Ties in the selection resolve toward the lower expert index, as
/// a stable sort would, so the CPU reference and the device agree on which
/// experts were picked.
///
/// `expert_ids` and `expert_weights` are [rows][num_used]; `shared_weight`
/// is [rows].
void LaunchMoeRouterTopK(const float* router_logits,
                         const float* shared_logit, std::int32_t* expert_ids,
                         float* expert_weights, float* shared_weight,
                         std::uint32_t num_experts, std::uint32_t num_used,
                         std::uint32_t rows, hipStream_t stream = nullptr);

/// Combines the routed experts and the shared expert into the feed-forward's
/// output.
///
/// `expert_out` is [rows][num_used][dim], `expert_weights` is
/// [rows][num_used], `shared_out` is [rows][dim] and `shared_weight` is
/// [rows]. The result is the weighted sum of the routed experts plus the
/// shared expert scaled by its gate.
void LaunchMoeCombine(const float* expert_out, const float* expert_weights,
                      const float* shared_out, const float* shared_weight,
                      float* out, std::uint32_t num_used, std::uint32_t dim,
                      std::uint32_t rows, hipStream_t stream = nullptr);

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_FAMILIES_QWEN_KERNELS_OPS_MOE_HPP_
