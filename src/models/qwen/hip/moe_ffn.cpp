#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/moe_ffn.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/families/qwen/kernels/ops/gemm.hpp"
#include "src/families/qwen/kernels/ops/moe.hpp"
#include "src/families/qwen/kernels/ops/swiglu.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::hip {

void MoeFfnToken(const models::QwenLayerWeights& layer,
                 const core::ModelConfig& config, const QwenMoeScratch& scratch,
                 std::uint32_t row, const float* x, float* out,
                 hipStream_t stream) {
  const int experts = static_cast<int>(config.num_experts);
  const int used = static_cast<int>(config.num_experts_used);
  const int ff = static_cast<int>(config.expert_ff);
  const int hidden = static_cast<int>(config.hidden_size);
  const int shared_ff = static_cast<int>(config.shared_expert_ff);

  // The scratch is laid out one row per token or verification row.
  float* router_logits = scratch.router_logits.data() + (row * experts);
  float* shared_logit = scratch.shared_logit.data() + row;
  std::int32_t* ids = scratch.ids.data() + (row * used);
  float* weights = scratch.weights.data() + (row * used);
  float* shared_weight = scratch.shared_weight.data() + row;
  float* act = scratch.act.data() + (row * used * ff);
  float* down_out = scratch.down_out.data() + (row * used * hidden);
  float* shared_act = scratch.shared_act.data() + (row * shared_ff);
  float* shared_out = scratch.shared_out.data() + (row * hidden);

  // The router's scores and the shared expert's own gate.
  LaunchGEMV(layer.ffn_gate_inp.data, layer.ffn_gate_inp.type, x, router_logits,
             experts, hidden, stream);
  LaunchGEMV(layer.ffn_gate_inp_shexp.data, layer.ffn_gate_inp_shexp.type, x,
             shared_logit, 1, hidden, stream);

  // The selection, on the device: the softmax, the top-k and the
  // renormalization of the selected weights.
  LaunchMoeRouterTopK(router_logits, shared_logit, ids, weights, shared_weight,
                      static_cast<std::uint32_t>(experts),
                      static_cast<std::uint32_t>(used), 1, stream);

  // The routed experts: the gate and up of each selected expert, then its
  // down projection.
  if (qfn_mmq_moe_gated_vec(static_cast<int>(layer.ffn_gate_exps.type),
                            layer.ffn_gate_exps.data, layer.ffn_up_exps.data, x,
                            ids, act, ff, hidden, 1, experts, used,
                            stream) != 0) {
    throw std::runtime_error("qwen35moe expert gate/up projection failed");
  }
  if (qfn_mmq_moe_vec(static_cast<int>(layer.ffn_down_exps.type),
                      layer.ffn_down_exps.data, act, ids, down_out, hidden, ff,
                      used, experts, 1, stream, nullptr, nullptr) != 0) {
    throw std::runtime_error("qwen35moe expert down projection failed");
  }

  // The shared expert runs on every token.
  LaunchFusedSwiGLUGEMV(layer.ffn_gate_shexp.data, layer.ffn_gate_shexp.type,
                        layer.ffn_up_shexp.data, layer.ffn_up_shexp.type, x,
                        shared_act, shared_ff, hidden, stream);
  LaunchGEMV(layer.ffn_down_shexp.data, layer.ffn_down_shexp.type, shared_act,
             shared_out, hidden, shared_ff, stream);

  // The routed sum plus the gated shared expert.
  LaunchMoeCombine(down_out, weights, shared_out, shared_weight, out,
                   static_cast<std::uint32_t>(used),
                   static_cast<std::uint32_t>(hidden), 1, stream);
}

void WarmMoeDecode(const models::QwenModelWeights& weights,
                   const core::ModelConfig& config,
                   const QwenMoeScratch& scratch, const float* x,
                   hipStream_t stream) {
  const int experts = static_cast<int>(config.num_experts);
  const int used = static_cast<int>(config.num_experts_used);
  const int ff = static_cast<int>(config.expert_ff);
  const int hidden = static_cast<int>(config.hidden_size);
  HIP_CHECK(hipMemsetAsync(scratch.ids.data(), 0,
                           scratch.ids.size() * sizeof(std::int32_t), stream));
  for (const auto& layer : weights.layers) {
    (void)qfn_mmq_moe_gated_vec(
        static_cast<int>(layer.ffn_gate_exps.type), layer.ffn_gate_exps.data,
        layer.ffn_up_exps.data, x, scratch.ids.data(), scratch.act.data(), ff,
        hidden, 1, experts, used, stream);
    (void)qfn_mmq_moe_vec(static_cast<int>(layer.ffn_down_exps.type),
                          layer.ffn_down_exps.data, scratch.act.data(),
                          scratch.ids.data(), scratch.down_out.data(), hidden,
                          ff, used, experts, 1, stream, nullptr, nullptr);
  }
  HIP_CHECK(hipStreamSynchronize(stream));
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
