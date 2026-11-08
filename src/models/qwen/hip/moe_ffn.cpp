#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/moe_ffn.hpp"

#include <algorithm>
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

namespace {

/// The most tokens one gated routed projection takes at a time. Q4_K shares
/// each weight row across a group of rows; the other formats the projection
/// accepts stop lower. Zero means the format has no batched route at all.
/// These mirror the row limits the routed kernel enforces on itself.
[[nodiscard]] std::size_t GatedRouteRows(core::GgmlType type) noexcept {
  switch (type) {
    case core::GgmlType::kQ4_K:
      return 64;
    case core::GgmlType::kQ5_K:
    case core::GgmlType::kQ8_0:
      return 8;
    default:
      return 0;
  }
}

/// True when every stage of this layer has a batched route. The router's
/// weights are F32 or BF16, so they go through the batched dense kernel; a
/// format either of them does not handle leaves the run to the per-token
/// path rather than changing what the layer computes.
[[nodiscard]] bool BatchedMoeSupported(const models::QwenLayerWeights& layer,
                                       const QwenMoeScratch& scratch) noexcept {
  const auto dense_form = [](core::GgmlType type) {
    return type == core::GgmlType::kF32 || type == core::GgmlType::kBF16;
  };
  return dense_form(layer.ffn_gate_inp.type) &&
         dense_form(layer.ffn_gate_inp_shexp.type) &&
         GatedRouteRows(layer.ffn_gate_exps.type) > 0 &&
         !scratch.shared_gate.empty() && !scratch.shared_up.empty();
}

}  // namespace

void MoeFfnBatch(const models::QwenLayerWeights& layer,
                 const core::ModelConfig& config, const QwenMoeScratch& scratch,
                 std::size_t rows, const float* x, float* out,
                 hipStream_t stream) {
  if (rows == 0) {
    return;
  }
  if (!BatchedMoeSupported(layer, scratch)) {
    for (std::size_t row = 0; row < rows; ++row) {
      MoeFfnToken(layer, config, scratch, static_cast<std::uint32_t>(row),
                  x + row * config.hidden_size, out + row * config.hidden_size,
                  stream);
    }
    return;
  }

  const std::size_t experts = config.num_experts;
  const std::size_t used = config.num_experts_used;
  const std::size_t ff = config.expert_ff;
  const std::size_t hidden = config.hidden_size;
  const std::size_t shared_ff = config.shared_expert_ff;

  // The router scores every token in the run, then picks on the device.
  LaunchBatchedGEMM(
      layer.ffn_gate_inp.data, layer.ffn_gate_inp.type == core::GgmlType::kBF16,
      x, scratch.router_logits.data(), rows, experts, hidden, stream);
  LaunchBatchedGEMM(layer.ffn_gate_inp_shexp.data,
                    layer.ffn_gate_inp_shexp.type == core::GgmlType::kBF16, x,
                    scratch.shared_logit.data(), rows, 1, hidden, stream);
  LaunchMoeRouterTopK(
      scratch.router_logits.data(), scratch.shared_logit.data(),
      scratch.ids.data(), scratch.weights.data(), scratch.shared_weight.data(),
      static_cast<std::uint32_t>(experts), static_cast<std::uint32_t>(used),
      static_cast<std::uint32_t>(rows), stream);

  // The shared expert runs on every token, so the whole run is one pair of
  // projections, one activation and one down projection. The quant-block
  // route keeps the decode GEMV's accumulation order per row, so a token's
  // shared-expert output does not depend on what shared the launch with it.
  LaunchBatchedQuantGEMMFp32(
      layer.ffn_gate_shexp.type, layer.ffn_gate_shexp.data, x,
      scratch.shared_gate.data(), rows, shared_ff, hidden, stream);
  LaunchBatchedQuantGEMMFp32(layer.ffn_up_shexp.type, layer.ffn_up_shexp.data,
                             x, scratch.shared_up.data(), rows, shared_ff,
                             hidden, stream);
  LaunchBatchedSwiGLUActivation(
      scratch.shared_gate.data(), scratch.shared_up.data(),
      scratch.shared_act.data(), nullptr, rows * shared_ff, stream);
  LaunchBatchedQuantGEMMFp32(
      layer.ffn_down_shexp.type, layer.ffn_down_shexp.data,
      scratch.shared_act.data(), scratch.shared_out.data(), rows, hidden,
      shared_ff, stream);

  // The routed experts, a tile at a time. The down projection takes the
  // gate/up result as [rows * used] slot rows with one slot each: the layout
  // gate/up wrote it in, and the layout the selected ids are stored in.
  const std::size_t tile_rows = GatedRouteRows(layer.ffn_gate_exps.type);
  for (std::size_t begin = 0; begin < rows; begin += tile_rows) {
    const std::size_t count = std::min(tile_rows, rows - begin);
    auto* const tile_act = scratch.act.data() + begin * used * ff;
    if (qfn_mmq_moe_gated_vec(
            static_cast<int>(layer.ffn_gate_exps.type),
            layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
            x + begin * hidden, scratch.ids.data() + begin * used, tile_act,
            static_cast<int>(ff), static_cast<int>(hidden),
            static_cast<int>(count), static_cast<int>(experts),
            static_cast<int>(used), stream) != 0) {
      throw std::runtime_error("qwen35moe batched expert gate/up failed");
    }
    if (qfn_mmq_moe_vec(static_cast<int>(layer.ffn_down_exps.type),
                        layer.ffn_down_exps.data, tile_act,
                        scratch.ids.data() + begin * used,
                        scratch.down_out.data() + begin * used * hidden,
                        static_cast<int>(hidden), static_cast<int>(ff),
                        static_cast<int>(count * used),
                        static_cast<int>(experts), 1, stream, nullptr,
                        nullptr) != 0) {
      throw std::runtime_error("qwen35moe batched expert down failed");
    }
  }

  LaunchMoeCombine(scratch.down_out.data(), scratch.weights.data(),
                   scratch.shared_out.data(), scratch.shared_weight.data(), out,
                   static_cast<std::uint32_t>(used),
                   static_cast<std::uint32_t>(hidden),
                   static_cast<std::uint32_t>(rows), stream);
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
