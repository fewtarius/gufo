#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/moe_ffn.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "src/core/hip/hip_utils.hpp"
#include "src/families/qwen/kernels/ops/gemm.hpp"
#include "src/families/qwen/kernels/ops/moe.hpp"
#include "src/families/qwen/kernels/ops/swiglu.hpp"
#include "src/models/qwen/hip/profile.hpp"
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

/// The down projection's grouped matrix route, which shares one gather and
/// one tiled activation quantization across all of an expert's tokens. The
/// formats below are the ones its kernel instantiates; anything else keeps
/// the vector route.
[[nodiscard]] bool DownMatrixSupported(core::GgmlType type) noexcept {
  return type == core::GgmlType::kQ4_K || type == core::GgmlType::kQ5_K ||
         type == core::GgmlType::kQ5_1 || type == core::GgmlType::kQ8_0;
}

/// Below this many tokens the grouped matrix route cannot fill the device and
/// the vector route's per-expert grouping wins. It is a floor on work, not a
/// correctness bound: both routes compute the same projection.
constexpr std::size_t kMatrixRouteRows = 16;

/// True when the gate and up projections both have a grouped matrix route and
/// share a shape, which is what lets one activation buffer feed both. The
/// formats are held to Q4_K because that is what this geometry is verified
/// for; the dispatch itself would take the others.
[[nodiscard]] bool PairMatrixSupported(
    const models::QwenLayerWeights& layer) noexcept {
  return layer.ffn_gate_exps.type == core::GgmlType::kQ4_K &&
         layer.ffn_up_exps.type == core::GgmlType::kQ4_K &&
         layer.ffn_gate_exps.num_elements == layer.ffn_up_exps.num_elements;
}

/// Dispatches the routed projections' grouped matrix route by weight format.
int LaunchMatrixMoe(core::GgmlType type, const void* w, const float* x,
                    const std::int32_t* ids, float* out, int m, int k,
                    int tokens, int experts, int used, hipStream_t stream) {
  switch (type) {
    case core::GgmlType::kQ4_K:
      return qfn_mmq_q4_K_moe_raw(w, x, ids, out, m, k, tokens, experts, used,
                                  stream);
    case core::GgmlType::kQ5_K:
      return qfn_mmq_q5_K_moe_raw(w, x, ids, out, m, k, tokens, experts, used,
                                  stream);
    case core::GgmlType::kQ5_1:
      return qfn_mmq_q5_1_moe_raw(w, x, ids, out, m, k, tokens, experts, used,
                                  stream);
    case core::GgmlType::kQ8_0:
      return qfn_mmq_q8_0_moe_raw(w, x, ids, out, m, k, tokens, experts, used,
                                  stream);
    default:
      return -1;
  }
}

/// Column-tile width for the routed expert GEMMs: the smallest tile at or above
/// twice the mean bucket, so most experts finish in one tile with little
/// padding. Mirrors the Flash-Next executor's choice for the same kernel.
[[nodiscard]] int RoutedTileCols(std::size_t rows, std::size_t used,
                                 std::size_t experts) noexcept {
  const std::size_t mean =
      std::max<std::size_t>(1, rows * used / std::max<std::size_t>(experts, 1));
  for (int cols = 16; cols < 80; cols += 16) {
    if (static_cast<std::size_t>(cols) >= 2 * mean) {
      return cols;
    }
  }
  return 80;
}

/// Tell the routed matrix kernel how wide a bucket can be and how wide a
/// column tile should be. Without this the column grid spans every selected
/// slot and every excess tile still costs a dispatch and a full shared-memory
/// reservation. The bound is the row count because no expert can receive more
/// slots than there are tokens, which is the same bound the Flash-Next executor
/// uses when it has no measured counts.
void SetRoutedHints(std::size_t rows, std::size_t used,
                    std::size_t experts) noexcept {
  qfn_mmq_set_routed_max_expert_rows(static_cast<int>(rows));
  qfn_mmq_set_routed_tile_cols(RoutedTileCols(rows, used, experts));
}

/// Dense router projection.
///
/// The warp-per-element kernel this replaces gave each output element its own
/// warp and looped over K with no reuse across the batch, so an expert row was
/// fetched once per token. At 1024 rows, 256 experts and K of 2048 that is over
/// a gigabyte of weight traffic for a projection whose weights are two
/// megabytes, and it measured at about one percent of peak.
///
/// F32 weights go through BLAS instead, which is the same FP32 product with a
/// tiled schedule and therefore the same arithmetic; only the summation order
/// differs, which the router's top-k is insensitive to at this magnitude. BF16
/// weights keep the old kernel because BLAS would have to round the activations
/// to BF16 first, and the router's expert choice is not worth that risk.
void DenseRouterGemm(const models::QwenTensorRef& weight, const float* x,
                     float* out, std::size_t rows, std::size_t m, std::size_t k,
                     hipStream_t stream, hipblasHandle_t blas) {
  if (blas != nullptr && weight.type == core::GgmlType::kF32) {
    LaunchHipblasGEMM(blas, weight.data, /*is_bf16=*/false, x, out, rows, m, k,
                      /*d_x_bf16_buf=*/nullptr, stream);
    return;
  }
  LaunchBatchedGEMM(weight.data, weight.type == core::GgmlType::kBF16, x, out,
                    rows, m, k, stream);
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
                 hipStream_t stream, hipblasHandle_t blas) {
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

  auto& ffn_clock = FfnClock();
  ffn_clock.SetStream(stream);
  ffn_clock.Mark();

  // The router scores every token in the run, then picks on the device.
  DenseRouterGemm(layer.ffn_gate_inp, x, scratch.router_logits.data(), rows,
                  experts, hidden, stream, blas);
  DenseRouterGemm(layer.ffn_gate_inp_shexp, x, scratch.shared_logit.data(),
                  rows, 1, hidden, stream, blas);
  LaunchMoeRouterTopK(
      scratch.router_logits.data(), scratch.shared_logit.data(),
      scratch.ids.data(), scratch.weights.data(), scratch.shared_weight.data(),
      static_cast<std::uint32_t>(experts), static_cast<std::uint32_t>(used),
      static_cast<std::uint32_t>(rows), stream);
  ffn_clock.Mark();

  // The shared expert runs on every token, so the whole run is one pair of
  // projections, one activation and one down projection.
  //
  // Q8_0 puts all three on the matrix-core route: the activation is quantized
  // once into the tiled Q8_1 layout, the gate and up projections run as one
  // fused pass that shares the activation reads, the SwiGLU result is quantized
  // in that pass's epilogue, and the down projection reads it directly. The
  // quant-block route it replaces grouped sixteen rows per weight load, which
  // is a twentieth of the tiles' reuse and was a quarter of the layer's time.
  // Other formats keep that route: it preserves the decode GEMV's accumulation
  // order per row, which the matrix-core path does not.
  static const bool shared_wmma_enabled = [] {
    const char* const v = std::getenv("GUFO_SHARED_WMMA");
    return v == nullptr || *v != '0';
  }();
  const bool shared_wmma = shared_wmma_enabled &&
                           layer.ffn_gate_shexp.type == core::GgmlType::kQ8_0 &&
                           layer.ffn_up_shexp.type == core::GgmlType::kQ8_0 &&
                           layer.ffn_down_shexp.type == core::GgmlType::kQ8_0 &&
                           !scratch.shared_q8_in.empty() &&
                           !scratch.shared_q8_act.empty();
  if (shared_wmma) {
    LaunchQuantizeActivationQ8_1FromFp32(x, scratch.shared_q8_in.data(), rows,
                                         hidden, stream);
    LaunchBatchedDualQuantGEMMPreQuantized(
        layer.ffn_gate_shexp.type, layer.ffn_gate_shexp.data,
        layer.ffn_up_shexp.data, scratch.shared_q8_in.data(),
        scratch.shared_gate.data(), scratch.shared_up.data(), rows, shared_ff,
        hidden, stream);
    LaunchBatchedFusedSwiGLUQuantizeQ8_1(
        scratch.shared_gate.data(), scratch.shared_up.data(),
        scratch.shared_q8_act.data(), rows, shared_ff, stream);
    LaunchBatchedQuantGEMMPreQuantized(
        layer.ffn_down_shexp.type, layer.ffn_down_shexp.data,
        scratch.shared_q8_act.data(), scratch.shared_out.data(), rows, hidden,
        shared_ff, stream);
  } else {
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
  }

  ffn_clock.Mark();
  // The routed experts. From 16 tokens up the gate/up pair takes the grouped
  // matrix route, which gathers every token's activation once and runs one
  // matrix pass per expert over all of that expert's tokens. The vector route
  // stays for narrow runs and for the formats the matrix route lacks. The pair
  // runs as two passes, not the paired entry point: that one fixes the second
  // weight's ids bound at the token count, which does not match the slot-space
  // expert bounds, and it is order-of-magnitude slower for it.
  const std::size_t tile_rows = GatedRouteRows(layer.ffn_gate_exps.type);
  SetRoutedHints(rows, used, experts);
  const bool pair_matrix = rows >= kMatrixRouteRows &&
                           PairMatrixSupported(layer) &&
                           !scratch.up_out.empty();
  if (pair_matrix) {
    if (LaunchMatrixMoe(layer.ffn_gate_exps.type, layer.ffn_gate_exps.data, x,
                        scratch.ids.data(), scratch.act.data(),
                        static_cast<int>(ff), static_cast<int>(hidden),
                        static_cast<int>(rows), static_cast<int>(experts),
                        static_cast<int>(used), stream) != 0 ||
        LaunchMatrixMoe(layer.ffn_up_exps.type, layer.ffn_up_exps.data, x,
                        scratch.ids.data(), scratch.up_out.data(),
                        static_cast<int>(ff), static_cast<int>(hidden),
                        static_cast<int>(rows), static_cast<int>(experts),
                        static_cast<int>(used), stream) != 0) {
      throw std::runtime_error("qwen35moe matrix expert gate/up failed");
    }
    LaunchBatchedSwiGLUActivation(scratch.act.data(), scratch.up_out.data(),
                                  scratch.routed_act.data(), nullptr,
                                  rows * used * ff, stream);
  } else {
    for (std::size_t begin = 0; begin < rows; begin += tile_rows) {
      const std::size_t count = std::min(tile_rows, rows - begin);
      if (qfn_mmq_moe_gated_vec(
              static_cast<int>(layer.ffn_gate_exps.type),
              layer.ffn_gate_exps.data, layer.ffn_up_exps.data,
              x + begin * hidden, scratch.ids.data() + begin * used,
              scratch.routed_act.data() + begin * used * ff,
              static_cast<int>(ff), static_cast<int>(hidden),
              static_cast<int>(count), static_cast<int>(experts),
              static_cast<int>(used), stream) != 0) {
        throw std::runtime_error("qwen35moe batched expert gate/up failed");
      }
    }
  }
  ffn_clock.Mark();

  // The down projection reads the gate/up result as [rows][used][ff]. The
  // grouped matrix route takes the run in one pass when the format has one;
  // otherwise the vector route walks it a tile at a time, one slot per row.
  //
  // The matrix route takes the slot form - one row per selected expert with a
  // single expert each - because it derives each gathered row from the token
  // index alone. A token's slots are distinct feed-forward vectors, so the
  // source row of slot (t, j) is t * used + j, which is exactly the row index
  // when the slots are flattened.
  if (rows >= kMatrixRouteRows &&
      DownMatrixSupported(layer.ffn_down_exps.type)) {
    if (LaunchMatrixMoe(layer.ffn_down_exps.type, layer.ffn_down_exps.data,
                        scratch.routed_act.data(), scratch.ids.data(),
                        scratch.down_out.data(), static_cast<int>(hidden),
                        static_cast<int>(ff), static_cast<int>(rows * used),
                        static_cast<int>(experts), 1, stream) != 0) {
      throw std::runtime_error("qwen35moe matrix expert down failed");
    }
  } else {
    for (std::size_t begin = 0; begin < rows; begin += tile_rows) {
      const std::size_t count = std::min(tile_rows, rows - begin);
      if (qfn_mmq_moe_vec(static_cast<int>(layer.ffn_down_exps.type),
                          layer.ffn_down_exps.data,
                          scratch.routed_act.data() + begin * used * ff,
                          scratch.ids.data() + begin * used,
                          scratch.down_out.data() + begin * used * hidden,
                          static_cast<int>(hidden), static_cast<int>(ff),
                          static_cast<int>(count * used),
                          static_cast<int>(experts), 1, stream, nullptr,
                          nullptr) != 0) {
        throw std::runtime_error("qwen35moe batched expert down failed");
      }
    }
  }

  ffn_clock.Mark();
  LaunchMoeCombine(scratch.down_out.data(), scratch.weights.data(),
                   scratch.shared_out.data(), scratch.shared_weight.data(), out,
                   static_cast<std::uint32_t>(used),
                   static_cast<std::uint32_t>(hidden),
                   static_cast<std::uint32_t>(rows), stream);
  ffn_clock.Mark();
}

SubStageClock& FfnClock() {
  // Static so the marks span every layer of a chunk. Reporting per call would
  // synchronise once per layer, which drains the queue and measures the stall
  // rather than the work: the same stages read five times slower that way.
  static SubStageClock clock("GUFO_PROFILE_FFN");
  static const bool registered = [] {
    clock.AddStage("router");
    clock.AddStage("shared");
    clock.AddStage("routed-gate-up");
    clock.AddStage("routed-down");
    clock.AddStage("combine");
    return true;
  }();
  (void)registered;
  return clock;
}

void FlushFfnProfile() {
  FfnClock().Report("ffn");
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
