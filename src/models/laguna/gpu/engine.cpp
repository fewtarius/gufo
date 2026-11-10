#include "src/models/laguna/gpu/engine.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <vector>

#include "src/core/hip/profile.hpp"
#include "src/models/laguna/gpu/device_model.hpp"
#include "src/models/laguna/gpu/kernels/kernels.hpp"
#include "src/models/laguna/gpu/kernels/kv_q8.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::laguna::gpu {
namespace {

// KV cache bytes for a session of `context` tokens. The sliding-window
// layers hold a ring of sliding_window + 2*tile rows; the full-attention
// layers hold one row per token. Both the K and the V cache store one
// half per element.
std::size_t SessionBytesForContextLocked(const Config& config,
                                         std::uint32_t context,
                                         std::uint32_t tile,
                                         KvCacheType kv_type) {
  const std::size_t kv_row =
      static_cast<std::size_t>(config.num_kv) * config.head_dim;
  const std::size_t row_bytes =
      kv_type == KvCacheType::kQ8_0
          ? rocm::detail::Q8KvRowBytes(static_cast<std::uint32_t>(kv_row))
          : kv_row * sizeof(__half);
  std::size_t bytes = 0;
  for (std::uint32_t il = 0; il < config.num_layers; ++il) {
    std::uint32_t rows = context;
    if (config.IsSwa(il)) {
      rows = std::min(context, config.sliding_window + 2 * tile);
    }
    bytes += 2 * row_bytes * rows;
  }
  return bytes;
}

constexpr std::uint32_t kVecBatch = 8;
// DFlash verify batches are n<=Session::kMaxVerifyRows (16) tokens, used=10
// -> rows up to 160. The underlying qfn_mmq_moe_vec/mul_mat_vec_moe_gated
// kernels chunk larger batches internally (mmvq_moe_max_batch groups) and
// assert tokens<=MMVQ_MAX_ROUTED_BATCH(64) - well above 16 - so this was a
// too-conservative proxy for "fits the vector path", tuned for n=1 decode
// only. Raised to let verify's small batches skip the wide/RoutedExperts
// path's host-sync overhead (see laguna-port-state.md); TEST EMPIRICALLY
// before trusting this further if expert_used or weight types change.
constexpr std::uint32_t kVecExpertRows = 160;
constexpr std::uint32_t kMaxTopK = 16;
constexpr std::uint32_t kMaxExperts = 256;
constexpr std::uint32_t kRoutePadRows = 16;
constexpr std::uint32_t kPairTileRows = 128;
constexpr std::uint32_t kPlainTileRows = 48;

// Token rows per routed gate/up tile. Kept at 128: narrowing to 64 measured
// slower, because each (expert, tile) block reads that expert's whole weight
// matrix, so the tile count is the weight-traffic multiplier. Overridable so
// one build still measures both arms.
std::uint32_t PairTileRows() {
  static const std::uint32_t value = [] {
    if (const char* const v = std::getenv("LAGUNA_PAIR_TILE")) {
      const int parsed = std::atoi(v);
      if (parsed == 64 || parsed == 128 || parsed == 256) {
        return static_cast<std::uint32_t>(parsed);
      }
    }
    return kPairTileRows;
  }();
  return value;
}

// F32 router weights through hipBLAS SGEMM. The tiled RouterGemmKernel reads
// the whole expert matrix once per 64-token tile, which at a 4096-row chunk is
// 64 re-reads of 3 MiB; SGEMM reads both operands once. Off by default: the
// summation order differs, and measured greedy output diverged on two of four
// probes, so the router's top-k is not insensitive enough to accept it without
// a quality evaluation.
bool RouterBlasEnabled() {
  static const bool value = [] {
    const char* const v = std::getenv("LAGUNA_ROUTER_BLAS");
    return v != nullptr && *v != '0';
  }();
  return value;
}
// 2026-10-30: experimenting with a 128-row Q8_0 down-projection tile via the
// additive RoutedF16GemmQ8Wide (bypasses RoutedF16Gemm's tile_rows<=64 cap
// for Q8_0, which was tuned on Flash-Next's shapes, not Laguna's). Toggle
// here; see laguna-port-state.md for the A/B result.
constexpr bool kQ8WideDownTile = false;
constexpr std::uint32_t kQ8TileRows = kQ8WideDownTile ? 128 : 64;

// Token rows per routed down-projection tile. Each (expert, tile) block reads
// that expert's whole weight matrix, so the tile count is the weight-traffic
// multiplier: a 4096-row chunk gives an expert 160 rows, which is 4 tiles of
// 48, 3 of 64, or 1 of 192. 192 exists in RoutedF16Gemm's dispatch. Overridden
// so one build measures both arms.
std::uint32_t DownTileRows(core::GgmlType type) {
  static const std::uint32_t override_value = [] {
    if (const char* const v = std::getenv("LAGUNA_DOWN_TILE")) {
      const int parsed = std::atoi(v);
      if (parsed == 16 || parsed == 48 || parsed == 64 || parsed == 192) {
        return static_cast<std::uint32_t>(parsed);
      }
    }
    return 0U;
  }();
  if (override_value != 0) {
    return override_value;
  }
  return type == core::GgmlType::kQ8_0 ? kQ8TileRows : kPlainTileRows;
}
constexpr std::uint32_t kRoutedTileMin = 16;
constexpr std::uint32_t kRoutedTileMax = 80;
constexpr std::uint32_t kRoutedTileStep = 16;
constexpr std::uint32_t kWarmTokens = 64;
// 2026-10-30: split-K decode attention (see kernels.hpp's AttentionSplit).
// Narrow decode/verify batches at depth otherwise sweep the whole KV range in
// one block per (head, token), so TG falls off linearly with context. 8 splits
// mirrors Flash-Next's kAttnSplits; only non-SWA layers need it (SWA is capped
// at sliding_window=512 and already flat). The depth threshold keeps shallow
// decode on the unchanged WMMA path, where the extra merge launch is pure cost.
constexpr std::uint32_t kSplitAttnSplits = 64;
constexpr std::uint32_t kSplitAttnMinDepth = 2048;
// Block-sparse attention (LAGUNA_SPARSE_ATTN): candidate older blocks beyond
// the sink + local window are kept exactly whenever there are at most this
// many of them; above that, only the top-scoring ones (Quest bound) survive.
// Same default as 27B/35B's GUFO_QWEN_SPARSE_TOPK; LAGUNA_SPARSE_TOPK
// overrides it.
constexpr std::uint32_t kSparseTopBlocksDefault = 256;
// The narrowest forward that uses block-sparse selection. The Quest bound
// scores every candidate block once per CUDA block, and a decode step runs
// one block per split per KV head - 64 splits - so the scoring is paid 64
// times over for a single row of queries. Measured at 99k tokens of
// context: prefill 740 -> 1001 tok/s with sparse on, but decode 14.9 ->
// 11.6 tok/s. Prefill is worth the cost, decode is not, so the narrow
// path stays dense. A prefill chunk is 4096 wide, a decode step is 1.
constexpr std::uint32_t kSparseMinTokens = 256;
std::uint32_t SparseTopBlocks() {
  static const std::uint32_t value = [] {
    if (const char* v = std::getenv("LAGUNA_SPARSE_TOPK"))
      return static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
    return kSparseTopBlocksDefault;
  }();
  return value;
}
// Routed-expert activations are stored divided by this (F16 range); the
// combine multiplies it back.
constexpr float kRoutedActScale = 64.0F;

using core::GgmlType;

models::QwenTensorRef TensorRef(const DeviceTensor& t) {
  models::QwenTensorRef ref;
  ref.data = t.data;
  ref.type = t.type;
  ref.num_elements = static_cast<std::size_t>(t.cols) * t.rows;
  return ref;
}

bool Fail(std::string* error_msg, const std::string& message) {
  if (error_msg != nullptr)
    *error_msg = message;
  return false;
}

constexpr std::uint32_t kSnapshotMagic = 0x4e474c41;  // "AGLN"
constexpr std::size_t kSnapshotHeaderBytes = 40;

#define LAGUNA_HIP(call)                                                   \
  do {                                                                     \
    const hipError_t status_ = (call);                                     \
    if (status_ != hipSuccess) {                                           \
      return Fail(error_msg,                                               \
                  std::string(#call) + ": " + hipGetErrorString(status_)); \
    }                                                                      \
  } while (false)

template<typename T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { Reset(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  bool Allocate(std::size_t count) {
    Reset();
    if (hipMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)) !=
        hipSuccess) {
      ptr_ = nullptr;
      return false;
    }
    count_ = count;
    return true;
  }
  void Reset() {
    if (ptr_ != nullptr)
      (void)hipFree(ptr_);
    ptr_ = nullptr;
    count_ = 0;
  }
  T* get() const noexcept { return ptr_; }
  std::size_t bytes() const noexcept { return count_ * sizeof(T); }

private:
  T* ptr_{nullptr};
  std::size_t count_{0};
};

[[maybe_unused]] int RoutedTileCols(std::uint32_t n_tokens,
                                    std::uint32_t n_used,
                                    std::uint32_t n_experts) {
  const std::uint32_t mean =
      std::max<std::uint32_t>(1, n_tokens * n_used / std::max(n_experts, 1U));
  for (std::uint32_t cols = kRoutedTileMin; cols < kRoutedTileMax;
       cols += kRoutedTileStep) {
    if (cols >= 2 * mean)
      return static_cast<int>(cols);
  }
  return static_cast<int>(kRoutedTileMax);
}

bool DenseVecType(GgmlType type) {
  return type == GgmlType::kQ4_K || type == GgmlType::kQ5_K ||
         type == GgmlType::kQ5_1 || type == GgmlType::kQ6_K;
}

std::uint32_t TileCapacityFor(std::size_t slots, std::uint32_t experts) {
  return static_cast<std::uint32_t>(
      (slots + std::size_t{experts} * (kRoutePadRows - 1)) / kRoutePadRows +
      experts + 1);
}

}  // namespace

struct ModelState {
  EngineOptions options;
  std::shared_ptr<core::GgufReader> reader;
  Config config;
  std::unique_ptr<DeviceModel> device;
  std::shared_ptr<const hip::QwenDFlashGpuModel> dflash_model;
  std::vector<std::uint32_t> dflash_taps;
  std::uint32_t dflash_window{0};
  std::uint32_t drafter_block{0};
};

struct Session::Impl {
  std::shared_ptr<ModelState> model;
  hipStream_t stream{nullptr};
  std::uint32_t capacity{0};
  std::uint32_t tile{0};
  std::uint32_t position{0};
  std::uint32_t logit_rows{1};
  float* h_logits{nullptr};
  std::int32_t* h_tokens{nullptr};
  std::uint32_t* h_pos{nullptr};
  std::int32_t* h_tiles{nullptr};
  __half* xh{nullptr};
  hipEvent_t ready{nullptr};
  std::uint32_t* h_counts{nullptr};

  // The KV cache holds F16 halves or q8_0 blocks; both are addressed as
  // bytes and the row stride comes from the session's cache type.
  std::vector<DeviceBuffer<std::byte>> k_cache, v_cache;
  std::vector<std::uint32_t> ring;
  /// Storage format of the KV cache, and the byte stride of one row.
  KvCacheType kv_cache_type{KvCacheType::kF16};
  std::size_t kv_row_bytes{0};
  // Block-sparse attention (LAGUNA_SPARSE_ATTN): per-64-key-block channel
  // min/max pooled keys, allocated only for the non-SWA (group-6,
  // il%4==0) layers - the SWA layers are already bounded by their ring
  // buffer and never use this. One shared buffer per layer, with a single
  // valid_end tracked on the host - see Attention().
  std::vector<DeviceBuffer<__half>> sparse_pooled;
  std::vector<std::uint32_t> sparse_valid_end;
  std::uint32_t sparse_max_blocks{0};
  std::unique_ptr<hip::QwenDFlashGpuExecutor> dflash;
  DeviceBuffer<float> tap_capture;
  std::uint32_t tap_pending{0}, tap_pending_pos{0}, tap_pending_first{0};
  bool inject_taps{true};
  DeviceBuffer<std::int32_t> tokens;
  DeviceBuffer<std::uint32_t> pos;
  // x/xn hold the residual stream and its normed copy; q/k/v/gate/attn/o are
  // transient per-layer scratch, written and fully consumed within one
  // layer's step, so a single shared copy reused sequentially is safe - HIP
  // executes one stream's kernels in launch order. `cx`/`cxn` are what every
  // per-layer helper (Attention/Moe/DenseFfn/RoutedExperts) actually reads
  // and writes.
  DeviceBuffer<float> x, xn, q, k, v, gate, attn, o, router;
  // Partial online-softmax states for the split-K decode attention path,
  // indexed [token][head][split][head_dim + 2] (acc then m, l).
  DeviceBuffer<float> attn_partials;
  DeviceBuffer<float> gate_e, up_e, sh_up, sh_out;
  // down_e/sh_gate/topk/expert_scale carry state from MoeRoute's return to
  // MoeFinish's start. Every per-layer scratch buffer
  // (q/k/v/gate/attn/o/router/ids/counts/routed-expert
  // tiles/gate_e/up_e/sh_up/sh_out/xh) is fully written-then-consumed
  // within a single MoeRoute or MoeFinish call, so reusing it
  // sequentially is still safe (HIP's in-order stream execution).
  DeviceBuffer<float> down_e, sh_gate, topk, expert_scale;
  DeviceBuffer<float> expert_const, dense_gate, dense_up, d_logits;
  DeviceBuffer<float> last;
  DeviceBuffer<std::int32_t> ids;
  DeviceBuffer<std::uint32_t> counts;
  DeviceBuffer<std::int32_t> route_bounds, route_cursors, rows_token, rows_slot;
  DeviceBuffer<std::int32_t> tiles_pair, tiles_down;
  DeviceBuffer<__half> act_half, qh;
  // Dense Q8_0 GEMM through the Q8_0 F16-dequant WMMA kernel. A dense
  // projection is that kernel with one bucket covering every row, so the
  // gather and scatter maps are the identity; they are built once here and
  // the bucket bound is a two-int async copy per call. GUFO_DENSE_F16=1
  // selects this route. See PERF-FINDINGS.md 17.2.
  DeviceBuffer<std::int32_t> dense_rows, dense_tiles, dense_bounds;
  DeviceBuffer<__half> xh_wide;
  std::size_t xh_wide_k{0};
  bool dense_ready{false};

  // Caches the MMQ-tile quantization of the last Dense() input so repeated
  // calls against the same activation (Q/K/V off attn_norm's output, or
  // gate/up off ffn_norm's) quantize once instead of once per projection.
  DeviceBuffer<char> quant_cache;
  const float* quant_cache_src{nullptr};
  std::uint32_t quant_cache_n{0}, quant_cache_k{0};
  bool Quantized(const float* in, std::uint32_t n, std::uint32_t k);
  // Same caching idea, for Dense()'s Q8_0 small-batch (n<=8) vec path -
  // Q/K/V off attn_norm's output (or gate/up off ffn_norm's) share one
  // input pointer, so quantize once per activation instead of once per
  // projection (same win Quantized() already gets on the wide path).
  DeviceBuffer<char> q8_vec_quant;
  const float* q8_vec_quant_src{nullptr};
  std::uint32_t q8_vec_quant_n{0}, q8_vec_quant_k{0};
  bool QuantizedVec(const float* in, std::uint32_t n, std::uint32_t k);
  // Caches QuantizeQ8Tiled's output for Dense()'s W8A8Gemm experiment
  // (2026-10-30): same dedup idea, for the int8-WMMA wide-batch path.
  DeviceBuffer<char> q8_tiled_quant;
  const float* q8_tiled_quant_src{nullptr};
  std::uint32_t q8_tiled_quant_n{0}, q8_tiled_quant_k{0};
  bool QuantizedTiled(const float* in, std::uint32_t n, std::uint32_t k);

  ~Impl() {
    if (hipblas_handle != nullptr) {
      (void)hipblasDestroy(hipblas_handle);
      hipblas_handle = nullptr;
    }
    if (stream != nullptr) {
      (void)hipStreamSynchronize(stream);
      (void)hipStreamDestroy(stream);
    }
    if (xh != nullptr)
      (void)hipFree(xh);
    if (ready != nullptr)
      (void)hipEventDestroy(ready);
    if (h_logits != nullptr)
      (void)hipHostFree(h_logits);
    if (h_tokens != nullptr)
      (void)hipHostFree(h_tokens);
    if (h_pos != nullptr)
      (void)hipHostFree(h_pos);
    if (h_tiles != nullptr)
      (void)hipHostFree(h_tiles);
    if (h_counts != nullptr)
      (void)hipHostFree(h_counts);
    for (auto& g : graphs) {
      if (g.exec != nullptr)
        (void)hipGraphExecDestroy(g.exec);
    }
  }

  bool Init(std::uint32_t max_context, std::string* error_msg);
  bool Dense(const DeviceTensor& w, const float* in, float* out,
             std::uint32_t n, std::string* error_msg);
  bool Experts(const DeviceTensor& w, const float* in, float* out,
               std::uint32_t rows, std::uint32_t used, std::uint32_t n_tokens,
               std::string* error_msg);
  bool Attention(std::uint32_t il, std::uint32_t n, std::string* error_msg);
  bool DenseFfn(const DeviceLayer& layer, std::uint32_t n,
                std::string* error_msg);
  bool Moe(const DeviceLayer& layer, std::uint32_t n, std::string* error_msg);
  bool MoeRoute(const DeviceLayer& layer, std::uint32_t n, bool* wide_out,
                bool* scaled_out, std::string* error_msg);
  bool MoeFinish(const DeviceLayer& layer, std::uint32_t n, bool wide,
                 bool scaled, std::string* error_msg);
  bool RoutedExperts(const DeviceLayer& layer, std::uint32_t n, bool* scaled,
                     std::string* error_msg);
  bool Forward(std::span<const std::int32_t> tokens, bool all_rows,
               std::uint32_t prefix_end, std::string* error_msg);
  bool Enqueue(std::uint32_t n, bool all_rows, std::uint32_t prefix_end,
               std::uint32_t* tap_rows_out, std::uint32_t* tap_first_out,
               std::string* error_msg);
  bool DflashInject(std::uint32_t rows, std::string* error_msg);
  std::size_t PooledSnapshotBytes(std::uint32_t il,
                                  std::uint32_t position) const noexcept;
  std::size_t PooledBlocks(std::uint32_t position) const noexcept;
  void SavePooled(std::uint32_t il, std::uint32_t position,
                  std::uint8_t* out) const;
  void RestorePooled(std::uint32_t il, std::uint32_t position,
                     const std::uint8_t* in);

  // One captured HIP graph per decode/verify width (1..kMaxVerifyRows),
  // mirroring Flint's pattern: the whole per-step kernel sequence launches
  // once from host code, then replays as a single `hipGraphLaunch` on every
  // later call at that width - cuts per-kernel launch overhead, the
  // dominant cost at n=1..16. Capture only applies when `all_rows` and
  // `n<=kMaxVerifyRows` (the decode/verify path, not full prefill chunks),
  // and only once the DFlash tap-capture window has reached its steady
  // state for that width (tap_rows==n - true for plain decode always,
  // since dflash==nullptr there, and true for DFlash verify once position
  // has advanced past the window-fill edge case) - capturing while
  // tap_first/tap_rows are still settling would bake a stale, position-
  // dependent slice into the graph.
  struct StepGraph {
    hipGraphExec_t exec{nullptr};
    bool failed{false};
    std::uint32_t steps{0};
  };
  StepGraph graphs[Session::kMaxVerifyRows + 1];

  // cx/cxn are the residual-stream buffers every per-layer helper reads
  // and writes.
  float* cx{nullptr};
  float* cxn{nullptr};

  // Env-gated GPU-event stage timing. Null unless the matching variable is
  // set, so an unset run pays one pointer test per boundary. Marks are events
  // on the compute stream, so the report is device time per stage rather than
  // host launch time.
  std::unique_ptr<gufo::hip::SubStageClock> layer_clock;
  std::unique_ptr<gufo::hip::SubStageClock> ffn_clock;
  hipblasHandle_t hipblas_handle{nullptr};
};

bool Session::Impl::Init(std::uint32_t max_context, std::string* error_msg) {
  const Config& c = model->config;
  if (c.expert_used > kMaxTopK || c.expert_count > kMaxExperts) {
    return Fail(error_msg, "Laguna: expert counts exceed the router limit");
  }
  capacity = max_context;
  tile = std::max<std::uint32_t>(1, model->options.prefill_tile);
  kv_cache_type = model->options.kv_cache_type;
  if (kv_cache_type == KvCacheType::kQ8_0 &&
      (c.head_dim % 32 != 0 || c.num_kv * c.head_dim % 32 != 0)) {
    return Fail(error_msg,
                "Laguna: q8_0 KV needs a head width that is a multiple of 32");
  }
  LAGUNA_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  LAGUNA_HIP(hipEventCreateWithFlags(&ready, hipEventDisableTiming));
  if (hipblasCreate(&hipblas_handle) != HIPBLAS_STATUS_SUCCESS ||
      hipblasSetStream(hipblas_handle, stream) != HIPBLAS_STATUS_SUCCESS) {
    return Fail(error_msg, "Laguna: hipblas initialization failed");
  }

  // Stage timers. GUFO_PROFILE splits each layer into attention and
  // feed-forward; the two sub-variables drill into one branch each. Events are
  // created per mark and read after the forward's own sync, so nothing here
  // changes the kernel sequence.
  const auto make_clock = [&](const char* variable) {
    const char* const value = std::getenv(variable);
    if (value == nullptr || *value == '0') {
      return std::unique_ptr<gufo::hip::SubStageClock>{};
    }
    auto clock = std::make_unique<gufo::hip::SubStageClock>(variable);
    clock->SetStream(stream);
    return clock;
  };
  layer_clock = make_clock("GUFO_PROFILE");
  if (layer_clock != nullptr) {
    layer_clock->AddStage("norm_attn");
    layer_clock->AddStage("qkv_dense");
    layer_clock->AddStage("norm_rope");
    layer_clock->AddStage("store_kv");
    layer_clock->AddStage("attn_core");
    layer_clock->AddStage("norm_ffn");
    layer_clock->AddStage("ffn");
  }
  ffn_clock = make_clock("GUFO_PROFILE_FFN");
  if (ffn_clock != nullptr) {
    ffn_clock->AddStage("router");
    ffn_clock->AddStage("shared_up");
    ffn_clock->AddStage("routed");
    ffn_clock->AddStage("shared_down");
  }
  if (qfn_mmq_init(0) != 0)
    return Fail(error_msg, "mmq initialization failed");

  const std::size_t kv_row = static_cast<std::size_t>(c.num_kv) * c.head_dim;
  kv_row_bytes =
      kv_cache_type == KvCacheType::kQ8_0
          ? rocm::detail::Q8KvRowBytes(static_cast<std::uint32_t>(kv_row))
          : kv_row * sizeof(__half);
  const std::uint32_t max_heads =
      *std::max_element(c.heads.begin(), c.heads.end());
  const std::size_t q_row = static_cast<std::size_t>(max_heads) * c.head_dim;
  const std::size_t slots = static_cast<std::size_t>(tile) * c.expert_used;

  // Size the KV cache before allocating it. On a unified-memory GPU the
  // device pool is backed by host RAM, so an over-large allocation can be
  // admitted and then fail to back its pages: the kernels read unbacked
  // memory and the forward pass returns non-finite logits instead of an
  // allocation error. Sizing it first lets the caller see the real
  // requirement and pick a context the device can hold.
  const std::size_t kv_bytes =
      SessionBytesForContextLocked(c, capacity, tile, kv_cache_type);
  {
    const std::size_t free_bytes = Model::FreeDeviceBytes();
    if (free_bytes != 0) {
      // The weights are already resident; the drafter's own allocations and
      // the session's scratch come out of the same pool. Require enough
      // headroom for the KV cache plus that scratch, and refuse rather than
      // run with a cache the device cannot back.
      if (kv_bytes * 2 > free_bytes) {
        return Fail(error_msg,
                    "Laguna: the KV cache for a " + std::to_string(capacity) +
                        "-token context needs " +
                        std::to_string(kv_bytes / (1024 * 1024)) +
                        " MiB but only " +
                        std::to_string(free_bytes / (1024 * 1024)) +
                        " MiB of device memory is free; use a smaller "
                        "context");
      }
    }
  }

  k_cache.resize(c.num_layers);
  v_cache.resize(c.num_layers);
  ring.assign(c.num_layers, 0);
  bool ok = true;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    std::uint32_t rows = capacity;
    if (c.IsSwa(il)) {
      // Sized for the window plus one prefill tile: a forward writes up to
      // `tile` new rows and then reads every key back to window+1 rows
      // behind its own queries, so the ring has to hold both at once.
      // Sizing it for one tile only let the new rows wrap around and
      // overwrite rows the same forward's sliding-window attention still
      // needed.
      ring[il] = c.sliding_window + tile;
      rows = std::min(capacity, ring[il]);
      if (rows == capacity)
        ring[il] = 0;
    }
    ok = ok && k_cache[il].Allocate(kv_row_bytes * rows) &&
         v_cache[il].Allocate(kv_row_bytes * rows);
  }
  const bool sparse_attn = std::getenv("LAGUNA_SPARSE_ATTN") != nullptr;
  sparse_pooled.resize(c.num_layers);
  sparse_valid_end.assign(c.num_layers, 0);
  if (sparse_attn) {
    sparse_max_blocks = (capacity + 63) / 64;
    for (std::uint32_t il = 0; il < c.num_layers && ok; ++il) {
      if (!c.IsSwa(il)) {
        ok = sparse_pooled[il].Allocate(std::size_t{c.num_kv} *
                                        sparse_max_blocks * 256);
      }
    }
  }
  ok = ok && tokens.Allocate(tile) && pos.Allocate(4) &&
       x.Allocate(std::size_t{tile} * c.hidden_size) &&
       xn.Allocate(std::size_t{tile} * c.hidden_size) &&
       q.Allocate(std::size_t{tile} * q_row) &&
       qh.Allocate(std::size_t{tile} * q_row) &&
       k.Allocate(std::size_t{tile} * kv_row) &&
       v.Allocate(std::size_t{tile} * kv_row) &&
       gate.Allocate(std::size_t{tile} * max_heads) &&
       attn.Allocate(std::size_t{tile} * q_row) &&
       attn_partials.Allocate(std::size_t{kMaxVerifyRows} * max_heads *
                              kSplitAttnSplits * (c.head_dim + 2)) &&
       o.Allocate(std::size_t{tile} * c.hidden_size) &&
       router.Allocate(std::size_t{tile} * c.expert_count) &&
       gate_e.Allocate(slots * c.expert_ff) &&
       up_e.Allocate(slots * c.expert_ff) &&
       down_e.Allocate(slots * c.hidden_size) && topk.Allocate(slots) &&
       ids.Allocate(slots) && counts.Allocate(c.expert_count) &&
       route_bounds.Allocate(kMaxExperts + 1) &&
       route_cursors.Allocate(kMaxExperts) &&
       rows_token.Allocate(
           qwen38_flash_next::rocm::RoutedCompactRows(slots, c.expert_count)) &&
       rows_slot.Allocate(
           qwen38_flash_next::rocm::RoutedCompactRows(slots, c.expert_count)) &&
       tiles_pair.Allocate(TileCapacityFor(slots, c.expert_count)) &&
       tiles_down.Allocate(TileCapacityFor(slots, c.expert_count)) &&
       act_half.Allocate(slots * c.expert_ff) && expert_scale.Allocate(slots) &&
       expert_const.Allocate(slots) &&
       sh_gate.Allocate(std::size_t{tile} * c.shared_ff) &&
       sh_up.Allocate(std::size_t{tile} * c.shared_ff) &&
       sh_out.Allocate(std::size_t{tile} * c.hidden_size) &&
       dense_gate.Allocate(std::size_t{tile} * c.dense_ff) &&
       dense_up.Allocate(std::size_t{tile} * c.dense_ff) &&
       dense_rows.Allocate(tile) &&
       dense_tiles.Allocate((std::size_t{tile} + 127) / 128) &&
       dense_bounds.Allocate(2) &&
       xh_wide.Allocate(
           std::size_t{tile} *
           (xh_wide_k = std::max(q_row, std::size_t{c.hidden_size}))) &&
       last.Allocate(std::size_t{kMaxVerifyRows} * c.hidden_size) &&
       d_logits.Allocate(std::size_t{kMaxVerifyRows} * c.vocab_size);
  ok = ok &&
       hipMalloc(reinterpret_cast<void**>(&xh),
                 std::size_t{tile} * c.hidden_size * sizeof(__half)) ==
           hipSuccess &&
       hipHostMalloc(reinterpret_cast<void**>(&h_logits),
                     std::size_t{kMaxVerifyRows} * c.vocab_size *
                         sizeof(float)) == hipSuccess &&
       hipHostMalloc(reinterpret_cast<void**>(&h_tokens),
                     std::size_t{tile} * sizeof(std::int32_t)) == hipSuccess &&
       hipHostMalloc(reinterpret_cast<void**>(&h_pos),
                     4 * sizeof(std::uint32_t)) == hipSuccess &&
       hipHostMalloc(reinterpret_cast<void**>(&h_counts),
                     kMaxExperts * sizeof(std::uint32_t)) == hipSuccess &&
       hipHostMalloc(reinterpret_cast<void**>(&h_tiles),
                     2 * std::size_t{TileCapacityFor(slots, c.expert_count)} *
                         sizeof(std::int32_t)) == hipSuccess;
  if (!ok)
    return Fail(error_msg, "Laguna: session allocation failed");
  {
    const std::vector<float> efill(slots, kRoutedActScale);
    LAGUNA_HIP(hipMemcpy(expert_const.get(), efill.data(),
                         efill.size() * sizeof(float), hipMemcpyHostToDevice));
  }
  {
    // Identity maps for the dense Q8_0 route: one bucket covering every row,
    // one macro tile per 128 rows. Only the bucket bound changes per call.
    std::vector<std::int32_t> rows(tile);
    for (std::uint32_t i = 0; i < tile; ++i) {
      rows[i] = static_cast<std::int32_t>(i);
    }
    std::vector<std::int32_t> dtiles((std::size_t{tile} + 127) / 128);
    for (std::size_t j = 0; j < dtiles.size(); ++j) {
      dtiles[j] = static_cast<std::int32_t>(j << 16);
    }
    const std::int32_t bounds[2] = {0, static_cast<std::int32_t>(tile)};
    dense_ready = hipMemcpy(dense_rows.get(), rows.data(),
                            rows.size() * sizeof(std::int32_t),
                            hipMemcpyHostToDevice) == hipSuccess &&
                  hipMemcpy(dense_tiles.get(), dtiles.data(),
                            dtiles.size() * sizeof(std::int32_t),
                            hipMemcpyHostToDevice) == hipSuccess &&
                  hipMemcpy(dense_bounds.get(), bounds, sizeof(bounds),
                            hipMemcpyHostToDevice) == hipSuccess;
  }
  std::fill(h_logits, h_logits + std::size_t{kMaxVerifyRows} * c.vocab_size,
            0.0F);
  cx = x.get();
  cxn = xn.get();
  return true;
}

bool Session::Impl::Quantized(const float* in, std::uint32_t n,
                              std::uint32_t k) {
  if (quant_cache_src == in && quant_cache_n == n && quant_cache_k == k)
    return true;
  const std::size_t bytes =
      qfn_mmq_q8_1_mmq_bytes(static_cast<int>(n), static_cast<int>(k));
  if (quant_cache.bytes() < bytes && !quant_cache.Allocate(bytes))
    return false;
  if (qfn_mmq_quantize_q8_1_mmq(in, quant_cache.get(), static_cast<int>(n),
                                static_cast<int>(k), stream) != 0) {
    return false;
  }
  quant_cache_src = in;
  quant_cache_n = n;
  quant_cache_k = k;
  return true;
}

bool Session::Impl::QuantizedVec(const float* in, std::uint32_t n,
                                 std::uint32_t k) {
  if (q8_vec_quant_src == in && q8_vec_quant_n == n && q8_vec_quant_k == k)
    return true;
  const std::size_t bytes =
      qfn_mmq_q8_1_bytes(static_cast<int>(n), static_cast<int>(k));
  if (q8_vec_quant.bytes() < bytes && !q8_vec_quant.Allocate(bytes))
    return false;
  if (qfn_mmq_quantize_q8_1(in, q8_vec_quant.get(), static_cast<int>(n),
                            static_cast<int>(k), stream) != 0) {
    return false;
  }
  q8_vec_quant_src = in;
  q8_vec_quant_n = n;
  q8_vec_quant_k = k;
  return true;
}

bool Session::Impl::QuantizedTiled(const float* in, std::uint32_t n,
                                   std::uint32_t k) {
  if (q8_tiled_quant_src == in && q8_tiled_quant_n == n &&
      q8_tiled_quant_k == k)
    return true;
  const std::size_t bytes = qwen38_flash_next::rocm::Q8TiledBytes(n, k);
  if (q8_tiled_quant.bytes() < bytes && !q8_tiled_quant.Allocate(bytes))
    return false;
  qwen38_flash_next::rocm::QuantizeQ8Tiled(in, q8_tiled_quant.get(), n, k,
                                           stream);
  q8_tiled_quant_src = in;
  q8_tiled_quant_n = n;
  q8_tiled_quant_k = k;
  return true;
}

bool Session::Impl::Dense(const DeviceTensor& w, const float* in, float* out,
                          std::uint32_t n, std::string* error_msg) {
  if (w.type == GgmlType::kQ8_0) {
    // qfn_mmq_q8_0_dense_vec_preq avoids mul_mat_q's WMMA-tile mmq_x>=16
    // granularity floor (gfx1151 is WMMA-only, no MFMA - mmq_x=8 is never
    // usable there), which otherwise wastes 15/16 of a GEMM tile at n=1
    // (confirmed via profiling: 43% of all decode kernel time). Verified:
    // per-call output differs from the GEMM path by ~0.03-0.04 absolute
    // (on values ~15-90), consistent with ordinary Q8 rounding, not a
    // bug - and a full generation run at n=300 stayed coherent and
    // measurably faster (+6%) at the same prompt/seed. The two paths'
    // outputs do drift apart over many layers/tokens, same as every other
    // numeric-path change in this file (MoE routing's near-tie
    // sensitivity) - not specific to this change. See
    // laguna-port-state.md.
    if (n <= 8) {
      if (!QuantizedVec(in, n, w.cols))
        return Fail(error_msg, "Q8_0 vec activation quantization failed");
      if (qfn_mmq_q8_0_dense_vec_preq(w.data, nullptr, q8_vec_quant.get(), out,
                                      static_cast<int>(w.rows),
                                      static_cast<int>(n),
                                      static_cast<int>(w.cols), stream) != 0) {
        return Fail(error_msg, "Q8_0 dense vec projection failed");
      }
      return true;
    }
    // W8A8Gemm (int8 WMMA GEMM, reads the same block_q8_0 weight bytes
    // directly - no re-layout) replaces mul_mat_q for the wide (n>8) Q8_0
    // dense path: measured 1,222 -> 1,275-1,287 tok/s pp2048 (+4-5%).
    // Falls through to the MMQ path if W8A8Gemm declines (k not a multiple
    // of 32). See laguna-port-state.md.
    //
    // GUFO_DENSE_F16=1 routes the same projection through the Q8_0
    // F16-dequant WMMA kernel instead, as a single-bucket dense GEMM. On the
    // same pass that kernel runs at about 33 TFLOP/s on Q4_K where the int8
    // path runs at 19, and it accumulates the whole K extent in F32 with no
    // per-K-block rescale. Measured in PERF-FINDINGS.md 17.2.
    static const bool kDenseF16 = std::getenv("GUFO_DENSE_F16") != nullptr;
    if (kDenseF16 && dense_ready && n >= 128 && n <= tile &&
        w.rows % 128 == 0 && w.cols % 64 == 0 &&
        std::size_t{w.cols} <= xh_wide_k) {
      rocm::ToHalf(in, xh_wide.get(), std::size_t{n} * w.cols, stream);
      const std::int32_t bound[2] = {0, static_cast<std::int32_t>(n)};
      if (hipMemcpyAsync(dense_bounds.get(), bound, sizeof(bound),
                         hipMemcpyHostToDevice, stream) == hipSuccess &&
          qwen38_flash_next::rocm::RoutedF16GemmQ8Wide(
              w.data, xh_wide.get(), dense_tiles.get(),
              static_cast<std::uint32_t>((n + 127) / 128), dense_bounds.get(),
              dense_rows.get(), dense_rows.get(), nullptr, out, nullptr, w.rows,
              w.cols, stream)) {
        return true;
      }
    }
    if (QuantizedTiled(in, n, w.cols) &&
        qwen38_flash_next::rocm::W8A8Gemm(w.data, q8_tiled_quant.get(), out, n,
                                          w.rows, w.cols, stream)) {
      return true;
    }
    if (!Quantized(in, n, w.cols))
      return Fail(error_msg, "Q8_0 activation quantization failed");
    if (qfn_mmq_q8_0_dense_gemm_preq(
            w.data, quant_cache.get(), out, static_cast<int>(w.rows),
            static_cast<int>(n), static_cast<int>(w.cols), stream) != 0) {
      return Fail(error_msg, "Q8_0 dense projection failed");
    }
    return true;
  }
  if (!DenseVecType(w.type))
    return Fail(error_msg, "unsupported dense weight type");
  for (std::uint32_t r0 = 0; r0 < n; r0 += kVecBatch) {
    const std::uint32_t rows = std::min(kVecBatch, n - r0);
    const float* src = in + static_cast<std::size_t>(r0) * w.cols;
    float* dst = out + static_cast<std::size_t>(r0) * w.rows;
    if (qfn_mmq_dense_vec(static_cast<int>(w.type), w.data, src, dst,
                          static_cast<int>(w.rows), static_cast<int>(w.cols),
                          static_cast<int>(rows), stream) != 0) {
      return Fail(error_msg, "quantized dense projection failed");
    }
  }
  return true;
}

bool Session::Impl::Experts(const DeviceTensor& w, const float* in, float* out,
                            std::uint32_t rows, std::uint32_t used,
                            std::uint32_t n_tokens, std::string* error_msg) {
  const int M = static_cast<int>(w.rows);
  const int K = static_cast<int>(w.cols);
  const int E = static_cast<int>(w.experts);
  const int T = static_cast<int>(rows);
  const int U = static_cast<int>(used);
  if (rows * used > kVecExpertRows) {
    return Fail(error_msg, "Laguna: wide expert batch on the vector path");
  }
  (void)n_tokens;
  if (qfn_mmq_moe_vec(static_cast<int>(w.type), w.data, in, ids.get(), out, M,
                      K, T, E, U, stream, nullptr, nullptr) != 0) {
    return Fail(error_msg, "routed expert projection failed");
  }
  return true;
}

bool Session::Impl::Attention(std::uint32_t il, std::uint32_t n,
                              std::string* error_msg) {
  const Config& c = model->config;
  const DeviceLayer& L = model->device->layers()[il];
  const std::uint32_t heads = c.Heads(il);
  const std::uint32_t kv_row = c.num_kv * c.head_dim;
  const bool swa = c.IsSwa(il);
  const RopeTables& rope = model->device->rope(swa);

  if (!Dense(L.attn_q, cxn, q.get(), n, error_msg) ||
      !Dense(L.attn_k, cxn, k.get(), n, error_msg) ||
      !Dense(L.attn_v, cxn, v.get(), n, error_msg)) {
    return false;
  }
  rocm::Q8Rows(L.attn_gate.data, cxn, gate.get(), n, L.attn_gate.rows,
               L.attn_gate.cols, stream);
  if (layer_clock != nullptr)
    layer_clock->Mark();
  rocm::RmsNormRows(q.get(), static_cast<const float*>(L.attn_q_norm.data),
                    q.get(), n, heads * c.head_dim, heads, c.rms_eps, stream);
  rocm::RmsNormRows(k.get(), static_cast<const float*>(L.attn_k_norm.data),
                    k.get(), n, kv_row, c.num_kv, c.rms_eps, stream);
  rocm::RopeTable(q.get(), n, heads, c.head_dim, rope.n_dims, rope.inv_freq,
                  rope.mscale, pos.get(), stream);
  rocm::RopeTable(k.get(), n, c.num_kv, c.head_dim, rope.n_dims, rope.inv_freq,
                  rope.mscale, pos.get(), stream);
  if (layer_clock != nullptr)
    layer_clock->Mark();
  if (kv_cache_type == KvCacheType::kQ8_0) {
    rocm::StoreKvQ8(k.get(), v.get(), k_cache[il].get(), v_cache[il].get(), n,
                    kv_row, pos.get(), ring[il], stream);
  } else {
    rocm::StoreKv(k.get(), v.get(),
                  reinterpret_cast<__half*>(k_cache[il].get()),
                  reinterpret_cast<__half*>(v_cache[il].get()), n, kv_row,
                  pos.get(), ring[il], stream);
  }
  if (layer_clock != nullptr)
    layer_clock->Mark();
  // Block-sparse attention's pooled-key maintenance is the CALLER's job
  // (Forward()'s live post-pass), never Attention()'s own - a decode/verify
  // HIP
  // graph can capture this function, and UpdatePooledKeysLaguna's grid
  // size and from_pos/end_pos would then be baked into the recorded graph
  // as fixed launch arguments (not read from a device pointer at replay
  // time), so every later replay would re-pool only the ONE block range
  // that existed at capture time, forever - silently degrading sparse's
  // accuracy the longer a captured decode run goes (invisible on a short
  // needle-in-haystack generation, real on a long completion). See
  // Forward()'s UpdateSparsePooling-equivalent post-pass.
  const __half* sparse_kmm =
      n >= kSparseMinTokens ? sparse_pooled[il].get() : nullptr;
  rocm::ToHalf(q.get(), qh.get(), std::size_t{n} * heads * c.head_dim, stream);
  // Four real bugs found and fixed in AttentionWmma/AttentionWmmaKernelLaguna
  // (shape guard; token stride missing *2; missing softplus(gate) multiply
  // the scalar reference applies but WMMA silently dropped; head address
  // using the tile width (kWmmaGroup=8) instead of the real per-kv_head
  // stride (heads/kv_heads=6) as the memory stride, which only worked by
  // coincidence for kv_head==0). Verified via `gufo bench
  // --validate-prefill` and a per-layer scalar-vs-WMMA trace
  // (LAGUNA_WMMA_DEBUG=1): every group-6 layer now matches scalar to
  // fp16-precision-level noise only. See laguna-port-state.md for the full
  // debugging trail, including why n=16 alone still fails top1_match (a
  // pre-existing near-tie numeric sensitivity, not a WMMA bug -- confirmed
  // by reproducing the same divergence with WMMA fully disabled).
  constexpr bool kWmmaAttentionEnabled = true;
  // Split-K only pays off where the single-block sweep is long: non-SWA
  // layers (SWA is capped by sliding_window) on the narrow decode/verify
  // batches (prefill's WMMA path is already fine and must stay untouched) past
  // kSplitAttnMinDepth. See kernels.hpp's AttentionSplit.
  const bool split_attn = !swa && n <= Session::kMaxVerifyRows &&
                          position + n >= kSplitAttnMinDepth;
  const bool use_wmma =
      kWmmaAttentionEnabled &&
      rocm::AttentionWmma(qh.get(), k_cache[il].get(), v_cache[il].get(),
                          gate.get(), attn.get(), pos.get(), heads, c.num_kv,
                          c.head_dim, swa ? c.sliding_window : 0, ring[il], n,
                          split_attn ? attn_partials.get() : nullptr,
                          split_attn ? kSplitAttnSplits : 1, stream, sparse_kmm,
                          sparse_max_blocks,
                          sparse_kmm != nullptr ? SparseTopBlocks() : 0,
                          kv_cache_type);
  if (!use_wmma) {
    rocm::Attention(q.get(), reinterpret_cast<const __half*>(k_cache[il].get()),
                    reinterpret_cast<const __half*>(v_cache[il].get()),
                    gate.get(), attn.get(), n, pos.get(), heads, c.num_kv,
                    c.head_dim, swa ? c.sliding_window : 0, ring[il], stream,
                    kv_cache_type);
  }
  if (layer_clock != nullptr)
    layer_clock->Mark();
  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  if (split_attn && use_wmma && std::getenv("LAGUNA_SPLIT_DEBUG") != nullptr &&
      hipStreamIsCapturing(stream, &capture_status) == hipSuccess &&
      capture_status == hipStreamCaptureStatusNone) {
    // Differential trace: the split path is the same WMMA kernel with its key
    // tiles round-robined over grid.z and merged, so against the single-block
    // kernel it must match to fp32 rounding. Any larger difference is a real
    // split/merge bug (suspects: tile round-robin at the tail, ring modulo at
    // split boundaries, merge base/log2 mismatch).
    DeviceBuffer<float> ref_attn;
    if (!ref_attn.Allocate(std::size_t{n} * heads * c.head_dim))
      return Fail(error_msg, "Laguna: split debug allocation failed");
    const bool ref_ok = rocm::AttentionWmma(
        qh.get(), k_cache[il].get(), v_cache[il].get(), gate.get(),
        ref_attn.get(), pos.get(), heads, c.num_kv, c.head_dim,
        swa ? c.sliding_window : 0, ring[il], n, nullptr, 1, stream, nullptr, 0,
        0, kv_cache_type);
    if (!ref_ok)
      return Fail(error_msg, "Laguna: split debug reference failed");
    LAGUNA_HIP(hipStreamSynchronize(stream));
    std::vector<float> split_host(std::size_t{n} * heads * c.head_dim);
    std::vector<float> ref_host(split_host.size());
    LAGUNA_HIP(hipMemcpy(split_host.data(), attn.get(),
                         split_host.size() * sizeof(float),
                         hipMemcpyDeviceToHost));
    LAGUNA_HIP(hipMemcpy(ref_host.data(), ref_attn.get(),
                         ref_host.size() * sizeof(float),
                         hipMemcpyDeviceToHost));
    double max_abs = 0.0, max_ref = 0.0;
    for (std::size_t i = 0; i < split_host.size(); ++i) {
      max_abs = std::max(
          max_abs, static_cast<double>(std::fabs(split_host[i] - ref_host[i])));
      max_ref = std::max(max_ref, static_cast<double>(std::fabs(ref_host[i])));
    }
    std::fprintf(stderr,
                 "[laguna] split debug layer=%u n=%u depth=%u max_abs=%g "
                 "max_ref=%g rel=%g\n",
                 il, n, position, max_abs, max_ref,
                 max_ref > 0.0 ? max_abs / max_ref : 0.0);
  }
  if (use_wmma && std::getenv("LAGUNA_WMMA_DEBUG") != nullptr &&
      hipStreamIsCapturing(stream, &capture_status) == hipSuccess &&
      capture_status == hipStreamCaptureStatusNone) {
    DeviceBuffer<float> scalar_attn;
    scalar_attn.Allocate(std::size_t{n} * heads * c.head_dim);
    rocm::Attention(q.get(), k_cache[il].get(), v_cache[il].get(), gate.get(),
                    scalar_attn.get(), n, pos.get(), heads, c.num_kv,
                    c.head_dim, swa ? c.sliding_window : 0, ring[il], stream,
                    kv_cache_type);
    LAGUNA_HIP(hipStreamSynchronize(stream));
    std::vector<float> wmma_host(std::size_t{n} * heads * c.head_dim);
    std::vector<float> scalar_host(wmma_host.size());
    LAGUNA_HIP(hipMemcpy(wmma_host.data(), attn.get(),
                         wmma_host.size() * sizeof(float),
                         hipMemcpyDeviceToHost));
    LAGUNA_HIP(hipMemcpy(scalar_host.data(), scalar_attn.get(),
                         scalar_host.size() * sizeof(float),
                         hipMemcpyDeviceToHost));
    for (std::uint32_t t = 0; t < n; ++t) {
      for (std::uint32_t h = 0; h < heads; ++h) {
        double max_abs = 0.0;
        double sum_abs_scalar = 0.0;
        const std::size_t base = (std::size_t{t} * heads + h) * c.head_dim;
        for (std::uint32_t d = 0; d < c.head_dim; ++d) {
          const double diff =
              std::fabs(wmma_host[base + d] - scalar_host[base + d]);
          max_abs = std::max(max_abs, diff);
          sum_abs_scalar += std::fabs(scalar_host[base + d]);
        }
        std::fprintf(stderr,
                     "[LAGUNA_WMMA_DEBUG] layer=%u token=%u head=%u "
                     "max_abs_diff=%.6f scalar_sum_abs=%.6f "
                     "wmma[0..3]=%.4f,%.4f,%.4f,%.4f "
                     "scalar[0..3]=%.4f,%.4f,%.4f,%.4f\n",
                     il, t, h, max_abs, sum_abs_scalar, wmma_host[base + 0],
                     wmma_host[base + 1], wmma_host[base + 2],
                     wmma_host[base + 3], scalar_host[base + 0],
                     scalar_host[base + 1], scalar_host[base + 2],
                     scalar_host[base + 3]);
      }
    }
  }
  if (!Dense(L.attn_output, attn.get(), o.get(), n, error_msg))
    return false;
  rocm::AddInPlace(cx, o.get(), std::size_t{n} * c.hidden_size, stream);
  return true;
}

bool Session::Impl::DenseFfn(const DeviceLayer& layer, std::uint32_t n,
                             std::string* error_msg) {
  const Config& c = model->config;
  // Same five boundaries as the MoE branch so the sub-stage report buckets
  // evenly whether a layer routes or not; a dense layer simply has nothing
  // between two of them.
  if (ffn_clock != nullptr)
    ffn_clock->Mark();
  if (ffn_clock != nullptr)
    ffn_clock->Mark();
  if (!Dense(layer.ffn_gate, cxn, dense_gate.get(), n, error_msg) ||
      !Dense(layer.ffn_up, cxn, dense_up.get(), n, error_msg)) {
    return false;
  }
  rocm::Swiglu(dense_gate.get(), dense_up.get(), std::size_t{n} * c.dense_ff,
               stream);
  if (ffn_clock != nullptr)
    ffn_clock->Mark();
  if (ffn_clock != nullptr)
    ffn_clock->Mark();
  if (!Dense(layer.ffn_down, dense_gate.get(), o.get(), n, error_msg))
    return false;
  rocm::AddInPlace(cx, o.get(), std::size_t{n} * c.hidden_size, stream);
  if (ffn_clock != nullptr)
    ffn_clock->Mark();
  return true;
}

bool Session::Impl::RoutedExperts(const DeviceLayer& layer, std::uint32_t n,
                                  bool* scaled, std::string* error_msg) {
  using qwen38_flash_next::rocm::WeightType;
  const Config& c = model->config;
  const std::uint32_t used = c.expert_used;
  const std::uint32_t experts = c.expert_count;
  const DeviceTensor& ge = layer.gate_exps;
  const DeviceTensor& ue = layer.up_exps;
  const DeviceTensor& de = layer.down_exps;
  // 2026-10-30: checked whether Laguna's gate/up experts are ever Q8_0 (the
  // fused RoutedGatedF16Gemm path only covers K-quant types) - confirmed via
  // instrumentation they are always Q4_K across every MoE layer in the
  // UD-Q4_K_M quant (de alone varies, Q5_K or Q8_0), so this model already
  // takes the fused gate/up path and there is nothing to add here. See
  // laguna-port-state.md for the full investigation.
  const bool gated = ge.type == ue.type &&
                     (ge.type == GgmlType::kQ4_K || ge.type == GgmlType::kQ5_K);
  *scaled = gated;
  const std::uint32_t pair_rows = gated ? PairTileRows() : kPlainTileRows;
  const std::uint32_t down_rows = DownTileRows(de.type);

  qwen38_flash_next::rocm::ExpertCounts(ids.get(), counts.get(), n, experts,
                                        used, stream);
  LAGUNA_HIP(hipMemcpyAsync(h_counts, counts.get(),
                            experts * sizeof(std::uint32_t),
                            hipMemcpyDeviceToHost, stream));
  LAGUNA_HIP(hipEventRecord(ready, stream));
  static const bool kSyncDebug = std::getenv("LAGUNA_SYNC_DEBUG") != nullptr;
  static double sync_total_us = 0.0;
  static std::uint32_t sync_calls = 0;
  if (kSyncDebug) {
    const auto t0 = std::chrono::steady_clock::now();
    LAGUNA_HIP(hipEventSynchronize(ready));
    const auto t1 = std::chrono::steady_clock::now();
    sync_total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
    if (++sync_calls % 47 == 0) {
      std::fprintf(stderr,
                   "[LAGUNA_SYNC_DEBUG] calls=%u cumulative=%.1f us "
                   "mean=%.2f us/call\n",
                   sync_calls, sync_total_us, sync_total_us / sync_calls);
    }
  } else {
    LAGUNA_HIP(hipEventSynchronize(ready));
  }
  std::uint32_t pair_tiles = 0;
  std::uint32_t down_tiles = 0;
  std::int32_t* host_pair = h_tiles;
  std::int32_t* host_down =
      h_tiles + TileCapacityFor(static_cast<std::size_t>(tile) * used, experts);
  for (std::uint32_t e = 0; e < experts; ++e) {
    const std::uint32_t padded =
        (h_counts[e] + kRoutePadRows - 1) / kRoutePadRows * kRoutePadRows;
    for (std::uint32_t j = 0; j * pair_rows < padded; ++j) {
      host_pair[pair_tiles++] = static_cast<std::int32_t>(e | (j << 16));
    }
    for (std::uint32_t j = 0; j * down_rows < padded; ++j) {
      host_down[down_tiles++] = static_cast<std::int32_t>(e | (j << 16));
    }
  }
  if (pair_tiles == 0)
    return Fail(error_msg, "routed experts: no tiles");
  LAGUNA_HIP(hipMemcpyAsync(tiles_pair.get(), host_pair,
                            pair_tiles * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream));
  LAGUNA_HIP(hipMemcpyAsync(tiles_down.get(), host_down,
                            down_tiles * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream));
  qwen38_flash_next::rocm::RoutedCompact(
      ids.get(), counts.get(), route_bounds.get(), route_cursors.get(),
      rows_token.get(), rows_slot.get(), n, used, experts, stream);
  rocm::NarrowRows(cxn, xh, n, c.hidden_size, c.hidden_size, stream);

  using qwen38_flash_next::rocm::RoutedF16Gemm;
  using qwen38_flash_next::rocm::RoutedGatedF16Gemm;
  const auto gate_type = static_cast<WeightType>(ge.type);
  const auto down_type = static_cast<WeightType>(de.type);
  bool ok;
  if (gated) {
    ok = RoutedGatedF16Gemm(ge.data, ue.data, gate_type, xh, tiles_pair.get(),
                            pair_tiles, pair_rows, route_bounds.get(),
                            rows_token.get(), rows_slot.get(), act_half.get(),
                            ge.rows, ge.cols, stream, 1.0F / kRoutedActScale);
  } else {
    ok = RoutedF16Gemm(ge.data, gate_type, xh, tiles_pair.get(), pair_tiles,
                       pair_rows, route_bounds.get(), rows_token.get(),
                       rows_slot.get(), nullptr, gate_e.get(), nullptr, ge.rows,
                       ge.cols, stream) &&
         RoutedF16Gemm(ue.data, static_cast<WeightType>(ue.type), xh,
                       tiles_pair.get(), pair_tiles, pair_rows,
                       route_bounds.get(), rows_token.get(), rows_slot.get(),
                       gate_e.get(), nullptr, act_half.get(), ue.rows, ue.cols,
                       stream);
  }
  if (!ok)
    return Fail(error_msg, "routed gate/up projection failed");
  const bool down_wide = kQ8WideDownTile && down_type == WeightType::kQ8_0;
  const bool down_ok =
      down_wide
          ? qwen38_flash_next::rocm::RoutedF16GemmQ8Wide(
                de.data, act_half.get(), tiles_down.get(), down_tiles,
                route_bounds.get(), rows_slot.get(), rows_slot.get(), nullptr,
                down_e.get(), nullptr, de.rows, de.cols, stream)
          : RoutedF16Gemm(de.data, down_type, act_half.get(), tiles_down.get(),
                          down_tiles, down_rows, route_bounds.get(),
                          rows_slot.get(), rows_slot.get(), nullptr,
                          down_e.get(), nullptr, de.rows, de.cols, stream);
  if (!down_ok) {
    return Fail(error_msg, "routed down projection failed");
  }
  return true;
}

bool Session::Impl::MoeRoute(const DeviceLayer& layer, std::uint32_t n,
                             bool* wide_out, bool* scaled_out,
                             std::string* error_msg) {
  const Config& c = model->config;
  const std::uint32_t used = c.expert_used;
  const bool wide = n * used > kVecExpertRows;
  *wide_out = wide;

  if (ffn_clock != nullptr)
    ffn_clock->Mark();
  if (RouterBlasEnabled() && hipblas_handle != nullptr) {
    const float alpha = 1.0F;
    const float beta = 0.0F;
    const auto status =
        hipblasGemmEx(hipblas_handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                      static_cast<int>(c.expert_count), static_cast<int>(n),
                      static_cast<int>(c.hidden_size), &alpha,
                      static_cast<const float*>(layer.router.data), HIP_R_32F,
                      static_cast<int>(c.hidden_size), cxn, HIP_R_32F,
                      static_cast<int>(c.hidden_size), &beta, router.get(),
                      HIP_R_32F, static_cast<int>(c.expert_count),
                      HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    if (status != HIPBLAS_STATUS_SUCCESS) {
      return Fail(error_msg, "Laguna: router GEMM failed");
    }
  } else {
    rocm::RouterF32(static_cast<const float*>(layer.router.data), cxn,
                    router.get(), n, c.expert_count, c.hidden_size, stream);
  }
  rocm::SigmoidBiasTopK(router.get(),
                        static_cast<const float*>(layer.router_bias.data),
                        ids.get(), topk.get(), n, c.expert_count, used,
                        c.expert_weights_norm, c.expert_weights_scale, stream);
  if (ffn_clock != nullptr)
    ffn_clock->Mark();

  if (!Dense(layer.gate_shexp, cxn, sh_gate.get(), n, error_msg) ||
      !Dense(layer.up_shexp, cxn, sh_up.get(), n, error_msg)) {
    return false;
  }
  rocm::Swiglu(sh_gate.get(), sh_up.get(), std::size_t{n} * c.shared_ff,
               stream);
  if (ffn_clock != nullptr)
    ffn_clock->Mark();

  bool scaled = false;
  if (wide) {
    if (!RoutedExperts(layer, n, &scaled, error_msg))
      return false;
  } else {
    const DeviceTensor& ge = layer.gate_exps;
    const DeviceTensor& ue = layer.up_exps;
    const bool paired =
        ge.type == ue.type && ge.rows == ue.rows && ge.cols == ue.cols;
    if (paired) {
      if (qfn_mmq_moe_vec(static_cast<int>(ge.type), ge.data, cxn, ids.get(),
                          gate_e.get(), static_cast<int>(ge.rows),
                          static_cast<int>(ge.cols), static_cast<int>(n),
                          static_cast<int>(ge.experts), static_cast<int>(used),
                          stream, ue.data, up_e.get()) != 0) {
        return Fail(error_msg, "paired expert projection failed");
      }
    } else if (!Experts(ge, cxn, gate_e.get(), n, used, n, error_msg) ||
               !Experts(ue, cxn, up_e.get(), n, used, n, error_msg)) {
      return false;
    }
    // Non-Q8_0 down projections read the half-precision sum of each Q8_1
    // block, which overflows for very large rows: keep rows inside range.
    scaled = layer.down_exps.type != GgmlType::kQ8_0;
    if (scaled) {
      rocm::SwigluScaled(gate_e.get(), up_e.get(), expert_scale.get(), n * used,
                         c.expert_ff, stream);
    } else {
      rocm::Swiglu(gate_e.get(), up_e.get(),
                   static_cast<std::size_t>(n) * used * c.expert_ff, stream);
    }
    if (!Experts(layer.down_exps, gate_e.get(), down_e.get(), n * used, 1, n,
                 error_msg)) {
      return false;
    }
  }
  *scaled_out = scaled;
  if (ffn_clock != nullptr)
    ffn_clock->Mark();
  return true;
}

bool Session::Impl::MoeFinish(const DeviceLayer& layer, std::uint32_t n,
                              bool wide, bool scaled, std::string* error_msg) {
  const Config& c = model->config;
  const std::uint32_t used = c.expert_used;
  if (!Dense(layer.down_shexp, sh_gate.get(), sh_out.get(), n, error_msg)) {
    return false;
  }
  const float* expert_scale_ptr =
      wide ? expert_const.get() : expert_scale.get();
  rocm::MoeCombine(cx, down_e.get(), topk.get(),
                   scaled ? expert_scale_ptr : nullptr, sh_out.get(), nullptr,
                   n, used, c.hidden_size, stream);
  if (ffn_clock != nullptr)
    ffn_clock->Mark();
  return true;
}

bool Session::Impl::Moe(const DeviceLayer& layer, std::uint32_t n,
                        std::string* error_msg) {
  bool wide = false, scaled = false;
  return MoeRoute(layer, n, &wide, &scaled, error_msg) &&
         MoeFinish(layer, n, wide, scaled, error_msg);
}

bool Session::Impl::Enqueue(std::uint32_t n, bool all_rows,
                            std::uint32_t prefix_end,
                            std::uint32_t* tap_rows_out,
                            std::uint32_t* tap_first_out,
                            std::string* error_msg) {
  // The activation-quantization caches below dedupe repeated Dense() calls
  // against the SAME activation within one forward (Q/K/V off attn_norm's
  // output, gate/up off ffn_norm's). They are keyed by (pointer, n, k), and
  // those triples recur across forwards with different content - most
  // importantly the LM head (last, head_rows, hidden), which is the ONLY
  // q8_vec-cache user in a prefill forward (n>8 routes the layers through
  // the wide cache). Without a reset, every prefill's final logits would be
  // the previous forward's head quantization - stale sampling all around.
  // Invalidate at the top of every forward; intra-forward dedup is kept.
  quant_cache_src = nullptr;
  quant_cache_n = 0;
  quant_cache_k = 0;
  q8_vec_quant_src = nullptr;
  q8_vec_quant_n = 0;
  q8_vec_quant_k = 0;
  q8_tiled_quant_src = nullptr;
  q8_tiled_quant_n = 0;
  q8_tiled_quant_k = 0;
  const Config& c = model->config;
  const DeviceModel& dm = *model->device;
  // Host time spent inside Attention() per prefill chunk. GPU-event marks
  // measure device time, so a gap between them is either device work the
  // marks do not cover or the host failing to keep the queue fed. This tells
  // the two apart.
  static const bool hub_host_debug =
      std::getenv("LAGUNA_HOST_DEBUG") != nullptr;
  static double attn_host_us = 0.0;
  static std::uint32_t attn_host_calls = 0;
  LAGUNA_HIP(hipMemcpyAsync(tokens.get(), h_tokens, n * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream));
  LAGUNA_HIP(hipMemcpyAsync(pos.get(), h_pos, 4 * sizeof(std::uint32_t),
                            hipMemcpyHostToDevice, stream));
  rocm::EmbedQ8(dm.token_embd().data, tokens.get(), cx, n, c.hidden_size,
                stream);
  std::uint32_t tap_rows = 0, tap_first = 0;
  if (dflash != nullptr) {
    // Only the drafter's sliding window matters: capture the rows ending the
    // session prefix and skip forwards that finish far before it.
    //
    // `x` holds only this call's own `n` rows ([0, n) locally, [position,
    // end) in absolute terms) - `tap_first` must be an offset into that
    // local range, not an absolute position. Compute the absolute start of
    // the trailing dflash_window-sized slice first, then clip it into this
    // call's local range; for a full-prompt prefill (position == 0) this
    // reduces to the original absolute math, but for a 1-row decode step
    // (position > 0) it correctly captures just that row instead of
    // reading `end` rows out of a 1-row buffer.
    const std::uint32_t end = position + n;
    if (end + model->dflash_window >= prefix_end) {
      const std::uint32_t want_first =
          end > model->dflash_window ? end - model->dflash_window : 0;
      tap_first = want_first > position ? want_first - position : 0;
      tap_rows = n - tap_first;
      // tap_pos (Forward()'s local, mirroring this) is used post-call.
    }
  }
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const DeviceLayer& L = dm.layers()[il];
    if (layer_clock != nullptr)
      layer_clock->Mark();
    rocm::RmsNormRows(cx, static_cast<const float*>(L.attn_norm.data), cxn, n,
                      c.hidden_size, 1, c.rms_eps, stream);
    if (layer_clock != nullptr)
      layer_clock->Mark();
    const auto attn_host_start = std::chrono::steady_clock::now();
    const bool attn_ok = Attention(il, n, error_msg);
    if (hub_host_debug) {
      attn_host_us += std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - attn_host_start)
                          .count();
      if (++attn_host_calls % 48 == 0) {
        std::fprintf(stderr,
                     "[laguna-host] attention host=%.1f ms/layer over %u\n",
                     attn_host_us / 1000.0 / 48.0, 48U);
        attn_host_us = 0.0;
      }
    }
    if (!attn_ok)
      return false;
    rocm::RmsNormRows(cx, static_cast<const float*>(L.ffn_norm.data), cxn, n,
                      c.hidden_size, 1, c.rms_eps, stream);
    if (layer_clock != nullptr)
      layer_clock->Mark();
    if (!(L.is_moe ? Moe(L, n, error_msg) : DenseFfn(L, n, error_msg)))
      return false;
    if (layer_clock != nullptr)
      layer_clock->Mark();
    if (tap_rows != 0 &&
        std::find(model->dflash_taps.begin(), model->dflash_taps.end(),
                  il + 1) != model->dflash_taps.end()) {
      const auto tap =
          static_cast<std::size_t>(std::find(model->dflash_taps.begin(),
                                             model->dflash_taps.end(), il + 1) -
                                   model->dflash_taps.begin());
      rocm::CaptureTap(
          tap_capture.get(),
          cx + static_cast<std::size_t>(tap_first) * c.hidden_size, tap_rows,
          c.hidden_size,
          static_cast<std::size_t>(model->dflash_taps.size()) * c.hidden_size,
          tap * c.hidden_size, stream);
    }
  }
  const std::uint32_t head_rows = all_rows ? n : 1;
  const float* head_in =
      cx + (all_rows ? 0 : std::size_t{n - 1} * c.hidden_size);
  rocm::RmsNormRows(head_in, static_cast<const float*>(dm.output_norm().data),
                    last.get(), head_rows, c.hidden_size, 1, c.rms_eps, stream);
  if (!Dense(dm.output(), last.get(), d_logits.get(), head_rows, error_msg))
    return false;
  LAGUNA_HIP(
      hipMemcpyAsync(h_logits, d_logits.get(),
                     std::size_t{head_rows} * c.vocab_size * sizeof(float),
                     hipMemcpyDeviceToHost, stream));
  *tap_rows_out = tap_rows;
  *tap_first_out = tap_first;
  return true;
}

bool Session::Impl::Forward(std::span<const std::int32_t> input, bool all_rows,
                            std::uint32_t prefix_end, std::string* error_msg) {
  const auto n = static_cast<std::uint32_t>(input.size());
  std::copy(input.begin(), input.end(), h_tokens);
  h_pos[0] = position;
  logit_rows = all_rows ? n : 1;

  // Same tap-window math as Enqueue() (duplicated, not shared, since it must
  // run every call - including replayed ones, where Enqueue()'s body does
  // not re-execute - to decide graphability and drive DflashInject below).
  std::uint32_t tap_rows = 0, tap_first = 0, tap_pos = 0;
  if (dflash != nullptr) {
    const std::uint32_t end = position + n;
    if (end + model->dflash_window >= prefix_end) {
      const std::uint32_t want_first =
          end > model->dflash_window ? end - model->dflash_window : 0;
      tap_first = want_first > position ? want_first - position : 0;
      tap_rows = n - tap_first;
      tap_pos = position + tap_first;
    }
  }

  // Captured HIP graph per decode/verify width: the whole per-step kernel
  // sequence (Enqueue) launches once from host code, then replays as a
  // single hipGraphLaunch - cuts per-kernel launch overhead, the dominant
  // cost at n=1..16 on gfx1151. Only when tap_rows==n (tap_first==0, the
  // DFlash drafter's tap window has reached steady state - always true for
  // plain decode since dflash==nullptr keeps tap_rows at 0==n only when
  // n==0, which never happens, so this also correctly excludes plain decode
  // from ever reading a stale tap_rows==n check... ) - see the StepGraph
  // comment at its declaration for the full safety argument.
  const bool steady = dflash == nullptr || tap_rows == n;
  StepGraph& g = graphs[std::min<std::uint32_t>(n, Session::kMaxVerifyRows)];
  // Only capture once past the split-K depth threshold: the per-layer
  // Attention() dispatch is a host-side decision baked into the recorded
  // graph, so a graph captured at shallow depth would replay the non-split
  // kernel forever, exactly where split-K starts mattering.
  const bool graphable = all_rows && n <= Session::kMaxVerifyRows && steady &&
                         position + n >= kSplitAttnMinDepth;
  std::uint32_t enq_tap_rows = tap_rows, enq_tap_first = tap_first;
  if (graphable)
    ++g.steps;
  if (graphable && g.steps > 1 && g.exec == nullptr && !g.failed) {
    LAGUNA_HIP(hipStreamSynchronize(stream));
    LAGUNA_HIP(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed));
    std::string capture_error;
    const bool enqueued = Enqueue(n, all_rows, prefix_end, &enq_tap_rows,
                                  &enq_tap_first, &capture_error);
    hipGraph_t graph = nullptr;
    const hipError_t ended = hipStreamEndCapture(stream, &graph);
    if (!enqueued || ended != hipSuccess ||
        hipGraphInstantiate(&g.exec, graph, nullptr, nullptr, 0) !=
            hipSuccess) {
      g.failed = true;
      g.exec = nullptr;
      (void)hipGetLastError();
      std::fprintf(stderr, "[laguna] decode graph unavailable: %s\n",
                   capture_error.c_str());
    }
    if (graph != nullptr)
      (void)hipGraphDestroy(graph);
  }
  if (graphable && g.exec != nullptr) {
    LAGUNA_HIP(hipGraphLaunch(g.exec, stream));
  } else if (!Enqueue(n, all_rows, prefix_end, &enq_tap_rows, &enq_tap_first,
                      error_msg)) {
    return false;
  }

  // Block-sparse attention's pooled-key maintenance, run live every call
  // (including ones a captured graph will replay) - see the long comment
  // on the equivalent, capture-skipped call inside Attention() for why
  // this can't just live there for the graphed decode path.
  //
  // It has to run AFTER this call's KV rows are written. The pooled
  // per-block key bounds are read out of k_cache, and a bound computed
  // from rows that were never written is not a bound on anything: every
  // block's summary was whatever the fresh allocation happened to hold,
  // and the Quest bound it produced selected the wrong blocks. That is
  // not a subtle quality loss - sparse attention missed a needle that
  // dense attention found at 33k tokens.
  //
  // Pooling before the layer loop, as this used to, read rows the layer
  // loop was about to write.
  {
    const Config& c = model->config;
    const std::uint32_t end = position + n;
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      if (sparse_pooled[il].get() == nullptr)
        continue;
      rocm::UpdatePooledKeysLaguna(
          k_cache[il].get(), sparse_pooled[il].get(), sparse_valid_end[il], end,
          c.num_kv, c.head_dim, sparse_max_blocks, stream, kv_cache_type);
      sparse_valid_end[il] = end;
    }
  }

  LAGUNA_HIP(hipStreamSynchronize(stream));
  {
    // Prefill chunks are reported. A graph-replayed decode step never reaches
    // Enqueue, so any marks it could have left are dropped rather than
    // accumulating across a generation.
    const bool report = n > Session::kMaxVerifyRows;
    if (report && (layer_clock != nullptr || ffn_clock != nullptr)) {
      std::fprintf(stderr, "[laguna] forward n=%u start_pos=%u\n", n, position);
    }
    if (layer_clock != nullptr) {
      if (report) {
        layer_clock->Report("laguna");
      } else {
        layer_clock->Discard();
      }
    }
    if (ffn_clock != nullptr) {
      if (report) {
        ffn_clock->Report("laguna-ffn");
      } else {
        ffn_clock->Discard();
      }
    }
  }
  position += n;
  {
    // A forward that produces non-finite logits fails later, in the
    // sampler, as "logit distribution contains no finite values" - which
    // reads as a model fault and hides a broken device or a bad
    // allocation. Catch it here, where the position and the GPU's own
    // state are known.
    //
    // Only the rows that will be sampled matter: the last row always, and
    // the rest for the narrow verify widths. A full prefill carries
    // thousands of rows and scanning them all would cost more than the
    // forward.
    const std::size_t vocab = model->config.vocab_size;
    const bool scan_all = logit_rows <= Session::kMaxVerifyRows;
    std::size_t bad = 0;
    float first = 0.0F;
    const auto scan_row = [&](std::size_t row) {
      for (std::size_t t = 0; t < vocab; ++t) {
        const float value = h_logits[row * vocab + t];
        if (!std::isfinite(value)) {
          if (bad == 0) {
            first = value;
          }
          ++bad;
        }
      }
    };
    if (scan_all) {
      for (std::size_t r = 0; r < logit_rows; ++r) {
        scan_row(r);
      }
    } else {
      scan_row(logit_rows - 1);
    }
    if (bad != 0) {
      const auto error = hipGetLastError();
      std::string detail = "Laguna: the forward produced non-finite logits";
      detail += " at position " + std::to_string(position);
      detail += " (" + std::to_string(bad) + " of " +
                std::to_string(logit_rows * vocab) + " values,";
      detail += " first=" + std::to_string(static_cast<double>(first));
      detail += "). ";
      if (error != hipSuccess) {
        detail += "The GPU reported " + std::string(hipGetErrorString(error)) +
                  " - the device is in an error state. Restart the server; if "
                  "it persists, reset the GPU.";
        (void)hipGetLastError();
      } else {
        detail +=
            "The GPU reported no error, so the device is still healthy and "
            "the model read memory it should not have - usually a KV cache "
            "larger than the device can back, or a device that has not "
            "settled after the previous server released it. Restart the "
            "server and wait about a minute before starting it again; use a "
            "smaller context if it keeps happening.";
      }
      if (std::getenv("LAGUNA_NAN_DEBUG") != nullptr) {
        std::fprintf(stderr,
                     "[LAGUNA_NAN_DEBUG] forward produced non-finite logits: "
                     "rows=%u nonfinite=%zu first=%g position=%u n=%u\n",
                     logit_rows, bad, static_cast<double>(first), position, n);
      }
      return Fail(error_msg, detail);
    }
  }
  if (tap_rows != 0) {
    tap_pending = tap_rows;
    tap_pending_pos = tap_pos;
    tap_pending_first = 0;
    if (inject_taps && !DflashInject(tap_pending, error_msg))
      return false;
  }
  return true;
}

Session::Session() : impl_(std::make_unique<Impl>()) {}
Session::~Session() = default;

bool Session::Append(std::span<const std::int32_t> tokens,
                     std::string* error_msg) {
  if (tokens.empty())
    return Fail(error_msg, "Laguna: no tokens to append");
  if (impl_->position + tokens.size() > impl_->capacity) {
    return Fail(error_msg, "Laguna: context capacity exceeded");
  }
  const std::uint32_t prefix_end =
      impl_->position + static_cast<std::uint32_t>(tokens.size());
  std::size_t off = 0;
  for (; off < tokens.size();) {
    const std::size_t n =
        std::min<std::size_t>(tokens.size() - off, impl_->tile);
    // all_rows=true when n==1 (plain single-token decode): identical to
    // "last row only" at n=1, but makes the step graphable (see Forward()'s
    // StepGraph comment) - this is what gives plain decode a flat per-step
    // graph replay instead of per-kernel launch overhead every step.
    const bool single = tokens.size() == 1;
    if (!impl_->Forward(tokens.subspan(off, n), single, prefix_end, error_msg))
      return false;
    off += n;
  }
  return true;
}

bool Session::AppendAll(std::span<const std::int32_t> tokens,
                        std::string* error_msg) {
  if (tokens.empty() || tokens.size() > kMaxVerifyRows) {
    return Fail(error_msg, "Laguna: unsupported verification width");
  }
  if (impl_->position + tokens.size() > impl_->capacity) {
    return Fail(error_msg, "Laguna: context capacity exceeded");
  }
  impl_->inject_taps = false;
  const bool ok = impl_->Forward(
      tokens, true, impl_->position + static_cast<std::uint32_t>(tokens.size()),
      error_msg);
  impl_->inject_taps = true;
  return ok;
}

void Session::Rewind(std::uint32_t position) noexcept {
  if (position < impl_->position) {
    impl_->position = position;
    // The drafter's injected context must follow the target's KV rewind.
    // Leaving it ahead makes the next injection land behind the drafter's
    // recorded frontier, which InjectTargetContextDevice rejects.
    if (impl_->dflash != nullptr) {
      impl_->dflash->RewindInjectedContext(position);
    }
    // A pooled block covering rows past the rewound position is stale (it
    // pooled KV that is about to be overwritten by a later Append) - drop
    // back to the block boundary at or below `position` so the next
    // UpdatePooledKeysLaguna call re-pools it from scratch instead of
    // trusting a min/max computed from since-discarded rows.
    for (auto& end : impl_->sparse_valid_end)
      end = std::min(end, (position / 64) * 64);
  }
}

std::span<const float> Session::Logits() const noexcept {
  const auto vocab = impl_->model->config.vocab_size;
  return {impl_->h_logits + std::size_t{impl_->logit_rows - 1} * vocab, vocab};
}

std::span<const float> Session::Logits(std::uint32_t row) const noexcept {
  const auto vocab = impl_->model->config.vocab_size;
  return {impl_->h_logits + std::size_t{row} * vocab, vocab};
}

std::uint32_t Session::Position() const noexcept {
  return impl_->position;
}

std::uint32_t Session::Capacity() const noexcept {
  return impl_->capacity;
}

void Session::Reset() noexcept {
  impl_->position = 0;
  std::fill(impl_->sparse_valid_end.begin(), impl_->sparse_valid_end.end(), 0);
  if (impl_->dflash != nullptr)
    impl_->dflash->Reset();
}

hip::QwenDFlashGpuExecutor* Session::DflashExecutor() const noexcept {
  return impl_->dflash.get();
}

bool Session::DflashInject(std::uint32_t rows, std::string* error_msg) {
  return impl_->DflashInject(rows, error_msg);
}

bool Session::Impl::DflashInject(std::uint32_t rows, std::string* error_msg) {
  if (dflash == nullptr)
    return true;
  if (rows > tap_pending) {
    tap_pending = 0;
    return Fail(error_msg, "Laguna: drafter capture out of range");
  }
  if (rows == 0)
    return true;
  if (!dflash->InjectTargetContextDevice(
          tap_capture.get() + static_cast<std::size_t>(tap_pending_first) *
                                  model->dflash_taps.size() *
                                  model->config.hidden_size,
          tap_pending_pos, rows)) {
    tap_pending = 0;
    return Fail(error_msg, "Laguna: drafter context injection failed");
  }
  tap_pending -= rows;
  tap_pending_pos += rows;
  tap_pending_first += rows;
  return true;
}

std::size_t Session::AllocatedBytes() const noexcept {
  std::size_t total = 0;
  for (const auto& b : impl_->k_cache)
    total += b.bytes();
  for (const auto& b : impl_->v_cache)
    total += b.bytes();
  return total;
}

std::size_t Session::LiveRows(std::uint32_t il,
                              std::uint32_t position) const noexcept {
  const Config& c = impl_->model->config;
  if (!c.IsSwa(il))
    return position;
  // A SWA layer only ever reads the rows its window can still reach.
  const std::size_t window = c.sliding_window;
  if (window == 0 || window >= position)
    return position;
  return window;
}

std::uint32_t Session::LiveFirstRow(std::uint32_t il,
                                    std::uint32_t position) const noexcept {
  const std::size_t rows = LiveRows(il, position);
  return static_cast<std::uint32_t>(position - rows);
}

std::size_t Session::SnapshotBytes() const noexcept {
  const Config& c = impl_->model->config;
  const std::size_t kv_row_bytes = impl_->kv_row_bytes;
  const std::size_t position = impl_->position;
  std::size_t total = kSnapshotHeaderBytes;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    // The per-layer row count is stored so a restore can detect a payload
    // whose window no longer matches this session's.
    total += sizeof(std::uint32_t);
    total += 2 * LiveRows(il, position) * kv_row_bytes;
    total +=
        impl_->PooledSnapshotBytes(il, static_cast<std::uint32_t>(position));
  }
  total += std::size_t{c.vocab_size} * sizeof(float);
  // The drafter's injected context is part of the session: a restore that
  // leaves it behind makes the next injection land in front of the
  // drafter's frontier and get rejected.
  if (impl_->dflash != nullptr) {
    total += impl_->dflash->PersistentPayloadBytes();
  }
  return total;
}

std::size_t Session::MaxSnapshotBytes() const noexcept {
  const Config& c = impl_->model->config;
  const std::size_t kv_row_bytes = impl_->kv_row_bytes;
  const std::uint32_t position = impl_->capacity;
  std::size_t total = kSnapshotHeaderBytes;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    total += sizeof(std::uint32_t);
    total += 2 * LiveRows(il, position) * kv_row_bytes;
    total += impl_->PooledSnapshotBytes(il, position);
  }
  total += std::size_t{c.vocab_size} * sizeof(float);
  if (impl_->dflash != nullptr) {
    total += impl_->dflash->PersistentPayloadBytes();
  }
  return total;
}

bool Session::SaveSnapshot(std::span<std::uint8_t> out,
                           std::string* error_msg) const {
  const Config& c = impl_->model->config;
  const std::size_t kv_row = std::size_t{c.num_kv} * c.head_dim;
  const std::size_t kv_row_bytes = impl_->kv_row_bytes;
  const std::size_t position = impl_->position;
  if (position == 0)
    return Fail(error_msg, "Laguna: snapshot needs a non-empty context");
  if (out.size() != SnapshotBytes())
    return Fail(error_msg, "Laguna: snapshot buffer has the wrong size");
  if (hipStreamSynchronize(impl_->stream) != hipSuccess)
    return Fail(error_msg, "Laguna: device synchronization failed");

  const std::uint32_t header[9] = {
      kSnapshotMagic,
      Session::kSnapshotPayloadVersion,
      static_cast<std::uint32_t>(position),
      c.num_layers,
      static_cast<std::uint32_t>(kv_row),
      c.vocab_size,
      c.sliding_window,
      static_cast<std::uint32_t>(impl_->dflash != nullptr
                                     ? impl_->dflash->PersistentPayloadBytes()
                                     : 0),
      static_cast<std::uint32_t>(impl_->kv_cache_type),
  };
  std::memcpy(out.data(), header, sizeof(header));
  std::uint8_t* at = out.data() + kSnapshotHeaderBytes;

  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const std::size_t rows = LiveRows(il, position);
    const std::uint32_t row_count = static_cast<std::uint32_t>(rows);
    std::memcpy(at, &row_count, sizeof(row_count));
    at += sizeof(row_count);
    if (rows == 0)
      continue;
    const std::size_t layer_bytes = rows * kv_row_bytes;
    for (const auto* cache : {&impl_->k_cache[il], &impl_->v_cache[il]}) {
      // A full-attention layer's live rows are contiguous in the device
      // cache. An SWA layer's live window wraps its ring, so it is copied
      // as up to two segments.
      const std::uint32_t first = LiveFirstRow(il, position);
      if (impl_->ring[il] == 0) {
        if (hipMemcpy(at, cache->get() + std::size_t{first} * kv_row_bytes,
                      layer_bytes, hipMemcpyDeviceToHost) != hipSuccess)
          return Fail(error_msg, "Laguna: snapshot copy failed");
        at += layer_bytes;
        continue;
      }
      const std::uint32_t ring_rows =
          static_cast<std::uint32_t>(cache->bytes() / kv_row_bytes);
      std::size_t copied = 0;
      while (copied < rows) {
        const std::uint32_t start = (first + copied) % ring_rows;
        const std::size_t chunk =
            std::min<std::size_t>(rows - copied, ring_rows - start);
        if (hipMemcpy(at, cache->get() + std::size_t{start} * kv_row_bytes,
                      chunk * kv_row_bytes,
                      hipMemcpyDeviceToHost) != hipSuccess)
          return Fail(error_msg, "Laguna: snapshot copy failed");
        at += chunk * kv_row_bytes;
        copied += chunk;
      }
    }
    // The pooled block bounds follow the layer's rows: they are derived from
    // the keys, so a restore that does not carry them has to re-derive them
    // from the whole KV cache.
    const std::size_t pooled_bytes =
        impl_->PooledSnapshotBytes(il, static_cast<std::uint32_t>(position));
    if (pooled_bytes != 0) {
      impl_->SavePooled(il, static_cast<std::uint32_t>(position), at);
      at += pooled_bytes;
    }
  }
  // The last row holds the distribution of the final token. A verify pass
  // (AppendAll) leaves several rows; only the final one is reusable.
  std::memcpy(
      at, impl_->h_logits + std::size_t{impl_->logit_rows - 1} * c.vocab_size,
      std::size_t{c.vocab_size} * sizeof(float));
  if (impl_->dflash != nullptr) {
    at += std::size_t{c.vocab_size} * sizeof(float);
    const auto dflash_bytes = impl_->dflash->PersistentPayloadBytes();
    if (at + dflash_bytes != out.data() + out.size()) {
      return Fail(error_msg, "Laguna: drafter snapshot size mismatch");
    }
    if (!impl_->dflash->SavePersistentSnapshot(
            std::span<std::uint8_t>(at, dflash_bytes), error_msg)) {
      return Fail(error_msg, "Laguna: drafter snapshot copy failed");
    }
  }
  return true;
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error_msg) {
  const Config& c = impl_->model->config;
  const std::size_t kv_row = std::size_t{c.num_kv} * c.head_dim;
  const std::size_t kv_row_bytes = impl_->kv_row_bytes;
  std::uint32_t header[9];
  if (payload.size() < kSnapshotHeaderBytes)
    return Fail(error_msg, "Laguna: snapshot payload is too short");
  std::memcpy(header, payload.data(), sizeof(header));
  const std::uint32_t position = header[2];
  if (header[0] != kSnapshotMagic ||
      header[1] != Session::kSnapshotPayloadVersion ||
      header[3] != c.num_layers || header[4] != kv_row ||
      header[5] != c.vocab_size || header[6] != c.sliding_window ||
      header[8] != static_cast<std::uint32_t>(impl_->kv_cache_type) ||
      position == 0 || position > impl_->capacity) {
    return Fail(error_msg, "Laguna: snapshot does not match this session");
  }
  std::size_t expected = kSnapshotHeaderBytes;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    expected +=
        sizeof(std::uint32_t) + 2 * LiveRows(il, position) * kv_row_bytes;
    expected += impl_->PooledSnapshotBytes(il, position);
  }
  expected += std::size_t{c.vocab_size} * sizeof(float);
  const std::size_t dflash_bytes = header[7];
  if (dflash_bytes != 0 &&
      (impl_->dflash == nullptr || expected > payload.size() ||
       dflash_bytes > payload.size() - expected)) {
    return Fail(error_msg, "Laguna: snapshot drafter section is invalid");
  }
  expected += dflash_bytes;
  if (payload.size() != expected)
    return Fail(error_msg, "Laguna: snapshot payload has the wrong size");
  if (hipStreamSynchronize(impl_->stream) != hipSuccess)
    return Fail(error_msg, "Laguna: device synchronization failed");

  const std::uint8_t* at = payload.data() + kSnapshotHeaderBytes;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    std::uint32_t row_count = 0;
    std::memcpy(&row_count, at, sizeof(row_count));
    at += sizeof(row_count);
    const std::size_t rows = LiveRows(il, position);
    if (row_count != rows)
      return Fail(error_msg, "Laguna: snapshot layer row count mismatch");
    if (rows == 0)
      continue;
    const std::size_t layer_bytes = rows * kv_row_bytes;
    for (auto* cache : {&impl_->k_cache[il], &impl_->v_cache[il]}) {
      const std::uint32_t first = LiveFirstRow(il, position);
      if (impl_->ring[il] == 0) {
        if (hipMemcpy(cache->get() + std::size_t{first} * kv_row_bytes, at,
                      layer_bytes, hipMemcpyHostToDevice) != hipSuccess)
          return Fail(error_msg, "Laguna: snapshot restore copy failed");
        at += layer_bytes;
        continue;
      }
      const std::uint32_t ring_rows =
          static_cast<std::uint32_t>(cache->bytes() / kv_row_bytes);
      std::size_t copied = 0;
      while (copied < rows) {
        const std::uint32_t start = (first + copied) % ring_rows;
        const std::size_t chunk =
            std::min<std::size_t>(rows - copied, ring_rows - start);
        if (hipMemcpy(cache->get() + std::size_t{start} * kv_row_bytes, at,
                      chunk * kv_row_bytes,
                      hipMemcpyHostToDevice) != hipSuccess)
          return Fail(error_msg, "Laguna: snapshot restore copy failed");
        at += chunk * kv_row_bytes;
        copied += chunk;
      }
    }
    const std::size_t pooled_bytes = impl_->PooledSnapshotBytes(il, position);
    if (pooled_bytes != 0) {
      impl_->RestorePooled(il, position, at);
      at += pooled_bytes;
    }
  }
  std::memcpy(impl_->h_logits, at, std::size_t{c.vocab_size} * sizeof(float));
  impl_->logit_rows = 1;
  impl_->position = position;
  if (impl_->dflash != nullptr) {
    at += std::size_t{c.vocab_size} * sizeof(float);
    if (dflash_bytes == 0) {
      // No drafter section: the frontier has to go back to the start of the
      // context, or the next injection lands behind it.
      impl_->dflash->Reset();
    } else {
      try {
        impl_->dflash->RestorePersistentSnapshot(
            std::span<const std::uint8_t>(at, dflash_bytes));
      } catch (const std::exception& error) {
        return Fail(error_msg, std::string("Laguna: drafter restore failed: ") +
                                   error.what());
      }
    }
  }
  // The pooled block bounds came with the snapshot. Rebuild them only if
  // this session has no pooled buffers to restore into - a session with
  // sparse attention off has none, and then nothing was carried either.
  if (impl_->sparse_max_blocks == 0 || impl_->sparse_pooled.empty()) {
    RebuildSparsePooled();
  }
  return true;
}

// The pooled block bounds are derived from the keys, so they are session
// state rather than a cache: a snapshot that does not carry them has to
// re-derive them from the whole restored KV cache, which is O(context) and
// dominates the restore at deep context (measured 215-600 ms at 158k
// tokens). Carrying them turns the restore into a copy.
//
// The snapshot stores the blocks that exist at the snapshot's position, in
// a contiguous [kv_head][block][256] layout. The device buffer strides by
// its own block count, so a restore copies one segment per kv head.
std::size_t Session::Impl::PooledSnapshotBytes(
    std::uint32_t il, std::uint32_t position) const noexcept {
  if (sparse_pooled.empty() || sparse_max_blocks == 0)
    return 0;
  if (il >= sparse_pooled.size() || sparse_pooled[il].get() == nullptr)
    return 0;
  const std::uint32_t blocks =
      std::min<std::uint32_t>((position + 63) / 64, sparse_max_blocks);
  if (blocks == 0)
    return 0;
  return static_cast<std::size_t>(model->config.num_kv) * blocks * 256 *
         sizeof(__half);
}

std::size_t Session::Impl::PooledBlocks(std::uint32_t position) const noexcept {
  if (sparse_max_blocks == 0)
    return 0;
  return std::min<std::uint32_t>((position + 63) / 64, sparse_max_blocks);
}

void Session::Impl::SavePooled(std::uint32_t il, std::uint32_t position,
                               std::uint8_t* out) const {
  const std::uint32_t blocks = PooledBlocks(position);
  if (blocks == 0 || sparse_pooled[il].get() == nullptr)
    return;
  const std::size_t kv = model->config.num_kv;
  const std::size_t stride = static_cast<std::size_t>(sparse_max_blocks) * 256;
  for (std::size_t head = 0; head < kv; ++head) {
    if (hipMemcpy(out + head * blocks * 256 * sizeof(__half),
                  sparse_pooled[il].get() + head * stride,
                  static_cast<std::size_t>(blocks) * 256 * sizeof(__half),
                  hipMemcpyDeviceToHost) != hipSuccess) {
      return;
    }
    out += static_cast<std::size_t>(blocks) * 256 * sizeof(__half);
  }
}

void Session::Impl::RestorePooled(std::uint32_t il, std::uint32_t position,
                                  const std::uint8_t* in) {
  const std::uint32_t blocks = PooledBlocks(position);
  if (blocks == 0 || sparse_pooled[il].get() == nullptr)
    return;
  const std::size_t kv = model->config.num_kv;
  const std::size_t stride = static_cast<std::size_t>(sparse_max_blocks) * 256;
  for (std::size_t head = 0; head < kv; ++head) {
    if (hipMemcpy(sparse_pooled[il].get() + head * stride,
                  in + head * blocks * 256 * sizeof(__half),
                  static_cast<std::size_t>(blocks) * 256 * sizeof(__half),
                  hipMemcpyHostToDevice) != hipSuccess) {
      return;
    }
    in += static_cast<std::size_t>(blocks) * 256 * sizeof(__half);
  }
  // The restored bounds cover every block that exists at this position, so
  // the next append only has to pool the tail of the last partial block.
  sparse_valid_end[il] = position;
}

void Session::RebuildSparsePooled() {
  const Config& c = impl_->model->config;
  if (impl_->sparse_pooled.empty() || impl_->sparse_max_blocks == 0)
    return;
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    if (impl_->sparse_pooled[il].get() == nullptr)
      continue;
    // A full-attention layer's cache is contiguous, so the whole range can
    // be re-pooled from scratch.
    rocm::UpdatePooledKeysLaguna(
        impl_->k_cache[il].get(), impl_->sparse_pooled[il].get(), 0,
        impl_->position, c.num_kv, c.head_dim, impl_->sparse_max_blocks,
        impl_->stream, impl_->kv_cache_type);
    impl_->sparse_valid_end[il] = impl_->position;
  }
  (void)hipStreamSynchronize(impl_->stream);
}

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const EngineOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> model(new Model());
  auto impl = std::make_shared<ModelState>();
  impl->options = options;
  if (const char* t = std::getenv("LAGUNA_PREFILL_TILE")) {
    const int v = std::atoi(t);
    if (v > 0)
      impl->options.prefill_tile = static_cast<std::uint32_t>(v);
  }
  // The KV cache format. q8_0 stores an ordinary K/V row in 53.125% of the
  // F16 bytes, at the cost of a lossy round trip; see docs/kv-codec.md.
  // F16 stays the default.
  if (const char* kv = std::getenv("LAGUNA_KV_CACHE_TYPE")) {
    if (std::strcmp(kv, "q8_0") == 0) {
      impl->options.kv_cache_type = KvCacheType::kQ8_0;
    } else if (std::strcmp(kv, "f16") != 0) {
      Fail(error_msg,
           "LAGUNA_KV_CACHE_TYPE must be f16 or q8_0, got: " + std::string(kv));
      return nullptr;
    }
  }
  std::string err;
  impl->reader = core::GgufReader::OpenFile(model_path, &err);
  if (!impl->reader) {
    Fail(error_msg, "failed to open GGUF: " + err);
    return nullptr;
  }
  const auto config = ParseConfig(*impl->reader, &err);
  if (!config) {
    Fail(error_msg, "failed to parse Laguna config: " + err);
    return nullptr;
  }
  impl->config = *config;
  impl->device = DeviceModel::Upload(*impl->reader, impl->config, &err);
  if (!impl->device) {
    Fail(error_msg, "failed to upload Laguna weights: " + err);
    return nullptr;
  }
  if (!options.dflash_model_path.empty()) {
    auto dflash_reader =
        core::GgufReader::OpenFile(options.dflash_model_path, &err);
    if (!dflash_reader) {
      Fail(error_msg, "failed to open the drafter: " + err);
      return nullptr;
    }
    std::string dflash_err;
    impl->dflash_model = hip::QwenDFlashGpuModel::Create(
        std::move(dflash_reader), impl->config.hidden_size,
        impl->config.vocab_size, impl->config.num_layers,
        TensorRef(impl->device->token_embd()),
        TensorRef(impl->device->output()), &dflash_err);
    if (!impl->dflash_model) {
      Fail(error_msg, "failed to load the drafter: " + dflash_err);
      return nullptr;
    }
    const auto& dcfg = impl->dflash_model->GetDFlashConfig();
    impl->dflash_taps = dcfg.target_layer_ids;
    impl->drafter_block =
        std::min<std::uint32_t>(dcfg.block_size, Session::kMaxVerifyRows);
    impl->dflash_window =
        dcfg.sliding_window == 0
            ? options.max_context
            : std::min<std::uint32_t>(dcfg.sliding_window, options.max_context);
    if (!impl->dflash_taps.empty() &&
        impl->dflash_taps.back() >= impl->config.num_layers) {
      Fail(error_msg, "the drafter taps exceed the Laguna layer count");
      return nullptr;
    }
  }
  // Release the GGUF mmap now that all weights are on the device: keeping it
  // alive pins ~model-size host pages and trips the SVM/resident-memory
  // limit. Neither the DFlash drafter (its own reader, opened above) nor
  // anything else needs Laguna's own reader past this point.
  impl->reader.reset();
  model->state_ = std::move(impl);
  return model;
}

std::unique_ptr<Session> Model::CreateSession(std::uint32_t max_context,
                                              std::string* error_msg) {
  std::unique_ptr<Session> session(new Session());
  session->impl_->model = state_;
  if (!session->impl_->Init(max_context, error_msg))
    return nullptr;
  if (HasDflash()) {
    session->impl_->dflash = hip::QwenDFlashGpuExecutor::Create(
        state_->dflash_model,
        std::min<std::uint32_t>(max_context, state_->options.max_context),
        error_msg);
    if (!session->impl_->dflash) {
      Fail(error_msg, "Laguna: drafter executor creation failed");
      return nullptr;
    }
    const std::size_t taps = state_->dflash_taps.size();
    if (taps != 0 &&
        !session->impl_->tap_capture.Allocate(
            std::size_t{state_->dflash_window + Session::kMaxVerifyRows} *
            taps * state_->config.hidden_size)) {
      Fail(error_msg, "Laguna: drafter capture allocation failed");
      return nullptr;
    }
  }
  const std::int32_t warm[kWarmTokens] = {};
  for (const std::uint32_t width : {1U, kWarmTokens}) {
    if (width > max_context)
      break;
    if (!session->impl_->Forward({warm, width}, false, width, error_msg)) {
      // A forward that produced non-finite logits means the device read
      // memory it should not have. Let that fail the session: the caller
      // can tell this apart from a structural failure and retry once the
      // device has settled, which is what it needs. Swallowing it here
      // would hand the caller a session that produces garbage.
      return nullptr;
    }
    session->impl_->position = 0;
    if (session->impl_->dflash != nullptr) {
      session->impl_->dflash->Reset();
      session->impl_->tap_pending = 0;
      session->impl_->tap_pending_pos = 0;
      session->impl_->tap_pending_first = 0;
    }
  }
  return session;
}

const Config& Model::config() const noexcept {
  return state_->config;
}

bool Model::HasDflash() const noexcept {
  return state_->dflash_model != nullptr;
}

std::uint32_t Model::DrafterBlockSize() const noexcept {
  return state_->drafter_block;
}

std::size_t Model::ResidentBytes() const noexcept {
  return state_->device->resident_bytes();
}

std::size_t Model::SessionBytesForContext(
    std::uint32_t context) const noexcept {
  const std::uint32_t tile =
      std::max<std::uint32_t>(1, state_->options.prefill_tile);
  return SessionBytesForContextLocked(state_->config, context, tile,
                                      state_->options.kv_cache_type);
}

std::size_t Model::FreeDeviceBytes() noexcept {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
    return 0;
  }
  return free_bytes;
}

}  // namespace gufo::models::laguna::gpu
