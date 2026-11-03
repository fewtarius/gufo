#ifndef GUFO_MODELS_LAGUNA_GPU_ENGINE_HPP_
#define GUFO_MODELS_LAGUNA_GPU_ENGINE_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "src/models/laguna/runtime/config.hpp"

namespace gufo::hip {
class QwenDFlashGpuExecutor;
class QwenDFlashGpuModel;
}  // namespace gufo::hip

namespace gufo::models::laguna::gpu {

/// How the attention K/V cache is stored.
enum class KvCacheType : std::uint8_t {
  /// Half precision. Two bytes per element. The default.
  kF16 = 0,
  /// One binary16 scale and 32 signed codes per 32 elements: 53.125% of the
  /// F16 bytes, at the cost of a lossy round trip. The layout and the
  /// numeric contract are in docs/kv-codec.md.
  kQ8_0 = 1,
};

struct EngineOptions {
  /// Tokens one prefill pass processes.
  std::uint32_t prefill_tile{4096};
  /// Optional DFlash-2 drafter; its trunk taps bind to this model.
  std::string dflash_model_path;
  /// Upper bound for session contexts; bounds the drafter's context.
  std::uint32_t max_context{65536};
  /// Storage format for the attention K/V cache.
  KvCacheType kv_cache_type{KvCacheType::kF16};
};

class Model;
struct ModelState;

/// One conversation: KV cache, position and the logits of the last token.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Appends `tokens` and leaves the logits of the last one in Logits().
  [[nodiscard]] bool Append(std::span<const std::int32_t> tokens,
                            std::string* error_msg);

  /// Appends 1..kMaxVerifyRows tokens and keeps the logits of every row:
  /// row i is the distribution after tokens[i]. Used to verify drafts.
  [[nodiscard]] bool AppendAll(std::span<const std::int32_t> tokens,
                               std::string* error_msg);
  static constexpr std::uint32_t kMaxVerifyRows = 16;

  /// Drops cache rows past `position`; later appends overwrite them.
  void Rewind(std::uint32_t position) noexcept;

  /// DFlash-2 executor bound to this session, or null when no drafter was
  /// loaded. Multi-row AppendAll leaves the drafter un-injected; the caller
  /// commits the accepted prefix with DflashInject after verification.
  [[nodiscard]] hip::QwenDFlashGpuExecutor* DflashExecutor() const noexcept;
  /// Injects the first `rows` captured rows of the last forward into the
  /// drafter and returns true when its context covers `Position()`.
  bool DflashInject(std::uint32_t rows, std::string* error_msg);

  [[nodiscard]] std::span<const float> Logits() const noexcept;
  /// Row `row` of the last AppendAll (or Append of up to kMaxVerifyRows).
  [[nodiscard]] std::span<const float> Logits(std::uint32_t row) const noexcept;
  [[nodiscard]] std::uint32_t Position() const noexcept;
  [[nodiscard]] std::uint32_t Capacity() const noexcept;

  /// Forgets everything: position 0, empty cache.
  void Reset() noexcept;

  /// Device bytes of the KV caches (scratch excluded).
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;

  /// Host bytes SaveSnapshot needs for the current position.
  [[nodiscard]] std::size_t SnapshotBytes() const noexcept;

  /// Copies the KV caches and the last logits to host memory. The full
  /// attention layers contribute their live rows; the sliding-window layers
  /// contribute only the rows their window can still reach.
  [[nodiscard]] bool SaveSnapshot(std::span<std::uint8_t> out,
                                  std::string* error_msg) const;

  /// Restores a snapshot into this session. The payload must come from a
  /// session of the same model with the same context capacity.
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                                     std::string* error_msg);

  /// Bumped when the payload layout changes.
  /// Bumped when the payload layout changes. Version 2 records the KV cache
  /// format in the header, so a snapshot taken in one format is rejected
  /// rather than restored into the other. Version 3 also carries the pooled
  /// block bounds, so a restore does not have to re-derive them from the
  /// whole KV cache.
  static constexpr std::uint32_t kSnapshotPayloadVersion = 3;

private:
  friend class Model;
  Session();
  struct Impl;
  /// Rows of layer `il` that a session at `position` must keep to continue
  /// generation: every row for a full-attention layer, the sliding window
  /// for a SWA layer.
  [[nodiscard]] std::size_t LiveRows(std::uint32_t il,
                                     std::uint32_t position) const noexcept;
  /// Absolute row index of the first live row of layer `il`.
  [[nodiscard]] std::uint32_t LiveFirstRow(
      std::uint32_t il, std::uint32_t position) const noexcept;
  /// Re-derives the block-sparse pooled keys from the restored KV cache.
  void RebuildSparsePooled();
  std::unique_ptr<Impl> impl_;
};

/// The Laguna weights resident on the GPU, shared by sessions.
class Model final {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const EngineOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      std::uint32_t max_context, std::string* error_msg = nullptr);

  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  [[nodiscard]] bool HasDflash() const noexcept;
  /// Tokens the drafter proposes and one verification pass holds at most.
  [[nodiscard]] std::uint32_t DrafterBlockSize() const noexcept;

  /// KV cache bytes a session of `context` tokens would allocate.
  ///
  /// The server uses this to pick the largest context the device can hold
  /// before it creates any session. The value matches what Session::Init
  /// allocates, so the two never disagree about the requirement.
  [[nodiscard]] std::size_t SessionBytesForContext(
      std::uint32_t context) const noexcept;

  /// Device bytes currently free, as the driver reports them.
  [[nodiscard]] static std::size_t FreeDeviceBytes() noexcept;

private:
  Model() = default;
  std::shared_ptr<ModelState> state_;
};

}  // namespace gufo::models::laguna::gpu

#endif  // GUFO_MODELS_LAGUNA_GPU_ENGINE_HPP_
