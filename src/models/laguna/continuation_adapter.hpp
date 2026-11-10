#ifndef GUFO_MODELS_LAGUNA_CONTINUATION_ADAPTER_HPP_
#define GUFO_MODELS_LAGUNA_CONTINUATION_ADAPTER_HPP_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <span>
#include <vector>

#include "src/cache/adapter.hpp"
#include "src/models/laguna/gpu/engine.hpp"

namespace gufo::models::laguna {

/// Compatibility identity for the in-process component cache. The process
/// identity is enough while the adapter does not advertise persistent
/// encoding: a snapshot never leaves this server.
[[nodiscard]] cache::Identity LagunaProcessIdentity(const void* model,
                                                    std::uint32_t max_context);

/// Bridges the Laguna session snapshot to the component cache.
///
/// A Laguna session is one flat snapshot: the attention KV caches, the last
/// logits and the drafter's injected context are saved and restored as a
/// single private payload. There are no append-row components, so prefix
/// reuse is exact-boundary only, which is what the previous continuation
/// cache did.
class LagunaComponentAdapter final : public cache::Adapter {
public:
  LagunaComponentAdapter(std::shared_ptr<gpu::Model> model,
                         std::uint32_t max_context,
                         cache::Identity identity);
  ~LagunaComponentAdapter() override;

  [[nodiscard]] gpu::Session& Session(cache::Slot&) const;

  [[nodiscard]] std::vector<cache::BackingClass> BackingPlan(
      std::size_t ram_bytes) const override;
  [[nodiscard]] cache::Capabilities GetCapabilities() const override;
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components()
      const override;
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override;
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(
      cache::MutationGuard&) override;
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      const cache::Slot&) const override;
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(
      const cache::Slot&, cache::Rows first, cache::Rows end) const override;
  void BeginRestore(cache::Slot&,
                    std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] cache::Completion CapturePrivate(const cache::Slot&,
                                                 cache::ComponentId,
                                                 std::span<std::byte>,
                                                 cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId,
                                              std::span<const std::byte>,
                                              cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsOut(
      const cache::Slot&, cache::ComponentId, cache::Rows first,
      cache::Rows end, std::span<std::byte>, cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId,
                                             cache::Rows first, cache::Rows end,
                                             std::span<const std::byte>,
                                             cache::Stream&) override;
  [[nodiscard]] bool Validate(
      cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] bool Invalidate(cache::Slot&) noexcept override;

  /// The runner consumes at most this many prompt tokens per pass.
  ///
  /// This bounds one Append() call. It was 4096 while the engine's own prefill
  /// tile was 4096; the routed expert GEMM gives each expert
  /// `rows * experts_used / experts` rows and tiles them 128 at a time, so
  /// the pass length decides how much of the last tile is padding. At 4096
  /// rows and 10-of-256 routing that is 160 rows, i.e. two tiles of 128 and a
  /// 37.5% empty second tile. Overridable with LAGUNA_PREFILL_TILE so one
  /// build measures both arms; it must not exceed the engine's tile.
  static constexpr std::size_t kPrefillPass = 16384;

  [[nodiscard]] static std::size_t PrefillPassRows() {
    static const std::size_t value = [] {
      if (const char* const v = std::getenv("LAGUNA_PREFILL_TILE")) {
        const int parsed = std::atoi(v);
        if (parsed > 0) {
          return static_cast<std::size_t>(parsed);
        }
      }
      return kPrefillPass;
    }();
    return value;
  }
  /// The live snapshot length is stored after the payload so a restore can
  /// hand the session an exact-size span while the cache buffer is fixed.
  static constexpr std::size_t kLengthTrailerBytes = sizeof(std::uint64_t);

private:
  class SlotState;
  [[nodiscard]] SlotState& As(cache::Slot&) const;
  [[nodiscard]] const SlotState& As(const cache::Slot&) const;
  [[nodiscard]] const cache::ComponentDescriptor& Component(
      cache::ComponentId) const;

  std::shared_ptr<gpu::Model> model_;
  std::uint32_t max_context_;
  std::size_t state_bytes_{0};
  cache::Identity identity_;
  std::vector<cache::ComponentDescriptor> descriptors_;
};

}  // namespace gufo::models::laguna

#endif  // GUFO_MODELS_LAGUNA_CONTINUATION_ADAPTER_HPP_
