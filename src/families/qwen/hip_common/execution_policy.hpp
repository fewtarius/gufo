#ifndef GUFO_MODELS_QWEN_HIP_EXECUTION_POLICY_HPP_
#define GUFO_MODELS_QWEN_HIP_EXECUTION_POLICY_HPP_

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "src/core/kv_codec/codec.hpp"

namespace gufo::hip {

enum class QwenKvCacheStorage : std::uint8_t {
  kFp32,
  kFp16,
  /// ggml q8_0: one binary16 scale and 32 signed codes per 32 elements. The
  /// layout and the numeric contract are in docs/kv-codec.md.
  kQ8_0,
};

/// The storage of the canonical K/V cache the attention kernels read.
///
/// The FP32 plane is the independent validation reference and travels
/// beside the canonical plane; it has no entry here.
enum class QwenKvFormat : std::uint8_t {
  kFp16,
  kQ8_0,
};

/// The canonical K/V cache planes an attention launch reads and writes.
struct QwenCompactKv {
  void* k{nullptr};
  void* v{nullptr};
  QwenKvFormat format{QwenKvFormat::kFp16};
};

/// The format of the canonical plane for a storage policy.
[[nodiscard]] constexpr QwenKvFormat QwenKvFormatFor(
    QwenKvCacheStorage storage) noexcept {
  return storage == QwenKvCacheStorage::kQ8_0 ? QwenKvFormat::kQ8_0
                                              : QwenKvFormat::kFp16;
}

/// Bytes one K/V row of `elements` values occupies in `storage`.
///
/// The FP32 plane is the independent validation reference and the FP16 and
/// q8_0 planes are the canonical token-major caches.
[[nodiscard]] inline std::size_t QwenKvRowBytes(QwenKvCacheStorage storage,
                                                 std::uint32_t elements) {
  switch (storage) {
    case QwenKvCacheStorage::kFp32:
      return static_cast<std::size_t>(elements) * sizeof(float);
    case QwenKvCacheStorage::kFp16:
      return static_cast<std::size_t>(elements) * sizeof(std::uint16_t);
    case QwenKvCacheStorage::kQ8_0:
      return core::kv_codec::RowBytes(
          elements, core::kv_codec::RowCodec::kQ8_0);
  }
  throw std::invalid_argument("unknown Qwen KV cache storage");
}

/// True for the canonical token-major caches, as opposed to the head-major
/// FP32 reference plane.
[[nodiscard]] constexpr bool QwenKvCacheIsTokenMajor(
    QwenKvCacheStorage storage) noexcept {
  return storage != QwenKvCacheStorage::kFp32;
}

enum class QwenRecurrentStateStorage : std::uint8_t {
  kFp32,
  kBf16,
};

[[nodiscard]] constexpr std::size_t QwenRecurrentStateElementBytes(
    QwenRecurrentStateStorage storage) noexcept {
  return storage == QwenRecurrentStateStorage::kBf16 ? sizeof(std::uint16_t)
                                                     : sizeof(float);
}

/// Immutable route policy for one Qwen GPU executor. Resolve this before HIP
/// graph capture. Alternate storage is used by independent operator controls.
enum class QwenExecutionMode : std::uint8_t {
  kDecode,
  kPrefill,
};

struct QwenExecutionPolicy {
  bool fuse_qk_norm_rope_kv{true};
  QwenKvCacheStorage kv_cache_storage{QwenKvCacheStorage::kFp16};
  QwenRecurrentStateStorage recurrent_state_storage{
      QwenRecurrentStateStorage::kFp32};

  [[nodiscard]] static constexpr QwenExecutionPolicy Production() noexcept {
    return {};
  }

  [[nodiscard]] constexpr bool UsesFp16AttentionKv() const noexcept {
    return kv_cache_storage == QwenKvCacheStorage::kFp16;
  }

  [[nodiscard]] constexpr bool UsesQ8AttentionKv() const noexcept {
    return kv_cache_storage == QwenKvCacheStorage::kQ8_0;
  }

  [[nodiscard]] constexpr bool UsesBf16RecurrentState() const noexcept {
    return recurrent_state_storage == QwenRecurrentStateStorage::kBf16;
  }

  /// Stable bit fingerprint suitable for telemetry and graph-cache identity.
  [[nodiscard]] constexpr std::uint64_t Fingerprint() const noexcept {
    return (static_cast<std::uint64_t>(fuse_qk_norm_rope_kv) << 0U) |
           (static_cast<std::uint64_t>(UsesFp16AttentionKv()) << 8U) |
           (static_cast<std::uint64_t>(UsesBf16RecurrentState()) << 9U) |
           (static_cast<std::uint64_t>(UsesQ8AttentionKv()) << 10U);
  }
};

/// Resolved route decisions for one layer invocation. This record is pure data
/// so policy selection can be tested independently from HIP launches.
struct QwenLayerRoutePlan {
  QwenExecutionMode mode{QwenExecutionMode::kDecode};
  bool full_attention{false};
  bool fuse_qk_norm_rope_kv{false};

  [[nodiscard]] constexpr std::uint64_t Fingerprint() const noexcept {
    return (static_cast<std::uint64_t>(mode == QwenExecutionMode::kPrefill)
            << 0U) |
           (static_cast<std::uint64_t>(full_attention) << 1U) |
           (static_cast<std::uint64_t>(fuse_qk_norm_rope_kv) << 2U);
  }
};

enum class QwenRouteRejection : std::uint32_t {
  kNone = 0,
  kQkNormRopeKvRequiresAttention = 1U << 0U,
};

[[nodiscard]] constexpr QwenRouteRejection operator|(
    QwenRouteRejection lhs, QwenRouteRejection rhs) noexcept {
  return static_cast<QwenRouteRejection>(static_cast<std::uint32_t>(lhs) |
                                         static_cast<std::uint32_t>(rhs));
}

constexpr QwenRouteRejection& operator|=(QwenRouteRejection& lhs,
                                         QwenRouteRejection rhs) noexcept {
  lhs = lhs | rhs;
  return lhs;
}

[[nodiscard]] constexpr bool HasQwenRouteRejection(
    QwenRouteRejection reasons, QwenRouteRejection reason) noexcept {
  return (static_cast<std::uint32_t>(reasons) &
          static_cast<std::uint32_t>(reason)) != 0U;
}

struct QwenLayerRouteResolution {
  QwenLayerRoutePlan plan;
  QwenRouteRejection rejected{QwenRouteRejection::kNone};
};

[[nodiscard]] constexpr QwenLayerRouteResolution
ResolveQwenLayerRouteWithReasons(const QwenExecutionPolicy& policy,
                                 QwenExecutionMode mode,
                                 bool full_attention) noexcept {
  const auto rejected = policy.fuse_qk_norm_rope_kv && !full_attention
                            ? QwenRouteRejection::kQkNormRopeKvRequiresAttention
                            : QwenRouteRejection::kNone;
  return {
      .plan =
          {
              .mode = mode,
              .full_attention = full_attention,
              .fuse_qk_norm_rope_kv =
                  full_attention && policy.fuse_qk_norm_rope_kv,
          },
      .rejected = rejected,
  };
}

[[nodiscard]] constexpr QwenLayerRoutePlan ResolveQwenLayerRoute(
    const QwenExecutionPolicy& policy, QwenExecutionMode mode,
    bool full_attention) noexcept {
  return ResolveQwenLayerRouteWithReasons(policy, mode, full_attention).plan;
}

/// FNV-1a-style composition helpers for a deterministic executor-local graph
/// workload identity. Callers add configuration fields and resolved route
/// fingerprints; raw addresses must never participate.
[[nodiscard]] constexpr std::uint64_t
BeginQwenGraphWorkloadIdentity() noexcept {
  return 14695981039346656037ULL;
}

[[nodiscard]] constexpr std::uint64_t ExtendQwenGraphWorkloadIdentity(
    std::uint64_t identity, std::uint64_t value) noexcept {
  return (identity ^ value) * 1099511628211ULL;
}

enum class QwenGraphRejection : std::uint32_t {
  kNone = 0,
  kLogitsNotRequested = 1U << 0U,
  kSplitKAttentionRequired = 1U << 1U,
  kGraphDisabled = 1U << 3U,
};

[[nodiscard]] constexpr QwenGraphRejection operator|(
    QwenGraphRejection lhs, QwenGraphRejection rhs) noexcept {
  return static_cast<QwenGraphRejection>(static_cast<std::uint32_t>(lhs) |
                                         static_cast<std::uint32_t>(rhs));
}

constexpr QwenGraphRejection& operator|=(QwenGraphRejection& lhs,
                                         QwenGraphRejection rhs) noexcept {
  lhs = lhs | rhs;
  return lhs;
}

[[nodiscard]] constexpr bool HasQwenGraphRejection(
    QwenGraphRejection reasons, QwenGraphRejection reason) noexcept {
  return (static_cast<std::uint32_t>(reasons) &
          static_cast<std::uint32_t>(reason)) != 0U;
}

[[nodiscard]] constexpr QwenGraphRejection ResolveQwenGraphRejections(
    bool compute_logits, bool split_k_attention, bool graph_enabled) noexcept {
  QwenGraphRejection rejected = QwenGraphRejection::kNone;
  if (!compute_logits) {
    rejected |= QwenGraphRejection::kLogitsNotRequested;
  }
  if (split_k_attention) {
    rejected |= QwenGraphRejection::kSplitKAttentionRequired;
  }
  if (!graph_enabled) {
    rejected |= QwenGraphRejection::kGraphDisabled;
  }
  return rejected;
}

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_EXECUTION_POLICY_HPP_
