#ifndef GUFO_MODELS_LAGUNA_POOLED_LAYOUT_HPP_
#define GUFO_MODELS_LAGUNA_POOLED_LAYOUT_HPP_

#include <cstddef>
#include <cstdint>

namespace gufo::models::laguna {

// Block-sparse attention keeps one pooled key summary per 64-token block per
// kv head. A summary is 256 values wide, and the session holds them on the
// device as [kv_head][block][256] with the block stride fixed at the session's
// block capacity.
inline constexpr std::size_t kPooledBlockValues = 256;
inline constexpr std::size_t kPooledValueBytes = 2;
inline constexpr std::uint32_t kPooledBlockTokens = 64;

/// Blocks that exist at `position`, capped at the session's block capacity.
[[nodiscard]] inline constexpr std::uint32_t PooledBlocksAt(
    std::uint32_t position, std::uint32_t max_blocks) noexcept {
  if (max_blocks == 0)
    return 0;
  const std::uint64_t blocks =
      (static_cast<std::uint64_t>(position) + kPooledBlockTokens - 1) /
      kPooledBlockTokens;
  return blocks < max_blocks ? static_cast<std::uint32_t>(blocks) : max_blocks;
}

/// Bytes one kv head occupies in a snapshot taken at `blocks` blocks.
[[nodiscard]] inline constexpr std::size_t PooledHeadBytes(
    std::uint32_t blocks) noexcept {
  return static_cast<std::size_t>(blocks) * kPooledBlockValues *
         kPooledValueBytes;
}

/// Byte offset of one kv head's segment inside a snapshot region.
///
/// The segments are contiguous and this is the layout's only stride. A
/// reader that advances its own pointer as well as applying this offset
/// doubles the stride, and the last segment then ends past the bytes the
/// payload reserved.
[[nodiscard]] inline constexpr std::size_t PooledHeadOffset(
    std::size_t head, std::uint32_t blocks) noexcept {
  return head * PooledHeadBytes(blocks);
}

/// Bytes a snapshot region reserves for `heads` kv heads at `blocks` blocks.
[[nodiscard]] inline constexpr std::size_t PooledRegionBytes(
    std::size_t heads, std::uint32_t blocks) noexcept {
  return heads * PooledHeadBytes(blocks);
}

}  // namespace gufo::models::laguna

#endif  // GUFO_MODELS_LAGUNA_POOLED_LAYOUT_HPP_
