// Pins the pooled-key snapshot layout.
//
// A block-sparse checkpoint carries its pooled key summaries in a contiguous
// [kv_head][block][256] layout, and the region it is given is sized for the
// session's whole block capacity. The copy that fills and drains that layout
// used to apply the head offset and advance its own pointer, which doubled the
// stride and put the last segments past the end of the payload. At a deep
// position that is several megabytes of the neighbouring pool block, read back
// as pooled keys and K/V rows.
//
// The failure only appears past the point where the doubled stride outgrows
// the region, so the geometry below is the one the log recorded: Laguna S 2.1
// at position 177136, eight kv heads, 4097 blocks of capacity.
#include "src/models/laguna/pooled_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using namespace gufo::models::laguna;

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "Assertion failed: %s\n", message);
    std::exit(1);
  }
}

// Laguna S 2.1: num_kv 8, capacity 262160 (262144 + 16 verify rows).
constexpr std::size_t kHeads = 8;
constexpr std::uint32_t kCapacity = 262160;
constexpr std::uint32_t kMaxBlocks =
    (kCapacity + kPooledBlockTokens - 1) / kPooledBlockTokens;

}  // namespace

int main() {
  // One 64-token block per 64 tokens, capped at the session's capacity.
  Expect(PooledBlocksAt(0, kMaxBlocks) == 0, "an empty frontier has no blocks");
  Expect(PooledBlocksAt(1, kMaxBlocks) == 1,
         "a one-token frontier has one block");
  Expect(PooledBlocksAt(64, kMaxBlocks) == 1, "64 tokens fill one block");
  Expect(PooledBlocksAt(65, kMaxBlocks) == 2, "65 tokens need a second block");
  Expect(PooledBlocksAt(kCapacity, kMaxBlocks) == kMaxBlocks,
         "a full context reaches the block capacity");
  Expect(PooledBlocksAt(UINT32_MAX, kMaxBlocks) == kMaxBlocks,
         "the block count is capped at the session capacity");
  Expect(PooledBlocksAt(177136, 0) == 0, "no sparse pool means no blocks");

  // Segments are contiguous: each one starts where the previous one ended.
  for (const std::uint32_t blocks : {1U, 2531U, 2768U, kMaxBlocks}) {
    const std::size_t head = PooledHeadBytes(blocks);
    Expect(PooledRegionBytes(kHeads, blocks) == kHeads * head,
           "the region is the heads' segments end to end");
    for (std::size_t h = 0; h < kHeads; ++h) {
      Expect(PooledHeadOffset(h, blocks) == h * head,
             "a segment starts at its head index");
      Expect(PooledHeadOffset(h, blocks) + head <=
                 PooledRegionBytes(kHeads, blocks),
             "a segment ends inside the region");
    }
    Expect(PooledHeadOffset(kHeads - 1, blocks) + head ==
               PooledRegionBytes(kHeads, blocks),
           "the last segment ends exactly at the region's end");
  }

  // The geometry that failed: 177136 tokens of a 262160-token session. The
  // region reserves the whole block capacity while the snapshot carries only
  // the blocks that exist, so the snapshot fits with room to spare - and a
  // doubled stride does not fit at all.
  const std::uint32_t blocks = PooledBlocksAt(177136, kMaxBlocks);
  Expect(blocks == 2768, "177136 tokens need 2768 blocks");
  const std::size_t head = PooledHeadBytes(blocks);
  const std::size_t region = PooledRegionBytes(kHeads, blocks);
  const std::size_t reserved = PooledRegionBytes(kHeads, kMaxBlocks);
  Expect(region <= reserved, "the snapshot fits the reserved region");
  Expect(15 * head > region,
         "a doubled stride overruns the region at this position");
  // The doubled stride keeps the device's block capacity, so at this depth it
  // outgrows even the region reserved for the whole session - the last layer's
  // segments land past the end of the payload, not merely past the blocks the
  // snapshot carries.
  Expect(15 * head > reserved, "a doubled stride overruns the reserved region");
  // The layout this file pins fits with room to spare.
  Expect(PooledHeadOffset(kHeads - 1, blocks) + head <= reserved,
         "the contiguous layout fits the reserved region");

  // Head offset plus its own length never crosses the reserved bytes, for any
  // position a session can reach.
  for (std::uint32_t position = 0; position <= kCapacity; position += 61) {
    const std::uint32_t at = PooledBlocksAt(position, kMaxBlocks);
    for (std::size_t h = 0; h < kHeads; ++h) {
      Expect(PooledHeadOffset(h, at) + PooledHeadBytes(at) <=
                 PooledRegionBytes(kHeads, kMaxBlocks),
             "every segment stays inside the reserved region");
    }
  }

  std::printf(
      "pooled layout: %u blocks at 177136, head %zu B, region %zu B, "
      "reserved %zu B\n",
      blocks, head, region, reserved);
  return 0;
}
