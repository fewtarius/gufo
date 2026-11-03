// Quantized attention K/V row codec.
//
// Model-agnostic, portable host code. It has no HIP dependency and knows
// nothing about attention, recurrence, positions or sessions: a model
// supplies a producer (where K/V values come from) and a consumer (how that
// model's attention reads them).
//
// The full contract is in docs/kv-codec.md. The rules that matter most:
//
//   - Both formats quantize per 32-element block, independently. No block's
//     scale depends on another block.
//
//   - q8_0: d = max(|x|)/127 in binary32, q = clamp(roundf(x/d), -127, 127).
//     `roundf` rounds halfway cases AWAY FROM ZERO, not to nearest-even, and
//     the division uses the UNROUNDED binary32 d - never the stored binary16
//     scale. Only the stored scale is rounded to binary16.
//
//   - q4_0: d = signed_max / -8 in binary32, q = clamp(trunc(x/d + 8.5), 0,
//     15) with the product and the addition rounded separately so they do
//     not contract into an FMA. A negative scale is valid.
//
//   - Decode rounds the product to binary16 BEFORE promoting to binary32.
//     The stored scale is binary16, so the product is generally not
//     representable in binary32 and reconstructing in binary32 would produce
//     values the format cannot represent.
//
//   - Rotation is NOT part of the codec. A rotated row is stored with the
//     same codec as an unrotated one.
//
//   - Validation completes before any output is written, so a rejected call
//     leaves the destination entirely unchanged.
//
//   - q8_0 scales are non-negative; q4_0 scales may be negative, including
//     negative zero.
//
// Every API throws std::invalid_argument for invalid geometry, non-finite
// input or payload, a forbidden scale sign, a q8_0 code of -128, or a
// reconstruction that overflows to binary16 infinity.
//
// The numeric contract was derived in the Gufetto community fork of Gufo
// for its Flash-Next q8 KV work and is documented there. This implementation
// is written independently against that contract so one codec can serve
// every model in this tree. See docs/kv-codec.md for provenance.
#ifndef GUFO_CORE_KV_CODEC_CODEC_HPP_
#define GUFO_CORE_KV_CODEC_CODEC_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace gufo::core::kv_codec {

/// Storage format for one attention K or V row.
enum class RowCodec : std::uint8_t {
  /// Half precision, two bytes per element. Not a packed format.
  kF16 = 0,
  /// ggml q8_0: one binary16 scale plus 32 signed 8-bit codes per block.
  kQ8_0 = 1,
  /// ggml q4_0: one binary16 scale plus 16 packed nibble pairs per block.
  kQ4_0 = 2,
};

/// Elements per quantized block, in every packed format.
inline constexpr std::uint32_t kBlockElements = 32;

/// Standard q8_0 block: a binary16 scale followed by 32 signed codes.
/// There is no inter-block padding. Serialized scales are little-endian;
/// the byte-buffer API does not require aligned storage.
struct BlockQ8_0 {
  std::uint16_t scale;
  std::int8_t values[kBlockElements];
};
static_assert(sizeof(BlockQ8_0) == 34);
static_assert(offsetof(BlockQ8_0, values) == 2);
static_assert(std::is_standard_layout_v<BlockQ8_0>);
static_assert(std::is_trivially_copyable_v<BlockQ8_0>);

/// Standard q4_0 block: a binary16 scale followed by 16 bytes, each holding
/// two 4-bit codes. The low nibble holds coordinates 0..15, the high nibble
/// holds 16..31. Each code is a signed value offset by eight.
struct BlockQ4_0 {
  std::uint16_t scale;
  std::uint8_t values[16];
};
static_assert(sizeof(BlockQ4_0) == 18);
static_assert(offsetof(BlockQ4_0, values) == 2);
static_assert(std::is_trivially_copyable_v<BlockQ4_0>);

[[nodiscard]] constexpr bool IsKnownCodec(RowCodec codec) noexcept {
  switch (codec) {
    case RowCodec::kF16:
    case RowCodec::kQ8_0:
    case RowCodec::kQ4_0:
      return true;
  }
  return false;
}

/// True for the formats that store a row in packed blocks.
[[nodiscard]] constexpr bool IsPacked(RowCodec codec) noexcept {
  return codec != RowCodec::kF16;
}

/// True when a row of `elements` values is representable in `codec`.
[[nodiscard]] constexpr bool IsValidRowGeometry(std::size_t elements,
                                                RowCodec codec) noexcept {
  switch (codec) {
    case RowCodec::kF16:
      return elements <= std::numeric_limits<std::size_t>::max() / 2;
    case RowCodec::kQ8_0:
    case RowCodec::kQ4_0:
      return elements % kBlockElements == 0 &&
             elements / kBlockElements <=
                 std::numeric_limits<std::size_t>::max() / sizeof(BlockQ8_0);
  }
  return false;
}

/// Bytes a row of `elements` values occupies in `codec`.
///
/// Throws std::invalid_argument for an unknown codec or invalid geometry.
[[nodiscard]] std::size_t RowBytes(std::size_t elements, RowCodec codec);

/// Encodes one row of binary32 values into `codec`.
///
/// `output` must be exactly RowBytes(input.size(), codec) bytes. Throws
/// std::invalid_argument for a mismatched size, overlapping buffers, a
/// non-finite input, or a reconstruction that overflows to binary16
/// infinity. On a throw the destination is unchanged.
void EncodeRow(std::span<const float> input, RowCodec codec,
               std::span<std::byte> output);

/// Decodes one row in `codec` into binary32 values.
///
/// `output` must hold exactly the row's element count. Throws
/// std::invalid_argument for a mismatched size, overlapping buffers, an
/// invalid payload, or a non-finite input. On a throw the destination is
/// unchanged.
void DecodeRow(std::span<const std::byte> input, RowCodec codec,
               std::span<float> output);

/// Checks one row in `codec` without decoding it.
///
/// `input` must be exactly RowBytes(elements, codec) bytes. Throws
/// std::invalid_argument for a mismatched size or an invalid payload.
void ValidateRow(std::span<const std::byte> input, std::size_t elements,
                 RowCodec codec);

/// Canonical ggml q8_0 rounding primitive: roundf ties away from zero,
/// clamped to [-127, 127]. Throws std::invalid_argument for a non-finite
/// scaled value.
[[nodiscard]] std::int8_t RoundQ8Code(float scaled);

}  // namespace gufo::core::kv_codec

#endif  // GUFO_CORE_KV_CODEC_CODEC_HPP_
