#include "src/core/kv_codec/codec.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

#include "src/core/kv_codec/half.hpp"

namespace gufo::core::kv_codec {
namespace {

/// Rounds to binary32 at an explicit boundary.
///
/// The q4_0 encoder multiplies and then adds 8.5. On a host with FMA those
/// two operations would contract into one, and the rounding of the product
/// would be lost. The volatile store forces the intermediate value to be
/// materialized in binary32.
inline float RoundToF32(float value) noexcept {
  volatile float rounded = value;
  return rounded;
}

[[nodiscard]] bool IsFiniteHalf(std::uint16_t bits) noexcept {
  return HalfBitsAreFinite(bits);
}

/// Rejects a source and destination that share storage.
void RejectOverlap(std::span<const std::byte> input,
                   std::span<std::byte> output) {
  if (input.empty() || output.empty()) {
    return;
  }
  const auto a = reinterpret_cast<std::uintptr_t>(input.data());
  const auto b = reinterpret_cast<std::uintptr_t>(output.data());
  if (a <= b ? b - a < input.size() : a - b < output.size()) {
    throw std::invalid_argument("overlapping KV row buffers");
  }
}

/// One q8_0 block, encoded from 32 binary32 values.
[[nodiscard]] BlockQ8_0 QuantizeQ8Block(const float* input) {
  float amax = 0.0F;
  for (std::size_t i = 0; i < kBlockElements; ++i) {
    if (!std::isfinite(input[i])) {
      throw std::invalid_argument("nonfinite KV row input");
    }
    amax = std::max(amax, std::abs(input[i]));
  }
  // The division uses this unrounded binary32 scale. Dividing by the stored
  // binary16 scale, or by a reciprocal approximation, changes tie bytes.
  const float d = amax / 127.0F;
  BlockQ8_0 block{};
  if (d == 0.0F) {
    return block;
  }
  block.scale = FloatToHalfBits(d);
  if (!IsFiniteHalf(block.scale)) {
    throw std::invalid_argument("KV row scale overflows F16");
  }
  const float stored_scale = HalfBitsToFloat(block.scale);
  for (std::size_t i = 0; i < kBlockElements; ++i) {
    const auto q = RoundQ8Code(input[i] / d);
    block.values[i] = q;
    if (!IsFiniteHalf(FloatToHalfBits(static_cast<float>(q) * stored_scale))) {
      throw std::invalid_argument("KV row reconstruction overflows F16");
    }
  }
  return block;
}

/// One q4_0 block, encoded from 32 binary32 values.
[[nodiscard]] BlockQ4_0 QuantizeQ4Block(const float* input) {
  float amax = 0.0F;
  float signed_max = 0.0F;
  for (std::size_t i = 0; i < kBlockElements; ++i) {
    if (!std::isfinite(input[i])) {
      throw std::invalid_argument("nonfinite KV row input");
    }
    if (std::abs(input[i]) > amax) {
      amax = std::abs(input[i]);
      signed_max = input[i];
    }
  }
  BlockQ4_0 block{};
  // The all-zero code is 8.
  std::fill(std::begin(block.values), std::end(block.values), 0x88);
  // A signed scale keeps the sign of the dominant value, which is what makes
  // the four-bit code symmetric around zero.
  const float d = RoundToF32(signed_max / -8.0F);
  if (d == 0.0F) {
    return block;
  }
  const float inverse = RoundToF32(1.0F / d);
  block.scale = FloatToHalfBits(d);
  if (!std::isfinite(inverse) || !IsFiniteHalf(block.scale)) {
    throw std::invalid_argument("invalid KV row scale");
  }
  const float scale = HalfBitsToFloat(block.scale);
  for (std::size_t j = 0; j < 16; ++j) {
    unsigned packed = 0;
    for (unsigned half = 0; half < 2; ++half) {
      const float shifted =
          RoundToF32(RoundToF32(input[j + half * 16] * inverse) + 8.5F);
      const auto code = static_cast<unsigned>(std::clamp(shifted, 0.0F, 15.0F));
      if (!IsFiniteHalf(FloatToHalfBits(static_cast<float>(
                          static_cast<int>(code) - 8) * scale))) {
        throw std::invalid_argument("KV row reconstruction overflows F16");
      }
      packed |= code << (half * 4);
    }
    block.values[j] = static_cast<std::uint8_t>(packed);
  }
  return block;
}

void WriteBlock(std::span<std::byte> output, std::size_t block_index,
                const BlockQ8_0& block) {
  auto* dst = output.data() + block_index * sizeof(BlockQ8_0);
  std::memcpy(dst, &block.scale, sizeof(block.scale));
  for (std::size_t j = 0; j < kBlockElements; ++j) {
    dst[2 + j] = std::byte(static_cast<std::uint8_t>(block.values[j]));
  }
}

void WriteBlock(std::span<std::byte> output, std::size_t block_index,
                const BlockQ4_0& block) {
  auto* dst = output.data() + block_index * sizeof(BlockQ4_0);
  std::memcpy(dst, &block.scale, sizeof(block.scale));
  for (std::size_t j = 0; j < 16; ++j) {
    dst[2 + j] = std::byte(block.values[j]);
  }
}

/// Largest code magnitude in a q8_0 block. One overflow check per block
/// covers every element, because a non-negative scale makes the maximum
/// magnitude the worst case.
void ValidateQ8Blocks(std::span<const std::byte> input) {
  for (std::size_t i = 0; i < input.size(); i += sizeof(BlockQ8_0)) {
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, input.data() + i, sizeof(scale_bits));
    // A q8_0 scale is a magnitude: the sign bit is always clear.
    if ((scale_bits & 0x8000U) != 0 || !IsFiniteHalf(scale_bits)) {
      throw std::invalid_argument("invalid KV row scale");
    }
    const float scale = HalfBitsToFloat(scale_bits);
    int largest = 0;
    for (std::size_t j = 0; j < kBlockElements; ++j) {
      const auto q = std::bit_cast<std::int8_t>(
          std::to_integer<std::uint8_t>(input[i + 2 + j]));
      if (q == -128) {
        throw std::invalid_argument("invalid KV row code");
      }
      largest = std::max(largest, std::abs(static_cast<int>(q)));
    }
    if (!IsFiniteHalf(FloatToHalfBits(static_cast<float>(largest) * scale))) {
      throw std::invalid_argument("invalid KV row reconstruction");
    }
  }
}

void ValidateQ4Blocks(std::span<const std::byte> input) {
  for (std::size_t i = 0; i < input.size(); i += sizeof(BlockQ4_0)) {
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, input.data() + i, sizeof(scale_bits));
    // A q4_0 scale may be negative, including negative zero.
    if (!IsFiniteHalf(scale_bits)) {
      throw std::invalid_argument("invalid KV row scale");
    }
    const float scale = HalfBitsToFloat(scale_bits);
    int largest = 0;
    for (std::size_t j = 0; j < 16; ++j) {
      const auto byte = std::to_integer<unsigned>(input[i + 2 + j]);
      const int low = static_cast<int>(byte & 15U) - 8;
      const int high = static_cast<int>(byte >> 4) - 8;
      largest = std::max(largest, std::max(std::abs(low), std::abs(high)));
    }
    // Half overflow is monotone in magnitude and symmetric in sign, so one
    // check covers every nibble in the block.
    if (!IsFiniteHalf(FloatToHalfBits(static_cast<float>(largest) * scale))) {
      throw std::invalid_argument("invalid KV row reconstruction");
    }
  }
}

}  // namespace

std::size_t RowBytes(std::size_t elements, RowCodec codec) {
  if (!IsKnownCodec(codec)) {
    throw std::invalid_argument("unknown KV row codec");
  }
  if (!IsValidRowGeometry(elements, codec)) {
    throw std::invalid_argument("invalid KV row geometry");
  }
  if (codec == RowCodec::kF16) {
    return elements * 2;
  }
  const auto blocks = elements / kBlockElements;
  return blocks * (codec == RowCodec::kQ8_0 ? sizeof(BlockQ8_0)
                                            : sizeof(BlockQ4_0));
}

std::int8_t RoundQ8Code(float scaled) {
  if (!std::isfinite(scaled)) {
    throw std::invalid_argument("nonfinite KV row code");
  }
  // std::round breaks ties away from zero. That is the canonical ggml q8_0
  // primitive and it is NOT the IEEE default rounding.
  return static_cast<std::int8_t>(std::clamp(std::round(scaled), -127.0F,
                                              127.0F));
}

void EncodeRow(std::span<const float> input, RowCodec codec,
               std::span<std::byte> output) {
  const auto bytes = RowBytes(input.size(), codec);
  if (output.size() != bytes) {
    throw std::invalid_argument("incorrect KV row output size");
  }
  RejectOverlap(std::as_bytes(input), output);
  // Two passes: the first validates the whole row, the second writes it.
  // That gives a fail-closed encode without a row-sized temporary.
  if (codec == RowCodec::kF16) {
    for (const auto x : input) {
      if (!std::isfinite(x) || !IsFiniteHalf(FloatToHalfBits(x))) {
        throw std::invalid_argument("KV row input is not finite F16");
      }
    }
    for (std::size_t i = 0; i < input.size(); ++i) {
      const auto bits = FloatToHalfBits(input[i]);
      std::memcpy(output.data() + i * 2, &bits, sizeof(bits));
    }
    return;
  }
  if (codec == RowCodec::kQ4_0) {
    for (std::size_t i = 0; i < input.size(); i += kBlockElements) {
      (void)QuantizeQ4Block(input.data() + i);
    }
    for (std::size_t i = 0; i < input.size(); i += kBlockElements) {
      WriteBlock(output, i / kBlockElements,
                 QuantizeQ4Block(input.data() + i));
    }
    return;
  }
  for (std::size_t i = 0; i < input.size(); i += kBlockElements) {
    (void)QuantizeQ8Block(input.data() + i);
  }
  for (std::size_t i = 0; i < input.size(); i += kBlockElements) {
    WriteBlock(output, i / kBlockElements, QuantizeQ8Block(input.data() + i));
  }
}

void ValidateRow(std::span<const std::byte> input, std::size_t elements,
                 RowCodec codec) {
  if (input.size() != RowBytes(elements, codec)) {
    throw std::invalid_argument("incorrect KV row input size");
  }
  if (codec == RowCodec::kF16) {
    for (std::size_t i = 0; i < input.size(); i += 2) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, input.data() + i, sizeof(bits));
      if (!IsFiniteHalf(bits)) {
        throw std::invalid_argument("nonfinite KV row payload");
      }
    }
    return;
  }
  if (codec == RowCodec::kQ4_0) {
    ValidateQ4Blocks(input);
    return;
  }
  ValidateQ8Blocks(input);
}

void DecodeRow(std::span<const std::byte> input, RowCodec codec,
               std::span<float> output) {
  ValidateRow(input, output.size(), codec);
  RejectOverlap(input, std::span<std::byte>(
                           reinterpret_cast<std::byte*>(output.data()),
                           output.size() * sizeof(float)));
  if (codec == RowCodec::kF16) {
    for (std::size_t i = 0; i < output.size(); ++i) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, input.data() + i * 2, sizeof(bits));
      output[i] = HalfBitsToFloat(bits);
    }
    return;
  }
  if (codec == RowCodec::kQ4_0) {
    for (std::size_t i = 0; i < output.size(); i += kBlockElements) {
      const auto* src = input.data() + (i / kBlockElements) * sizeof(BlockQ4_0);
      std::uint16_t scale_bits = 0;
      std::memcpy(&scale_bits, src, sizeof(scale_bits));
      const float scale = HalfBitsToFloat(scale_bits);
      for (std::size_t j = 0; j < kBlockElements; ++j) {
        const auto byte = std::to_integer<unsigned>(src[2 + j % 16]);
        const int code = static_cast<int>((byte >> (j / 16 * 4)) & 15U) - 8;
        // Round through binary16 before promoting: the stored scale is
        // binary16, so the product is generally not representable in
        // binary32.
        output[i + j] = HalfBitsToFloat(FloatToHalfBits(
            static_cast<float>(code) * scale));
      }
    }
    return;
  }
  for (std::size_t i = 0; i < output.size(); i += kBlockElements) {
    const auto* src = input.data() + (i / kBlockElements) * sizeof(BlockQ8_0);
    std::uint16_t scale_bits = 0;
    std::memcpy(&scale_bits, src, sizeof(scale_bits));
    const float scale = HalfBitsToFloat(scale_bits);
    for (std::size_t j = 0; j < kBlockElements; ++j) {
      const auto q = std::bit_cast<std::int8_t>(
          std::to_integer<std::uint8_t>(src[2 + j]));
      output[i + j] = HalfBitsToFloat(
          FloatToHalfBits(static_cast<float>(q) * scale));
    }
  }
}

}  // namespace gufo::core::kv_codec
