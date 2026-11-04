// Portable IEEE binary32 <-> binary16 conversion.
//
// The KV row codec stores its block scales as binary16 and reconstructs
// elements through binary16, so the conversion has to be exact and must not
// depend on the host's floating-point environment: the codec is portable
// host code and its output is compared against a device path that uses the
// hardware conversion.
//
// Properties this implementation guarantees:
//   - round to nearest, ties to even;
//   - independent of the host rounding mode;
//   - signed zero and infinity are preserved;
//   - a NaN input becomes a signed quiet NaN;
//   - a finite input that overflows rounds to infinity;
//   - binary16 subnormals are produced and consumed exactly.
//
// See docs/kv-codec.md for the contract this exists to satisfy.
#ifndef GUFO_CORE_KV_CODEC_HALF_HPP_
#define GUFO_CORE_KV_CODEC_HALF_HPP_

#include <bit>
#include <cstdint>

namespace gufo::core::kv_codec {

/// Binary32 value to binary16 bits.
[[nodiscard]] inline std::uint16_t FloatToHalfBits(float value) noexcept {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000);
  const auto exponent = (bits >> 23) & 0xffU;
  const auto fraction = bits & 0x7fffffU;
  if (exponent == 0xffU) {
    // Infinity keeps its encoding; a NaN becomes a quiet NaN.
    return sign | (fraction == 0 ? 0x7c00 : 0x7e00);
  }
  if (exponent < 102U) {
    // Below 2^-126, which is far under half the smallest binary16 subnormal.
    return sign;
  }
  if (exponent > 142U) {
    // Above the largest finite binary16. A finite overflow rounds to
    // infinity; there is no finite value to round to here.
    return sign | 0x7c00;
  }
  if (exponent < 113U) {
    // Binary16 subnormal: include the hidden bit and round once into the
    // subnormal field. exponent == 113 is the tie at half the smallest
    // subnormal.
    const auto significand = fraction | 0x800000U;
    const auto shift = 126U - exponent;
    const auto truncated = significand >> shift;
    const auto remainder = significand & ((std::uint32_t{1} << shift) - 1U);
    const auto halfway = std::uint32_t{1} << (shift - 1U);
    return sign | static_cast<std::uint16_t>(
                      truncated + (remainder > halfway ||
                                   (remainder == halfway && (truncated & 1U))));
  }
  auto magnitude = ((exponent - 112U) << 10) | (fraction >> 13);
  const auto remainder = fraction & 0x1fffU;
  magnitude +=
      remainder > 0x1000U || (remainder == 0x1000U && (magnitude & 1U));
  return sign | static_cast<std::uint16_t>(magnitude);
}

/// Binary16 bits to binary32 value.
[[nodiscard]] inline float HalfBitsToFloat(std::uint16_t bits) noexcept {
  const auto sign = std::uint32_t{bits & 0x8000U} << 16;
  const auto exponent = (bits >> 10) & 0x1fU;
  auto fraction = std::uint32_t{bits & 0x3ffU};
  std::uint32_t magnitude = 0;
  if (exponent == 0) {
    if (fraction != 0) {
      std::uint32_t e = 113;
      while ((fraction & 0x400U) == 0) {
        fraction <<= 1;
        --e;
      }
      magnitude = (e << 23) | ((fraction & 0x3ffU) << 13);
    }
  } else if (exponent == 31) {
    magnitude = 0x7f800000U | (fraction == 0 ? 0U : 0x400000U);
  } else {
    magnitude = ((exponent + 112U) << 23) | (fraction << 13);
  }
  return std::bit_cast<float>(sign | magnitude);
}

/// True when the binary16 bits are neither infinity nor NaN.
[[nodiscard]] inline bool HalfBitsAreFinite(std::uint16_t bits) noexcept {
  return (bits & 0x7c00U) != 0x7c00U;
}

}  // namespace gufo::core::kv_codec

#endif  // GUFO_CORE_KV_CODEC_HALF_HPP_
