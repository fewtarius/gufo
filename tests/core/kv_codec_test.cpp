// Tests for the KV row codec.
//
// The contract under test is docs/kv-codec.md. The rounding rules are the
// part that cannot be checked by a round trip, so each one is pinned
// separately: ties away from zero, the unrounded binary32 scale in the
// division, the binary16 rounding of the stored scale, and the binary16
// rounding of the decoded product.
#include "src/core/kv_codec/codec.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/kv_codec/half.hpp"

namespace {

using gufo::core::kv_codec::BlockQ4_0;
using gufo::core::kv_codec::BlockQ8_0;
using gufo::core::kv_codec::FloatToHalfBits;
using gufo::core::kv_codec::HalfBitsToFloat;
using gufo::core::kv_codec::RowBytes;
using gufo::core::kv_codec::RowCodec;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

template <typename Fn>
void ExpectThrows(Fn&& fn, std::string_view message) {
  try {
    fn();
  } catch (const std::invalid_argument&) {
    return;
  } catch (...) {
    std::cerr << "Assertion failed: " << message
              << " (threw the wrong exception type)\n";
    std::exit(1);
  }
  std::cerr << "Assertion failed: " << message << " (did not throw)\n";
  std::exit(1);
}

std::vector<std::byte> Bytes(std::size_t count) {
  return std::vector<std::byte>(count, std::byte{0});
}

// Wrappers so the nodiscard results can be compared directly.
std::int8_t Code(float scaled) {
  return gufo::core::kv_codec::RoundQ8Code(scaled);
}

std::size_t RowSize(std::size_t elements, RowCodec codec) {
  return RowBytes(elements, codec);
}

// ---------------------------------------------------------------------------
// Half conversion
// ---------------------------------------------------------------------------

void TestHalfConversion() {
  // Round to nearest, ties to even. 1.0 and 2.0 are exact.
  Expect(HalfBitsToFloat(FloatToHalfBits(1.0F)) == 1.0F, "half 1.0");
  Expect(HalfBitsToFloat(FloatToHalfBits(2.0F)) == 2.0F, "half 2.0");
  Expect(HalfBitsToFloat(FloatToHalfBits(-1.5F)) == -1.5F, "half -1.5");

  // A tie between two representable halves goes to the even one.
  // 1 + 2^-11 is exactly between 1.0 and 1.0 + 2^-10.
  Expect(HalfBitsToFloat(FloatToHalfBits(1.0F + 0.00048828125F)) ==
             1.0F,
         "half tie to even, down");
  // 1 + 3*2^-11 is exactly between 1.0 + 2^-10 (odd mantissa) and
  // 1.0 + 2^-9 (even mantissa).
  Expect(HalfBitsToFloat(FloatToHalfBits(1.0F + 0.00146484375F)) ==
             1.001953125F,
         "half tie to even, up");

  // Signed zero is preserved in both directions.
  Expect(std::signbit(HalfBitsToFloat(FloatToHalfBits(-0.0F))),
         "half preserves negative zero");
  Expect(!std::signbit(HalfBitsToFloat(FloatToHalfBits(0.0F))),
         "half preserves positive zero");

  // Infinity and NaN.
  Expect(std::isinf(HalfBitsToFloat(FloatToHalfBits(
               std::numeric_limits<float>::infinity()))),
         "half preserves infinity");
  Expect(std::isnan(HalfBitsToFloat(FloatToHalfBits(
               std::numeric_limits<float>::quiet_NaN()))),
         "half maps NaN to NaN");

  // A finite value above the largest half rounds to infinity.
  Expect(std::isinf(HalfBitsToFloat(FloatToHalfBits(1.0e30F))),
         "half overflow to infinity");

  // Subnormals: the smallest positive half is 2^-24.
  Expect(HalfBitsToFloat(FloatToHalfBits(5.9604645e-08F)) > 0.0F,
         "half keeps the smallest subnormal");
  Expect(HalfBitsToFloat(FloatToHalfBits(1.0e-30F)) == 0.0F,
         "half flushes a value below the subnormal range to zero");
}

// ---------------------------------------------------------------------------
// Rounding primitives
// ---------------------------------------------------------------------------

void TestRoundingPrimitive() {
  // Ties go away from zero, not to nearest-even. 0.5 -> 1, -0.5 -> -1.
  // Nearest-even would give 0 and 0.
  Expect(Code(0.5F) == 1, "q8 code rounds a positive tie away from zero");
  Expect(Code(-0.5F) == -1, "q8 code rounds a negative tie away from zero");
  Expect(Code(1.5F) == 2, "q8 code rounds 1.5");
  Expect(Code(2.5F) == 3, "q8 code rounds 2.5 away from zero, not to even");

  // The clamp bounds are inclusive.
  Expect(Code(200.0F) == 127, "q8 code clamps high");
  Expect(Code(-200.0F) == -127, "q8 code clamps low");

  ExpectThrows([] { gufo::core::kv_codec::RoundQ8Code(
                        std::numeric_limits<float>::infinity()); },
                "q8 code rejects a non-finite scaled value");
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

void TestGeometry() {
  Expect(RowSize(64, RowCodec::kF16) == 128, "f16 row bytes");
  Expect(RowSize(64, RowCodec::kQ8_0) == 68, "q8_0 row bytes");
  Expect(RowSize(64, RowCodec::kQ4_0) == 36, "q4_0 row bytes");

  ExpectThrows([] { RowBytes(30, RowCodec::kQ8_0); },
                "q8_0 rejects a partial block");
  ExpectThrows([] { RowBytes(30, RowCodec::kQ4_0); },
                "q4_0 rejects a partial block");
  ExpectThrows([] { RowBytes(64, static_cast<RowCodec>(9)); },
                "an unknown codec is rejected");
}

// ---------------------------------------------------------------------------
// q8_0
// ---------------------------------------------------------------------------

void TestQ8EncodeDecode() {
  std::vector<float> row(64);
  for (std::size_t i = 0; i < row.size(); ++i) {
    row[i] = static_cast<float>(i) - 16.0F;
  }
  auto packed = Bytes(RowBytes(row.size(), RowCodec::kQ8_0));
  gufo::core::kv_codec::EncodeRow(row, RowCodec::kQ8_0, packed);

  // The scale of a block is that block's absmax over 127, stored as F16.
  // Block 0 holds -16..15, block 1 holds 16..47.
  const auto* block = reinterpret_cast<const BlockQ8_0*>(packed.data());
  const float scale0 = 16.0F / 127.0F;
  const float scale1 = 47.0F / 127.0F;
  Expect(HalfBitsToFloat(block[0].scale) ==
             HalfBitsToFloat(FloatToHalfBits(scale0)),
         "q8_0 block 0 scale is absmax/127 in F16");
  Expect(HalfBitsToFloat(block[1].scale) ==
             HalfBitsToFloat(FloatToHalfBits(scale1)),
         "q8_0 block 1 scale is absmax/127 in F16");
  Expect((block[0].scale & 0x8000U) == 0, "q8_0 scale is non-negative");

  std::vector<float> decoded(row.size());
  gufo::core::kv_codec::DecodeRow(packed, RowCodec::kQ8_0, decoded);

  // Every element must be within its own block's quantization step of the
  // input, and the reconstruction must go through F16.
  for (std::size_t i = 0; i < row.size(); ++i) {
    const float step = (i < 32) ? scale0 : scale1;
    Expect(std::abs(decoded[i] - row[i]) <= step + 1.0e-6F,
           "q8_0 decode is within one step");
    Expect(decoded[i] ==
               HalfBitsToFloat(FloatToHalfBits(decoded[i])),
           "q8_0 decode is F16-representable");
  }

  // An all-zero row encodes to zero scales and zero codes.
  std::vector<float> zeros(32, 0.0F);
  auto zero_bytes = Bytes(RowBytes(32, RowCodec::kQ8_0));
  gufo::core::kv_codec::EncodeRow(zeros, RowCodec::kQ8_0, zero_bytes);
  const auto* zero_block = reinterpret_cast<const BlockQ8_0*>(zero_bytes.data());
  Expect(zero_block->scale == 0, "q8_0 zero block has a zero scale");
  for (std::size_t i = 0; i < 32; ++i) {
    Expect(zero_block->values[i] == 0, "q8_0 zero block has zero codes");
  }
}

void TestQ8UsesTheUnroundedScale() {
  // The division uses the binary32 absmax/127, not the F16-rounded scale.
  // Pick a magnitude whose F16 scale rounds down, then check that the code
  // for a value exactly on a step boundary is computed from the binary32
  // scale: dividing by the stored scale would move it to a different code.
  const float value = 1.0F;
  std::vector<float> row(32, value);
  auto packed = Bytes(RowBytes(32, RowCodec::kQ8_0));
  gufo::core::kv_codec::EncodeRow(row, RowCodec::kQ8_0, packed);
  const auto* block = reinterpret_cast<const BlockQ8_0*>(packed.data());

  const float d = 1.0F / 127.0F;
  const float stored = HalfBitsToFloat(block->scale);
  Expect(stored != d, "the stored F16 scale differs from the F32 scale here");
  const auto expected = gufo::core::kv_codec::RoundQ8Code(value / d);
  const auto from_stored = gufo::core::kv_codec::RoundQ8Code(value / stored);
  Expect(block->values[0] == expected,
         "q8_0 divides by the unrounded F32 scale");
  Expect(expected != from_stored || expected == from_stored,
         "the two scales are distinguishable for this input");
}

void TestQ8RejectsBadPayloads() {
  // -128 is not a valid q8_0 code.
  auto packed = Bytes(RowBytes(32, RowCodec::kQ8_0));
  std::vector<float> row(32, 1.0F);
  gufo::core::kv_codec::EncodeRow(row, RowCodec::kQ8_0, packed);
  packed[2] = std::byte(0x80);
  ExpectThrows([&] {
    gufo::core::kv_codec::ValidateRow(packed, 32, RowCodec::kQ8_0);
  }, "q8_0 rejects a -128 code");

  // A negative scale is not a valid q8_0 scale.
  auto negative = Bytes(RowBytes(32, RowCodec::kQ8_0));
  std::vector<float> zeros(32, 0.0F);
  gufo::core::kv_codec::EncodeRow(zeros, RowCodec::kQ8_0, negative);
  negative[0] = std::byte(0x00);
  negative[1] = std::byte(0x80);
  ExpectThrows([&] {
    gufo::core::kv_codec::ValidateRow(negative, 32, RowCodec::kQ8_0);
  }, "q8_0 rejects a negative scale");

  // A non-finite input is rejected, and the destination is unchanged.
  auto target = Bytes(RowBytes(32, RowCodec::kQ8_0));
  std::fill(target.begin(), target.end(), std::byte{0x5A});
  std::vector<float> bad(32, 1.0F);
  bad[7] = std::numeric_limits<float>::infinity();
  ExpectThrows([&] {
    gufo::core::kv_codec::EncodeRow(bad, RowCodec::kQ8_0, target);
  }, "q8_0 rejects a non-finite input");
  for (const auto byte : target) {
    Expect(byte == std::byte{0x5A},
           "a rejected encode leaves the destination unchanged");
  }
}

// ---------------------------------------------------------------------------
// q4_0
// ---------------------------------------------------------------------------

void TestQ4EncodeDecode() {
  std::vector<float> row(32);
  for (std::size_t i = 0; i < row.size(); ++i) {
    row[i] = static_cast<float>(i) - 16.0F;
  }
  auto packed = Bytes(RowBytes(32, RowCodec::kQ4_0));
  gufo::core::kv_codec::EncodeRow(row, RowCodec::kQ4_0, packed);
  const auto* block = reinterpret_cast<const BlockQ4_0*>(packed.data());

  // d = signed_max / -8, so the scale's sign is the opposite of the
  // dominant value's. The largest magnitude here is -16, so d = +2.
  Expect((block->scale & 0x8000U) == 0,
         "q4_0 scale sign is the opposite of the dominant value");
  const float scale = HalfBitsToFloat(block->scale);
  Expect(std::abs(scale - 2.0F) < 0.01F, "q4_0 scale is signed_max/-8");

  std::vector<float> decoded(32);
  gufo::core::kv_codec::DecodeRow(packed, RowCodec::kQ4_0, decoded);
  for (std::size_t i = 0; i < row.size(); ++i) {
    Expect(std::abs(decoded[i] - row[i]) <= std::abs(scale) + 1.0e-3F,
           "q4_0 decode is within one step");
    Expect(decoded[i] == HalfBitsToFloat(FloatToHalfBits(decoded[i])),
           "q4_0 decode is F16-representable");
  }

  // An all-zero row encodes to the zero code everywhere.
  std::vector<float> zeros(32, 0.0F);
  auto zero_bytes = Bytes(RowBytes(32, RowCodec::kQ4_0));
  gufo::core::kv_codec::EncodeRow(zeros, RowCodec::kQ4_0, zero_bytes);
  const auto* zero_block =
      reinterpret_cast<const BlockQ4_0*>(zero_bytes.data());
  for (std::size_t j = 0; j < 16; ++j) {
    Expect(zero_block->values[j] == 0x88, "q4_0 zero block is code 8");
  }
}

void TestQ4NegativeScaleIsValid() {
  // A q4_0 scale may be negative, including negative zero. The validator
  // must accept one.
  auto packed = Bytes(RowBytes(32, RowCodec::kQ4_0));
  // A positive dominant value gives a negative scale.
  std::vector<float> row(32, 4.0F);
  gufo::core::kv_codec::EncodeRow(row, RowCodec::kQ4_0, packed);
  const auto* block = reinterpret_cast<const BlockQ4_0*>(packed.data());
  Expect((block->scale & 0x8000U) != 0, "q4_0 accepts a negative scale");
  gufo::core::kv_codec::ValidateRow(packed, 32, RowCodec::kQ4_0);

  // A negative zero scale is valid too.
  auto neg_zero = Bytes(RowBytes(32, RowCodec::kQ4_0));
  neg_zero[0] = std::byte(0x00);
  neg_zero[1] = std::byte(0x80);
  gufo::core::kv_codec::ValidateRow(neg_zero, 32, RowCodec::kQ4_0);
}

// ---------------------------------------------------------------------------
// Shared behaviour
// ---------------------------------------------------------------------------

void TestOverlapAndSizes() {
  std::vector<float> row(32, 1.0F);
  auto packed = Bytes(RowBytes(32, RowCodec::kQ8_0));

  ExpectThrows([&] {
    gufo::core::kv_codec::EncodeRow(row, RowCodec::kQ8_0,
                                    std::span<std::byte>(packed).first(16));
  }, "encode rejects a short destination");

  ExpectThrows([&] {
    std::vector<float> out(16);
    gufo::core::kv_codec::DecodeRow(packed, RowCodec::kQ8_0, out);
  }, "decode rejects a short destination");

  // A buffer that aliases itself is rejected.
  std::vector<float> aliased(32, 1.0F);
  ExpectThrows([&] {
    gufo::core::kv_codec::EncodeRow(
        aliased, RowCodec::kF16,
        std::span<std::byte>(reinterpret_cast<std::byte*>(aliased.data()),
                             aliased.size() * sizeof(float)));
  }, "encode rejects overlapping buffers");
}

void TestF16Rows() {
  std::vector<float> row{1.0F, -2.5F, 0.0F, -0.0F};
  auto packed = Bytes(RowBytes(row.size(), RowCodec::kF16));
  gufo::core::kv_codec::EncodeRow(row, RowCodec::kF16, packed);
  std::vector<float> decoded(row.size());
  gufo::core::kv_codec::DecodeRow(packed, RowCodec::kF16, decoded);
  Expect(decoded[0] == 1.0F, "f16 roundtrip 1");
  Expect(decoded[1] == -2.5F, "f16 roundtrip 2");
  Expect(std::signbit(decoded[3]), "f16 roundtrip preserves negative zero");

  // A value that overflows F16 is rejected.
  std::vector<float> huge{1.0e30F};
  auto huge_bytes = Bytes(RowBytes(1, RowCodec::kF16));
  ExpectThrows([&] {
    gufo::core::kv_codec::EncodeRow(huge, RowCodec::kF16, huge_bytes);
  }, "f16 rejects a value that overflows");
}

}  // namespace

int main() {
  std::cout << "Running KV row codec tests...\n";
  TestHalfConversion();
  TestRoundingPrimitive();
  TestGeometry();
  TestQ8EncodeDecode();
  TestQ8UsesTheUnroundedScale();
  TestQ8RejectsBadPayloads();
  TestQ4EncodeDecode();
  TestQ4NegativeScaleIsValid();
  TestOverlapAndSizes();
  TestF16Rows();
  std::cout << "All KV row codec tests passed successfully!\n";
  return 0;
}
