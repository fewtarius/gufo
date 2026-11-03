// Checks the Laguna q8_0 KV path against the host codec.
//
// The device store kernel and the device dequantizer have to agree with
// src/core/kv_codec exactly: the store kernel writes the packed bytes the
// host would write, and the dequantizer reconstructs what the host would
// reconstruct. Any difference is a numeric divergence that would show up
// as a model output change nobody could explain.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <gufo/ext/experimental/hip_utils.hpp>
#include "src/core/kv_codec/codec.hpp"
#include "src/models/laguna/gpu/kernels/kernels.hpp"
#include "src/models/laguna/gpu/kernels/kv_q8.hpp"

namespace {

using gufo::core::kv_codec::RowBytes;
using gufo::core::kv_codec::RowCodec;

void Expect(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "Assertion failed: %s\n", message);
    std::exit(1);
  }
}

constexpr std::uint32_t kRowDim = 1024;  // num_kv 8 x head_dim 128
constexpr std::uint32_t kNumTokens = 4;

}  // namespace

int main() {
  std::mt19937 rng(20261103);
  std::uniform_real_distribution<float> uniform(-4.0F, 4.0F);

  // The host reference for one row.
  std::vector<float> row(kRowDim);
  for (auto& v : row) {
    v = uniform(rng);
  }
  std::vector<float> v_row(kRowDim);
  for (auto& v : v_row) {
    v = uniform(rng);
  }

  std::vector<std::byte> host_k(RowBytes(kRowDim, RowCodec::kQ8_0));
  std::vector<std::byte> host_v(RowBytes(kRowDim, RowCodec::kQ8_0));
  gufo::core::kv_codec::EncodeRow(row, RowCodec::kQ8_0, host_k);
  gufo::core::kv_codec::EncodeRow(v_row, RowCodec::kQ8_0, host_v);

  // The device store kernel writes one row per token.
  std::vector<float> device_k(static_cast<std::size_t>(kNumTokens) * kRowDim);
  std::vector<float> device_v(static_cast<std::size_t>(kNumTokens) * kRowDim);
  for (std::size_t t = 0; t < kNumTokens; ++t) {
    for (std::uint32_t i = 0; i < kRowDim; ++i) {
      device_k[t * kRowDim + i] = row[i];
      device_v[t * kRowDim + i] = v_row[i];
    }
  }

  const std::size_t row_bytes = RowBytes(kRowDim, RowCodec::kQ8_0);
  float* d_k_in = nullptr;
  float* d_v_in = nullptr;
  std::byte* d_k = nullptr;
  std::byte* d_v = nullptr;
  std::uint32_t* d_pos = nullptr;
  HIP_CHECK(hipMalloc(&d_k_in, device_k.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v_in, device_v.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, row_bytes * kNumTokens));
  HIP_CHECK(hipMalloc(&d_v, row_bytes * kNumTokens));
  HIP_CHECK(hipMalloc(&d_pos, sizeof(std::uint32_t)));
  const std::uint32_t start_pos = 0;
  HIP_CHECK(hipMemcpy(d_pos, &start_pos, sizeof(start_pos),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k_in, device_k.data(),
                      device_k.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v_in, device_v.data(),
                      device_v.size() * sizeof(float), hipMemcpyHostToDevice));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  gufo::models::laguna::gpu::rocm::StoreKvQ8(
      d_k_in, d_v_in, d_k, d_v, kNumTokens, kRowDim, d_pos, 0, stream);
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<std::byte> got_k(row_bytes * kNumTokens);
  std::vector<std::byte> got_v(row_bytes * kNumTokens);
  HIP_CHECK(hipMemcpy(got_k.data(), d_k, got_k.size(), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(got_v.data(), d_v, got_v.size(), hipMemcpyDeviceToHost));

  // The device store must produce the host's bytes exactly.
  bool k_match = std::memcmp(got_k.data(), host_k.data(), row_bytes) == 0;
  bool v_match = std::memcmp(got_v.data(), host_v.data(), row_bytes) == 0;
  if (!k_match) {
    for (std::size_t i = 0; i < row_bytes; ++i) {
      if (got_k[i] != host_k[i]) {
        std::fprintf(stderr,
                     "K byte %zu differs: device=%02x host=%02x\n", i,
                     static_cast<unsigned>(
                         std::to_integer<std::uint8_t>(got_k[i])),
                     static_cast<unsigned>(
                         std::to_integer<std::uint8_t>(host_k[i])));
        break;
      }
    }
  }
  Expect(k_match, "the device K store matches the host codec byte for byte");
  Expect(v_match, "the device V store matches the host codec byte for byte");

  // And the device dequantizer must reconstruct what the host reconstructs.
  std::vector<float> host_decoded(kRowDim);
  gufo::core::kv_codec::DecodeRow(host_k, RowCodec::kQ8_0, host_decoded);

  // The attention kernel reconstructs eight elements at a time from the
  // block that holds them. Mirror that exactly.
  double max_abs = 0.0;
  for (std::uint32_t group = 0; group < kRowDim / 8; ++group) {
    const std::size_t element = group * 8;
    const auto* block = reinterpret_cast<const gufo::models::laguna::gpu::rocm::
                                           detail::Q8KvBlock*>(
        got_k.data()) + element / 32;
    const auto packed = gufo::models::laguna::gpu::rocm::detail::
        DequantizeQ8Group(block, static_cast<std::uint32_t>(element % 32));
    __half halves[8];
    std::memcpy(halves, &packed, sizeof(packed));
    for (int i = 0; i < 8; ++i) {
      const float device_value = __half2float(halves[i]);
      max_abs = std::max(
          max_abs, std::fabs(static_cast<double>(device_value) -
                              static_cast<double>(host_decoded[element + i])));
    }
  }
  std::printf("max reconstruction difference: %g\n", max_abs);
  Expect(max_abs == 0.0,
         "the device dequantizer matches the host decode exactly");

  HIP_CHECK(hipFree(d_k_in));
  HIP_CHECK(hipFree(d_v_in));
  HIP_CHECK(hipFree(d_k));
  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_pos));
  HIP_CHECK(hipStreamDestroy(stream));
  std::printf("Laguna q8_0 KV test passed on gfx1151.\n");
  return 0;
}
