#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/kv_codec/kv_q8_device.hpp"
#include "src/families/qwen/kernels/kv_q8.hpp"
#include "src/families/qwen/kernels/ops/kv.hpp"

namespace gufo::hip {
namespace {

using detail::Q8KvBlock;

// One warp per q8_0 block: lane i owns element i of the block, so the
// block's absmax is a warp reduction and each lane writes its own code.
// K and V are separate rows and get separate scales.
__global__ void EncodeQ8KvRowsKernel(
    const float* __restrict__ source_k, const float* __restrict__ source_v,
    Q8KvBlock* __restrict__ k_cache, Q8KvBlock* __restrict__ v_cache,
    std::uint32_t layer_idx, std::uint32_t start_position,
    const std::uint32_t* __restrict__ d_position, std::uint32_t rows,
    std::uint32_t max_context, std::uint32_t blocks_per_row,
    std::uint32_t kv_width) {
  const std::uint32_t block_index = blockIdx.x;
  const std::uint32_t row = blockIdx.y;
  if (row >= rows) {
    return;
  }
  // The decode path keeps its position on the device.
  if (d_position != nullptr) {
    start_position = *d_position;
  }
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t element = (block_index * 32) + lane;
  const std::size_t source_offset =
      (static_cast<std::size_t>(row) * kv_width) + element;
  const float key = source_k[source_offset];
  const float value = source_v[source_offset];

  const float key_scale = gufo::core::kv_codec::device::Q8KvScale(
      gufo::core::kv_codec::device::WarpMaxAbs(fabsf(key)));
  const float value_scale = gufo::core::kv_codec::device::Q8KvScale(
      gufo::core::kv_codec::device::WarpMaxAbs(fabsf(value)));

  const std::size_t row_offset =
      (static_cast<std::size_t>(layer_idx) * max_context +
       (start_position + row)) *
      blocks_per_row;
  auto* key_block = k_cache + row_offset + block_index;
  auto* value_block = v_cache + row_offset + block_index;
  if (lane == 0) {
    key_block->scale = __float2half(key_scale);
    value_block->scale = __float2half(value_scale);
  }
  key_block->values[lane] = gufo::core::kv_codec::device::Q8KvClamp(
      gufo::core::kv_codec::device::Q8KvCode(key, key_scale));
  value_block->values[lane] = gufo::core::kv_codec::device::Q8KvClamp(
      gufo::core::kv_codec::device::Q8KvCode(value, value_scale));
}

}  // namespace

void LaunchEncodeQ8KvRows(const float* source_k, const float* source_v,
                          void* k_cache_q8, void* v_cache_q8,
                          std::uint32_t layer_idx, std::uint32_t start_position,
                          std::uint32_t rows, std::uint32_t max_context,
                          std::uint32_t kv_width, hipStream_t stream) {
  if (source_k == nullptr || source_v == nullptr || k_cache_q8 == nullptr ||
      v_cache_q8 == nullptr || rows == 0) {
    return;
  }
  const std::uint32_t blocks_per_row = detail::Q8KvBlocksPerRow(kv_width);
  if (blocks_per_row == 0) {
    return;
  }
  const dim3 grid(blocks_per_row, rows);
  EncodeQ8KvRowsKernel<<<grid, 32, 0, stream>>>(
      source_k, source_v, static_cast<Q8KvBlock*>(k_cache_q8),
      static_cast<Q8KvBlock*>(v_cache_q8), layer_idx, start_position, nullptr,
      rows, max_context, blocks_per_row, kv_width);
}

void LaunchEncodeQ8KvRowsAtDevicePosition(
    const float* source_k, const float* source_v, void* k_cache_q8,
    void* v_cache_q8, std::uint32_t layer_idx, const std::uint32_t* d_position,
    std::uint32_t rows, std::uint32_t max_context, std::uint32_t kv_width,
    hipStream_t stream) {
  if (source_k == nullptr || source_v == nullptr || k_cache_q8 == nullptr ||
      v_cache_q8 == nullptr || rows == 0 || d_position == nullptr) {
    return;
  }
  const std::uint32_t blocks_per_row = detail::Q8KvBlocksPerRow(kv_width);
  if (blocks_per_row == 0) {
    return;
  }
  const dim3 grid(blocks_per_row, rows);
  EncodeQ8KvRowsKernel<<<grid, 32, 0, stream>>>(
      source_k, source_v, static_cast<Q8KvBlock*>(k_cache_q8),
      static_cast<Q8KvBlock*>(v_cache_q8), layer_idx, 0, d_position, rows,
      max_context, blocks_per_row, kv_width);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
