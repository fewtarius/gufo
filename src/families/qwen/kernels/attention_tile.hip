#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/families/qwen/hip_common/attention_policy.hpp"
#include "src/families/qwen/kernels/kv_read.hpp"
#include "src/families/qwen/kernels/ops/attention.hpp"
#include "src/families/qwen/kernels/ops/kv.hpp"

namespace gufo::hip {

__global__ void SyncTiledAttentionKvPrefixKernel(
    const float* __restrict__ k_cache, const float* __restrict__ v_cache,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    std::uint32_t layer_idx, std::uint32_t prefix_length,
    std::uint32_t max_context, std::uint32_t num_kv_heads,
    std::uint32_t head_dim) {
  const std::size_t pos = blockIdx.y;
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  const std::size_t kv_width =
      static_cast<std::size_t>(num_kv_heads) * head_dim;
  if (pos >= prefix_length || index >= kv_width) {
    return;
  }
  const std::size_t kv_head = index / head_dim;
  const std::size_t dimension = index % head_dim;
  const std::size_t f32_cache_offset =
      (((static_cast<std::size_t>(layer_idx) * num_kv_heads + kv_head) *
            max_context +
        pos) *
       head_dim) +
      dimension;
  const std::size_t f16_cache_offset =
      ((static_cast<std::size_t>(layer_idx) * max_context + pos) * kv_width) +
      index;
  k_cache_f16[f16_cache_offset] = __float2half_rn(k_cache[f32_cache_offset]);
  v_cache_f16[f16_cache_offset] = __float2half_rn(v_cache[f32_cache_offset]);
}

constexpr std::uint32_t kTileAttentionHeadDim = detail::kTiledAttentionHeadDim;
constexpr std::uint32_t kTileAttentionQueryRows = 16;
constexpr std::uint32_t kTileAttentionHeadsPerBlock = 2;
constexpr std::uint32_t kTileAttentionColumns =
    kTileAttentionQueryRows * kTileAttentionHeadsPerBlock;
static_assert((kTileAttentionColumns % 8) == 0,
              "the query tile must divide evenly across the eight waves");
constexpr std::uint32_t kTileAttentionKeyRows = 64;
constexpr std::uint32_t kTileAttentionHalf2Dim = kTileAttentionHeadDim / 2;
constexpr std::uint32_t kTileAttentionKeyHalf2Dim = kTileAttentionHalf2Dim / 2;
constexpr std::uint32_t kTileAttentionValueRows = 32;
constexpr std::uint32_t kTileAttentionHalf2Padding = 1;
constexpr std::uint32_t kTileAttentionKeyRowStride =
    kTileAttentionKeyHalf2Dim + kTileAttentionHalf2Padding;
static_assert((kTileAttentionKeyRowStride % 2) == 1,
              "Wave32 key rows require an odd stride across 32 LDS banks");

__device__ __forceinline__ float TileWaveReduceMax(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down(value, offset));
  }
  return __shfl(value, 0);
}

__device__ __forceinline__ float TileWaveReduceSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down(value, offset);
  }
  return __shfl(value, 0);
}

__device__ __forceinline__ void TileHalf2Dot(float& accumulator, __half2 lhs,
                                             __half2 rhs) {
  asm volatile("v_dot2_f32_f16 %0, %1, %2, %0"
               : "+v"(accumulator)
               : "v"(lhs), "v"(rhs));
}

template<std::uint32_t KvHeads>
__global__ void PackTiledAttentionKvKernel(
    const float* __restrict__ k, const float* __restrict__ v,
    __half* __restrict__ k_cache_f16, __half* __restrict__ v_cache_f16,
    float* __restrict__ k_cache, float* __restrict__ v_cache,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context) {
  const std::size_t token = blockIdx.y;
  const std::size_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
  constexpr std::size_t kv_width = KvHeads * kTileAttentionHeadDim;
  if (token >= batch_size || index >= kv_width) {
    return;
  }

  const std::size_t source_offset = (token * kv_width) + index;
  const std::size_t position = start_pos + token;
  const std::size_t kv_head = index / kTileAttentionHeadDim;
  const std::size_t dimension = index % kTileAttentionHeadDim;
  const std::size_t f16_cache_offset =
      ((static_cast<std::size_t>(layer_idx) * max_context + position) *
       kv_width) +
      index;
  const std::size_t f32_cache_offset =
      (((static_cast<std::size_t>(layer_idx) * KvHeads + kv_head) *
            max_context +
        position) *
       kTileAttentionHeadDim) +
      dimension;
  const float key = k[source_offset];
  const float value = v[source_offset];
  k_cache_f16[f16_cache_offset] = __float2half_rn(key);
  v_cache_f16[f16_cache_offset] = __float2half_rn(value);
  if (k_cache != nullptr) {
    k_cache[f32_cache_offset] = key;
  }
  if (v_cache != nullptr) {
    v_cache[f32_cache_offset] = value;
  }
}

template<QwenKvFormat Format, std::uint32_t QueryHeads, std::uint32_t KvHeads>
__launch_bounds__(256, 3) __global__ void QwenTiledAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const void* __restrict__ k_cache, const void* __restrict__ v_cache,
    float* __restrict__ out_context, std::size_t plane_offset,
    std::uint32_t start_pos, std::size_t batch_size,
    float* __restrict__ lse_out, std::uint32_t key_begin) {
  constexpr std::uint32_t kGqa = QueryHeads / KvHeads;
  static_assert((kGqa % kTileAttentionHeadsPerBlock) == 0,
                "GQA heads must divide evenly across tiled blocks");
  const std::uint32_t lane = threadIdx.x;
  const std::uint32_t wave = threadIdx.y;
  const std::uint32_t query_start = blockIdx.x * kTileAttentionQueryRows;
  const std::uint32_t head_pair = blockIdx.y;
  const std::uint32_t kv_head = head_pair / (kGqa / 2);
  const std::uint32_t pair_in_group = head_pair % (kGqa / 2);
  const std::uint32_t first_query_head =
      (kv_head * kGqa) + (pair_in_group * kTileAttentionHeadsPerBlock);
  constexpr std::size_t attention_width = QueryHeads * kTileAttentionHeadDim;
  constexpr std::size_t kv_width = KvHeads * kTileAttentionHeadDim;
  constexpr float attention_scale = 1.0F / 16.0F;

  __shared__ __half2 q_tile[kTileAttentionColumns][kTileAttentionHalf2Dim];
  __shared__ __half2
      kv_tile[kTileAttentionKeyRows * kTileAttentionKeyRowStride];
  __shared__ __half weights[kTileAttentionColumns][kTileAttentionKeyRows];

  constexpr std::uint32_t columns_per_wave = kTileAttentionColumns / 8;
  float running_max[columns_per_wave];
  float running_sum[columns_per_wave];
  float2 output[columns_per_wave][kTileAttentionHalf2Dim / 32];
#pragma unroll
  for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
    running_max[column] = -INFINITY;
    running_sum[column] = 0.0F;
#pragma unroll
    for (std::uint32_t slot = 0; slot < kTileAttentionHalf2Dim / 32; ++slot) {
      output[column][slot] = make_float2(0.0F, 0.0F);
    }
  }

#pragma unroll
  for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
    const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
    const std::uint32_t query_offset =
        tiled_column / kTileAttentionHeadsPerBlock;
    const std::uint32_t query_head =
        first_query_head + (tiled_column % kTileAttentionHeadsPerBlock);
    const std::uint32_t local_query_position = query_start + query_offset;
#pragma unroll
    for (std::uint32_t half2_index = lane; half2_index < kTileAttentionHalf2Dim;
         half2_index += 32) {
      __half2 packed =
          __halves2half2(__float2half_rn(0.0F), __float2half_rn(0.0F));
      if (local_query_position < batch_size) {
        const std::size_t source =
            (static_cast<std::size_t>(local_query_position) * attention_width) +
            (static_cast<std::size_t>(query_head) * kTileAttentionHeadDim) +
            (2 * half2_index);
        packed =
            __halves2half2(__float2half_rn(q[source] * attention_scale),
                           __float2half_rn(q[source + 1] * attention_scale));
      }
      q_tile[tiled_column][half2_index] = packed;
    }
  }
  __syncthreads();

  const std::uint32_t context_end =
      start_pos + static_cast<std::uint32_t>(batch_size);
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kTileAttentionQueryRows);
  for (std::uint32_t key_start = key_begin; key_start < max_visible;
       key_start += kTileAttentionKeyRows) {
    float scores[columns_per_wave][2] = {};
#pragma unroll
    for (std::uint32_t key_dimension = 0; key_dimension < 2; ++key_dimension) {
      for (std::uint32_t key_row = wave; key_row < kTileAttentionKeyRows;
           key_row += 8) {
        const std::uint32_t key_position = key_start + key_row;
#pragma unroll
        for (std::uint32_t half2_index = lane;
             half2_index < kTileAttentionKeyHalf2Dim; half2_index += 32) {
          __half2 packed =
              __halves2half2(__float2half_rn(0.0F), __float2half_rn(0.0F));
          if (key_position < context_end) {
            const std::size_t element =
                plane_offset +
                (static_cast<std::size_t>(key_position) * kv_width) +
                (static_cast<std::size_t>(kv_head) * kTileAttentionHeadDim) +
                (key_dimension * kTileAttentionKeyHalf2Dim * 2) +
                (2 * half2_index);
            packed = detail::KvPlaneAt<Format>(k_cache, element).LoadHalf2(0);
          }
          kv_tile[(key_row * kTileAttentionKeyRowStride) + half2_index] =
              packed;
        }
      }
      __syncthreads();

#pragma unroll 4
      for (std::uint32_t half2_index = 0;
           half2_index < kTileAttentionKeyHalf2Dim; ++half2_index) {
        const __half2 key_values[2] = {
            kv_tile[(lane * kTileAttentionKeyRowStride) + half2_index],
            kv_tile[((lane + 32) * kTileAttentionKeyRowStride) + half2_index]};
#pragma unroll
        for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
          const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
          const __half2 query_value =
              q_tile[tiled_column]
                    [(key_dimension * kTileAttentionKeyHalf2Dim) + half2_index];
          TileHalf2Dot(scores[column][0], key_values[0], query_value);
          TileHalf2Dot(scores[column][1], key_values[1], query_value);
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
      const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
      const std::uint32_t query_offset =
          tiled_column / kTileAttentionHeadsPerBlock;
      const std::uint32_t local_query_position = query_start + query_offset;
      const std::uint32_t absolute_query_position =
          start_pos + local_query_position;
      const std::uint32_t key_positions[2] = {key_start + lane,
                                              key_start + lane + 32};
      const bool valid[2] = {local_query_position < batch_size &&
                                 key_positions[0] <= absolute_query_position &&
                                 key_positions[0] < context_end,
                             local_query_position < batch_size &&
                                 key_positions[1] <= absolute_query_position &&
                                 key_positions[1] < context_end};
      const float score_values[2] = {valid[0] ? scores[column][0] : -INFINITY,
                                     valid[1] ? scores[column][1] : -INFINITY};
      const float tile_max =
          TileWaveReduceMax(fmaxf(score_values[0], score_values[1]));
      const float next_max = fmaxf(running_max[column], tile_max);
      const float prior_scale = isfinite(running_max[column])
                                    ? expf(running_max[column] - next_max)
                                    : 0.0F;
      const float weight_values[2] = {
          valid[0] ? expf(score_values[0] - next_max) : 0.0F,
          valid[1] ? expf(score_values[1] - next_max) : 0.0F};
      running_max[column] = next_max;
      running_sum[column] = (running_sum[column] * prior_scale) +
                            weight_values[0] + weight_values[1];
#pragma unroll
      for (std::uint32_t slot = 0; slot < kTileAttentionHalf2Dim / 32; ++slot) {
        output[column][slot].x *= prior_scale;
        output[column][slot].y *= prior_scale;
      }
      weights[tiled_column][lane] = __float2half_rn(weight_values[0]);
      weights[tiled_column][lane + 32] = __float2half_rn(weight_values[1]);
    }
    __syncthreads();

    for (std::uint32_t value_group = 0; value_group < 2; ++value_group) {
      for (std::uint32_t value_row = wave; value_row < kTileAttentionValueRows;
           value_row += 8) {
        const std::uint32_t key_position =
            key_start + (value_group * kTileAttentionValueRows) + value_row;
#pragma unroll
        for (std::uint32_t half2_index = lane;
             half2_index < kTileAttentionHalf2Dim; half2_index += 32) {
          __half2 packed =
              __halves2half2(__float2half_rn(0.0F), __float2half_rn(0.0F));
          if (key_position < context_end) {
            const std::size_t element =
                plane_offset +
                (static_cast<std::size_t>(key_position) * kv_width) +
                (static_cast<std::size_t>(kv_head) * kTileAttentionHeadDim) +
                (2 * half2_index);
            packed = detail::KvPlaneAt<Format>(v_cache, element).LoadHalf2(0);
          }
          kv_tile[(value_row * kTileAttentionHalf2Dim) + half2_index] = packed;
        }
      }
      __syncthreads();

#pragma unroll
      for (std::uint32_t value_row = 0; value_row < kTileAttentionValueRows;
           value_row += 2) {
        __half2 weight_values[columns_per_wave];
#pragma unroll
        for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
          const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
          weight_values[column] = __halves2half2(
              weights[tiled_column]
                     [(value_group * kTileAttentionValueRows) + value_row],
              weights[tiled_column]
                     [(value_group * kTileAttentionValueRows) + value_row + 1]);
        }
#pragma unroll
        for (std::uint32_t slot = 0; slot < kTileAttentionHalf2Dim / 32;
             ++slot) {
          const std::uint32_t half2_index = lane + (slot * 32);
          const __half2 value0 =
              kv_tile[(value_row * kTileAttentionHalf2Dim) + half2_index];
          const __half2 value1 =
              kv_tile[((value_row + 1) * kTileAttentionHalf2Dim) + half2_index];
          const __half2 values_x =
              __halves2half2(__low2half(value0), __low2half(value1));
          const __half2 values_y =
              __halves2half2(__high2half(value0), __high2half(value1));
#pragma unroll
          for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
            TileHalf2Dot(output[column][slot].x, values_x,
                         weight_values[column]);
            TileHalf2Dot(output[column][slot].y, values_y,
                         weight_values[column]);
          }
        }
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (std::uint32_t column = 0; column < columns_per_wave; ++column) {
    const std::uint32_t tiled_column = (wave * columns_per_wave) + column;
    const std::uint32_t query_offset =
        tiled_column / kTileAttentionHeadsPerBlock;
    const std::uint32_t query_head =
        first_query_head + (tiled_column % kTileAttentionHeadsPerBlock);
    const std::uint32_t local_query_position = query_start + query_offset;
    const float denominator = TileWaveReduceSum(running_sum[column]);
    if (local_query_position >= batch_size) {
      continue;
    }
    if (lse_out != nullptr && lane == 0) {
      // log-sum-exp of this partial, in [head, token] layout.
      lse_out[(static_cast<std::size_t>(query_head) * batch_size) +
              local_query_position] =
          (denominator > 0.0F) ? (running_max[column] + logf(denominator))
                               : -INFINITY;
    }
    if (denominator == 0.0F) {
      if (lse_out != nullptr) {
#pragma unroll
        for (std::uint32_t slot = 0; slot < kTileAttentionHalf2Dim / 32;
             ++slot) {
          const std::uint32_t dimension = 2 * (lane + (slot * 32));
          const std::size_t output_offset =
              (static_cast<std::size_t>(local_query_position) *
               attention_width) +
              (static_cast<std::size_t>(query_head) * kTileAttentionHeadDim) +
              dimension;
          out_context[output_offset] = 0.0F;
          out_context[output_offset + 1] = 0.0F;
        }
      }
      continue;
    }
#pragma unroll
    for (std::uint32_t slot = 0; slot < kTileAttentionHalf2Dim / 32; ++slot) {
      const std::uint32_t half2_index = lane + (slot * 32);
      const std::uint32_t dimension = 2 * half2_index;
      const std::size_t output_offset =
          (static_cast<std::size_t>(local_query_position) * attention_width) +
          (static_cast<std::size_t>(query_head) * kTileAttentionHeadDim) +
          dimension;
      const float2 result = output[column][slot];
      // When this is one half of a split the gate is applied once on the merged
      // result instead, so only normalize here.
      const float gate0 = (gate == nullptr || lse_out != nullptr)
                              ? 1.0F
                              : 1.0F / (1.0F + expf(-gate[output_offset]));
      const float gate1 = (gate == nullptr || lse_out != nullptr)
                              ? 1.0F
                              : 1.0F / (1.0F + expf(-gate[output_offset + 1]));
      out_context[output_offset] = (result.x / denominator) * gate0;
      out_context[output_offset + 1] = (result.y / denominator) * gate1;
    }
  }
}

template<QwenKvFormat Format, std::uint32_t QueryHeads, std::uint32_t KvHeads>
bool LaunchTiledAttention(const float* q, const float* k, const float* v,
                          const float* gate, float* k_cache, float* v_cache,
                          QwenCompactKv compact, float* out_context,
                          std::uint32_t layer_idx, std::uint32_t start_pos,
                          std::size_t batch_size, std::uint32_t max_context,
                          std::uint32_t num_kv_heads, std::uint32_t head_dim,
                          hipStream_t stream, float* lse_out,
                          std::uint32_t key_begin, bool skip_kv_write) {
  constexpr std::size_t block_size = 256;
  constexpr std::size_t kv_width = KvHeads * kTileAttentionHeadDim;
  const dim3 pack_grid(
      static_cast<unsigned int>((kv_width + block_size - 1) / block_size),
      static_cast<unsigned int>(batch_size));
  if (!skip_kv_write && start_pos > 0 && k_cache != nullptr &&
      v_cache != nullptr && compact.format == QwenKvFormat::kFp16) {
    const dim3 sync_grid(
        static_cast<unsigned int>((kv_width + block_size - 1) / block_size),
        static_cast<unsigned int>(start_pos));
    SyncTiledAttentionKvPrefixKernel<<<sync_grid, block_size, 0, stream>>>(
        k_cache, v_cache, static_cast<__half*>(compact.k),
        static_cast<__half*>(compact.v), layer_idx, start_pos, max_context,
        num_kv_heads, head_dim);
  }
  if (!skip_kv_write) {
    if (compact.format == QwenKvFormat::kQ8_0) {
      LaunchEncodeQ8KvRows(k, v, compact.k, compact.v, layer_idx, start_pos,
                           static_cast<std::uint32_t>(batch_size), max_context,
                           static_cast<std::uint32_t>(kv_width), stream);
    } else {
      // The binary16 plane only exists for the FP16 format. The q8_0 plane
      // is written by the encoder above.
      const bool binary16_plane = compact.format == QwenKvFormat::kFp16;
      PackTiledAttentionKvKernel<KvHeads><<<pack_grid, block_size, 0, stream>>>(
          k, v, binary16_plane ? static_cast<__half*>(compact.k) : nullptr,
          binary16_plane ? static_cast<__half*>(compact.v) : nullptr, k_cache,
          v_cache, layer_idx, start_pos, batch_size, max_context);
    }
  }

  const dim3 attention_grid(
      static_cast<unsigned int>((batch_size + kTileAttentionQueryRows - 1) /
                                kTileAttentionQueryRows),
      KvHeads * ((QueryHeads / KvHeads) / kTileAttentionHeadsPerBlock));
  const dim3 attention_block(32, 8);
  // The canonical plane holds every attention layer back to back, each
  // max_context rows of kv_width elements.
  const std::size_t plane_offset =
      static_cast<std::size_t>(layer_idx) * max_context * kv_width;
  QwenTiledAttentionKernel<Format, QueryHeads, KvHeads>
      <<<attention_grid, attention_block, 0, stream>>>(
          q, gate, compact.k, compact.v, out_context, plane_offset, start_pos,
          batch_size, lse_out, key_begin);
  return true;
}

// The kernel is compiled per head geometry: it pairs two query heads that
// share one KV head and reads the K/V plane with the row stride of its own
// KV-head count. A geometry without an instantiation falls back to the
// baseline attention.
template<QwenKvFormat Format>
bool DispatchTiledAttention(std::uint32_t num_heads, std::uint32_t num_kv_heads,
                            const float* q, const float* k, const float* v,
                            const float* gate, float* k_cache, float* v_cache,
                            QwenCompactKv compact, float* out_context,
                            std::uint32_t layer_idx, std::uint32_t start_pos,
                            std::size_t batch_size, std::uint32_t max_context,
                            std::uint32_t head_dim, hipStream_t stream,
                            float* lse_out, std::uint32_t key_begin,
                            bool skip_kv_write) {
  switch ((num_heads * 1000U) + num_kv_heads) {
    case (24U * 1000U) + 4U:
      return LaunchTiledAttention<Format, 24, 4>(
          q, k, v, gate, k_cache, v_cache, compact, out_context, layer_idx,
          start_pos, batch_size, max_context, num_kv_heads, head_dim, stream,
          lse_out, key_begin, skip_kv_write);
    case (16U * 1000U) + 2U:
      return LaunchTiledAttention<Format, 16, 2>(
          q, k, v, gate, k_cache, v_cache, compact, out_context, layer_idx,
          start_pos, batch_size, max_context, num_kv_heads, head_dim, stream,
          lse_out, key_begin, skip_kv_write);
    default:
      return false;
  }
}

bool LaunchBatchedAttentionTile(
    const float* q, const float* k, const float* v, const float* gate,
    float* k_cache, float* v_cache, QwenCompactKv compact, float* out_context,
    std::uint32_t layer_idx, std::uint32_t start_pos, std::size_t batch_size,
    std::uint32_t max_context, std::uint32_t num_heads,
    std::uint32_t num_kv_heads, std::uint32_t head_dim, hipStream_t stream,
    float* lse_out, std::uint32_t key_begin, bool skip_kv_write) {
  const detail::AttentionSupportParams support_params{
      .batch_size = batch_size,
      .start_pos = start_pos,
      .max_context = max_context,
      .num_heads = num_heads,
      .num_kv_heads = num_kv_heads,
      .head_dim = head_dim,
      .has_k_cache_f16 = compact.k != nullptr,
      .has_v_cache_f16 = compact.v != nullptr,
  };
  if (!detail::IsTiledAttentionLayoutSupported(support_params) ||
      !detail::IsTiledAttentionGeometrySupported(num_heads, num_kv_heads)) {
    return false;
  }

  if (compact.format == QwenKvFormat::kQ8_0) {
    return DispatchTiledAttention<QwenKvFormat::kQ8_0>(
        num_heads, num_kv_heads, q, k, v, gate, k_cache, v_cache, compact,
        out_context, layer_idx, start_pos, batch_size, max_context, head_dim,
        stream, lse_out, key_begin, skip_kv_write);
  }
  return DispatchTiledAttention<QwenKvFormat::kFp16>(
      num_heads, num_kv_heads, q, k, v, gate, k_cache, v_cache, compact,
      out_context, layer_idx, start_pos, batch_size, max_context, head_dim,
      stream, lse_out, key_begin, skip_kv_write);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
