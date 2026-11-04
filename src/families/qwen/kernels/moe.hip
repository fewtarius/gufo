#if defined(ENGINE_ENABLE_HIP)
#include "src/families/qwen/kernels/ops/moe.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace gufo::hip {
namespace {

constexpr std::uint32_t kMoeRouterThreads = 256;
constexpr std::uint32_t kMoeRouterWarps = kMoeRouterThreads / 32;
/// The largest expert vocabulary the router kernel holds in one lane's
/// register array: 32 experts per lane across 32 lanes.
constexpr std::uint32_t kMoeMaxExperts = 1024;

__device__ float WarpReduceMax(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor(value, offset));
  }
  return value;
}

__device__ float WarpReduceSum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor(value, offset);
  }
  return value;
}

/// One block per token row.
///
/// The softmax runs over the whole expert vocabulary in shared memory, then
/// the first warp picks the top-k one slot at a time: every lane holds a
/// strided slice of the probabilities, so a lane-local scan plus a butterfly
/// reduction finds the row's best, and the winner is masked out before the
/// next slot.
__global__ void MoeRouterTopKKernel(const float* __restrict__ router_logits,
                                    const float* __restrict__ shared_logit,
                                    std::int32_t* __restrict__ expert_ids,
                                    float* __restrict__ expert_weights,
                                    float* __restrict__ shared_weight,
                                    std::uint32_t num_experts,
                                    std::uint32_t num_used) {
  // Dynamic shared memory: the probabilities, then the picked values and
  // indices.
  extern __shared__ float shared[];
  // The block-wide reductions, kept out of the dynamic block so they do
  // not overlap the probabilities.
  __shared__ float s_reduce[2 * kMoeRouterWarps];
  float* probs = shared;
  float* pick_value = probs + num_experts;
  std::int32_t* pick_index =
      reinterpret_cast<std::int32_t*>(pick_value + num_used);

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31U;
  const std::uint32_t warp = tid >> 5;
  const std::uint32_t warps = blockDim.x >> 5;

  const std::size_t row = blockIdx.x;
  const float* logits = router_logits + (row * num_experts);
  std::int32_t* ids = expert_ids + (row * num_used);
  float* weights = expert_weights + (row * num_used);

  // The peak of the softmax.
  float local_max = -INFINITY;
  for (std::uint32_t e = tid; e < num_experts; e += blockDim.x) {
    local_max = fmaxf(local_max, logits[e]);
  }
  local_max = WarpReduceMax(local_max);
  if (lane == 0) {
    s_reduce[warp] = local_max;
  }
  __syncthreads();
  float peak = s_reduce[0];
  for (std::uint32_t w = 1; w < warps; ++w) {
    peak = fmaxf(peak, s_reduce[w]);
  }

  // The exponentials and their sum.
  float local_sum = 0.0F;
  for (std::uint32_t e = tid; e < num_experts; e += blockDim.x) {
    const float value = __expf(logits[e] - peak);
    probs[e] = value;
    local_sum += value;
  }
  local_sum = WarpReduceSum(local_sum);
  if (lane == 0) {
    s_reduce[kMoeRouterWarps + warp] = local_sum;
  }
  __syncthreads();
  float total = s_reduce[kMoeRouterWarps];
  for (std::uint32_t w = 1; w < warps; ++w) {
    total += s_reduce[kMoeRouterWarps + w];
  }
  for (std::uint32_t e = tid; e < num_experts; e += blockDim.x) {
    probs[e] /= total;
  }
  __syncthreads();

  // The top-k. The first warp picks; the rest wait.
  if (tid < 32) {
    constexpr std::uint32_t kMaxItems = kMoeMaxExperts / 32;
    float values[kMaxItems];
    const std::uint32_t items = (num_experts + 31U) / 32U;
#pragma unroll
    for (std::uint32_t j = 0; j < kMaxItems; ++j) {
      const std::uint32_t e = tid + (j * 32U);
      values[j] = (j < items && e < num_experts) ? probs[e] : -1.0F;
    }
    for (std::uint32_t slot = 0; slot < num_used; ++slot) {
      float best = -1.0F;
      std::uint32_t index = 0xffffffffU;
#pragma unroll
      for (std::uint32_t j = 0; j < kMaxItems; ++j) {
        if (j < items && values[j] > best) {
          best = values[j];
          index = tid + (j * 32U);
        }
      }
      // Ties resolve toward the lower expert index.
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_xor(best, offset);
        const std::uint32_t other_index = __shfl_xor(index, offset);
        if (other > best || (other == best && other_index < index)) {
          best = other;
          index = other_index;
        }
      }
      if (lane == 0) {
        pick_value[slot] = best;
        pick_index[slot] = static_cast<std::int32_t>(index);
      }
      // Mask the picked expert out so the next slot finds a different one.
#pragma unroll
      for (std::uint32_t j = 0; j < kMaxItems; ++j) {
        if (j < items && tid + (j * 32U) == index) {
          values[j] = -1.0F;
        }
      }
      __syncwarp();
    }
    if (lane == 0) {
      float selected_sum = 0.0F;
      for (std::uint32_t slot = 0; slot < num_used; ++slot) {
        selected_sum += pick_value[slot];
      }
      const float inverse = selected_sum > 0.0F ? 1.0F / selected_sum : 0.0F;
      for (std::uint32_t slot = 0; slot < num_used; ++slot) {
        ids[slot] = pick_index[slot];
        weights[slot] = pick_value[slot] * inverse;
      }
    }
  }

  // The shared expert's gate.
  if (tid == 0) {
    shared_weight[row] = 1.0F / (1.0F + __expf(-shared_logit[row]));
  }
}

__global__ void MoeCombineKernel(float* __restrict__ out,
                                 const float* __restrict__ slot_out,
                                 const float* __restrict__ slot_weight,
                                 const float* __restrict__ shared_out,
                                 const float* __restrict__ shared_weight,
                                 std::uint32_t used, std::uint32_t dim) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) {
    return;
  }
  const std::size_t row = blockIdx.y;
  const float* slots = slot_out + (row * used * dim);
  const float* slot_weights = slot_weight + (row * used);
  const float* shared = shared_out + (row * dim);
  float sum = 0.0F;
  for (std::uint32_t s = 0; s < used; ++s) {
    sum += slot_weights[s] * slots[(static_cast<std::size_t>(s) * dim) + i];
  }
  out[(row * dim) + i] = sum + shared_weight[row] * shared[i];
}

}  // namespace

void LaunchMoeRouterTopK(const float* router_logits,
                         const float* shared_logit, std::int32_t* expert_ids,
                         float* expert_weights, float* shared_weight,
                         std::uint32_t num_experts, std::uint32_t num_used,
                         std::uint32_t rows, hipStream_t stream) {
  if (rows == 0 || num_experts == 0 || num_used == 0) {
    return;
  }
  if (num_experts > kMoeMaxExperts) {
    return;
  }
  // The probabilities, the picked values and the picked indices.
  const std::size_t shared_bytes =
      (static_cast<std::size_t>(num_experts) + (2U * num_used)) *
      sizeof(float);
  hipLaunchKernelGGL(MoeRouterTopKKernel, dim3(rows), dim3(kMoeRouterThreads),
                     shared_bytes, stream, router_logits, shared_logit,
                     expert_ids, expert_weights, shared_weight, num_experts,
                     num_used);
}

void LaunchMoeCombine(const float* expert_out, const float* expert_weights,
                      const float* shared_out, const float* shared_weight,
                      float* out, std::uint32_t num_used, std::uint32_t dim,
                      std::uint32_t rows, hipStream_t stream) {
  if (rows == 0 || dim == 0) {
    return;
  }
  constexpr std::uint32_t kThreads = 256;
  const std::uint32_t blocks = (dim + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(MoeCombineKernel, dim3(blocks, rows), dim3(kThreads), 0,
                     stream, out, expert_out, expert_weights, shared_out,
                     shared_weight, num_used, dim);
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
