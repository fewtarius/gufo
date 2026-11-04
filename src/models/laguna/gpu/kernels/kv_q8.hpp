// Device-side q8_0 KV storage for the Laguna attention path.
//
// The layout matches the host codec's BlockQ8_0 exactly: one binary16 scale
// followed by 32 signed codes, 34 bytes per block, no inter-block padding.
// The device primitives live in src/core/kv_codec/kv_q8_device.hpp so the
// numeric contract has one home; this header keeps the Laguna spelling.
//
// See docs/kv-codec.md.
#ifndef GUFO_MODELS_LAGUNA_GPU_KERNELS_KV_Q8_HPP_
#define GUFO_MODELS_LAGUNA_GPU_KERNELS_KV_Q8_HPP_

#include "src/core/kv_codec/kv_q8_device.hpp"

namespace gufo::models::laguna::gpu::rocm::detail {

using Q8KvBlock = gufo::core::kv_codec::device::Q8KvBlock;
using gufo::core::kv_codec::device::DequantizeQ8Group;
using gufo::core::kv_codec::device::Q8KvClamp;
using gufo::core::kv_codec::device::Q8KvCode;
using gufo::core::kv_codec::device::Q8KvRowBytes;
using gufo::core::kv_codec::device::Q8KvScale;

#if defined(__HIPCC__)
using gufo::core::kv_codec::device::WarpMaxAbs;
#endif  // defined(__HIPCC__)

}  // namespace gufo::models::laguna::gpu::rocm::detail

#endif  // GUFO_MODELS_LAGUNA_GPU_KERNELS_KV_Q8_HPP_
