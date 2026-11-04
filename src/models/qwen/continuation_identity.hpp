#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

#include "src/families/qwen/hip_common/execution_policy.hpp"
#include "src/models/qwen/dflash_policy.hpp"

namespace gufo::speculative {
struct SpeculativeOptions;
}

namespace gufo::models::qwen {

inline constexpr std::uint32_t kQwenComponentPayloadVersion = 2;

inline constexpr std::string_view QwenStateAbi(
    bool speculative, hip::QwenKvCacheStorage kv_storage,
    bool bf16_recurrent_state) noexcept {
  const bool fp16_attention_kv = kv_storage == hip::QwenKvCacheStorage::kFp16;
  const bool q8_attention_kv = kv_storage == hip::QwenKvCacheStorage::kQ8_0;
  if (bf16_recurrent_state) {
    if (speculative) {
      if (q8_attention_kv) {
        return "qwen-gfx1151-dflash-state-v4-q8_0-kv-bf16-recurrent";
      }
      return fp16_attention_kv
                 ? "qwen-gfx1151-dflash-state-v4-fp16-kv-bf16-recurrent"
                 : "qwen-gfx1151-dflash-state-v4-fp32-kv-bf16-recurrent";
    }
    if (q8_attention_kv) {
      return "qwen-gfx1151-state-v3-q8_0-kv-bf16-recurrent";
    }
    return fp16_attention_kv ? "qwen-gfx1151-state-v3-fp16-kv-bf16-recurrent"
                             : "qwen-gfx1151-state-v3-fp32-kv-bf16-recurrent";
  }
  if (speculative) {
    if (q8_attention_kv) {
      return "qwen-gfx1151-dflash-state-v3-q8_0-kv";
    }
    return fp16_attention_kv ? "qwen-gfx1151-dflash-state-v3-fp16-kv"
                             : "qwen-gfx1151-dflash-state-v3-fp32-kv";
  }
  if (q8_attention_kv) {
    return "qwen-gfx1151-state-v2-q8_0-kv";
  }
  return fp16_attention_kv ? "qwen-gfx1151-state-v2-fp16-kv"
                           : "qwen-gfx1151-state-v2-fp32-kv";
}

bool IsSha256Hex(std::string_view value) noexcept;

std::vector<std::uint8_t> QwenCompatibilityIdentity(
    std::string_view artifact_fingerprint,
    std::string_view draft_artifact_fingerprint, std::uint32_t max_context,
    const hip::QwenExecutionPolicy& execution_policy, bool speculative,
    const speculative::SpeculativeOptions& speculative_options,
    speculative::DFlashDraftPolicy draft_policy);

std::vector<std::uint8_t> QwenProcessIdentity(const void* model,
                                              const void* draft,
                                              std::uint32_t max_context);

}  // namespace gufo::models::qwen

namespace gufo::server {
using models::qwen::kQwenComponentPayloadVersion;
using models::qwen::QwenCompatibilityIdentity;
using models::qwen::QwenProcessIdentity;
using models::qwen::QwenStateAbi;
}  // namespace gufo::server
