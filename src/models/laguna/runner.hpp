#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "gufo/ext/text.hpp"

namespace gufo::models::laguna {

#if defined(ENGINE_ENABLE_HIP)
/// Creates the Laguna text runner from a loaded GGUF.
///
/// Returns nullptr and fills `error` when the model or tokenizer cannot be
/// loaded, or when the device did not settle.
[[nodiscard]] std::shared_ptr<ext::TextModelRunner> CreateLagunaTextRunner(
    const ext::TextRunnerOptions& options, std::string* error);
#endif

}  // namespace gufo::models::laguna
