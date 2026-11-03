#pragma once

#include <gufo/ext/gguf_reader.hpp>
#include <string_view>

namespace gufo::models::laguna {

/// True when the GGUF declares the Laguna architecture.
[[nodiscard]] bool ProbeLagunaGguf(const core::GgufReader& reader);

/// True when the path names a GGUF that declares the Laguna architecture.
[[nodiscard]] bool ProbeLaguna(std::string_view model_root);

}  // namespace gufo::models::laguna
