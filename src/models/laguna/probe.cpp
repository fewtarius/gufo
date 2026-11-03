#include "src/models/laguna/probe.hpp"

#include <filesystem>
#include <gufo/ext/gguf_reader.hpp>
#include <string>

namespace gufo::models::laguna {

bool ProbeLagunaGguf(const core::GgufReader& reader) {
  const auto architecture =
      reader.GetMetadataString("general.architecture").value_or("");
  return architecture == "laguna";
}

bool ProbeLaguna(std::string_view model_root) {
  try {
    const std::filesystem::path path(model_root);
    if (!std::filesystem::exists(path)) {
      return false;
    }
    if (std::filesystem::is_regular_file(path)) {
      std::string error;
      auto reader = core::GgufReader::OpenFile(path.string(), &error);
      if (!reader) {
        return false;
      }
      return ProbeLagunaGguf(*reader);
    }
    return false;
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::laguna
