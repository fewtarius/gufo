#include <gufo/ext/generated/laguna_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "src/models/laguna/drivers.hpp"
#include "src/models/laguna/probe.hpp"
#include "src/models/laguna/runner.hpp"

namespace gufo::models::laguna {}  // namespace gufo::models::laguna

void laguna_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_laguna::kProvider_laguna);
#if defined(ENGINE_ENABLE_HIP)
  registry.AddTextProvider({
      .id = "laguna",
      .probe = &gufo::models::laguna::ProbeLaguna,
      .probe_gguf = &gufo::models::laguna::ProbeLagunaGguf,
      .create = &gufo::models::laguna::CreateLagunaTextRunner,
      .bench_driver = &gufo::models::laguna::RunLagunaBenchmark,
      .prompt_driver = &gufo::models::laguna::RunLagunaPrompt,
      .chat_driver = &gufo::models::laguna::RunLagunaChat,
  });
#endif
}
