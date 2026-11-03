#ifndef GUFO_MODELS_LAGUNA_DRIVERS_HPP_
#define GUFO_MODELS_LAGUNA_DRIVERS_HPP_

#include <chrono>
#include <memory>

#include "gufo/ext/text.hpp"

namespace gufo::models::laguna {

#if defined(ENGINE_ENABLE_HIP) && ENGINE_ENABLE_HIP
/// Benchmarks the Laguna S 2.1 ROCm engine: prompt processing and token
/// generation throughput, plus the optional prefill validation pass.
///
/// Returns 0 on success and a non-zero code when the workload cannot run.
int RunLagunaBenchmark(const ext::TextBenchOptions& opt,
                       const std::shared_ptr<const core::GgufReader>& reader,
                       std::chrono::steady_clock::time_point model_load_start);

/// Runs one Laguna prompt: renders the chat template (or the raw text),
/// prefills and decodes through the DFlash-2 drafter when one is loaded.
int RunLagunaPrompt(const ext::TextPromptOptions& opt,
                    const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point model_load_start);

/// Runs an interactive Laguna chat session over the same engine.
int RunLagunaChat(const ext::TextPromptOptions& opt,
                  const core::GgufReader& reader,
                  std::chrono::steady_clock::time_point model_load_start);
#endif

}  // namespace gufo::models::laguna

#endif  // GUFO_MODELS_LAGUNA_DRIVERS_HPP_
