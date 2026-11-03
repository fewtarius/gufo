// Laguna S 2.1 CLI drivers.
//
// The prompt, chat and bench commands reach a model through the provider's
// driver slots, so this file owns Laguna's command-line behaviour. The
// behaviour itself comes from the Laguna prompt path this repository used to
// carry inside src/cli/prompt/prompt.cpp: the model loads its own tokenizer,
// the chat template renders the turns, the engine prefills the prompt and the
// DFlash-2 drafter verifies its blocks during decode.
//
// The engine lives in src/models/laguna/gpu/. The chat template and the
// native tool-call dialect live in this directory; the tokenizer is the
// shared Qwen BPE tokenizer the Laguna GGUF carries.

#include "src/models/laguna/drivers.hpp"

// The engine, the DFlash drafter and the HIP runtime are the ROCm tier's
// only environment, so the drivers exist there. The value test keeps the
// file compilable on its own outside that build, where the build defines
// ENGINE_ENABLE_HIP=1 rather than leaving it undefined.
#if defined(ENGINE_ENABLE_HIP) && ENGINE_ENABLE_HIP

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gufo/ext/crypto.hpp"
#include "gufo/ext/text.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/models/laguna/chat_template.hpp"
#include "src/models/laguna/gpu/engine.hpp"
#include "src/models/qwen/hip/dflash.hpp"
#include "src/testing/compare/logit_comparator.hpp"

namespace gufo::models::laguna {
namespace {

/// The CLI works inside one fixed context window, like the other models'
/// drivers.
constexpr std::uint32_t kDefaultContext = 4096;

struct BenchStats {
  double mean = 0.0;
  double stddev = 0.0;
};

BenchStats ComputeStats(const std::vector<double>& values) {
  if (values.empty())
    return {};
  if (values.size() == 1)
    return {.mean = values.front(), .stddev = 0.0};
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double mean = sum / static_cast<double>(values.size());
  double sq_sum = 0.0;
  for (const double val : values) {
    sq_sum += (val - mean) * (val - mean);
  }
  const double variance = sq_sum / static_cast<double>(values.size() - 1);
  return {.mean = mean, .stddev = std::sqrt(variance)};
}

std::string MakeTestName(std::string_view prefix, std::size_t count,
                         std::size_t depth) {
  std::string result = std::string(prefix) + std::to_string(count);
  if (depth > 0) {
    result += " @ d" + std::to_string(depth);
  }
  return result;
}

void PrintModelLoadTime(std::chrono::steady_clock::time_point start,
                        bool success = true) {
  if (start == std::chrono::steady_clock::time_point{})
    return;
  const auto end = std::chrono::steady_clock::now();
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cerr << "Model loaded in " << ms << " ms ("
            << (success ? "success" : "failed") << ").\n";
}

void PrintTokenTrace(std::span<const tokenization::TokenId> tokens) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(tokens.size() * sizeof(std::uint32_t));
  for (const auto token : tokens) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
      bytes.push_back(static_cast<std::uint8_t>(token >> shift));
    }
  }
  std::cerr << "[TokenTrace]: count=" << tokens.size()
            << " sha256=" << crypto::Sha256Hex(bytes) << '\n';
}

std::optional<gufo::ReasoningEffort> ParseReasoningEffort(
    std::string_view value) {
  if (value == "minimal")
    return gufo::ReasoningEffort::kMinimal;
  if (value == "low")
    return gufo::ReasoningEffort::kLow;
  if (value == "medium")
    return gufo::ReasoningEffort::kMedium;
  if (value == "high")
    return gufo::ReasoningEffort::kHigh;
  if (value == "xhigh")
    return gufo::ReasoningEffort::kXHigh;
  if (value == "max")
    return gufo::ReasoningEffort::kMax;
  return std::nullopt;
}

gufo::ReasoningOptions PromptReasoningOptions(
    const ext::TextPromptOptions& options) {
  gufo::ReasoningOptions reasoning;
  if (options.reasoning_mode == "on") {
    reasoning.enabled = true;
  } else if (options.reasoning_mode == "off") {
    reasoning.enabled = false;
  }
  if (options.reasoning_effort != "auto") {
    reasoning.effort = ParseReasoningEffort(options.reasoning_effort);
    reasoning.enabled = true;
  }
  if (options.preserve_thinking == "on") {
    reasoning.preserve_thinking = true;
  } else if (options.preserve_thinking == "off") {
    reasoning.preserve_thinking = false;
  }
  return reasoning;
}

/// The tokens that end a turn. The GGUF carries the EOS, EOT and EOM ids
/// separately; a model may also ship a list of end-of-generation ids.
std::vector<tokenization::TokenId> LagunaStopTokens(
    const core::GgufReader& reader) {
  std::vector<tokenization::TokenId> stop_tokens;
  if (const auto eos =
          reader.GetMetadataUint32("tokenizer.ggml.eos_token_id")) {
    stop_tokens.push_back(*eos);
  }
  for (const char* key :
       {"tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"}) {
    if (const auto id = reader.GetMetadataUint32(key)) {
      stop_tokens.push_back(*id);
    }
  }
  if (const auto* meta = reader.FindMetadata("tokenizer.ggml.eog_token_ids")) {
    if (const auto* ids =
            std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (const auto id : *ids) {
        stop_tokens.push_back(static_cast<tokenization::TokenId>(id));
      }
    } else if (const auto* ids =
                   std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      for (const auto id : *ids) {
        stop_tokens.push_back(static_cast<tokenization::TokenId>(id));
      }
    }
  }
  return stop_tokens;
}

/// The model and the tokenizer a Laguna command needs. Both are loaded once
/// and shared by the prompt and chat paths.
struct LagunaSessionBundle {
  std::shared_ptr<gpu::Model> model;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer;
  std::vector<tokenization::TokenId> stop_tokens;
};

/// Loads the Laguna model and its tokenizer.
///
/// Returns false after printing the reason; the caller turns that into an
/// exit code.
bool LoadLaguna(const ext::TextPromptOptions& opt,
                const core::GgufReader& reader,
                std::chrono::steady_clock::time_point load_start,
                LagunaSessionBundle* out) {
  if (opt.force_cpu) {
    std::cerr << "Laguna S 2.1 is supported only by the ROCm backend\n";
    PrintModelLoadTime(load_start, false);
    return false;
  }
  if (!opt.image_paths.empty() || !opt.vision_model_path.empty()) {
    std::cerr << "Laguna does not support image input\n";
    PrintModelLoadTime(load_start, false);
    return false;
  }
  const bool dflash = opt.speculative_backend == "dflash2";
  if (!opt.speculative_backend.empty() && !dflash) {
    std::cerr << "Laguna supports only --speculative dflash2\n";
    PrintModelLoadTime(load_start, false);
    return false;
  }
  if (dflash && (opt.dflash_model_path.empty() || opt.draft_tokens < 1 ||
                 opt.draft_tokens > 15)) {
    std::cerr << "Laguna DFlash2 requires --dflash-model and 1..15 draft "
                 "tokens\n";
    PrintModelLoadTime(load_start, false);
    return false;
  }
  if (dflash && opt.min_draft_tokens != 1) {
    std::cerr << "Laguna DFlash2 requires --min-draft-tokens 1\n";
    PrintModelLoadTime(load_start, false);
    return false;
  }

  std::string error;
  auto tokenizer = tokenization::QwenTokenizer::CreateFromGguf(reader, &error);
  if (!tokenizer) {
    std::cerr << "Laguna tokenizer failed: " << error << '\n';
    PrintModelLoadTime(load_start, false);
    return false;
  }

  gpu::EngineOptions engine_options;
  engine_options.max_context = kDefaultContext;
  if (dflash) {
    engine_options.dflash_model_path = opt.dflash_model_path;
  }
  auto model = gpu::Model::Load(opt.model_path, engine_options, &error);
  if (!model) {
    std::cerr << "Laguna GPU load failed: " << error << '\n';
    PrintModelLoadTime(load_start, false);
    return false;
  }
  PrintModelLoadTime(load_start);

  out->model = std::move(model);
  out->tokenizer = std::move(tokenizer);
  out->stop_tokens = LagunaStopTokens(reader);
  return true;
}

/// Renders the chat template for one turn, or passes the text through when
/// the caller asked for raw input.
std::string RenderLagunaPrompt(const ext::TextPromptOptions& opt,
                               const std::string& system_prompt,
                               const std::string& user_text,
                               const gufo::ReasoningOptions& reasoning) {
  if (!opt.use_chat_template)
    return user_text;
  ChatTemplateOptions chat_options;
  chat_options.enable_thinking = reasoning.enabled.value_or(false);
  return RenderPrompt(system_prompt, user_text, chat_options);
}

std::vector<tokenization::TokenId> EncodeLagunaPrompt(
    const tokenization::QwenTokenizer& tokenizer, const std::string& rendered) {
  tokenization::TokenizerOptions encode_options;
  encode_options.parse_special_tokens = true;
  return tokenizer.Encode(rendered, encode_options);
}

/// The generation loop.
///
/// The drafter proposes a block of tokens from the last accepted token; the
/// engine verifies the whole block in one pass and keeps the prefix the
/// target distribution accepts. A greedy target accepts a draft when it is
/// the argmax; a sampled target accepts it with probability
/// P_target(draft) and resamples from the residual when it misses. Both rules
/// leave the published token distributed as the target distribution.
int GenerateLagunaResponse(const ext::TextPromptOptions& opt,
                           const LagunaSessionBundle& bundle,
                           gpu::Session& session,
                           std::span<const tokenization::TokenId> prompt,
                           std::string* reply = nullptr) {
  const auto& tokenizer = *bundle.tokenizer;
  const auto is_stop = [&](tokenization::TokenId token) {
    return std::find(bundle.stop_tokens.begin(), bundle.stop_tokens.end(),
                     token) != bundle.stop_tokens.end();
  };
  const auto emit = [&](tokenization::TokenId token) {
    const auto piece = tokenizer.DecodeToken(token);
    if (reply != nullptr)
      reply->append(piece);
    std::cout << piece << std::flush;
  };

  std::string error;
  const std::vector<std::int32_t> input(prompt.begin(), prompt.end());
  const auto start = std::chrono::steady_clock::now();
  if (!session.Append(input, &error)) {
    std::cerr << "Laguna GPU prefill failed: " << error << '\n';
    return 1;
  }
  const auto prefill_end = std::chrono::steady_clock::now();

  sampling::SamplerState sampler(opt.sampling, prompt);
  const bool sampled = opt.sampling.uses_random_sampling();
  auto* drafter = session.DflashExecutor();
  const bool use_dflash = bundle.model->HasDflash() && drafter != nullptr;
  const std::uint32_t max_draft =
      use_dflash ? std::min<std::uint32_t>(opt.draft_tokens,
                                           bundle.model->DrafterBlockSize() - 1)
                 : 0;

  std::size_t generated = 0;
  std::size_t drafted = 0;
  std::size_t accepted_drafts = 0;
  std::vector<tokenization::TokenId> generated_ids;

  while (generated < opt.max_tokens) {
    const auto anchor_logits = session.Logits();
    if (anchor_logits.empty()) {
      std::cerr << "\nLaguna GPU token selection failed\n";
      return 1;
    }
    const auto anchor =
        static_cast<tokenization::TokenId>(sampler.Sample(anchor_logits));
    if (is_stop(anchor))
      break;
    sampler.Accept(anchor);
    emit(anchor);
    if (opt.verbose)
      generated_ids.push_back(anchor);
    ++generated;
    if (generated >= opt.max_tokens)
      break;

    // The drafter proposes a block anchored on the token just published.
    // It may only draft while its injected context covers the session
    // position; a cold start after the prefill re-syncs it during the plain
    // decode step.
    const std::uint32_t base = session.Position();
    std::vector<std::int32_t> feed{static_cast<std::int32_t>(anchor)};
    if (use_dflash && max_draft > 0 &&
        drafter->GetInjectedContextLength() == session.Position()) {
      // A drafter that samples needs one random value per proposal.
      std::vector<float> draft_uniforms;
      if (sampled) {
        auto draft_rng = sampling::NextRandom(sampler.mutable_rng_state());
        draft_uniforms.reserve(max_draft);
        for (std::uint32_t row = 0; row < max_draft; ++row) {
          draft_uniforms.push_back(
              static_cast<float>(sampling::Uniform(&draft_rng)));
        }
      }
      const std::uint32_t wanted = std::min<std::uint32_t>(
          max_draft, static_cast<std::uint32_t>(opt.max_tokens - generated));
      const auto drafts = drafter->ForwardBlock(
          anchor, base, wanted, sampled ? opt.sampling.temperature : 0.0F,
          draft_uniforms);
      feed.insert(feed.end(), drafts.begin(), drafts.end());
    }

    if (feed.size() == 1) {
      if (!session.Append(feed, &error)) {
        std::cerr << "\nLaguna GPU decode failed: " << error << '\n';
        return 1;
      }
      continue;
    }

    drafted += feed.size() - 1;
    if (!session.AppendAll(feed, &error)) {
      std::cerr << "\nLaguna GPU verify failed: " << error << '\n';
      return 1;
    }
    // Row i holds the distribution after feed[i]; draft feed[i + 1] is
    // tested against it, and only the accepted rows are committed to the
    // drafter.
    std::uint32_t kept = 0;
    bool stopped = false;
    bool have_pending = false;
    tokenization::TokenId pending = 0;
    for (std::uint32_t row = 0;; ++row) {
      const auto logits = session.Logits(row);
      if (row + 1 >= feed.size())
        break;
      const auto guess = static_cast<tokenization::TokenId>(feed[row + 1]);
      tokenization::TokenId next = 0;
      bool hit = false;
      if (sampled) {
        const double p = sampler.Distribution(logits).probability(guess);
        hit = sampler.Uniform() < p;
        if (hit) {
          next = guess;
        } else {
          const float one = 1.0F;
          next = sampler.SampleResidual(logits, {&guess, 1}, {&one, 1});
        }
      } else {
        next = static_cast<tokenization::TokenId>(sampler.Sample(logits));
        hit = next == guess;
      }
      if (!hit) {
        // The rejected draft's replacement is a valid sample of the target
        // distribution at the last accepted token. It is published after
        // the accepted prefix has been committed.
        pending = next;
        have_pending = true;
        break;
      }
      if (is_stop(guess)) {
        // A stop token ends the turn and is never published.
        stopped = true;
        break;
      }
      ++kept;
      sampler.Accept(guess);
      emit(guess);
      if (opt.verbose)
        generated_ids.push_back(guess);
      ++accepted_drafts;
      ++generated;
      if (generated >= opt.max_tokens)
        break;
    }

    // AppendAll advanced the KV cache and the position by the whole verified
    // batch (the anchor plus every drafted token); only the accepted prefix
    // sticks. Discard the rest, or the full-attention layers keep the
    // rejected drafts' key/value state at positions nothing overwrites.
    session.Rewind(base + 1 + kept);
    if (!session.DflashInject(kept + 1, &error)) {
      std::cerr << "\nLaguna drafter commit failed: " << error << '\n';
      return 1;
    }
    if (have_pending && generated < opt.max_tokens) {
      const std::int32_t pending_token = static_cast<std::int32_t>(pending);
      if (!session.Append(std::span(&pending_token, 1), &error)) {
        std::cerr << "\nLaguna GPU decode failed: " << error << '\n';
        return 1;
      }
      if (is_stop(pending)) {
        // The stop token reached the KV cache before it could be known for
        // one; roll it back so the cache matches the published tokens.
        session.Rewind(session.Position() - 1);
        stopped = true;
      } else {
        sampler.Accept(pending);
        emit(pending);
        if (opt.verbose)
          generated_ids.push_back(pending);
        ++generated;
      }
    }
    if (stopped)
      break;
  }
  std::cout << '\n';

  if (opt.verbose) {
    const auto seconds = [](auto a, auto b) {
      return std::chrono::duration<double>(b - a).count();
    };
    const auto end = std::chrono::steady_clock::now();
    const double prefill_seconds = seconds(start, prefill_end);
    const double decode_seconds = seconds(prefill_end, end);
    std::cerr << "[Laguna GPU] prompt=" << prompt.size() << " tok in "
              << prefill_seconds << " s ("
              << prompt.size() / std::max(prefill_seconds, 1e-9)
              << " tok/s), generated=" << generated << " tok in "
              << decode_seconds << " s ("
              << generated / std::max(decode_seconds, 1e-9) << " tok/s)";
    if (drafted != 0) {
      std::cerr << ", drafted=" << drafted << " accepted=" << accepted_drafts
                << '/' << drafted;
    }
    std::cerr << '\n';
    PrintTokenTrace(generated_ids);
  }
  return 0;
}

/// Compares the logits of one multi-token append against the same tokens
/// appended one at a time. A mismatch means a token-count-dependent
/// attention bug rather than noise.
bool ValidateLagunaPrefill(gpu::Model& model,
                           std::span<const std::int32_t> tokens,
                           std::string* error_msg) {
  auto batched = model.CreateSession(
      static_cast<std::uint32_t>(tokens.size() + 1), error_msg);
  if (!batched || !batched->Append(tokens, error_msg))
    return false;
  const std::vector<float> batched_logits(batched->Logits().begin(),
                                          batched->Logits().end());

  auto sequential = model.CreateSession(
      static_cast<std::uint32_t>(tokens.size() + 1), error_msg);
  if (!sequential)
    return false;
  for (const std::int32_t token : tokens) {
    if (!sequential->Append(std::span(&token, 1), error_msg))
      return false;
  }
  const auto sequential_logits = sequential->Logits();

  const auto comparison = testing::CompareLogits(
      std::vector<float>(sequential_logits.begin(), sequential_logits.end()),
      batched_logits);
  std::cout << "[Laguna Prefill Validation] tokens=" << tokens.size()
            << " top1_match=" << (comparison.top1_match ? "yes" : "no")
            << " finite=" << (comparison.finite ? "yes" : "no") << '\n'
            << "  max_abs_diff=" << comparison.max_abs_diff
            << " mean_abs_diff=" << comparison.mean_abs_diff
            << " rmse=" << comparison.root_mean_square_error
            << " cosine_similarity=" << comparison.cosine_similarity << '\n';
  if (!comparison.top1_match) {
    const float ref_top1 = sequential_logits[comparison.reference_argmax];
    const float cand_top1 = batched_logits[comparison.candidate_argmax];
    const float ref_at_cand = sequential_logits[comparison.candidate_argmax];
    const float cand_at_ref = batched_logits[comparison.reference_argmax];
    std::cout << "  reference_argmax=" << comparison.reference_argmax
              << " (logit=" << ref_top1 << ")"
              << " candidate_argmax=" << comparison.candidate_argmax
              << " (logit=" << cand_top1 << ")\n"
              << "  reference's logit at candidate's token=" << ref_at_cand
              << " (margin=" << (ref_top1 - ref_at_cand) << ")"
              << " candidate's logit at reference's token=" << cand_at_ref
              << " (margin=" << (cand_top1 - cand_at_ref) << ")\n";
  }
  return comparison.finite && comparison.top1_match;
}

int RunLagunaBenchmarkImpl(
    const ext::TextBenchOptions& options,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start) {
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
    std::cerr << "Error: no HIP GPU is available for Laguna S 2.1\n";
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  if (options.concurrency != std::vector<std::size_t>{1}) {
    std::cerr << "Error: Laguna bench supports C1 only\n";
    return 1;
  }
  const bool dflash = options.speculative_backend == "dflash2";
  if (!options.speculative_backend.empty() && !dflash) {
    std::cerr << "Error: Laguna supports only --speculative dflash2 or off\n";
    return 1;
  }
  if (dflash && options.dflash_model_path.empty()) {
    std::cerr << "Error: --speculative dflash2 requires --dflash-model\n";
    return 1;
  }
  if (dflash && options.min_draft_tokens != 1) {
    std::cerr << "Error: Laguna DFlash2 requires --min-draft-tokens 1\n";
    return 1;
  }

  const auto max_or_zero = [](const std::vector<std::size_t>& values) {
    return values.empty() ? std::size_t{0}
                          : *std::max_element(values.begin(), values.end());
  };
  const std::size_t max_depth = max_or_zero(options.n_depths);
  const std::size_t max_prompt = max_or_zero(options.n_prompts);
  const std::size_t max_generation = max_or_zero(options.n_gens);
  constexpr std::size_t context_limit = std::numeric_limits<int>::max() - 1;
  if (max_depth > context_limit || max_prompt > context_limit - max_depth ||
      max_generation > context_limit - std::max<std::size_t>(max_depth, 16) ||
      options.validate_prefill_tokens > context_limit) {
    std::cerr << "Error: requested Laguna benchmark context is too large\n";
    return 1;
  }
  const std::size_t required_context =
      std::max({std::size_t{4096}, max_depth + max_prompt + 1,
                std::max<std::size_t>(max_depth, 16) + max_generation + 1,
                options.validate_prefill_tokens + 1});

  std::string error;
  auto tokenizer = tokenization::QwenTokenizer::CreateFromGguf(*reader, &error);
  if (!tokenizer) {
    std::cerr << "Laguna tokenizer failed: " << error << '\n';
    return 1;
  }
  gpu::EngineOptions engine_options;
  engine_options.max_context = static_cast<std::uint32_t>(required_context);
  if (dflash) {
    engine_options.dflash_model_path = options.dflash_model_path;
  }
  auto model = gpu::Model::Load(options.model_path, engine_options, &error);
  if (!model) {
    std::cerr << "Error creating Laguna model: " << error << '\n';
    PrintModelLoadTime(model_load_start, false);
    return 1;
  }
  PrintModelLoadTime(model_load_start);

  // A fixed token pattern keeps the timing reproducible. It uses far fewer
  // distinct rows than varied requests.
  std::vector<std::int32_t> tokens;
  {
    const auto pattern = tokenizer->Encode(
        "The quick brown fox jumps over the lazy dog. "
        "Strix Halo executes this deterministic benchmark sequence. ",
        tokenization::TokenizerOptions{});
    if (pattern.empty()) {
      std::cerr << "Error: benchmark token pattern is empty\n";
      return 1;
    }
    tokens.resize(required_context);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      tokens[i] = static_cast<std::int32_t>(pattern[i % pattern.size()]);
    }
  }

  if (options.validate_prefill_tokens != 0) {
    const std::size_t width =
        std::min(options.validate_prefill_tokens, tokens.size());
    if (!ValidateLagunaPrefill(*model, std::span(tokens).first(width),
                               &error)) {
      std::cerr << "Laguna prefill validation FAILED: " << error << '\n';
      return 1;
    }
  }

  const double size_gib =
      static_cast<double>(reader->GetSize()) / (1024.0 * 1024.0 * 1024.0);
  std::cout << "| " << std::left << std::setw(32) << "model"
            << " | " << std::right << std::setw(10) << "size"
            << " | " << std::left << std::setw(10) << "backend"
            << " | " << std::right << std::setw(18) << "test"
            << " | " << std::right << std::setw(21) << "t/s"
            << " |\n"
            << "| " << std::string(32, '-') << " | " << std::string(10, '-')
            << " | " << std::string(10, '-') << " | " << std::string(18, '-')
            << " | " << std::string(21, '-') << " |\n";
  std::ostringstream size_text;
  size_text << std::fixed << std::setprecision(2) << size_gib << " GiB";
  const auto print_result = [&](std::string_view test_name,
                                const BenchStats& stats) {
    std::ostringstream throughput;
    throughput << std::fixed << std::setprecision(2) << stats.mean << " ± "
               << stats.stddev;
    std::cout << "| " << std::left << std::setw(32) << "Laguna" << " | "
              << std::right << std::setw(10) << size_text.str() << " | "
              << std::left << std::setw(10) << "ROCm (HIP)"
              << " | " << std::right << std::setw(18) << test_name << " | "
              << std::right << std::setw(21) << throughput.str() << " |\n"
              << std::flush;
  };

  for (const std::size_t depth : options.n_depths) {
    for (const std::size_t prompt_length : options.n_prompts) {
      if (depth + prompt_length >= required_context) {
        std::cerr << "Error: prompt benchmark exceeds context\n";
        return 1;
      }
      std::vector<double> runs;
      // The first repetition warms the kernels; only the measured ones count.
      for (std::size_t repetition = 0; repetition <= options.repetitions;
           ++repetition) {
        auto session = model->CreateSession(
            static_cast<std::uint32_t>(required_context), &error);
        if (!session ||
            (depth > 0 &&
             !session->Append(std::span(tokens).first(depth), &error))) {
          std::cerr << "Error preparing depth: " << error << '\n';
          return 1;
        }
        const auto start = std::chrono::steady_clock::now();
        if (!session->Append(std::span(tokens).subspan(depth, prompt_length),
                             &error)) {
          std::cerr << "Error running prefill: " << error << '\n';
          return 1;
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        if (repetition > 0)
          runs.push_back(static_cast<double>(prompt_length) / seconds);
      }
      print_result(MakeTestName("pp", prompt_length, depth),
                   ComputeStats(runs));
    }

    for (const std::size_t generation_length : options.n_gens) {
      const std::size_t prefix_length = depth > 0 ? depth : 16;
      if (prefix_length + generation_length >= required_context) {
        std::cerr << "Error: generation benchmark exceeds context\n";
        return 1;
      }
      std::vector<double> runs;
      for (std::size_t repetition = 0; repetition < options.repetitions;
           ++repetition) {
        auto session = model->CreateSession(
            static_cast<std::uint32_t>(required_context), &error);
        if (!session ||
            !session->Append(std::span(tokens).first(prefix_length), &error)) {
          std::cerr << "Error preparing generation: " << error << '\n';
          return 1;
        }
        // Two untimed steps warm the decode path before the measurement.
        std::int32_t next = tokens[prefix_length % tokens.size()];
        for (int warm = 0; warm < 2; ++warm) {
          const std::int32_t token = next;
          if (!session->Append(std::span(&token, 1), &error)) {
            std::cerr << "Error running Laguna decode: " << error << '\n';
            return 1;
          }
          next = tokens[(prefix_length + warm + 1) % tokens.size()];
        }
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < generation_length; ++i) {
          const std::int32_t token = next;
          if (!session->Append(std::span(&token, 1), &error)) {
            std::cerr << "Error running Laguna decode: " << error << '\n';
            return 1;
          }
          next = tokens[(prefix_length + i + 3) % tokens.size()];
        }
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        runs.push_back(static_cast<double>(generation_length) / seconds);
      }
      print_result(MakeTestName("tg", generation_length, depth),
                   ComputeStats(runs));
    }
  }
  std::cout << '\n';
  return 0;
}

int RunLagunaPromptImpl(
    const ext::TextPromptOptions& opt, const core::GgufReader& reader,
    std::chrono::steady_clock::time_point model_load_start) {
  LagunaSessionBundle bundle;
  if (!LoadLaguna(opt, reader, model_load_start, &bundle))
    return 1;

  const auto reasoning = PromptReasoningOptions(opt);
  const std::string rendered =
      RenderLagunaPrompt(opt, opt.system_prompt, opt.prompt_text, reasoning);
  const auto prompt = EncodeLagunaPrompt(*bundle.tokenizer, rendered);
  if (prompt.empty()) {
    std::cerr << "Laguna prompt produced no tokens\n";
    return 1;
  }
  // The session holds the prompt, the published tokens and the rows one
  // verification block can add.
  const std::uint32_t needed = static_cast<std::uint32_t>(
      prompt.size() + opt.max_tokens + 1 + gpu::Session::kMaxVerifyRows);
  if (needed > kDefaultContext) {
    std::cerr << "Laguna prompt and output exceed the 4096-token CLI "
                 "context\n";
    return 1;
  }

  std::string error;
  const auto session = bundle.model->CreateSession(needed, &error);
  if (!session) {
    std::cerr << "Laguna GPU session creation failed: " << error << '\n';
    return 1;
  }
  return GenerateLagunaResponse(opt, bundle, *session, prompt);
}

int RunLagunaChatImpl(const ext::TextPromptOptions& opt,
                      const core::GgufReader& reader,
                      std::chrono::steady_clock::time_point model_load_start) {
  if (!opt.use_chat_template) {
    std::cerr << "Laguna interactive chat requires chat framing; use prompt "
                 "--raw for raw text\n";
    return 1;
  }
  LagunaSessionBundle bundle;
  if (!LoadLaguna(opt, reader, model_load_start, &bundle))
    return 1;

  const auto reasoning = PromptReasoningOptions(opt);
  std::cout << "=== Gufo Interactive Chat (Laguna S 2.1) ===\n"
            << "Type 'exit' or Ctrl+D to quit.\n\n";

  std::vector<ChatMessage> history;
  if (!opt.system_prompt.empty()) {
    history.push_back({.role = "system",
                       .content = opt.system_prompt,
                       .reasoning_content = {},
                       .tool_calls = {}});
  }

  for (std::string input;;) {
    std::cout << ">>> User: " << std::flush;
    if (!std::getline(std::cin, input) || input == "exit" || input == "quit")
      break;
    if (input.empty())
      continue;

    history.push_back({.role = "user",
                       .content = input,
                       .reasoning_content = {},
                       .tool_calls = {}});
    ChatTemplateOptions chat_options;
    chat_options.enable_thinking = reasoning.enabled.value_or(false);
    const std::string rendered = RenderPrompt(history, chat_options);
    const auto prompt = EncodeLagunaPrompt(*bundle.tokenizer, rendered);
    if (prompt.empty()) {
      std::cerr << "Laguna prompt produced no tokens\n";
      return 1;
    }
    if (prompt.size() + opt.max_tokens + 1 + gpu::Session::kMaxVerifyRows >
        kDefaultContext) {
      std::cerr << "Laguna prompt and output exceed the 4096-token CLI "
                   "context\n";
      return 1;
    }

    // Every turn is a fresh request over the whole history, so it gets a
    // session of its own.
    std::string error;
    const auto session = bundle.model->CreateSession(
        static_cast<std::uint32_t>(prompt.size() + opt.max_tokens + 1 +
                                   gpu::Session::kMaxVerifyRows),
        &error);
    if (!session) {
      std::cerr << "Laguna GPU session creation failed: " << error << '\n';
      return 1;
    }

    std::cout << "<<< Assistant: ";
    std::string assistant_reply;
    if (GenerateLagunaResponse(opt, bundle, *session, prompt,
                               &assistant_reply) != 0) {
      return 1;
    }

    std::cout << '\n';
    ChatMessage reply{.role = "assistant",
                      .content = {},
                      .reasoning_content = {},
                      .tool_calls = {}};
    if (chat_options.enable_thinking) {
      constexpr std::string_view end = "</think>";
      const auto boundary = assistant_reply.find(end);
      reply.reasoning_content = assistant_reply.substr(0, boundary);
      if (boundary != std::string::npos) {
        reply.content = assistant_reply.substr(boundary + end.size());
      }
    } else {
      reply.content = std::move(assistant_reply);
    }
    history.push_back(std::move(reply));
  }
  return 0;
}

}  // namespace

int RunLagunaBenchmark(const ext::TextBenchOptions& opt,
                       const std::shared_ptr<const core::GgufReader>& reader,
                       std::chrono::steady_clock::time_point model_load_start) {
  return RunLagunaBenchmarkImpl(opt, reader, model_load_start);
}

int RunLagunaPrompt(const ext::TextPromptOptions& opt,
                    const core::GgufReader& reader,
                    std::chrono::steady_clock::time_point model_load_start) {
  return RunLagunaPromptImpl(opt, reader, model_load_start);
}

int RunLagunaChat(const ext::TextPromptOptions& opt,
                  const core::GgufReader& reader,
                  std::chrono::steady_clock::time_point model_load_start) {
  return RunLagunaChatImpl(opt, reader, model_load_start);
}

}  // namespace gufo::models::laguna

#endif  // defined(ENGINE_ENABLE_HIP) && ENGINE_ENABLE_HIP
