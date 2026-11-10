// Laguna S 2.1 text runner.
//
// Adapts the Laguna GPU engine to the shared ext::TextModelRunner contract.
// The engine itself lives in src/models/laguna/gpu/; this file is the
// serving side: prefill, decode, speculative decoding through the DFlash
// drafter, session snapshots and tool-call aware prompting.

#include "src/models/laguna/runner.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/host.hpp>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gufo/ext/component_cache_resources.hpp"
#include "gufo/ext/host_budget.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/models/laguna/chat_template.hpp"
#include "src/models/laguna/continuation_adapter.hpp"
#include "src/models/laguna/gpu/engine.hpp"
#include "src/models/laguna/tool_calls.hpp"
#include "src/models/qwen/hip/dflash.hpp"

namespace gufo::models::laguna {
namespace {

using namespace gufo::ext;

using LagunaModel = gpu::Model;
using LagunaSession = gpu::Session;

std::vector<std::int32_t> LagunaEngineTokens(
    std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(
                    std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("Laguna token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

constexpr std::string_view kLagunaStateAbi = "laguna-rocm-session-v1";

[[nodiscard]] inline hip::ComponentCacheHooks HipComponentCacheHooks() {
  return {[] { return ext::HostSnapshotCeilingBytes(); },
          [](std::string_view) {}, [](std::string_view) {}};
}

}  // namespace

}  // namespace gufo::models::laguna

namespace gufo::models::laguna {

using namespace gufo::ext;
using gufo::server::Logger;

class LagunaTextRunnerState final : public TextRunnerState {
public:
  LagunaTextRunnerState(const std::shared_ptr<LagunaModel>& model,
                        std::uint32_t max_context) {
    std::string error;
    owned_ = model->CreateSession(max_context + LagunaSession::kMaxVerifyRows,
                                  &error);
    if (owned_ == nullptr) {
      throw std::runtime_error("Failed to create Laguna session: " + error);
    }
    session_ = owned_.get();
  }

  // The component cache owns execution storage. The runner state executes on
  // the session the slot bound rather than allocating one of its own.
  LagunaTextRunnerState(std::shared_ptr<LagunaModel> model,
                        std::uint32_t max_context, LagunaSession& session)
      : model_(std::move(model)), session_(&session) {
    (void)max_context;
  }

  void Invalidate() noexcept override {
    session_->Reset();
    position_ = 0;
    drafter_stats_ = {};
  }
  [[nodiscard]] TextRunnerMeasuredResources MeasuredResources()
      const noexcept override {
    return {.per_request_state_bytes = session_->AllocatedBytes(),
            .temporary_scratch_bytes = 0};
  }

  [[nodiscard]] LagunaSession& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

  struct DrafterStats {
    std::uint32_t proposed = 0;
    std::uint32_t accepted = 0;
    bool suspended = false;
  };
  [[nodiscard]] const DrafterStats& drafter_stats() const noexcept {
    return drafter_stats_;
  }
  [[nodiscard]] DrafterStats& drafter_stats() noexcept {
    return drafter_stats_;
  }

private:
  std::shared_ptr<LagunaModel> model_;
  std::unique_ptr<LagunaSession> owned_;
  LagunaSession* session_{nullptr};
  std::size_t position_{0};
  DrafterStats drafter_stats_;
};

LagunaTextRunnerState& RequireLagunaState(TextRunnerState& state) {
  auto* laguna = dynamic_cast<LagunaTextRunnerState*>(&state);
  if (laguna == nullptr)
    throw std::logic_error("text runner state is not Laguna");
  return *laguna;
}

const LagunaTextRunnerState& RequireLagunaState(const TextRunnerState& state) {
  const auto* laguna = dynamic_cast<const LagunaTextRunnerState*>(&state);
  if (laguna == nullptr)
    throw std::logic_error("text runner state is not Laguna");
  return *laguna;
}

class LagunaTextRunnerSnapshot final : public TextRunnerSnapshot {
public:
  LagunaTextRunnerSnapshot(std::shared_ptr<LagunaModel> model,
                           std::vector<std::uint8_t> bytes,
                           std::size_t position)
      : model(std::move(model)), bytes(std::move(bytes)), position(position) {}

  [[nodiscard]] std::size_t PayloadBytes() const noexcept override {
    return bytes.size();
  }

  std::shared_ptr<LagunaModel> model;
  std::vector<std::uint8_t> bytes;
  std::size_t position;
};

class LagunaTextRunner final : public HipTextModelRunner {
public:
  static constexpr std::size_t kPrefillChunk = 16384;

  /// Tokens one Prefill() call hands the engine. Bounded by the engine's own
  /// prefill tile and by how the routed expert GEMM tiles a bucket: at 4096
  /// rows each expert gets 160 rows against a 128-row tile, so 37.5% of the
  /// second tile is padding. LAGUNA_PREFILL_TILE raises it for the wider-pass
  /// experiment; it must not exceed that tile.
  [[nodiscard]] static std::size_t PrefillChunkRows() {
    static const std::size_t value = [] {
      if (const char* const v = std::getenv("LAGUNA_PREFILL_TILE")) {
        const int parsed = std::atoi(v);
        if (parsed > 0) {
          return static_cast<std::size_t>(parsed);
        }
      }
      return kPrefillChunk;
    }();
    return value;
  }
  static constexpr std::size_t kWarmupTokens = 4096;
  // A device that has just released ~109 GiB of GTT needs a moment before
  // its next allocation lands correctly. Two attempts with a pause
  // between them covers a restart straight after the previous server
  // exited.
  static constexpr int kProbeAttempts = 3;
  static constexpr int kProbeRetryWaitSeconds = 45;

  LagunaTextRunner(std::shared_ptr<LagunaModel> model,
                   std::unique_ptr<tokenization::QwenTokenizer> tokenizer,
                   std::vector<std::int32_t> stop_tokens,
                   std::uint32_t max_context,
                   std::uint32_t max_draft_tokens = 0)
      : model_(std::move(model)),
        tokenizer_(std::move(tokenizer)),
        stop_tokens_(std::move(stop_tokens)),
        max_context_(max_context),
        max_draft_tokens_(max_draft_tokens) {
    MeasureSessionBytes();
    Warmup();
  }

  // Resource admission needs an upfront per-request state estimate; the
  // engine has no closed-form size (KV plus banked NPU scratch vary with
  // config), so probe it once with a throwaway session of the real size.
  //
  // The probe also catches a context the device cannot hold. On a
  // unified-memory GPU the device pool is backed by host RAM, so an
  // over-large KV cache can be admitted and then fail to back its pages:
  // the kernels read unbacked memory and every forward returns non-finite
  // logits. That reads as a model fault rather than a memory fault, so
  // refuse to serve instead.
  //
  // It also catches a device that is still settling. When the previous
  // server exited, it left ~109 GiB of GTT mapped; a new server started
  // before the device recovered gets an allocation that does not match
  // reality and every forward reads non-finite values. The device
  // recovers on its own, so retry after a pause rather than failing:
  // measured on the Nimo Axis N161, one start in four was broken without
  // the pause and none of four were broken with it.
  void MeasureSessionBytes() {
    std::string error;
    std::unique_ptr<LagunaSession> probe;
    for (int attempt = 1; attempt <= kProbeAttempts; ++attempt) {
      probe = model_->CreateSession(
          max_context_ + LagunaSession::kMaxVerifyRows, &error);
      if (probe != nullptr) {
        break;
      }
      if (!IsDeviceUnsettled(error) || attempt == kProbeAttempts) {
        break;
      }
      Logger::Warn(
          "loader",
          "event=session_probe_retry attempt=" + std::to_string(attempt) +
              " wait_s=" + std::to_string(kProbeRetryWaitSeconds) +
              " reason=device_settling");
      std::this_thread::sleep_for(std::chrono::seconds(kProbeRetryWaitSeconds));
    }
    if (probe == nullptr) {
      Logger::Error("loader", "event=session_probe_failed reason=" + error);
      if (IsDeviceUnsettled(error)) {
        // Every attempt produced non-finite logits. The device is not
        // settling on its own, and serving anyway would fail every
        // request with the same error.
        device_unsettled_ = true;
      }
      return;
    }
    session_bytes_ = probe->AllocatedBytes();
  }

  // A non-finite forward means the device produced garbage, not that the
  // model or the context is wrong.
  [[nodiscard]] static bool IsDeviceUnsettled(const std::string& error) {
    return error.find("non-finite logits") != std::string::npos;
  }

  // True when the load-time probe never got a finite forward out of the
  // device.
  [[nodiscard]] bool DeviceUnsettled() const noexcept {
    return device_unsettled_;
  }

  void Warmup() const {
    const char* disabled = std::getenv("GUFO_WARMUP");
    if (disabled != nullptr && disabled[0] == '0')
      return;
    try {
      const std::size_t room = max_context_ > 128 ? max_context_ - 128 : 0;
      const std::size_t length = std::min<std::size_t>(kWarmupTokens, room);
      if (length < 1024)
        return;
      const auto started = std::chrono::steady_clock::now();
      std::string error;
      // The warmup is the first real forward on the device, so it is where
      // a device that has not settled shows itself. The probe above
      // already covers the same ground, but the warmup runs a wider
      // forward and can fail where the probe did not; retry the same way.
      std::unique_ptr<LagunaSession> session;
      for (int attempt = 1; attempt <= kProbeAttempts; ++attempt) {
        session = model_->CreateSession(static_cast<std::uint32_t>(length + 16),
                                        &error);
        if (session != nullptr) {
          break;
        }
        if (!IsDeviceUnsettled(error) || attempt == kProbeAttempts) {
          break;
        }
        Logger::Warn("loader",
                     "event=warmup_retry attempt=" + std::to_string(attempt) +
                         " wait_s=" + std::to_string(kProbeRetryWaitSeconds) +
                         " reason=device_settling");
        std::this_thread::sleep_for(
            std::chrono::seconds(kProbeRetryWaitSeconds));
      }
      if (session == nullptr) {
        Logger::Warn("loader", "event=warmup_skipped reason=" + error);
        return;
      }
      const auto unit = tokenizer_->Encode(
          "The quick brown fox jumps over the lazy dog while the engine warms "
          "its kernels. ");
      if (unit.empty())
        return;
      std::vector<std::int32_t> prompt;
      prompt.reserve(length);
      for (std::size_t i = 0; i < length; ++i)
        prompt.push_back(static_cast<std::int32_t>(unit[i % unit.size()]));
      if (!session->Append(prompt, &error)) {
        Logger::Warn("loader", "event=warmup_failed reason=" + error);
        return;
      }
      for (int step = 0; step < 4; ++step) {
        const std::int32_t next = 1;
        if (!session->Append({&next, 1}, &error))
          break;
      }
      const auto elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();
      Logger::Info("loader",
                   "event=warmup_completed tokens=" + std::to_string(length) +
                       " elapsed_s=" + std::to_string(elapsed));
    } catch (const std::exception& failure) {
      Logger::Warn("loader",
                   std::string("event=warmup_failed reason=") + failure.what());
    }
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = "Laguna S 2.1",
        .state_abi = std::string(kLagunaStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .incremental_prefill = true,
                .snapshot = true,
                .fork = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = max_draft_tokens_ != 0,
                .prefix_reuse = true,
                .prefill_grid_tokens = 4096,
            },
        .persistence = std::nullopt,
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess)
      capacity = free_bytes;
    return {
        .resident_weights_bytes = model_->ResidentBytes(),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = session_bytes_,
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = HostSnapshotCeilingBytes(),
        .requires_device_runtime_lock = true,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {{.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        tokenizer_->GetVocabSize(), [this](std::uint32_t id) {
          return sampling::ConstraintVocabulary::Piece{
              tokenizer_->DecodeTokenCopy(id), IsStop(id)};
        });
  }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(
      std::string_view text) const override {
    return tokenizer_->Encode(text);
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    std::vector<models::laguna::ChatMessage> messages;
    messages.reserve(request.messages.size());
    for (const auto& message : request.messages) {
      if (!message.images.empty())
        throw std::invalid_argument("Laguna does not support image input");
      models::laguna::ChatMessage rendered{
          std::string(tokenization::ToString(message.role)),
          message.content + message.framing_suffix,
          message.thought,
          {}};
      for (const auto& call : message.tool_calls) {
        models::laguna::ChatToolCall native;
        native.name = call.name;
        native.arguments.reserve(call.arguments.size());
        for (const auto& argument : call.arguments) {
          native.arguments.push_back(
              {argument.name, argument.value, argument.is_string});
        }
        rendered.tool_calls.push_back(std::move(native));
      }
      messages.push_back(std::move(rendered));
      if (message.role == tokenization::ChatRole::kDeveloper)
        messages.back().role = "developer";
    }
    models::laguna::ChatTemplateOptions chat_options;
    chat_options.enable_thinking = request.reasoning.enabled.value_or(false);
    chat_options.tools.reserve(request.tools.size());
    for (const auto& tool : request.tools) {
      chat_options.tools.push_back(
          {tool.name, tool.description, tool.parameters_json});
    }
    tokenization::TokenizerOptions options;
    options.parse_special_tokens = true;
    return tokenizer_->Encode(
        models::laguna::RenderPrompt(messages, chat_options), options);
  }

  [[nodiscard]] sampling::JsonConstraint::ToolFormat ToolFormat()
      const override {
    // Laguna has no built-in format: its envelope is a provider syntax, so the
    // host builds the grammar from the registered dialect named by
    // ToolDialectId().
    return sampling::JsonConstraint::ToolFormat::kJson;
  }

  [[nodiscard]] std::string ToolDialectId() const override {
    return "laguna";
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    auto prepared = TextModelRunner::PreparePrompt(request);
    if (prepared) {
      // The generation prefix (`<|assistant|>`) is not what the next turn
      // replays for this reply, so the reusable frontier ends where a
      // further user message would diverge from it.
      auto continuation = request;
      continuation.messages.emplace_back(tokenization::ChatRole::kUser, "");
      const auto continued = RenderAndTokenize(continuation);
      if (continued) {
        prepared->cache_prefix_tokens = static_cast<std::size_t>(
            std::mismatch(prepared->tokens.begin(), prepared->tokens.end(),
                          continued->begin(), continued->end())
                .first -
            prepared->tokens.begin());
      }
    }
    return prepared;
  }

  [[nodiscard]] std::string Decode(
      std::span<const TextRunnerToken> tokens) const override {
    return tokenizer_->Decode(tokens);
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> CreateState() const override {
    throw std::logic_error(
        "Laguna execution slots are owned by the component cache");
  }

  [[nodiscard]] bool UsesComponentCache() const override { return true; }

  [[nodiscard]] std::unique_ptr<ComponentCacheResources>
  CreateComponentCacheResources(cache::ResourceLedger& ledger,
                                std::size_t) const override {
    return hip::CreateHipComponentCacheResources(
        std::make_unique<LagunaComponentAdapter>(
            model_, max_context_,
            LagunaProcessIdentity(model_.get(), max_context_)),
        ledger, HipComponentCacheHooks());
  }

  [[nodiscard]] std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    auto& laguna = dynamic_cast<LagunaComponentAdapter&>(adapter);
    return std::make_unique<LagunaTextRunnerState>(model_, max_context_,
                                                   laguna.Session(slot));
  }

  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t position) const override {
    RequireLagunaState(state).set_position(position);
  }

  void PreparePrefixReuse(
      TextRunnerState& state,
      std::span<const TextRunnerToken> prefix) const override {
    if (RequireLagunaState(state).position() != prefix.size())
      throw std::logic_error("Laguna reused prefix does not match checkpoint");
  }

  [[nodiscard]] TextPrefillStep Prefill(
      TextRunnerState& state, std::span<const TextRunnerToken> prompt,
      std::size_t offset, std::size_t max_input_tokens) const override {
    auto& laguna = RequireLagunaState(state);
    if (offset != laguna.position())
      throw std::logic_error("Laguna prefill offset does not match state");
    if (offset >= prompt.size())
      throw std::logic_error("Laguna prefill has no remaining input");
    // A fresh request starts a new acceptance history; a continued
    // conversation keeps the previous request's, which still describes
    // how well the drafter tracks this context.
    if (offset == 0)
      laguna.drafter_stats() = {};
    const std::size_t consumed = std::min<std::size_t>(
        {max_input_tokens, prompt.size() - offset, PrefillChunkRows()});
    const auto chunk = LagunaEngineTokens(prompt.subspan(offset, consumed));
    std::string error;
    if (!laguna.session().Append(chunk, &error)) {
      laguna.Invalidate();
      throw std::runtime_error("Laguna prefill failed: " + error);
    }
    laguna.set_position(offset + consumed);
    return {.consumed_tokens = consumed,
            .decode_ready = offset + consumed == prompt.size()};
  }

  [[nodiscard]] TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& laguna = RequireLagunaState(state);
    if (laguna.position() >= max_context_)
      return {.stop = true, .piece = {}, .logprobs = std::nullopt};
    const auto logits = laguna.session().Logits();
    if (logits.empty())
      throw std::runtime_error("Laguna token selection has no logits");
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits));
    if (IsStop(token))
      return {.stop = true, .token = 0, .piece = {}, .logprobs = std::nullopt};
    return {.stop = false,
            .token = static_cast<TextRunnerToken>(token),
            .piece = std::string(tokenizer_->DecodeToken(token)),
            .logprobs = std::nullopt};
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& laguna = RequireLagunaState(state);
    const std::int32_t input = LagunaEngineTokens({&token, 1}).front();
    std::string error;
    if (!laguna.session().Append({&input, 1}, &error)) {
      laguna.Invalidate();
      throw std::runtime_error("Laguna decode failed: " + error);
    }
    laguna.set_position(laguna.position() + 1);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  // DFlash2 speculative decode, mirroring the prompt CLI's verify loop:
  // one anchor plus up to max_draft_tokens_ drafted tokens are appended in
  // a single AppendAll, each draft is accepted on an exact greedy match
  // against its row's logits, the rejected suffix is rewound and the
  // accepted prefix is committed to the drafter. Only used when the
  // drafter's injected context covers the session position; otherwise
  // (cold start after prefill or a rewind) the engine's own tap injection
  // re-syncs it during the plain decode step.
  [[nodiscard]] TextDecodeStep DecodeStep(
      TextRunnerState& state, std::size_t max_tokens,
      sampling::SamplerState& sampler) const override {
    if (max_tokens == 0) {
      throw std::invalid_argument("decode budget must be at least one token");
    }
    auto& laguna = RequireLagunaState(state);
    if (laguna.position() >= max_context_)
      return {
          .selections = {}, .stop = true, .terminal_logprobs = std::nullopt};
    auto* drafter = laguna.session().DflashExecutor();
    // Speculation runs for greedy AND sampled targets. The prompt CLI
    // (src/cli/prompt/prompt.cpp) has always done so: it samples the
    // drafter's proposals at the target's temperature and accepts a draft
    // with probability P_target(draft), resampling from the residual when
    // it misses. That rule is unbiased - P(publish = z) works out to
    // q(z)p(z) + p(z)(1 - q(z)) = p(z) - and it accepts far more often than
    // drawing a token from the target and comparing, which is what this
    // path used to do for sampled requests.
    //
    // The gate that used to refuse sampled requests here left the server
    // running plain decode for every sampled request, which is most of
    // them: an agent client that sends a temperature gets none of the
    // documented DFlash2 decode speed.
    //
    // Acceptance gate: drafting only pays when enough drafts stick.
    // Measured on gfx1151 (docs/models/laguna/KV-Q8.md, BENCHMARKS.md),
    // break-even sits near 45-50% acceptance at depth: 57% acceptance
    // decodes 24.5 tok/s against 18-21 plain, 40% decodes 16.1 against
    // 18.1. Agent traffic at temperature 0.7+ sits at 17-36% (30% steady
    // in serving logs), so once a session's acceptance drops below the
    // threshold over a minimum sample, the drafter is suspended for the
    // rest of the request and plain decode runs instead. GUFO_DFLASH_GATE
    // =0 restores the ungated behaviour.
    static const bool kDraftGateEnabled = [] {
      const char* v = std::getenv("GUFO_DFLASH_GATE");
      return v == nullptr || std::strcmp(v, "0") != 0;
    }();
    static const std::uint32_t kDraftGatePercent = [] {
      if (const char* v = std::getenv("GUFO_DFLASH_GATE_THRESHOLD")) {
        const auto pct =
            static_cast<std::uint32_t>(std::strtoul(v, nullptr, 10));
        if (pct > 0 && pct <= 100)
          return pct;
      }
      return 45U;
    }();
    constexpr std::uint32_t kDraftGateMinProposals = 24;
    const auto& stats = laguna.drafter_stats();
    const bool acceptance_ok =
        !kDraftGateEnabled || !stats.suspended ||
        stats.proposed < kDraftGateMinProposals ||
        stats.accepted * 100U >= kDraftGatePercent * stats.proposed;
    const bool drafter_ready =
        acceptance_ok && max_tokens > 1 && max_draft_tokens_ > 0 &&
        drafter != nullptr && model_->HasDflash() &&
        drafter->GetInjectedContextLength() == laguna.session().Position();
    if (!drafter_ready) {
      if (std::getenv("LAGUNA_DFLASH_DEBUG") != nullptr) {
        std::fprintf(
            stderr,
            "[LAGUNA_DFLASH_DEBUG] drafting skipped: budget=%zu "
            "max_draft=%u drafter=%s has_dflash=%d injected=%u "
            "position=%u sampled=%d\n",
            max_tokens, max_draft_tokens_, drafter == nullptr ? "null" : "ok",
            model_->HasDflash() ? 1 : 0,
            drafter == nullptr ? 0U : drafter->GetInjectedContextLength(),
            laguna.session().Position(),
            sampler.config().uses_random_sampling() ? 1 : 0);
      }
      return TextModelRunner::DecodeStep(state, max_tokens, sampler);
    }
    const auto anchor_logits = laguna.session().Logits();
    if (anchor_logits.empty())
      throw std::runtime_error("Laguna token selection has no logits");
    // A grammar constrains the token stream and its state advances with
    // every published token. Verifying a batch of rows means sampling row
    // k against the state that follows row k's token, not against the
    // state the step started with: a draft the grammar forbids at its
    // own position must be rejected, and the sample that replaced it
    // must be valid there. Sample from a tentative copy of the sampler
    // that advances as drafts are accepted, and publish only its draw
    // state; the pool accepts the committed tokens once.
    sampling::SamplerState working_sampler = sampler;
    const auto anchor =
        static_cast<std::int32_t>(working_sampler.Sample(anchor_logits));
    if (IsStop(anchor)) {
      sampler.CopyDrawStateFrom(working_sampler);
      return {
          .selections = {}, .stop = true, .terminal_logprobs = std::nullopt};
    }
    const std::uint32_t base = laguna.session().Position();
    // A drafter that samples needs one random value per proposal. Draw them
    // from the working sampler's RNG so the draws are part of the state
    // published at the end of the step.
    const auto wanted = std::min<std::uint32_t>(
        max_draft_tokens_, static_cast<std::uint32_t>(max_tokens - 1));
    const bool sampled = sampler.config().uses_random_sampling();
    std::vector<float> draft_uniforms;
    if (sampled) {
      auto draft_rng =
          sampling::NextRandom(working_sampler.mutable_rng_state());
      draft_uniforms.reserve(wanted);
      for (std::uint32_t row = 0; row < wanted; ++row) {
        draft_uniforms.push_back(
            static_cast<float>(sampling::Uniform(&draft_rng)));
      }
    }
    const auto drafts = drafter->ForwardBlock(
        anchor, base, wanted, sampled ? sampler.config().temperature : 0.0F,
        draft_uniforms);
    std::vector<std::int32_t> feed{anchor};
    feed.insert(feed.end(), drafts.begin(), drafts.end());
    std::string error;
    if (!laguna.session().AppendAll(feed, &error)) {
      laguna.Invalidate();
      throw std::runtime_error("Laguna verify failed: " + error);
    }
    TextDecodeStep step;
    step.draft_tokens = drafts.size();
    step.draft_rounds = 1;
    step.selections.push_back(
        {.stop = false,
         .token = static_cast<TextRunnerToken>(anchor),
         .piece = std::string(tokenizer_->DecodeToken(anchor)),
         .logprobs = std::nullopt});
    // Row i holds the distribution after feed[i], so row 0's distribution
    // follows the anchor: advance the tentative sampler past it before
    // sampling any row.
    working_sampler.Accept(anchor);
    // Row i holds the distribution after feed[i]; draft feed[i + 1] is
    // tested against it, and only accepted rows are committed to the
    // drafter. A stop-token draft ends the request exactly like the base
    // path: it is committed to the KV cache but never published.
    std::uint32_t kept = 0;
    bool stopped = false;
    std::int32_t pending = 0;
    bool have_pending = false;
    for (std::uint32_t row = 0;; ++row) {
      const auto logits = laguna.session().Logits(row);
      if (row + 1 >= feed.size())
        break;  // full acceptance: next step re-samples the new anchor
      const auto guess = feed[row + 1];
      // The acceptance rule mirrors the prompt CLI's. A greedy target
      // draws its argmax and compares. A sampled target accepts the draft
      // with probability P_target(guess) and, on a miss, resamples from
      // the residual (the target distribution with the draft removed).
      // Both are unbiased: for a draft sampled from the drafter's
      // distribution q, the published token z lands with probability
      // q(z)p(z) + p(z)(1 - q(z)) = p(z).
      std::int32_t next = 0;
      bool hit = false;
      if (sampled) {
        const sampling::TokenId guess_id =
            static_cast<sampling::TokenId>(guess);
        const double p =
            working_sampler.Distribution(logits).probability(guess_id);
        hit = working_sampler.Uniform() < p;
        if (hit) {
          next = guess;
        } else {
          const float one = 1.0F;
          next = static_cast<std::int32_t>(working_sampler.SampleResidual(
              logits, {&guess_id, 1}, {&one, 1}));
        }
      } else {
        next = static_cast<std::int32_t>(working_sampler.Sample(logits));
        hit = next == guess;
      }
      if (!hit) {
        // Rejected draft: this draw is a valid sample of the target
        // distribution at the last accepted token. Rewind() does not
        // roll the logits rows back (Logits() still returns the last
        // row's distribution), so the token is committed with its own
        // single-token forward below, which also injects the drafter tap
        // that keeps the next step synced.
        pending = next;
        have_pending = true;
        break;
      }
      if (IsStop(guess)) {
        // The stop draft is NOT accepted: it is never published, and
        // CheckpointPosition() must equal prompt + published tokens. The
        // Rewind below drops its KV row (kept still excludes it), and the
        // drafter never sees it. The request ends here.
        stopped = true;
        break;
      }
      ++kept;
      // The draft matched a sample drawn under the current grammar
      // state, so it is valid there. Advance the tentative sampler so
      // the next row is sampled against the state this token produces.
      working_sampler.Accept(guess);
      step.selections.push_back(
          {.stop = false,
           .token = static_cast<TextRunnerToken>(guess),
           .piece = std::string(tokenizer_->DecodeToken(guess)),
           .logprobs = std::nullopt});
    }
    // Publish the tentative sampler's draw state so the next step
    // continues the same RNG stream and residual draw.
    sampler.CopyDrawStateFrom(working_sampler);
    // AppendAll advanced the KV cache and position by the whole verified
    // batch; only the accepted prefix (anchor + kept drafts) sticks.
    laguna.session().Rewind(base + 1 + kept);
    if (!laguna.session().DflashInject(kept + 1, &error)) {
      laguna.Invalidate();
      throw std::runtime_error("Laguna drafter commit failed: " + error);
    }
    if (have_pending) {
      if (step.selections.size() >= max_tokens) {
        // Budget exhausted: leave the state rewound at the accepted prefix
        // and publish nothing more. The request completes at the token
        // limit; the state position must equal prompt + published tokens.
        laguna.set_position(laguna.session().Position());
      } else {
        if (!laguna.session().Append({&pending, 1}, &error)) {
          laguna.Invalidate();
          throw std::runtime_error("Laguna decode failed: " + error);
        }
        laguna.set_position(laguna.session().Position());
        if (IsStop(pending)) {
          // The stop token was appended to the KV cache before we could
          // know it was one - roll it back so the state matches the
          // published tokens, then stop. `stopped` (not step.stop) drives
          // the final assignment below; setting step.stop here would be
          // overwritten and the request would continue past the stop.
          laguna.session().Rewind(laguna.session().Position() - 1);
          laguna.set_position(laguna.session().Position());
          stopped = true;
        } else {
          step.selections.push_back(
              {.stop = false,
               .token = static_cast<TextRunnerToken>(pending),
               .piece = std::string(tokenizer_->DecodeToken(pending)),
               .logprobs = std::nullopt});
        }
      }
    } else {
      laguna.set_position(base + 1 + kept);
    }
    step.draft_accepted_tokens = kept;
    step.stop = stopped;
    {
      auto& draft_stats = laguna.drafter_stats();
      draft_stats.proposed += drafts.size();
      draft_stats.accepted += kept;
      if (kDraftGateEnabled && !draft_stats.suspended &&
          draft_stats.proposed >= kDraftGateMinProposals &&
          draft_stats.accepted * 100U <
              kDraftGatePercent * draft_stats.proposed) {
        draft_stats.suspended = true;
        Logger::Info("laguna",
                     "event=dflash_suspended accepted=" +
                         std::to_string(draft_stats.accepted) +
                         " proposed=" + std::to_string(draft_stats.proposed) +
                         " acceptance_pct=" +
                         std::to_string(100.0 * draft_stats.accepted /
                                        draft_stats.proposed) +
                         " gate_pct=" + std::to_string(kDraftGatePercent));
      }
    }
    return step;
  }

  [[nodiscard]] std::size_t CheckpointPosition(
      const TextRunnerState& state) const override {
    return RequireLagunaState(state).position();
  }

  [[nodiscard]] std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const override {
    return RequireLagunaState(state).session().SnapshotBytes();
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    const auto& laguna = RequireLagunaState(state);
    std::vector<std::uint8_t> bytes(laguna.session().SnapshotBytes());
    std::string error;
    if (!laguna.session().SaveSnapshot(bytes, &error)) {
      throw std::runtime_error("Laguna snapshot failed: " + error);
    }
    if (std::getenv("LAGUNA_SNAPSHOT_DEBUG") != nullptr) {
      std::fprintf(
          stderr,
          "[LAGUNA_SNAPSHOT_DEBUG] save runner_pos=%zu session_pos=%u "
          "dflash_frontier=%u bytes=%zu dflash_sum=%.6f\n",
          laguna.position(), laguna.session().Position(),
          laguna.session().DflashExecutor() != nullptr
              ? laguna.session().DflashExecutor()->GetInjectedContextLength()
              : 0U,
          bytes.size(),
          laguna.session().DflashExecutor() != nullptr
              ? laguna.session().DflashExecutor()->InjectedContextChecksum()
              : 0.0);
    }
    return std::make_unique<LagunaTextRunnerSnapshot>(model_, std::move(bytes),
                                                      laguna.position());
  }

  [[nodiscard]] std::unique_ptr<TextRunnerSnapshot> SnapshotForPersistence(
      const TextRunnerState& state) const override {
    return Snapshot(state);
  }

  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& snapshot) const override {
    const auto* laguna_snapshot =
        dynamic_cast<const LagunaTextRunnerSnapshot*>(&snapshot);
    if (laguna_snapshot == nullptr ||
        laguna_snapshot->model.get() != model_.get()) {
      throw std::invalid_argument(
          "Laguna snapshot does not belong to this model");
    }
    auto& restored = RequireLagunaState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(laguna_snapshot->bytes, &error)) {
      restored.Invalidate();
      throw std::runtime_error("Laguna snapshot restore failed: " + error);
    }
    if (std::getenv("LAGUNA_SNAPSHOT_DEBUG") != nullptr) {
      std::fprintf(
          stderr,
          "[LAGUNA_SNAPSHOT_DEBUG] restore snapshot_pos=%zu "
          "session_pos=%u dflash_frontier=%u dflash_sum=%.6f\n",
          laguna_snapshot->position, restored.session().Position(),
          restored.session().DflashExecutor() != nullptr
              ? restored.session().DflashExecutor()->GetInjectedContextLength()
              : 0U,
          restored.session().DflashExecutor() != nullptr
              ? restored.session().DflashExecutor()->InjectedContextChecksum()
              : 0.0);
    }
    restored.set_position(laguna_snapshot->position);
  }

  [[nodiscard]] std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& snapshot) const override {
    const auto* laguna_snapshot =
        dynamic_cast<const LagunaTextRunnerSnapshot*>(&snapshot);
    if (laguna_snapshot == nullptr ||
        laguna_snapshot->model.get() != model_.get()) {
      throw std::invalid_argument(
          "Laguna persistent snapshot does not belong to this model");
    }
    return laguna_snapshot->bytes.size();
  }

  [[nodiscard]] std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& snapshot,
      std::span<std::uint8_t> destination) const override {
    const auto* laguna_snapshot =
        dynamic_cast<const LagunaTextRunnerSnapshot*>(&snapshot);
    if (laguna_snapshot == nullptr ||
        laguna_snapshot->model.get() != model_.get() ||
        destination.size() != laguna_snapshot->bytes.size()) {
      throw std::invalid_argument(
          "Laguna persistent snapshot serialization failed");
    }
    std::memcpy(destination.data(), laguna_snapshot->bytes.data(),
                laguna_snapshot->bytes.size());
    return destination.size();
  }

  void StreamPersistentSnapshot(const TextRunnerSnapshot& snapshot,
                                const SnapshotSink& sink) const override {
    (void)PersistentSnapshotPayloadBytes(snapshot);
    sink(dynamic_cast<const LagunaTextRunnerSnapshot&>(snapshot).bytes);
  }

  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    auto& restored = RequireLagunaState(state);
    std::string error;
    if (!restored.session().RestoreSnapshot(payload, &error)) {
      restored.Invalidate();
      throw std::runtime_error("Laguna persistent snapshot restore failed: " +
                               error);
    }
    const std::uint32_t position = restored.session().Position();
    if (position == 0 || position > max_context_) {
      restored.Invalidate();
      throw std::runtime_error(
          "Laguna persistent snapshot restored an invalid position");
    }
    restored.set_position(position);
  }

private:
  [[nodiscard]] bool IsStop(std::int32_t token) const {
    return std::find(stop_tokens_.begin(), stop_tokens_.end(), token) !=
           stop_tokens_.end();
  }

  std::shared_ptr<LagunaModel> model_;
  std::unique_ptr<tokenization::QwenTokenizer> tokenizer_;
  std::vector<std::int32_t> stop_tokens_;
  std::uint32_t max_context_;
  std::uint32_t max_draft_tokens_{0};
  std::size_t session_bytes_{0};
  // Set when the load-time probe could not get a finite forward out of the
  // device even after retrying. Every request would fail the same way, so
  // the load fails with the reason instead.
  bool device_unsettled_{false};
};
}  // namespace gufo::models::laguna

namespace gufo::models::laguna {

std::shared_ptr<ext::TextModelRunner> CreateLagunaTextRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  // The context comes from the model unless the caller asked for one. The
  // GGUF carries the native window under the architecture-prefixed key
  // the other extensions read.
  std::uint32_t max_context = options.max_context;
  if (max_context == 0) {
    const std::uint32_t native =
        options.reader->GetMetadataUint32("laguna.context_length").value_or(0);
    if (native < 2) {
      if (error != nullptr) {
        *error = "GGUF has no valid native context length; specify --context";
      }
      return nullptr;
    }
    max_context = native;
  }
  std::string load_error;
  gpu::EngineOptions engine_options;
  engine_options.max_context = max_context;
  engine_options.kv_cache_type = gpu::KvCacheType::kF16;
  // The engine owns the drafter: it loads the DFlash model, binds the taps
  // to this model's trunk and hands the executor to every session.
  if (options.speculative.backend == ext::TextSpeculativeBackend::kDFlash) {
    if (options.speculative.draft_model_path.empty()) {
      if (error != nullptr) {
        *error = "DFlash decoding requires --dflash-model";
      }
      return nullptr;
    }
    engine_options.dflash_model_path = options.speculative.draft_model_path;
  }
  auto model =
      gpu::Model::Load(options.model_path, engine_options, &load_error);
  if (model == nullptr) {
    if (error != nullptr) {
      *error = "Failed to load Laguna model: " + load_error;
    }
    return nullptr;
  }
  // Pick the largest context the device can hold. The weights and the
  // drafter are resident now, so what is left is what the KV cache can
  // use. An explicit --context is the user's decision to honour or to
  // refuse, never to reduce silently.
  if (options.max_context == 0) {
    const std::size_t free_bytes = LagunaModel::FreeDeviceBytes();
    if (free_bytes != 0) {
      const std::uint32_t native = max_context;
      std::uint32_t chosen = native;
      // A session holds the window plus the verify rows, so size the cache
      // for what the session will really ask for.
      while (chosen > 4096 && model->SessionBytesForContext(
                                  chosen + LagunaSession::kMaxVerifyRows) *
                                      2 >
                                  free_bytes) {
        chosen /= 2;
      }
      if (chosen != native) {
        Logger::Info(
            "loader",
            "event=context_clamped native=" + std::to_string(native) +
                " chosen=" + std::to_string(chosen) +
                " reason=kv_cache_exceeds_free_device_memory free_mib=" +
                std::to_string(free_bytes / (1024 * 1024)) + " kv_mib=" +
                std::to_string(model->SessionBytesForContext(
                                   chosen + LagunaSession::kMaxVerifyRows) /
                               (1024 * 1024)));
        max_context = chosen;
      }
    }
  }
  auto tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(*options.reader, &load_error);
  if (tokenizer == nullptr) {
    if (error != nullptr) {
      *error = "Laguna tokenizer failed: " + load_error;
    }
    return nullptr;
  }
  std::vector<std::int32_t> stop_tokens;
  for (const char* key :
       {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
        "tokenizer.ggml.eom_token_id"}) {
    if (const auto id = options.reader->GetMetadataUint32(key))
      stop_tokens.push_back(static_cast<std::int32_t>(*id));
  }
  const std::uint32_t max_draft_tokens =
      options.speculative.draft_model_path.empty()
          ? 0
          : std::max(options.speculative.max_draft_tokens, 1u);
  auto runner = std::make_shared<LagunaTextRunner>(
      std::move(model), std::move(tokenizer), std::move(stop_tokens),
      max_context, max_draft_tokens);
  if (runner->DeviceUnsettled()) {
    if (error != nullptr) {
      *error =
          "Laguna: the GPU did not settle - it produced non-finite values on "
          "every attempt. Another server probably just exited and released "
          "the device; wait a minute and start again";
    }
    return nullptr;
  }
  return runner;
}

}  // namespace gufo::models::laguna

#endif  // defined(ENGINE_ENABLE_HIP)
