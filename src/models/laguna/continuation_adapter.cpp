#include "src/models/laguna/continuation_adapter.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gufo::models::laguna {
namespace {

using gpu::Session;

class Settled final : public cache::CompletionSignal {
public:
  explicit Settled(bool ok) : ok_(ok) {}
  [[nodiscard]] bool Ready() const noexcept override { return true; }
  [[nodiscard]] cache::TransferResult Wait() noexcept override {
    return ok_ ? cache::TransferResult::kSucceeded
               : cache::TransferResult::kFailed;
  }

private:
  bool ok_;
};

[[nodiscard]] cache::Completion Completed(bool ok) {
  return cache::Completion(std::make_unique<Settled>(ok));
}

constexpr std::size_t kSnapshotHeaderBytes = 9 * sizeof(std::uint32_t);
constexpr std::size_t kRowChunk = 256;

}  // namespace

cache::Identity LagunaProcessIdentity(const void* model,
                                      std::uint32_t max_context) {
  std::ostringstream identity;
  identity << "schema=gufo-process-laguna-component-v2\n"
           << "model_instance=" << model << '\n'
           << "context_tokens=" << max_context << '\n'
           << "state_abi=laguna-rocm-session-v1-component-v2\n"
           << "snapshot_payload_version=" << Session::kSnapshotPayloadVersion
           << '\n';
  const auto canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

class LagunaComponentAdapter::SlotState final : public cache::Slot {
public:
  SlotState(std::shared_ptr<gpu::Model> model, std::uint32_t capacity,
            std::uint32_t context, cache::MutationGuard& guard)
      : guard(guard), context(context) {
    std::string error;
    session_ = model->CreateSession(capacity, &error);
    if (session_ == nullptr) {
      throw std::runtime_error("cannot create Laguna component session: " +
                               error);
    }
  }

  ~SlotState() override = default;

  [[nodiscard]] bool IsValid() const noexcept override {
    return !restoring && !failed;
  }

  [[nodiscard]] gpu::Session& session() const { return *session_; }

  cache::MutationGuard& guard;
  std::uint32_t context;
  std::unique_ptr<gpu::Session> session_;
  bool restoring{false};
  bool failed{false};
  bool private_loaded{false};
  std::size_t restored_position{0};
};

LagunaComponentAdapter::LagunaComponentAdapter(
    std::shared_ptr<gpu::Model> model, std::uint32_t max_context,
    cache::Identity identity)
    : model_(std::move(model)),
      max_context_(max_context),
      identity_(std::move(identity)) {
  // A drafter's persistent payload is position-dependent and has no cheap
  // fixed bound here, so its adapter keeps the legacy single-payload path.
  // Without one, the full-attention K/V is exposed as shared append rows and
  // only the sliding-window ring, pooled keys and logits stay private.
  decomposed_ = model_ != nullptr && !model_->HasDflash();
  std::string error;
  auto probe =
      model_->CreateSession(max_context_ + Session::kMaxVerifyRows, &error);
  if (probe == nullptr) {
    throw std::runtime_error("cannot size the Laguna component cache: " +
                             error);
  }
  if (!decomposed_) {
    // The cache sizes one checkpoint buffer up front, so take the widest
    // snapshot this context can produce.
    state_bytes_ = probe->MaxSnapshotBytes() + kLengthTrailerBytes;
    descriptors_.push_back(cache::ComponentDescriptor{
        .id = cache::ComponentId{1},
        .layout_version = 1,
        .kind = cache::ComponentKind::kPrivateState,
        .row_bytes = 0,
        .rows_per_chunk = 0,
        .state_bytes = state_bytes_,
    });
    components_.push_back({cache::ComponentId{1}, 0, false, true});
    return;
  }
  kv_row_bytes_ = probe->KvRowBytes();
  private_bytes_ = probe->PrivateSnapshotBytes();
  const auto layers = probe->LayerCount();
  std::uint32_t next_id = 1;
  for (std::uint32_t il = 0; il < layers; ++il) {
    if (!probe->LayerIsFullAttention(il))
      continue;
    for (const bool value : {false, true}) {
      const cache::ComponentId id{next_id++};
      descriptors_.push_back(cache::ComponentDescriptor{
          .id = id,
          .layout_version = 1,
          .kind = cache::ComponentKind::kAppendRows,
          .row_bytes = kv_row_bytes_,
          .rows_per_chunk = static_cast<cache::Rows>(kRowChunk),
          .state_bytes = 0,
      });
      components_.push_back({id, il, value, false});
    }
  }
  const cache::ComponentId private_id{next_id++};
  descriptors_.push_back(cache::ComponentDescriptor{
      .id = private_id,
      .layout_version = 1,
      .kind = cache::ComponentKind::kPrivateState,
      .row_bytes = 0,
      .rows_per_chunk = 0,
      .state_bytes = private_bytes_,
  });
  components_.push_back({private_id, 0, false, true});
}

LagunaComponentAdapter::~LagunaComponentAdapter() = default;

LagunaComponentAdapter::SlotState& LagunaComponentAdapter::As(
    cache::Slot& slot) const {
  auto* state = dynamic_cast<SlotState*>(&slot);
  if (state == nullptr) {
    throw std::logic_error("cache slot is not a Laguna session");
  }
  return *state;
}

const LagunaComponentAdapter::SlotState& LagunaComponentAdapter::As(
    const cache::Slot& slot) const {
  return As(const_cast<cache::Slot&>(slot));
}

const cache::ComponentDescriptor& LagunaComponentAdapter::Component(
    cache::ComponentId id) const {
  const auto found =
      std::ranges::find(descriptors_, id, &cache::ComponentDescriptor::id);
  if (found == descriptors_.end())
    throw std::invalid_argument("invalid Laguna component ID");
  return *found;
}

const LagunaComponentAdapter::ComponentInfo& LagunaComponentAdapter::Info(
    cache::ComponentId id) const {
  const auto found = std::ranges::find(components_, id, &ComponentInfo::id);
  if (found == components_.end())
    throw std::invalid_argument("invalid Laguna component ID");
  return *found;
}

gpu::Session& LagunaComponentAdapter::Session(cache::Slot& slot) const {
  return As(slot).session();
}

std::vector<cache::BackingClass> LagunaComponentAdapter::BackingPlan(
    std::size_t ram_bytes) const {
  if (decomposed_) {
    // Rows and the private payload share fine pages, as for the other
    // component adapters. A fixed max-context block per checkpoint would
    // charge a shallow snapshot the whole context.
    constexpr std::size_t kPage = 65536;
    constexpr std::size_t kMetadataPerBlock = 1024;
    constexpr std::size_t kMetadataBase = 4096;
    const auto budget =
        ram_bytes > (4ULL << 20) ? (ram_bytes - (4ULL << 20)) / 10 * 9 : 0;
    if (budget <= kMetadataBase)
      return {};
    const auto blocks = (budget - kMetadataBase) / (kPage + kMetadataPerBlock);
    if (!blocks)
      return {};
    return {cache::BackingClass{
        .block_bytes = kPage,
        .blocks = blocks,
        .metadata_bytes = kMetadataBase + blocks * kMetadataPerBlock}};
  }
  // Legacy: one checkpoint-sized block per retained snapshot.
  constexpr std::size_t kPage = 4096;
  constexpr std::size_t kMetadataPerBlock = 4096;
  constexpr std::size_t kMetadataBase = 1 << 16;
  const std::size_t block = (state_bytes_ + kPage - 1) / kPage * kPage;
  const std::size_t blocks =
      std::max<std::size_t>(1, ram_bytes == 0 ? 1 : ram_bytes / block);
  return {cache::BackingClass{
      .block_bytes = block,
      .blocks = blocks,
      .metadata_bytes = blocks * kMetadataPerBlock + kMetadataBase}};
}

cache::Capabilities LagunaComponentAdapter::GetCapabilities() const {
  return {.continuation = true, .persistent_encoding = false};
}

std::span<const cache::ComponentDescriptor> LagunaComponentAdapter::Components()
    const {
  return descriptors_;
}

cache::Identity LagunaComponentAdapter::CompatibilityIdentity() const {
  return identity_;
}

std::unique_ptr<cache::Slot> LagunaComponentAdapter::CreateSlot(
    cache::MutationGuard& guard) {
  return std::make_unique<SlotState>(
      model_, max_context_ + Session::kMaxVerifyRows, max_context_, guard);
}

std::vector<cache::ComponentPosition> LagunaComponentAdapter::Positions(
    const cache::Slot& slot) const {
  const auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("cannot query an invalid Laguna slot");
  const auto position = state.session().Position();
  std::vector<cache::ComponentPosition> positions;
  positions.reserve(descriptors_.size());
  for (const auto& descriptor : descriptors_)
    positions.push_back({descriptor.id, position});
  return positions;
}

std::vector<cache::Rows> LagunaComponentAdapter::PlanPrefill(
    const cache::Slot& slot, cache::Rows first, cache::Rows end) const {
  const auto& state = As(slot);
  if (!state.IsValid() || first > end || end > max_context_)
    throw std::invalid_argument("invalid Laguna prefill plan");
  std::vector<cache::Rows> passes;
  while (first < end) {
    first += std::min<cache::Rows>(end - first, PrefillPassRows());
    passes.push_back(first);
  }
  return passes;
}

void LagunaComponentAdapter::BeginRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (state.restoring)
    throw std::logic_error("Laguna slot is already restoring");
  if (positions.size() != descriptors_.size())
    throw std::invalid_argument("incomplete Laguna component positions");
  const auto target = positions.front().valid_rows;
  if (target == 0 || target > max_context_)
    throw std::invalid_argument("Laguna restore frontier is out of range");
  for (const auto& position : positions)
    if (position.valid_rows != target ||
        std::ranges::count(descriptors_, position.id,
                           &cache::ComponentDescriptor::id) != 1)
      throw std::invalid_argument("inconsistent Laguna component positions");
  state.restoring = true;
  state.failed = false;
  state.private_loaded = false;
  state.restored_position = static_cast<std::size_t>(target);
}

cache::Completion LagunaComponentAdapter::CapturePrivate(
    const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes,
    cache::Stream&) {
  const auto& state = As(slot);
  const auto& descriptor = Component(id);
  if (bytes.size() != descriptor.state_bytes)
    throw std::invalid_argument("incorrect Laguna private buffer");
  if (!state.IsValid())
    throw std::logic_error("Laguna capture requires a valid slot");
  if (!decomposed_) {
    auto& session = state.session();
    const std::size_t length = session.SnapshotBytes();
    if (length == 0)
      return Completed(false);
    if (length + kLengthTrailerBytes > bytes.size()) {
      std::fprintf(stderr, "laguna capture: snapshot %zu exceeds buffer %zu\n",
                   length, bytes.size());
      return Completed(false);
    }
    std::string error;
    const std::span<std::uint8_t> out{
        reinterpret_cast<std::uint8_t*>(bytes.data()), length};
    if (!session.SaveSnapshot(out, &error)) {
      std::fprintf(stderr, "laguna capture: %s\n", error.c_str());
      return Completed(false);
    }
    std::memcpy(bytes.data() + bytes.size() - kLengthTrailerBytes, &length,
                sizeof(length));
    return Completed(true);
  }
  if (!Info(id).private_state)
    throw std::logic_error("Laguna row component is not private");
  std::string error;
  if (!state.session().SavePrivate(
          std::span<std::uint8_t>(reinterpret_cast<std::uint8_t*>(bytes.data()),
                                  bytes.size()),
          &error)) {
    std::fprintf(stderr, "laguna private capture: %s\n", error.c_str());
    return Completed(false);
  }
  return Completed(true);
}

cache::Completion LagunaComponentAdapter::LoadPrivate(
    cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes,
    cache::Stream&) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("Laguna load requires an unfailed restore");
  try {
    const auto& descriptor = Component(id);
    if (bytes.size() != descriptor.state_bytes)
      throw std::invalid_argument("incorrect Laguna private buffer");
    if (!decomposed_) {
      std::size_t length = 0;
      std::memcpy(&length, bytes.data() + bytes.size() - kLengthTrailerBytes,
                  sizeof(length));
      if (length < kSnapshotHeaderBytes ||
          length + kLengthTrailerBytes > bytes.size())
        throw std::invalid_argument("Laguna snapshot length is invalid");
      std::string error;
      const std::span<const std::uint8_t> payload{
          reinterpret_cast<const std::uint8_t*>(bytes.data()), length};
      if (!state.session().RestoreSnapshot(payload, &error))
        throw std::runtime_error("Laguna snapshot restore failed: " + error);
      if (state.session().Position() != state.restored_position)
        throw std::runtime_error("Laguna snapshot position does not match");
      state.private_loaded = true;
      return Completed(true);
    }
    if (!Info(id).private_state)
      throw std::logic_error("Laguna row component is not private");
    std::string error;
    const std::span<const std::uint8_t> input{
        reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()};
    if (!state.session().LoadPrivate(
            input, static_cast<std::uint32_t>(state.restored_position), &error))
      throw std::runtime_error("Laguna private restore failed: " + error);
    state.private_loaded = true;
    return Completed(true);
  } catch (...) {
    state.failed = true;
    return Completed(false);
  }
}

cache::Completion LagunaComponentAdapter::CopyRowsOut(
    const cache::Slot& slot, cache::ComponentId id, cache::Rows first,
    cache::Rows end, std::span<std::byte> bytes, cache::Stream&) {
  const auto& state = As(slot);
  if (!decomposed_ || !state.IsValid())
    throw std::logic_error("Laguna row capture requires a decomposed slot");
  const auto& info = Info(id);
  const auto& descriptor = Component(id);
  if (info.private_state || first > end || end > state.session().Position() ||
      bytes.size() != (end - first) * descriptor.row_bytes)
    throw std::invalid_argument("invalid Laguna row capture");
  std::string error;
  if (!state.session().SaveKvRows(
          info.layer, info.value_cache, static_cast<std::uint32_t>(first),
          static_cast<std::uint32_t>(end),
          std::span<std::uint8_t>(reinterpret_cast<std::uint8_t*>(bytes.data()),
                                  bytes.size()),
          &error))
    throw std::runtime_error("Laguna row capture failed: " + error);
  return Completed(true);
}

cache::Completion LagunaComponentAdapter::CopyRowsIn(
    cache::Slot& slot, cache::ComponentId id, cache::Rows first,
    cache::Rows end, std::span<const std::byte> bytes, cache::Stream&) {
  auto& state = As(slot);
  if (!decomposed_ || !state.restoring || state.failed)
    throw std::logic_error("Laguna row restore requires an unfailed restore");
  try {
    const auto& info = Info(id);
    const auto& descriptor = Component(id);
    if (info.private_state || first > end || end > state.restored_position ||
        bytes.size() != (end - first) * descriptor.row_bytes)
      throw std::invalid_argument("invalid Laguna row restore");
    std::string error;
    if (!state.session().LoadKvRows(
            info.layer, info.value_cache, static_cast<std::uint32_t>(first),
            static_cast<std::uint32_t>(end),
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(bytes.data()),
                bytes.size()),
            &error))
      throw std::runtime_error("Laguna row restore failed: " + error);
    return Completed(true);
  } catch (...) {
    state.failed = true;
    return Completed(false);
  }
}

bool LagunaComponentAdapter::Validate(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.restoring)
    throw std::logic_error("Laguna validation requires an active restore");
  const auto refuse = [&] {
    state.failed = true;
    return false;
  };
  if (state.failed || positions.size() != descriptors_.size() ||
      !state.private_loaded)
    return refuse();
  for (const auto& position : positions)
    if (position.valid_rows != state.restored_position ||
        std::ranges::count(descriptors_, position.id,
                           &cache::ComponentDescriptor::id) != 1)
      return refuse();
  if (state.session().Position() != state.restored_position)
    return refuse();
  state.restoring = false;
  return true;
}

bool LagunaComponentAdapter::Invalidate(cache::Slot& slot) noexcept {
  try {
    auto& state = As(slot);
    state.session().Reset();
    state.restoring = false;
    state.failed = false;
    state.private_loaded = false;
    state.restored_position = 0;
    state.guard.AfterReset();
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::laguna

#endif  // ENGINE_ENABLE_HIP
