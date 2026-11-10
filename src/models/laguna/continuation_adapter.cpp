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

}  // namespace

cache::Identity LagunaProcessIdentity(const void* model,
                                      std::uint32_t max_context) {
  std::ostringstream identity;
  identity << "schema=gufo-process-laguna-component-v1\n"
           << "model_instance=" << model << '\n'
           << "context_tokens=" << max_context << '\n'
           << "state_abi=laguna-rocm-session-v1-component-v1\n"
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
  std::size_t restored_position{0};
};

LagunaComponentAdapter::LagunaComponentAdapter(
    std::shared_ptr<gpu::Model> model, std::uint32_t max_context,
    cache::Identity identity)
    : model_(std::move(model)),
      max_context_(max_context),
      identity_(std::move(identity)) {
  // The cache sizes one checkpoint buffer up front, so take the widest
  // snapshot this context can produce. SnapshotBytes() shrinks with the
  // position; MaxSnapshotBytes() is the position-independent bound.
  std::string error;
  auto probe =
      model_->CreateSession(max_context_ + Session::kMaxVerifyRows, &error);
  if (probe == nullptr) {
    throw std::runtime_error("cannot size the Laguna component cache: " +
                             error);
  }
  state_bytes_ = probe->MaxSnapshotBytes() + kLengthTrailerBytes;
  descriptors_.push_back(cache::ComponentDescriptor{
      .id = cache::ComponentId{1},
      .layout_version = 1,
      .kind = cache::ComponentKind::kPrivateState,
      .row_bytes = 0,
      .rows_per_chunk = 0,
      .state_bytes = state_bytes_,
  });
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
  if (id.value != 1 || descriptors_.empty()) {
    throw std::invalid_argument("invalid Laguna component ID");
  }
  return descriptors_.front();
}

gpu::Session& LagunaComponentAdapter::Session(cache::Slot& slot) const {
  return As(slot).session();
}

std::vector<cache::BackingClass> LagunaComponentAdapter::BackingPlan(
    std::size_t ram_bytes) const {
  // The committed backing pool requires page-aligned blocks and enough
  // metadata headroom for one state record plus a kilobyte per block. Keep
  // the block at or above the cache's own rounded component size.
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
  if (!state.IsValid()) {
    throw std::logic_error("cannot query an invalid Laguna slot");
  }
  return {{descriptors_.front().id, state.session().Position()}};
}

std::vector<cache::Rows> LagunaComponentAdapter::PlanPrefill(
    const cache::Slot& slot, cache::Rows first, cache::Rows end) const {
  const auto& state = As(slot);
  if (!state.IsValid() || first > end || end > max_context_) {
    throw std::invalid_argument("invalid Laguna prefill plan");
  }
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
  if (state.restoring) {
    throw std::logic_error("Laguna slot is already restoring");
  }
  if (positions.size() != 1 || positions.front().id.value != 1) {
    throw std::invalid_argument("incomplete Laguna component position");
  }
  state.restoring = true;
  state.failed = false;
  state.restored_position = positions.front().valid_rows;
}

cache::Completion LagunaComponentAdapter::CapturePrivate(
    const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes,
    cache::Stream&) {
  const auto& state = As(slot);
  const auto& descriptor = Component(id);
  if (bytes.size() != descriptor.state_bytes) {
    throw std::invalid_argument("incorrect Laguna private buffer");
  }
  if (!state.IsValid()) {
    throw std::logic_error("Laguna capture requires a valid slot");
  }
  auto& session = state.session();
  const std::size_t length = session.SnapshotBytes();
  if (length == 0) {
    // The cache never asks for an empty prefix; a zero here means the slot
    // was reset while the capture was being prepared.
    return Completed(false);
  }
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

cache::Completion LagunaComponentAdapter::LoadPrivate(
    cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes,
    cache::Stream&) {
  auto& state = As(slot);
  if (!state.restoring || state.failed) {
    throw std::logic_error("Laguna load requires an unfailed restore");
  }
  try {
    const auto& descriptor = Component(id);
    if (bytes.size() != descriptor.state_bytes) {
      throw std::invalid_argument("incorrect Laguna private buffer");
    }
    std::size_t length = 0;
    std::memcpy(&length, bytes.data() + bytes.size() - kLengthTrailerBytes,
                sizeof(length));
    if (length < kSnapshotHeaderBytes ||
        length + kLengthTrailerBytes > bytes.size()) {
      throw std::invalid_argument("Laguna snapshot length is invalid");
    }
    std::string error;
    const std::span<const std::uint8_t> payload{
        reinterpret_cast<const std::uint8_t*>(bytes.data()), length};
    if (!state.session().RestoreSnapshot(payload, &error)) {
      throw std::runtime_error("Laguna snapshot restore failed: " + error);
    }
    if (state.session().Position() != state.restored_position) {
      throw std::runtime_error("Laguna snapshot position does not match");
    }
    return Completed(true);
  } catch (...) {
    state.failed = true;
    return Completed(false);
  }
}

cache::Completion LagunaComponentAdapter::CopyRowsOut(const cache::Slot&,
                                                      cache::ComponentId,
                                                      cache::Rows, cache::Rows,
                                                      std::span<std::byte>,
                                                      cache::Stream&) {
  throw std::logic_error("Laguna has no append-row components");
}

cache::Completion LagunaComponentAdapter::CopyRowsIn(cache::Slot&,
                                                     cache::ComponentId,
                                                     cache::Rows, cache::Rows,
                                                     std::span<const std::byte>,
                                                     cache::Stream&) {
  throw std::logic_error("Laguna has no append-row components");
}

bool LagunaComponentAdapter::Validate(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.restoring) {
    throw std::logic_error("Laguna validation requires an active restore");
  }
  const bool ok = !state.failed && positions.size() == 1 &&
                  positions.front().id.value == 1 &&
                  positions.front().valid_rows == state.restored_position &&
                  state.session().Position() == state.restored_position;
  if (!ok) {
    state.failed = true;
    return false;
  }
  state.restoring = false;
  return true;
}

bool LagunaComponentAdapter::Invalidate(cache::Slot& slot) noexcept {
  try {
    auto& state = As(slot);
    state.session().Reset();
    state.restoring = false;
    state.failed = false;
    state.restored_position = 0;
    state.guard.AfterReset();
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::laguna

#endif  // ENGINE_ENABLE_HIP
