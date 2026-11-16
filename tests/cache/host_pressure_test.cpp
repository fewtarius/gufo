#include "src/cache/host_pressure.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "src/cache/retention.hpp"
#include "tests/cache/prefix_index_fixture.hpp"

using namespace gufo::cache;
using namespace gufo::cache::testing;
namespace {

constexpr std::size_t kGiB = std::size_t{1} << 30;
constexpr std::size_t kMiB = std::size_t{1} << 20;

void UnreadableHostLeavesTheBudgetAlone() {
  // A failed read is evidence about the reader. Trimming a cache because
  // /proc could not be opened is the failure mode this path must avoid.
  const auto verdict = HostPressure(8 * kGiB, std::nullopt, 4 * kGiB, 0);
  assert(verdict.ram_ceiling_bytes == 8 * kGiB);
  assert(!verdict.should_trim);
}

void ComfortableHostKeepsTheConfiguredCeiling() {
  const auto verdict =
      HostPressure(8 * kGiB, std::uint64_t{64} * kGiB, 4 * kGiB, 0);
  assert(verdict.ram_ceiling_bytes == 8 * kGiB);
  assert(!verdict.should_trim);
}

void TightHostTightensTheCeiling() {
  const auto verdict =
      HostPressure(8 * kGiB, std::uint64_t{10} * kGiB, 4 * kGiB, 0);
  assert(verdict.ram_ceiling_bytes == 6 * kGiB);
  assert(!verdict.should_trim);
}

void FloorExhaustedAffordsNothing() {
  assert(HostPressure(8 * kGiB, std::uint64_t{4} * kGiB, 4 * kGiB, 0)
             .ram_ceiling_bytes == 0);
  assert(HostPressure(8 * kGiB, std::uint64_t{3} * kGiB, 4 * kGiB, 0)
             .ram_ceiling_bytes == 0);
  // Only the overshoot test asks for a trim, and a cache holding nothing is
  // not trimming however short the host is.
  assert(!HostPressure(8 * kGiB, std::uint64_t{1} * kMiB, 4 * kGiB, 0)
              .should_trim);
}

void CeilingNeverRisesAboveTheConfiguredBudget() {
  const auto verdict =
      HostPressure(1 * kGiB, std::uint64_t{64} * kGiB, 4 * kGiB, 0);
  assert(verdict.ram_ceiling_bytes == 1 * kGiB);
}

void TrimNeedsAWholeMarginOfOvershoot() {
  // 12 GiB available with a 4 GiB floor affords exactly the configured 8 GiB.
  const auto sample = std::uint64_t{12} * kGiB;
  assert(HostPressure(8 * kGiB, sample, 4 * kGiB, 0).ram_ceiling_bytes ==
         8 * kGiB);
  // One byte short of the margin is noise, and reacting to it would evict a
  // checkpoint the next turn wants back.
  assert(!HostPressure(8 * kGiB, sample, 4 * kGiB,
                       8 * kGiB + kHostPressureMarginBytes - 1)
              .should_trim);
  assert(HostPressure(8 * kGiB, sample, 4 * kGiB,
                      8 * kGiB + kHostPressureMarginBytes)
             .should_trim);
  assert(HostPressure(8 * kGiB, sample, 4 * kGiB, 16 * kGiB).should_trim);
  assert(!HostPressure(8 * kGiB, sample, 4 * kGiB, 8 * kGiB).should_trim);
}

void OvershootTestDoesNotWrap() {
  // Comparing down from held rather than up from the ceiling keeps a held
  // value near SIZE_MAX from flipping the margin test.
  const auto verdict = HostPressure(8 * kGiB, std::uint64_t{4} * kGiB, 4 * kGiB,
                                    std::numeric_limits<std::size_t>::max());
  assert(verdict.ram_ceiling_bytes == 0);
  assert(verdict.should_trim);
}

void ZeroFloorAffordsAllAvailability() {
  const auto verdict = HostPressure(8 * kGiB, std::uint64_t{2} * kGiB, 0, 0);
  assert(verdict.ram_ceiling_bytes == 2 * kGiB);
}

struct PressureFixture {
  ResourceLedger ledger{kIndexLimits};
  PrefixIndex index{ledger};
  RetentionPolicy policy;
  PressureFixture() : policy(ledger, index) {
    index.Register({1}, kIndexComponents);
  }
  bool Offer(const std::vector<Token>& tokens, std::size_t payload) {
    RetentionRequest request;
    request.tokens = tokens;
    request.compatibility = {1};
    request.new_payload_bytes = payload;
    return policy.Admit(request,
                        [&] { return IndexCheckpoint(ledger, tokens, {1}); });
  }
};

void RamBudgetTakesTheMinimum() {
  PressureFixture fixture;
  assert(fixture.policy.RamBudget() == kIndexLimits.ram_bytes);
  fixture.policy.SetHostRamCeiling(1024);
  assert(fixture.policy.RamBudget() == 1024);
  // Clearing restores the ledger's own limit rather than leaving the last
  // sample in force, which is what an unreadable host must do.
  fixture.policy.SetHostRamCeiling(std::nullopt);
  assert(fixture.policy.RamBudget() == kIndexLimits.ram_bytes);
  // A ceiling above the ledger limit is not a relaxation.
  fixture.policy.SetHostRamCeiling(kIndexLimits.ram_bytes * 2);
  assert(fixture.policy.RamBudget() == kIndexLimits.ram_bytes);
}

void CeilingGatesAdmission() {
  PressureFixture fixture;
  // The ledger allows 4 GiB per category here, so this payload is admitted
  // freely until a host ceiling says otherwise.
  assert(fixture.Offer({1, 2, 3, 4}, 64));
  fixture.policy.SetHostRamCeiling(16);
  assert(!fixture.Offer({5, 6, 7, 8}, 64));
  fixture.policy.SetHostRamCeiling(std::nullopt);
  assert(fixture.Offer({9, 10, 11, 12}, 64));
}

void CeilingGatesReclaim() {
  PressureFixture fixture;
  RetentionRequest request;
  request.compatibility = {1};
  request.new_payload_bytes = 64;
  fixture.policy.SetHostRamCeiling(16);
  // The host cannot afford the payload, so reclaiming a victim for it would
  // evict a checkpoint for nothing.
  assert(!fixture.policy.Reclaim(request));
}

}  // namespace

int main() {
  UnreadableHostLeavesTheBudgetAlone();
  ComfortableHostKeepsTheConfiguredCeiling();
  TightHostTightensTheCeiling();
  FloorExhaustedAffordsNothing();
  CeilingNeverRisesAboveTheConfiguredBudget();
  TrimNeedsAWholeMarginOfOvershoot();
  OvershootTestDoesNotWrap();
  ZeroFloorAffordsAllAvailability();
  RamBudgetTakesTheMinimum();
  CeilingGatesAdmission();
  CeilingGatesReclaim();
  return 0;
}
