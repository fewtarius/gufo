#include "src/models/laguna/settle.hpp"

#include <cassert>
#include <cstdint>

using gufo::models::laguna::DeviceStillReleasing;
using gufo::models::laguna::kDeviceSettleGrowthBytes;

namespace {

constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;
constexpr std::uint64_t kMiB = std::uint64_t{1} << 20;

// The regression this guards is a retry that sleeps for nothing. On the Nimo
// Axis N161 six in-process attempts across four minutes failed identically
// while a fresh process served first try, so elapsed time is not the signal.

void AnUnchangedPoolDoesNotJustifyAnotherAttempt() {
  // The device released its memory long before this start. Nothing is coming
  // back, so a second attempt allocates exactly what the first one did.
  assert(!DeviceStillReleasing(20 * kGiB, 20 * kGiB));
  // A pool that went backwards is worse, not better.
  assert(!DeviceStillReleasing(20 * kGiB, 19 * kGiB));
}

void AReleasingPoolJustifiesAnotherAttempt() {
  // A server just exited and its ~109 GiB is still being torn down.
  assert(DeviceStillReleasing(20 * kGiB, 40 * kGiB));
  // Exactly the threshold counts.
  assert(DeviceStillReleasing(20 * kGiB, 20 * kGiB + kDeviceSettleGrowthBytes));
  // One byte short is noise between two samples of the same pool.
  assert(!DeviceStillReleasing(20 * kGiB,
                               20 * kGiB + kDeviceSettleGrowthBytes - 1));
  assert(!DeviceStillReleasing(20 * kGiB, 20 * kGiB + 1 * kMiB));
}

void AnUnreadablePoolFailsFast() {
  // FreeDeviceBytes reports 0 when it cannot ask the driver. With no evidence
  // that anything is being released, sleeping is the one thing not to do.
  assert(!DeviceStillReleasing(0, 0));
  assert(!DeviceStillReleasing(0, 100 * kMiB));
  // And a pool that recovers from an unreadable sample is genuine progress.
  assert(DeviceStillReleasing(0, 1 * kGiB));
}

void AGrowingPoolStillCountsWhenItStartedLarge() {
  // Size is irrelevant; only the change across the wait matters.
  assert(DeviceStillReleasing(100 * kGiB, 101 * kGiB));
  assert(!DeviceStillReleasing(100 * kGiB, 100 * kGiB));
}

}  // namespace

int main() {
  AnUnchangedPoolDoesNotJustifyAnotherAttempt();
  AReleasingPoolJustifiesAnotherAttempt();
  AnUnreadablePoolFailsFast();
  AGrowingPoolStillCountsWhenItStartedLarge();
  return 0;
}
