#ifndef GUFO_MODELS_LAGUNA_SETTLE_HPP_
#define GUFO_MODELS_LAGUNA_SETTLE_HPP_

#include <cstdint>

namespace gufo::models::laguna {

// A forward that returns non-finite logits on a unified-memory device has read
// memory the device could not back. Waiting before another session attempt
// helps only while something outside this process is still releasing device
// memory - a server that just exited, whose GTT the driver is still tearing
// down. Once the free pool stops growing, the bad state belongs to this
// process's own allocation and no amount of sleeping changes it.
//
// Measured on the Nimo Axis N161: six in-process attempts across four minutes
// failed identically, then a fresh process started two minutes later served
// first try. So the retry test is progress across the wait, not elapsed time.
inline constexpr std::uint64_t kDeviceSettleGrowthBytes = std::uint64_t{256}
                                                          << 20;

/// Whether the device released enough memory during a wait for another session
/// attempt to land somewhere different from the one that just failed.
///
/// Both arguments are free device bytes sampled around the wait. A pool that
/// shrank or held still means nothing outside this process is giving memory
/// back, so the caller should stop retrying and let its supervisor restart it,
/// which is what actually recovers.
[[nodiscard]] inline bool DeviceStillReleasing(
    std::uint64_t free_before, std::uint64_t free_after) noexcept {
  return free_after > free_before &&
         free_after - free_before >= kDeviceSettleGrowthBytes;
}

}  // namespace gufo::models::laguna

#endif  // GUFO_MODELS_LAGUNA_SETTLE_HPP_
