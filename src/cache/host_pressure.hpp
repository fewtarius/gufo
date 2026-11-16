#ifndef GUFO_CACHE_HOST_PRESSURE_HPP_
#define GUFO_CACHE_HOST_PRESSURE_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>

namespace gufo::cache {

// Hysteresis for host-pressure trimming. Pressure only acts once the cache is
// over its affordable ceiling by at least this much, so a sample landing close
// to the boundary cannot make the cache give up a checkpoint it wants again on
// the next turn. Matches the drift allowance the budget contract test already
// permits between two reads of the same host.
inline constexpr std::size_t kHostPressureMarginBytes = std::size_t{256} << 20;

struct HostPressureVerdict {
  // Ceiling a byte-pressure decision should apply. Never above the configured
  // ceiling, so a sample can only ever tighten the budget.
  std::size_t ram_ceiling_bytes{};
  // The cache holds at least kHostPressureMarginBytes over the ceiling, so the
  // owner should reclaim before committing more payload.
  bool should_trim{false};
};

// Ceiling implied by one host sample.
//
// available_bytes nullopt means the host could not be read. That must not
// collapse the budget: the configured ceiling comes back and no trim is asked
// for, because a failed read is evidence about the reader, not about pressure.
//
// A host at or below floor_bytes affords the cache nothing new. Callers that
// must keep checkpoints alive at all costs clamp the ceiling themselves; this
// function reports what the host says, not what the caller hopes to hear.
[[nodiscard]] HostPressureVerdict HostPressure(
    std::size_t configured_ram_bytes,
    std::optional<std::uint64_t> available_bytes, std::uint64_t floor_bytes,
    std::size_t held_bytes) noexcept;

// Whether a pressure trim should evict one more record.
//
// Two conditions, and the first is the one that matters. A cache holding a
// single record still serves reuse of that prefix; evicting it trades that for
// a guaranteed re-prefill of the whole prefix next turn at the same memory
// cost, so trimming below one record buys nothing and can repeat every request.
// That is not hypothetical: a workload whose single checkpoint is larger than
// the ceiling evicted and re-captured 48 GiB of pinned host backing in thirteen
// minutes while prefill was running. Host protection still applies through the
// ceiling, which gates what new admissions may add.
[[nodiscard]] bool ShouldTrimRecord(std::size_t records, std::size_t held_bytes,
                                    std::size_t ceiling_bytes) noexcept;

}  // namespace gufo::cache

#endif  // GUFO_CACHE_HOST_PRESSURE_HPP_
