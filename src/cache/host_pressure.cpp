#include "src/cache/host_pressure.hpp"

#include <algorithm>

namespace gufo::cache {

HostPressureVerdict HostPressure(std::size_t configured_ram_bytes,
                                 std::optional<std::uint64_t> available_bytes,
                                 std::uint64_t floor_bytes,
                                 std::size_t held_bytes) noexcept {
  if (!available_bytes)
    return {configured_ram_bytes, false};
  // Subtraction rather than comparison against available: an availability at
  // or below the floor affords the cache nothing, and the difference can never
  // overflow because it is only taken when positive.
  const std::size_t affordable =
      *available_bytes > floor_bytes
          ? static_cast<std::size_t>(*available_bytes - floor_bytes)
          : 0;
  const std::size_t ceiling = std::min(configured_ram_bytes, affordable);
  // Compare down from held instead of up from ceiling so a ceiling near
  // SIZE_MAX cannot wrap the margin test into a false trim.
  const bool trim =
      held_bytes > ceiling && held_bytes - ceiling >= kHostPressureMarginBytes;
  return {ceiling, trim};
}

}  // namespace gufo::cache
