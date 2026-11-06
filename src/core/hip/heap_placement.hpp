#ifndef GUFO_CORE_HIP_HEAP_PLACEMENT_HPP_
#define GUFO_CORE_HIP_HEAP_PLACEMENT_HPP_

#include <algorithm>
#include <cstdint>

namespace gufo::hip {

/// Keep this much free in the device heap for driver command submission.
inline constexpr std::uint64_t kDispatchReserve = 4ull << 30;

/// The dispatch reserve a heap with `free_bytes` free keeps back.
///
/// The reserve is the flat kDispatchReserve while the heap has room for it,
/// and a quarter of what is free when it does not. A flat reserve the heap
/// cannot satisfy rejects every size: on a 26 GiB APU whose model already
/// holds 21 GiB, every fit test failed and the context clamp collapsed to
/// its floor.
[[nodiscard]] inline std::uint64_t DispatchReserve(std::uint64_t free_bytes) {
  return std::min(kDispatchReserve, free_bytes / 4);
}

/// The bytes one allocation may take from a heap with `free_bytes` free.
[[nodiscard]] inline std::uint64_t HeapBudget(std::uint64_t free_bytes) {
  return free_bytes - DispatchReserve(free_bytes);
}

/// True when `bytes` can be taken from `free_bytes` and still leave the
/// dispatch reserve.
[[nodiscard]] inline bool HeapFitsAboveReserve(std::uint64_t free_bytes,
                                               std::uint64_t bytes) {
  return bytes <= HeapBudget(free_bytes);
}

enum class Placement { kMalloc, kManaged };

[[nodiscard]] inline Placement ChooseWeightPlacement(
    std::uint64_t free_bytes, std::uint64_t total_weight_bytes) {
  return free_bytes != 0 && HeapFitsAboveReserve(free_bytes, total_weight_bytes)
             ? Placement::kMalloc
             : Placement::kManaged;
}

}  // namespace gufo::hip

#endif  // GUFO_CORE_HIP_HEAP_PLACEMENT_HPP_
