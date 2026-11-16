#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string_view>

namespace gufo::ext {

/// MemAvailable from /proc/meminfo text minus CmaFree, in bytes. Free CMA
/// pages count as available but only hold movable pages, not GPU
/// allocations; the kernel's KHO scratch area can make that several GiB.
/// Nullopt when MemAvailable is missing.
[[nodiscard]] std::optional<std::uint64_t> MeminfoAvailableBytes(
    std::string_view meminfo);
/// `host_available` bounded by every cgroup v2 memory limit from the process's
/// cgroup (named in `membership_text`, the /proc/self/cgroup contents) up to
/// `root`. Inactive file pages in memory.stat are reclaimable page cache and
/// are not counted as used; without that line, memory.current is used as is.
[[nodiscard]] std::uint64_t CgroupAvailableBytes(
    const std::filesystem::path& root, std::string_view membership_text,
    std::uint64_t host_available);
/// Automatic snapshot budget: half the host RAM available after loading
/// (excluding free CMA pages), after cgroup limits. Read once, where the cache
/// is constructed; it is not re-measured as the session consumes RAM.
[[nodiscard]] std::size_t HostSnapshotBudgetBytes();
/// RAM an explicit --cache-ram-bytes leaves to the host's other users: the
/// kernel, the desktop and every other process's resident memory.
///
/// Not a device reservation, and no device path needs one. A firmware carveout
/// is absent from MemTotal, so it never competes for the pool this protects.
/// GTT pages are charged against MemAvailable as the driver faults them in, so
/// an availability figure already nets out what the device holds at the moment
/// it is read. What remains at risk is growth after the read: on a unified
/// memory host the window and this budget are caps on overlapping bytes, and
/// neither is sized from the other.
inline constexpr std::uint64_t kHostSnapshotHeadroomBytes = std::uint64_t{4}
                                                            << 30;
/// Most an explicit --cache-ram-bytes may claim: available host RAM minus
/// kHostSnapshotHeadroomBytes.
[[nodiscard]] std::size_t HostSnapshotCeilingBytes();

}  // namespace gufo::ext
