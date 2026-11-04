#ifndef GUFO_CORE_DIAGNOSTICS_GPU_ARCHITECTURE_H_
#define GUFO_CORE_DIAGNOSTICS_GPU_ARCHITECTURE_H_

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

namespace gufo::diagnostics {

// The AMD GPU architectures the HIP backend is built and qualified for.
//
// gfx1151 (Strix Halo) is the reference target. The gfx11 family (RDNA3,
// e.g. the Radeon 780M) and gfx12 also run the production kernels: the
// wave64-only translation units are compiled with -mwavefrontsize64, which
// RDNA3 accepts and executes.
//
// This list must stay in step with GUFO_SUPPORTED_HIP_ARCHITECTURES in
// CMakeLists.txt. A build for an architecture outside this list is
// rejected at configure time.
inline constexpr std::array<std::string_view, 11> kSupportedGpuArchitectures{
    "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151",
    "gfx1152", "gfx1153", "gfx1200", "gfx1201", "gfx1250"};

// Strip the target-feature suffix the driver appends to the architecture
// name ("gfx1030:xnack-" -> "gfx1030") and any whitespace.
[[nodiscard]] inline std::string_view NormalizeGpuArchitecture(
    std::string_view raw) {
  const auto colon = raw.find(':');
  if (colon != std::string_view::npos)
    raw = raw.substr(0, colon);
  while (!raw.empty() && (raw.back() == ' ' || raw.back() == '\t'))
    raw.remove_suffix(1);
  return raw;
}

[[nodiscard]] inline bool IsSupportedGpuArchitecture(std::string_view raw) {
  const auto arch = NormalizeGpuArchitecture(raw);
  for (const auto supported : kSupportedGpuArchitectures) {
    if (arch == supported)
      return true;
  }
  return false;
}

// The architecture a PCI device id belongs to, or an empty view when the id
// is not one gufo knows.
//
// A PCI id cannot always identify an architecture: AMD reuses ids across
// silicon. The HIP query is authoritative; this table only provides a
// fallback for the case where the HIP runtime is unavailable.
[[nodiscard]] inline std::string_view GpuArchitectureFromPciId(
    std::string_view pci_id) {
  struct Entry {
    std::string_view pci_id;
    std::string_view architecture;
  };
  static constexpr std::array<Entry, 22> kEntries{{
      // Strix Halo / Strix Point (gfx1151)
      {"1002:1586", "gfx1151"},
      {"1002:1587", "gfx1151"},
      {"1002:1588", "gfx1151"},
      // Phoenix (gfx1103)
      {"1002:15bf", "gfx1103"},
      {"1002:15be", "gfx1103"},
      {"1002:15c0", "gfx1103"},
      {"1002:15e0", "gfx1103"},
      {"1002:15e1", "gfx1103"},
      {"1002:15c8", "gfx1103"},
      {"1002:15c9", "gfx1103"},
      {"1002:15ca", "gfx1103"},
      {"1002:15cb", "gfx1103"},
      // Navi 31 / 32 / 33 (gfx1100 / gfx1101 / gfx1102)
      {"1002:1636", "gfx1100"},
      {"1002:1637", "gfx1100"},
      {"1002:1638", "gfx1100"},
      {"1002:1639", "gfx1100"},
      {"1002:164c", "gfx1100"},
      {"1002:164d", "gfx1101"},
      {"1002:164e", "gfx1101"},
      {"1002:164f", "gfx1101"},
      {"1002:1650", "gfx1102"},
      {"1002:1651", "gfx1102"},
  }};
  for (const auto& entry : kEntries) {
    if (entry.pci_id == pci_id)
      return entry.architecture;
  }
  return {};
}

}  // namespace gufo::diagnostics

#endif  // GUFO_CORE_DIAGNOSTICS_GPU_ARCHITECTURE_H_
