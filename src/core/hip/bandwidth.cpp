#include "src/core/diagnostics/bandwidth.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

#include "src/core/diagnostics/bandwidth_run.hpp"

#ifdef ENGINE_ENABLE_HIP
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#endif

namespace gufo::diagnostics {

#ifdef ENGINE_ENABLE_HIP

namespace {

void ComputeStats(BandwidthPathResult& result) {
  if (result.raw_repetitions_gbps.empty()) {
    return;
  }
  std::vector<double> sorted = result.raw_repetitions_gbps;
  std::ranges::sort(sorted);

  result.min_gbps = sorted.front();
  result.max_gbps = sorted.back();

  const std::size_t n = sorted.size();
  if (n % 2 == 0) {
    result.median_gbps = (sorted[(n / 2) - 1] + sorted[n / 2]) / 2.0;
  } else {
    result.median_gbps = sorted[n / 2];
  }

  const std::size_t p95_idx =
      static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(n))) - 1;
  result.p95_gbps = sorted[std::min(p95_idx, n - 1)];
}

std::string HipReason(hipError_t status) {
  return std::string(hipGetErrorName(status)) + ": " +
         hipGetErrorString(status);
}

BandwidthPathResult UnavailablePath(std::string path_name, std::string reason) {
  BandwidthPathResult path;
  path.backend = "hip";
  path.path_name = std::move(path_name);
  path.allocation_type = "hip_device_memory";
  path.status = "unavailable";
  path.reason = std::move(reason);
  return path;
}

// Frees a device allocation when the scope ends. A cleanup failure is
// reported and not thrown: the benchmark already has its result.
class DeviceAllocation {
public:
  DeviceAllocation() = default;
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  ~DeviceAllocation() {
    if (pointer != nullptr) {
      hip::LogCleanupError(hipFree(pointer));
    }
  }

  hipError_t Allocate(std::size_t bytes) {
    pointer = nullptr;
    return hipMalloc(&pointer, bytes);
  }

  void* Pointer() const noexcept { return pointer; }

private:
  void* pointer{nullptr};
};

// Releases a HIP event when the scope ends.
class DeviceEvent {
public:
  DeviceEvent() = default;
  DeviceEvent(const DeviceEvent&) = delete;
  DeviceEvent& operator=(const DeviceEvent&) = delete;
  ~DeviceEvent() {
    if (event != nullptr) {
      hip::LogCleanupError(hipEventDestroy(event));
    }
  }

  hipError_t Create() {
    event = nullptr;
    return hipEventCreate(&event);
  }

  hipEvent_t Event() const noexcept { return event; }

private:
  hipEvent_t event{nullptr};
};

// Runs one measured transfer path. The first HIP failure ends the path and
// is reported instead of leaving the timings unqualified.
template<typename Prepare, typename Copy, typename Verify>
hipError_t MeasureTransfer(const BandwidthOptions& options,
                           BandwidthPathResult& path, hipEvent_t start_event,
                           hipEvent_t stop_event, double bytes_per_run,
                           Prepare&& prepare, Copy&& copy, Verify&& verify) {
  for (std::uint32_t w = 0; w < options.warmup; ++w) {
    const hipError_t status = copy();
    if (status != hipSuccess)
      return status;
  }
  for (detail::BandwidthRun run(options, path); run.ShouldContinue();) {
    prepare();
    const hipError_t record = hipEventRecord(start_event, nullptr);
    if (record != hipSuccess)
      return record;
    const hipError_t copy_status = copy();
    if (copy_status != hipSuccess)
      return copy_status;
    const hipError_t stop = hipEventRecord(stop_event, nullptr);
    if (stop != hipSuccess)
      return stop;
    const hipError_t sync = hipEventSynchronize(stop_event);
    if (sync != hipSuccess)
      return sync;
    float elapsed_ms = 0.0F;
    const hipError_t elapsed =
        hipEventElapsedTime(&elapsed_ms, start_event, stop_event);
    if (elapsed != hipSuccess)
      return elapsed;
    const double elapsed_sec = static_cast<double>(elapsed_ms) / 1000.0;
    run.Record(elapsed_sec, bytes_per_run);
    verify();
  }
  return hipSuccess;
}

}  // namespace

std::vector<BandwidthPathResult> MeasureHipBandwidth(
    const BandwidthOptions& options) {
  std::vector<BandwidthPathResult> results;

  int device_count = 0;
  hipError_t err = hipGetDeviceCount(&device_count);
  if (err != hipSuccess || device_count <= 0) {
    BandwidthPathResult unavailable_path;
    unavailable_path.backend = "hip";
    unavailable_path.path_name = "hip_device_visible";
    unavailable_path.allocation_type = "hip_device_memory";
    unavailable_path.status = "unavailable";
    unavailable_path.reason =
        (err != hipSuccess)
            ? std::string("hipGetDeviceCount failed: ") + hipGetErrorName(err)
            : "No HIP GPU device found";
    results.push_back(std::move(unavailable_path));
    return results;
  }

  const std::size_t size = options.working_set_bytes;
  std::vector<std::uint8_t> h_src(size);
  std::vector<std::uint8_t> h_dst(size);

  for (std::size_t i = 0; i < size; ++i) {
    h_src[i] = static_cast<std::uint8_t>((i * 31 + 17) & 0xFF);
  }

  DeviceAllocation d_src;
  DeviceAllocation d_dst;
  const hipError_t alloc_src = d_src.Allocate(size);
  const hipError_t alloc_dst =
      alloc_src == hipSuccess ? d_dst.Allocate(size) : alloc_src;
  if (alloc_src != hipSuccess || alloc_dst != hipSuccess) {
    results.push_back(UnavailablePath(
        "hip_device_memory",
        "hipMalloc failed: " +
            HipReason(alloc_src != hipSuccess ? alloc_src : alloc_dst)));
    return results;
  }

  DeviceEvent start_event;
  DeviceEvent stop_event;
  const hipError_t event_status = start_event.Create();
  if (event_status != hipSuccess) {
    results.push_back(
        UnavailablePath("hip_device_memory",
                        "hipEventCreate failed: " + HipReason(event_status)));
    return results;
  }
  const hipError_t stop_event_status = stop_event.Create();
  if (stop_event_status != hipSuccess) {
    results.push_back(UnavailablePath(
        "hip_device_memory",
        "hipEventCreate failed: " + HipReason(stop_event_status)));
    return results;
  }

  // 1. Host-to-Device (H2D)
  {
    BandwidthPathResult path;
    path.backend = "hip";
    path.path_name = "hip_h2d";
    path.allocation_type = "hip_device_memory";
    path.working_set_bytes = size;
    path.warmup_runs = options.warmup;

    const hipError_t status = MeasureTransfer(
        options, path, start_event.Event(), stop_event.Event(),
        static_cast<double>(size), [] {},
        [&] {
          return hipMemcpy(d_src.Pointer(), h_src.data(), size,
                           hipMemcpyHostToDevice);
        },
        [] {});
    if (status != hipSuccess) {
      path.status = "unavailable";
      path.reason = HipReason(status);
      path.sentinel_verified = false;
    } else {
      path.sentinel_verified = true;
    }
    ComputeStats(path);
    results.push_back(std::move(path));
  }

  // 2. Device-to-Device (D2D)
  {
    BandwidthPathResult path;
    path.backend = "hip";
    path.path_name = "hip_device_copy";
    path.allocation_type = "hip_device_memory";
    path.working_set_bytes = size;
    path.warmup_runs = options.warmup;

    // D2D reads size bytes and writes size bytes = 2 * size
    const hipError_t status = MeasureTransfer(
        options, path, start_event.Event(), stop_event.Event(),
        static_cast<double>(size) * 2, [] {},
        [&] {
          return hipMemcpy(d_dst.Pointer(), d_src.Pointer(), size,
                           hipMemcpyDeviceToDevice);
        },
        [] {});
    if (status != hipSuccess) {
      path.status = "unavailable";
      path.reason = HipReason(status);
      path.sentinel_verified = false;
    } else {
      path.sentinel_verified = true;
    }
    ComputeStats(path);
    results.push_back(std::move(path));
  }

  // 3. Device-to-Host (D2H)
  {
    BandwidthPathResult path;
    path.backend = "hip";
    path.path_name = "hip_d2h";
    path.allocation_type = "hip_device_memory";
    path.working_set_bytes = size;
    path.warmup_runs = options.warmup;

    bool sentinel_ok = true;
    const hipError_t status = MeasureTransfer(
        options, path, start_event.Event(), stop_event.Event(),
        static_cast<double>(size),
        [&] { std::fill(h_dst.begin(), h_dst.end(), 0); },
        [&] {
          return hipMemcpy(h_dst.data(), d_dst.Pointer(), size,
                           hipMemcpyDeviceToHost);
        },
        [&] {
          if (h_dst[0] != h_src[0] || h_dst[size / 2] != h_src[size / 2] ||
              h_dst[size - 1] != h_src[size - 1]) {
            sentinel_ok = false;
          }
        });
    if (status != hipSuccess) {
      path.status = "unavailable";
      path.reason = HipReason(status);
      path.sentinel_verified = false;
    } else {
      path.sentinel_verified = sentinel_ok;
    }
    ComputeStats(path);
    results.push_back(std::move(path));
  }

  return results;
}

#else

std::vector<BandwidthPathResult> MeasureHipBandwidth(
    const BandwidthOptions& /*options*/) {
  BandwidthPathResult uncompiled;
  uncompiled.backend = "hip";
  uncompiled.path_name = "hip_device_visible";
  uncompiled.allocation_type = "hip_device_memory";
  uncompiled.status = "unavailable";
  uncompiled.reason = "uncompiled (ENGINE_ENABLE_HIP=OFF)";
  return {uncompiled};
}

#endif

}  // namespace gufo::diagnostics
