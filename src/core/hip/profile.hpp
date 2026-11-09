#ifndef GUFO_CORE_HIP_PROFILE_HPP_
#define GUFO_CORE_HIP_PROFILE_HPP_

// GPU-event sub-stage timing, shared by the layer loop and the feed-forward so
// both can be broken down with the same tool and the same output shape.

#if defined(ENGINE_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace gufo::hip {

/// Sub-stage timing for one branch of the layer loop.
///
/// The chunk clock splits a layer into attention and feed-forward. That is too
/// coarse to act on when the attention half is a mix of projections and a
/// convolution-plus-recurrence: they are different kinds of work and they move
/// independently. This accumulates named stages across the layers that have
/// them, so the branch can be attacked in order of cost rather than by guess.
///
/// Stage boundaries are events on the arena stream, so consecutive events bound
/// a stage regardless of what else is queued between them.
class SubStageClock {
public:
  explicit SubStageClock(const char* variable) {
    const char* const value = std::getenv(variable);
    on_ = value != nullptr && *value != '0';
  }

  void SetStream(hipStream_t stream) { stream_ = stream; }
  void AddStage(const char* name) { names_.emplace_back(name); }

  /// Records a stage boundary. One call starts the branch and one closes each
  /// named stage, so a layer contributes names_.size() + 1 events.
  void Mark() {
    if (!on_) {
      return;
    }
    hipEvent_t event{};
    if (hipEventCreate(&event) != hipSuccess ||
        hipEventRecord(event, stream_) != hipSuccess) {
      on_ = false;
      return;
    }
    marks_.push_back(event);
  }

  /// Drops marks recorded for a forward that will not be reported, so a
  /// graph-replayed step that never reaches the report does not accumulate
  /// events across an entire generation.
  void Discard() {
    for (hipEvent_t event : marks_) {
      (void)hipEventDestroy(event);
    }
    marks_.clear();
  }

  /// Buckets marks into layers only when their count divides evenly, so a layer
  /// that skipped a stage cannot silently shift every later measurement.
  void Report(const char* label) {
    if (!on_ || names_.empty()) {
      return;
    }
    const std::size_t per_layer = names_.size() + 1;
    if (marks_.size() < per_layer || marks_.size() % per_layer != 0) {
      std::fprintf(stderr,
                   "[%s] skipped: %zu events is not a multiple of %zu\n", label,
                   marks_.size(), per_layer);
      std::fflush(stderr);
      for (hipEvent_t event : marks_) {
        (void)hipEventDestroy(event);
      }
      marks_.clear();
      return;
    }
    if (hipStreamSynchronize(stream_) != hipSuccess) {
      return;
    }
    std::vector<double> totals(names_.size(), 0.0);
    const std::size_t layers = marks_.size() / per_layer;
    for (std::size_t layer = 0; layer < layers; ++layer) {
      for (std::size_t stage = 0; stage < names_.size(); ++stage) {
        const std::size_t base = layer * per_layer + stage;
        float elapsed = 0.0F;
        (void)hipEventElapsedTime(&elapsed, marks_[base], marks_[base + 1]);
        totals[stage] += elapsed;
      }
    }
    for (hipEvent_t event : marks_) {
      (void)hipEventDestroy(event);
    }
    marks_.clear();
    double all = 0.0;
    for (const double total : totals) {
      all += total;
    }
    for (std::size_t stage = 0; stage < names_.size(); ++stage) {
      std::fprintf(
          stderr, "[%s] %-12s %8.1f ms  %5.1f%%  %6.2f ms/layer\n", label,
          names_[stage], totals[stage],
          all > 0.0 ? 100.0 * totals[stage] / all : 0.0,
          layers > 0 ? totals[stage] / static_cast<double>(layers) : 0.0);
    }
    std::fprintf(stderr, "[%s] %-12s %8.1f ms  over %zu layers\n", label,
                 "total", all, layers);
    std::fflush(stderr);
  }

private:
  bool on_ = false;
  hipStream_t stream_ = nullptr;
  std::vector<const char*> names_;
  std::vector<hipEvent_t> marks_;
};

}  // namespace gufo::hip

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_CORE_HIP_PROFILE_HPP_
