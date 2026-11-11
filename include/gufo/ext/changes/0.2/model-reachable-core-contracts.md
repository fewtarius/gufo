# Model-reachable core contracts

Publish three forwarding headers so a model or extension source reaches the
core contracts it already implements through the public surface instead of a
private `src/core` path:

- `include/gufo/ext/logging.hpp` forwards `src/core/logging/logging.hpp`
  (`gufo::server::Logger`).
- `include/gufo/ext/kv_codec.hpp` forwards `src/core/kv_codec/kv_q8_device.hpp`,
  the q8_0 K/V row codec a model stores its cache with.
- `include/gufo/ext/experimental/hip_profile.hpp` forwards
  `src/core/hip/profile.hpp`, the HIP profiling scopes a GPU provider wraps its
  kernels in.

The three sources are added to the `SURFACE` closure. They re-export code that
was already in the closure and add no new declaration, so the change is
additive and a model that previously included the private path keeps building.
