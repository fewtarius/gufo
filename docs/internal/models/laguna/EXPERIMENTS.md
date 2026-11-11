# Laguna S 2.1 experiments

Retained decisions and the measurements that settled them.

## The NPU path was dropped

The Laguna engine carried an NPU offload for the shared-expert gate/up, worth
+0.2% to +2.4% depending on depth. Upstream removed the NPU stack entirely
(commit `f6d3bbb7`, "refactor: remove unused NPU stack and consolidate model
docs"), including the XCLBIN blobs the path needs. Re-vendoring that stack to
keep a 1% gain was not worth the maintenance, so the path was removed rather
than carried.

The two-bank prefill that existed only to hide NPU work behind GPU work went
with it. The plain prefill path is unchanged.

## Block-sparse attention is opt-in

`LAGUNA_SPARSE_ATTN` is presence-checked, so any value enables it and unset
leaves it off; the default is off. It is a prefill win that grows with depth:
694.6 tok/s dense to 958.0 tok/s sparse at 100,616 tokens (+37.9%), and 633.4 to
965.5 at 141,211 (+52%). It is approximate above about 18.5k tokens of context
and exact below it, and it applies to prefill only: a single-row decode
measured 14.9 tok/s dense against 11.6 with it on, because the block scoring is
paid once per split per K/V head.

Those figures come from `gufo prompt` with the whole document as the prompt, so
the prefill rate is a full-context pass. The 11 November 2026 tables in
[BENCHMARKS.md](BENCHMARKS.md) measure a 2,048-token continuation instead and
report lower prefill rates at the same depth; the gain table here has not been
re-measured against the current engine.

## The pooled block bounds travel in the snapshot

The bounds are derived from the keys, so restoring a session used to
re-derive them from the whole restored K/V cache. That is O(context) and it
dominated the restore: 290-672 ms at 158k tokens, against 75 ms with sparse
attention off. Carrying them in the snapshot cut the restore to 58-77 ms and
the per-turn time-to-first-token from 507-894 ms to 275 ms, with the same
output.

## Speculation runs for sampled requests

The server used to refuse to draft whenever the request sampled, so an agent
client that sends a temperature got plain decode and none of the documented
DFlash2 speed. The prompt CLI never had that gate. The server now uses the
CLI's rule: sample the drafter's proposals at the target's temperature and
accept a draft with probability `P_target(draft)`, resampling from the
residual on a miss. That rule is unbiased, and `sampling_test` checks it.

## The q8_0 KV cache is opt-in

It saves 6.2 GiB of device memory at a 262144-token context (-7%) and costs
3.7% of prefill and plain decode. With the drafter on it costs more, because
draft acceptance falls from 57% to 41%. F16 stays the default.

## The DFlash executor gained three methods

The upstream executor had no device-side target-context injection, no rewind,
and no persistent snapshot of the injected context. Laguna needs all three:
it captures the drafter's taps on the device, rewinds the drafter's frontier
with the target's K/V rewind, and carries the drafter's frontier in the
session snapshot. The methods were ported from the fork this work started
from and adapted to the upstream executor's internals.

## The upstream build had two link errors

`src/cli/extension_diagnostics.cpp` was moved out of the library that
`src/cli/diagnose/diagnose.cpp` lives in, so every target that linked the
diagnostics library failed to link. And the DeepSeek V4 Flash tests linked
the CPU tier while calling into the HIP tier. Both were fixed here.
