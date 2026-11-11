# Qwen3.8 Flash-Next benchmarks

Hybrid attention with per-layer embeddings, 103.7 GiB. Unsloth UD-Q4_K_XL,
sharded over four files. Greedy decode, temperature 0, thinking off, one
session. Measured on Gufo `0e7b3333`. Method and protocol:
[benchmark methodology](../../benchmarks.md).

## Nimo Axis N161

Strix Halo: Ryzen AI Max+ 395, Radeon 8060S `gfx1151`, 40 CUs, 124.9 GiB RAM,
114,688 MiB pool. Context 133,760, the largest that runs, with F16 K/V.

The model's native window is 262,144 and it does not survive it: the server
loads, then aborts on its first real request with
`HSA_STATUS_ERROR_INVALID_PACKET_FORMAT: The AQL packet is malformed` and
`fatal_signal signal=6`. Loading at 262,144 reports 87,431 MiB of the pool, so
this is not an allocation failure. The same weights and binary at 133,760
complete a full sweep without error, which is what the tables below report.

Rows here are a single measurement rather than the two the other pages report,
so they carry no deviation. A second server launch for this model aborted
before finishing its first request, which is the same failure seen at native
context, so the sweep runs once per depth.

<!-- bench:single-ar -->
| Qwen3.8-Flash-Next AR<br>Depth (tokens) | pp (tok/s) | tg (tok/s) |
| ---: | ---: | ---: |
| 0 | 1753.95 | 29.25 |
| 8,192 | 1671.08 | 27.70 |
| 32,768 | 1490.71 | 28.58 |
| 65,536 | 1571.97 | 29.26 |
| 131,072 | 1504.27 | 29.07 |
<!-- /bench -->

<!-- bench:loading -->
| Qwen3.8-Flash-Next<br>Target | ready (s) |
| --- | ---: |
| Q4 | 51.02 |
<!-- /bench -->

<!-- bench:memory -->
| Qwen3.8-Flash-Next AR<br>Workload | GiB |
| --- | ---: |
| pp2048 + tg128 | 85.51 |
| 32K prefix, pp4096 + tg128 | 85.64 |
<!-- /bench -->

## AYANEO Flip KB

The model file is not installed on the handheld, and 103.7 GiB of weights would
not fit its 20,480 MiB pool in any case.

## Not measured in this pass

Speculative decoding with the MTP sidecar. The shipped head is bound by the
loader as of `799a9a48`, but no MTP sweep has been run on these weights since
the move to the Unsloth Dynamic 3.1 set, so this page reports autoregressive
throughput only.

Context beyond 133,760. See above.
