# Qwen3.8 Flash-Next benchmarks

Hybrid attention with per-layer embeddings, 103.7 GiB. Unsloth UD-Q4_K_XL,
sharded over four files. Greedy decode, temperature 0, thinking off, one
session. Measured on Gufo `0e7b3333`. Method and protocol:
[benchmark methodology](../../benchmarks.md).

## Nimo Axis N161

Strix Halo: Ryzen AI Max+ 395, Radeon 8060S `gfx1151`, 40 CUs, 124.9 GiB RAM,
114,688 MiB pool. Context 262,144, the model's native window, with F16 K/V.
Loading reports 87,431 MiB of the pool.

Three consecutive sweeps at this context completed without error, so native
context is usable. One earlier launch aborted with
`HSA_STATUS_ERROR_INVALID_PACKET_FORMAT: The AQL packet is malformed` and
`fatal_signal signal=6` on its first request, minutes after a previous 82 GiB
model exited, and did not recur once the device was given time to settle. That
is the between-run settling behaviour the methodology describes, showing up as
a GPU queue error rather than non-finite logits. Space consecutive large loads
before concluding anything about this model's context ceiling.

<!-- bench:single-ar -->
| Qwen3.8-Flash-Next AR<br>Depth (tokens) | pp (tok/s) | tg (tok/s) |
| ---: | ---: | ---: |
| 0 | 1763.88 ± 8.78 | 28.30 ± 1.21 |
| 8,192 | 1652.39 ± 27.77 | 27.69 ± 0.03 |
| 32,768 | 1608.38 ± 37.66 | 27.97 ± 1.92 |
| 65,536 | 1567.54 ± 12.70 | 27.25 ± 2.85 |
| 131,072 | 1512.59 ± 0.83 | 27.50 ± 2.19 |
| 196,608 | 1404.72 ± 33.16 | 28.81 ± 0.34 |
<!-- /bench -->

<!-- bench:loading -->
| Qwen3.8-Flash-Next<br>Target | ready (s) |
| --- | ---: |
| Q4 | 50.30 ± 7.06 |
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


