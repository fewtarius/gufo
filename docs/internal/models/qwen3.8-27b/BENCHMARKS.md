# Qwen3.8-27B benchmarks

Dense, 27.3B parameters. Unsloth UD-Q4_K_XL, 16.35 GiB. Greedy decode,
temperature 0, thinking off, one session. Measured on Gufo `0e7b3333`. Method
and protocol: [benchmark methodology](../../benchmarks.md).

## Nimo Axis N161

Strix Halo: Ryzen AI Max+ 395, Radeon 8060S `gfx1151`, 40 CUs, 124.9 GiB RAM,
114,688 MiB pool. Context 131,072, F16 K/V.

<!-- bench:single-ar -->
| Qwen3.8-27B AR<br>Depth (tokens) | pp (tok/s) | tg (tok/s) |
| ---: | ---: | ---: |
| 0 | 579.04 ± 51.89 | 11.56 ± 0.00 |
| 8,192 | 502.58 ± 4.14 | 11.28 ± 0.01 |
| 32,768 | 388.01 ± 29.85 | 10.50 ± 0.00 |
| 65,536 | 303.86 ± 9.49 | 9.61 ± 0.01 |
| 98,304 | 256.31 ± 3.16 | 8.86 ± 0.00 |
<!-- /bench -->

<!-- bench:loading -->
| Qwen3.8-27B<br>Target | ready (s) |
| --- | ---: |
| Q4 | 7.04 ± 0.03 |
<!-- /bench -->

<!-- bench:memory -->
| Qwen3.8-27B AR<br>Workload | GiB |
| --- | ---: |
| pp2048 + tg128 | 27.03 |
| 32K prefix, pp4096 + tg128 | 27.03 |
<!-- /bench -->

## AYANEO Flip KB

Ryzen 7 7840U, Radeon 780M `gfx1103`, 8 CUs, 25.2 GiB RAM, 20,480 MiB pool.
Context 32,768 with q8_0 K/V.

The weights are 16.35 GiB of a 20,480 MiB pool, so this model loads by leaving
part of itself in host memory. Decode tracks active parameters, and the spill
adds host round trips on top, so the rate is far below the same model on the
Nimo. The numbers describe a configuration that works, not one worth using.

<!-- bench:single-ar-handheld@artifacts-flip -->
| Qwen3.8-27B AR<br>Depth (tokens) | pp (tok/s) | tg (tok/s) |
| ---: | ---: | ---: |
| 0 | 113.91 ± 0.43 | 3.79 ± 0.01 |
| 8,192 | 107.16 ± 0.47 | 3.65 ± 0.00 |
| 16,384 | 100.39 ± 0.06 | 3.53 ± 0.00 |
| 24,576 | 94.34 ± 0.14 | 3.41 ± 0.00 |
| 28,672 | 88.46 ± 0.10 | 3.36 ± 0.00 |
<!-- /bench -->

<!-- bench:loading-handheld@artifacts-flip -->
| Qwen3.8-27B<br>Target | ready (s) |
| --- | ---: |
| Q4 | 9.41 ± 3.26 |
<!-- /bench -->

<!-- bench:memory-handheld@artifacts-flip -->
| Qwen3.8-27B AR<br>Workload | GiB |
| --- | ---: |
| pp2048 + tg128 | 16.83 |
| 16K prefix, pp4096 + tg128 | 16.83 |
<!-- /bench -->

## K/V cache format

[q8_0 against F16 on this model](KV-Q8.md) covers what the cache format costs in
throughput and saves in memory.

## What the numbers show

Decode is low for the weight size: 11.6 tok/s at depth 0 on the Nimo, against
43.4 for Qwen3.6-35B-A3B at 21.28 GiB. Dense decode reads every weight per
token, so 27.3B dense costs more per token than a 35.5B mixture that activates
about 3B. Prefill is closer between them (579 against 1,923 tok/s at depth 0)
because prompt processing is compute bound on the same class of GEMM either way.

On the handheld the model spills out of the 20,480 MiB pool, and the result is
113.9 tok/s of prefill and 3.8 tok/s of decode at depth 0. That is 20% and 33%
of the Nimo's rates for a model that would otherwise be a reasonable fit at a
smaller context.
