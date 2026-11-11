# Qwen3.6-35B-A3B benchmarks

Mixture of experts, 35.5B parameters with about 3B active per token. Unsloth
UD-Q4_K_XL, 21.28 GiB. Greedy decode, temperature 0, thinking off, one session.
Measured on Gufo `0e7b3333`. Method and protocol:
[benchmark methodology](../../benchmarks.md).

## Nimo Axis N161

Strix Halo: Ryzen AI Max+ 395, Radeon 8060S `gfx1151`, 40 CUs, 124.9 GiB RAM,
114,688 MiB pool. Context 131,072, F16 K/V.

<!-- bench:single-ar -->
| Qwen3.6-35B-A3B AR<br>Depth (tokens) | pp (tok/s) | tg (tok/s) |
| ---: | ---: | ---: |
| 0 | 1922.71 ± 5.59 | 43.43 ± 0.02 |
| 8,192 | 1741.13 ± 3.12 | 40.61 ± 0.04 |
| 32,768 | 1259.29 ± 18.29 | 34.06 ± 0.02 |
| 65,536 | 831.88 ± 15.37 | 28.58 ± 0.03 |
| 98,304 | 719.64 ± 9.97 | 24.74 ± 0.03 |
<!-- /bench -->

<!-- bench:loading -->
| Qwen3.6-35B-A3B<br>Target | ready (s) |
| --- | ---: |
| Q4 | 8.68 ± 0.00 |
<!-- /bench -->

<!-- bench:memory -->
| Qwen3.6-35B-A3B AR<br>Workload | GiB |
| --- | ---: |
| pp2048 + tg128 | 29.32 |
| 32K prefix, pp4096 + tg128 | 29.70 |
<!-- /bench -->

## AYANEO Flip KB

Ryzen 7 7840U, Radeon 780M `gfx1103`, 8 CUs, 25.2 GiB RAM, 20,480 MiB pool.
Context 49,152 with q8_0 K/V. F16 at a context worth measuring does not fit
this pool.

<!-- bench:single-ar-handheld@artifacts-flip -->
| Qwen3.6-35B-A3B AR<br>Depth (tokens) | pp (tok/s) | tg (tok/s) |
| ---: | ---: | ---: |
| 0 | 454.74 ± 2.09 | 22.38 ± 0.04 |
| 8,192 | 386.59 ± 0.23 | 19.63 ± 0.00 |
| 16,384 | 335.59 ± 0.47 | 17.43 ± 0.00 |
| 32,768 | 261.25 ± 2.82 | 14.09 ± 0.01 |
| 40,960 | 237.89 ± 0.10 | 12.83 ± 0.00 |
<!-- /bench -->

<!-- bench:loading-handheld@artifacts-flip -->
| Qwen3.6-35B-A3B<br>Target | ready (s) |
| --- | ---: |
| Q4 | 12.72 ± 1.16 |
<!-- /bench -->

<!-- bench:memory-handheld@artifacts-flip -->
| Qwen3.6-35B-A3B AR<br>Workload | GiB |
| --- | ---: |
| pp2048 + tg128 | 19.63 |
| 32K prefix, pp4096 + tg128 | 19.63 |
<!-- /bench -->

## What the numbers show

Prefill falls 63% from depth 0 to 98,304 on the Nimo (1,923 to 720 tok/s) while
decode falls 43% (43.4 to 24.7). The model reads a wider attention window for
every token as depth grows, and prompt processing pays that cost per token while
decode amortises it across the generated sequence.

The handheld reaches 24% of the Nimo's prefill and 52% of its decode at depth 0:
455 against 1,923 tok/s, and 22.4 against 43.4. The 780M has 8 compute units
against 40, which is what prefill spends, while decode is bound by reading the
active weights and gains less from extra units.

Depth costs prefill on both devices for the same reason. The handheld loses 48%
by 40,960 tokens, which is 83% of the 49,152-token context it can hold with q8_0
K/V. The Nimo loses 63% by 98,304, which is 75% of its 131,072-token context at
F16. Comparing equal fractions of each device's reachable context, the two are
close.
