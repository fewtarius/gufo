# Laguna S 2.1 benchmarks

Mixture of experts. Two weight sets are installed and measured: UD-Q4_K_XL at
68.35 GiB and UD-Q5_K_XL at 82.02 GiB. Greedy decode, temperature 0, thinking
off, one session. Measured on Gufo `0e7b3333`. Method and protocol:
[benchmark methodology](../../benchmarks.md).

The two weight sets get different cache formats because they need them: Q4 at
131,072 tokens with F16 K/V, Q5 at 262,144 tokens with q8_0 K/V. Q5 with F16 at
262,144 tokens needs more memory than the pool holds. See
[the K/V cache page](KV-Q8.md) for what the format costs and saves.

## Nimo Axis N161

Strix Halo: Ryzen AI Max+ 395, Radeon 8060S `gfx1151`, 40 CUs, 124.9 GiB RAM,
114,688 MiB pool.

### UD-Q4_K_XL, context 131,072, F16 K/V

<!-- bench:single-ar-q4 -->
| Laguna S 2.1 Q4 AR<br>Depth (tokens) | pp (tok/s) | tg (tok/s) |
| ---: | ---: | ---: |
| 0 | 1183.87 ± 1.95 | 24.69 ± 0.22 |
| 8,192 | 1059.19 ± 0.46 | 23.90 ± 0.01 |
| 32,768 | 810.33 ± 0.29 | 21.26 ± 0.01 |
| 65,536 | 613.15 ± 6.04 | 18.53 ± 0.00 |
| 98,304 | 489.65 ± 1.55 | 16.35 ± 0.01 |
<!-- /bench -->

### UD-Q5_K_XL, context 262,144, q8_0 K/V

The allocation sits at 109,195 MiB of the 114,688 MiB pool, and a filled
262,144-token context intermittently produces non-finite logits mid-prefill.
Rows below are the runs that completed; two repetitions per row where they did.
Treat 131,072 as the deepest prefix measured here, not as the model's limit.

<!-- bench:single-ar-q5 -->
| Laguna S 2.1 Q5 AR<br>Depth (tokens) | pp (tok/s) | tg (tok/s) |
| ---: | ---: | ---: |
| 0 | 1140.98 ± 1.41 | 22.27 ± 0.16 |
| 8,192 | 1013.25 ± 0.04 | 21.48 ± 0.02 |
| 32,768 | 737.75 ± 0.19 | 19.12 ± 0.00 |
| 65,536 | 532.70 ± 0.78 | 16.67 ± 0.01 |
| 131,072 | 336.64 ± 1.90 | 13.25 ± 0.01 |
<!-- /bench -->

### Load time

<!-- bench:loading -->
| Laguna S 2.1<br>Target | ready (s) |
| --- | ---: |
| Q4 | 58.19 ± 0.54 |
| Q5 | 93.74 ± 2.31 |
<!-- /bench -->

## AYANEO Flip KB

Neither weight set fits the 20,480 MiB pool. Q4 is 68.35 GiB of weights, so it
cannot be admitted at all, let alone run. Laguna is a Nimo-only model here.
