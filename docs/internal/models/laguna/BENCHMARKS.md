# Laguna S 2.1 benchmarks

Laguna S 2.1 UD-Q4_K_XL and UD-Q5_K_XL, measured on a Nimo Axis N161 (Strix
Halo: Ryzen AI Max+ 395, Radeon 8060S `gfx1151`, 124.9 GiB RAM visible,
112.5 GiB GPU-visible).

## Protocol

Two protocols appear on this page and they are not interchangeable.

- **Serving sweep** (`scratch/kv_sweep.sh`): `gufo serve llm --sessions 1`,
  temperature 0, 64 generated tokens, one fresh request per row, prompt cache a
  miss on every row, no speculative decoding. Prompt sizes are whatever the
  generated prose tokenized to, so they differ slightly between paired runs;
  both are shown.
- **Ported-tree** rows: the upstream benchmark driver at `pp2048`, quoted from
  the port. These are not matched to the serving sweeps.

A first forward that returns non-finite logits immediately after a previous
server exited is a device-settling artifact, not a result. Allow about 75
seconds between back-to-back large loads; see
[the methodology](../../benchmarks.md#between-run-settling).

## UD-Q4_K_XL at 131072: F16 against q8_0

Both runs asked for a 131072-token context. This is the paired A/B for the
cache format.

| Prompt tokens | pp F16 | pp q8_0 | tg F16 | tg q8_0 |
| ---: | ---: | ---: | ---: | ---: |
| 5,939 / 6,009 | 1,250.5 | 1,260.2 | 24.4 | 24.0 |
| 23,707 / 23,986 | 1,168.4 | 1,120.4 | 22.5 | 21.9 |
| 35,586 / 36,005 | 1,074.0 | 1,011.3 | 21.3 | 20.5 |
| 47,415 / 47,973 | 1,001.0 | 932.2 | 20.3 | 19.4 |
| 71,101 / 71,938 | 869.9 | 794.4 | 18.4 | 17.5 |

Device at load: F16 90,446 MiB, q8_0 83,563 MiB. **6,883 MiB saved (-7.6%).**

q8_0 costs 4 to 9% of prefill and 2 to 5% of decode past 24k. Unlike the Qwen
family the cost rises smoothly with depth, which is what a dequantized
attention read should do.

## UD-Q5_K_XL

This weight set did not load before the session-budget fix. It does now, and it
is the only Laguna configuration that reaches the model's native context on this
device.

Two runs, different contexts:

- F16 at 131072, which is the widest context F16 can hold.
- q8_0 at 262144, the model's native context.

| Prompt tokens | pp F16 @128k | pp q8_0 @262k | tg F16 @128k | tg q8_0 @262k |
| ---: | ---: | ---: | ---: | ---: |
| 6,289 / 6,359 | 1,255.5 | 1,204.1 | 22.3 | 21.9 |
| 25,102 / 25,381 | 1,159.5 | 1,098.1 | 20.6 | 20.0 |
| 37,681 / 38,100 | 1,067.7 | 1,003.6 | 19.6 | 18.9 |
| 50,205 / 50,763 | 987.9 | 917.5 | 18.6 | 17.9 |
| 75,286 / 76,123 | 858.5 | 778.8 | 16.9 | 16.2 |

Device at load: F16 at 131072 uses 109,963 MiB; q8_0 at **262144** uses
109,195 MiB. The q8_0 cache doubles the usable context and still lands 768 MiB
lower than F16 at half of it.

That is the whole reason the pair exists: F16 at 262144 needs 16,968 MiB of KV
cache plus 7,971 MiB of session scratch against 21,376 MiB free, so it cannot
reach native context at all. See [KV-Q8.md](KV-Q8.md).

Q5_K_XL costs a few percent against Q4_K_XL at the same context, which is what
the extra weight fidelity is paid for.

## Ported-tree measurements

Retained from the port, at the upstream driver's `pp2048`. UD-Q4_K_XL,
262,144-token context, drafter off.

| KV format | Device memory | Prefill tok/s | Plain decode tok/s |
| --- | --- | --- | --- |
| F16 | 85,943 MiB | 1,278.2 | 20.7 |
| q8_0 | 79,895 MiB | - | - |

**-6,048 MiB (-7.0%)** at 262,144 tokens.

The prefill figure is the ported tree's own measurement. It is higher than the
fork's because the port inherits upstream's newer W8A8 prefill GEMM, which
splits a launch's tail into a separate kernel instead of widening the main one.

Note the conflict: the paired sweep above shows 6,883 MiB saved at **131072**,
and this row shows 6,048 MiB saved at **262,144**. A saving proportional to
context would roughly double between the two. It does not, and that is
unexplained. Either the ported row's context was not the whole 262,144, or the
sliding-window ring buffers absorb more of the cache than the layer count
suggests. Do not quote the larger context's saving until this is re-measured
against a paired control on one revision.

## Block-sparse attention (ported tree)

UD-Q4_K_XL, block-sparse attention on unless the row says otherwise.

### Prefill

| Prompt tokens | tok/s | Notes |
| --- | --- | --- |
| 45,205 | 1,057.6 | block-sparse attention on |
| 141,170 | 965.6 | block-sparse attention on |
| 141,211 | 633.4 | dense (block-sparse off) |

Block-sparse attention is a prefill win that grows with depth: the candidate
pool it prunes grows with position. It is approximate above about 18.5k tokens
of context - exact below that, because every candidate block fits inside the
256-block budget.

### Decode

| Context tokens | tok/s | Mode |
| --- | --- | --- |
| 45,205 | 26.4 | DFlash2, temperature 0.7, 40% accepted |
| 45,205 | 24.7 | DFlash2, greedy, 56% accepted |
| 141,170 | 16.1 | DFlash2, temperature 0.7, 40% accepted |
| 141,211 | 18.1 | plain decode |

Decode flattens with depth: the sliding-window layers and the FFN/MoE cost
become a larger share of the step once the split-K attention layers' own cost
shrinks.

## TODO

- Reconcile the q8_0 saving between 131072 and 262144 (above).
- A DFlash2 sweep at depth with content a drafter accepts well. The measured
  decode numbers above use filler prose, and DFlash's acceptance is sharply
  content-dependent.
- The q8_0 rows are measured with the drafter off. With DFlash2 on, acceptance
  falls from 57% to 41%, because the drafter predicts from the target's tapped
  hidden states, which a lossy K/V cache perturbs.
- Confirm the full-attention layer count. `swa_period` defaults to 4 and is not
  read from the GGUF, which gives 12 full-attention layers of 48. The cache
  sizes measured above imply more. Either the default is wrong for this model or
  the accounting is; the paired sweep is the measurement that would settle it.
