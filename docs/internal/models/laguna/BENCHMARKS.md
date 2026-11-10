# Laguna S 2.1 benchmarks

All numbers were measured on the Nimo Axis N161 (gfx1151, Radeon 8060S,
118 GiB GPU-visible) with the Laguna S 2.1 UD-Q4_K_XL weights, unless a row
says otherwise.

## Prefill

| Prompt tokens | tok/s | Notes |
| --- | --- | --- |
| 45,205 | 1,057.6 | block-sparse attention on |
| 141,170 | 965.6 | block-sparse attention on |
| 141,211 | 633.4 | dense (block-sparse off) |

Block-sparse attention is a prefill win that grows with depth: the candidate
pool it prunes grows with position. It is approximate above about 18.5k tokens
of context - exact below that, because every candidate block fits inside the
256-block budget.

## Decode

| Context tokens | tok/s | Mode |
| --- | --- | --- |
| 45,205 | 26.4 | DFlash2, temperature 0.7, 40% accepted |
| 45,205 | 24.7 | DFlash2, greedy, 56% accepted |
| 141,170 | 16.1 | DFlash2, temperature 0.7, 40% accepted |
| 141,211 | 18.1 | plain decode |

Decode flattens with depth: the sliding-window layers and the FFN/MoE cost
become a larger share of the step once the split-K attention layers' own cost
shrinks.

## Memory

Measured on the ported tree, at a 262,144-token context with the drafter off:

| KV format | Device memory | Prefill tok/s | Plain decode tok/s |
| --- | --- | --- | --- |
| F16 | 85,943 MiB | 1,278.2 | 20.7 |
| q8_0 | 79,895 MiB | - | - |

**-6,048 MiB of device memory (-7.0%).** See [KV-Q8.md](KV-Q8.md).

The prefill figure above is the ported tree's own `pp2048` measurement. It is
higher than the fork's because the port inherits upstream's newer W8A8
prefill GEMM, which splits a launch's tail into a separate kernel instead of
widening the main one.

## TODO

- A DFlash2 sweep at depth with content that a drafter accepts well. The
  measured decode numbers above use filler prose, and DFlash's acceptance is
  sharply content-dependent.
- The q8_0 KV row above is measured with the drafter off. With DFlash2 on,
  the drafter's acceptance falls from 57% to 41% because it predicts from the
  target's tapped hidden states, which a lossy K/V cache perturbs.
