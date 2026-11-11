# Laguna S 2.1

Mixture-of-experts text model on gfx1151: 48 layers, hidden 3,072, 8 KV heads
of 128, heads per layer [48,72,72,72] repeating, 256 experts with 10 routed
plus a shared expert, sliding-window attention of 512 on 3 of every 4 layers.
The tested weights are `Laguna-S-2.1-UD-Q4_K_M.gguf` (68 GB; Q8_0 attention,
Q4_K/Q5_K experts). The NPU kernel is `tools/npu/runtime_mk/`; see its README
for adding a shape.

## Run

```sh
gufo prompt -m Laguna-S-2.1-UD-Q4_K_M.gguf -f prompt.txt -n 128 -t 0 -v
```

`--dflash-model laguna-s-2.1-DFlash-BF16.gguf --speculative dflash2 -d 3` adds
the v1 (block-diffusion) drafter. `GUFO_NPU_DIR=<repo>/tools/npu` uses the NPU
(shared-expert gate/up, banked by default); needs the fabric clock held and a
binary named `gufo*`.

## Speed (GPU, interleaved runs, one machine)

| Prefill, tok/s | GPU only | GPU + NPU |
| --- | --- | --- |
| `-p 2048` | 1,145 | **1,277** (+11.5%) |
| `-p 4096` | 1,241 | **1,253** (+1.0%) |
| `-p 8192` | 1,193 | **1,204** (+0.9%) |

| Decode, tok/s | plain | DFlash v1 |
| --- | --- | --- |
| code | 25.9 | **44.5** (211/267 drafts accepted, 79%) |
| prose | 25.9 | 27.9 (153/441 accepted, 35%) |
| JSON | 25.9 | **45.4** (90/111 accepted, 81%) |

| Decode at depth, tok/s | plain (split-K attention) |
| --- | --- |
| 13,305 tokens | 24.0 |
| 47,280 tokens | 20.3 |
| 133,284 tokens | 14.8 |

Those plain-decode numbers are what a server request sees when the drafter is
off or the context is too deep for it to help. At 141,170 tokens on the N161,
`gufo serve` measured 14.4 tok/s plain and 16.1 tok/s with the drafter at
`temperature=0.7`.

DFlash v1's gain is sharply content-dependent: 79% of drafts accepted on code
(44.5 tok/s), 35% on prose (27.9 tok/s), and a short or trivial prompt can land
below plain decode. Measure on real target text, not a filler prompt, before
trusting a single decode number.

`gufo prompt` takes `--speculative dflash2 --dflash-model ...`; `gufo serve`
takes `--speculative` and drafts for greedy and sampled requests alike.

NPU offload of shared-expert gate/up is a small win (+0.2% to +2.4% depending
on depth) because those projections are a small fraction of per-layer FLOPs at
this model's shape. QKV on the NPU is built but a measured net loss for Laguna,
so it stays off by default.

Measurement protocol, acceptance rules and the port's full investigation:
[internal/models/laguna/BENCHMARKS.md](../internal/models/laguna/BENCHMARKS.md)
and [laguna-port-state.md](../../src/models/laguna/laguna-port-state.md).

## Block-sparse attention (opt-in)

`LAGUNA_SPARSE_ATTN=1` replaces full attention beyond the sink + local
window with block-sparse attention on the 12 full-attention layers only
(the 36 sliding-window layers are already bounded by `sliding_window=512`
and don't need it): each query reads 2 sink blocks, a 32-block local
window, and the 256 best-scoring older 64-token blocks (Quest bound,
per-block channel min/max). Exact below ~18.5k tokens of context (every
candidate fits inside the 256-block budget); approximate above it, same
accepted-divergence class as this port's other approximations.
Needle-in-a-haystack (a unique fact buried mid-document) matches dense on
every depth tested:

| Depth | Dense | Sparse | Gain |
| --- | --- | --- | --- |
| 41,327 tok | 937.6 tok/s, correct | 999.5 tok/s, correct | +6.6% |
| 61,108 tok | 835.3 tok/s, correct | 973.2 tok/s, correct | +16.5% |
| 100,616 tok | 694.6 tok/s, correct | 958.0 tok/s, correct | +37.9% |

Gain grows with depth, because the candidate pool to prune grows with
position. It stacks cleanly with the NPU and the two-bank pipeline.
`LAGUNA_SPARSE_TOPK=N` overrides the 256-block budget. The selection bugs and
the graph-capture boundary found while porting it are in
[laguna-port-state.md](../../src/models/laguna/laguna-port-state.md).

## Options (environment)

| Variable | Effect |
| --- | --- |
| `GUFO_NPU_DIR=<repo>/tools/npu` | Use the NPU (shared-expert gate/up, banked by default) |
| `LAGUNA_NPU_NO_BANK=1` | NPU without the two-bank pipeline (same-call overlap only) |
| `LAGUNA_NPU_QKV=1` | Also offload QKV to the NPU - measured net loss, off by default |
| `LAGUNA_SPARSE_ATTN=1` | Block-sparse attention on the 12 full-attention layers - off by default |
| `LAGUNA_SPARSE_TOPK=N` | Candidate-block budget for sparse attention (default 256) |
| `LAGUNA_PREFILL_TILE=N` | Prefill chunk size (default 4,096); a different N needs a matching NPU instruction file |
| `LAGUNA_KV_CACHE_TYPE=q8_0` | Store K/V as q8_0 instead of F16: 53.125% of the bytes, lossy. See [KV-Q8.md](../internal/models/laguna/KV-Q8.md) |

## Numerical notes

Routed-expert activations are stored in F32 divided by 64; the combine
multiplies it back. The NPU path is not bit-identical to the GPU path
(block-float, as for every other NPU offload here), so output diverges after
several tokens on real prose - the same accepted-divergence class as the other
numeric changes here. The port's precision investigation is in
[laguna-port-state.md](../../src/models/laguna/laguna-port-state.md).

## Status

Prefill and decode are verified against greedy text and `--validate-prefill`.
DFlash2 speculation works from both `gufo prompt` and `gufo serve`, on greedy
and sampled requests. NPU shared-expert offload is on by default; QKV offload
is built and off by default.

`gufo serve` reuses work through the component cache, so a repeated prompt or
a restored session continues from its own retained state. Block-sparse
attention is verified (needle-in-a-haystack at three depths, two-bank and NPU
interaction, the decode graph-capture boundary) and off by default;
`LAGUNA_SPARSE_ATTN=1` opts in.
