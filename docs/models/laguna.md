# Laguna S 2.1

Mixture-of-experts text model on `gfx1151`, carried by this fork: 48 layers,
hidden 3,072, 8 K/V heads of 128, heads per layer `[48,72,72,72]` repeating,
256 experts with 10 routed plus a shared expert, and sliding-window attention
of 512 on 3 of every 4 layers. Context is 262,144 tokens.

## Files

Download the sharded GGUF as a set; the first shard alone will not load.

| File | Size | Notes |
| --- | ---: | --- |
| `UD-Q4_K_XL` × 3 shards | 68.35 GiB | The measured configuration below |
| `UD-Q5_K_XL` × 3 shards | 82.02 GiB | Needs a q8_0 K/V cache for a long context |
| `Laguna-S-2.1-DFlash-BF16-gufo.gguf` | 2.08 GiB | Optional drafter for speculative decoding |

## Run

```sh
MODEL=models/laguna/Laguna-S-2.1-UD-Q4_K_XL-00001-of-00003.gguf
DRAFT=models/laguna/Laguna-S-2.1-DFlash-BF16-gufo.gguf
gufo serve llm --model "$MODEL" --context 131072
gufo serve llm --model "$MODEL" --context 131072 \
  --speculative dflash2 --dflash-model "$DRAFT" --draft-tokens 3
```

The model path is one shard of the set; keep all three beside it. Laguna is
text-only, so an image input is refused. `dflash2` is the only speculative
backend it accepts, and it takes 1 to 15 draft tokens.

## Speed

Measured on a Nimo Axis N161 (Ryzen AI Max+ 395, Radeon 8060S, 124.9 GiB RAM,
114,688 MiB GPU pool) on 11 November 2026, Gufo `0e7b3333`, UD-Q4_K_XL at a
131,072-token context with an F16 K/V cache. Each row is a server request: a
2,048-token prompt against the cached prefix in the first column, 128 generated
tokens, greedy, thinking off, two repetitions. **pp** is prompt processing and
**tg** is token generation, both in tokens per second.

| Depth (tokens) | pp | tg |
| ---: | ---: | ---: |
| 0 | 1,183.87 ± 1.95 | 24.69 ± 0.22 |
| 8,192 | 1,059.19 ± 0.46 | 23.90 ± 0.01 |
| 32,768 | 810.33 ± 0.29 | 21.26 ± 0.01 |
| 65,536 | 613.15 ± 6.04 | 18.53 ± 0.00 |
| 98,304 | 489.65 ± 1.55 | 16.35 ± 0.01 |

Prefill falls with depth because attention reads a longer cache for a
2,048-token pass. A full-context prefill reports higher tokens per second, so
do not compare across protocols.

UD-Q5_K_XL at its native 262,144-token context needs the q8_0 cache and holds
109,195 MiB of the pool. It loads in 93.74 s against the Q4 set's 58.19 s, and
decodes at 22.25 tok/s at depth 0, 16.66 at 65,536 and 13.25 at 131,072. Both
weight sets, the load times and the full protocol are in the
[internal benchmarks](../internal/models/laguna/BENCHMARKS.md).

## Speculative decoding

The drafter proposes a block of tokens from the target's tapped hidden states,
and each proposal is verified against the target's own logits, so an accepted
token is the target's token. The gain depends on how predictable the text is: a
long refactor or a JSON body accepts most drafts, prose accepts few, and a
trivial prompt can end up slower than plain decode. Measure on your own text
before relying on a single number.

Draft acceptance also depends on the K/V cache format: a paired sweep at one
prompt accepted 57.4% of drafts with F16 and 41.2% with q8_0.

## K/V cache format

`LAGUNA_KV_CACHE_TYPE=q8_0` stores a K/V row in 53.125% of its F16 bytes,
which is what makes a 262,144-token context fit. It is lossy and costs about
3.7% of prefill and plain decode. F16 is the default. The format, the numeric
contract and the paired measurement are in
[the q8_0 cache page](../internal/models/laguna/KV-Q8.md) and
[the codec reference](../kv-codec.md).

## Block-sparse attention (opt-in)

`LAGUNA_SPARSE_ATTN=1` replaces full attention beyond the sink and local window
with block-sparse attention on the 12 full-attention layers; the 36
sliding-window layers are already bounded at 512 tokens. Each query reads 2 sink
blocks, a 32-block local window and the 256 best-scoring older 64-token blocks,
scored with a per-block channel min/max bound. Below about 18.5k tokens of
context every candidate fits the budget and the result is exact; above it the
selection is approximate.

Only prefill takes this path. A single-row decode pays the block scoring once
per split per K/V head and measured slower with it on, so decode stays dense.
`LAGUNA_SPARSE_TOPK=N` changes the candidate budget.

## Options

| Variable | Effect |
| --- | --- |
| `LAGUNA_KV_CACHE_TYPE` | `f16` (default) or `q8_0` for the attention K/V cache |
| `LAGUNA_SPARSE_ATTN` | Set to any value to enable block-sparse attention; unset is off |
| `LAGUNA_SPARSE_TOPK` | Candidate-block budget for sparse attention (default 256) |
| `LAGUNA_PREFILL_TILE` | Tokens one prefill pass processes (default 16384) |

## Numerical notes

Routed-expert activations are stored in F32 divided by 64. The q8_0 cache and
block-sparse attention are both lossy by construction; which outputs they change
is in [the quality report](../internal/models/laguna/QUALITY.md).

## Status

Prefill is checked with `gufo bench --validate-prefill`, which compares one
multi-token prefill against the same tokens appended one at a time. The CPU
reference in `src/models/laguna/reference.cpp` is the intended oracle, but no
test wires it to the GPU path yet. Speculative decoding works from `gufo prompt`
and `gufo serve`, on greedy and sampled requests. `gufo serve` reuses work
through the component cache, so a repeated prompt continues from its own state.
