# Qwen3.6 35B-A3B

A hybrid mixture-of-experts text model on `gfx1103` and `gfx1151`, carried by
this fork. Production target is **UD-Q4_K_XL**, 21.28 GiB of weights.

[Benchmarks](../internal/models/qwen3.6-35b-a3b/BENCHMARKS.md) · [Quality](../internal/models/qwen3.6-35b-a3b/QUALITY.md) · [q8_0 K/V cache](../internal/models/qwen3.8-27b/KV-Q8.md)

## Run

```sh
MODEL=models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
gufo serve llm --model "$MODEL" --context 131072
```

The model is a `qwen35moe` GGUF: 41 layers with `full_attention_interval=4`,
so 31 linear-attention layers and 10 full-attention ones, 256 experts of which
8 are routed plus a shared expert, hidden 2048, and 16 query heads over 2
key/value heads of 256. The context length is 262,144 tokens.

The runner this model shares with the Qwen3.8 27B is `qwen38/qwen35moe`. The
embedded MTP head the model carries is served by the Qwen Flash-Next runner, so
this model decodes without a drafter.

## Memory

Only the 10 full-attention layers grow with the context, so an F16 K/V row
costs 20 KiB per token and a q8_0 row costs 53.125% of that:

| Context | F16 cache | q8_0 cache |
| --- | ---: | ---: |
| 8,192 | 160 MiB | 85 MiB |
| 32,768 | 640 MiB | 340 MiB |
| 49,152 | 960 MiB | 510 MiB |
| 131,072 | 2,560 MiB | 1,360 MiB |

On the Nimo Axis N161 the model at a 131,072-token context with an F16 cache
peaks at 29.32 GiB of device memory and decodes at 43.4 tok/s at depth 0.

The AYANEO Flip KB has a 20,480 MiB pool, which the weights exceed on their
own, so part of them stays in host memory. There it runs at a 49,152-token
context with the q8_0 cache and peaks at 19.63 GiB, decoding at 22.4 tok/s at
depth 0. An F16 cache at a context worth measuring does not fit. See the
[q8_0 K/V cache](../internal/models/qwen3.8-27b/KV-Q8.md) for the format and
its cost.

```sh
QWEN_KV_CACHE_TYPE=q8_0 gufo serve llm --model "$MODEL" --context 49152
```

`QWEN_KV_CACHE_TYPE` accepts `f16` or `q8_0` and rejects anything else at model
load. It is a memory trade, not a speed one: a decode step reads the weights
rather than the cache.

## Speculation

None. This runner accepts the DFlash2 drafter, but no DFlash2 drafter exists
for this model and the model's own MTP head is not served by this runner. The
[benchmarks](../internal/models/qwen3.6-35b-a3b/BENCHMARKS.md) are plain
decode.
