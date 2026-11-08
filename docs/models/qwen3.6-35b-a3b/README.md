# Qwen3.6 35B-A3B

A hybrid mixture-of-experts text model on gfx1103 and gfx1151. Production
target is **UD-Q4_K_XL** (21.3 GiB weights); a draft model and a projector
need additional memory.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [q8_0 K/V cache](../qwen3.8-27b/KV-Q8.md)

## Load and run

```sh
MODEL=models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
./gufo-run.sh --model Qwen3.6-35B-A3B
```

The model is a `qwen35moe` GGUF: 41 layers with
`full_attention_interval=4`, so 31 linear-attention layers and 10
full-attention ones, 256 experts of which 8 are routed plus a shared
expert, hidden 2048, and 16 query heads over 2 key/value heads of 256.
The context length is 262144 tokens.

The runner this model shares with the Qwen3.8 27B is `qwen38/qwen35moe`.
The embedded MTP head the model carries is served by the Qwen Flash-Next
runner, so this model decodes without a drafter.

## Memory

The weights are 21.3 GiB, which is most of a 24.6 GiB machine. What is left
has to hold the K/V cache and the per-request state, and the K/V cache
grows with the context:

| context | F16 cache | q8_0 cache |
| --- | --- | --- |
| 8192 | 160 MiB | 85 MiB |
| 32768 | 640 MiB | 340 MiB |
| 131072 | 2560 MiB | 1360 MiB |

At a 131072-token context the F16 cache does not fit in the 1.9 GiB of
device memory the weights leave behind. The q8_0 cache does, and the
context loads. See the [q8_0 K/V cache](../qwen3.8-27b/KV-Q8.md) for the
format and its cost.

```sh
QWEN_KV_CACHE_TYPE=q8_0 ./gufo-run.sh --model Qwen3.6-35B-A3B
```

`QWEN_KV_CACHE_TYPE` accepts `f16` or `q8_0` and rejects anything else at
model load. It is a memory trade, not a speed one: the decode rate is the
same either way, because a decode step reads the weights rather than the
cache.

## Speculation

None. The DFlash2 drafter is accepted by this runner, but no DFlash2
drafter for this model is available on the measurement host, and the
model's own MTP head is not served by this runner. The
[Benchmarks](BENCHMARKS.md) are plain decode.
