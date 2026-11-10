# Qwen q8_0 KV cache

## What it is

`QWEN_KV_CACHE_TYPE=q8_0` stores the Qwen attention K/V cache in q8_0
instead of F16. One binary16 scale and 32 signed codes per 32 elements: 34
bytes where F16 used 64. A row keeps **53.125%** of its F16 bytes.

The format and the numeric contract are in
[../../kv-codec.md](../../kv-codec.md). The codec itself is model-agnostic
and lives in `src/core/kv_codec/`; this page is about the Qwen side of
it.

This page covers the Qwen3.5 and Qwen3.8 families, which share one
runner and one cache layout: the Qwen3.8 27B and the Qwen3.6 35B-A3B.

```bash
QWEN_KV_CACHE_TYPE=q8_0 ./gufo-run.sh --model Qwen3.6-35B-A3B-UD-Q4_K_XL
```

F16 is the default. `QWEN_KV_CACHE_TYPE` accepts `f16` or `q8_0` and
rejects anything else at model load. The variable is read by the model
runner, not the server, so it applies to every session the process
serves.

## Why it matters here

The Qwen3.5 and Qwen3.8 families are hybrid: most of their layers are
linear attention and only every fourth layer is full attention, so the
K/V cache is small next to the weights. What it is not is negligible once
the context grows.

Qwen3.6-35B-A3B has 41 layers with `full_attention_interval=4`, so 10 of
them are full attention. Each of those holds a row of
`num_key_value_heads * head_dim` = 2 x 256 = 512 values per token, in
both a K plane and a V plane.

| context | F16 cache | q8_0 cache |
| --- | --- | --- |
| 8192 | 160 MiB | 85 MiB |
| 32768 | 640 MiB | 340 MiB |
| 131072 | 2560 MiB | 1360 MiB |

On a 24.6 GiB APU the 2560 MiB of F16 cache is the difference between a
context that loads and one that does not. The weights are 21.3 GiB of the
same pool, and what was left after them was 1907 MiB.

## What it changes

The cache is addressed through one descriptor, `QwenCompactKv`, which
names the K plane, the V plane and the format they are stored in. The
attention routes read the plane through `KvElements`, which is
templated on the format, so the F16 and q8_0 paths share one arithmetic
and differ only in how a value is fetched.

- **Store**: the q8_0 plane is written by `LaunchEncodeQ8KvRows`, which
  encodes a row from the binary32 K and V the layer produced. The F16
  plane keeps its own writer.
- **Attention**: the decode split-K, decode online and decode baseline
  kernels, the causal verification kernel, the batched prefill kernel,
  the tiled prefill kernel, the WMMA prefill kernel and the graph decode
  kernel all take the format as a template parameter. The q8_0 path
  reconstructs each value from its block's scale as it reads it.
- **Snapshots**: a q8_0 session has its own state ABI, so a checkpoint
  taken in one format is refused by a session in the other rather than
  restored into it.

The device encoder matches the host codec byte for byte, including the
binary16 scale and the tie-breaking, and the read path reproduces the
dequantized values exactly. Both are checked in
`tests/models/qwen/hip/attention/kv_storage_ops_test.cpp`.

## Measured

Ayaneo Flip KB (780M, gfx1103), 24.6 GiB of RAM, 6 GiB of firmware VRAM
and a 26 GiB GTT window. Qwen3.6-35B-A3B UD-Q4_K_XL, one session,
greedy, 128 new tokens.

| context | format | device used | decode tok/s | prompt tok/s |
| --- | --- | --- | --- | --- |
| 8192 | F16 | 25149 MiB | 27.3 | 62.8 |
| 8192 | q8_0 | 25075 MiB | 27.3 | 62.4 |
| 32768 | F16 | 25629 MiB | 27.3 | 59.6 |
| 32768 | q8_0 | 25329 MiB | 27.3 | 62.5 |
| 131072 | F16 | does not load | - | - |
| 131072 | q8_0 | 26351 MiB | 27.2 | 25.3 |

At 131072 the F16 cache does not fit:

```text
text runner request-state claim exceeds state capacity: 1 sessions at
131072-token context need 2938028244 bytes (2.7 GiB) each, but only
2000326656 bytes (1.9 GiB) of device memory is available
```

**The decode rate is unchanged.** A decode step reads the weights, not
the cache, so the extra work of reconstructing a value from its block is
lost in the noise. This is a memory trade, not a speed one, and it is the
opposite of what the Laguna measurement shows, where the drafter's draft
acceptance collapses because the lossy cache perturbs the attention
output it predicts from. The Qwen runner has no drafter on this box, so
there is nothing to lose.

The savings are bounded by the full-attention layers. The 31 linear
attention layers hold no K/V cache at all, so q8_0 can only ever touch
the 10 full-attention ones. At a 131072-token context that is the whole
2560 MiB, and the whole 1360 MiB it becomes.

### Output

A lossy cache moves the logits, so a near-tie in the argmax can flip.
The q8_0 path is not bit-identical to F16; it is the same answer, from
slightly different numbers.

## When to use it

Use q8_0 when the context has to fit somewhere F16 cannot: a smaller
device, several concurrent sessions, or a longer context. On a machine
with room to spare, F16 is exact and costs nothing extra at decode time.

The saving is proportional to the context. At a short context it is
noise; the measurements above show 74 MiB at 8192 tokens against 300 MiB
at 32768 and 1200 MiB at 131072.

## See also

- [../../kv-codec.md](../../kv-codec.md) - the codec contract and the numeric rules
- [../../KV-CACHE.md](../../KV-CACHE.md) - cross-request cache reuse
- [../laguna/KV-Q8.md](../laguna/KV-Q8.md) - the same codec on the Laguna
  model, where the drafter makes the loss visible
