# Laguna q8_0 KV cache

## What it is

`LAGUNA_KV_CACHE_TYPE=q8_0` stores the Laguna attention K/V cache in q8_0
instead of F16. One binary16 scale and 32 signed codes per 32 elements: 34
bytes where F16 used 64. A row keeps **53.125%** of its F16 bytes.

The format and the numeric contract are in [../kv-codec.md](../../kv-codec.md).
The codec itself is model-agnostic and lives in `src/core/kv_codec/`; this
page is about the Laguna side of it.

```bash
LAGUNA_KV_CACHE_TYPE=q8_0 ./gufo-run.sh --model Laguna-S-2.1-UD-Q4_K_XL
```

F16 is the default. `LAGUNA_KV_CACHE_TYPE` accepts `f16` or `q8_0` and
rejects anything else at model load.

## What it changes

The cache is now addressed as bytes with a byte row stride, and the storage
format selects between two store kernels and two attention kernels:

- **Store**: `StoreKvKernel` (F16) or `StoreKvQ8Kernel` (q8_0). The q8_0
  store assigns one warp per block, so the block's absmax is a warp
  reduction and each lane writes its own code.
- **Attention**: `AttentionWmmaKernelLaguna<kHeadsPerKv, kQ8>` and
  `AttentionKernel<kQ8>`. The F16 path is unchanged; the q8_0 path
  reconstructs each group of eight elements on the way from global memory
  into LDS. Everything after that - the LDS tile, the WMMA math, the online
  softmax - is shared.
- **Pooled keys**: the sparse Quest bounds are computed from the
  reconstructed values, so the bound matches what attention reads.
- **Snapshots**: the payload version is now 2 and the header records the
  cache format. A snapshot taken in one format is rejected by a session in
  the other rather than restored into it.

The device store kernel and the device dequantizer are checked against the
host codec byte for byte in `tests/models/laguna/kv_q8_test.cpp`.

## Measured

Nimo Axis N161 (gfx1151), Laguna S 2.1 UD-Q4_K_XL, 56494-token prompt,
block-sparse attention on, greedy.

| | device memory | prefill tok/s | plain decode tok/s |
| --- | --- | --- | --- |
| F16 | 87534 MiB | 1052.0 | 18.8 |
| q8_0 | 81198 MiB | 1013.1 | 18.1 |

**-6.2 GiB of device memory (-7%), -3.7% prefill, -3.7% plain decode.**

The F16 path is unchanged: a greedy request produces byte-identical output
before and after this change.

### The drafter is the real cost

With DFlash2 enabled on the same prompt:

| | decode tok/s | drafts accepted |
| --- | --- | --- |
| F16 | 24.5 | 54/94 (57.4%) |
| q8_0 | 20.8 | 42/102 (41.2%) |

The plain-decode cost is 3.7%, but the decode with the drafter is 15%. The
difference is the drafter: DFlash2 predicts from the target's tapped hidden
states, and a lossy K/V cache perturbs the attention output that produces
them. Draft acceptance falls from 57% to 41%, so the same output needs more
verification passes.

That is the cost of the format, not of the implementation: the device path
matches the host codec exactly, so the attention output is what a q8_0 cache
must produce.

### Output

Greedy output differs from F16. Both answers are correct, but a lossy cache
moves the logits, and a near-tie in the argmax flips. This is the same
property gufetto documents for its q8 KV work.

## When to use it

q8_0 is a **memory** trade, not a speed one. Use it when the KV cache has to
fit somewhere F16 cannot - a smaller device, or several concurrent
sessions. On a machine with room to spare, F16 is faster and exact.

The saving is bounded by the full-attention layers. Laguna S 2.1 has 48
layers with `swa_period=4`, so 12 are full-attention and 36 are sliding
window. The sliding window layers only ever hold `sliding_window + 2 *
prefill_tile` rows, so they contribute about 1.2 GiB of the cache at a
262144-token context and q8_0 barely touches them. The 6.2 GiB measured here
is almost entirely the 12 full-attention layers.

## See also

- [../kv-codec.md](../../kv-codec.md) - the codec contract and the numeric rules
- [../KV-CACHE.md](../../KV-CACHE.md) - cross-request cache reuse
- [EXPERIMENTS.md](EXPERIMENTS.md) - the retained decisions behind this
  model, including the ones the q8_0 cache was chosen against
