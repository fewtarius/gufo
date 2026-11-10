# Qwen35B-A3B benchmarks

**AYANEO FLIP KB** (Ryzen 7 7840U, Radeon 780M `gfx1103`, 25.2 GiB RAM
visible, 26 GiB GPU-visible: a 6 GiB firmware carveout plus a 20 GiB GTT
window). Qwen3.6-35B-A3B UD-Q4_K_XL, one session, greedy.

## Method

Prefill is measured through `/v1/completions` at `temperature: 0` with prompts
generated so that no two of them share a prefix. This matters: both engines
reuse a repeated prompt prefix, so a set of probes built by repeating one
sentence measures only the new tail and reports a rate with no meaning. Every
figure below was taken with the prompt cache a miss, confirmed from the
server's own completed-request line.

The llama.cpp columns are from a llama.cpp Vulkan build on the same host, same
weights, same prompts, in the same session, with `--flash-attn on` and q8_0
K/V. Both engines were measured sequentially because the model does not fit
twice; that forces one of them to run hotter. llama.cpp's own rate fell from
400 to 316 tok/s across three identical 4804-token prompts in one run, so a few
percent either way is thermal, not a difference between engines.

## Strix Halo - the same model, F16 against q8_0

A **Nimo Axis N161** (Strix Halo: Ryzen AI Max+ 395, Radeon 8060S `gfx1151`,
124.9 GiB RAM visible, 112.5 GiB GPU-visible). Every other table on this page
is the AYANEO FLIP KB.

Context 131072, `gufo serve llm --sessions 1`, temperature 0, 64 generated
tokens, one fresh request per row, prompt cache a miss. Two separate runs, so
the prompt token counts differ slightly.

| Prompt F16 / q8_0 | pp F16 | pp q8_0 | tg F16 | tg q8_0 |
| --- | ---: | ---: | ---: | ---: |
| 6,009 / 6,079 | 1,786.3 | 1,791.9 | 41.6 | 41.7 |
| 23,986 / 24,265 | 1,690.0 | 1,696.9 | 36.6 | 35.9 |
| 47,961 / 48,519 | 1,318.9 | 1,301.5 | 31.3 | 30.2 |
| 71,937 / 72,774 | 1,070.3 | 906.0 | 28.0 | 26.2 |

Device at load: F16 29,928 MiB, q8_0 28,728 MiB.

Decode is within 3.5% through 48k, matching the FLIP KB finding that the format is a
memory trade rather than a speed one: a decode step reads the weights, not the
cache. The 72k prefill row is 15.4% slower on q8_0 after being 1.3% slower at
48k, with TTFT 80,325 ms against 67,215 ms. A 14x jump in the gap across one
step is not a smooth dequant cost, so that row needs a repeat before it is
trusted.

On this device the cache format buys nothing at 131072 - 1,200 MiB against a
112.5 GiB pool. It matters on the FLIP KB, where the same 1,200 MiB is the difference
between loading and not.

## Prefill

Tokens per second, cumulative rate for the whole prompt, which is how
llama.cpp's own log reports it.

| Prompt tokens | llama.cpp Vulkan | Gufo | Gufo vs reference |
| ---: | ---: | ---: | ---: |
| 4,637 | ~401 | 383.3 | -4.5% |
| 9,598 | ~380 | 363.2 | -4.5% |
| 14,699 | 337.7 | 343.3 | +1.6% |
| 18,664 | 319.4 | 329.3 | +3.1% |
| 19,888 | 315.4 | 325.3 | +3.1% |
| 93,011 | not measured | 197.7 | |

### Prompt chunk width

The chunk width is chosen at load from the memory left after the context the
caller asked for, so a shorter context runs a wider chunk. That is a real speed
difference, not a rounding error:

| prompt tokens | 1024-row chunk (context 102400) | 2048-row chunk (context 16384) |
| ---: | ---: | ---: |
| 4,010 | 370.4 | 401.7 |
| 4,673 | 383.3 | 397.6 |
| 9,616 | 363.2 | 381.3 |
| 14,653 | 343.3 | 360.3 |

Ask for the context you need rather than the largest one available. The routed
expert path fills fixed-width tiles, and a wider chunk fills more of them.

The rate decays with prompt length on both engines because attention grows with
context while the feed-forward does not. Gufo is a few percent behind at short
prompts and a few percent ahead from roughly 14,000 tokens up.

The 4,637 and 9,598 reference figures are interpolated from llama.cpp's
neighbouring measurements; the rest are its own `prompt eval time` lines from
the same run.

## Decode

26 to 27 tokens per second against the Vulkan reference's 16 to 17. Decode
reads the weights per step rather than the context, so it does not move with
the cache format.

## Configuration these numbers were taken with

```
./gufo-run.sh --model Qwen3.6-35B-A3B
QWEN_KV_CACHE_TYPE=q8_0
```

The RAM cache is enabled. It used to be a liability here: with it on, four of
five identical short requests failed with `logit distribution contains no finite
values`, because the cache's ledger allowance was the whole remaining device pool
and the component cache sizes device resources from that allowance. The
allowance now keeps a request-transient reserve and is capped at a third of the
headroom, which fixed five of five. At this memory size the cache still declines
to retain anything (it logs `ram_cache_refused`), because a single checkpoint is
a 69 MB fixed payload dominated by the recurrent state and there is no room for
one alongside the context. That is a capacity limit, not a defect.

The MoE prompt chunk is 1024 rows here and it is why the loader clamps the native
262,144-token context to 102,400: the prompt chunk and the context draw on the
same budget. The width is resolved from free device memory at load and only
widens on a host with at least 8 GiB to spare, so this clamp is specific to a
machine this tight.

## What is verified

- 8 of 8 exact-answer questions, warm server, this configuration.
- A 93,011-token prompt prefills at 197.7 tok/s, so the clamped context is
  usable rather than theoretical.
- Three concurrent requests each return content.
- A prompt beyond the context limit is refused with a clean error rather than a
  device fault.
- Server survives the whole battery.

## K/V cache footprint

Per-context cost at three capacities, F16 against q8_0. The q8_0 row stores a
row in 53.125% of the F16 bytes.

| Context | F16 K/V | q8_0 K/V |
| ---: | ---: | ---: |
| 8,192 | 160 MiB | 85 MiB |
| 32,768 | 640 MiB | 340 MiB |
| 131,072 | 2,560 MiB | 1,360 MiB |

The F16 cache at 131,072 tokens does not fit the memory this model leaves on
the host, which is why the 131,072-token configuration requires
`QWEN_KV_CACHE_TYPE=q8_0`. See [EXPERIMENTS.md](EXPERIMENTS.md) for the
scratch and state accounting behind that ceiling.

## Measurement setup

`./gufo-run.sh --model Qwen3.6-35B-A3B` with `QWEN_KV_CACHE_TYPE=q8_0`, one
session, `--prefill-chunk 512`, speculative decoding off. The loader clamps
the native 262,144-token context to 131,072 against the device budget and
reports the plan on stderr.

Requests go to `/v1/completions` with `temperature: 0`. Each prompt carries a
unique nonce so the component cache cannot answer it from a checkpoint.

## Where the time goes

`GUFO_PROFILE=1`, `GUFO_PROFILE_SSM=1` and `GUFO_PROFILE_FFN=1` break a prompt
chunk down by stage. On a 1889-token chunk at 397 tok/s the chunk is 4.75 s:

| block | ms | share |
| ---: | ---: | ---: |
| feed-forward | 2993 | 63% |
| the state-space mixer (30 layers) | 1401 | 29% |
| remainder | ~360 | 7% |

The routed experts dominate the feed-forward and are the clearest headroom: they
run at 1.1 to 1.8 TFLOP/s where the shared expert, at the same 512-wide shape on
the same data, reaches 4.2. A routed weight row is reused once per expert bucket,
so the rate follows `rows * experts_used / experts`, which is 64 rows per bucket
at a 2048-row chunk and 128 at 4096. That is why asking for a shorter context,
and so a wider chunk, is faster here.

## Prompt throughput

Context 16384, q8_0 KV cache, resolved chunk width 3072:

| prompt tokens | tok/s |
| ---: | ---: |
| 9795 | 449.2 |
| 9795 | 447.5 |
| 9795 | 447.1 |
| 14368 | 433.3 |
| 14368 | 432.6 |

At width 2048 the same probes measure 425.7 at 9795 tokens and 405.1 at 14368, so
the wider chunk is worth 5% and 7%. llama.cpp on this host and model reports
385 t/s at 12288 tokens and 349 at 15516, so gufo is ahead of it across this
range.

## Long context

A 131072-token context runs on this host. It is the ceiling, and the arithmetic
says why: the weights are 21794 MiB of a 20480 MiB device pool, so the KV cache
comes out of the 1907 MiB that is left, and at 128K it already takes 1360 of it
(q8_0, 10 attention layers). Per-request state is 1487 MiB and the MoE scratch is
264 MiB at the 512-row chunk the resolver narrows to.

| context | outcome |
| ---: | --- |
| 131072 | runs |
| 135168 | refused, reduce --context |
| 143360 | loads at the narrowest chunk, then hangs the GPU |
| 200000 | refused |

At 131072 the chunk narrows to 512 rows on its own. The scratch and the context
draw on the same pool, so the width is what gives way: at 1024 rows the claim is
7 MiB short of what the pool can carry.

Prefill of a 117987-token prompt, which is 90% of the context:

| configuration | prefill | decode |
| --- | ---: | ---: |
| 512-row chunk | 189.4 tok/s | 7.2 tok/s |
| 768-row chunk | 192.1 tok/s | 7.2 tok/s |
| default cache | 186.8 tok/s | 7.2 tok/s |

Against 449 tok/s prefill at 16K, the fall to 187 is the attention cost over an
ever-larger cache, and decode falls from 27 to 7.2 for the same reason.

128K is at the edge of this pool: about 156 MiB is free once the state is charged,
so a request that needs more fails, and it fails as a logit distribution with no
finite values rather than as an allocation error. Use `--cache-ram-bytes off` at
this context. Higher contexts need a larger pool, not a code change.

Two cache defects surfaced here and are fixed: a long context could fail the load
outright with a bare `std::bad_alloc`, because the cache's budget was computed
from a claim taken before the sessions charged their state; and the refusal path
still handed the cache a device allowance, which starved the first request after
the refusal.

## Open work

- **The prompt chunk is capped at 3072 rows.** 4096 is absent from the resolver
  because it cannot be made to fit this device at any context: it needs 1502 to
  1630 MiB against a budget near 1430, so it has never been executed and its
  correctness is not established. Everything the packed attention prefix needs to
  be correct is now in place, so 4096 should join the list once it is verified on
  a host with the memory. The last attention layer's source ends 256 elements
  short of the plane at the full context, so that margin is worth re-checking at
  any larger width or context.
- **A 4096-row chunk is 198 MiB short of the heap budget here**: 1630 MiB needed
  against 1432. The largest reclaimable block is `down_out`, the per-expert
  down-projection output, which is materialised in full before the combine and is
  268 MiB of the per-row scratch at that width.
- **The mixer's recurrence is a token-sequential scan** at 18.5 ms per layer, the
  largest single mixer stage. The chunked form of the delta rule would replace it
  with a few matrix products per chunk of tokens.
- **Checkpoint retention does not fit here.** The context and the weights leave
  too little device memory for a 69 MB checkpoint, so the cache declines to
  retain and every prompt is a fresh prefill. On a host with the headroom it
  retains normally; the refusal is reported rather than silent.
- **Context ceiling.** The prompt chunk and the context draw on the same budget,
  so the clamp lands below the native 262,144 at this context. The chunk width is
  resolved from available memory, so a host with the headroom takes a wider chunk
  and a longer context together.
- **Multi-user rows.** One session is the whole memory budget at this context.
