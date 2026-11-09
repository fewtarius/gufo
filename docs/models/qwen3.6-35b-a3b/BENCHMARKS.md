# Qwen35B-A3B benchmarks

AMD Radeon 780M (gfx1103), 25.8 GiB of usable system memory, 6 GiB of firmware
VRAM and a 20 GiB GTT window. Qwen3.6-35B-A3B UD-Q4_K_XL, one session, greedy.

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

## Open work

- **Checkpoint retention does not fit here.** The context and the weights leave
  too little device memory for a 69 MB checkpoint, so the cache declines to
  retain and every prompt is a fresh prefill. On a host with the headroom it
  retains normally; the refusal is reported rather than silent.
- **Context ceiling.** The prompt chunk is a compile-time arena property, and it
  and the context draw on the same budget, so the clamp lands at 102,400 rather
  than the native 262,144. Sizing the chunk from available memory instead would
  let a larger host take a wider chunk and a longer context together.
- **Multi-user rows.** One session is the whole memory budget at this context.
- **Wider prompt chunks.** 1024 rows fills the routed expert tiles; 2048 would
  fill them further, but it needs the context budget back first.
