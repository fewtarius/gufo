# Qwen35B-A3B benchmarks

AMD Radeon 780M (gfx1103), 24.6 GiB of system memory, 6 GiB of firmware VRAM
and a 26 GiB GTT window. Qwen3.6-35B-A3B UD-Q4_K_XL, one session, greedy,
thinking off.

**Throughput is not published for this model yet.** Prefill under Gufo sits
well below the llama.cpp Vulkan reference on the same weights and the same
host, and Gufo's prefill rate barely moves as the prompt grows where the
reference scales with it. Decode runs the other way, in Gufo's favour. That
gap is open work, so publishing pp and tg tables now would document numbers
we expect to move substantially. They will appear in one pass, from a single
measurement run, once the prefill path is fixed.

What follows is the memory geometry, which is a property of the cache format
rather than a performance claim, and which does not move when the prefill
path changes.

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

- **Prefill throughput.** Gufo's curve is flat against prompt length, which
  points at per-chunk overhead rather than kernel throughput. Dispatch
  telemetry reports a rejected compute graph (`rejectionMask=2`) on this
  workload; the Vulkan reference reuses its graph on the same model.
- **The comparison itself.** A llama.cpp Vulkan build runs on this host and
  serves these weights at the native 262,144-token context, so the reference
  columns that were previously marked unmeasurable are measurable.
- **Multi-user rows.** One session is the whole memory budget at this
  context.
