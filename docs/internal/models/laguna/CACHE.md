# Laguna S 2.1 checkpoint caching

How the component cache represents a Laguna session, what a checkpoint costs,
and what a restore moves. [cache.md](../../cache.md) describes the cache
itself; this page is the Laguna side of it.

## Session state

A Laguna session holds one K/V cache per layer, plus a few small buffers:

- The 12 full-attention layers (`swa_period=4`) keep every row from 0 to the
  current position.
- The 36 sliding-window layers keep a ring of `sliding_window + prefill_tile`
  rows, because a forward writes up to one tile and then reads every key back
  to `window + 1` rows behind its own queries.
- The pooled block bounds, when `LAGUNA_SPARSE_ATTN` is set, are derived from
  the keys and cover the blocks that exist at the current position.
- The last row of logits is needed to continue without a forward.

## Components

`LagunaComponentAdapter` maps that state onto the cache's two component kinds.

**Append rows.** Each full-attention layer contributes a K component and a V
component. A row is `num_kv * head_dim` elements in the configured format:
1088 bytes at `q8_0`, 2048 at F16. Rows are chunked every 256. A row is final
once written, so a checkpoint at position `n` shares every chunk with the
checkpoint before it and copies only the rows past that one. Retaining twelve
full-attention layers costs about 26 KiB per token at `q8_0`; the first
checkpoint of a conversation therefore pays for its whole prefix, and later
checkpoints pay only for the growth.

**Private state.** The sliding-window rings, the pooled block bounds, the
logits row and a small header are one private component. Its size is fixed by
the context rather than by the position, so it is the same for every
checkpoint:

| Part | `q8_0` | F16 |
| --- | ---: | ---: |
| 36 sliding-window rings, `window=512` | 38 MiB | 72 MiB |
| 12 pooled-key sets (only with `LAGUNA_SPARSE_ATTN`) | 192 MiB | 192 MiB |
| Logits | 0.4 MiB | 0.4 MiB |

A session with a loaded drafter does not use these components. The drafter's
persistent payload is position-dependent and has no fixed upper bound in the
current interface, so such a session keeps the earlier representation: one
private payload the size of the widest snapshot the context can produce, about
7 GiB at 262,144 tokens with `q8_0`. Attribution and restore work the same
way; only the cost of a checkpoint differs.

## Capture and restore

Rows are materialized at capture rather than borrowed from the live session.
The cache owns every checkpoint, so a later write to the session cannot
invalidate a retained one and no overwrite guard is needed. The cost is one
copy out per new row, which the cache already performs for adapters that
cannot lend device memory.

A restore writes rows straight into the layer's device cache at their absolute
offsets, loads the private payload, and sets the session position from the
payload header. Row offsets do not depend on the position, so the order in
which the cache submits row and private transfers does not matter.

The private payload records the SWA rings sized for the maximum context and
fills the rows past the frontier with zeros. A restore at a shallower position
therefore overwrites the whole ring, and a slot that previously held a longer
prefix cannot leak its stale window rows or its pooled bounds into the
restored one.

## Budget

Two policies decide how much host memory the cache may retain.

An **explicit** `--cache-ram-bytes` is the amount the user asked for, capped at
the host RAM still available after the sessions allocate, minus the 4 GiB
`kHostSnapshotHeadroomBytes` reserve. A request above that cap is trimmed and
reported as `event=ram_cache_trimmed`.

An **automatic** budget (no `--cache-ram-bytes`, or `0`) is half of the same
available figure, capped at 32 GiB. The other half is the reserve, so the fixed
4 GiB reserve is not applied a second time: doing so clamped the automatic
budget to zero whenever the host had less than 4 GiB free and disabled
checkpoints entirely instead of trimming them.

The HIP resource provider commits backing pages lazily and reports no
long-term pins, so `slot_pinned_bytes` is zero for the HIP models. The
pinned-byte accounting still applies to providers that do pin.

## Measured

Nimo Axis N161, `gfx1151`, UD-Q4_K_XL, `q8_0` K/V, 262,144-token context, two
sessions, no drafter, greedy, 16 output tokens. One server per run, a shared
system prompt, and a different instruction per request so the second request
reuses the first one's prefix and nothing else.

| Run | Prompt | Reused | Prefilled | Restore | TTFT | prefill (tok/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1, no checkpoint | 9,838 | 0 | 9,838 | - | 8.43 s | 1,176 |
| A2, shared prefix | 9,838 | 8,192 | 1,646 | 9.4 ms | 1.60 s | 1,047 |
| B1, no checkpoint | 42,039 | 0 | 42,039 | - | 43.2 s | 977 |
| B2, shared prefix | 42,039 | 40,960 | 1,079 | 34.7 ms | 1.76 s | 641 |

`automatic_bytes` was 10.7 GiB. At the 42,039-token request the ledger held
about 3.9 GiB of it, host RAM reported 17.9 GiB available, and no capture was
refused. Decode measured 15.2 tok/s at that depth. The cache changes how much
prompt is prefilled and leaves the decode path alone, so decode moves only with
the KV format and the achieved occupancy.

## Limits

- A loaded drafter keeps the flat representation described above, so
  speculative sessions pay the full-context cost per checkpoint.
- Capture is synchronous: it drains the compute stream once per component
  rather than transferring on the cache's stream. It did not measurably slow
  prefill in the runs above, but it does not overlap with other work.
- The SWA rings are one private component rather than a `kWindowRows`
  component, because a ring wraps and the row API takes a contiguous range.
