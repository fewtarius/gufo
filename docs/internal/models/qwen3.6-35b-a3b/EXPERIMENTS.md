# Qwen35B-A3B experiments

Short retained decisions. Measurements belong in
[BENCHMARKS.md](BENCHMARKS.md); quality evidence in
[QUALITY.md](QUALITY.md).

## The K/V cache is q8_0, and the q8_0 cache is what buys the context

| | scratch | K/V cache | state | total | fits in 1907 MiB? |
| --- | ---: | ---: | ---: | ---: | --- |
| F16 | 877 MiB | 2560 MiB | 2687 MiB | 3564 MiB | no, 1657 MiB over |
| q8_0 | 877 MiB | 1360 MiB | 1487 MiB | 2364 MiB | no, 457 MiB over |
| q8_0, right-sized scratch | 114 MiB | 1360 MiB | 1487 MiB | 1764 MiB | yes |

Three changes turned "does not load" into "loads at 131072 tokens":

- The q8_0 K/V cache. A row keeps 53.125% of its F16 bytes.
- The arena's activation scratch now follows the rows a prompt chunk can
  actually use. It was sized for 4096 rows, but a mixture-of-experts chunk
  is bounded by the expert scratch at 64 rows. That was 763 MiB reserved
  for rows no pass could reach.
- The component cache's transfer pool creates its HIP streams when a slot
  is first leased rather than 64 of them at startup. A HIP stream reserves
  a command buffer in the device heap, and the pool was holding 540 MiB
  that one session never used.

The third is not Qwen-specific, and the second applies to every
mixture-of-experts model.

## A q8_0 session cannot restore an F16 checkpoint

The state ABI records the cache format, so a checkpoint taken in one format
is refused by a session in the other rather than restored into it. A
restored cache is read by a kernel that assumes the other format's layout,
and the failure would be silent.

## The write path has to know the format

The q8_0 plane is written by its own encoder. The F16 writers - the fused
QK-norm/RoPE kernel, the decode cache write, the batched prefill write,
the tiled and WMMA packers - must not touch it. Writing F16 values into a
q8_0 plane corrupts it and runs past the end of the buffer.

This is the failure that produced a model that answered correctly at short
context and produced garbage at depth: the first requests were served from
rows the encoder had written, and the corruption appeared once the
attention started reading rows the F16 writers had overwritten.

## The device encoder has to match the host codec exactly

The host codec computes the block scale in binary32 and divides by that
unrounded scale. Dividing by the stored binary16 scale, or by a reciprocal
approximation, changes the tie bytes. The device encoder was checked
against the host codec byte for byte, and the check is a test.

## The MTP head is not this runner's

The model carries an embedded MTP head, and the solver selected it. The
Qwen38/Qwen35moe runner refuses the MTP backend: it is a Qwen Flash-Next
feature, and that runner serves the MTP head. The solver now selects the
embedded head for Flash-Next only, and this model decodes plain.
