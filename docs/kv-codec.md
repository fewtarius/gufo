# KV row codec

## Scope

This document specifies the **row codec**: how one attention K or V row is
stored when it is not stored as F16.

The codec is deliberately small and model-agnostic. It lives in
`src/core/kv_codec/`, has no HIP dependency, and knows nothing about
attention, recurrence, positions or sessions. A model supplies a producer
(where K/V values come from) and a consumer (how that model's attention reads
them); everything between those two points is this document's subject.

F16 storage is the default everywhere and is not described here. This
document covers the *packed* row formats: `q8_0` and `q4_0`.

## Why a codec at all

Attention K/V grows linearly with context and is read on every decode step.
On a **Nimo Axis N161** (Strix Halo: Ryzen AI Max+ 395, Radeon 8060S
`gfx1151`, 124.9 GiB RAM visible, 112.5 GiB GPU-visible) a Laguna S 2.1
conversation at 106k tokens retains 10.6 GB of ordinary K/V - about 99 KB per
token. At the model's 262144-token context that is roughly 26 GB.

`q8_0` stores an ordinary K/V row in 53.125% of the F16 bytes. That is a
memory trade, not a speed one: the codec adds work on write and on read, and
it is lossy. See [Quantization contract](#quantization-contract) for what
the loss looks like and
[docs/models/*/QUALITY.md](models) for per-model measurements.

## Row layouts

Both formats are sequences of fixed-size blocks. There is no inter-block
padding, no row header, and no per-row metadata: a row is exactly
`elements / 32` blocks.

### q8_0

Standard ggml `block_q8_0`: an IEEE binary16 scale followed by 32 signed
quantized values.

```
struct KvBlockQ8_0 {
  std::uint16_t scale;   // F16 bits, little-endian when serialized
  std::int8_t values[32];
};
// sizeof == 34, offsetof(values) == 2
```

A row of `n` elements occupies `n / 32 * 34` bytes. `n` must be a multiple of
32.

### q4_0

Standard ggml `block_q4_0`: an IEEE binary16 scale followed by 16 bytes, each
holding two 4-bit codes. The low nibble holds coordinate 0..15, the high
nibble holds 16..31. Each code is a signed value offset by eight.

```
struct KvBlockQ4_0 {
  std::uint16_t scale;   // F16 bits
  std::uint8_t values[16];
};
// sizeof == 18, offsetof(values) == 2
```

A row of `n` elements occupies `n / 32 * 18` bytes. `n` must be a multiple of
32.

## Quantization contract

Both formats quantize **per block**, independently. No block's scale depends
on any other block.

### q8_0

Given 32 F32 input values `x[0..31]`:

1. `amax = max(|x[i]|)`. A non-finite input is an error.
2. `d = amax / 127` computed in **binary32**.
3. If `d == 0` the block is all zero: `scale = 0`, every `q = 0`. The block is
   returned without further work.
4. `scale` is the F16 (binary16, round-to-nearest-even) representation of `d`.
   A scale that overflows to infinity is an error.
5. `stored_scale` is the binary32 value of that F16 scale.
6. For each `i`: `q = clamp(roundf(x[i] / d), -127, 127)`.

Two details in step 6 carry the whole contract:

- **`roundf` rounds halfway cases away from zero**, not to nearest-even. This
  is the canonical ggml `q8_0` rounding primitive, and it is not the IEEE
  default. `std::round` has the required behaviour; `std::nearbyint` does
  not.
- **The division uses the unrounded binary32 `d`**, never `stored_scale`. A
  reciprocal approximation or a division by the stored scale changes tie
  bytes.

### q4_0

Given 32 F32 input values `x[0..31]`:

1. Find `amax = max(|x[i]|)` and `signed_max`, the input value with the
   largest magnitude, keeping its sign. A non-finite input is an error.
2. `d = signed_max / -8` in binary32. If `d == 0` the block is all zero:
   `scale = 0`, every nibble is `8` (the code for zero).
3. `scale` is the F16 representation of `d`; `inverse = 1 / d` in binary32.
   A non-finite `inverse` or a scale that overflows to infinity is an error.
   **A negative scale is valid for q4_0**, including negative zero.
4. For each pair `j` (coordinates `j` and `j + 16`):
   `shifted = fl32(fl32(x * inverse) + 8.5)`, then
   `code = clamp(truncate(shifted), 0, 15)`.

`fl32` marks an explicit binary32 rounding boundary: the product and the
`+ 8.5` must not contract into an FMA, and the result must be rounded to
binary32 before the next operation.

## Decode contract

### q8_0

For each element: `value = F16(q * stored_scale)`, then promote that F16 to
binary32.

The intermediate rounding to F16 is **not** optional. The stored scale is F16,
so `q * stored_scale` is generally not representable in binary32, and
reconstructing in binary32 instead would produce values the F16-scale format
was never able to represent.

### q4_0

For each element: `value = F16((code - 8) * stored_scale)`, then promote.

Signed zero is preserved: a negative scale with a zero code yields `-0`.

## F16 conversion

The codec needs binary32 to binary16 conversion that does not depend on the
host's floating-point environment:

- Round to nearest, ties to even.
- Independent of the host rounding mode.
- Signed zero and infinity are preserved.
- NaN becomes a signed quiet NaN.
- Finite overflow rounds to infinity.

## Validation

Validation is a separate pass from quantization. It exists so that a payload
read back from disk, a peer, or another process can be rejected before it is
published.

- A payload is rejected if any scale is not finite.
- A q8_0 payload is rejected if it contains the value -128. The encoder never
  produces -128, so it means the bytes are not a q8_0 payload.
- A payload is rejected if any reconstructed element would overflow to F16
  infinity.
- A q4_0 payload is rejected if any reconstructed element would overflow to
  F16 infinity. Its scale may be negative, including negative zero.

Geometry is checked before content: an element count that is not a multiple
of 32, or a buffer length that is not an exact multiple of the block size, is
an error rather than a partial read.

**Validation completes before any output is written.** A rejected call leaves
the destination buffer entirely unchanged.

## Error handling

Every API throws `std::invalid_argument` for:

- invalid geometry (unknown type, element count not a multiple of 32, buffer
  length not an exact multiple of the block size, `size_t` overflow),
- non-finite input or payload,
- a scale that is negative where the format forbids it, or that overflows to
  F16 infinity,
- a q8_0 payload containing -128,
- a reconstruction that overflows to F16 infinity.

`EncodeRow` and `DecodeRow` also reject **overlapping** input and output
buffers. Partial overlap is an error even when the overlap is harmless,
because the check is what makes the "validate before write" guarantee
meaningful.

An empty buffer is accepted only for an empty valid row.

## Rotation

Rotation is **not** part of the codec.

The transform is a Sylvester-order fast Walsh-Hadamard transform applied to
one 256-element head: for K and query it is a single H256; for V and output it
is four contiguous H64 blocks. Normalization comes first (1/16 for H256, 1/8
for H64), then ascending round-to-nearest-even binary32 butterflies with
`high = u - v`.

The transform is not an involution in floating point: a round trip is not
promised to be exact. It requires round-to-nearest and gradual underflow
(FTZ and DAZ clear), and on the CPU it has strict signed-zero semantics.

A rotated row is stored with the same codec as an unrotated one. The codec
never rotates, inverse-rotates or migrates bytes; the producer chooses the
transform and the consumer chooses the inverse. A caller that stores
already-rotated coordinates must not expect the codec to detect or correct it.

A transformed value can be finite and still fail the codec's finite-F16
reconstruction constraint. Rotation and encoding are separate steps, and the
encode step can fail after a successful transform.

## What is deliberately out of scope

- **Attention.** The codec does not select blocks, score candidates, or
  reorder anything.
- **Session, cache and persistence policy.** Which rows are retained, when
  they are captured, and how they are versioned belong to the model and the
  server, not here.
- **Mixed K/V policies.** A session that stores K as q8_0 and V as q4_0
  resolves to two concrete row codecs before it reaches this layer.
- **GPU kernels.** The codec is portable host code. Device-side packing
  belongs to the model that owns the attention path.

## Provenance

The numeric contract above is not new. It was derived in the **Gufetto**
community fork of Gufo (`github.com/fewtarius/gufetto`) for its Flash-Next
q8 KV work, where it is documented in
`src/models/qwen38_flash_next/kv_cache.hpp`. Gufo's implementation here is
written independently against that documented contract, with its own tests,
so that one codec can serve every model in this tree rather than only
Flash-Next.

The two implementations were cross-checked directly: 4500 rows of ordinary,
tiny, huge, zero and random-exponent values were encoded by both and the
bytes and decoded values compared. They are identical, and they accept and
reject the same rows.

The formats themselves are ggml's. `q8_0` is the canonical ggml block format
and its rounding rule is ggml's; `q4_0` is likewise. The Sylvester transform
used for the rotation mode is the standard one, as used by Flash-Next.

Both Gufo and Gufetto are MIT licensed and carry the same
`Copyright (c) 2026 gufo contributors` notice.

## See also

- [internal/cache.md](internal/cache.md) - cross-request cache reuse and the ContinuationCache
- [models/](models) - per-model documentation and measurements
