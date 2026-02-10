# Plan: Remove All Remaining Unsafe Code

**Goal**: Eliminate all 47 remaining `unsafe` occurrences (15 `unsafe fn` + 32 `unsafe` blocks)
across 9 files, reaching `#![forbid(unsafe_code)]`.

**Current state**: 47 unsafe occurrences in 9 files.

---

## Inventory

| File | unsafe fn | unsafe {} | Total | Category |
|------|-----------|-----------|-------|----------|
| `src/src/opus_encoder.rs` | 10 | 3 | 13 | Encoder API + internals |
| `src/src/opus_decoder.rs` | 5 | 3 | 8 | Decoder API + internals |
| `src/celt/bands.rs` | 0 | 16 | 16 | Band processing sub-slicing |
| `src/celt/celt_decoder.rs` | 0 | 2 | 2 | Channel view splitting |
| `src/celt/celt_encoder.rs` | 0 | 1 | 1 | energy_mask raw pointer |
| `src/celt/mdct.rs` | 0 | 2 | 2 | ndarray view casting |
| `src/silk/enc_API.rs` | 0 | 2 | 2 | addr_of_mut disjoint borrows |
| `src/util/nalgebra.rs` | 0 | 2 | 2 | nalgebra ViewStorage |
| `src/src/opus.rs` | 0 | 1 | 1 | pointer offset_from |

---

## Stages (ordered by dependency — bottom-up)

### Stage 1: bands.rs — Replace raw pointer sub-slicing (16 unsafe blocks)

**Problem**: `quant_all_bands` takes raw pointers `x_ptr`, `y_ptr`, `norm_ptr`
and uses `from_raw_parts_mut` to create overlapping mutable sub-slices of
the same buffer at different offsets per band iteration. The C code relies on
the fact that each band's X/Y region and its lowband/scratch regions are
non-overlapping within the same allocation.

**Strategy**: Replace raw pointers with index-based sub-slicing.

1. Change `x_ptr: *mut f32` parameters to pass the full `&mut [f32]` slice
   and use `split_at_mut` or index ranges to carve out non-overlapping regions.
2. For the `_norm` buffer where we need simultaneous `lowband` (read) and
   `lowband_out` (write) sub-slices: use `split_at_mut` at the boundary
   between the two regions, since `effective_lowband + N <= norm_band_out_off`
   is guaranteed by the algorithm.
3. For the scratch buffer that aliases `X_` at `decode_scratch_off`: use
   `split_at_mut` on `X_` to separate the scratch region (at the end) from
   the band regions (at the beginning), since `decode_scratch_off = M*eBands[nbEBands-1]`
   is always past all band data.
4. For the `x_band` / `y_band` sub-slices: simple `&mut x_buf[band_start..band_start+n]`.

**Specific changes**:
- Refactor `quant_all_bands` signature: replace `*mut f32` with `&mut [f32]`
  for X, Y, norm buffers
- Split X buffer into `(band_data, scratch)` via `split_at_mut(decode_scratch_off)`
  at the top of the loop
- Split norm buffer into `(lowband_region, lowband_out_region)` where needed
- Update all 16 `from_raw_parts_mut` calls to safe index slicing
- Update callers in `celt_encoder.rs` and `celt_decoder.rs`

**Risk**: Medium — complex overlapping buffer logic, must verify non-overlap
invariants carefully. Run vector tests after every sub-step.

---

### Stage 2: celt_decoder.rs — Safe channel view array (2 unsafe blocks)

**Problem 1** (line ~390): Creates `[&mut [celt_sig]; 2]` array from two
already-split mutable slices by going through raw pointers, because Rust
can't put two `&mut` slices from the same `split_at_mut` into an array
without the borrow checker complaining.

**Strategy**: The two slices `out_syn_ch0` and `out_syn_ch1` come from
`split_at_mut` and are already disjoint. Restructure the code to avoid
needing them in an array — either:
- (a) Use a helper that takes the two slices and an index, returning the
  right one (avoids the array entirely), or
- (b) Move the slices into the array directly after `split_at_mut` since
  they're separate bindings (this should work — the issue is likely that
  they're re-borrowed from parameters; if so, restructure the caller).

**Problem 2** (line ~910): `from_raw_parts_mut` on a local `Vec`'s pointer.
This is trivially replaced with `&mut data_copy[..]`.

**Risk**: Low.

---

### Stage 3: celt_encoder.rs — Replace energy_mask raw pointer (1 unsafe block)

**Problem**: `OpusCustomEncoder.energy_mask` is `*const opus_val16`, set via
CTL from `opus_encoder.rs`. The CELT encoder reads it as
`from_raw_parts(st.energy_mask, len)`.

**Strategy**: Change `energy_mask: *const opus_val16` to
`energy_mask: Option<&[opus_val16]>` (with appropriate lifetime, or use
a `Vec<opus_val16>` copy). Since the mask is set once per encode call via
`OPUS_SET_ENERGY_MASK_REQUEST` and read within the same call, a slice
reference with appropriate scoping should work. If lifetime issues arise,
copy the mask data into a `Vec` field on the encoder.

**Specific changes**:
- Change field type in `OpusCustomEncoder`
- Update the CTL setter in `celt_encoder.rs` (OPUS_SET_ENERGY_MASK_REQUEST)
- Update the CTL setter in `opus_encoder.rs` (OPUS_SET_ENERGY_MASK_REQUEST)  
- Update the reader in `celt_encode_with_ec`
- Remove the `from_raw_parts` call

**Risk**: Medium — lifetime management across the encode call boundary.
May need to restructure how the mask flows from opus_encoder to celt_encoder.

---

### Stage 4: mdct.rs — Safe ndarray view splitting (2 unsafe blocks)

**Problem**: `split_interleaving_opposite` and `_mut` create strided ndarray
views via `raw_view()` → `deref_into_view()`. The unsafe is needed because
ndarray doesn't provide a safe API for creating two strided views that are
provably disjoint.

**Strategy options**:
- (a) Use `ndarray::Zip` or manual iteration to avoid needing simultaneous
  views — process elements in-place with index arithmetic.
- (b) Keep the unsafe but wrap it in a well-documented `# Safety` comment
  and move it into a dedicated helper module with unit tests proving
  disjointness. Mark the containing functions as safe (the unsafe is
  internal implementation detail).
- (c) Use `split_at` on the underlying slice, then construct two separate
  ArrayViews from the halves with appropriate strides.

**Recommended**: Option (c) if feasible, otherwise (b) with thorough safety
documentation and test coverage. The interleaving pattern (even indices
forward, odd indices backward) guarantees disjointness.

**Risk**: Low — well-contained, clear disjointness invariant.

---

### Stage 5: silk/enc_API.rs — Safe disjoint array borrows (2 unsafe blocks)

**Problem**: `silk_Encode` needs `&mut psEnc.sStereo` for `silk_stereo_LR_to_MS`
while simultaneously borrowing `&mut psEnc.sStereo.predIx[nfe]` and
`&mut psEnc.sStereo.mid_only_flags[nfe]`. Uses `addr_of_mut!` to work
around the borrow checker.

**Strategy**: Restructure `silk_stereo_LR_to_MS` to take `predIx` and
`mid_only_flags` outputs as separate parameters instead of reading/writing
them through the `&mut stereo_state` reference. This eliminates the
overlapping borrow entirely.

Alternative: Extract `predIx` and `mid_only_flags` from `sStereo` into
separate fields on the parent struct, so they can be borrowed independently.

**Risk**: Low — straightforward parameter refactoring.

---

### Stage 6: util/nalgebra.rs — Safe matrix view construction (2 unsafe blocks)

**Problem**: `ViewStorageMut::from_raw_parts` and `ViewStorage::from_raw_parts`
require unsafe because nalgebra's API doesn't provide a safe constructor
for row-major views from slices.

**Strategy**: Check if nalgebra provides `MatrixSlice::from_slice` or
similar safe constructors. If so, use those. If not:
- Use `nalgebra::DMatrix::from_row_slice` for owned matrices
- Or use `MatrixView::from_slice_generic` which is safe in recent nalgebra
  versions (check the version we depend on)
- Or wrap the unsafe in a clearly-documented helper with bounds assertions
  (already has `assert!` checks) and accept this as "safe enough" until
  nalgebra provides better APIs

**Risk**: Low — bounds are already checked via assert.

---

### Stage 7: opus.rs — Replace pointer offset_from (1 unsafe block)

**Problem**: `opus_packet_parse_impl` uses `fp.offset_from(dp)` to calculate
the byte offset of a frame pointer within the packet data slice.

**Strategy**: Replace with index tracking. Instead of computing pointer
differences, track the current parse position as a `usize` index into the
slice and compute frame offsets as `current_index - start_index`.

**Risk**: Low — straightforward index arithmetic replacement.

---

### Stage 8: opus_decoder.rs — Safe decode path (5 unsafe fn + 3 unsafe blocks)

**Problem**: The main decode path uses `unsafe fn` for `opus_decode_frame`,
`opus_decode_native`, and the legacy C API functions.

**Unsafe fn declarations**:
1. `opus_decoder_init` — deprecated, raw pointer deref → already delegates to `OpusDecoder::new()`
2. `opus_decoder_create` — deprecated, `Box::into_raw` → keep as deprecated unsafe
3. `opus_decoder_destroy` — deprecated, `Box::from_raw` → keep as deprecated unsafe
4. `opus_decode_frame` — internal, takes `&mut OpusDecoder` + `Option<&[u8]>` + `&mut [opus_val16]`
5. `opus_decode_native` — internal, orchestrates multi-frame decode

**Unsafe blocks**:
1. `from_raw_parts_mut` on `data.as_ptr() as *mut u8` for ec_dec init (line ~263)
2. `opus_decode_native()` call from `opus_decode` (line ~824)
3. `opus_decode_native()` call from `opus_decode_float` (line ~843)

**Strategy**:
1. Make `opus_decode_frame` safe:
   - The `from_raw_parts_mut` for ec_dec init: use `to_vec()` + pass `&mut vec` 
     (or restructure ec_dec to accept `&[u8]` for init since it only needs read access initially)
2. Make `opus_decode_native` safe by converting its signature to take
   `&mut OpusDecoder`, `&mut [opus_val16]`, etc.
3. Remove `unsafe` from `opus_decode` and `opus_decode_float` once
   `opus_decode_native` is safe
4. Keep deprecated functions (`opus_decoder_init/create/destroy`) as `unsafe`
   — they exist only for backward compat and will be removed in Stage 11

**Risk**: Medium — decode path is complex but already mostly uses safe types.

---

### Stage 9: opus_encoder.rs — Safe encode path (10 unsafe fn + 3 unsafe blocks)

This is the largest and most complex stage.

**Unsafe fn declarations**:
1. `downmix_float` — raw pointer arithmetic on `*const c_void`
2. `downmix_int` — raw pointer arithmetic on `*const c_void`
3. `encode_multiframe_packet` — calls `opus_encode_native`
4. `opus_encode_native` — main encode function (~1300 lines), raw pointer params
5. `opus_encode` — public i16 API, calls `opus_encode_native`
6. `opus_encode_float` — public f32 API, calls `opus_encode_native`
7. `opus_encoder_ctl_impl` — VarArgs dispatch, dereferences `*mut OpusEncoder`
8. `opus_encoder_init` — deprecated, dereferences raw pointer
9. `opus_encoder_create` — deprecated, `Box::into_raw`
10. `opus_encoder_destroy` — deprecated, `Box::from_raw`

**Unsafe blocks** (inside `opus_encode_native`):
1. `std::mem::zeroed()` in `OpusEncoder::new()` (line ~241)
2. `from_raw_parts_mut` for redundancy data (line ~2240)
3. `from_raw_parts_mut` for redundancy data (line ~2340)
4. `std::ptr::copy` for data relocation (line ~2317)
5. `offset_from` for reset state (line ~2867)

**Strategy — bottom-up**:

**Step 9a: Safe downmix functions**
- Replace `downmix_float` signature: take `&[f32]` instead of `*const c_void`
- Replace `downmix_int` signature: take `&[i16]` instead of `*const c_void`
- Change `downmix_func` type alias from `unsafe fn(*const c_void, ...)` to
  a safe enum or trait:
  ```rust
  enum DownmixSource<'a> {
      Float(&'a [f32]),
      Int(&'a [i16]),
  }
  ```
  Then a single safe `downmix` function handles both cases.
- Update `opus_encode_native`, `opus_encode`, `opus_encode_float`,
  `analysis.rs` callers

**Step 9b: Safe opus_encode_native**
- Change signature: `*mut OpusEncoder` → `&mut OpusEncoder`
- Change `pcm: *const opus_val16` → `pcm: &[f32]`
- Change `data: *mut u8` → `data: &mut [u8]`
- Replace `data.offset(nb_compr_bytes)` with `&mut data[nb_compr_bytes..]`
- Replace `std::ptr::copy` with `data.copy_within(..)`
- Replace `from_raw_parts_mut` for redundancy slices with index slicing
- The `*data.offset(1)` etc. become `data[1]`
- This is the hardest part — 1300+ lines of pointer-heavy code

**Step 9c: Safe encode_multiframe_packet**
- Change signature to take `&mut OpusEncoder`, `&[f32]`, `&mut [u8]`
- Delegates to safe `opus_encode_native`

**Step 9d: Safe opus_encode / opus_encode_float**
- Change signatures to take `&mut OpusEncoder`, slices
- Delegate to safe `opus_encode_native`

**Step 9e: Safe opus_encoder_ctl_impl**
- Change `*mut OpusEncoder` → `&mut OpusEncoder`
- Replace VarArgs with typed method dispatch (or at minimum, remove the
  raw pointer deref)
- The `OPUS_RESET_STATE` handler uses `ptr::write_bytes` to zero a range
  of the struct — replace with field-by-field reset
- The `OPUS_SET_ENERGY_MASK_REQUEST` handler passes a raw pointer — fix
  after Stage 3 changes the field type

**Step 9f: Replace mem::zeroed in OpusEncoder::new**
- Implement `Default` for `OpusEncoder` (all fields are primitive/Copy)
- Or use field-by-field initialization

**Risk**: High — this is the largest, most complex stage. Break into
multiple commits, test after each sub-step.

---

### Stage 10: Implement safe public API wrappers

Implement the `Encoder`, `Decoder`, `Repacketizer`, `SoftClip` wrapper
structs defined in `phase4-integration-safety.md` stages 4.4-4.7.

These wrap the now-safe internal functions with typed parameters and
`Result` returns. This stage has no new unsafe code to eliminate — it's
about providing the final public API.

**Risk**: Low — wrapper code only.

---

### Stage 11: Final cleanup

1. Remove deprecated `unsafe fn` C-API functions (opus_encoder_create/init/destroy,
   opus_decoder_create/init/destroy)
2. Remove `varargs.rs` (no longer needed after CTL methods are typed)
3. Add `#![forbid(unsafe_code)]` to `lib.rs`
4. Verify: `cargo build`, `cargo test --all`, `cargo clippy`, vector tests

**Risk**: Low — deletion and verification only.

---

## Execution Order & Dependencies

```
Stage 1 (bands.rs)          ──┐
Stage 2 (celt_decoder.rs)   ──┤
Stage 3 (celt_encoder.rs)   ──┤── All independent, can be done in any order
Stage 4 (mdct.rs)            ──┤
Stage 5 (enc_API.rs)         ──┤
Stage 6 (nalgebra.rs)        ──┤
Stage 7 (opus.rs)            ──┘
                               │
Stage 8 (opus_decoder.rs)   ───┤── Depends on Stage 2 (celt_decoder changes)
                               │
Stage 9 (opus_encoder.rs)   ───┤── Depends on Stages 1, 3, 5 (callee changes)
                               │
Stage 10 (public API)        ──┤── Depends on Stages 8, 9
                               │
Stage 11 (forbid unsafe)    ───┘── Depends on all above
```

---

## Estimated Commit Count

| Stage | Commits | Description |
|-------|---------|-------------|
| 1 | 2-3 | bands.rs sub-slicing refactor |
| 2 | 1 | celt_decoder channel views |
| 3 | 1 | energy_mask pointer → slice |
| 4 | 1 | mdct view splitting |
| 5 | 1 | enc_API disjoint borrows |
| 6 | 1 | nalgebra view construction |
| 7 | 1 | opus.rs offset_from |
| 8 | 2 | decoder safe path |
| 9 | 4-6 | encoder safe path (largest) |
| 10 | 2-3 | public API wrappers |
| 11 | 1 | forbid(unsafe_code) |
| **Total** | **~18-21** | |

---

## Testing Strategy

After every stage:
1. `cargo build` — must compile
2. `cargo test --all` — all unit + integration tests pass
3. `cargo clippy` — no warnings
4. Vector tests (for stages touching encoder/decoder/codec internals):
   `cargo run --release -p unsafe-libopus-tools --bin run_vectors2 -- opus_newvectors`

---

## What We Start With

I recommend starting with **Stages 1-7 in parallel** (the independent
bottom-up work on CELT/SILK/util), then moving to **Stage 8** (decoder),
then **Stage 9** (encoder — the hardest), then **Stages 10-11** (API + cleanup).

The single hardest piece is **Stage 9b** (making `opus_encode_native` safe) —
a 1300-line function with pervasive raw pointer usage. This will likely need
to be broken into multiple sub-commits.
