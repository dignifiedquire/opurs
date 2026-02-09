# Phase 3: SILK Module Safety

**Goal**: Make all code under `src/silk/` and `src/silk/float/` free of
`unsafe` blocks and `unsafe fn` declarations. Every function has an
`/// Upstream C:` comment.

**Prerequisites**: Phase 1 (tests). Can proceed in parallel with Phase 2
since CELT and SILK have independent dependency trees.

**Approach**: Bottom-up. Many leaf modules are already safe on the `master`
branch — merge those first, then continue upward.

---

## Already Safe (from master branch or dig-safe work)

These modules have been made safe, either from `master` merges or
direct refactoring on `dig-safe`:

- [x] `silk_PLC_Reset`, `silk_PLC_update` (PLC.rs partial — glue/main PLC still unsafe)
- [x] `silk_sum_sqr_shift`
- [x] `silk_CNG`, `silk_CNG_exc`
- [x] `silk_decode_core`
- [x] `silk_stereo_MS_to_LR`
- [x] `silk_stereo_decode_pred`, `silk_stereo_decode_mid_only`
- [x] `silk_decode_pulses`
- [x] `silk_decode_signs`, `silk_encode_signs`
- [x] `silk_decode_parameters`
- [x] `silk_NLSF2A`
- [x] `silk_LPC_inverse_pred_gain_c`
- [x] `silk_LPC_fit`
- [x] `silk_bwexpander_32`
- [x] `silk_NLSF2A_find_poly`

### Made safe on dig-safe branch

- [x] `silk_encode_indices` (encode_indices.rs)
- [x] `silk_NLSF_del_dec_quant` (NLSF_del_dec_quant.rs)
- [x] `silk_NLSF_encode` (NLSF_encode.rs)
- [x] `silk_process_NLSFs` (process_NLSFs.rs)
- [x] `silk_process_NLSFs_FLP` (float/wrappers_FLP.rs)
- [x] `silk_decoder_set_fs` (decoder_set_fs.rs)
- [x] `silk_VAD_Init` (VAD.rs)
- [x] `silk_VAD_GetNoiseLevels` (VAD.rs)
- [x] `silk_init_encoder` (init_encoder.rs)
- [x] `silk_control_encoder` (control_codec.rs)
- [x] `silk_setup_resamplers` (control_codec.rs)
- [x] `silk_setup_fs` (control_codec.rs)
- [x] `silk_setup_complexity` (control_codec.rs)
- [x] `silk_setup_LBRR` (control_codec.rs)
- [x] `silk_find_LPC_FLP` (float/find_LPC_FLP.rs)
- [x] `silk_quant_LTP_gains` (quant_LTP_gains.rs)
- [x] Various float leaf functions (energy, inner_product, autocorrelation, etc.)
- [x] Default impls for silk_encoder_state, silk_encoder_state_FLP, silk_nsq_state, etc.

---

## Dependency DAG

```
                    enc_API.rs ←── opus_encoder.rs
                  /  |   |  \
    encode_frame_FLP |   |   control_codec.rs
         / | \       |   |       |
   find_*  NSQ  noise_shape  resampler/
     |      |        |
   float helpers   silk math
     |
  SigProc_FIX / Inlines / macros
```

Decoder side:
```
                    dec_API.rs ←── opus_decoder.rs
                  /    |     \
        decode_frame  init_decoder  resampler/
          /    |   \
  decode_core  PLC  decode_parameters
       |        |
  decode_pulses  CNG
```

---

## Stages

### Stage 3.1 — Math/Utility Primitives

- [ ] `Inlines.rs`
  - Upstream C: `silk/Inlines.h`
  - Scope: inline math helpers (CLZ, shifts, saturation)
  - Risk: Low — pure math
- [ ] `SigProc_FIX.rs`
  - Upstream C: `silk/SigProc_FIX.h`
  - Scope: fixed-point signal processing macros/inlines
  - Risk: Low
- [ ] `macros.rs`
  - Upstream C: `silk/macros.h`
  - Scope: multiply-accumulate, saturation macros
  - Risk: Low
- [ ] `mathops.rs` (silk version)
  - Upstream C: `silk/SigProc_FIX.h` (various)
  - Risk: Low
- [ ] `sigm_Q15.rs`
  - Upstream C: `silk/sigm_Q15.c`
  - Risk: Low — lookup table + interpolation
- [ ] `lin2log.rs`, `log2lin.rs`
  - Upstream C: `silk/lin2log.c`, `silk/log2lin.c`
  - Risk: Low
- [ ] `sort.rs`
  - Upstream C: `silk/sort.c`
  - Function: `silk_insertion_sort_increasing`
  - Risk: Low — convert pointer+len to slice
- [ ] **Commit**: `refactor: make silk math/utility primitives safe`

### Stage 3.2 — Tables (static data, trivially safe)

- [ ] `tables_gain.rs`, `tables_LTP.rs`, `tables_NLSF_CB_NB_MB.rs`, `tables_NLSF_CB_WB.rs`, `tables_other.rs`, `tables_pitch_lag.rs`, `tables_pulses_per_block.rs`, `table_LSF_cos.rs`, `pitch_est_tables.rs`
  - Upstream C: `silk/tables_*.c`
  - Risk: Very Low — static const arrays
  - Some may have `unsafe` from c2rust that can simply be removed
- [ ] **Commit**: `refactor: make silk table modules safe`

### Stage 3.3 — Filter/Processing Leaf Modules

- [ ] `biquad_alt.rs`
  - Upstream C: `silk/biquad_alt.c`
  - Function: `silk_biquad_alt_stride1`
  - Risk: Low — IIR filter with state array
- [ ] `bwexpander.rs`
  - Upstream C: `silk/bwexpander.c`
  - Risk: Low — chirp filter
- [ ] `ana_filt_bank_1.rs`
  - Upstream C: `silk/ana_filt_bank_1.c`
  - Risk: Low — analysis filter bank
- [ ] `inner_prod_aligned.rs`
  - Upstream C: `silk/inner_prod_aligned.c`
  - Risk: Low — pointer+len → slice
- [ ] `interpolate.rs`
  - Upstream C: `silk/interpolate.c`
  - Risk: Low
- [ ] `LPC_analysis_filter.rs`
  - Upstream C: `silk/LPC_analysis_filter.c`
  - Risk: Medium — FIR filter with pointer arithmetic
- [ ] `LPC_inv_pred_gain.rs`
  - Upstream C: `silk/LPC_inv_pred_gain.c`
  - Risk: Medium — already partially safe on master
- [ ] **Commit per file or small group**: `refactor: make silk::<module> safe`

### Stage 3.4 — NLSF Codec (self-contained subsystem)

- [ ] `NLSF_stabilize.rs`
  - Upstream C: `silk/NLSF_stabilize.c`
- [ ] `NLSF_VQ.rs`, `NLSF_VQ_weights_laroia.rs`
  - Upstream C: `silk/NLSF_VQ.c`, `silk/NLSF_VQ_weights_laroia.c`
- [ ] `NLSF_unpack.rs`
  - Upstream C: `silk/NLSF_unpack.c`
- [ ] `NLSF_decode.rs`
  - Upstream C: `silk/NLSF_decode.c`
- [x] `NLSF_del_dec_quant.rs` ✓ (dig-safe: 0abfdd6, b088d75)
  - Upstream C: `silk/NLSF_del_dec_quant.c`
  - Risk: Medium — complex quantization with memcpy
- [x] `NLSF_encode.rs` ✓ (dig-safe: da3c42e)
  - Upstream C: `silk/NLSF_encode.c`
- [ ] `A2NLSF.rs`
  - Upstream C: `silk/A2NLSF.c`
  - Risk: Medium — polynomial root finding
- [x] `process_NLSFs.rs` ✓ (dig-safe: da3c42e)
  - Upstream C: `silk/process_NLSFs.c`
- [ ] **Commit**: `refactor: make silk NLSF subsystem safe`

### Stage 3.5 — Coding Modules

- [ ] `code_signs.rs`
  - Upstream C: `silk/code_signs.c`
- [ ] `shell_coder.rs`
  - Upstream C: `silk/shell_coder.c`
- [ ] `encode_pulses.rs`, `decode_pulses.rs`
  - Upstream C: `silk/encode_pulses.c`, `silk/decode_pulses.c`
- [x] `encode_indices.rs` ✓ (dig-safe: 0abfdd6), [ ] `decode_indices.rs`
  - Upstream C: `silk/encode_indices.c`, `silk/decode_indices.c`
- [ ] `gain_quant.rs`
  - Upstream C: `silk/gain_quant.c`
- [x] `quant_LTP_gains.rs` ✓ (dig-safe: d7966e9)
  - Upstream C: `silk/quant_LTP_gains.c`
- [ ] `VQ_WMat_EC.rs`
  - Upstream C: `silk/VQ_WMat_EC.c`
- [ ] **Commit**: `refactor: make silk coding modules safe`

### Stage 3.6 — Stereo Processing

- [ ] `stereo_LR_to_MS.rs` (1 unsafe fn remaining)
  - Upstream C: `silk/stereo_LR_to_MS.c`
  - Risk: Medium — uses memcpy
- [ ] `stereo_find_predictor.rs`
  - Upstream C: `silk/stereo_find_predictor.c`
- [ ] `stereo_quant_pred.rs`
  - Upstream C: `silk/stereo_quant_pred.c`
- [ ] `stereo_encode_pred.rs`
  - Upstream C: `silk/stereo_encode_pred.c`
- [ ] **Commit**: `refactor: make silk stereo processing safe`

### Stage 3.7 — Control/Configuration

- [ ] `check_control_input.rs`
  - Upstream C: `silk/check_control_input.c`
- [ ] `control_SNR.rs`
  - Upstream C: `silk/control_SNR.c`
- [ ] `control_audio_bandwidth.rs`
  - Upstream C: `silk/control_audio_bandwidth.c`
- [ ] `HP_variable_cutoff.rs`
  - Upstream C: `silk/HP_variable_cutoff.c`
- [ ] `LP_variable_cutoff.rs`
  - Upstream C: `silk/LP_variable_cutoff.c`
- [x] `control_codec.rs` ✓ (dig-safe: 4a97b0b) — all 5 functions safe
  - Upstream C: `silk/control_codec.c`
- [x] **Commit**: `refactor: make all control_codec functions safe`

### Stage 3.8 — Resampler

- [ ] `resampler/down2.rs`, `resampler/down2_3.rs`
  - Upstream C: `silk/resampler_down2.c`, `silk/resampler_down2_3.c`
- [ ] `resampler/up2_hq.rs`
  - Upstream C: `silk/resampler_up2_hq.c` (name may vary)
- [ ] `resampler/iir_fir.rs`
  - Upstream C: resampler IIR-FIR implementation
- [ ] `resampler/ar2.rs`
  - Upstream C: `silk/resampler_private_AR2.c`
- [ ] `resampler/mod.rs` (orchestrator)
  - Upstream C: `silk/resampler.c`
  - Risk: Medium — dispatches to sub-resamplers
- [ ] **Commit**: `refactor: make silk resampler safe`

### Stage 3.9 — Float Processing (src/silk/float/)

Many of these follow the same pattern: pointer+length → slice.

- [ ] `SigProc_FLP.rs`, `structs_FLP.rs`
  - Data types and helpers
- [ ] `energy_FLP.rs`, `inner_product_FLP.rs`, `scale_copy_vector_FLP.rs`
  - Upstream C: `silk/float/energy_FLP.c`, etc.
  - Risk: Low — simple math kernels
- [ ] `autocorrelation_FLP.rs`, `warped_autocorrelation_FLP.rs`
  - Upstream C: respective `.c` files
- [ ] `burg_modified_FLP.rs`
  - Upstream C: `silk/float/burg_modified_FLP.c`
  - Risk: Medium — Burg's method with array manipulation
- [ ] `schur_FLP.rs`, `k2a_FLP.rs`, `sort_FLP.rs`
  - Low complexity helpers
- [ ] `apply_sine_window_FLP.rs`, `bwexpander_FLP.rs`
  - Low complexity
- [ ] `corrMatrix_FLP.rs`, `residual_energy_FLP.rs`
  - Medium — matrix operations
- [ ] `LPC_analysis_filter_FLP.rs`
  - Upstream C: `silk/float/LPC_analysis_filter_FLP.c`
  - Risk: Medium — 6 unsafe fn, FIR filter
- [ ] `LTP_analysis_filter_FLP.rs`, `LTP_scale_ctrl_FLP.rs`
  - Medium
- [x] `find_LPC_FLP.rs` ✓ (dig-safe: 6ee1f2d), [ ] `find_LTP_FLP.rs`, [ ] `find_pitch_lags_FLP.rs` (1 unsafe), [ ] `find_pred_coefs_FLP.rs` (1 unsafe)
  - Medium — analysis routines that combine multiple helpers
- [ ] `noise_shape_analysis_FLP.rs` (1 unsafe fn)
  - Upstream C: `silk/float/noise_shape_analysis_FLP.c`
  - Risk: High — complex analysis calling many modules
- [ ] `pitch_analysis_core_FLP.rs` (1 unsafe fn)
  - Upstream C: `silk/float/pitch_analysis_core_FLP.c`
  - Risk: High — 743 lines, heavy pointer arithmetic
- [ ] `process_gains_FLP.rs`
  - Medium
- [ ] `wrappers_FLP.rs` (1 unsafe fn: silk_NSQ_wrapper_FLP)
  - Upstream C: `silk/float/wrappers_FLP.c`
  - Risk: Medium — bridges float/fixed-point, blocked by NSQ safety
- [ ] **Commit per small group**: `refactor: make silk::float::<group> safe`

### Stage 3.10 — Core Quantizers

- [ ] `NSQ.rs` (3 unsafe fn)
  - Upstream C: `silk/NSQ.c`
  - Risk: **Very High** — 572 lines, core noise shaping quantizer, heavy memcpy
- [ ] `NSQ_del_dec.rs` (2 unsafe fn)
  - Upstream C: `silk/NSQ_del_dec.c`
  - Risk: **Very High** — 1065 lines, delayed-decision NSQ variant
- [ ] **Commit**: `refactor: make silk::NSQ safe`
- [ ] **Commit**: `refactor: make silk::NSQ_del_dec safe`

### Stage 3.11 — Decoder Integration

- [ ] `decode_frame.rs` (1 unsafe fn — blocked by PLC)
  - Upstream C: `silk/decode_frame.c`
  - Partial cleanup done: memmove/memcpy replaced with copy_within/copy_from_slice
- [x] `decode_core.rs` ✓ (safe from master)
- [ ] `decode_pitch.rs`
  - Upstream C: `silk/decode_pitch.c`
- [ ] `PLC.rs` (2 unsafe fn: silk_PLC, silk_PLC_glue_frames)
  - Upstream C: `silk/PLC.c`
  - Risk: High — 803 lines, memcpy/memmove heavy
- [x] `CNG.rs` ✓ (safe from master)
- [ ] `VAD.rs` (1 unsafe fn: silk_VAD_GetSA_Q8_c — raw pIn pointer)
  - Upstream C: `silk/VAD.c`
  - silk_VAD_Init and silk_VAD_GetNoiseLevels already safe (dig-safe: ab4dea9)
- [x] `init_decoder.rs` ✓, [x] `decoder_set_fs.rs` ✓ (dig-safe: ab4dea9)
  - Upstream C: respective `.c` files
- [ ] `dec_API.rs` (1 unsafe fn: silk_Decode)
  - Upstream C: `silk/dec_API.c`
  - Risk: High — 490 lines, decoder hub
- [ ] **Commit**: `refactor: make silk decoder pipeline safe`

### Stage 3.12 — Encoder Integration

- [ ] `float/encode_frame_FLP.rs` (2 unsafe fn: silk_encode_do_VAD_FLP, silk_encode_frame_FLP)
  - Upstream C: `silk/float/encode_frame_FLP.c`
  - Risk: High — 571 lines, calls 10+ float modules
  - All float helpers must be safe first (Stage 3.9)
- [x] `init_encoder.rs` ✓ (dig-safe: ab4dea9) — uses Default::default()
  - Upstream C: `silk/init_encoder.c`
- [ ] `enc_API.rs` (3 unsafe fn: silk_Get_Encoder_Size, silk_InitEncoder, silk_Encode)
  - Upstream C: `silk/enc_API.c`
  - Risk: **Very High** — 877 lines, encoder hub
  - Calls: control_codec, encode_indices, encode_pulses, float/encode_frame_FLP, init_encoder, resampler, stereo functions
  - Uses: memcpy, memset
- [ ] **Commit**: `refactor: make silk encoder pipeline safe`

---

## Definition of Done

- [ ] Zero `unsafe fn` declarations in `src/silk/`
- [ ] Zero `unsafe {}` blocks in `src/silk/`
- [ ] Every function has `/// Upstream C:` comment
- [ ] All tests pass (cargo test + vector tests)
- [ ] Clippy clean, formatted
- [ ] `externs::{memcpy,memmove,memset}` no longer called from silk/
