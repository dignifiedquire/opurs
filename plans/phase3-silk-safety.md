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

### Stage 3.1 — Math/Utility Primitives ✅

- [x] `Inlines.rs` ✓ — safe
- [x] `SigProc_FIX.rs` ✓ — safe
- [x] `macros.rs` ✓ — safe
- [x] `mathops.rs` ✓ — safe
- [x] `sigm_Q15.rs` ✓ — safe
- [x] `lin2log.rs`, `log2lin.rs` ✓ — safe
- [x] `sort.rs` ✓ — safe

### Stage 3.2 — Tables (static data, trivially safe) ✅

- [x] All table modules safe — static const arrays, no unsafe

### Stage 3.3 — Filter/Processing Leaf Modules ✅

- [x] `biquad_alt.rs` ✓ — safe
- [x] `bwexpander.rs` ✓ — safe
- [x] `ana_filt_bank_1.rs` ✓ — safe
- [x] `inner_prod_aligned.rs` ✓ — safe
- [x] `interpolate.rs` ✓ — safe
- [x] `LPC_analysis_filter.rs` ✓ — safe
- [x] `LPC_inv_pred_gain.rs` ✓ — safe

### Stage 3.4 — NLSF Codec (self-contained subsystem) ✅

- [x] `NLSF_stabilize.rs` ✓ — safe
- [x] `NLSF_VQ.rs`, `NLSF_VQ_weights_laroia.rs` ✓ — safe
- [x] `NLSF_unpack.rs` ✓ — safe
- [x] `NLSF_decode.rs` ✓ — safe
- [x] `NLSF_del_dec_quant.rs` ✓ (dig-safe: 0abfdd6, b088d75)
- [x] `NLSF_encode.rs` ✓ (dig-safe: da3c42e)
- [x] `A2NLSF.rs` ✓ — safe
- [x] `process_NLSFs.rs` ✓ (dig-safe: da3c42e)

### Stage 3.5 — Coding Modules ✅

- [x] `code_signs.rs` ✓ — safe
- [x] `shell_coder.rs` ✓ — safe (`#![forbid(unsafe_code)]`)
- [x] `encode_pulses.rs`, `decode_pulses.rs` ✓ — safe
- [x] `encode_indices.rs` ✓ (dig-safe: 0abfdd6), [x] `decode_indices.rs` ✓ — safe
- [x] `gain_quant.rs` ✓ — safe
- [x] `quant_LTP_gains.rs` ✓ (dig-safe: d7966e9)
- [x] `VQ_WMat_EC.rs` ✓ — safe

### Stage 3.6 — Stereo Processing ✅

- [x] `stereo_LR_to_MS.rs` ✓ — safe
- [x] `stereo_find_predictor.rs` ✓ — safe
- [x] `stereo_quant_pred.rs` ✓ — safe
- [x] `stereo_encode_pred.rs` ✓ — safe

### Stage 3.7 — Control/Configuration ✅

- [x] `check_control_input.rs` ✓ — safe
- [x] `control_SNR.rs` ✓ — safe
- [x] `control_audio_bandwidth.rs` ✓ — safe
- [x] `HP_variable_cutoff.rs` ✓ — safe
- [x] `LP_variable_cutoff.rs` ✓ — safe
- [x] `control_codec.rs` ✓ (dig-safe: 4a97b0b) — all 5 functions safe

### Stage 3.8 — Resampler ✅

- [x] All resampler modules safe (`#![forbid(unsafe_code)]` on mod.rs, iir_fir.rs)

### Stage 3.9 — Float Processing (src/silk/float/) ✅

Many of these follow the same pattern: pointer+length → slice.

- [x] `SigProc_FLP.rs`, `structs_FLP.rs` ✓ — safe (data types and helpers)
- [x] `energy_FLP.rs`, `scale_copy_vector_FLP.rs` ✓ — safe
- [x] `autocorrelation_FLP.rs`, `warped_autocorrelation_FLP.rs` ✓ — safe
- [x] `burg_modified_FLP.rs` ✓ — safe
- [x] `schur_FLP.rs`, `k2a_FLP.rs`, `sort_FLP.rs` ✓ — safe
- [x] `apply_sine_window_FLP.rs`, `bwexpander_FLP.rs` ✓ — safe
- [x] `corrMatrix_FLP.rs`, `residual_energy_FLP.rs` ✓ — safe
- [x] `LPC_analysis_filter_FLP.rs` ✓ — safe
- [x] `LTP_analysis_filter_FLP.rs`, `LTP_scale_ctrl_FLP.rs` ✓ — safe
- [x] `find_LPC_FLP.rs` ✓, `find_LTP_FLP.rs` ✓, `find_pitch_lags_FLP.rs` ✓, `find_pred_coefs_FLP.rs` ✓
  - All converted to safe slice APIs (dig-safe)
- [x] `noise_shape_analysis_FLP.rs` ✓ — safe (dig-safe: 93ef28c)
- [x] `pitch_analysis_core_FLP.rs` ✓ — safe (dig-safe: be723e4)
- [x] `process_gains_FLP.rs` ✓ — safe
- [x] `inner_product_FLP.rs` ✓ — safe (dig-safe: b1d21a5, removed unsafe get_unchecked)
- [x] `wrappers_FLP.rs` ✓ — safe (dig-safe: 19b1c2c)
- [x] `encode_frame_FLP.rs` ✓ — safe (dig-safe: a408ae6, NsqConfig extraction)

### Stage 3.10 — Core Quantizers ✅

- [x] `NSQ.rs` ✓ — safe (dig-safe: 0fbb745)
- [x] `NSQ_del_dec.rs` ✓ — safe (dig-safe: c84e14d)

### Stage 3.11 — Decoder Integration ✅

- [x] `decode_frame.rs` ✓ — safe (dig-safe: cf6d7c9)
- [x] `decode_core.rs` ✓ (safe from master)
- [x] `decode_pitch.rs` ✓ — safe
- [x] `PLC.rs` ✓ — fully safe (dig-safe: fd9ea78)
- [x] `CNG.rs` ✓ (safe from master)
- [x] `VAD.rs` ✓ — all 3 functions safe (dig-safe: silk_VAD_GetSA_Q8_c converted)
- [x] `init_decoder.rs` ✓, [x] `decoder_set_fs.rs` ✓ (dig-safe: ab4dea9)
- [x] `dec_API.rs` ✓ — safe (dig-safe: c1d46f4)

### Stage 3.12 — Encoder Integration ✅

- [x] `float/encode_frame_FLP.rs` ✓ — safe (dig-safe: a408ae6)
- [x] `init_encoder.rs` ✓ (dig-safe: ab4dea9) — uses Default::default()
- [x] `enc_API.rs` ✓ — all 4 public functions safe (dig-safe: a5c05ff, then final cleanup)
  - 2 `unsafe` blocks remain: disjoint field borrows for `sStereo.predIx`/`mid_only_flags` via `addr_of_mut!`
  - Zero `unsafe fn` declarations

---

## Definition of Done

- [x] Zero `unsafe fn` declarations in `src/silk/`
- [~] Zero `unsafe {}` blocks in `src/silk/` — 2 remain in `enc_API.rs` (disjoint field borrows, unavoidable without API refactoring of `silk_stereo_LR_to_MS`)
- [ ] Every function has `/// Upstream C:` comment
- [x] All tests pass (cargo test + vector tests) — 92/92 tests, 228/228 vectors
- [ ] Clippy clean, formatted
- [x] `externs::{memcpy,memmove,memset}` no longer called from silk/
