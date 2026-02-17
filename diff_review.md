# Upstream Parity Diff Review

## Scope
Rust sources compared against upstream C in `libopus-sys/opus`.

## Findings

1. [HIGH] CELT PLC attenuation ratio mismatch.
- Rust: `src/celt/celt_decoder.rs:983`
- Upstream: `libopus-sys/opus/celt/celt_decoder.c:1002`
- Detail: Rust uses `celt_sqrt((S1 + 1.0) / (S2 + 1.0))`; upstream uses halved S1 before ratio (`SHR32(S1,1)+1`), equivalent to `celt_sqrt((0.5*S1 + 1)/(S2 + 1))`.

2. [HIGH] Missing restricted application mode parity in encoder.
- Rust: `src/opus/opus_encoder.rs:162`, `src/opus/opus_encoder.rs:1878`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:219`, `libopus-sys/opus/src/opus_encoder.c:1467`, `libopus-sys/opus/src/opus_encoder.c:1470`
- Detail: Rust path does not mirror `OPUS_APPLICATION_RESTRICTED_SILK` and `OPUS_APPLICATION_RESTRICTED_CELT` control-flow branches present upstream.

3. [HIGH][QEXT] `run_prefilter` does not use qext-scaled periods/memory layout.
- Rust: `src/celt/celt_encoder.rs:1697`, `src/celt/celt_encoder.rs:1704`, `src/celt/celt_encoder.rs:1754`, `src/celt/celt_encoder.rs:1953`
- Upstream: `libopus-sys/opus/celt/celt_encoder.c:1418`, `libopus-sys/opus/celt/celt_encoder.c:1423`, `libopus-sys/opus/celt/celt_encoder.c:1469`, `libopus-sys/opus/celt/celt_encoder.c:1590`
- Detail: Rust uses fixed `COMBFILTER_MAXPERIOD/MINPERIOD`; upstream uses `QEXT_SCALE(...)` in this path.

4. [HIGH][QEXT] Overlap copy into encoder input uses unscaled combfilter period.
- Rust: `src/celt/celt_encoder.rs:2403`
- Upstream: `libopus-sys/opus/celt/celt_encoder.c:2017`
- Detail: Rust indexes via `COMBFILTER_MAXPERIOD`; upstream indexes via `QEXT_SCALE(COMBFILTER_MAXPERIOD)`.

5. [HIGH][QEXT] Decoder PLC/buffer path appears unscaled vs upstream.
- Rust: `src/celt/celt_decoder.rs:577`, `src/celt/celt_decoder.rs:598`, `src/celt/celt_decoder.rs:653`
- Upstream: `libopus-sys/opus/celt/celt_decoder.c:704`, `libopus-sys/opus/celt/celt_decoder.c:705`, `libopus-sys/opus/celt/celt_decoder.c:573`
- Detail: Rust uses fixed `DECODE_BUFFER_SIZE` assumptions in PLC path where upstream uses `QEXT_SCALE(DECODE_BUFFER_SIZE)` and `QEXT_SCALE(MAX_PERIOD)`.

6. [MEDIUM][QEXT] QEXT payload not wired in main encoder CELT call path.
- Rust: `src/opus/opus_encoder.rs:2818`
- Upstream refs: `libopus-sys/opus/src/opus_encoder.c:1746`, `libopus-sys/opus/src/opus_encoder.c:1792`, `libopus-sys/opus/src/opus_encoder.c:2394`
- Detail: Rust currently passes `None` payload with TODO, preventing upstream-equivalent QEXT framing/entropy path.

7. [MEDIUM] Missing upstream application constants in Rust defines.
- Rust: `src/opus/opus_defines.rs`
- Upstream: `libopus-sys/opus/include/opus_defines.h:224`, `libopus-sys/opus/include/opus_defines.h:226`
- Detail: `OPUS_APPLICATION_RESTRICTED_SILK` (2052) and `OPUS_APPLICATION_RESTRICTED_CELT` (2053) are defined upstream but absent in Rust constants, contributing to encoder mode-parity gaps.

8. [LOW] Missing ignore-extensions CTL request constants in Rust defines.
- Rust: `src/opus/opus_defines.rs`
- Upstream: `libopus-sys/opus/include/opus_defines.h` (`OPUS_SET_IGNORE_EXTENSIONS_REQUEST`=4058, `OPUS_GET_IGNORE_EXTENSIONS_REQUEST`=4059)
- Detail: Upstream defines these CTLs, but Rust constant surface currently omits them.

9. [LOW] Additional upstream CTL constants missing in Rust defines.
- Rust: `src/opus/opus_defines.rs`
- Upstream: `libopus-sys/opus/include/opus_defines.h`
- Missing names (present upstream, absent in Rust constants):
  - `OPUS_SET_QEXT_REQUEST`
  - `OPUS_GET_QEXT_REQUEST`
  - `OPUS_SET_OSCE_BWE_REQUEST`
  - `OPUS_GET_OSCE_BWE_REQUEST`
- Detail: API surface parity for CTL constants is incomplete.

10. [MEDIUM] Missing `opus_encode24` API parity.
- Rust: `src/opus/opus_encoder.rs` (has `opus_encode`, `opus_encode_float` only)
- Upstream: `libopus-sys/opus/src/opus_encoder.c:2697`, `libopus-sys/opus/src/opus_encoder.c:2706`
- Detail: Upstream exports 24-bit integer encode API (`opus_encode24`); Rust API surface currently does not provide a matching entry point.

11. [MEDIUM] Missing `opus_decode24` API parity.
- Rust: `src/opus/opus_decoder.rs` (has `opus_decode`, `opus_decode_float` only)
- Upstream: `libopus-sys/opus/src/opus_decoder.c:937`, `libopus-sys/opus/src/opus_decoder.c:946`
- Detail: Upstream exports 24-bit integer decode API (`opus_decode24`); Rust API surface currently does not provide a matching entry point.

12. [MEDIUM][DRED] Missing decoder DRED API entry points.
- Rust: `src/opus/opus_decoder.rs`
- Upstream: `libopus-sys/opus/src/opus_decoder.c:1609`, `libopus-sys/opus/src/opus_decoder.c:1643`, `libopus-sys/opus/src/opus_decoder.c:1677`
- Detail: Upstream exports `opus_decoder_dred_decode`, `opus_decoder_dred_decode24`, and `opus_decoder_dred_decode_float`; matching public Rust entry points were not found.

13. [LOW] C-style `opus_encoder_ctl` / `opus_decoder_ctl` API entry points not mirrored as direct functions.
- Rust: `src/opus/opus_encoder.rs`, `src/opus/opus_decoder.rs`
- Upstream: `libopus-sys/opus/src/opus_encoder.c` (`opus_encoder_ctl`), `libopus-sys/opus/src/opus_decoder.c:1031` (`opus_decoder_ctl`)
- Detail: Rust provides typed setters/getters on structs, but direct C-style variadic CTL function parity is not present.

14. [HIGH][Deep PLC/QEXT] Missing 96 kHz guard for neural PLC selection in CELT decoder loss path.
- Rust: `src/celt/celt_decoder.rs:663-667`
- Upstream: `libopus-sys/opus/celt/celt_decoder.c` (condition around `curr_frame_type = FRAME_PLC_NEURAL` includes `st->mode->Fs != 96000`)
- Detail: Upstream explicitly avoids Deep PLC path at 96 kHz; Rust selection condition currently omits this guard.

15. [HIGH][QEXT] `run_prefilter` tone/pitch scaling logic diverges from upstream.
- Rust: `src/celt/celt_encoder.rs:1720-1731`, `src/celt/celt_encoder.rs:1756-1767`
- Upstream: `libopus-sys/opus/celt/celt_encoder.c:1440-1453`, `libopus-sys/opus/celt/celt_encoder.c:1475-1479`
- Detail: Upstream applies `QEXT_SCALE(tone_freq)` in tone-threshold logic and divides `pitch_index` by `qext_scale`; Rust currently uses unscaled tone-frequency comparisons and does not apply `pitch_index /= qext_scale`.

16. [HIGH][QEXT] `celt_plc_pitch_search` return value is not QEXT-scaled.
- Rust: `src/celt/celt_decoder.rs:574-590`
- Upstream: `libopus-sys/opus/celt/celt_decoder.c:556-573`
- Detail: Upstream returns `QEXT_SCALE(pitch_index)`; Rust currently returns unscaled `pitch_index`.

17. [HIGH][QEXT] PLC pitch-search downsampling inputs are not QEXT-aware.
- Rust: `src/celt/celt_decoder.rs:575-588`
- Upstream: `libopus-sys/opus/celt/celt_decoder.c:567-571`
- Detail: Upstream uses QEXT-aware sizing/stride in `pitch_downsample(..., QEXT_SCALE(2), ...)`; Rust currently drives downsampling with fixed `DECODE_BUFFER_SIZE`-based lengths.

18. [MEDIUM] Decoder `ignore_extensions` behavior/CTL parity missing.
- Rust: `src/opus/opus_decoder.rs` (no `ignore_extensions` field/ctl path found)
- Upstream: `libopus-sys/opus/src/opus_decoder.c` (uses `st->ignore_extensions` in parse path and exposes ignore-extensions CTLs)
- Detail: Upstream can ignore packet extensions at decode time via CTL/state; matching Rust control surface/behavior was not found.

19. [MEDIUM] Multistream/projection API surface parity missing.
- Rust: `src/` (no `opus_multistream_*` / `opus_projection_*` implementation files found)
- Upstream: `libopus-sys/opus/src/opus_multistream_encoder.c`, `libopus-sys/opus/src/opus_multistream_decoder.c`, `libopus-sys/opus/src/opus_projection_encoder.c`, `libopus-sys/opus/src/opus_projection_decoder.c`
- Detail: Upstream includes full multistream/projection encoder/decoder APIs (including 24-bit variants and ctl paths); matching Rust API surface is not present.

20. [LOW] Multistream packet pad/unpad helpers parity missing.
- Rust: `src/opus/repacketizer.rs` (single-stream repacketizer only; notes no multistream manipulation)
- Upstream: `libopus-sys/opus/src/repacketizer.c` (`opus_multistream_packet_pad`, `opus_multistream_packet_unpad`)
- Detail: Upstream multistream packet pad/unpad helpers are not mirrored as Rust public functions.

21. [HIGH][QEXT] CELT encoder prefilter memory/state sizing is not qext-scaled.
- Rust: `src/celt/celt_encoder.rs:113`, `src/celt/celt_encoder.rs:212`, `src/celt/celt_encoder.rs:277`
- Upstream: `libopus-sys/opus/celt/celt_encoder.c:167` (+ related QEXT-scale use sites)
- Detail: Rust prefilter memory/state handling is dimensioned and reset with fixed `COMBFILTER_MAXPERIOD`; upstream scales combfilter-period-dependent state by `QEXT_SCALE(COMBFILTER_MAXPERIOD)` for 96 kHz QEXT operation.

22. [HIGH][QEXT] CELT decoder loss-concealment core still uses hardcoded 2048/1024 instead of scaled buffer/period.
- Rust: `src/celt/celt_decoder.rs:853-857`, `src/celt/celt_decoder.rs:869-871`, `src/celt/celt_decoder.rs:894`, `src/celt/celt_decoder.rs:926-927`, `src/celt/celt_decoder.rs:941`, `src/celt/celt_decoder.rs:950`
- Upstream: `libopus-sys/opus/celt/celt_decoder.c:835`, `libopus-sys/opus/celt/celt_decoder.c:851-853`, `libopus-sys/opus/celt/celt_decoder.c:913-916`, `libopus-sys/opus/celt/celt_decoder.c:934`, `libopus-sys/opus/celt/celt_decoder.c:939`, `libopus-sys/opus/celt/celt_decoder.c:958`
- Detail: Upstream computes and uses `decode_buffer_size = QEXT_SCALE(DECODE_BUFFER_SIZE)` and `max_period = QEXT_SCALE(MAX_PERIOD)` throughout PLC extrapolation/energy logic. Rust still hardcodes `2048`/`1024` in multiple indexing and shift/copy sites, diverging for QEXT scale 2.

23. [HIGH][QEXT] Stereo split in `quant_band` uses wrong `qext_extra` basis in one branch.
- Rust: `src/celt/bands.rs:1707-1712`, `src/celt/bands.rs:1763-1789`
- Upstream: `libopus-sys/opus/celt/bands.c:1517-1519`, `libopus-sys/opus/celt/bands.c:1537-1539`
- Detail: Rust computes `qext_extra` once from `mbits` and reuses it in both branch paths. Upstream computes `qext_extra` from `mbits` in the `mbits >= sbits` branch and from `sbits` in the opposite branch. This changes QEXT side/mid bit reallocation and entropy path.

24. [HIGH][QEXT] Decoder state buffer sizing is fixed to 48 kHz assumptions.
- Rust: `src/celt/celt_decoder.rs:75`, `src/celt/celt_decoder.rs:183`
- Upstream: `libopus-sys/opus/celt/celt_decoder.c:201` (size formula includes `QEXT_SCALE(DECODE_BUFFER_SIZE)` and runtime overlap)
- Detail: Rust allocates `decode_mem` as `2 * (DECODE_BUFFER_SIZE + 120)`. Upstream sizing scales decode buffer for QEXT and uses mode overlap; 96 kHz path needs larger state footprint. This can alter history indexing and PLC behavior under QEXT.

25. [HIGH][QEXT] `pitch_downsample` API/behavior is fixed to factor-2 and cannot represent upstream QEXT factor-4 path.
- Rust: `src/celt/pitch.rs:271-314`
- Upstream: `libopus-sys/opus/celt/pitch.c:140-142` (takes `C`, `factor`, `arch`), call-site uses `QEXT_SCALE(2)` in QEXT decode path (`libopus-sys/opus/celt/celt_decoder.c:567-571`)
- Detail: Rust downsampler infers channels from slice count and hardcodes stride/filtering for factor 2 (`2*i±1` indexing). Upstream uses a parameterized factor and explicitly passes scaled factor under QEXT. This creates a structural mismatch in PLC pitch preprocessing at 96 kHz.

26. [HIGH][QEXT] `qext_scale` state is not updated from mode/frame configuration in Rust CELT encoder/decoder.
- Rust: `src/celt/celt_encoder.rs:127`, `src/celt/celt_encoder.rs:220`, `src/celt/celt_decoder.rs:83`, `src/celt/celt_decoder.rs:190` (defaulted to 1; no matching init/update path found)
- Upstream: `libopus-sys/opus/celt/celt_encoder.c:224-225`, `libopus-sys/opus/celt/celt_decoder.c:268-269`
- Detail: Upstream sets `st->qext_scale = 2` for 96 kHz-compatible short MDCT modes and 1 otherwise. Rust code consumes `qext_scale` in multiple QEXT decisions but appears to leave it at default 1, causing wrong scaling/mode decisions for 96 kHz QEXT paths.

27. [MEDIUM][QEXT] CELT reset paths do not clear QEXT old-band history arrays in Rust.
- Rust: `src/celt/celt_encoder.rs:233-282`, `src/celt/celt_decoder.rs:206-228`
- Upstream: `libopus-sys/opus/celt/celt_encoder.c:3074-3085`, `libopus-sys/opus/celt/celt_decoder.c:1790-1806`
- Detail: Upstream `OPUS_RESET_STATE` clears full runtime state from `ENCODER_RESET_START` / `DECODER_RESET_START`, which includes QEXT runtime histories. Rust reset methods currently clear core band histories but do not reset `qext_oldBandE`, leaving stale QEXT energy history across resets.

28. [HIGH][QEXT] CELT decoder validation hardcodes 48 kHz mode and rejects 96 kHz QEXT mode.
- Rust: `src/celt/celt_decoder.rs:101-103`
- Upstream: `libopus-sys/opus/celt/celt_decoder.c` validation permits configured mode/state (including QEXT mode paths), not a fixed 48 kHz assertion.
- Detail: Rust `validate_celt_decoder()` asserts `st.mode == opus_custom_mode_create(48000, 960, ...)`, which is incompatible with 96 kHz/QEXT decoder operation and can trip asserts during decode on valid 96 kHz state.

29. [HIGH][QEXT] CELT encoder overlap history buffer is fixed-size instead of mode-overlap-sized.
- Rust: `src/celt/celt_encoder.rs:111`, `src/celt/celt_encoder.rs:276`
- Upstream: `libopus-sys/opus/celt/celt_encoder.c:136` and size computation in `libopus-sys/opus/celt/celt_encoder.c:167`
- Detail: Upstream allocates `in_mem` as `channels*mode->overlap` in the variable tail of the encoder state. Rust fixes `in_mem` to `2*120`, which does not scale with 96 kHz overlap (`240`) and diverges from upstream state layout/behavior.

30. [HIGH][QEXT] CELT decoder uses fixed-size synthesis/prefilter scratch buffers where upstream is runtime-sized.
- Rust: `src/celt/celt_decoder.rs:369`, `src/celt/celt_decoder.rs:394`, `src/celt/celt_decoder.rs:437`, `src/celt/celt_decoder.rs:604`
- Upstream: `libopus-sys/opus/celt/celt_decoder.c:432`, `libopus-sys/opus/celt/celt_decoder.c:597`
- Detail: Rust hardcodes `freq`/`freq2` to `960(+7)` and `etmp` to `120`. Upstream allocates `freq` with size `N` and `etmp` with size `overlap`. For 96 kHz/QEXT operation (`N=1920`, `overlap=240`) this diverges from upstream buffer model and risks truncation/index errors.

31. [HIGH][QEXT] Decoder synthesis path does not pass/apply QEXT denormalisation inputs.
- Rust: `src/celt/celt_decoder.rs:341`, `src/celt/celt_decoder.rs:1787`
- Upstream: `libopus-sys/opus/celt/celt_decoder.c:416`, `libopus-sys/opus/celt/celt_decoder.c:456-458`, `libopus-sys/opus/celt/celt_decoder.c:480-483`, `libopus-sys/opus/celt/celt_decoder.c:496-498`, `libopus-sys/opus/celt/celt_decoder.c:1534-1535`
- Detail: Upstream `celt_synthesis()` takes `qext_mode`, `qext_bandLogE`, `qext_end` and applies extra `denormalise_bands()` contributions when QEXT is active. Rust `celt_synthesis()` has no QEXT parameters and call-sites never feed QEXT band energies into synthesis, changing decoded HF reconstruction.

32. [HIGH][QEXT] `deemphasis` misses upstream custom/QEXT IIR branch (`coef[1]`/`coef[3]` path).
- Rust: `src/celt/celt_decoder.rs:259-337`
- Upstream: `libopus-sys/opus/celt/celt_decoder.c:345-359`
- Detail: Upstream has an alternate deemphasis branch (compiled for custom/QEXT builds) that uses `coef[1]` and `coef[3]` and saturating arithmetic before downsampling. Rust only implements the simpler single-coefficient path, so output filtering diverges when non-default deemphasis coefficients are used.

33. [MEDIUM][QEXT] `deemphasis` scratch allocation is fixed to 960 instead of `N`.
- Rust: `src/celt/celt_decoder.rs:281`
- Upstream: `libopus-sys/opus/celt/celt_decoder.c:335`
- Detail: Upstream allocates `scratch` with runtime length `N`; Rust hardcodes `[f32; 960]`. This mismatches larger-frame/QEXT decode configurations where `N > 960`.

34. [HIGH][QEXT] Decoder energy finalisation updates `oldEBands` even when QEXT payload is present.
- Rust: `src/celt/celt_decoder.rs:1744-1754`
- Upstream: `libopus-sys/opus/celt/celt_decoder.c:1520`
- Detail: Upstream calls `unquant_energy_finalise(..., (qext_bytes > 0) ? NULL : oldBandE, ...)`, explicitly skipping base-band history updates when QEXT bits are present. Rust always passes `&mut st.oldEBands`, so history evolution differs in QEXT packets.

35. [HIGH][Extensions] Public extension parse/count path does not implement upstream "repeat these extensions" expansion semantics.
- Rust: `src/opus/extensions.rs:35-84`, `src/opus/extensions.rs:90-168`
- Upstream: `libopus-sys/opus/src/extensions.c:154-220`, `libopus-sys/opus/src/extensions.c:226-379`
- Detail: Upstream `opus_packet_extensions_count/parse` iterate through `OpusExtensionIterator` and include repeated extensions (ID=2 mechanism). Rust public `opus_packet_extensions_count()` / `opus_packet_extensions_parse()` use a simplified scanner based on `skip_extension()` and do not perform iterator-driven repetition, so extension lists/counts can differ on packets using repeat markers.

36. [MEDIUM][Extensions] Extension generation path omits upstream repeat-compaction logic.
- Rust: `src/opus/extensions.rs:176-273`, `src/opus/extensions.rs:667-713`
- Upstream: `libopus-sys/opus/src/extensions.c:471-624`
- Detail: Upstream `opus_packet_extensions_generate()` performs cross-frame repeat analysis and may emit ID=2 repeat indicators to compact payloads. Rust generator writes extensions frame-by-frame without repeat-compaction analysis, producing different padding bitstreams and sizes for repeatable extension sets.

37. [LOW][Extensions] `_ext` extension helper APIs are missing.
- Rust: `src/opus/extensions.rs` (no `opus_packet_extensions_count_ext` / `opus_packet_extensions_parse_ext`)
- Upstream: `libopus-sys/opus/src/extensions.c:341-421`
- Detail: Upstream exposes frame-counted parse helpers (`*_count_ext`, `*_parse_ext`) used for frame-order extraction workflows; matching Rust public functions are not present.

38. [HIGH][Repacketizer+Extensions] Repacketizer state does not retain parsed padding/extension metadata from input packets.
- Rust: `src/opus/repacketizer.rs:38-44`, `src/opus/repacketizer.rs:128-137`
- Upstream: `libopus-sys/opus/src/repacketizer.c:147-148` and struct usage in `libopus-sys/opus/src/repacketizer.c:143-177`
- Detail: Upstream `OpusRepacketizer` stores per-input padding pointer/length/frame-count and parses those extensions during output packing. Rust state only keeps frame offsets/lengths and drops padding metadata during `cat`, so extension-preserving behavior diverges.

39. [HIGH][Repacketizer+Extensions] `out_range_impl_ext` does not merge extensions embedded in source packet paddings.
- Rust: `src/opus/repacketizer.rs:240`, `src/opus/repacketizer.rs:332-340`, `src/opus/repacketizer.rs:390-397`
- Upstream: `libopus-sys/opus/src/repacketizer.c:143-177`, `libopus-sys/opus/src/repacketizer.c:260-265`, `libopus-sys/opus/src/repacketizer.c:308-311`
- Detail: Upstream computes `total_ext_count = passed_in + parsed_from_rp_paddings`, renumbers frame indices, and generates combined extension payload. Rust only serializes caller-provided `extensions` and never incorporates extensions already present in repacketized input packets.

40. [MEDIUM][Repacketizer+Extensions] Non-pad extension overhead formula is off by one for `ext_len` multiples of 254.
- Rust: `src/opus/repacketizer.rs:339`
- Upstream: `libopus-sys/opus/src/repacketizer.c:267`
- Detail: Upstream uses `pad_amount = ext_len + (ext_len ? (ext_len+253)/254 : 1)`. Rust uses `ext_len + ext_len/254 + 1`, which is larger by 1 when `ext_len % 254 == 0` and `ext_len > 0`, changing packing decisions and output layout.

41. [MEDIUM][Repacketizer+Extensions] Extension frame-index validation is not constrained to output packet frame count.
- Rust: `src/opus/extensions.rs:176-193`, `src/opus/repacketizer.rs:334-336`, `src/opus/repacketizer.rs:392-396`
- Upstream: `libopus-sys/opus/src/extensions.c:471-473`, `libopus-sys/opus/src/extensions.c:495`, `libopus-sys/opus/src/repacketizer.c:263-264`, `libopus-sys/opus/src/repacketizer.c:309-310`
- Detail: Upstream generation takes `nb_frames` and rejects any extension with `frame >= nb_frames`. Rust generator APIs do not take packet frame count and only enforce `frame < 48`, so repacketizer extension emission can accept frame indices that upstream would reject.

42. [HIGH][QEXT] Encoder fade helpers omit `IMAX(1, 48000/Fs)` guard and can compute zero increment at 96 kHz.
- Rust: `src/opus/opus_encoder.rs:833`, `src/opus/opus_encoder.rs:875`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:554`, `libopus-sys/opus/src/opus_encoder.c:588`
- Detail: Upstream clamps `inc` to at least 1. Rust uses raw integer division (`48000/Fs`), which becomes `0` for `Fs=96000` and can break overlap/index math in `stereo_fade`/`gain_fade` on QEXT 96 kHz paths.

43. [MEDIUM] Missing CELT custom 24-bit API parity.
- Rust: `src/celt/celt_encoder.rs`, `src/celt/celt_decoder.rs` (no `opus_custom_encode24` / `opus_custom_decode24` entry points found)
- Upstream: `libopus-sys/opus/celt/celt_encoder.c:2871`, `libopus-sys/opus/celt/celt_decoder.c:1658`
- Detail: Upstream custom CELT API exports 24-bit integer encode/decode functions; matching Rust entry points are not present.

44. [LOW] Missing direct CELT custom C-style CTL/destroy entry points.
- Rust: `src/celt/celt_encoder.rs`, `src/celt/celt_decoder.rs`
- Upstream: `libopus-sys/opus/celt/celt_encoder.c:260`, `libopus-sys/opus/celt/celt_encoder.c:2941`, `libopus-sys/opus/celt/celt_decoder.c:278`, `libopus-sys/opus/celt/celt_decoder.c:1722`
- Detail: Upstream exposes direct C-style `opus_custom_encoder_ctl` / `opus_custom_decoder_ctl` and explicit destroy functions. Rust exposes struct-based APIs but does not mirror these direct entry points.

45. [MEDIUM][DRED] Missing full DRED object API surface (`opus_dred_*` / `opus_dred_decoder_*`).
- Rust: `src/opus/` (no matching `opus_dred_decoder_get_size/init/create/destroy/ctl`, `opus_dred_get_size/alloc/free`, `opus_dred_parse`, `opus_dred_process`)
- Upstream: `libopus-sys/opus/src/opus_decoder.c:1363`, `libopus-sys/opus/src/opus_decoder.c:1381`, `libopus-sys/opus/src/opus_decoder.c:1395`, `libopus-sys/opus/src/opus_decoder.c:1417`, `libopus-sys/opus/src/opus_decoder.c:1423`, `libopus-sys/opus/src/opus_decoder.c:1511`, `libopus-sys/opus/src/opus_decoder.c:1520`, `libopus-sys/opus/src/opus_decoder.c:1539`, `libopus-sys/opus/src/opus_decoder.c:1548`, `libopus-sys/opus/src/opus_decoder.c:1587`
- Detail: Upstream exposes a complete parsed-DRED object workflow API in addition to direct `opus_decoder_dred_decode*` calls. Rust does not mirror this object lifecycle/parse/process API surface.

46. [LOW] Missing direct CELT custom encode/decode entry point functions.
- Rust: `src/celt/celt_encoder.rs`, `src/celt/celt_decoder.rs` (no direct `opus_custom_encode`, `opus_custom_encode_float`, `opus_custom_decode`, `opus_custom_decode_float` free-function exports)
- Upstream: `libopus-sys/opus/celt/celt_encoder.c:2838`, `libopus-sys/opus/celt/celt_encoder.c:2906`, `libopus-sys/opus/celt/celt_decoder.c:1629`, `libopus-sys/opus/celt/celt_decoder.c:1690`
- Detail: Upstream provides stable C-style entry points for custom CELT encode/decode variants. Rust currently exposes struct-centric APIs instead of matching these direct function symbols.

47. [LOW][Analysis] `downmix_and_resample` lacks upstream unsupported-Fs assertion path.
- Rust: `src/opus/analysis.rs:642-672`
- Upstream: `libopus-sys/opus/src/analysis.c:181`
- Detail: Upstream asserts `Fs` must be one of `{16000,24000,48000}` in this helper. Rust has no equivalent assertion branch, so unsupported `Fs` values fall through without upstream-equivalent failure behavior.

48. [LOW] Missing direct top-level create/get_size/destroy C entry points for encoder/decoder.
- Rust: `src/opus/opus_encoder.rs` (`OpusEncoder::new`), `src/opus/opus_decoder.rs` (`OpusDecoder::new`)
- Upstream: `libopus-sys/opus/src/opus_encoder.c:194`, `libopus-sys/opus/src/opus_encoder.c:622`, `libopus-sys/opus/src/opus_encoder.c:3362`, `libopus-sys/opus/src/opus_decoder.c:121`, `libopus-sys/opus/src/opus_decoder.c:186`, `libopus-sys/opus/src/opus_decoder.c:1244`
- Detail: Upstream exports C-style `opus_encoder_get_size/init/create/destroy` and `opus_decoder_get_size/init/create/destroy` entry points. Rust uses struct constructors/methods and does not mirror these direct symbols.

49. [LOW][Packet Parser] `opus_packet_parse_impl` does not clear padding output on entry like upstream.
- Rust: `src/opus/packet.rs:172-195`
- Upstream: `libopus-sys/opus/src/opus.c:239-244`
- Detail: Upstream sets padding outputs to null/zero before validation so callers get deterministic output on error paths. Rust only writes `padding_out` at successful end of parse, so caller-provided storage may retain stale values after early parse errors.

50. [MEDIUM] Encoder analysis gating misses upstream restricted-SILK application guard.
- Rust: `src/opus/opus_encoder.rs:1677-1681`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:1252`
- Detail: Upstream only runs `run_analysis(...)` when `application != OPUS_APPLICATION_RESTRICTED_SILK`. Rust gating currently checks only complexity/Fs and can run analysis in restricted-SILK mode, diverging from upstream mode behavior.

51. [HIGH] `frame_size_select` omits application-dependent restricted-SILK minimum-frame rule.
- Rust: `src/opus/opus_encoder.rs:942-974`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:827-851` (notably `:849-850`)
- Detail: Upstream rejects frame sizes below 10 ms for `OPUS_APPLICATION_RESTRICTED_SILK` (`new_size < Fs/100`). Rust `frame_size_select` has no `application` parameter and therefore cannot enforce this rule, allowing frame-size selections upstream would reject.

52. [HIGH][QEXT] Encoder `delay_buffer` capacity is fixed to non-QEXT size.
- Rust: `src/opus/opus_encoder.rs:104`, `src/opus/opus_encoder.rs:250`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:63-66`, `libopus-sys/opus/src/opus_encoder.c:145`
- Detail: Upstream uses `delay_buffer[MAX_ENCODER_BUFFER*2]` with `MAX_ENCODER_BUFFER=960` under QEXT builds (capacity 1920 samples). Rust hardcodes `delay_buffer: [opus_val16; 960]`, matching only non-QEXT sizing and risking state truncation for 96 kHz/QEXT operation.

53. [MEDIUM] Encoder init sets `encoder_buffer` unconditionally, missing restricted-app zero-buffer behavior.
- Rust: `src/opus/opus_encoder.rs:222`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:304-307`
- Detail: Upstream sets `encoder_buffer=0` for restricted CELT/SILK applications and `Fs/100` otherwise. Rust initializes `encoder_buffer` to `Fs/100` unconditionally, diverging from restricted-application startup behavior.

54. [LOW] Missing CELT custom create/get_size entry points.
- Rust: `src/celt/celt_encoder.rs`, `src/celt/celt_decoder.rs` (struct constructors only)
- Upstream: `libopus-sys/opus/celt/celt_encoder.c:146`, `libopus-sys/opus/celt/celt_encoder.c:178`, `libopus-sys/opus/celt/celt_decoder.c:180`, `libopus-sys/opus/celt/celt_decoder.c:208`
- Detail: Upstream exposes `opus_custom_encoder_get_size`/`opus_custom_encoder_create` and `opus_custom_decoder_get_size`/`opus_custom_decoder_create`. Rust does not mirror these direct allocation-size/create APIs.

55. [LOW][Packet API] `opus_packet_parse` frame output type differs from upstream pointer API.
- Rust: `src/opus/packet.rs:389-405`
- Upstream: `libopus-sys/opus/src/opus.c:392-398`
- Detail: Upstream returns direct frame pointers (`const unsigned char *frames[48]`) into the packet buffer. Rust returns frame byte offsets (`[usize; 48]`). This is an API-surface mismatch for callers expecting C-equivalent pointer outputs.

56. [MEDIUM][Encoder CTL] Bitrate validation/clamping diverges from upstream (`300000*channels` vs `750000*channels`, and non-positive handling).
- Rust: `src/opus/opus_encoder.rs:321`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:2826-2827`
- Detail: Upstream `OPUS_SET_BITRATE_REQUEST` rejects non-positive explicit values (`goto bad_arg`) and clamps high values to `750000*channels`. Rust `set_bitrate()` clamps arbitrary explicit values into `[500, 300000*channels]`, changing control semantics and resulting encode configuration at both low and high out-of-range inputs.

57. [LOW][Packet API] `opus_packet_has_lbrr` entry point is missing.
- Rust: `src/opus/packet.rs` (no `opus_packet_has_lbrr` symbol found)
- Upstream: `libopus-sys/opus/src/opus_decoder.c:1306`
- Detail: Upstream exports packet-level LBRR presence detection API; matching Rust public function is not present.

58. [LOW][Extensions API] Direct C-style extension-iterator entry points are not mirrored.
- Rust: `src/opus/extensions.rs` (struct methods only: `new`, `next`, `find`)
- Upstream: `libopus-sys/opus/src/extensions.c:44`, `libopus-sys/opus/src/extensions.c:102`, `libopus-sys/opus/src/extensions.c:126`, `libopus-sys/opus/src/extensions.c:149`, `libopus-sys/opus/src/extensions.c:171`
- Detail: Upstream exposes `opus_extension_iterator_init/reset/next/set_frame_max/find` as direct APIs. Rust currently provides method-based iterator usage and lacks equivalent direct function symbols/surface.

59. [LOW][Repacketizer API] Direct C-style repacketizer entry points are not mirrored.
- Rust: `src/opus/repacketizer.rs` (struct-centric API)
- Upstream: `libopus-sys/opus/src/repacketizer.c:67`, `libopus-sys/opus/src/repacketizer.c:72`, `libopus-sys/opus/src/repacketizer.c:79`, `libopus-sys/opus/src/repacketizer.c:86`, `libopus-sys/opus/src/repacketizer.c:138`, `libopus-sys/opus/src/repacketizer.c:197`, `libopus-sys/opus/src/repacketizer.c:347`, `libopus-sys/opus/src/repacketizer.c:353`
- Detail: Upstream exports `opus_repacketizer_get_size/init/create/destroy/cat/out/out_range/get_nb_frames` function entry points. Rust exposes equivalent operations as struct methods but does not mirror these direct symbols.

60. [LOW][Packet API] `opus_pcm_soft_clip_impl` entry point is missing.
- Rust: `src/opus/packet.rs` (exports `opus_pcm_soft_clip` only)
- Upstream: `libopus-sys/opus/src/opus.c:39`
- Detail: Upstream includes internal/publicly-linkable `opus_pcm_soft_clip_impl` with `arch` parameter, while Rust surface exposes only the simplified `opus_pcm_soft_clip` wrapper.

61. [MEDIUM][Error Handling] Decoder has unconditional panics where upstream uses `celt_assert` fallback behavior.
- Rust: `src/opus/opus_decoder.rs:535`, `src/opus/opus_decoder.rs:694`
- Upstream: `libopus-sys/opus/src/opus_decoder.c:444`, `libopus-sys/opus/src/opus_decoder.c:562` (assert sites in switch defaults)
- Detail: Upstream `celt_assert(0)` is assertion-only and retains defined fallback assignments in normal builds. Rust uses unconditional `panic!`, which can terminate decoding on malformed/unexpected packet-state combinations instead of continuing with upstream-compatible fallback behavior.

62. [MEDIUM][Error Handling] CELT decoder init paths panic on invalid args instead of returning Opus error codes.
- Rust: `src/celt/celt_decoder.rs:141`, `src/celt/celt_decoder.rs:149-152`
- Upstream: `libopus-sys/opus/celt/celt_decoder.c:232-236`, `libopus-sys/opus/celt/celt_decoder.c:241-245`
- Detail: Upstream `celt_decoder_init`/`opus_custom_decoder_init` return `OPUS_BAD_ARG`/`OPUS_ALLOC_FAIL` for unsupported sampling rates or channel counts. Rust currently panics in these argument-validation paths, changing failure semantics and making API behavior non-equivalent.

63. [LOW][Encoder API] Top-level `opus_encode` / `opus_encode_float` function entry points are not public.
- Rust: `src/opus/opus_encoder.rs:3023`, `src/opus/opus_encoder.rs:3054` (private `fn`)
- Upstream: `libopus-sys/opus/src/opus_encoder.c:2662`, `libopus-sys/opus/src/opus_encoder.c:2735`
- Detail: Upstream exports direct C API functions `opus_encode` and `opus_encode_float`. Rust keeps same-named functions private and exposes method-based encoding on `OpusEncoder`, so symbol-level/public function parity differs.

64. [LOW][CPU Dispatch] Runtime architecture selection is stubbed to scalar-only in Rust.
- Rust: `src/opus/opus_encoder.rs:16-21` (`opus_select_arch()` returns `0`), plus decoder/CELT states that keep `arch=0`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:248`, `libopus-sys/opus/src/opus_decoder.c:177`, `libopus-sys/opus/celt/celt_decoder.c:264` (runtime `opus_select_arch()` usage)
- Detail: Upstream initializes encoder/decoder/CELT state with runtime architecture dispatch. Rust hardcodes architecture selection to `0`, so optimized arch-specific paths are never selected, diverging from upstream runtime-path selection behavior.

65. [LOW][Decoder Integration] `silk_Decode` is called with fixed `arch=0` instead of decoder runtime arch.
- Rust: `src/opus/opus_decoder.rs:578`
- Upstream: `libopus-sys/opus/src/opus_decoder.c:478-480`
- Detail: Upstream passes `st->arch` through to `silk_Decode(...)`. Rust passes literal `0`, so SILK decode path ignores runtime arch selection even when upstream would route arch-specific kernels.

66. [MEDIUM][Runtime Semantics] Decoder validation asserts run unconditionally in Rust hot paths.
- Rust: `src/opus/opus_decoder.rs:919`, `src/celt/celt_decoder.rs:1028`
- Upstream: `libopus-sys/opus/src/opus_decoder.c:92-117`, `libopus-sys/opus/src/opus_decoder.c:119-123`; `libopus-sys/opus/celt/celt_decoder.c:136-151`, `libopus-sys/opus/celt/celt_decoder.c:153-155`
- Detail: Upstream wraps `VALIDATE_OPUS_DECODER`/`VALIDATE_CELT_DECODER` behind `ENABLE_HARDENING || ENABLE_ASSERTIONS` and no-ops otherwise. Rust calls validation functions unconditionally, so release/runtime behavior can panic on invariant violations that upstream production builds would not check.

67. [MEDIUM][Runtime Semantics] Decode-loop invariants use unconditional `assert!` in Rust where upstream uses `celt_assert`.
- Rust: `src/opus/opus_decoder.rs:946`, `src/opus/opus_decoder.rs:1004`, `src/opus/opus_decoder.rs:1095`
- Upstream: `libopus-sys/opus/src/opus_decoder.c:777`, `libopus-sys/opus/src/opus_decoder.c:813`, `libopus-sys/opus/src/opus_decoder.c:864`
- Detail: Upstream uses `celt_assert(...)` for internal invariants (often compiled out in non-assert builds). Rust uses unconditional `assert!`, which can abort decoding at runtime on mismatch conditions where upstream typically continues/returns errors depending on build flags.

68. [LOW][Runtime Semantics] Encoder internal invariants use unconditional `assert!` in Rust where upstream uses `celt_assert`.
- Rust: `src/opus/opus_encoder.rs:2372`, `src/opus/opus_encoder.rs:2508`, `src/opus/opus_encoder.rs:2988`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:2118`, `libopus-sys/opus/src/opus_encoder.c:2230`, `libopus-sys/opus/src/opus_encoder.c:2628`
- Detail: Upstream marks these as `celt_assert(...)` internal checks. Rust promotes them to unconditional `assert!`, which can terminate runtime encode paths in situations where upstream non-assert builds would not abort.

69. [LOW][Encoder CTL] LFE control request parity is missing.
- Rust: `src/opus/opus_encoder.rs` (no `set_lfe`/`OPUS_SET_LFE`-equivalent public control path found)
- Upstream: `libopus-sys/opus/src/opus_encoder.c:3283-3289`
- Detail: Upstream supports `OPUS_SET_LFE_REQUEST` and forwards it to CELT when applicable. Rust has an internal `lfe` field but no matching public/direct control entry point.

70. [LOW][Encoder CTL] Energy-mask control request parity is missing.
- Rust: `src/opus/opus_encoder.rs` (no `set_energy_mask`/`OPUS_SET_ENERGY_MASK`-equivalent public control path found)
- Upstream: `libopus-sys/opus/src/opus_encoder.c:3291-3296`
- Detail: Upstream supports `OPUS_SET_ENERGY_MASK_REQUEST` to inject per-band masking. Rust encoder contains energy-mask storage but lacks a matching external control entry point.

71. [LOW][Typed API] Rust `Application` enum omits upstream restricted application variants.
- Rust: `src/enums.rs:19-31`, `src/enums.rs:36-40`, `src/enums.rs:47-51`
- Upstream: `libopus-sys/opus/include/opus_defines.h` (`OPUS_APPLICATION_RESTRICTED_SILK`=2052, `OPUS_APPLICATION_RESTRICTED_CELT`=2053)
- Detail: Rust typed API exposes only `Voip`/`Audio`/`LowDelay`. Upstream defines two additional restricted application modes, so typed control/API parity is incomplete even before lower-level encoder support.

72. [LOW][Runtime Semantics][QEXT] `compute_qext_mode` uses `panic!` on unsupported mode tuples where upstream uses `celt_assert(0)`.
- Rust: `src/celt/modes.rs:113`
- Upstream: `libopus-sys/opus/celt/modes.c:511`
- Detail: For unexpected `shortMdctSize/Fs` combinations, upstream issues an internal assertion; Rust unconditionally panics. This changes failure behavior and can hard-abort on state/config mismatches.

73. [LOW][CELT API] Non-custom CELT size/init wrapper entry points are incomplete.
- Rust: `src/celt/celt_encoder.rs`, `src/celt/celt_decoder.rs` (no `celt_encoder_get_size`, `celt_decoder_get_size`, `celt_encoder_init` symbols)
- Upstream: `libopus-sys/opus/celt/celt.h:157`, `libopus-sys/opus/celt/celt.h:166`, `libopus-sys/opus/celt/celt.h:160`; implementations in `libopus-sys/opus/celt/celt_encoder.c:145`, `libopus-sys/opus/celt/celt_decoder.c:179`, `libopus-sys/opus/celt/celt_encoder.c:240`
- Detail: Upstream exposes top-level CELT wrapper APIs (`celt_*`) in addition to custom-object APIs; Rust currently exposes struct constructors/helpers but not equivalent wrapper function symbols.

74. [LOW][CELT API] `celt_decode_with_ec_dred` entry point is missing.
- Rust: `src/celt/celt_decoder.rs` (has `celt_decode_with_ec`, no dedicated `celt_decode_with_ec_dred` symbol)
- Upstream: `libopus-sys/opus/celt/celt.h:169`; implementation in `libopus-sys/opus/celt/celt_decoder.c:1099`
- Detail: Upstream provides a distinct DRED-capable decode entry point (`celt_decode_with_ec_dred`) used by decoder integration. Rust folds behavior into `celt_decode_with_ec` call signatures and does not mirror the dedicated symbol/surface.

75. [LOW][QEXT] 96 kHz CELT decoder init uses a different `opus_custom_mode_create` frame-size argument (`1920` vs `960`) than upstream.
- Rust: `src/celt/celt_decoder.rs:126`
- Upstream: `libopus-sys/opus/celt/celt_decoder.c:229`
- Detail: This is a code-shape/API-call difference. In practice both calls resolve to the same static mode via `opus_custom_mode_create` frame-size shift matching, so this is not expected to change behavior but remains a direct source-level divergence.

76. [MEDIUM][DRED] `opus_decode_native` signature/flow omits upstream DRED input arguments and pre-processing path.
- Rust: `src/opus/opus_decoder.rs:898-907`
- Upstream: `libopus-sys/opus/src/opus_decoder.c:718` plus DRED pre-processing block at `:732-766`
- Detail: Upstream `opus_decode_native` accepts `const OpusDRED *dred` and `dred_offset`, with explicit feature-frame staging for deep PLC when provided. Rust `opus_decode_native` has no DRED object arguments and therefore cannot mirror this decode-time DRED-assisted flow.

77. [LOW][Decoder API] `opus_decoder_get_nb_samples` requires mutable decoder reference in Rust.
- Rust: `src/opus/opus_decoder.rs:1211`
- Upstream: `libopus-sys/opus/src/opus_decoder.c:1333` (`const OpusDecoder *dec`)
- Detail: Upstream sample-count query is const and does not require mutable decoder access. Rust signature takes `&mut OpusDecoder`, narrowing call-sites and diverging from C API constness/usage semantics.

78. [LOW][Packet API] Packet header helper signatures take TOC byte instead of packet pointer.
- Rust: `src/opus/opus_decoder.rs:1151` (`opus_packet_get_bandwidth(toc: u8)`), `src/opus/packet.rs:149` (`opus_packet_get_samples_per_frame(data: u8, fs: i32)`), `src/opus/opus_decoder.rs:1173` (`opus_packet_get_nb_channels(toc: u8)`)
- Upstream: `libopus-sys/opus/src/opus.c:173`, `libopus-sys/opus/src/opus.c:205`, `libopus-sys/opus/src/opus.c:170`
- Detail: Upstream helper APIs accept packet pointers (`const unsigned char *data`) and read `data[0]`. Rust exposes TOC-byte forms directly, which is behaviorally equivalent for well-formed calls but not a symbol/signature match with the upstream C API.

79. [MEDIUM][Error Handling] `resampling_factor` panics on unsupported rates instead of returning 0 like upstream.
- Rust: `src/celt/common.rs:34-42`
- Upstream: `libopus-sys/opus/celt/celt.c:432-445`
- Detail: Upstream `resampling_factor()` returns `0` for unsupported rates and callers convert this into `OPUS_BAD_ARG`. Rust uses `panic!` for unsupported rates, changing error propagation semantics to process aborts.

80. [LOW][Validation] CELT decoder validation hardcodes `arch <= 0` rather than upstream arch-mask check.
- Rust: `src/celt/celt_decoder.rs:110-111`
- Upstream: `libopus-sys/opus/celt/celt_decoder.c:158-162`
- Detail: Upstream validates `st->arch` in range `[0, OPUS_ARCHMASK]` (when enabled). Rust enforces `st.arch <= 0`, over-constraining valid non-zero arch IDs and diverging from upstream validation semantics.

81. [LOW][Decoder Integration] `celt_float2int16` conversion path omits upstream arch parameter.
- Rust: `src/opus/opus_decoder.rs:1139`, `src/celt/float_cast.rs:26`
- Upstream: `libopus-sys/opus/src/opus_decoder.c:928`
- Detail: Upstream integer-decode wrapper calls `celt_float2int16(..., st->arch)` to allow arch-specific conversion. Rust conversion helper has no arch argument and call-site cannot pass decoder arch, diverging from upstream dispatch behavior.

82. [MEDIUM][Error Handling] CELT encoder init path can panic on unsupported sampling rates instead of returning Opus error codes.
- Rust: `src/celt/celt_encoder.rs:144-151` (uses `resampling_factor(sampling_rate)` which panics on unsupported rates)
- Upstream: `libopus-sys/opus/celt/celt_encoder.c:251-256`
- Detail: Upstream `celt_encoder_init` returns an Opus status code on invalid rates/init failures. Rust init path can panic through `resampling_factor`, changing API failure semantics from error-return to abort.

83. [LOW][Validation] CELT encoder constructor does not mirror upstream channel-range validation.
- Rust: `src/celt/celt_encoder.rs:136-151` (no explicit `channels` range check before state setup)
- Upstream: `libopus-sys/opus/celt/celt_encoder.c:197-199`
- Detail: Upstream validates `channels` bounds in `opus_custom_encoder_init_arch` and returns `OPUS_BAD_ARG` when invalid. Rust constructor lacks equivalent upfront validation/error return in the same place, relying on later behavior.

84. [LOW][State Layout] `OpusDecoder` state omits upstream `arch` field.
- Rust: `src/opus/opus_decoder.rs:38-58` (`OpusDecoder` struct)
- Upstream: `libopus-sys/opus/src/opus_decoder.c:62-76` (`struct OpusDecoder` includes `int arch`)
- Detail: Upstream decoder stores selected architecture in decoder state and uses it in downstream helper calls. Rust decoder struct has no corresponding persisted `arch` member, which changes state-layout parity and limits exact mirroring of arch-dependent call plumbing.

85. [LOW][Packet Parser] Rust adds per-frame bounds checks in `opus_packet_parse_impl` that are not present in upstream loop.
- Rust: `src/opus/packet.rs:368-374`
- Upstream: `libopus-sys/opus/src/opus.c:368-374`
- Detail: Upstream advances `data += size[i]` directly after prior size accounting. Rust additionally checks `size > data.len()` each iteration and returns `OPUS_INVALID_PACKET` early. This is stricter error-path behavior and not a byte-for-byte control-flow match.

86. [LOW][Packet API] `opus_packet_unpad` signature differs (no explicit `len` argument).
- Rust: `src/opus/repacketizer.rs:529`
- Upstream: `libopus-sys/opus/src/repacketizer.c:371`
- Detail: Upstream API accepts `(data, len)` and can operate on a packet prefix inside a larger buffer. Rust uses full-slice length (`data.len()`) only, so call semantics differ when buffer capacity exceeds packet length.

87. [LOW][Runtime Semantics] CELT encoder internal invariants are enforced with unconditional `assert!` in Rust.
- Rust: `src/celt/celt_encoder.rs:369`, `src/celt/celt_encoder.rs:370`, `src/celt/celt_encoder.rs:2256`, `src/celt/celt_encoder.rs:2530`, `src/celt/celt_encoder.rs:2604`
- Upstream: `libopus-sys/opus/celt/celt_encoder.c:423`, `libopus-sys/opus/celt/celt_encoder.c:424`, `libopus-sys/opus/celt/celt_encoder.c:1895`, `libopus-sys/opus/celt/celt_encoder.c:2093`, `libopus-sys/opus/celt/celt_encoder.c:2136`
- Detail: Upstream marks these checks as `celt_assert(...)` (typically compiled out in non-assert builds). Rust uses unconditional `assert!`, changing runtime failure behavior.

88. [HIGH][QEXT][Validation] CELT decoder validation hardcodes `end <= 21`, missing upstream QEXT allowance up to 25.
- Rust: `src/celt/celt_decoder.rs:109`
- Upstream: `libopus-sys/opus/celt/celt_decoder.c:149-155`
- Detail: Upstream validates `end <= 21` for non-QEXT and `end <= 25` when QEXT is enabled. Rust always asserts `st.end <= 21`, which can incorrectly reject valid QEXT decoder states.

89. [LOW][Projection/Multistream] `mapping_matrix` module/API is missing.
- Rust: `src/opus/` (no `mapping_matrix.rs` equivalent found)
- Upstream: `libopus-sys/opus/src/mapping_matrix.c`
- Detail: Upstream includes mapping-matrix utilities and static FOA/SOA/TOA HOA matrices used by projection/multistream components. Rust codebase does not mirror this module, consistent with missing projection/multistream API coverage.

90. [MEDIUM][Analysis] Float downmix path clamps samples and replaces NaNs earlier than upstream.
- Rust: `src/opus/analysis.rs:78-83`
- Upstream: `libopus-sys/opus/src/analysis.c:561-567`
- Detail: Rust `DownmixInput::Float::downmix` clamps to `[-65536, 65536]` and converts NaN to `0` before analysis FFT. Upstream does not clamp there; it checks for NaN after FFT output and bails from analysis if detected. This alters analysis feature generation under extreme/NaN inputs.

91. [LOW][CPU Dispatch] Analysis path uses stubbed architecture selection (`arch=0`) in Rust.
- Rust: `src/opus/analysis.rs:194-200`, `src/opus/analysis.rs:677`
- Upstream: `libopus-sys/opus/src/analysis.c:220`, `libopus-sys/opus/src/analysis.c:558`
- Detail: Upstream analysis state stores selected runtime architecture and passes it into FFT calls. Rust analysis also stores `arch` but `opus_select_arch()` is stubbed to `0`, so arch-dispatched analysis kernels are never selected.

92. [LOW][Packet API] Additional packet helper signatures are slice-based rather than upstream pointer+length forms.
- Rust: `src/opus/packet.rs:388-405` (`opus_packet_parse(data: &[u8], ...)`), `src/opus/opus_decoder.rs:1178` (`opus_packet_get_nb_frames(packet: &[u8])`)
- Upstream: `libopus-sys/opus/src/opus.c:392-398` (`opus_packet_parse(const unsigned char *data, opus_int32 len, ...)`), `libopus-sys/opus/src/opus.c:216` (`opus_packet_get_nb_frames(const unsigned char packet[], opus_int32 len)`)
- Detail: Rust APIs infer length from slice and do not expose explicit `len` parameters, which diverges from upstream C signatures and affects FFI-level compatibility semantics.

93. [MEDIUM][Runtime Semantics][SILK] Multiple SILK control/validation paths use unconditional `panic!(\"libopus: assert(0) called\")`.
- Rust: `src/silk/check_control_input.rs:56-96`, `src/silk/decoder_set_fs.rs:55`, `src/silk/enc_API.rs:79`, `src/silk/enc_API.rs:86`, `src/silk/enc_API.rs:148`, `src/silk/enc_API.rs:186`, `src/silk/enc_API.rs:216-219`, `src/silk/dec_API.rs:144-148`, `src/silk/resampler/mod.rs:127-138`
- Upstream: representative assert sites such as `libopus-sys/opus/silk/enc_API.c:264`, `libopus-sys/opus/silk/enc_API.c:526`, `libopus-sys/opus/silk/decoder_set_fs.c:89`
- Detail: Upstream uses `silk_assert`/`celt_assert` in these invariant paths (typically compile-time no-op in non-assert builds). Rust panics unconditionally, changing runtime failure semantics.

94. [LOW][Runtime Semantics][DRED] DRED 16 kHz conversion helper panics on unsupported sample rates.
- Rust: `src/dnn/dred/encoder.rs:160`
- Upstream: `libopus-sys/opus/dnn/dred_encoder.c:165`, `libopus-sys/opus/dnn/dred_encoder.c:207`
- Detail: Upstream uses assertion-style checks in unreachable/default sample-rate branches; Rust uses unconditional `panic!(\"Unsupported sample rate\")`, changing failure mode from assert-gated to hard abort.

95. [LOW][Lib Info] `opus_get_version_string()` does not match upstream format/content.
- Rust: `src/celt/common.rs:426`
- Upstream: `libopus-sys/opus/celt/celt.c:360-372`
- Detail: Upstream returns `"libopus " PACKAGE_VERSION` with optional `-fixed`/`-fuzzing` suffixes. Rust returns hardcoded `"opurs (rust port) 1.5.2"`, which diverges from upstream identifier/version semantics (and currently from the target 1.6.1 line).

96. [LOW][Lib Info] `opus_strerror()` message text differs from upstream.
- Rust: `src/celt/common.rs:408-424`
- Upstream: `libopus-sys/opus/celt/celt.c:342-358`
- Detail: Rust returns strings including numeric suffixes (e.g., `"success (0)"`), while upstream returns plain canonical messages (e.g., `"success"`). API behavior is functionally similar but string outputs are not byte-identical.

97. [HIGH][Extensions][RTE Semantics] Public extension count/parse/find helpers ignore `nb_frames` and do not apply full repeat-extension (ID=2) iterator semantics.
- Rust: `src/opus/extensions.rs:90-121`, `src/opus/extensions.rs:279-285`
- Upstream: `libopus-sys/opus/src/extensions.c:329-383` (count/parse via `opus_extension_iterator_next`)
- Detail: Rust helpers explicitly discard `nb_frames` (`let _ = nb_frames`) and use a simplified parser path. Upstream routes these APIs through `OpusExtensionIterator`, including repeated-extension expansion and frame-limit validation, so outputs differ for packets using RTE (ID=2).

98. [MEDIUM][Extensions][API Coverage] Frame-ordered extension helpers are missing (`opus_packet_extensions_count_ext`, `opus_packet_extensions_parse_ext`).
- Rust: `src/opus/extensions.rs` (no equivalents found)
- Upstream: `libopus-sys/opus/src/extensions.c:341-420`
- Detail: Upstream exposes per-frame extension counting and frame-ordered parse APIs required by the full extension workflow. Rust only exposes count/parse/find/generate variants and omits `_ext` forms.

99. [MEDIUM][Extensions][Generator Semantics] Extension generator signature/logic diverges from upstream (`nb_frames` omitted and repeat optimization not implemented).
- Rust: `src/opus/extensions.rs:176-180`, `src/opus/extensions.rs:667-669`
- Upstream: `libopus-sys/opus/src/extensions.c:471-635`
- Detail: Upstream `opus_packet_extensions_generate(..., nb_frames, pad)` validates against frame count and can emit repeat indicators to compact repeated extensions. Rust generator infers `max_frame` from extension list and does not run upstream repeat-pointer/indicator logic, producing different packet layouts.

100. [LOW][Extensions][Iterator API] Iterator control helpers are missing (`reset`, `set_frame_max`).
- Rust: `src/opus/extensions.rs:456-660` (`OpusExtensionIterator` has `new/next/find` only)
- Upstream: `libopus-sys/opus/src/extensions.c:134-153`
- Detail: Upstream iterator API supports in-place reset and dynamic `frame_max` truncation. Rust struct stores `frame_max` but exposes no equivalent public control methods.

101. [MEDIUM][Platform/State Layout] `align()` helper is hardcoded to 8-byte alignment instead of upstream computed union alignment.
- Rust: `src/opus/opus_private.rs:11-16`
- Upstream: `libopus-sys/opus/src/opus_private.h:213-223`
- Detail: Upstream computes alignment from `offsetof(struct foo, u)` (platform-dependent). Rust hardcodes `alignment = 8`, which can diverge on targets where required alignment differs.

102. [MEDIUM][Constants/CTL Coverage] New upstream 1.6.1 request IDs are missing from Rust `opus_defines`.
- Rust: `src/opus/opus_defines.rs:55-57` (stops at `OPUS_SET_DNN_BLOB_REQUEST`)
- Upstream: `libopus-sys/opus/include/opus_defines.h:176-181`
- Detail: Rust omits `OPUS_SET/GET_OSCE_BWE_REQUEST` (4054/4055), `OPUS_SET/GET_QEXT_REQUEST` (4056/4057), and `OPUS_SET/GET_IGNORE_EXTENSIONS_REQUEST` (4058/4059).

103. [MEDIUM][Constants/Application Coverage] Restricted application constants are missing (`OPUS_APPLICATION_RESTRICTED_SILK`, `OPUS_APPLICATION_RESTRICTED_CELT`).
- Rust: `src/opus/opus_defines.rs:62-64`, `src/enums.rs:22-29`
- Upstream: `libopus-sys/opus/include/opus_defines.h:223-226`
- Detail: Upstream defines application IDs 2052 and 2053. Rust constants and typed `Application` enum only include VoIP/Audio/Restricted-LowDelay.

104. [HIGH][API Coverage][Multistream/Projection] Upstream multistream/projection API modules are missing.
- Rust: `src/lib.rs`, `src/opus/mod.rs` (no `opus_multistream*`/`opus_projection*` modules exported)
- Upstream: `libopus-sys/opus/src/opus_multistream.c`, `libopus-sys/opus/src/opus_multistream_encoder.c`, `libopus-sys/opus/src/opus_multistream_decoder.c`, `libopus-sys/opus/src/opus_projection_encoder.c`, `libopus-sys/opus/src/opus_projection_decoder.c`
- Detail: Beyond missing `mapping_matrix`, Rust currently lacks the core multistream/projection encoder/decoder implementations and public API surface present in upstream.

105. [MEDIUM][API Coverage][24-bit Paths] 24-bit PCM encode/decode APIs are missing.
- Rust: `src/opus/opus_encoder.rs` (no `opus_encode24`), `src/opus/opus_decoder.rs` (no `opus_decode24`), `src/lib.rs` exports
- Upstream: `libopus-sys/opus/include/opus.h` (`opus_encode24`, `opus_decode24` declarations)
- Detail: Upstream exposes dedicated 24-bit integer entry points; Rust currently exposes 16-bit and float paths only.

106. [LOW][Runtime Semantics][QEXT] `compute_qext_mode` uses unconditional `panic!` on unsupported mode relation.
- Rust: `src/celt/modes.rs:113`
- Upstream: `libopus-sys/opus/celt/modes.c:511`
- Detail: Upstream uses `celt_assert(0)` in this unreachable branch. Rust panics unconditionally, changing behavior under non-assert builds.

107. [LOW][CPU Dispatch][Soft Clip] Decoder soft-clip path does not pass runtime arch to implementation.
- Rust: `src/opus/opus_decoder.rs:1103`, `src/opus/packet.rs:16-21`
- Upstream: `libopus-sys/opus/src/opus_decoder.c:874`, `libopus-sys/opus/src/opus.c:39`, `libopus-sys/opus/src/opus.c:163-165`
- Detail: Upstream decoder calls `opus_pcm_soft_clip_impl(..., st->arch)`. Rust decoder calls `opus_pcm_soft_clip(...)` that has no arch parameter, so decoder-selected arch is not propagated to soft-clip internals.

108. [LOW][Version Targeting] Crate-level compatibility/docs still target libopus 1.5.2 rather than 1.6.1.
- Rust: `src/lib.rs:1`
- Upstream target for this review: `libopus-sys/opus` (1.6.1 line)
- Detail: Top-level crate docs claim "bit-exact with libopus 1.5.2". This indicates version-target drift and can mask new 1.6.x behavior/feature gaps during parity work.

109. [MEDIUM][Error Handling] CELT decoder init path can panic on invalid sampling rate/channel config instead of returning Opus error codes.
- Rust: `src/celt/celt_decoder.rs:141-152`
- Upstream: `libopus-sys/opus/celt/celt_decoder.c:224-245`
- Detail: Upstream `celt_decoder_init`/`opus_custom_decoder_init` return `OPUS_BAD_ARG` (or other status) for invalid sampling rate/channel inputs. Rust uses `panic!` branches for unsupported sample rates and channel validation, changing error semantics from return-code to abort.

110. [HIGH][C API Surface] C-style create/destroy/ctl entrypoints are not exposed in the Rust API layer.
- Rust: `src/lib.rs` exports typed structs/methods, but no `opus_encoder_create/destroy/ctl` or `opus_decoder_create/destroy/ctl` symbols
- Upstream: `libopus-sys/opus/include/opus.h:211`, `libopus-sys/opus/include/opus.h:351`, `libopus-sys/opus/include/opus.h:367`, `libopus-sys/opus/include/opus.h:477`, `libopus-sys/opus/include/opus.h:586`, `libopus-sys/opus/include/opus.h:591`
- Detail: Upstream public API is centered around C entrypoints with vararg CTL control plane. Rust exposes idiomatic methods/fields instead, so behavior/coverage parity for request dispatch and ABI-level compatibility is incomplete.

111. [HIGH][QEXT][Encoder Path] QEXT payload path is not wired in main Opus encoder call-site.
- Rust: `src/opus/opus_encoder.rs:2810-2820`, `src/celt/celt_encoder.rs:3283-3287`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:2491-2495` (enables QEXT via ctl before CELT encode)
- Detail: Rust sets `enable_qext` on CELT state but always calls `celt_encode_with_ec(..., qext_payload=None, qext_bytes=0)`. Rust CELT QEXT path is gated on `qext_bytes > 0`, so no QEXT payload is produced through this path despite QEXT enable state.

112. [MEDIUM][Decoder Controls] Ignore-extensions control path from upstream decoder is missing.
- Rust: `src/opus/opus_decoder.rs` (no `ignore_extensions` control field/request API)
- Upstream: `libopus-sys/opus/src/opus_decoder.c:1197-1207` (`OPUS_SET/GET_IGNORE_EXTENSIONS_REQUEST`)
- Detail: Upstream decoder exposes controls to ignore packet extensions during decode. Rust decoder does not mirror this control plane, so extension parsing behavior cannot be toggled equivalently.

113. [LOW][MLP Runtime Semantics] GRU layer neuron bound check is stricter than upstream (`< 32` vs allowing `== 32`).
- Rust: `src/opus/mlp/layers.rs:95-103`
- Upstream: `libopus-sys/opus/src/mlp.c:97-103`
- Detail: Upstream allocates local arrays of size `MAX_NEURONS` and allows `N` up to that bound. Rust asserts `n < MAX_NEURONS`, which rejects the edge case `n == 32` and can panic where upstream would proceed.

114. [HIGH][Frame Size Selection] `frame_size_select` omits upstream restricted-SILK application guard.
- Rust: `src/opus/opus_encoder.rs:942-974`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:827-851`
- Detail: Upstream signature includes `application` and rejects sub-10 ms frames for `OPUS_APPLICATION_RESTRICTED_SILK` (`new_size < Fs/100`). Rust function omits `application` entirely and therefore cannot enforce this branch.

115. [HIGH][Repacketizer][Extensions Preservation] Repacketizer `cat` path does not capture packet padding/extension metadata, so repacketized output cannot preserve incoming extensions.
- Rust: `src/opus/repacketizer.rs:38-44` (state lacks padding fields), `src/opus/repacketizer.rs:128-137` (`opus_packet_parse_impl(..., None, None)`)
- Upstream: `libopus-sys/opus/src/repacketizer.c:86-99` (stores `paddings/padding_len/padding_nb_frames`), `libopus-sys/opus/src/repacketizer.c:143-177` (parses and reindexes existing extensions during output)
- Detail: Upstream repacketizer tracks per-frame padding extension regions and merges them into output packets. Rust `OpusRepacketizer` drops this metadata at ingest, so extension payloads from source packets are not propagated through repacketization.

116. [HIGH][Opus Custom API Coverage] `opus_custom.h` API surface is largely missing (create/destroy/get_size/init/encode/decode/ctl entrypoints).
- Rust: `src/lib.rs:83-88` (exports only `OpusCustomEncoder`, `OpusCustomDecoder`, `opus_custom_mode_create`, with note about no mode destroy)
- Upstream: `libopus-sys/opus/include/opus_custom.h:122-373`
- Detail: Upstream exposes full Opus Custom C API (mode create/destroy, encoder/decoder lifecycle, float/int/int24 encode/decode, ctl). Rust only provides partial typed internals and omits the majority of `opus_custom_*` entrypoints and explicit mode destroy semantics.

117. [HIGH][Application Modes] Encoder initialization rejects upstream restricted application modes.
- Rust: `src/opus/opus_encoder.rs:162-165`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:218-221`, `libopus-sys/opus/src/opus_encoder.c:632-635`
- Detail: Upstream accepts `OPUS_APPLICATION_RESTRICTED_SILK` (2052) and `OPUS_APPLICATION_RESTRICTED_CELT` (2053) in both init and create paths. Rust validation only allows VoIP/Audio/Restricted-LowDelay and returns `OPUS_BAD_ARG` for restricted SILK/CELT.

118. [MEDIUM][CTL Semantics] Encoder bitrate setter clamps to a lower maximum than upstream.
- Rust: `src/opus/opus_encoder.rs:321-322`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:2826-2827`
- Detail: Upstream clamps explicit bitrate to `750000 * channels`. Rust clamps to `300000 * channels`, reducing accepted bitrate range and changing CTL behavior for high-bitrate configurations.

119. [HIGH][DRED API Coverage] Public DRED API from `opus.h` is not exposed in the Rust surface.
- Rust: no exported `opus_dred_*` / `opus_decoder_dred_decode*` API in `src/lib.rs`
- Upstream: `libopus-sys/opus/include/opus.h:596-697`
- Detail: Upstream 1.6.1 exposes DRED decoder/state lifecycle (`opus_dred_decoder_*`, `opus_dred_*`) plus decode entrypoints (`opus_decoder_dred_decode`, `...decode24`, `...decode_float`). Rust contains internal DRED modules but does not provide the corresponding public API set.

120. [MEDIUM][Packet API Coverage] Multistream packet pad/unpad helpers are missing.
- Rust: no `opus_multistream_packet_pad` / `opus_multistream_packet_unpad` exports in `src/lib.rs`
- Upstream: `libopus-sys/opus/include/opus.h:1152`, `libopus-sys/opus/include/opus.h:1167`
- Detail: Upstream provides helpers to pad/unpad multi-stream Opus packets. Rust currently exposes only single-stream packet pad/unpad helpers.

121. [MEDIUM][CTL Semantics] Rust bitrate setter does not preserve upstream `<=0` rejection semantics.
- Rust: `src/opus/opus_encoder.rs:318-325`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:2820-2825`
- Detail: Upstream returns `OPUS_BAD_ARG` for explicit bitrate values `<= 0` (except sentinels `OPUS_AUTO`/`OPUS_BITRATE_MAX`). Rust `set_bitrate()` accepts any `Bitrate::Bits(v)` and clamps values below 500 up to 500, including non-positive values.

122. [MEDIUM][API Coverage][Sizing/Lifecycle] `*_get_size`/C-style lifecycle functions are broadly missing for core and repacketizer APIs.
- Rust: no exported `opus_encoder_get_size`, `opus_decoder_get_size`, `opus_repacketizer_get_size` in `src/lib.rs`
- Upstream: `libopus-sys/opus/include/opus.h:174`, `libopus-sys/opus/include/opus.h:460`, `libopus-sys/opus/include/opus.h:953`
- Detail: Upstream C API exposes size-query and explicit init/create/destroy workflow for externally allocated states. Rust API is object-centric and does not mirror these ABI-level allocation/lifecycle entrypoints.

123. [LOW][DNN Module Coverage] Several upstream DNN runtime modules are not ported in Rust (e.g., FWGAN/lossgen paths).
- Rust: `src/dnn/` (no `fwgan.rs`, no `lossgen.rs` module equivalents)
- Upstream: `libopus-sys/opus/dnn/fwgan.c`, `libopus-sys/opus/dnn/lossgen.c` (and related headers/data)
- Detail: Rust DNN coverage includes LPCNet/OSCE/DRED/FARGAN components, but upstream repository contains additional DNN runtime components not mirrored in Rust sources.

124. [MEDIUM][CTL Semantics] `set_bandwidth(None)` does not mirror upstream side-effect on SILK max internal sample rate.
- Rust: `src/opus/opus_encoder.rs:356-369`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:2890-2903`
- Detail: Upstream `OPUS_SET_BANDWIDTH(OPUS_AUTO)` still sets `silk_mode.maxInternalSampleRate = 16000`. Rust `set_bandwidth(None)` only sets `user_bandwidth = OPUS_AUTO` and leaves prior `maxInternalSampleRate` unchanged.

125. [LOW][Lookahead Semantics] `lookahead()` omits upstream restricted-CELT exclusion when adding delay compensation.
- Rust: `src/opus/opus_encoder.rs:531-535`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:3089-3091`
- Detail: Upstream adds `delay_compensation` only when application is neither `RESTRICTED_LOWDELAY` nor `RESTRICTED_CELT`. Rust checks only `RESTRICTED_LOWDELAY`.

126. [MEDIUM][DRED API Semantics] Rust `opus_dred_process` helper has different signature/return semantics from upstream API.
- Rust: `src/dnn/dred/decoder.rs:195-208`
- Upstream: `libopus-sys/opus/include/opus.h:661`, `libopus-sys/opus/src/opus_decoder.c:1587-1600`
- Detail: Upstream API is `int opus_dred_process(OpusDREDDecoder*, const OpusDRED*, OpusDRED*)` returning Opus status codes with argument validation and copy-on-src!=dst behavior. Rust helper is in-place `fn opus_dred_process(&OpusDREDDecoder, &mut OpusDRED)` returning `()`, uses assertion for loaded-state, and omits status/error reporting semantics.

127. [LOW][Header/Feature Flags] Upstream projection-feature indicator macro has no Rust equivalent.
- Rust: no exported equivalent of `OPUS_HAVE_OPUS_PROJECTION_H` in API constants
- Upstream: `libopus-sys/opus/include/opus_defines.h:184`
- Detail: Upstream headers expose `OPUS_HAVE_OPUS_PROJECTION_H` to signal projection API availability at compile time. Rust constant surface does not provide an equivalent feature-availability marker.

128. [LOW][Packet Parser API] `opus_packet_parse_impl` does not expose upstream padding-pointer output semantics.
- Rust: `src/opus/packet.rs:172-181` (only `padding_out: Option<&mut i32>`)
- Upstream: `libopus-sys/opus/src/opus.c:224-228` (`const unsigned char **padding, opus_int32 *padding_len`)
- Detail: Upstream parse API can return both padding length and a pointer to padding bytes. Rust implementation returns only padding length, which is insufficient for API-equivalent pointer-based padding reuse workflows.

129. [LOW][Private CTL Coverage] Encoder voice-ratio control API is not surfaced.
- Rust: `src/opus/opus_encoder.rs` (no `set_voice_ratio`/`voice_ratio` public API)
- Upstream: `libopus-sys/opus/src/opus_private.h:152-169`, `libopus-sys/opus/src/opus_encoder.c:3022-3040`
- Detail: Upstream implements `OPUS_SET_VOICE_RATIO_REQUEST` / `OPUS_GET_VOICE_RATIO_REQUEST` (range `-1..100`) to bias mode decisions. Rust keeps an internal `voice_ratio` state but does not expose a corresponding control/getter surface.

130. [LOW][API Surface Divergence] Upstream-private force-mode control is exposed as public Rust API.
- Rust: `src/opus/opus_encoder.rs:515-520`, `src/lib.rs:90-91`
- Upstream: `libopus-sys/opus/src/opus_private.h:172-173` (`OPUS_SET_FORCE_MODE_REQUEST` is private, not in `include/opus.h`)
- Detail: Rust publicly exposes `set_force_mode()` and re-exports `opus_private`, while upstream treats force-mode as an internal/private control. This is an API-surface divergence in the opposite direction (extra public controls not present in the public C API).

131. [LOW][Version Targeting][Docs] DNN subsystem documentation still targets 1.5.2.
- Rust: `src/dnn/README.md:3`
- Upstream target for this review: `libopus-sys/opus` (1.6.1 line)
- Detail: The DNN README states "Opus 1.5.2". Even though primarily documentation, this version marker diverges from the 1.6.1 parity target and can mislead maintenance/prioritization for newer upstream DNN behavior.

132. [LOW][Decoder API Semantics] Rust decode entrypoints cannot represent upstream `(data == NULL, len > 0)` call form.
- Rust: `src/opus/opus_decoder.rs:124-131`, `src/opus/opus_decoder.rs:142-149`, `src/opus/opus_decoder.rs:900-927`
- Upstream: `libopus-sys/opus/src/opus_decoder.c:896-921`, `libopus-sys/opus/src/opus_decoder.c:985-990`
- Detail: Upstream decode APIs take `(const unsigned char *data, opus_int32 len)` and treat `data == NULL` as packet-loss/PLC regardless of `len`. Rust slice-based decode APIs (`data: &[u8]`) can only encode PLC via empty slices, so this pointer/length-state API form is not representable.

133. [LOW][Version Targeting][Docs] Top-level project README still advertises 1.5.2 parity.
- Rust: `README.md:7`, `README.md:12`
- Upstream target for this review: `libopus-sys/opus` (1.6.1 line)
- Detail: Repository-level docs still claim bit-exactness with libopus 1.5.2, which is inconsistent with the ongoing 1.6.1 parity target and may obscure newer 1.6.x feature/behavior gaps.

134. [LOW][Version Targeting][Comments] Source-level implementation comments still reference 1.5.2 in active modules.
- Rust: `src/dnn/simd/x86.rs:6`, `src/dnn/simd/aarch64.rs:6`, `src/opus/mlp/tansig.rs:5`
- Upstream target for this review: `libopus-sys/opus` (1.6.1 line)
- Detail: Several in-code provenance comments still pin behavior/ports to 1.5.2, which can mislead future parity verification against 1.6.1 when reviewing these function paths.

135. [LOW][Runtime Semantics][DRED] DRED encoder loaded-state checks are unconditional panics in Rust.
- Rust: `src/dnn/dred/encoder.rs:290`, `src/dnn/dred/encoder.rs:334`
- Upstream: `libopus-sys/opus/dnn/dred_encoder.c:95`, `libopus-sys/opus/dnn/dred_encoder.c:215`
- Detail: Upstream guards these paths with `celt_assert(enc->loaded)` (assert-gated behavior). Rust uses unconditional `assert!(enc.loaded)`, which can abort at runtime in builds where upstream would typically compile assertions out.

136. [LOW][Runtime Semantics][DNN] LPCNet/FARGAN state precondition checks are unconditional assertions in Rust.
- Rust: `src/dnn/lpcnet.rs:623`, `src/dnn/lpcnet.rs:726`, `src/dnn/fargan.rs:441`, `src/dnn/fargan.rs:671`
- Upstream: `libopus-sys/opus/dnn/lpcnet_plc.c:110`, `libopus-sys/opus/dnn/lpcnet_plc.c:159`, `libopus-sys/opus/dnn/fargan.c:85`, `libopus-sys/opus/dnn/fargan.c:202`
- Detail: Upstream uses `celt_assert(...)` for loaded/continuation-state invariants in these paths. Rust uses unconditional `assert!`, which changes failure mode from assert-gated behavior to hard runtime aborts when state preconditions are violated.

137. [LOW][Runtime Semantics][DNN] Core `nnet`/`nndsp` invariant checks are unconditional assertions in Rust.
- Rust: `src/dnn/nnet.rs:220-221`, `src/dnn/nnet.rs:254`, `src/dnn/nnet.rs:281`, `src/dnn/nnet.rs:310`, `src/dnn/nndsp.rs:171`, `src/dnn/nndsp.rs:396-400`
- Upstream: `libopus-sys/opus/dnn/nnet.c:85-86`, `libopus-sys/opus/dnn/nnet.c:111`, `libopus-sys/opus/dnn/nnet.c:128`, `libopus-sys/opus/dnn/nnet.c:142`, `libopus-sys/opus/dnn/nndsp.c:169`, `libopus-sys/opus/dnn/nndsp.c:364-366`
- Detail: Upstream encodes these as `celt_assert(...)` internal invariants. Rust promotes them to unconditional `assert!`, so malformed/inconsistent dimensions can panic in production builds where upstream often compiles these checks out.

138. [MEDIUM][Packet API][Error Handling] `opus_packet_pad_impl` can panic on slice bounds instead of returning Opus error codes.
- Rust: `src/opus/repacketizer.rs:480`, `src/opus/repacketizer.rs:491`
- Upstream: `libopus-sys/opus/src/repacketizer.c:341-347`, `libopus-sys/opus/src/repacketizer.c:354`
- Detail: Rust indexes `data[..len as usize]` and `data[..new_len as usize]` without checking against `data.len()`. If caller passes inconsistent `len/new_len` relative to slice size, Rust panics. Upstream follows pointer+length conventions and returns Opus status codes for its explicit argument checks rather than triggering a safe-slice panic mode.

139. [LOW][Runtime Semantics][Repacketizer] `opus_packet_unpad` uses unconditional `assert!` for postcondition check.
- Rust: `src/opus/repacketizer.rs:546`
- Upstream: `libopus-sys/opus/src/repacketizer.c:388`
- Detail: Upstream uses `celt_assert(ret > 0 && ret <= len)` for an internal invariant. Rust uses unconditional `assert!`, which can hard-abort at runtime where upstream non-assert builds would typically not.

140. [LOW][Runtime Semantics][CELT Entropy/Rate/VQ] Internal invariant checks are unconditional in Rust.
- Rust: `src/celt/entenc.rs:215`, `src/celt/entenc.rs:238`, `src/celt/entenc.rs:262`, `src/celt/entenc.rs:280`, `src/celt/entdec.rs:204`, `src/celt/laplace.rs:67-68`, `src/celt/laplace.rs:106-109`, `src/celt/cwrs.rs:298`, `src/celt/cwrs.rs:319`, `src/celt/cwrs.rs:331-332`, `src/celt/rate.rs:229`, `src/celt/rate.rs:286`, `src/celt/rate.rs:333-334`, `src/celt/rate.rs:340`, `src/celt/rate.rs:660`, `src/celt/vq.rs:509-510`, `src/celt/vq.rs:592-593`
- Upstream: `libopus-sys/opus/celt/entenc.c:191`, `libopus-sys/opus/celt/entenc.c:209`, `libopus-sys/opus/celt/entenc.c:228`, `libopus-sys/opus/celt/entenc.c:249`, `libopus-sys/opus/celt/entdec.c:224`, `libopus-sys/opus/celt/laplace.c:88-89`, `libopus-sys/opus/celt/laplace.c:128-131`, `libopus-sys/opus/celt/cwrs.c:448`, `libopus-sys/opus/celt/cwrs.c:463`, `libopus-sys/opus/celt/cwrs.c:473-474`, `libopus-sys/opus/celt/rate.c:394`, `libopus-sys/opus/celt/rate.c:445`, `libopus-sys/opus/celt/rate.c:516-517`, `libopus-sys/opus/celt/rate.c:527`, `libopus-sys/opus/celt/rate.c:750`, `libopus-sys/opus/celt/vq.c:46`, `libopus-sys/opus/celt/vq.c:53`
- Detail: Upstream marks these checks as `celt_assert(...)` internal invariants (typically assertion-gated). Rust uses unconditional `assert!`, changing failure behavior to hard runtime aborts when invariants are violated.

141. [LOW][Runtime Semantics][CELT LPC/Pitch/Bands] Additional internal invariants are unconditional in Rust.
- Rust: `src/celt/celt_lpc.rs:18`, `src/celt/celt_lpc.rs:103`, `src/celt/celt_lpc.rs:175-176`, `src/celt/pitch.rs:327`, `src/celt/pitch.rs:349-350`, `src/celt/bands.rs:265`, `src/celt/bands.rs:441`, `src/celt/bands.rs:505-506`, `src/celt/bands.rs:531`, `src/celt/bands.rs:626`, `src/celt/bands.rs:803`, `src/celt/bands.rs:1314`, `src/celt/bands.rs:1959`
- Upstream: `libopus-sys/opus/celt/celt_lpc.c:225`, `libopus-sys/opus/celt/celt_lpc.c:302-303`, `libopus-sys/opus/celt/pitch.c:265`, `libopus-sys/opus/celt/pitch.c:325-326`, `libopus-sys/opus/celt/bands.c:254`, `libopus-sys/opus/celt/bands.c:480`, `libopus-sys/opus/celt/bands.c:536-537`, `libopus-sys/opus/celt/bands.c:582`, `libopus-sys/opus/celt/bands.c:660`, `libopus-sys/opus/celt/bands.c:840`, `libopus-sys/opus/celt/bands.c:1182`, `libopus-sys/opus/celt/bands.c:1705`
- Detail: These upstream checks are `celt_assert(...)` invariants; Rust turns them into unconditional `assert!`, so invariant violations can terminate runtime processing in configurations where upstream non-assert builds would not.

142. [LOW][Runtime Semantics][SILK] Additional SILK invariant checks are unconditional `assert!` in Rust.
- Rust: `src/silk/control_codec.rs:167-168`, `src/silk/control_codec.rs:222`, `src/silk/control_codec.rs:228`, `src/silk/control_codec.rs:307-312`, `src/silk/decode_frame.rs:49`, `src/silk/decode_frame.rs:84`, `src/silk/decode_frame.rs:110`, `src/silk/decode_frame.rs:132`, `src/silk/decode_core.rs:133`, `src/silk/decode_core.rs:204`, `src/silk/NSQ.rs:220`, `src/silk/NSQ.rs:423`, `src/silk/NSQ.rs:441`, `src/silk/NSQ_del_dec.rs:311`, `src/silk/NSQ_del_dec.rs:541`, `src/silk/NSQ_del_dec.rs:608`, `src/silk/float/pitch_analysis_core_FLP.rs:90-92`, `src/silk/float/pitch_analysis_core_FLP.rs:218`, `src/silk/float/pitch_analysis_core_FLP.rs:244`, `src/silk/float/find_pred_coefs_FLP.rs:42`
- Upstream: `libopus-sys/opus/silk/control_codec.c:241-242`, `libopus-sys/opus/silk/control_codec.c:302`, `libopus-sys/opus/silk/control_codec.c:315`, `libopus-sys/opus/silk/control_codec.c:393-398`, `libopus-sys/opus/silk/decode_frame.c:68`, `libopus-sys/opus/silk/decode_frame.c:104`, `libopus-sys/opus/silk/decode_frame.c:127`, `libopus-sys/opus/silk/decode_frame.c:145`, `libopus-sys/opus/silk/NSQ.c:402`, `libopus-sys/opus/silk/NSQ_del_dec.c:686`, `libopus-sys/opus/silk/float/pitch_analysis_core_FLP.c:118-119`, `libopus-sys/opus/silk/float/pitch_analysis_core_FLP.c:532`, `libopus-sys/opus/silk/float/find_pred_coefs_FLP.c:55`
- Detail: Upstream uses `celt_assert`/`silk_assert` for many of these invariants. Rust promotes them to unconditional `assert!`, which can hard-abort runtime encode/decode flows in non-assert configurations where upstream would usually not.

143. [LOW][Runtime Semantics][OSCE/Freq/SILK-FLP] Additional assert-gated upstream checks are unconditional in Rust.
- Rust: `src/dnn/osce.rs:1753`, `src/dnn/osce.rs:1860`, `src/dnn/freq.rs:307-308`, `src/silk/float/LPC_analysis_filter_FLP.rs:110`, `src/silk/float/LPC_analysis_filter_FLP.rs:128`
- Upstream: `libopus-sys/opus/dnn/osce_features.c:363`, `libopus-sys/opus/dnn/osce_features.c:548`, `libopus-sys/opus/silk/float/LPC_analysis_filter_FLP.c:218`, `libopus-sys/opus/silk/float/LPC_analysis_filter_FLP.c:242`
- Detail: Upstream uses `celt_assert(...)` for these invariant/default branches. Rust uses unconditional `assert!` and `panic!(\"libopus: assert(0) called\")`, so invalid/precondition-breaking inputs can hard-abort where upstream non-assert builds would typically not.

144. [LOW][Runtime Semantics][CELT FFT] KISS FFT precondition checks are unconditional in Rust.
- Rust: `src/celt/kiss_fft.rs:29-30`, `src/celt/kiss_fft.rs:69`, `src/celt/kiss_fft.rs:212`, `src/celt/kiss_fft.rs:247-248`
- Upstream: `libopus-sys/opus/celt/kiss_fft.c:64`, `libopus-sys/opus/celt/kiss_fft.c:80`
- Detail: Upstream uses `celt_assert(...)` for internal butterfly preconditions (e.g., radix staging constraints). Rust enforces corresponding shape/length assumptions with unconditional `assert_eq!`, which can abort at runtime instead of remaining assert-gated behavior.

145. [LOW][Runtime Semantics][SILK FLP] Additional float-path invariant checks are unconditional in Rust.
- Rust: `src/silk/float/apply_sine_window_FLP.rs:12-13`, `src/silk/float/schur_FLP.rs:13`, `src/silk/float/sort_FLP.rs:10-12`, `src/silk/float/burg_modified_FLP.rs:36`, `src/silk/float/warped_autocorrelation_FLP.rs:25`, `src/silk/float/find_pitch_lags_FLP.rs:34`
- Upstream: `libopus-sys/opus/silk/float/apply_sine_window_FLP.c:48`, `libopus-sys/opus/silk/float/apply_sine_window_FLP.c:51`, `libopus-sys/opus/silk/float/schur_FLP.c:44`, `libopus-sys/opus/silk/float/sort_FLP.c:50-52`, `libopus-sys/opus/silk/float/burg_modified_FLP.c:56`, `libopus-sys/opus/silk/float/warped_autocorrelation_FLP.c:49`, `libopus-sys/opus/silk/float/find_pitch_lags_FLP.c:59`
- Detail: Upstream marks these as `celt_assert`/`silk_assert` invariants. Rust uses unconditional `assert!`, so invariant violations can hard-abort runtime float encoder paths where upstream non-assert builds typically would not.

146. [LOW][Runtime Semantics][CELT MDCT] MDCT wrappers enforce unconditional slice-length assertions in Rust.
- Rust: `src/celt/mdct.rs:48-49`, `src/celt/mdct.rs:52`, `src/celt/mdct.rs:55`, `src/celt/mdct.rs:57`
- Upstream: `libopus-sys/opus/celt/mdct.c:122`, `libopus-sys/opus/celt/mdct.c:268`
- Detail: Upstream MDCT APIs operate on raw pointers and internal size assumptions; Rust wrappers add unconditional `assert!`/`assert_eq!` checks for window/trig/input/output sizing. This introduces hard-abort behavior on mismatched buffers instead of upstream-style unchecked pointer semantics.

147. [LOW][Validation Semantics][SILK] Some upstream assert-gated range checks are missing in Rust counterparts.
- Rust: `src/silk/process_NLSFs.rs:25-35`, `src/silk/LPC_analysis_filter.rs:27-29`, `src/silk/NLSF_encode.rs:40`
- Upstream: `libopus-sys/opus/silk/process_NLSFs.c:49-51`, `libopus-sys/opus/silk/process_NLSFs.c:63-64`, `libopus-sys/opus/silk/process_NLSFs.c:84`, `libopus-sys/opus/silk/LPC_analysis_filter.c:67-72`, `libopus-sys/opus/silk/NLSF_encode.c:63-64`
- Detail: Upstream includes additional invariant/range checks (for example speech-activity bounds, `NLSF_mu_Q20` upper bound, per-weight lower bounds, and `d <= SILK_MAX_ORDER_LPC`) in these paths. Rust validates only a subset, so behavior on malformed/internal-state-corrupt inputs diverges from the upstream assert-guard model.

148. [LOW][Runtime Semantics][SILK Helpers] Additional helper-module invariants are unconditional in Rust.
- Rust: `src/silk/CNG.rs:154`, `src/silk/VAD.rs:82-84`, `src/silk/interpolate.rs:17`, `src/silk/NLSF_VQ_weights_laroia.rs:27`, `src/silk/decode_pulses.rs` (no equivalent frame-length assert branch), `src/silk/encode_pulses.rs` (no equivalent frame-length assert branch), `src/silk/encode_indices.rs:41-42`, `src/silk/encode_indices.rs:92`, `src/silk/stereo_encode_pred.rs:15`, `src/silk/stereo_encode_pred.rs:19-20`, `src/silk/sort.rs:10-12`, `src/silk/resampler/down2.rs:16-17`, `src/silk/resampler/up2_hq.rs` (coefficient-sign asserts absent), `src/silk/shell_coder.rs` (shell-frame-length assert not mirrored)
- Upstream: `libopus-sys/opus/silk/CNG.c:153`, `libopus-sys/opus/silk/VAD.c:104-106`, `libopus-sys/opus/silk/interpolate.c:45-46`, `libopus-sys/opus/silk/NLSF_VQ_weights_laroia.c:51-52`, `libopus-sys/opus/silk/decode_pulses.c:56,59`, `libopus-sys/opus/silk/encode_pulses.c:86,89`, `libopus-sys/opus/silk/encode_indices.c:59-60,93`, `libopus-sys/opus/silk/stereo_encode_pred.c:44,47-48`, `libopus-sys/opus/silk/sort.c:51-53`, `libopus-sys/opus/silk/resampler_down2.c:46-47`, `libopus-sys/opus/silk/resampler_private_up2_HQ.c:48-53`, `libopus-sys/opus/silk/shell_coder.c:86,128`
- Detail: These helper paths rely on assert-gated invariants upstream. Rust either enforces them with unconditional `assert!` (hard-abort behavior) or omits some of the assert checks entirely, creating mixed divergence in failure semantics under invalid/internal-corrupt states.

149. [LOW][Runtime Semantics][DNN Burg] Burg-analysis invariants are unconditional `assert!` in Rust.
- Rust: `src/dnn/burg.rs:29`, `src/dnn/burg.rs:76-77`, `src/dnn/burg.rs:139-140`, `src/dnn/burg.rs:144`
- Upstream: `libopus-sys/opus/dnn/burg.c:66`, `libopus-sys/opus/dnn/burg.c:115`, `libopus-sys/opus/dnn/burg.c:173-174`, `libopus-sys/opus/dnn/burg.c:178`
- Detail: Upstream uses C `assert(...)` invariants in this path. Rust uses unconditional `assert!`, so invalid-state handling remains abortive rather than explicit error returns and mirrors assert-enabled behavior only.

150. [HIGH][QEXT][Sampling Rate] `resampling_factor` is missing upstream 96 kHz handling under QEXT.
- Rust: `src/celt/common.rs:32-40`, call sites `src/celt/celt_encoder.rs:142`, `src/celt/celt_encoder.rs:148`, `src/celt/celt_decoder.rs:130`, `src/celt/celt_decoder.rs:136`
- Upstream: `libopus-sys/opus/celt/celt.c:67-72` (96 kHz case under `ENABLE_QEXT`), `libopus-sys/opus/celt/celt.c:85-90`
- Detail: Upstream maps `96000 -> 1` when QEXT is enabled, and falls back to `ret=0` in default branch. Rust has no `96000` arm and panics in default. With Rust `qext` feature enabled, 96 kHz init paths can still trip this panic despite upstream-supported handling.

151. [LOW][Validation Semantics][SILK LP] `silk_LP_variable_cutoff` omits multiple upstream invariant checks.
- Rust: `src/silk/LP_variable_cutoff.rs:97-104`, `src/silk/LP_variable_cutoff.rs:120`
- Upstream: `libopus-sys/opus/silk/LP_variable_cutoff.c:122-123`, `libopus-sys/opus/silk/LP_variable_cutoff.c:132`
- Detail: Upstream includes assert-gated checks for interpolation index bounds (`ind >= 0`, `ind < TRANSITION_INT_NUM`) and filter-shape invariant (`TRANSITION_NB == 3 && TRANSITION_NA == 2`). Rust enforces only a subset of LP-state invariants here, so malformed/internal-corrupt state coverage diverges from upstream assertion model.

152. [LOW][Runtime Semantics][SILK Wrapper Preconditions] Several Rust helper wrappers add unconditional slice-shape asserts not present in upstream pointer APIs.
- Rust: `src/silk/LPC_fit.rs:22`, `src/silk/NLSF_VQ.rs:22`, `src/silk/NLSF_decode.rs:22-23`, `src/silk/biquad_alt.rs:35`, `src/silk/code_signs.rs:40`, `src/silk/code_signs.rs:77`, `src/silk/resampler/ar2.rs:13`, `src/silk/stereo_MS_to_LR.rs:32-34`, `src/silk/float/find_LTP_FLP.rs:40`, `src/silk/float/corrMatrix_FLP.rs:64`, `src/silk/float/inner_product_FLP.rs:47`
- Upstream: `libopus-sys/opus/silk/LPC_fit.c:36`, `libopus-sys/opus/silk/NLSF_VQ.c:35`, `libopus-sys/opus/silk/NLSF_decode.c:63`, `libopus-sys/opus/silk/biquad_alt.c:42`, `libopus-sys/opus/silk/code_signs.c:41`, `libopus-sys/opus/silk/code_signs.c:75`, `libopus-sys/opus/silk/resampler_private_AR2.c:36`, `libopus-sys/opus/silk/stereo_MS_to_LR.c:35`, `libopus-sys/opus/silk/float/find_LTP_FLP.c:35`, `libopus-sys/opus/silk/float/corrMatrix_FLP.c:60`, `libopus-sys/opus/silk/float/inner_product_FLP.c:35`
- Detail: These C entry points are pointer-based and do not expose equivalent runtime shape checks. Rust adds unconditional `assert!`/`assert_eq!` preconditions, which can panic on mismatched buffer geometry where upstream behavior is unchecked/assert-gated.

153. [LOW][Runtime Semantics][SILK] Additional upstream assert-gated invariants are unconditional `assert!` in Rust.
- Rust: `src/silk/NLSF2A.rs:66`, `src/silk/NLSF_stabilize.rs:39`, `src/silk/PLC.rs:298`, `src/silk/PLC.rs:364`, `src/silk/decode_indices.rs:90`, `src/silk/float/encode_frame_FLP.rs:322`, `src/silk/float/encode_frame_FLP.rs:356`, `src/silk/float/find_LPC_FLP.rs:102`
- Upstream: `libopus-sys/opus/silk/NLSF2A.c:89`, `libopus-sys/opus/silk/NLSF_stabilize.c:58`, `libopus-sys/opus/silk/PLC.c:319`, `libopus-sys/opus/silk/PLC.c:373`, `libopus-sys/opus/silk/decode_indices.c:82`, `libopus-sys/opus/silk/float/encode_frame_FLP.c:261`, `libopus-sys/opus/silk/float/encode_frame_FLP.c:291`, `libopus-sys/opus/silk/float/find_LPC_FLP.c:103`
- Detail: Upstream keeps these as `celt_assert`/`silk_assert` checks. Rust enforces corresponding conditions with unconditional `assert!`, so invariant violations hard-abort at runtime in builds where upstream would typically compile such checks out.

154. [LOW][Runtime Semantics][SILK Wrapper Preconditions] Additional Rust-only shape/size assertions have no upstream equivalent checks.
- Rust: `src/silk/NLSF2A.rs:30`, `src/silk/gain_quant.rs:129`, `src/silk/float/SigProc_FLP.rs:18`, `src/silk/float/SigProc_FLP.rs:29`
- Upstream: `libopus-sys/opus/silk/NLSF2A.c:80`, `libopus-sys/opus/silk/gain_quant.c:95`, `libopus-sys/opus/silk/float/SigProc_FLP.h:118`, `libopus-sys/opus/silk/float/SigProc_FLP.h:126`
- Detail: These Rust ports impose slice-length/cardinality preconditions via `assert!`/`assert_eq!` that are not present in the upstream C entry points. Mismatched inputs therefore panic in Rust instead of following upstream pointer-level behavior.

155. [LOW][Validation Semantics][SILK] Additional upstream assert-gated checks are missing in Rust counterparts.
- Rust: `src/silk/NLSF_del_dec_quant.rs:53-230`, `src/silk/stereo_find_predictor.rs:54-68`, `src/silk/stereo_LR_to_MS.rs:148-156`, `src/silk/float/process_gains_FLP.rs:78-84`, `src/silk/float/wrappers_FLP.rs:126-149`, `src/silk/float/energy_FLP.rs:8-13`, `src/silk/float/noise_shape_analysis_FLP.rs:29-103`, `src/silk/float/noise_shape_analysis_FLP.rs:107-133`
- Upstream: `libopus-sys/opus/silk/NLSF_del_dec_quant.c:82`, `libopus-sys/opus/silk/NLSF_del_dec_quant.c:208-213`, `libopus-sys/opus/silk/stereo_find_predictor.c:64`, `libopus-sys/opus/silk/stereo_LR_to_MS.c:113`, `libopus-sys/opus/silk/float/process_gains_FLP.c:101-102`, `libopus-sys/opus/silk/float/wrappers_FLP.c:148`, `libopus-sys/opus/silk/float/energy_FLP.c:57`, `libopus-sys/opus/silk/float/noise_shape_analysis_FLP.c:113`, `libopus-sys/opus/silk/float/noise_shape_analysis_FLP.c:143`
- Detail: Upstream includes assert-gated invariants in these paths (quantizer-state shape, output-index bounds, smooth/rate constraints, lambda/gain/energy bounds, and convergence asserts in noise-shape helpers). Rust implementations currently omit these checks, so invalid/internal-corrupt states are no longer guarded the same way.

156. [LOW][Runtime Semantics][SILK Pitch Decode] Invalid `(Fs_kHz, nb_subfr)` combinations panic in Rust where upstream keeps assert-gated fallback table selection.
- Rust: `src/silk/decode_pitch.rs:26-34`
- Upstream: `libopus-sys/opus/silk/decode_pitch.c:49-66`
- Detail: Upstream uses `celt_assert(nb_subfr == PE_MAX_NB_SUBFR >> 1)` in fallback branches but still assigns stage2/stage3 10 ms codebooks. Rust uses a strict `(Fs_kHz, nb_subfr)` match and calls `unreachable!` otherwise, producing hard panic instead of upstream assert-gated fallback behavior.

157. [MEDIUM][Module Coverage][SILK FLP] `silk_residual_energy_covar_FLP` is not implemented in Rust.
- Rust: `src/silk/float/residual_energy_FLP.rs` (contains `silk_residual_energy_FLP` only)
- Upstream: `libopus-sys/opus/silk/float/residual_energy_FLP.c:38-87`, declaration in `libopus-sys/opus/silk/float/main_FLP.h:199`
- Detail: Upstream provides both `silk_residual_energy_FLP` and `silk_residual_energy_covar_FLP` (weighted covariance-form residual energy with regularization loop). Rust ports only the former, leaving this helper absent from the FLP module surface.

158. [LOW][Validation Semantics][CELT Quant Bands] Upstream channel-index invariant check is not mirrored in Rust.
- Rust: `src/celt/quant_bands.rs` (`unquant_coarse_energy` path; no equivalent `c < 2` assert check found)
- Upstream: `libopus-sys/opus/celt/quant_bands.c:463`
- Detail: Upstream keeps an explicit `celt_sig_assert(c < 2)` invariant in coarse energy decode loop. Rust omits the corresponding check, so this internal-state guard is not preserved under parity review.

159. [LOW][Validation Semantics][CELT Entcode] Upstream divisor precondition asserts are not mirrored in Rust helpers.
- Rust: `src/celt/entcode.rs:115-123`
- Upstream: `libopus-sys/opus/celt/entcode.h:125`, `libopus-sys/opus/celt/entcode.h:141`
- Detail: Upstream `celt_udiv`/`celt_sudiv` include `celt_sig_assert(d > 0)` preconditions. Rust wrappers call `wrapping_div`/`/` directly without equivalent invariant checks, so the upstream assert-guard contract is not preserved.

160. [LOW][Tooling Semantics][opus_demo] Rust demo wrapper rejects upstream-supported FEC/loss simulation options.
- Rust: `src/tools/demo/mod.rs:75-80`, `src/tools/demo/mod.rs:176-181`
- Upstream: `libopus-sys/opus/src/opus_demo.c:644-645`, `libopus-sys/opus/src/opus_demo.c:663-669`, `libopus-sys/opus/src/opus_demo.c:672-690`, `libopus-sys/opus/src/opus_demo.c:822`, `libopus-sys/opus/src/opus_demo.c:825`, `libopus-sys/opus/src/opus_demo.c:1082-1093`
- Detail: Upstream `opus_demo` supports `-inbandfec` and multiple loss simulation modes (`-loss`, `-sim_loss`, `-lossfile`, `-enc_loss`). Rust `opus_demo_encode/decode` currently `panic!` when `inbandfec` is enabled or `loss != 0`, so operational behavior diverges for these option paths.

161. [LOW][Tooling Coverage][opus_demo] Rust demo option surface is a subset of upstream CLI features.
- Rust: `src/tools/demo/input.rs:249-274` (limited `CommonOptions`/`EncoderOptions` set)
- Upstream: `libopus-sys/opus/src/opus_demo.c:145-151`, `libopus-sys/opus/src/opus_demo.c:682-700`
- Detail: Upstream CLI includes additional packet-loss and stress-test options (for example `-sim_loss`, `-lossfile`, `-enc_loss`, `-random_fec` and related sweep/test modes). Rust demo input model intentionally omits these controls, so tool-level parity is incomplete.

162. [LOW][Tooling Coverage][opus_demo] Decoder `ignore_extensions` CLI control is not represented in Rust demo args.
- Rust: `src/tools/demo/input.rs:249-311` (`CommonOptions`/`DecodeArgs` do not include an ignore-extensions toggle)
- Upstream: `libopus-sys/opus/src/opus_demo.c:742-744`, `libopus-sys/opus/src/opus_demo.c:857`
- Detail: Upstream `opus_demo` supports `-ignore_extensions` and forwards it to decoder CTL (`OPUS_SET_IGNORE_EXTENSIONS`). Rust demo argument surface has no corresponding decode option, so this behavior cannot be exercised through the Rust tool wrapper.

163. [LOW][API Surface][DNN PitchDNN] Upstream one-shot model-loading helper is not mirrored as a direct Rust API.
- Rust: `src/dnn/pitchdnn.rs:154-166` (`PitchDNNState::init(&[WeightArray])`), `src/dnn/weights.rs:37-38` (`load_weights` parser helper)
- Upstream: `libopus-sys/opus/dnn/pitchdnn.c:71-79` (`pitchdnn_load_model(PitchDNNState*, const void*, int)`)
- Detail: Upstream exposes a direct blob-based `pitchdnn_load_model` convenience entry point that parses and installs weights in one call. Rust currently provides the same flow as two separate steps (parse blob, then `init` with arrays), but no direct PitchDNN blob-loading method.

164. [LOW][Tooling Semantics][Demo Backend] Rust demo backend uses `panic!` for some feature-gated DNN/DRED operations instead of recoverable status flow.
- Rust: `src/tools/demo/backend.rs:120`, `src/tools/demo/backend.rs:131`, `src/tools/demo/backend.rs:175`
- Upstream: `libopus-sys/opus/src/opus_demo.c:830-833`, `libopus-sys/opus/src/opus_demo.c:923-927` (feature-gated paths are compile-time conditioned and wired through ctl calls)
- Detail: Rust demo backend aborts with `panic!` when required features (`dred`, `builtin-weights`, `deep-plc`) are unavailable for requested operations. Upstream demo behavior is controlled via compile-time feature paths and ctl calls rather than Rust-style hard panics in wrapper methods.

165. [LOW][Runtime Semantics][SILK Resampler] Down-FIR interpolation default branch is an unconditional panic in Rust.
- Rust: `src/silk/resampler/down_fir.rs:214`
- Upstream: `libopus-sys/opus/silk/resampler_private_down_FIR.c:138-141`
- Detail: Upstream uses `celt_assert(0)` in the `FIR_Order` switch default and then returns (assert-gated behavior). Rust uses `unreachable!()` for the same branch, producing an unconditional hard panic if reached.
