# Upstream Parity Diff Review

## Scope
Rust sources compared against upstream C in `libopus-sys/opus`.

## Remaining Items (Grouped)
Snapshot from the current findings list (open items only; stale/resolved entries removed in this refresh).

Resolved/removed in this refresh (now implemented in Rust): `1,3,4,5,6,10,11,14,15,16,17,18,21,22,23,24,25,26,27,28,29,30,31,32,33,34,42,52,64,65,88,91,105,111,112,150,166,175,214,224,229,236`.

Priority groups for execution:

1. QEXT correctness blockers (bitstream/PLC/sizing)
IDs: `none (resolved)`

2. Extensions and repacketizer semantic parity
IDs: `35,36,37,38,39,40,41,97,98,99,100,115,139`

3. Public API surface parity (core, custom, multistream/projection, 24-bit)
IDs: `12,43,45,98,104,110,116,119,120,122,163,173,177,178,186,199`

4. DNN/DRED/OSCE model loading and constant parity
IDs: `12,45,76,94,135,136,137,175,176,177,178,179,180,181,182,187,191,192,193,194,201,206,210,211,216,217,219,220,235`

5. SIMD/arch-dispatch/build-flag parity
IDs: `107,194,202,203,204,205,212,213,230,231,232,233,234,235,237`

6. Documentation/version/metadata drift
IDs: `95,96,108,131,133,134,217,223,228`

7. Runtime semantics/assert-vs-status cleanup (non-blocking but broad)
IDs (representative): `61,62,66,67,68,72,79,82,87,93,94,106,135,136,137,139,140,141,142,143,144,145,146,148,149,153,156,165,168,170,171,172,174`

## Findings

2. [HIGH] Missing restricted application mode parity in encoder.
- Rust: `src/opus/opus_encoder.rs:162`, `src/opus/opus_encoder.rs:1878`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:219`, `libopus-sys/opus/src/opus_encoder.c:1467`, `libopus-sys/opus/src/opus_encoder.c:1470`
- Detail: Rust path does not mirror `OPUS_APPLICATION_RESTRICTED_SILK` and `OPUS_APPLICATION_RESTRICTED_CELT` control-flow branches present upstream.

6. [RESOLVED][QEXT] QEXT payload is now wired in the main encoder CELT call path.
- Rust: `src/opus/opus_encoder.rs`
- Upstream refs: `libopus-sys/opus/src/opus_encoder.c:1746`, `libopus-sys/opus/src/opus_encoder.c:1792`, `libopus-sys/opus/src/opus_encoder.c:2394`
- Detail: Rust now computes QEXT allocation, passes `qext_payload`/`qext_bytes` into `celt_encode_with_ec`, and emits extension ID `124` in packet padding.

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

12. [MEDIUM][DRED] Missing decoder DRED API entry points.
- Rust: `src/opus/opus_decoder.rs`
- Upstream: `libopus-sys/opus/src/opus_decoder.c:1609`, `libopus-sys/opus/src/opus_decoder.c:1643`, `libopus-sys/opus/src/opus_decoder.c:1677`
- Detail: Upstream exports `opus_decoder_dred_decode`, `opus_decoder_dred_decode24`, and `opus_decoder_dred_decode_float`; matching public Rust entry points were not found.

13. [LOW] C-style `opus_encoder_ctl` / `opus_decoder_ctl` API entry points not mirrored as direct functions.
- Rust: `src/opus/opus_encoder.rs`, `src/opus/opus_decoder.rs`
- Upstream: `libopus-sys/opus/src/opus_encoder.c` (`opus_encoder_ctl`), `libopus-sys/opus/src/opus_decoder.c:1031` (`opus_decoder_ctl`)
- Detail: Rust provides typed setters/getters on structs, but direct C-style variadic CTL function parity is not present.

19. [MEDIUM] Multistream/projection API surface parity missing.
- Rust: `src/` (no `opus_multistream_*` / `opus_projection_*` implementation files found)
- Upstream: `libopus-sys/opus/src/opus_multistream_encoder.c`, `libopus-sys/opus/src/opus_multistream_decoder.c`, `libopus-sys/opus/src/opus_projection_encoder.c`, `libopus-sys/opus/src/opus_projection_decoder.c`
- Detail: Upstream includes full multistream/projection encoder/decoder APIs (including 24-bit variants and ctl paths); matching Rust API surface is not present.

20. [LOW] Multistream packet pad/unpad helpers parity missing.
- Rust: `src/opus/repacketizer.rs` (single-stream repacketizer only; notes no multistream manipulation)
- Upstream: `libopus-sys/opus/src/repacketizer.c` (`opus_multistream_packet_pad`, `opus_multistream_packet_unpad`)
- Detail: Upstream multistream packet pad/unpad helpers are not mirrored as Rust public functions.

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

51. [HIGH][Bitexact] CELT float math path used `FLOAT_APPROX` polynomial implementations unconditionally.
- Rust (before fix): `src/celt/mathops.rs:celt_log2`, `src/celt/mathops.rs:celt_exp2`
- Upstream: `libopus-sys/opus/celt/mathops.h:346-351` (default non-`FLOAT_APPROX` uses libc `log/exp`)
- Detail: Rust always used the Remez polynomial path for `celt_log2/celt_exp2`, while upstream C in this build uses the non-`FLOAT_APPROX` path (`1.442695...*log(x)` and `exp(0.693147...*x)`). This produced systematic CELT encode drift.
- Validation: after switching Rust to the upstream default float path, `opus_newvectors` improved from `109/228` to `216/228` (remaining failures are all `ENC @ 010kbps`, SILK-mode dominated).
- Detail: Upstream only runs `run_analysis(...)` when `application != OPUS_APPLICATION_RESTRICTED_SILK`. Rust gating currently checks only complexity/Fs and can run analysis in restricted-SILK mode, diverging from upstream mode behavior.

51. [HIGH] `frame_size_select` omits application-dependent restricted-SILK minimum-frame rule.
- Rust: `src/opus/opus_encoder.rs:942-974`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:827-851` (notably `:849-850`)
- Detail: Upstream rejects frame sizes below 10 ms for `OPUS_APPLICATION_RESTRICTED_SILK` (`new_size < Fs/100`). Rust `frame_size_select` has no `application` parameter and therefore cannot enforce this rule, allowing frame-size selections upstream would reject.

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

89. [LOW][Projection/Multistream] `mapping_matrix` module/API is missing.
- Rust: `src/opus/` (no `mapping_matrix.rs` equivalent found)
- Upstream: `libopus-sys/opus/src/mapping_matrix.c`
- Detail: Upstream includes mapping-matrix utilities and static FOA/SOA/TOA HOA matrices used by projection/multistream components. Rust codebase does not mirror this module, consistent with missing projection/multistream API coverage.

90. [MEDIUM][Analysis] Float downmix path clamps samples and replaces NaNs earlier than upstream.
- Rust: `src/opus/analysis.rs:78-83`
- Upstream: `libopus-sys/opus/src/analysis.c:561-567`
- Detail: Rust `DownmixInput::Float::downmix` clamps to `[-65536, 65536]` and converts NaN to `0` before analysis FFT. Upstream does not clamp there; it checks for NaN after FFT output and bails from analysis if detected. This alters analysis feature generation under extreme/NaN inputs.

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

166. [HIGH][QEXT][SILK Resampler] `silk_resampler_init` lacks upstream 96 kHz delay-table/rate-index coverage.
- Rust: `src/silk/resampler/mod.rs:53-67`, `src/silk/resampler/mod.rs:70-78`, `src/silk/resampler/mod.rs:121-143`
- Upstream: `libopus-sys/opus/silk/resampler.c:53-68`, `libopus-sys/opus/silk/resampler.c:70-72`, `libopus-sys/opus/silk/resampler.c:92-114`
- Detail: Upstream includes QEXT-aware 96 kHz support in `delay_matrix_enc`/`delay_matrix_dec` and `rateID()` mapping (with `ENABLE_QEXT` validation arms). Rust tables/indexing only cover up to 48 kHz, so QEXT 96 kHz init paths cannot be represented and fall into panic paths.

167. [MEDIUM][API Semantics][SILK Resampler] `silk_resampler_init` return surface is non-equivalent to upstream status-return API.
- Rust: `src/silk/resampler/mod.rs:120` (returns `ResamplerState` directly)
- Upstream: `libopus-sys/opus/silk/resampler.c:79-84`, `libopus-sys/opus/silk/resampler.c:99-101`, `libopus-sys/opus/silk/resampler.c:110-112`, `libopus-sys/opus/silk/resampler.c:178`
- Detail: Upstream `silk_resampler_init(...)` returns `opus_int` and uses `-1` failure status on invalid tuples/ratios. Rust always returns a state object and uses panic paths for invalid inputs, removing upstream error-code semantics from this API boundary.

168. [LOW][Runtime Semantics][SILK Resampler] Unsupported downsampling ratio handling is panic-based rather than assert+status.
- Rust: `src/silk/resampler/mod.rs:207-209`
- Upstream: `libopus-sys/opus/silk/resampler.c:161-165`
- Detail: Upstream handles unreachable ratio combinations with `celt_assert(0)` and `return -1`. Rust uses `unreachable!(...)`, which unconditionally panics instead of preserving upstream status-return flow.

169. [LOW][Validation Semantics][SILK] `silk_A2NLSF` omits upstream non-negative root assertion.
- Rust: `src/silk/A2NLSF.rs:147-149` (root assignment without explicit non-negative assert)
- Upstream: `libopus-sys/opus/silk/A2NLSF.c:214-217`
- Detail: Upstream checks `silk_assert( NLSF[root_ix] >= 0 )` immediately after each root interpolation. Rust computes/stores `NLSF[root_ix]` without the matching assertion, so this specific invariant is not mirrored.

170. [LOW][Runtime Semantics][SILK] `silk_sum_sqr_shift` uses `debug_assert!` where upstream keeps assert-gated post-loop invariants.
- Rust: `src/silk/sum_sqr_shift.rs:25`
- Upstream: `libopus-sys/opus/silk/sum_sqr_shift.c:61`, `libopus-sys/opus/silk/sum_sqr_shift.c:77`
- Detail: Upstream checks `nrg >= 0` after both accumulation passes with `silk_assert`. Rust only checks this condition inside helper `silk_sum_sqr_shift_inner` via `debug_assert!`, so release builds drop the invariant checks and there is no explicit second-pass postcondition check mirroring upstream placement.

171. [LOW][Runtime Semantics][SILK] `silk_LPC_inverse_pred_gain_c` keeps several upstream invariant checks only as `debug_assert!`.
- Rust: `src/silk/LPC_inv_pred_gain.rs:44-45`, `src/silk/LPC_inv_pred_gain.rs:50-51`, `src/silk/LPC_inv_pred_gain.rs:102-103`
- Upstream: `libopus-sys/opus/silk/LPC_inv_pred_gain.c:62-63`, `libopus-sys/opus/silk/LPC_inv_pred_gain.c:68-69`, `libopus-sys/opus/silk/LPC_inv_pred_gain.c:112-113`
- Detail: Upstream enforces `rc_mult1_Q30` and `invGain_Q30` bounds with `silk_assert` at each stage. Rust downgrades these to `debug_assert!`, so release builds drop the checks and no longer mirror upstream invariant enforcement points.

172. [LOW][Runtime Semantics][CELT Mathops] `celt_atan2p_norm` precondition check is `debug_assert!` in Rust.
- Rust: `src/celt/mathops.rs:92-94`
- Upstream: `libopus-sys/opus/celt/mathops.h:170-173`
- Detail: Upstream guards non-negative-domain inputs with `celt_sig_assert(x>=0 && y>=0)`. Rust uses `debug_assert!`, so release builds drop this invariant check and no longer mirror upstream assert-gated behavior at that call boundary.

173. [LOW][API Surface][SILK Decoder Init] `silk_init_decoder`/`silk_reset_decoder` do not mirror upstream in-place status-return API.
- Rust: `src/silk/init_decoder.rs:14`, `src/silk/init_decoder.rs:58` (reset is `fn(...){...}`; init constructs/returns `silk_decoder_state`)
- Upstream: `libopus-sys/opus/silk/init_decoder.c:43-45`, `libopus-sys/opus/silk/init_decoder.c:73-75`, `libopus-sys/opus/silk/init_decoder.c:66-67`, `libopus-sys/opus/silk/init_decoder.c:82`
- Detail: Upstream initializes/reset an existing decoder struct and returns `opus_int` status (`0` on success). Rust exposes a constructor-style init returning the state directly and a void reset path, so the initialization API contract differs from upstream C.

174. [LOW][Runtime Semantics][SILK+OSCE] Decoder reset initializes OSCE method to `NONE` rather than upstream `OSCE_DEFAULT_METHOD`.
- Rust: `src/silk/init_decoder.rs:50`
- Upstream: `libopus-sys/opus/silk/init_decoder.c:61-64`, `libopus-sys/opus/dnn/osce.h:59-67`
- Detail: With OSCE enabled, upstream `silk_reset_decoder` calls `osce_reset(..., OSCE_DEFAULT_METHOD)` (typically NOLACE/LACE when compiled). Rust reset uses `OSCE_METHOD_NONE`, changing default post-reset OSCE state semantics.

175. [HIGH][QEXT][DRED] DRED 16 kHz conversion helper omits upstream 96 kHz handling.
- Rust: `src/dnn/dred/encoder.rs:154-161`
- Upstream: `libopus-sys/opus/dnn/dred_encoder.c:143-163`
- Detail: Upstream includes `case 96000: up = 1` under `ENABLE_QEXT` in `dred_convert_to_16k`. Rust switch has no `96000` arm and falls into panic on that rate, so QEXT-enabled 96 kHz DRED conversion is not mirrored.

176. [MEDIUM][Initialization Semantics][DRED] `dred_encoder_init` does not mirror upstream built-in model auto-load behavior.
- Rust: `src/dnn/dred/encoder.rs:110-115`
- Upstream: `libopus-sys/opus/dnn/dred_encoder.c:79-87`
- Detail: Upstream `dred_encoder_init` sets `loaded=1` when built-in RDOVAE weights are compiled (`!USE_WEIGHTS_FILE`) before reset. Rust `DREDEnc::init` unconditionally sets `loaded=false` and resets without an equivalent built-in auto-load path at this API boundary.

177. [LOW][API Coverage][DRED] Upstream encoder deinitialization helper is not mirrored as a dedicated Rust API.
- Rust: `src/dnn/dred/encoder.rs` (no `dred_deinit_encoder` equivalent found)
- Upstream: `libopus-sys/opus/dnn/dred_encoder.h:65`
- Detail: Upstream declares `dred_deinit_encoder(DREDEnc*)` as part of the DRED encoder lifecycle surface. Rust relies on ownership/drop and does not expose a direct deinit helper, so lifecycle API parity is incomplete.

178. [LOW][API Coverage][MLP] Upstream analysis-layer helper entry points are not mirrored as direct Rust APIs.
- Rust: `src/opus/mlp/mod.rs:13-15`, `src/opus/mlp/analysis_mlp.rs:267-276` (only high-level `run_analysis_mlp` exposed)
- Upstream: `libopus-sys/opus/src/mlp.h:56-58`, `libopus-sys/opus/src/mlp.c:70-131`
- Detail: Upstream provides callable helper functions `analysis_compute_dense(...)` and `analysis_compute_gru(...)`. Rust implements equivalent logic inside layer methods (`src/opus/mlp/layers.rs`) but does not expose matching standalone entry points, so API-level parity for these MLP helpers is incomplete.

179. [MEDIUM][DNN API Coverage][LPCNet] Standalone LPCNet decoder/synthesis API surface from upstream headers is not mirrored.
- Rust: `src/dnn/lpcnet.rs` (provides `LPCNetEncState`, `LPCNetPLCState`, feature/PLC helpers; no `LPCNetDecState`/`LPCNetState` create/decode/synthesize API)
- Upstream: `libopus-sys/opus/dnn/lpcnet.h:40-79`, `libopus-sys/opus/dnn/lpcnet.h:134-166`
- Detail: Upstream declares standalone LPCNet decoder/synthesis lifecycle and processing entry points (`lpcnet_decoder_*`, `lpcnet_*`, `lpcnet_synthesize`, `lpcnet_decode`). Rust currently exposes encoder-analysis and PLC components but not the equivalent decoder/synth API set.

180. [LOW][DNN API Surface][LPCNet] Direct blob-based LPCNet load helpers are not mirrored as one-shot APIs.
- Rust: `src/dnn/lpcnet.rs:215-216`, `src/dnn/lpcnet.rs:571-580`, plus parser helper `src/dnn/weights.rs:37-38`
- Upstream: `libopus-sys/opus/dnn/lpcnet.h:97`, `libopus-sys/opus/dnn/lpcnet.h:180-181`
- Detail: Upstream exposes direct blob loaders (`lpcnet_encoder_load_model`, `lpcnet_load_model`, `lpcnet_plc_load_model`) taking `(const void*, len)`. Rust exposes `load_model(&[WeightArray])` methods and a separate weight parser, but no equivalent one-call LPCNet blob-loading entry points.

181. [MEDIUM][Constants][DRED] Experimental DRED version constant differs from upstream.
- Rust: `src/dnn/dred/config.rs:8`
- Upstream: `libopus-sys/opus/dnn/dred_config.h:35`
- Detail: Rust sets `DRED_EXPERIMENTAL_VERSION = 10`, while upstream 1.6.1 defines `DRED_EXPERIMENTAL_VERSION 12`. This can alter extension compatibility/version-tag behavior for experimental DRED packet signaling.

182. [HIGH][Model Constants][DRED] RDOVAE dimension/stat-table constants diverge from upstream generated headers.
- Rust: `src/dnn/dred/config.rs:28-31`, `src/dnn/dred/stats.rs:6`, `src/dnn/dred/stats.rs:84`
- Upstream: `libopus-sys/opus/dnn/dred_rdovae_constants.h:12-19`, `libopus-sys/opus/dnn/dred_rdovae_stats_data.h:15-24`
- Detail: Upstream defines `DRED_LATENT_DIM=25`, `DRED_STATE_DIM=50`, `DRED_PADDED_LATENT_DIM=32`, `DRED_PADDED_STATE_DIM=56`, with corresponding stats table sizes `400` and `800`. Rust currently uses `21/19/24/24` and stats arrays sized `336/304`, indicating a different model-constant/data layout than upstream 1.6.1.

183. [HIGH][Data Layout][DRED Decoder] `OpusDRED.latents` layout omits upstream per-latent extra slot (`DRED_LATENT_DIM+1`).
- Rust: `src/dnn/dred/decoder.rs:38`, `src/dnn/dred/decoder.rs:179`, `src/dnn/dred/decoder.rs:200-206`
- Upstream: `libopus-sys/opus/dnn/dred_decoder.h:40`, `libopus-sys/opus/dnn/dred_decoder.c:117`, `libopus-sys/opus/dnn/dred_decoder.c:123`
- Detail: Upstream stores latents as `(DRED_NUM_REDUNDANCY_FRAMES/2)*(DRED_LATENT_DIM+1)` and writes an extra trailing value per latent (`q_level*.125 - 1`). Rust allocates/stores only `DRED_LATENT_DIM` values per latent frame, so decoded DRED latent layout differs from upstream.

184. [LOW][Runtime Semantics][LPCNet PLC] FEC queue-full behavior differs from upstream assert-gated contract.
- Rust: `src/dnn/lpcnet.rs:597-607`
- Upstream: `libopus-sys/opus/dnn/lpcnet_plc.c:96-99`
- Detail: Upstream `lpcnet_plc_fec_add` asserts `fec_fill_pos < PLC_MAX_FEC` and then appends. Rust instead handles `fec_fill_pos == PLC_MAX_FEC` by shifting buffered entries and continuing insertion. This changes overflow/invariant behavior compared with upstream's assert-based contract.

185. [LOW][API Signature][DNN NNet] Generic DNN compute helpers omit upstream `arch` argument.
- Rust: `src/dnn/nnet.rs:197`, `src/dnn/nnet.rs:214`, `src/dnn/nnet.rs:253`, `src/dnn/nnet.rs:272`, `src/dnn/nnet.rs:299`
- Upstream: `libopus-sys/opus/dnn/nnet.h:89-93`, `libopus-sys/opus/dnn/nnet.c:60`, `libopus-sys/opus/dnn/nnet.c:76`, `libopus-sys/opus/dnn/nnet.c:107`, `libopus-sys/opus/dnn/nnet.c:124`, `libopus-sys/opus/dnn/nnet.c:136`
- Detail: Upstream signatures carry an explicit `arch` parameter through generic dense/GRU/GLU/conv helpers (matching runtime dispatch conventions). Rust equivalents expose no `arch` parameter, so function-level API surface and call-shape are not source-equivalent.

186. [LOW][API Coverage][DNN NNDSP] Explicit adaptive-filter state init helpers from upstream headers are not mirrored as direct functions.
- Rust: `src/dnn/nndsp.rs:30`, `src/dnn/nndsp.rs:54`, `src/dnn/nndsp.rs:74` (`Default` impls for `AdaConvState`/`AdaCombState`/`AdaShapeState`)
- Upstream: `libopus-sys/opus/dnn/nndsp.h:80-84`
- Detail: Upstream exports `init_adaconv_state`, `init_adacomb_state`, and `init_adashape_state`. Rust initializes these states via struct defaults but does not expose matching named init helpers, leaving API-level parity incomplete.

187. [LOW][API Signature][LPCNet] Single-frame feature extraction helpers omit upstream `arch` argument.
- Rust: `src/dnn/lpcnet.rs:435`, `src/dnn/lpcnet.rs:454`
- Upstream: `libopus-sys/opus/dnn/lpcnet.h:123`, `libopus-sys/opus/dnn/lpcnet.h:132`
- Detail: Upstream `lpcnet_compute_single_frame_features*` signatures include an explicit `arch` parameter. Rust helpers expose no `arch` input, so API call shape differs and does not mirror upstream arch-parameterized entry points.

188. [LOW][Tooling Coverage][Compare] Rust compare utility does not mirror upstream QEXT compare tool path (`qext_compare`).
- Rust: `src/tools/compare.rs:100-106`, `src/tools/demo/input.rs:41-47` (band table/sample-rate model tops out at 48 kHz)
- Upstream: `libopus-sys/opus/src/qext_compare.c:60-62`, `libopus-sys/opus/src/qext_compare.c:308`
- Detail: Upstream includes a dedicated 96 kHz/QEXT-aware compare utility with different band/frequency/window configuration (`NBANDS=28`, `NFREQS=480`, `TEST_WIN_SIZE=960`) and explicit 96 kHz handling. Rust currently exposes a single `opus_compare`-style path for up to 48 kHz.

189. [LOW][Tooling Coverage][Compare] DRED-specific compare utility from upstream is not mirrored.
- Rust: `src/tools/mod.rs:5-8` (exports `demo` and `opus_compare` utilities)
- Upstream: `libopus-sys/opus/dnn/dred_compare.c`
- Detail: Upstream provides `dred_compare` for DRED-focused quality evaluation workflows. Rust tools module has no dedicated equivalent utility surface for this comparison path.

190. [MEDIUM][Runtime Semantics][DNN Vec x86] x86 non-AVX2 quantization/bias path diverges from upstream `USE_SU_BIAS` behavior.
- Rust: `src/dnn/simd/mod.rs:100`, `src/dnn/simd/mod.rs:110`, `src/dnn/simd/mod.rs:135`, `src/dnn/simd/mod.rs:145`, `src/dnn/simd/mod.rs:302`
- Rust fallback kernels: `src/dnn/vec.rs:202`, `src/dnn/vec.rs:216`
- Upstream: `libopus-sys/opus/dnn/vec.h:38-39`, `libopus-sys/opus/dnn/vec_avx.h:41`, `libopus-sys/opus/dnn/vec.h:187`, `libopus-sys/opus/dnn/vec.h:221`
- Detail: Upstream x86/SSE2 builds include `vec_avx.h` and use `USE_SU_BIAS` unsigned-input quantization (`127+round(127*x)`) for int8 GEMV paths. Rust only enables SU-bias semantics when AVX2 is detected and otherwise falls back to signed scalar GEMV, so x86 non-AVX2 behavior is not upstream-equivalent.

191. [LOW][API Signature][OSCE] `osce_load_models` does not mirror upstream blob-loading entry point.
- Rust: `src/dnn/osce.rs:1597`
- Upstream: `libopus-sys/opus/dnn/osce.h:89`
- Detail: Upstream signature is `osce_load_models(OSCEModel*, const void *data, int len)`. Rust exposes `osce_load_models(&mut OSCEModel, &[WeightArray]) -> bool`, requiring pre-parsed arrays and omitting the direct `(blob,len)` API shape.

192. [LOW][API Signature][OSCE] `osce_enhance_frame`/`osce_bwe` omit upstream `arch` parameter.
- Rust: `src/dnn/osce.rs:2527`, `src/dnn/osce.rs:3339`
- Upstream: `libopus-sys/opus/dnn/osce.h:79`, `libopus-sys/opus/dnn/osce.h:93`
- Detail: Upstream OSCE processing entry points include explicit run-time `arch` dispatch arguments. Rust equivalents do not take `arch`, so function-level call signatures are not source-equivalent.

193. [LOW][API Signature][OSCE] `osce_bwe_reset` signature differs by requiring explicit feature-state argument.
- Rust: `src/dnn/osce.rs:3299`
- Upstream: `libopus-sys/opus/dnn/osce.h:102`, `libopus-sys/opus/dnn/osce.c:1410`
- Detail: Upstream reset API takes only `silk_OSCE_BWE_struct*`. Rust requires both `OSCEBWEState` and `OSCEBWEFeatureState`, changing reset-call contract and API parity.

194. [LOW][Arch Dispatch Coverage][DNN] Upstream RTCD x86/ARM specialized NNet kernels are not mirrored as architecture-specific backends.
- Rust: `src/dnn/` (no `src/dnn/x86/*` or `src/dnn/arm/*` backend modules for `compute_linear/activation/conv2d`)
- Upstream: `libopus-sys/opus/dnn/x86/dnn_x86.h`, `libopus-sys/opus/dnn/x86/x86_dnn_map.c`, `libopus-sys/opus/dnn/x86/nnet_sse2.c`, `libopus-sys/opus/dnn/x86/nnet_sse4_1.c`, `libopus-sys/opus/dnn/x86/nnet_avx2.c`, `libopus-sys/opus/dnn/arm/nnet_dotprod.c`, `libopus-sys/opus/dnn/arm/nnet_neon.c`, `libopus-sys/opus/dnn/arm/arm_dnn_map.c`
- Detail: Upstream provides RTCD-selected architecture-specific implementations for core NNet ops beyond generic C. Rust currently provides generic kernels plus a vec-level SIMD layer, but no equivalent x86/ARM NNet backend surface.

195. [MEDIUM][Validation Semantics][DNN Weights] `linear_init` omits upstream sparse-index structural validation.
- Rust: `src/dnn/nnet.rs:501-517`
- Upstream: `libopus-sys/opus/dnn/parse_lpcnet_weights.c:99-120`, `libopus-sys/opus/dnn/parse_lpcnet_weights.c:151`
- Detail: Upstream `find_idx_check` validates sparse index stream shape (`remain < nb_blocks+1`), 4-column alignment (`pos&0x3`), and bounds (`pos+3 < nb_in`) before accepting `weights_idx`. Rust only counts blocks and output groups, then accepts the index array without these checks, so malformed sparse index blobs that upstream rejects can be accepted.

196. [LOW][Initialization Semantics][DNN Weights] Optional float-weight size mismatches are silently ignored instead of treated as init errors.
- Rust: `src/dnn/nnet.rs:523-526`, `src/dnn/nnet.rs:534-538`, `src/dnn/nnet.rs:578-582`
- Upstream: `libopus-sys/opus/dnn/parse_lpcnet_weights.c:92`, `libopus-sys/opus/dnn/parse_lpcnet_weights.c:155-157`, `libopus-sys/opus/dnn/parse_lpcnet_weights.c:163-165`, `libopus-sys/opus/dnn/parse_lpcnet_weights.c:193-195`
- Detail: Upstream uses `opt_array_check(..., &err)` and fails initialization when an optional named float array exists with wrong size. Rust uses `if let Some(...)` and simply leaves fields empty when size check fails, changing error behavior and potentially masking corrupted/incompatible blobs.

197. [LOW][Validation Semantics][DNN Weights] `parse_weights` accepts zero-sized records that upstream rejects.
- Rust: `src/dnn/nnet.rs:629-635`
- Upstream: `libopus-sys/opus/dnn/parse_lpcnet_weights.c:52`, `libopus-sys/opus/dnn/parse_lpcnet_weights.c:64`
- Detail: Upstream `parse_record` returns `array->size` and `parse_weights` only accepts `ret > 0`; records with `size == 0` are treated as parse failure. Rust currently pushes arrays regardless of `size` (including zero), so malformed/empty records can be accepted where upstream rejects the blob.

198. [LOW][Validation Semantics][DNN Weights] Record-name termination check is weaker than upstream.
- Rust: `src/dnn/nnet.rs:620-623`
- Upstream: `libopus-sys/opus/dnn/parse_lpcnet_weights.c:43`
- Detail: Upstream requires `h->name[43] == 0` (fixed-field null termination) and rejects records otherwise. Rust scans for the first NUL and allows fully non-terminated 44-byte names (`unwrap_or(44)`), so some headers rejected upstream are accepted in Rust.

199. [LOW][API Coverage][DNN NNet] `compute_gated_activation` helper from upstream header is not mirrored.
- Rust: `src/dnn/nnet.rs` (no `compute_gated_activation` function)
- Upstream: `libopus-sys/opus/dnn/nnet.h:94`
- Detail: Upstream exposes `compute_gated_activation(const LinearLayer*, float*, const float*, int activation, int arch)` as part of the generic NNet API surface. Rust currently exposes `compute_glu` and other helpers but no equivalent gated-activation entry point.

200. [LOW][API Signature][DNN Weights] `parse_weights` return contract differs from upstream C API.
- Rust: `src/dnn/nnet.rs:591`
- Upstream: `libopus-sys/opus/dnn/nnet.h:97`, `libopus-sys/opus/dnn/parse_lpcnet_weights.c:55-79`
- Detail: Upstream `parse_weights(WeightArray **list, const void *data, int len)` allocates a null-terminated C array and returns the array count (or `-1`). Rust returns `Option<Vec<WeightArray>>` with copied payloads and no C-style sentinel/list ownership contract, so parser API shape and memory/lifecycle semantics are not source-equivalent.

201. [MEDIUM][Builtin Weights Coverage][OSCE] `compiled_weights()` omits BBWENet arrays needed for OSCE-BWE model loading.
- Rust: `src/dnn/weights.rs:15-30`
- Rust BBWE arrays exist but are not included: `src/dnn/bbwenet_data.rs:112975`
- Rust OSCE loader consumes BBWENet weights if present: `src/dnn/osce.rs:1597-1601`
- Upstream builtin path initializes BBWENet arrays alongside LACE/NoLACE: `libopus-sys/opus/dnn/osce.c:1465-1468`
- Detail: Upstream built-in OSCE model loading includes `bbwenetlayers_arrays` when OSCE-BWE is enabled. Rust `compiled_weights()` includes `lacelayers` and `nolacelayers` but not `bbwenetlayers`, so one-shot built-in weight aggregation is not upstream-equivalent for OSCE-BWE coverage.

202. [LOW][Arch Dispatch Coverage][SILK VAD] Full-function SSE4.1 VAD RTCD path is not mirrored.
- Rust: `src/silk/VAD.rs:60`, `src/silk/VAD.rs:154`, `src/silk/simd/mod.rs:101-113`
- Upstream: `libopus-sys/opus/silk/x86/x86_silk_map.c:61-69`
- Detail: Upstream x86 RTCD dispatches `silk_VAD_GetSA_Q8` to an SSE4.1 implementation at higher arch levels. Rust keeps `silk_VAD_GetSA_Q8_c` as the top-level path and only SIMD-accelerates the inner energy accumulator helper, so full-function RTCD parity for VAD is incomplete.

203. [LOW][Arch Dispatch Coverage][SILK NSQ] Full-function SSE4.1 NSQ RTCD replacement is not mirrored.
- Rust: `src/silk/NSQ.rs:170-176`, `src/silk/NSQ.rs:244-270`, `src/silk/simd/mod.rs:226-233`, `src/silk/simd/mod.rs:241`
- Upstream: `libopus-sys/opus/silk/x86/x86_silk_map.c:72-92`
- Detail: Upstream x86 RTCD can dispatch the entire `silk_NSQ` function to SSE4.1. Rust uses scalar `silk_NSQ_c` as the top-level flow and conditionally swaps only the inner `silk_noise_shape_quantizer_10_16` kernel, so function-level RTCD parity is partial.

204. [MEDIUM][Arch Dispatch Semantics][CELT x86] `celt_pitch_xcorr` uses SSE fallback in Rust where upstream keeps scalar for non-AVX2.
- Rust: `src/celt/simd/mod.rs:113-126`
- Rust SSE implementation exists: `src/celt/simd/x86.rs:538`
- Upstream: `libopus-sys/opus/celt/x86/x86_celt_map.c:95-108`
- Detail: Upstream x86 RTCD table dispatches `celt_pitch_xcorr` to AVX2 only; non-AVX2 arch levels stay on `celt_pitch_xcorr_c`. Rust dispatch adds an SSE path before scalar fallback, changing architecture-dependent execution path (and potentially floating-point accumulation behavior) relative to upstream.

205. [MEDIUM][Arch Dispatch Semantics][CELT VQ] `op_pvq_search` ignores caller-provided `arch` level in Rust SIMD path.
- Rust: `src/celt/simd/mod.rs:170-175`, `src/celt/simd/mod.rs:180`
- Rust call path still passes `arch`: `src/celt/vq.rs:25-26`, `src/celt/vq.rs:554`
- Upstream: `libopus-sys/opus/celt/vq.h:60-64`, `libopus-sys/opus/celt/x86/x86_celt_map.c:175-182`
- Detail: Upstream dispatch can be controlled via the `arch` argument (through `OP_PVQ_SEARCH_IMPL[(arch)&mask]`). Rust SIMD dispatch chooses SSE2 solely from host CPUID and ignores `_arch`, so callers cannot force scalar/lower-arch behavior in the same way.

206. [MEDIUM][Initialization Semantics][OSCE] Rust OSCE model loader treats partial model initialization as success.
- Rust: `src/dnn/osce.rs:1597-1602`
- Upstream: `libopus-sys/opus/dnn/osce.c:1438-1449`, `libopus-sys/opus/dnn/osce.c:1457-1468`, `libopus-sys/opus/dnn/osce.c:1473-1474`
- Detail: Upstream chains init calls with `if (ret == 0)` and returns failure (`-1`) if any enabled model init fails. Rust sets `loaded=true` when *any* of LACE/NoLACE/BBWENet init succeeds, allowing partial-load success that upstream would report as failure.

207. [LOW][API Signature][DRED RDOVAE] RDOVAE encode/decode entry points omit upstream `arch` parameter.
- Rust: `src/dnn/dred/rdovae_enc.rs:373`, `src/dnn/dred/rdovae_dec.rs:441`, `src/dnn/dred/rdovae_dec.rs:483`, `src/dnn/dred/rdovae_dec.rs:662`
- Upstream: `libopus-sys/opus/dnn/dred_rdovae_enc.h:49`, `libopus-sys/opus/dnn/dred_rdovae_dec.h:49-51`
- Detail: Upstream RDOVAE APIs (`dred_rdovae_encode_dframe`, `dred_rdovae_dec_init_states`, `dred_rdovae_decode_qframe`, `DRED_rdovae_decode_all`) all take explicit run-time `arch` arguments. Rust equivalents do not expose `arch`, so call signatures and RTCD control surface are not source-equivalent.

208. [LOW][API Signature][DRED Encoder] Encoder entry points omit upstream `arch` parameter.
- Rust: `src/dnn/dred/encoder.rs:333`, `src/dnn/dred/encoder.rs:429`
- Upstream: `libopus-sys/opus/dnn/dred_encoder.h:67`, `libopus-sys/opus/dnn/dred_encoder.h:69`
- Detail: Upstream `dred_compute_latents(...)` and `dred_encode_silk_frame(...)` include explicit run-time `arch` arguments for dispatch parity. Rust equivalents do not expose `arch`, so call signatures and arch-control surface differ.

209. [MEDIUM][Initialization/API Semantics][FARGAN] Rust state init does not mirror upstream built-in auto-load and blob loader entry points.
- Rust: `src/dnn/fargan.rs:345-370` (`FARGANState::new`, `FARGANState::init(&[WeightArray])`)
- Upstream: `libopus-sys/opus/dnn/fargan.c:174-185` (`fargan_init` builtin auto-load), `libopus-sys/opus/dnn/fargan.c:187-195` (`fargan_load_model(const void*, len)`), `libopus-sys/opus/dnn/fargan.h:59-60`
- Detail: Upstream exposes C entry points that initialize state (including builtin-model load when compiled) and a direct blob loader. Rust exposes constructor/reset plus array-based init only, without equivalent one-call blob loader or builtin auto-load behavior at init entry.

210. [LOW][API Signature][PitchDNN] `compute_pitchdnn` omits upstream `arch` parameter.
- Rust: `src/dnn/pitchdnn.rs:174-178`
- Upstream: `libopus-sys/opus/dnn/pitchdnn.h:27-31`, `libopus-sys/opus/dnn/pitchdnn.c:12-17`
- Detail: Upstream pitch estimator entry point takes explicit run-time `arch` for DNN op dispatch. Rust `compute_pitchdnn` has no `arch` argument, so function signature and arch-control surface differ.

211. [LOW][State Layout][PitchDNN] `PitchDNNState` omits upstream `xcorr_mem3` field.
- Rust: `src/dnn/pitchdnn.rs:129-134`
- Upstream: `libopus-sys/opus/dnn/pitchdnn.h:18-20`
- Detail: Upstream `PitchDNNState` includes `xcorr_mem1`, `xcorr_mem2`, and `xcorr_mem3`. Rust state currently stores only `xcorr_mem1` and `xcorr_mem2`, so struct/state-surface parity is incomplete.

212. [MEDIUM][Arch Dispatch/API Surface][DNN Core] Core DNN compute APIs omit upstream `arch` parameter and lose explicit RTCD control surface.
- Rust: `src/dnn/nnet.rs:108`, `src/dnn/nnet.rs:137`, `src/dnn/nnet.rs:197-202`, `src/dnn/nnet.rs:214-219`, `src/dnn/nnet.rs:253`, `src/dnn/nnet.rs:272-279`, `src/dnn/nnet.rs:299-307`, `src/dnn/nnet.rs:386-394`
- Upstream API signatures: `libopus-sys/opus/dnn/nnet.h:89-94`
- Upstream arch-dispatch macros: `libopus-sys/opus/dnn/nnet.h:145-154`
- Upstream RTCD tables: `libopus-sys/opus/dnn/x86/x86_dnn_map.c:39-78`, `libopus-sys/opus/dnn/arm/arm_dnn_map.c:39-82`
- Detail: Upstream threads an explicit `arch` argument through DNN compute entry points and can route through `DNN_COMPUTE_*_IMPL[(arch)&OPUS_ARCHMASK]`. Rust DNN compute functions select SIMD path from host capabilities internally and expose no `arch` parameter, so callers cannot force/mirror upstream arch-tier execution semantics.

213. [MEDIUM][Arch Dispatch Coverage][DNN x86] Rust DNN SIMD dispatch has AVX2-only acceleration on x86, missing upstream SSE2/SSE4.1 tiers.
- Rust: `src/dnn/simd/mod.rs:19-20`, `src/dnn/simd/mod.rs:44-57`, `src/dnn/simd/mod.rs:71-84`, `src/dnn/simd/mod.rs:98-110`
- Upstream: `libopus-sys/opus/dnn/x86/x86_dnn_map.c:46-48`, `libopus-sys/opus/dnn/x86/x86_dnn_map.c:59-61`, `libopus-sys/opus/dnn/x86/x86_dnn_map.c:75-77`
- Detail: Upstream RTCD supports SSE2 and SSE4.1 dispatch levels for linear/activation/conv2d before AVX2. Rust x86 DNN dispatch checks only `avx2+fma` and otherwise falls back to scalar, so non-AVX2 x86 behavior/perf tiering is not source-equivalent.

215. [LOW][Algorithmic Path][DNN Conv2D] Rust `compute_conv2d` lacks upstream 3x3 specialized convolution branch.
- Rust: `src/dnn/nnet.rs:351-381`, `src/dnn/nnet.rs:406-416`
- Upstream: `libopus-sys/opus/dnn/nnet_arch.h:230-233`
- Detail: Upstream selects `conv2d_3x3_float` when `ktime == 3 && kheight == 3`; otherwise it uses generic `conv2d_float`. Rust always executes the generic loop nest, so operation ordering/optimization path differs for the common 3x3 case.

216. [LOW][API Signature][LPCNet] LPCNet feature extraction entry points omit upstream `arch` parameter.
- Rust: `src/dnn/lpcnet.rs:277`, `src/dnn/lpcnet.rs:435-439`, `src/dnn/lpcnet.rs:454-458`
- Upstream: `libopus-sys/opus/dnn/lpcnet_private.h:76`, `libopus-sys/opus/dnn/lpcnet.h:123`, `libopus-sys/opus/dnn/lpcnet.h:132`
- Detail: Upstream `compute_frame_features`, `lpcnet_compute_single_frame_features`, and `lpcnet_compute_single_frame_features_float` all take explicit run-time `arch`. Rust equivalents do not expose `arch`, so API surface and explicit RTCD control differ.

217. [LOW][Documentation/Versioning][DNN] DNN module documentation still references Opus 1.5.2 rather than 1.6.1.
- Rust: `src/dnn/README.md:3`
- Upstream baseline being tracked in this repo: `libopus-sys/build.rs:65`, `libopus-sys/build.rs:68`
- Detail: The DNN README claims the port targets Opus 1.5.2, while the build/config baseline in-tree is 1.6.1. This is a documentation parity drift that can mislead future reviews and regeneration workflows.

218. [MEDIUM][Validation Semantics][DNN Weights] Sparse index validation in `linear_init` is weaker than upstream `find_idx_check`.
- Rust: `src/dnn/nnet.rs:501-517`
- Upstream: `libopus-sys/opus/dnn/parse_lpcnet_weights.c:99-121`, `libopus-sys/opus/dnn/parse_lpcnet_weights.c:149-152`
- Detail: Upstream validates each sparse block index (`remain < nb_blocks+1`, `pos+3 < nb_in`, and `pos` 4-aligned) before accepting `weights_idx`. Rust only counts blocks and output groups, without per-index bounds/alignment checks, so malformed sparse index blobs that upstream rejects may be accepted.

219. [LOW][Boundary Condition][MLP] GRU neuron-capacity assertion is off-by-one versus upstream fixed-size buffers.
- Rust: `src/opus/mlp/layers.rs:95`, `src/opus/mlp/layers.rs:102`, `src/opus/mlp/layers.rs:119-124`
- Upstream: `libopus-sys/opus/src/mlp.h:35`, `libopus-sys/opus/src/mlp.c:97-100`, `libopus-sys/opus/src/mlp.c:101-103`
- Detail: Upstream allocates `tmp/z/r/h` with `MAX_NEURONS` (=32) and supports `N == 32`. Rust asserts `n < MAX_NEURONS`, which rejects `n == 32` and can panic on valid upstream-size models.

220. [LOW][Numeric Fidelity][MLP] `tansig_approx` polynomial coefficients are truncated compared with upstream constants.
- Rust: `src/opus/mlp/tansig.rs:10-15`
- Upstream: `libopus-sys/opus/src/mlp.c:41-46`
- Detail: Rust uses shortened constants (e.g. `952.528`, `952.724`, `413.368`) while upstream uses higher-precision literals (`952.52801514f`, `952.72399902f`, `413.36801147f`, etc.). This can introduce small output differences in MLP activations relative to upstream.

221. [MEDIUM][Initialization Semantics][LPCNet PLC] Rust `init()` ignores encoder/FARGAN init outcomes when determining loaded state.
- Rust: `src/dnn/lpcnet.rs:552-560` (calls `self.fargan.init(arrays)` and `self.enc.load_model(arrays)` without checking return values; sets `loaded=true` if `init_plcmodel` succeeds)
- Rust checked path for comparison: `src/dnn/lpcnet.rs:571-583` (`load_model()` requires PLC model, encoder model, and FARGAN init to all succeed)
- Upstream init flow: `libopus-sys/opus/dnn/lpcnet_plc.c:58-67`, `libopus-sys/opus/dnn/lpcnet_plc.c:70`
- Detail: Upstream init path relies on builtin-model initialization assertions and marks loaded only on successful model init. Rust `init()` can mark `loaded=true` even if encoder/FARGAN model init fails, which diverges from the stricter all-components-ready contract already enforced in Rust `load_model()`.

222. [MEDIUM][API Behavior/Version Reporting][Public API] `opus_get_version_string()` does not mirror upstream version-format and build-suffix semantics.
- Rust: `src/celt/common.rs:426-427`
- Upstream implementation: `libopus-sys/opus/celt/celt.c:360-372`
- Upstream API contract notes: `libopus-sys/opus/include/opus_defines.h:852-861`
- Detail: Upstream returns `"libopus " PACKAGE_VERSION` and conditionally appends `-fixed` / `-fuzzing`, and documentation explicitly allows apps to detect fixed-point builds from this string. Rust returns a hardcoded `"opurs (rust port) 1.5.2"`, which drops upstream format, version baseline, and build-suffix signaling semantics.

223. [LOW][Documentation/Versioning][Top-Level Docs] Top-level crate and `libopus-sys` README still claim Opus 1.5.2.
- Rust: `src/lib.rs:1`, `libopus-sys/README.md:1`
- Upstream baseline being tracked in-tree: `libopus-sys/build.rs:65`, `libopus-sys/build.rs:68`
- Detail: User-facing docs advertise 1.5.2 while the vendored/build baseline declares 1.6.1. This is documentation/versioning drift that can mislead reviewers and downstream users about compatibility targets.

225. [LOW][Build Config Defaults][libopus-sys DNN] `DISABLE_DEBUG_FLOAT` default from upstream build systems is not mirrored in generated config.
- Rust build config: `libopus-sys/build.rs:45-84`, `libopus-sys/build.rs:277-322`
- Upstream default option and define path: `libopus-sys/opus/meson_options.txt:13`, `libopus-sys/opus/meson.build:191-193`
- Affected generated weights guards: `libopus-sys/opus/dnn/bbwenet_data.c:11673-11678`
- Detail: Upstream defaults `dnn-debug-float` to disabled and defines `DISABLE_DEBUG_FLOAT`. `build.rs` never emits that macro, so debug-float guarded weight sections remain enabled when compiling C DNN sources, diverging from upstream default build footprint/config semantics.

226. [LOW][Build Feature Parity][libopus-sys/QEXT] `libopus-sys` does not expose or map upstream `ENABLE_QEXT` build feature.
- Rust features: `Cargo.toml:117`, `Cargo.toml:119-120`, `libopus-sys/Cargo.toml:13-22`
- Rust build config path: `libopus-sys/build.rs:112-115`, `libopus-sys/build.rs:277-322`
- Upstream feature define: `libopus-sys/opus/configure.ac:158-165`
- Detail: Upstream has an explicit `--enable-qext` build switch that defines `ENABLE_QEXT`. The workspace has a Rust `qext` feature, but `libopus-sys` has no corresponding feature and `build.rs` never defines `ENABLE_QEXT`, so C-side tool/comparison builds cannot be configured to mirror upstream QEXT-enabled configuration.

227. [LOW][API Behavior][Public API] `opus_strerror()` returns non-upstream message strings.
- Rust: `src/celt/common.rs:409-418`
- Upstream: `libopus-sys/opus/celt/celt.c:344-353`
- Detail: Upstream returns canonical short strings (e.g. `"success"`, `"invalid argument"`), while Rust appends numeric suffixes (e.g. `"success (0)"`). Applications/tests comparing exact libopus error text will observe divergence.

228. [LOW][Documentation/Metadata][Package] Crate package metadata still states bit-exactness against 1.5.2.
- Rust: `Cargo.toml:8`
- Upstream baseline being tracked in-tree: `libopus-sys/build.rs:65`, `libopus-sys/build.rs:68`
- Detail: Published package description advertises 1.5.2 despite in-repo baseline updates to 1.6.1, creating metadata drift for downstream users evaluating compatibility.

229. [HIGH][Feature Behavior][QEXT Encoder] Rust Opus encoder never provisions QEXT extension payload bytes, effectively disabling upstream QEXT bitstream path.
- Rust caller path: `src/opus/opus_encoder.rs:2810-2820`
- Rust CELT entry shape: `src/celt/celt_encoder.rs:2119-2127`, `src/celt/celt_encoder.rs:3283`
- Upstream behavior: `libopus-sys/opus/celt/celt_encoder.c:2535-2590`, `libopus-sys/opus/celt/celt_encoder.c:2816-2818`
- Detail: Upstream `celt_encode_with_ec` computes `qext_bytes`, carves extension payload space from the packet, and entropy-codes QEXT data when `st->enable_qext` is set. Rust currently calls `celt_encode_with_ec` with `qext_payload=None` and `qext_bytes=0` (TODO noted), so QEXT coding branches guarded by `qext_bytes > 0` never activate, diverging from upstream QEXT-enabled encoding behavior.

230. [MEDIUM][Arch Dispatch Semantics][CELT/SILK SIMD] Rust SIMD dispatch ignores upstream `arch`-masked control path and always uses host CPUID result.
- Rust dispatch: `src/celt/simd/mod.rs:40-55`, `src/celt/simd/mod.rs:60-76`, `src/celt/simd/mod.rs:104-133`, `src/celt/simd/mod.rs:170-181`, `src/silk/simd/mod.rs:37-62`, `src/silk/simd/mod.rs:78-97`, `src/silk/simd/mod.rs:152-206`
- Rust evidence of dropped `arch` parameter: `src/celt/simd/mod.rs:170` (`_arch` unused)
- Upstream arch-masked dispatch macros/tables: `libopus-sys/opus/celt/x86/pitch_sse.h:74-77`, `libopus-sys/opus/celt/x86/pitch_sse.h:126-129`, `libopus-sys/opus/celt/x86/vq_sse.h:43-47`, `libopus-sys/opus/silk/x86/main_sse.h:266-269`, `libopus-sys/opus/silk/x86/main_sse.h:293`
- Detail: Upstream SIMD selection is keyed by `(arch & OPUS_ARCHMASK)` (allowing explicit arch-tier control by caller/state). Rust dispatches directly off runtime CPUID checks inside wrappers and does not honor an `arch`-tier input for SIMD selection, so execution-tier semantics differ from upstream RTCD behavior.

231. [MEDIUM][SIMD Coverage/Semantics][SILK FLP] Rust enables aarch64 NEON SIMD for `silk_inner_product_FLP`, while upstream only defines SIMD override on x86 AVX2.
- Rust: `src/silk/simd/mod.rs:78-82`, `src/silk/float/inner_product_FLP.rs:13-15`
- Upstream default macro path (scalar): `libopus-sys/opus/silk/float/SigProc_FLP.h:132-133`
- Upstream SIMD override path (x86 AVX2 only): `libopus-sys/opus/silk/x86/main_sse.h:279-293`, `libopus-sys/opus/silk/x86/x86_silk_map.c:165-175`
- Detail: Upstream routes `silk_inner_product_FLP` to AVX2 on x86 (or scalar otherwise). Rust adds an aarch64 NEON implementation and dispatches to it unconditionally on aarch64, introducing architecture-dependent numeric/decision-path behavior that upstream does not have.

232. [LOW][Build SIMD Flags][libopus-sys x86 AVX2] `libopus-sys/build.rs` uses weaker AVX2 flags for `SILK_SOURCES_AVX2` than upstream build scripts.
- Rust build flags: `libopus-sys/build.rs:169-173`
- Upstream CMake AVX2 flags for same source group: `libopus-sys/opus/CMakeLists.txt:528`, `libopus-sys/opus/CMakeLists.txt:531`
- Detail: Upstream applies `-mavx2 -mfma -mavx` to `silk_sources_avx2`; `build.rs` applies only `-mavx2` for `SILK_SOURCES_AVX2` (while using `-mavx2 -mfma` for `SILK_SOURCES_FLOAT_AVX2`). This is a compile-flag parity drift in SIMD build configuration.

233. [LOW][SIMD Coverage][SILK VAD] Rust SIMD path accelerates only VAD energy inner loop, not upstream full-function SSE4.1 override surface.
- Rust: `src/silk/VAD.rs:59-60`, `src/silk/VAD.rs:154`, `src/silk/simd/mod.rs:99-124`
- Upstream SIMD override: `libopus-sys/opus/silk/x86/main_sse.h:250-269`, `libopus-sys/opus/silk/x86/VAD_sse4_1.c:45-260`
- Detail: Upstream x86 can replace the full `silk_VAD_GetSA_Q8` routine with `silk_VAD_GetSA_Q8_sse4_1` via RTCD/presume macros. Rust keeps the scalar `silk_VAD_GetSA_Q8_c` control flow and only swaps in SIMD for the subframe energy accumulation helper, so SIMD coverage and dispatch granularity differ from upstream.

234. [MEDIUM][SIMD Coverage][DNN x86] Rust DNN SIMD runtime dispatch only uses AVX2+FMA tier, while upstream includes SSE2/SSE4.1 RTCD tiers.
- Rust dispatch: `src/dnn/simd/mod.rs:44-57`, `src/dnn/simd/mod.rs:71-85`, `src/dnn/simd/mod.rs:98-112`, `src/dnn/simd/mod.rs:133-147`, `src/dnn/simd/mod.rs:268-281`, `src/dnn/simd/mod.rs:322-335`
- Upstream x86 RTCD tables: `libopus-sys/opus/dnn/x86/x86_dnn_map.c:39-49`, `libopus-sys/opus/dnn/x86/x86_dnn_map.c:51-62`, `libopus-sys/opus/dnn/x86/x86_dnn_map.c:64-78`
- Upstream x86 dispatch macros: `libopus-sys/opus/dnn/x86/dnn_x86.h:82-114`
- Detail: On x86 CPUs without AVX2+FMA but with SSE2/SSE4.1, upstream still uses SIMD DNN implementations via RTCD tables. Rust falls back to scalar for those CPUs, reducing SIMD coverage/perf relative to upstream architecture tiers.

235. [MEDIUM][Arch Dispatch Semantics][DNN] Rust DNN path does not preserve upstream arch-indexed RTCD selection semantics.
- Rust call path: `src/dnn/nnet.rs:108-112`, `src/dnn/nnet.rs:137-160`, `src/dnn/simd/mod.rs:26-147`
- Upstream arch-indexed dispatch: `libopus-sys/opus/dnn/x86/dnn_x86.h:89`, `libopus-sys/opus/dnn/x86/dnn_x86.h:100`, `libopus-sys/opus/dnn/x86/dnn_x86.h:114`, `libopus-sys/opus/dnn/arm/dnn_arm.h:62`, `libopus-sys/opus/dnn/arm/dnn_arm.h:84`, `libopus-sys/opus/dnn/arm/dnn_arm.h:98`
- Detail: Upstream DNN compute entry points select implementations by `(arch & OPUS_ARCHMASK)` through RTCD tables/macros. Rust DNN wrappers select by host target/cpufeatures directly and expose no arch-tier control, so forced-tier testing/reproducibility semantics differ from upstream.

237. [LOW][Build SIMD Flags][libopus-sys x86 AVX2] `libopus-sys/build.rs` omits upstream `-mavx` companion flag for CELT/DNN AVX2 groups.
- Rust AVX2 group flags: `libopus-sys/build.rs:165-168`, `libopus-sys/build.rs:178`
- Upstream AVX2 flags assignment and usage: `libopus-sys/opus/CMakeLists.txt:528`, `libopus-sys/opus/CMakeLists.txt:530`, `libopus-sys/opus/CMakeLists.txt:535`, `libopus-sys/opus/meson.build:509`
- Detail: Upstream builds AVX2 groups with `-mavx -mfma -mavx2`; Rust uses `-mavx2 -mfma` for `CELT_SOURCES_AVX2` and `DNN_SOURCES_AVX2` (and only `-mavx2` for one SILK group in finding 232). This is additional AVX2 compile-flag parity drift.

238. [HIGH][Encode Input Semantics][SILK API] Rust `silk_Encode` accepted quantized `i16` input while upstream float build consumes `opus_res` and quantizes after stereo summation, causing SILK mono downmix divergence (resolved).
- Rust pre-fix path: `src/opus/opus_encoder.rs:2332`, `src/opus/opus_encoder.rs:2525`, `src/opus/opus_encoder.rs:2542`, `src/silk/enc_API.rs:120`, `src/silk/enc_API.rs:314`
- Upstream float path: `libopus-sys/opus/src/opus_encoder.c:2246`, `libopus-sys/opus/silk/enc_API.c:327`
- Fix: `src/silk/enc_API.rs:120`, `src/silk/enc_API.rs:276`, `src/silk/enc_API.rs:299`, `src/silk/enc_API.rs:314`, `src/silk/enc_API.rs:368`, `src/opus/opus_encoder.rs:2332`, `src/opus/opus_encoder.rs:2525`, `src/opus/opus_encoder.rs:2542`
- Evidence:
  - Before fix, packet-1 traces diverged at downmix/input buffer:
    - `RSILK pkt=1 downmix ... mix_hash=2d201ffa371a95d1` vs `CSILK ... mix_hash=cd93df8163d0318c`
    - `RSILK pkt=1 pre-frame ... in_hash=3573f440da2905bd` vs `CSILK ... in_hash=f978183424e60343`
  - After fix, packet-1 traces matched exactly through SILK entropy state:
    - downmix/input hashes equal, `frame_call tell/rng/val/nbits_total` equal.
  - Vector result moved to full parity:
    - `opus_newvectors`: `216/228` -> `228/228` passed.
