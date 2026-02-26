# Upstream Parity Diff Review

## Scope
Rust sources compared against upstream C in `libopus-sys/opus`.

## Remaining Items (Grouped)
Snapshot from the current alignment plans (`fix-g1` ... `fix-g7`) as of 2026-02-26.

Excluded from functional-equivalence tracking (C API-surface only): `177,178,179,186`.

Priority groups for execution:

1. QEXT correctness blockers (bitstream/PLC/sizing)
IDs: `none (resolved)`

2. Extensions and repacketizer semantic parity
IDs: `none (resolved)`

3. Public API surface parity (core, custom, multistream/projection, 24-bit)
IDs: `43,104,110,116,119,120,122`

4. DNN/DRED/OSCE model loading and constant parity
IDs: `none (resolved)`

5. SIMD/arch-dispatch/build-flag parity
IDs: `none (resolved)`

6. Documentation/version/metadata drift
IDs: `95,222,225,226`

7. Runtime semantics/assert-vs-status cleanup (non-blocking but broad)
IDs: `61,62,72,79,82,87,106,142,144,145,146,148,149,153,170,171,172`

## Findings

2. [RESOLVED][Encoder] Restricted application mode parity in encoder.
- Rust: `src/opus/opus_encoder.rs`, `tests/restricted_application_parity.rs`
- Upstream: `libopus-sys/opus/src/opus_encoder.c`
- Detail: Rust now mirrors restricted SILK/CELT control-flow behavior; C-vs-Rust parity tests cover restricted-SILK sub-10 ms encode rejection and `OPUS_SET_APPLICATION_REQUEST` rejection semantics for restricted modes.

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

12. [RESOLVED][DRED] Decoder DRED API entry points are implemented.
- Rust: `src/opus/opus_decoder.rs`, `src/lib.rs`, `tests/dred_decode_parity.rs`
- Upstream: `libopus-sys/opus/src/opus_decoder.c:1609`, `libopus-sys/opus/src/opus_decoder.c:1643`, `libopus-sys/opus/src/opus_decoder.c:1677`
- Detail: Rust now exports `opus_decoder_dred_decode`, `opus_decoder_dred_decode24`, and `opus_decoder_dred_decode_float` (plus `OpusDecoder` convenience methods), with tools-dnn parity coverage against C decode behavior.

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

35. [RESOLVED][Extensions] Public extension parse/count path now uses iterator semantics including repeat-extension expansion.
- Rust: `src/opus/extensions.rs`, `tests/extensions_repacketizer_parity.rs`
- Upstream: `libopus-sys/opus/src/extensions.c:154-220`, `libopus-sys/opus/src/extensions.c:226-383`
- Detail: `opus_packet_extensions_count()` and `opus_packet_extensions_parse()` are iterator-based and cover ID=2 repeat semantics and frame-limit handling; tests validate repeat roundtrip behavior.

36. [RESOLVED][Extensions] Extension generation now includes upstream repeat-compaction logic.
- Rust: `src/opus/extensions.rs`, `tests/extensions_repacketizer_parity.rs`
- Upstream: `libopus-sys/opus/src/extensions.c:471-624`
- Detail: Generator performs cross-frame repeat analysis and emits repeat indicators; parity tests assert repeat markers and consistent parse/count behavior.

37. [RESOLVED][Extensions] `_ext` helper APIs are implemented.
- Rust: `src/opus/extensions.rs`, `src/lib.rs`, `tests/extensions_repacketizer_parity.rs`
- Upstream: `libopus-sys/opus/src/extensions.c:341-421`
- Detail: Rust provides `opus_packet_extensions_count_ext` and `opus_packet_extensions_parse_ext`; tools-gated parity tests compare outputs against upstream C.

38. [RESOLVED][Repacketizer+Extensions] Repacketizer state retains padding/extension metadata from input packets.
- Rust: `src/opus/repacketizer.rs`, `tests/extensions_repacketizer_parity.rs`
- Upstream: `libopus-sys/opus/src/repacketizer.c:86-99`, `libopus-sys/opus/src/repacketizer.c:143-177`
- Detail: Rust stores per-packet padding bytes and frame counts from parse, enabling extension-preserving repacketization.

39. [RESOLVED][Repacketizer+Extensions] `out_range_impl_ext` merges caller and source-packet extensions.
- Rust: `src/opus/repacketizer.rs`, `tests/extensions_repacketizer_parity.rs`
- Upstream: `libopus-sys/opus/src/repacketizer.c:143-177`, `libopus-sys/opus/src/repacketizer.c:260-265`, `libopus-sys/opus/src/repacketizer.c:308-311`
- Detail: Rust parses stored per-input padding extensions, renumbers frame indices relative to output range, and emits combined extension payload.

40. [RESOLVED][Repacketizer+Extensions] Non-pad extension overhead formula now matches upstream.
- Rust: `src/opus/repacketizer.rs`
- Upstream: `libopus-sys/opus/src/repacketizer.c:267`
- Detail: Rust uses `ext_len + (ext_len ? (ext_len+253)/254 : 1)` semantics, matching upstream behavior at `ext_len % 254 == 0`.

41. [RESOLVED][Repacketizer+Extensions] Extension frame-index validation is constrained by output packet frame count.
- Rust: `src/opus/extensions.rs`, `src/opus/repacketizer.rs`
- Upstream: `libopus-sys/opus/src/extensions.c:471-473`, `libopus-sys/opus/src/extensions.c:495`, `libopus-sys/opus/src/repacketizer.c:263-264`, `libopus-sys/opus/src/repacketizer.c:309-310`
- Detail: Generator takes `nb_frames` and rejects out-of-range frame indices (`frame >= nb_frames`) as upstream does.

43. [MEDIUM] Missing CELT custom 24-bit API parity.
- Rust: `src/celt/celt_encoder.rs`, `src/celt/celt_decoder.rs` (no `opus_custom_encode24` / `opus_custom_decode24` entry points found)
- Upstream: `libopus-sys/opus/celt/celt_encoder.c:2871`, `libopus-sys/opus/celt/celt_decoder.c:1658`
- Detail: Upstream custom CELT API exports 24-bit integer encode/decode functions; matching Rust entry points are not present.

44. [LOW] Missing direct CELT custom C-style CTL/destroy entry points.
- Rust: `src/celt/celt_encoder.rs`, `src/celt/celt_decoder.rs`
- Upstream: `libopus-sys/opus/celt/celt_encoder.c:260`, `libopus-sys/opus/celt/celt_encoder.c:2941`, `libopus-sys/opus/celt/celt_decoder.c:278`, `libopus-sys/opus/celt/celt_decoder.c:1722`
- Detail: Upstream exposes direct C-style `opus_custom_encoder_ctl` / `opus_custom_decoder_ctl` and explicit destroy functions. Rust exposes struct-based APIs but does not mirror these direct entry points.

45. [RESOLVED][DRED] Full DRED object API surface is now mirrored in Rust wrappers.
- Rust: `src/opus/opus_decoder.rs`, `src/lib.rs`
- Upstream: `libopus-sys/opus/src/opus_decoder.c:1363`, `libopus-sys/opus/src/opus_decoder.c:1381`, `libopus-sys/opus/src/opus_decoder.c:1395`, `libopus-sys/opus/src/opus_decoder.c:1417`, `libopus-sys/opus/src/opus_decoder.c:1423`, `libopus-sys/opus/src/opus_decoder.c:1511`, `libopus-sys/opus/src/opus_decoder.c:1520`, `libopus-sys/opus/src/opus_decoder.c:1539`, `libopus-sys/opus/src/opus_decoder.c:1548`, `libopus-sys/opus/src/opus_decoder.c:1587`
- Detail: Rust now provides `opus_dred_decoder_get_size/init/create/destroy/ctl`, `opus_dred_get_size/alloc/free`, `opus_dred_parse`, and `opus_dred_process` wrappers, re-exported at crate root for public API parity.

46. [LOW] Missing direct CELT custom encode/decode entry point functions.
- Rust: `src/celt/celt_encoder.rs`, `src/celt/celt_decoder.rs` (no direct `opus_custom_encode`, `opus_custom_encode_float`, `opus_custom_decode`, `opus_custom_decode_float` free-function exports)
- Upstream: `libopus-sys/opus/celt/celt_encoder.c:2838`, `libopus-sys/opus/celt/celt_encoder.c:2906`, `libopus-sys/opus/celt/celt_decoder.c:1629`, `libopus-sys/opus/celt/celt_decoder.c:1690`
- Detail: Upstream provides stable C-style entry points for custom CELT encode/decode variants. Rust currently exposes struct-centric APIs instead of matching these direct function symbols.

47. [RESOLVED][Runtime Semantics][Analysis] `downmix_and_resample` now mirrors upstream unsupported-Fs assertion behavior.
- Rust: `src/opus/analysis.rs`, `src/opus/analysis.rs` (unit tests)
- Upstream: `libopus-sys/opus/src/analysis.c:181`
- Detail: Added upstream-equivalent debug assertion (`Fs` in `{16000,24000,48000}`) and tests for valid 24 kHz behavior plus debug-assert failure on unsupported `Fs`.

48. [LOW] Missing direct top-level create/get_size/destroy C entry points for encoder/decoder.
- Rust: `src/opus/opus_encoder.rs` (`OpusEncoder::new`), `src/opus/opus_decoder.rs` (`OpusDecoder::new`)
- Upstream: `libopus-sys/opus/src/opus_encoder.c:194`, `libopus-sys/opus/src/opus_encoder.c:622`, `libopus-sys/opus/src/opus_encoder.c:3362`, `libopus-sys/opus/src/opus_decoder.c:121`, `libopus-sys/opus/src/opus_decoder.c:186`, `libopus-sys/opus/src/opus_decoder.c:1244`
- Detail: Upstream exports C-style `opus_encoder_get_size/init/create/destroy` and `opus_decoder_get_size/init/create/destroy` entry points. Rust uses struct constructors/methods and does not mirror these direct symbols.

49. [RESOLVED][Packet Parser] `opus_packet_parse_impl` now clears padding output on entry like upstream.
- Rust: `src/opus/packet.rs`, `tests/packet_parse_impl_parity.rs`
- Upstream: `libopus-sys/opus/src/opus.c:239-244`
- Detail: Rust now zeroes `padding_out` before validation, matching upstream deterministic error-path behavior; parity test compares Rust vs C `opus_packet_parse_impl` outputs for invalid packets.

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

56. [RESOLVED][Encoder CTL] Bitrate validation/clamping now matches upstream semantics.
- Rust: `src/opus/opus_encoder.rs`, `tests/restricted_application_parity.rs`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:2826-2827`
- Detail: Rust now mirrors upstream non-positive explicit bitrate rejection and `750000*channels` clamp behavior; parity test `bitrate_ctl_semantics_match_c` validates request-path behavior against C `OPUS_SET/GET_BITRATE_REQUEST`.

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

66. [RESOLVED][Runtime Semantics] Decoder validation checks now use debug-only assertion gates.
- Rust: `src/opus/opus_decoder.rs`, `src/celt/celt_decoder.rs`
- Upstream: `libopus-sys/opus/src/opus_decoder.c:92-123`; `libopus-sys/opus/celt/celt_decoder.c:136-155`
- Detail: Rust now uses `debug_assert!` in validation helpers, aligning production behavior with upstream `VALIDATE_*` checks that are compiled out unless assertion/hardening gates are enabled.

67. [RESOLVED][Runtime Semantics] Decode-loop invariants now match upstream `celt_assert` gating.
- Rust: `src/opus/opus_decoder.rs`
- Upstream: `libopus-sys/opus/src/opus_decoder.c:777`, `libopus-sys/opus/src/opus_decoder.c:813`, `libopus-sys/opus/src/opus_decoder.c:864`
- Detail: Rust decode-loop consistency checks were converted from unconditional `assert!` to `debug_assert_eq!`, preventing release aborts while retaining debug diagnostics.

68. [RESOLVED][Runtime Semantics] Encoder internal invariants now match upstream `celt_assert` behavior.
- Rust: `src/opus/opus_encoder.rs`
- Upstream: `libopus-sys/opus/src/opus_encoder.c:2118`, `libopus-sys/opus/src/opus_encoder.c:2230`, `libopus-sys/opus/src/opus_encoder.c:2628`
- Detail: Rust now uses debug-only assertions for these internal invariants, eliminating unconditional runtime abort risk in release builds.

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

76. [RESOLVED][DRED] `opus_decode_native` now mirrors upstream DRED input flow and pre-processing path.
- Rust: `src/opus/opus_decoder.rs:1006`, `src/opus/opus_multistream_decoder.rs:235`
- Upstream: `libopus-sys/opus/src/opus_decoder.c:718` plus DRED pre-processing block at `:732-766`
- Detail: Rust `opus_decode_native` now accepts DRED input (`dred`, `dred_offset`) and stages DRED feature frames into Deep PLC state before decode, matching upstream decode-time DRED-assisted flow. Existing non-DRED call sites explicitly pass `None, 0` to preserve behavior.

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

93. [RESOLVED][Runtime Semantics][SILK] Resampler init now returns status codes instead of panicking.
- Rust: `src/silk/resampler/mod.rs`, `src/silk/control_codec.rs`, `src/silk/decoder_set_fs.rs`
- Upstream: `libopus-sys/opus/silk/resampler.c:silk_resampler_init`
- Detail: Rust now mirrors upstream by returning `-1` on invalid rate tuples/unsupported ratios and threading status through encoder/decoder setup call paths.

94. [RESOLVED][Runtime Semantics][DRED] Unsupported sample-rate branch now uses assertion-style fallback.
- Rust: `src/dnn/dred/encoder.rs`
- Upstream: `libopus-sys/opus/dnn/dred_encoder.c:165`, `libopus-sys/opus/dnn/dred_encoder.c:207`
- Detail: Rust no longer panics in the default sample-rate branch and now mirrors upstream assert-style behavior.

95. [LOW][Lib Info] `opus_get_version_string()` does not match upstream format/content.
- Rust: `src/celt/common.rs:426`
- Upstream: `libopus-sys/opus/celt/celt.c:360-372`
- Detail: Upstream returns `"libopus " PACKAGE_VERSION` with optional `-fixed`/`-fuzzing` suffixes. Rust returns `"opurs {crate_version}"`, which intentionally diverges from upstream identifier/version/suffix semantics.

96. [RESOLVED][Lib Info] `opus_strerror()` now matches upstream canonical message text.
- Rust: `src/celt/common.rs`
- Upstream: `libopus-sys/opus/celt/celt.c:342-358`
- Detail: Rust now returns the same canonical short strings as upstream (e.g., `"success"`, `"invalid argument"`, `"unknown error"`), removing prior numeric suffix divergence.

97. [RESOLVED][Extensions][RTE Semantics] Public extension count/parse/find helpers now apply full iterator semantics.
- Rust: `src/opus/extensions.rs`
- Upstream: `libopus-sys/opus/src/extensions.c:329-383`
- Detail: Helpers are `nb_frames`-aware and route through `OpusExtensionIterator`, including repeat-extension expansion and frame-limit semantics.

98. [RESOLVED][Extensions][API Coverage] Frame-ordered extension helpers are implemented.
- Rust: `src/opus/extensions.rs`, `src/lib.rs`
- Upstream: `libopus-sys/opus/src/extensions.c:341-420`
- Detail: Rust implements `opus_packet_extensions_count_ext` and `opus_packet_extensions_parse_ext` and re-exports them for parity tests.

99. [RESOLVED][Extensions][Generator Semantics] Generator signature and repeat optimization match upstream shape.
- Rust: `src/opus/extensions.rs`
- Upstream: `libopus-sys/opus/src/extensions.c:471-635`
- Detail: Rust generator takes `nb_frames`, validates frame indices against it, and executes repeat-indicator compaction logic.

100. [RESOLVED][Extensions][Iterator API] Iterator control helpers are implemented.
- Rust: `src/opus/extensions.rs`
- Upstream: `libopus-sys/opus/src/extensions.c:134-153`
- Detail: `OpusExtensionIterator` exposes `reset()` and `set_frame_max()` alongside `new()/next()/find()`.

101. [RESOLVED][Platform/State Layout] `align()` now uses computed union alignment semantics.
- Rust: `src/opus/opus_private.rs`
- Upstream: `libopus-sys/opus/src/opus_private.h:213-223`
- Detail: Rust now computes alignment from a C-repr union equivalent (`i32/i64/f32/pointer`) via `align_of`, matching upstream platform-dependent alignment intent instead of hardcoding 8.

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

107. [RESOLVED][CPU Dispatch][Soft Clip] Decoder soft-clip path now threads runtime arch to implementation.
- Rust: `src/opus/opus_decoder.rs`, `src/opus/packet.rs`
- Upstream: `libopus-sys/opus/src/opus_decoder.c:874`, `libopus-sys/opus/src/opus.c:39`, `libopus-sys/opus/src/opus.c:163-165`
- Detail: Rust now exposes internal `opus_pcm_soft_clip_impl(..., arch)` and decoder calls it with `st.celt_dec.arch`, matching upstream arch-threaded call shape.

108. [RESOLVED][Version Targeting] Crate-level compatibility/docs now target libopus 1.6.1.
- Rust: `src/lib.rs`
- Upstream target for this review: `libopus-sys/opus` (1.6.1 line)
- Detail: Top-level crate docs now state bit-exactness against libopus 1.6.1, removing version-target drift.

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

115. [RESOLVED][Repacketizer][Extensions Preservation] Repacketizer ingest captures padding metadata and preserves extensions through output.
- Rust: `src/opus/repacketizer.rs`, `tests/extensions_repacketizer_parity.rs`
- Upstream: `libopus-sys/opus/src/repacketizer.c:86-99`, `libopus-sys/opus/src/repacketizer.c:143-177`
- Detail: Rust stores parse-time padding metadata in repacketizer state and merges source-packet extensions into repacketized output.

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

131. [RESOLVED][Version Targeting][Docs] DNN subsystem documentation now targets libopus 1.6.1.
- Rust: `src/dnn/README.md`
- Upstream target for this review: `libopus-sys/opus` (1.6.1 line)
- Detail: Updated DNN README baseline wording to 1.6.1 to match current parity target and build metadata.

132. [LOW][Decoder API Semantics] Rust decode entrypoints cannot represent upstream `(data == NULL, len > 0)` call form.
- Rust: `src/opus/opus_decoder.rs:124-131`, `src/opus/opus_decoder.rs:142-149`, `src/opus/opus_decoder.rs:900-927`
- Upstream: `libopus-sys/opus/src/opus_decoder.c:896-921`, `libopus-sys/opus/src/opus_decoder.c:985-990`
- Detail: Upstream decode APIs take `(const unsigned char *data, opus_int32 len)` and treat `data == NULL` as packet-loss/PLC regardless of `len`. Rust slice-based decode APIs (`data: &[u8]`) can only encode PLC via empty slices, so this pointer/length-state API form is not representable.

133. [RESOLVED][Version Targeting][Docs] Top-level project README now advertises 1.6.1 parity.
- Rust: `README.md`
- Upstream target for this review: `libopus-sys/opus` (1.6.1 line)
- Detail: Project README now states bit-exactness with libopus 1.6.1.

134. [RESOLVED][Version Targeting][Comments] Source-level implementation comments target 1.6.1 baseline.
- Rust: `src/dnn/simd/x86.rs`, `src/dnn/simd/aarch64.rs`, `src/opus/mlp/tansig.rs`
- Upstream target for this review: `libopus-sys/opus` (1.6.1 line)
- Detail: Updated/re-audited provenance comments to reference the current 1.6.1 baseline in active modules.

135. [RESOLVED][Runtime Semantics][DRED] DRED loaded-state checks now follow upstream assert-gated behavior.
- Rust: `src/dnn/dred/encoder.rs`, `src/dnn/dred/decoder.rs`
- Upstream: `libopus-sys/opus/dnn/dred_encoder.c:95`, `libopus-sys/opus/dnn/dred_encoder.c:215`, `libopus-sys/opus/src/opus_decoder.c` (`opus_dred_process` assert-gated expectation)
- Detail: Converted unconditional `assert!` preconditions to `debug_assert!` for DRED encoder/decoder loaded-state checks, matching upstream `celt_assert` gating in non-assert builds.

136. [RESOLVED][Runtime Semantics][DNN] LPCNet/FARGAN state precondition checks now follow assert-gated behavior.
- Rust: `src/dnn/lpcnet.rs`, `src/dnn/fargan.rs`
- Upstream: `libopus-sys/opus/dnn/lpcnet_plc.c:110`, `libopus-sys/opus/dnn/lpcnet_plc.c:159`, `libopus-sys/opus/dnn/fargan.c:85`, `libopus-sys/opus/dnn/fargan.c:202`
- Detail: Converted unconditional `assert!` checks to `debug_assert!` in LPCNet/FARGAN precondition sites, matching upstream `celt_assert` gating semantics.

137. [RESOLVED][Runtime Semantics][DNN] Core `nnet`/`nndsp` invariant checks now follow assert-gated behavior.
- Rust: `src/dnn/nnet.rs`, `src/dnn/nndsp.rs`
- Upstream: `libopus-sys/opus/dnn/nnet.c:85-86`, `libopus-sys/opus/dnn/nnet.c:111`, `libopus-sys/opus/dnn/nnet.c:128`, `libopus-sys/opus/dnn/nnet.c:142`, `libopus-sys/opus/dnn/nndsp.c:169`, `libopus-sys/opus/dnn/nndsp.c:364-366`
- Detail: Converted these internal invariant checks from unconditional `assert!`/`assert_eq!` to `debug_assert!`/`debug_assert_eq!`, aligning with upstream `celt_assert`-style gating.

138. [RESOLVED][Packet API][Error Handling] `opus_packet_pad_impl` now returns `OPUS_BAD_ARG` for out-of-bounds `len/new_len`.
- Rust: `src/opus/repacketizer.rs`, `tests/extensions_repacketizer_parity.rs`
- Upstream: `libopus-sys/opus/src/repacketizer.c:341-347`, `libopus-sys/opus/src/repacketizer.c:354`
- Detail: Added explicit `len/new_len <= data.len()` guards before slice indexing, removing panic-only behavior and aligning with status-return argument handling. Added tests for both oversize `len` and `new_len`.

139. [RESOLVED][Runtime Semantics][Repacketizer] `opus_packet_unpad` postcondition check now matches upstream assert gating.
- Rust: `src/opus/repacketizer.rs`
- Upstream: `libopus-sys/opus/src/repacketizer.c:388`
- Detail: Converted unconditional `assert!` to `debug_assert!`, matching upstream `celt_assert` release behavior.

140. [RESOLVED][Runtime Semantics][CELT Entropy/Rate/VQ] Internal invariant checks now follow assert-gated semantics.
- Rust: `src/celt/entenc.rs`, `src/celt/entdec.rs`, `src/celt/laplace.rs`, `src/celt/cwrs.rs`, `src/celt/rate.rs`, `src/celt/vq.rs`
- Upstream: `libopus-sys/opus/celt/entenc.c`, `libopus-sys/opus/celt/entdec.c`, `libopus-sys/opus/celt/laplace.c`, `libopus-sys/opus/celt/cwrs.c`, `libopus-sys/opus/celt/rate.c`, `libopus-sys/opus/celt/vq.c`
- Detail: Converted these internal invariant checks from unconditional `assert!` to `debug_assert!`, matching upstream `celt_assert` release behavior.

141. [RESOLVED][Runtime Semantics][CELT LPC/Pitch/Bands] Internal invariant checks now follow assert-gated semantics.
- Rust: `src/celt/celt_lpc.rs`, `src/celt/pitch.rs`, `src/celt/bands.rs`
- Upstream: `libopus-sys/opus/celt/celt_lpc.c`, `libopus-sys/opus/celt/pitch.c`, `libopus-sys/opus/celt/bands.c`
- Detail: Converted tracked internal invariant checks from unconditional `assert!` to `debug_assert!`, matching upstream `celt_assert` release behavior.

142. [LOW][Runtime Semantics][SILK] Additional SILK invariant checks are unconditional `assert!` in Rust.
- Rust: `src/silk/control_codec.rs:167-168`, `src/silk/control_codec.rs:222`, `src/silk/control_codec.rs:228`, `src/silk/control_codec.rs:307-312`, `src/silk/decode_frame.rs:49`, `src/silk/decode_frame.rs:84`, `src/silk/decode_frame.rs:110`, `src/silk/decode_frame.rs:132`, `src/silk/decode_core.rs:133`, `src/silk/decode_core.rs:204`, `src/silk/NSQ.rs:220`, `src/silk/NSQ.rs:423`, `src/silk/NSQ.rs:441`, `src/silk/NSQ_del_dec.rs:311`, `src/silk/NSQ_del_dec.rs:541`, `src/silk/NSQ_del_dec.rs:608`, `src/silk/float/pitch_analysis_core_FLP.rs:90-92`, `src/silk/float/pitch_analysis_core_FLP.rs:218`, `src/silk/float/pitch_analysis_core_FLP.rs:244`, `src/silk/float/find_pred_coefs_FLP.rs:42`
- Upstream: `libopus-sys/opus/silk/control_codec.c:241-242`, `libopus-sys/opus/silk/control_codec.c:302`, `libopus-sys/opus/silk/control_codec.c:315`, `libopus-sys/opus/silk/control_codec.c:393-398`, `libopus-sys/opus/silk/decode_frame.c:68`, `libopus-sys/opus/silk/decode_frame.c:104`, `libopus-sys/opus/silk/decode_frame.c:127`, `libopus-sys/opus/silk/decode_frame.c:145`, `libopus-sys/opus/silk/NSQ.c:402`, `libopus-sys/opus/silk/NSQ_del_dec.c:686`, `libopus-sys/opus/silk/float/pitch_analysis_core_FLP.c:118-119`, `libopus-sys/opus/silk/float/pitch_analysis_core_FLP.c:532`, `libopus-sys/opus/silk/float/find_pred_coefs_FLP.c:55`
- Detail: Upstream uses `celt_assert`/`silk_assert` for many of these invariants. Rust promotes them to unconditional `assert!`, which can hard-abort runtime encode/decode flows in non-assert configurations where upstream would usually not.

143. [RESOLVED][Runtime Semantics][OSCE/Freq/SILK-FLP] OSCE/Freq invariant checks now follow assert-gated semantics.
- Rust: `src/dnn/osce.rs`, `src/dnn/freq.rs`
- Upstream: `libopus-sys/opus/dnn/osce_features.c`, `libopus-sys/opus/dnn/freq.c`
- Detail: Converted the tracked OSCE/Freq invariant checks from unconditional `assert!`/`assert_eq!` to `debug_assert!`/`debug_assert_eq!`, matching upstream release-mode assertion gating.

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

156. [RESOLVED][Runtime Semantics][SILK Pitch Decode] `silk_decode_pitch` now matches upstream fallback table selection.
- Rust: `src/silk/decode_pitch.rs`
- Upstream: `libopus-sys/opus/silk/decode_pitch.c:49-66`
- Detail: Replaced strict tuple match + `unreachable!` with upstream-style branching: `Fs_kHz==8` uses stage2 tables, otherwise stage3 tables, with debug-assert-gated `nb_subfr` checks on 10 ms fallback branches.

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

163. [RESOLVED][API Surface][DNN PitchDNN] One-shot blob model-loading helper is now mirrored.
- Rust: `src/dnn/pitchdnn.rs`, `tests/dnn_integration.rs`
- Upstream: `libopus-sys/opus/dnn/pitchdnn.c:71-79` (`pitchdnn_load_model(PitchDNNState*, const void*, int)`)
- Detail: Added direct blob-loading API (`pitchdnn_load_model` plus `PitchDNNState::load_model`) that parses and initializes in one call, with integration coverage for valid and invalid blobs.

164. [LOW][Tooling Semantics][Demo Backend] Rust demo backend uses `panic!` for some feature-gated DNN/DRED operations instead of recoverable status flow.
- Rust: `src/tools/demo/backend.rs:120`, `src/tools/demo/backend.rs:131`, `src/tools/demo/backend.rs:175`
- Upstream: `libopus-sys/opus/src/opus_demo.c:830-833`, `libopus-sys/opus/src/opus_demo.c:923-927` (feature-gated paths are compile-time conditioned and wired through ctl calls)
- Detail: Rust demo backend aborts with `panic!` when required features (`dred`, `builtin-weights`, `deep-plc`) are unavailable for requested operations. Upstream demo behavior is controlled via compile-time feature paths and ctl calls rather than Rust-style hard panics in wrapper methods.

165. [RESOLVED][Runtime Semantics][SILK Resampler] Down-FIR interpolation default branch now uses assert-gated fallback behavior.
- Rust: `src/silk/resampler/down_fir.rs`
- Upstream: `libopus-sys/opus/silk/resampler_private_down_FIR.c:138-141`
- Detail: Replaced `unreachable!()` with `debug_assert!(false, ...)` and return of the remaining output slice, matching upstream `celt_assert(0)` + return semantics.

166. [RESOLVED][QEXT][SILK Resampler] `silk_resampler_init` now includes upstream 96 kHz delay-table/rate-index coverage.
- Rust: `src/silk/resampler/mod.rs` (+ qext unit coverage)
- Upstream: `libopus-sys/opus/silk/resampler.c:53-68`, `libopus-sys/opus/silk/resampler.c:70-72`, `libopus-sys/opus/silk/resampler.c:92-114`
- Detail: Added 96 kHz QEXT rate-ID/table support and validated with qext-gated resampler init tests.

167. [RESOLVED][API Semantics][SILK Resampler] `silk_resampler_init` now mirrors upstream status-return semantics.
- Rust: `src/silk/resampler/mod.rs`, `src/silk/control_codec.rs`, `src/silk/decoder_set_fs.rs`
- Upstream: `libopus-sys/opus/silk/resampler.c:79-84`, `libopus-sys/opus/silk/resampler.c:99-101`, `libopus-sys/opus/silk/resampler.c:110-112`, `libopus-sys/opus/silk/resampler.c:178`
- Detail: Rust now uses in-place init + status return (`0`/`-1`) and propagates errors at call-sites instead of panic-based failure flow.

168. [RESOLVED][Runtime Semantics][SILK Resampler] Unsupported downsampling ratio handling now matches upstream assert+status flow.
- Rust: `src/silk/resampler/mod.rs`
- Upstream: `libopus-sys/opus/silk/resampler.c:161-165`
- Detail: Replaced panic/unreachable paths with debug-assert + `SILK_RESAMPLER_INVALID` status returns, with debug/release unit coverage.

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

173. [RESOLVED][API Surface][SILK Decoder Init] Decoder init/reset now mirror upstream in-place status-return API.
- Rust: `src/silk/init_decoder.rs`, `src/silk/dec_API.rs`
- Upstream: `libopus-sys/opus/silk/init_decoder.c:43-45`, `libopus-sys/opus/silk/init_decoder.c:73-75`, `libopus-sys/opus/silk/dec_API.c:98-101`, `libopus-sys/opus/silk/dec_API.c:176-179`
- Detail: `silk_reset_decoder` and `silk_init_decoder` now operate in-place and return status (`0`), with decode-path call sites updated to use the in-place return-flow semantics.

174. [RESOLVED][Runtime Semantics][SILK+OSCE] Decoder reset now uses upstream default OSCE method.
- Rust: `src/dnn/osce.rs`, `src/silk/init_decoder.rs`
- Upstream: `libopus-sys/opus/silk/init_decoder.c:61-64`, `libopus-sys/opus/dnn/osce.h:59-67`
- Detail: Added `OSCE_DEFAULT_METHOD` constant and switched `silk_reset_decoder` to call `osce_reset(..., OSCE_DEFAULT_METHOD)` instead of `OSCE_METHOD_NONE`.

175. [RESOLVED][QEXT][DRED] DRED 16 kHz conversion now mirrors upstream 96 kHz support.
- Rust: `src/dnn/dred/encoder.rs` (including `dred_convert_to_16k_supports_96k_qext` unit test)
- Upstream: `libopus-sys/opus/dnn/dred_encoder.c:143-163`
- Detail: 96 kHz QEXT conversion arm is present and now covered by dedicated unit test to lock non-panic behavior.

176. [RESOLVED][Initialization Semantics][DRED] `dred_encoder_init` now mirrors upstream built-in model auto-load behavior.
- Rust: `src/dnn/dred/encoder.rs`, `src/opus/opus_encoder.rs`, `src/opus/opus_encoder.rs` (unit tests)
- Upstream: `libopus-sys/opus/dnn/dred_encoder.c:79-87`, `libopus-sys/opus/src/opus_encoder.c:288-290`
- Detail: `DREDEnc::init` now auto-loads built-in RDOVAE weights when available, and `OpusEncoder::new` now invokes DRED init like upstream encoder initialization.

177. [EXCLUDED][API Coverage][DRED] Encoder deinitialization helper is excluded from functional-equivalence scope.
- Rust: `src/dnn/dred/encoder.rs`
- Upstream: `libopus-sys/opus/dnn/dred_encoder.h:65`
- Detail: This is a lifecycle API-surface item (no runtime algorithmic divergence). We track functional behavior parity and do not require adding C-specific lifecycle wrappers.

178. [EXCLUDED][API Coverage][MLP] Analysis helper entry points are excluded from functional-equivalence scope.
- Rust: `src/opus/mlp/analysis_mlp.rs`
- Upstream: `libopus-sys/opus/src/mlp.h:56-58`, `libopus-sys/opus/src/mlp.c:70-131`
- Detail: Rust already implements equivalent layer behavior in the active analysis path; this item is direct API-shape parity only and is excluded from functional-equivalence comparisons.

179. [EXCLUDED][DNN API Coverage][LPCNet] Standalone LPCNet decoder/synthesis API surface is excluded from functional-equivalence scope.
- Rust: `src/dnn/lpcnet.rs` (provides `LPCNetEncState`, `LPCNetPLCState`, feature/PLC helpers; no `LPCNetDecState`/`LPCNetState` create/decode/synthesize API)
- Upstream: `libopus-sys/opus/dnn/lpcnet.h:40-79`, `libopus-sys/opus/dnn/lpcnet.h:134-166`
- Detail: This is a standalone API-surface delta outside Opus packet encode/decode functional parity scope in this repo; DNN parity tracking here focuses on model loading and runtime paths that are exercised by Opus encoder/decoder behavior.

180. [RESOLVED][DNN API Surface][LPCNet] Direct blob-based LPCNet load helpers are mirrored as one-shot APIs.
- Rust: `src/dnn/lpcnet.rs`, `tests/dnn_integration.rs`
- Upstream: `libopus-sys/opus/dnn/lpcnet.h:97`, `libopus-sys/opus/dnn/lpcnet.h:180-181`
- Detail: Added blob-loading entry points `lpcnet_encoder_load_model` and `lpcnet_plc_load_model` plus corresponding state methods (`load_model_blob`), and coverage that blob and parsed-array model paths produce equivalent loaded states.

181. [RESOLVED][Constants][DRED] Experimental DRED version constant now matches upstream.
- Rust: `src/dnn/dred/config.rs`
- Upstream: `libopus-sys/opus/dnn/dred_config.h:35`
- Detail: Updated Rust `DRED_EXPERIMENTAL_VERSION` to `12`, matching upstream signaling/version-tag behavior.

182. [RESOLVED][Model Constants][DRED] RDOVAE stats-table dimensions now match upstream generated data.
- Rust: `src/dnn/dred/stats.rs`
- Upstream: `libopus-sys/opus/dnn/dred_rdovae_constants.h:12-19`, `libopus-sys/opus/dnn/dred_rdovae_stats_data.h:15-24`
- Detail: Regenerated Rust DRED stats tables from upstream data source so latent/state quantization arrays are now `400/800` like upstream.

183. [RESOLVED][Data Layout][DRED Decoder] `OpusDRED.latents` layout matches upstream `(DRED_LATENT_DIM+1)` storage.
- Rust: `src/dnn/dred/decoder.rs`
- Upstream: `libopus-sys/opus/dnn/dred_decoder.h:40`, `libopus-sys/opus/dnn/dred_decoder.c:117`, `libopus-sys/opus/dnn/dred_decoder.c:123`
- Detail: Rust stores latents as `(DRED_NUM_REDUNDANCY_FRAMES/2)*(DRED_LATENT_DIM+1)` and writes the trailing `q_level` slot, matching upstream decoder layout semantics.

184. [LOW][Runtime Semantics][LPCNet PLC] FEC queue-full behavior differs from upstream assert-gated contract.
- Rust: `src/dnn/lpcnet.rs:597-607`
- Upstream: `libopus-sys/opus/dnn/lpcnet_plc.c:96-99`
- Detail: Upstream `lpcnet_plc_fec_add` asserts `fec_fill_pos < PLC_MAX_FEC` and then appends. Rust instead handles `fec_fill_pos == PLC_MAX_FEC` by shifting buffered entries and continuing insertion. This changes overflow/invariant behavior compared with upstream's assert-based contract.

185. [LOW][API Signature][DNN NNet] Generic DNN compute helpers omit upstream `arch` argument.
- Rust: `src/dnn/nnet.rs:197`, `src/dnn/nnet.rs:214`, `src/dnn/nnet.rs:253`, `src/dnn/nnet.rs:272`, `src/dnn/nnet.rs:299`
- Upstream: `libopus-sys/opus/dnn/nnet.h:89-93`, `libopus-sys/opus/dnn/nnet.c:60`, `libopus-sys/opus/dnn/nnet.c:76`, `libopus-sys/opus/dnn/nnet.c:107`, `libopus-sys/opus/dnn/nnet.c:124`, `libopus-sys/opus/dnn/nnet.c:136`
- Detail: Upstream signatures carry an explicit `arch` parameter through generic dense/GRU/GLU/conv helpers (matching runtime dispatch conventions). Rust equivalents expose no `arch` parameter, so function-level API surface and call-shape are not source-equivalent.

186. [EXCLUDED][API Coverage][DNN NNDSP] Explicit adaptive-filter init helpers are excluded from functional-equivalence scope.
- Rust: `src/dnn/nndsp.rs`
- Upstream: `libopus-sys/opus/dnn/nndsp.h:80-84`, `libopus-sys/opus/dnn/nndsp.c:48-60`
- Detail: Rust state reset behavior is functionally equivalent via defaults/resets in active runtime paths; standalone init-function API parity is excluded.

187. [RESOLVED][API Signature][LPCNet] Single-frame feature extraction helpers thread upstream `arch` argument.
- Rust: `src/dnn/lpcnet.rs`
- Upstream: `libopus-sys/opus/dnn/lpcnet.h:123`, `libopus-sys/opus/dnn/lpcnet.h:132`
- Detail: Rust `lpcnet_compute_single_frame_features` and `lpcnet_compute_single_frame_features_float` take and pass `arch`, matching upstream call-shape semantics.

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

191. [RESOLVED][API Signature][OSCE] `osce_load_models` now mirrors upstream blob-loading entry point.
- Rust: `src/dnn/osce.rs`, `src/opus/opus_decoder.rs`, `tests/dnn_integration.rs`
- Upstream: `libopus-sys/opus/dnn/osce.h:89`
- Detail: Rust `osce_load_models` now takes binary blob input (with builtin fallback on empty input when enabled), and array-based loading is preserved as explicit helper `osce_load_models_from_arrays` for internal call sites.

192. [RESOLVED][API Signature][OSCE] `osce_enhance_frame` and `osce_bwe` already thread upstream `arch` parameter.
- Rust: `src/dnn/osce.rs:2588`, `src/dnn/osce.rs:3421`
- Upstream: `libopus-sys/opus/dnn/osce.h:79`, `libopus-sys/opus/dnn/osce.h:93`
- Detail: Re-audit confirmed both Rust entry points already include and pass through `arch`; this finding was stale.

193. [RESOLVED][API Signature][OSCE] `osce_bwe_reset` now uses upstream-style single-state argument.
- Rust: `src/dnn/osce.rs`, `src/silk/structs.rs`, `src/silk/dec_API.rs`
- Upstream: `libopus-sys/opus/dnn/osce.h:102`, `libopus-sys/opus/dnn/osce.c:1410`
- Detail: Introduced combined `OSCEBWE` (`features + state`) and updated `osce_bwe_reset`/`osce_bwe` plus SILK decoder wiring to pass one OSCE BWE state object, matching upstream `silk_OSCE_BWE_struct*` call-shape.

194. [RESOLVED][Arch Dispatch Coverage][DNN] Upstream RTCD x86/ARM NNet backend surface is now mirrored.
- Rust: `src/dnn/nnet.rs`
- Upstream: `libopus-sys/opus/dnn/x86/dnn_x86.h`, `libopus-sys/opus/dnn/x86/x86_dnn_map.c`, `libopus-sys/opus/dnn/x86/nnet_sse2.c`, `libopus-sys/opus/dnn/x86/nnet_sse4_1.c`, `libopus-sys/opus/dnn/x86/nnet_avx2.c`, `libopus-sys/opus/dnn/arm/nnet_dotprod.c`, `libopus-sys/opus/dnn/arm/nnet_neon.c`, `libopus-sys/opus/dnn/arm/arm_dnn_map.c`
- Detail: Added explicit RTCD backend shims (`x86_rtcd`, `arm_rtcd`) and arch-tier dispatch wrappers for `compute_linear`, `compute_activation`, and `compute_conv2d`, with table-order matching upstream x86/arm maps while preserving existing kernel behavior.

195. [RESOLVED][Validation Semantics][DNN Weights] `linear_init` now mirrors upstream sparse-index structural validation.
- Rust: `src/dnn/nnet.rs`
- Upstream: `libopus-sys/opus/dnn/parse_lpcnet_weights.c:99-120`, `libopus-sys/opus/dnn/parse_lpcnet_weights.c:151`
- Detail: Rust now uses `find_idx_check`-equivalent validation before accepting sparse indices, including stream-shape (`remain < nb_blocks+1`), 4-column alignment (`pos&0x3`), per-index bounds (`pos+3 < nb_in`), and output-group accounting. Added unit tests for short streams, unaligned indices, out-of-bounds indices, and a valid sparse layout.

196. [RESOLVED][Initialization Semantics][DNN Weights] Optional float-weight size mismatches now fail init like upstream.
- Rust: `src/dnn/nnet.rs`
- Upstream: `libopus-sys/opus/dnn/parse_lpcnet_weights.c:92`, `libopus-sys/opus/dnn/parse_lpcnet_weights.c:155-157`, `libopus-sys/opus/dnn/parse_lpcnet_weights.c:163-165`, `libopus-sys/opus/dnn/parse_lpcnet_weights.c:193-195`
- Detail: Rust now uses `opt_array_check`-equivalent validation for optional float arrays in `linear_init` and `conv2d_init`: missing optional arrays are allowed, but present arrays with wrong size cause initialization failure. Added regression tests for dense/sparse `linear_init` and `conv2d_init` mismatch cases.

197. [RESOLVED][Validation Semantics][DNN Weights] `parse_weights` now rejects zero-sized records like upstream.
- Rust: `src/dnn/nnet.rs`
- Upstream: `libopus-sys/opus/dnn/parse_lpcnet_weights.c:52`, `libopus-sys/opus/dnn/parse_lpcnet_weights.c:64`
- Detail: Rust now enforces `size > 0` when parsing each record, matching upstream `ret > 0` acceptance semantics. Added regression coverage for a zero-sized record header.

198. [RESOLVED][Validation Semantics][DNN Weights] Record-name termination check now matches upstream.
- Rust: `src/dnn/nnet.rs`
- Upstream: `libopus-sys/opus/dnn/parse_lpcnet_weights.c:43`
- Detail: Rust now enforces `name[43] == 0` in the 44-byte record name field before accepting a record, matching upstream parse rejection semantics. Added regression coverage for non-terminated name fields.

199. [RESOLVED][API Coverage][DNN NNet] `compute_gated_activation` helper is now mirrored.
- Rust: `src/dnn/nnet.rs`
- Upstream: `libopus-sys/opus/dnn/nnet.h:94`
- Detail: Added `compute_gated_activation` with upstream-equivalent signature and behavior (`output = input * activation(W*input)`), plus unit tests covering linear activation and GLU-equivalent sigmoid behavior.

200. [LOW][API Signature][DNN Weights] `parse_weights` return contract differs from upstream C API.
- Rust: `src/dnn/nnet.rs:591`
- Upstream: `libopus-sys/opus/dnn/nnet.h:97`, `libopus-sys/opus/dnn/parse_lpcnet_weights.c:55-79`
- Detail: Upstream `parse_weights(WeightArray **list, const void *data, int len)` allocates a null-terminated C array and returns the array count (or `-1`). Rust returns `Option<Vec<WeightArray>>` with copied payloads and no C-style sentinel/list ownership contract, so parser API shape and memory/lifecycle semantics are not source-equivalent.

201. [RESOLVED][Builtin Weights Coverage][OSCE] `compiled_weights()` now includes BBWENet arrays for OSCE-BWE model loading.
- Rust: `src/dnn/weights.rs`
- Rust BBWE arrays included from: `src/dnn/bbwenet_data.rs`
- Rust OSCE loader consumes BBWENet weights: `src/dnn/osce.rs`
- Upstream builtin path initializes BBWENet arrays alongside LACE/NoLACE: `libopus-sys/opus/dnn/osce.c:1465-1468`
- Detail: `compiled_weights()` now extends OSCE builtin arrays with `bbwenetlayers_arrays` under the `osce` feature, matching upstream one-shot built-in model aggregation semantics.

202. [RESOLVED][Arch Dispatch Coverage][SILK VAD] Full-function SSE4.1 VAD RTCD path is now mirrored.
- Rust: `src/silk/VAD.rs`, `src/silk/simd/mod.rs`, `src/silk/simd/x86.rs`, `src/silk/float/encode_frame_FLP.rs`
- Upstream: `libopus-sys/opus/silk/x86/x86_silk_map.c:61-69`, `libopus-sys/opus/silk/x86/main_sse.h:248-269`
- Detail: Added function-level `silk_VAD_GetSA_Q8` dispatch wrapper and x86 SSE4.1 entrypoint, and switched encode path call-sites to wrapper dispatch (instead of hard-calling `_c`). This mirrors upstream RTCD call-shape semantics.

203. [RESOLVED][Arch Dispatch Coverage][SILK NSQ] Full-function SSE4.1 NSQ RTCD replacement surface is now mirrored.
- Rust: `src/silk/NSQ.rs`, `src/silk/NSQ_del_dec.rs`, `src/silk/simd/mod.rs`, `src/silk/simd/x86.rs`, `src/silk/float/wrappers_FLP.rs`
- Upstream: `libopus-sys/opus/silk/x86/x86_silk_map.c:72-116`, `libopus-sys/opus/silk/x86/main_sse.h:198-246`
- Detail: Added function-level dispatch wrappers for `silk_NSQ` and `silk_NSQ_del_dec` with x86 SSE4.1/AVX2 tier selection and moved FLP wrapper call-sites to these RTCD wrappers, aligning upstream top-level dispatch surface.

204. [RESOLVED][Arch Dispatch Semantics][CELT x86] `celt_pitch_xcorr` non-AVX2 dispatch now matches upstream scalar fallback.
- Rust: `src/celt/simd/mod.rs`
- Upstream: `libopus-sys/opus/celt/x86/x86_celt_map.c:95-108`
- Detail: Rust now dispatches `celt_pitch_xcorr` to AVX2-only on x86 and uses scalar for lower tiers, matching upstream RTCD table behavior.

205. [RESOLVED][Arch Dispatch Semantics][CELT VQ] `op_pvq_search` now honors caller-provided arch tier.
- Rust: `src/celt/simd/mod.rs`, `src/celt/vq.rs`
- Upstream: `libopus-sys/opus/celt/vq.h:60-64`, `libopus-sys/opus/celt/x86/x86_celt_map.c:175-182`
- Detail: Rust dispatch now uses the passed `arch` tier for SSE2/scalar selection, matching upstream arch-masked RTCD control semantics.

206. [RESOLVED][Initialization Semantics][OSCE] Rust OSCE model loader now requires full model-set initialization for success.
- Rust: `src/dnn/osce.rs`
- Upstream: `libopus-sys/opus/dnn/osce.c:1438-1449`, `libopus-sys/opus/dnn/osce.c:1457-1468`, `libopus-sys/opus/dnn/osce.c:1473-1474`
- Detail: `osce_load_models` now sets `loaded=true` only when all enabled OSCE components (`lace`, `nolace`, `bbwenet`) initialize successfully, matching upstream failure-on-partial-init behavior.

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

210. [RESOLVED][API Signature][PitchDNN] `compute_pitchdnn` includes upstream `arch` parameter.
- Rust: `src/dnn/pitchdnn.rs`
- Upstream: `libopus-sys/opus/dnn/pitchdnn.h:27-31`, `libopus-sys/opus/dnn/pitchdnn.c:12-17`
- Detail: Rust `compute_pitchdnn(..., arch)` now matches upstream signature shape and threads `arch` through DNN compute paths.

211. [RESOLVED][State Layout][PitchDNN] `PitchDNNState` now includes upstream `xcorr_mem3` field.
- Rust: `src/dnn/pitchdnn.rs`
- Upstream: `libopus-sys/opus/dnn/pitchdnn.h:18-20`
- Detail: Added `xcorr_mem3` storage and initialization/reset behavior to mirror upstream state layout.

212. [RESOLVED][Arch Dispatch/API Surface][DNN Core] Core DNN compute APIs thread upstream-style arch control.
- Rust: `src/dnn/nnet.rs`, `src/dnn/simd/mod.rs`, `tests/osce_nndsp.rs`
- Upstream: `libopus-sys/opus/dnn/nnet.h:89-94`, `libopus-sys/opus/dnn/x86/x86_dnn_map.c`, `libopus-sys/opus/dnn/arm/arm_dnn_map.c`
- Detail: Rust DNN compute entry points carry `arch`, dispatch uses this parameter, and tiered C-vs-Rust regression coverage is present (`test_compute_linear_int8_arch_tiers_match_c`).

213. [RESOLVED][Arch Dispatch Coverage][DNN x86] Rust DNN SIMD dispatch now mirrors upstream SSE2/SSE4.1/AVX2 tiers.
- Rust: `src/dnn/simd/mod.rs`, `src/dnn/simd/x86.rs`, `tests/osce_nndsp.rs`, `libopus-sys/src/osce_test_harness.c`
- Upstream: `libopus-sys/opus/dnn/x86/x86_dnn_map.c:46-48`, `libopus-sys/opus/dnn/x86/x86_dnn_map.c:59-61`, `libopus-sys/opus/dnn/x86/x86_dnn_map.c:75-77`
- Detail: Rust x86 DNN dispatch now covers non-AVX2 tiers for float/int8 GEMV and activation EXP paths, with explicit SSE4.1 vs SSE2 behavior for `lpcnet_exp/softmax`. Forced-tier C-vs-Rust regression coverage now includes activation EXP (`test_compute_activation_exp_arch_tiers_match_c`), in addition to int8 linear and conv2d parity checks.

215. [RESOLVED][Algorithmic Path][DNN Conv2D] `compute_conv2d` now mirrors upstream 3x3 specialized convolution branch.
- Rust: `src/dnn/nnet.rs`, `tests/osce_nndsp.rs`
- Upstream: `libopus-sys/opus/dnn/nnet_arch.h:230-233`
- Detail: Rust now selects `conv2d_3x3_float` when `ktime == 3 && kheight == 3`, matching upstream `compute_conv2d_c` branch selection. Added deterministic C-vs-Rust regression `test_compute_conv2d_3x3` (via `osce_test_compute_conv2d_3x3`) to lock bit-exact behavior.

216. [RESOLVED][API Signature][LPCNet] LPCNet feature extraction entry points include upstream `arch` parameter.
- Rust: `src/dnn/lpcnet.rs`
- Upstream: `libopus-sys/opus/dnn/lpcnet_private.h:76`, `libopus-sys/opus/dnn/lpcnet.h:123`, `libopus-sys/opus/dnn/lpcnet.h:132`
- Detail: Rust `compute_frame_features` and the public single-frame helpers take and thread `arch`, matching upstream API-shape semantics.

217. [RESOLVED][Documentation/Versioning][DNN] DNN module documentation now references Opus 1.6.1 baseline.
- Rust: `src/dnn/README.md`
- Upstream baseline being tracked in this repo: `libopus-sys/build.rs:65`, `libopus-sys/build.rs:68`
- Detail: Updated DNN README baseline text to 1.6.1 to match the in-tree vendored/build baseline metadata.

218. [RESOLVED][Validation Semantics][DNN Weights] Sparse index validation in `linear_init` now matches upstream `find_idx_check`.
- Rust: `src/dnn/nnet.rs`
- Upstream: `libopus-sys/opus/dnn/parse_lpcnet_weights.c:99-121`, `libopus-sys/opus/dnn/parse_lpcnet_weights.c:149-152`
- Detail: Rust now performs the same sparse-index validity checks as upstream before accepting `weights_idx`, closing acceptance of malformed sparse index blobs that upstream rejects.

219. [RESOLVED][Boundary Condition][MLP] GRU neuron-capacity assertion now matches upstream fixed-size buffer limit.
- Rust: `src/opus/mlp/layers.rs`
- Upstream: `libopus-sys/opus/src/mlp.h:35`, `libopus-sys/opus/src/mlp.c:97-100`, `libopus-sys/opus/src/mlp.c:101-103`
- Detail: Rust now allows `N == MAX_NEURONS` (`n <= MAX_NEURONS`), matching upstream buffer sizing semantics. Added a regression test that runs GRU compute with exactly 32 neurons.

220. [RESOLVED][Numeric Fidelity][MLP] `tansig_approx` polynomial coefficients now match upstream constants.
- Rust: `src/opus/mlp/tansig.rs`
- Upstream: `libopus-sys/opus/src/mlp.c:41-46`
- Detail: Replaced shortened coefficient literals with upstream-precision constants in `tansig_approx` and kept them exact with a targeted `clippy::excessive_precision` allow on the function.

221. [RESOLVED][Initialization Semantics][LPCNet PLC] Rust `init()` now requires encoder/FARGAN init success before setting loaded state.
- Rust: `src/dnn/lpcnet.rs`
- Rust checked path for comparison: `src/dnn/lpcnet.rs` (`load_model()` requires PLC model, encoder model, and FARGAN init to all succeed)
- Upstream init flow: `libopus-sys/opus/dnn/lpcnet_plc.c:58-67`, `libopus-sys/opus/dnn/lpcnet_plc.c:70`
- Detail: `init()` now leaves `loaded=false` unless PLC model init, encoder model load, and FARGAN init all succeed. Added integration regression coverage (`lpcnet_plc_init_rejects_partial_weights`) to ensure partial model bundles are rejected.

222. [MEDIUM][API Behavior/Version Reporting][Public API] `opus_get_version_string()` does not mirror upstream version-format and build-suffix semantics.
- Rust: `src/celt/common.rs:426-427`
- Upstream implementation: `libopus-sys/opus/celt/celt.c:360-372`
- Upstream API contract notes: `libopus-sys/opus/include/opus_defines.h:852-861`
- Detail: Upstream returns `"libopus " PACKAGE_VERSION` and conditionally appends `-fixed` / `-fuzzing`. Rust intentionally returns `"opurs {crate_version}"`, which remains a deliberate API-behavior divergence from upstream format/suffix signaling semantics.

223. [RESOLVED][Documentation/Versioning][Top-Level Docs] Top-level crate and `libopus-sys` README now reflect 1.6.1 baseline.
- Rust: `src/lib.rs`, `libopus-sys/README.md`
- Upstream baseline being tracked in-tree: `libopus-sys/build.rs:65`, `libopus-sys/build.rs:68`
- Detail: Updated user-facing crate/module docs to 1.6.1, matching in-tree vendored/build baseline metadata.

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

227. [RESOLVED][API Behavior][Public API] `opus_strerror()` now returns upstream-canonical message strings.
- Rust: `src/celt/common.rs`
- Upstream: `libopus-sys/opus/celt/celt.c:344-353`
- Detail: Rust now matches upstream canonical short strings, removing prior numeric-suffix divergence.

228. [RESOLVED][Documentation/Metadata][Package] Crate package metadata now states libopus 1.6.1 compatibility target.
- Rust: `Cargo.toml`
- Upstream baseline being tracked in-tree: `libopus-sys/build.rs:65`, `libopus-sys/build.rs:68`
- Detail: Package description now references 1.6.1, removing stale baseline metadata drift.

229. [HIGH][Feature Behavior][QEXT Encoder] Rust Opus encoder never provisions QEXT extension payload bytes, effectively disabling upstream QEXT bitstream path.
- Rust caller path: `src/opus/opus_encoder.rs:2810-2820`
- Rust CELT entry shape: `src/celt/celt_encoder.rs:2119-2127`, `src/celt/celt_encoder.rs:3283`
- Upstream behavior: `libopus-sys/opus/celt/celt_encoder.c:2535-2590`, `libopus-sys/opus/celt/celt_encoder.c:2816-2818`
- Detail: Upstream `celt_encode_with_ec` computes `qext_bytes`, carves extension payload space from the packet, and entropy-codes QEXT data when `st->enable_qext` is set. Rust currently calls `celt_encode_with_ec` with `qext_payload=None` and `qext_bytes=0` (TODO noted), so QEXT coding branches guarded by `qext_bytes > 0` never activate, diverging from upstream QEXT-enabled encoding behavior.

230. [RESOLVED][Arch Dispatch Semantics][CELT/SILK SIMD] Runtime SIMD dispatch honors upstream arch-tier control path.
- Rust: `src/celt/simd/mod.rs`, `src/silk/simd/mod.rs`
- Upstream: `libopus-sys/opus/celt/x86/pitch_sse.h`, `libopus-sys/opus/celt/x86/vq_sse.h`, `libopus-sys/opus/silk/x86/main_sse.h`
- Detail: Rust CELT/SILK SIMD wrappers now use the threaded `arch` tier for dispatch decisions instead of host-only CPUID checks, matching upstream arch-masked control semantics.

231. [RESOLVED][SIMD Coverage/Semantics][SILK FLP] `silk_inner_product_FLP` override now matches upstream x86-only AVX2 behavior.
- Rust: `src/silk/simd/mod.rs`, `src/silk/float/inner_product_FLP.rs`
- Upstream default macro path (scalar): `libopus-sys/opus/silk/float/SigProc_FLP.h:132-133`
- Upstream SIMD override path (x86 AVX2 only): `libopus-sys/opus/silk/x86/main_sse.h:279-293`, `libopus-sys/opus/silk/x86/x86_silk_map.c:165-175`
- Detail: Removed the aarch64 NEON dispatch override for `silk_inner_product_FLP`; Rust now uses AVX2 override on x86 and scalar fallback elsewhere, matching upstream override coverage.

232. [LOW][Build SIMD Flags][libopus-sys x86 AVX2] `libopus-sys/build.rs` uses weaker AVX2 flags for `SILK_SOURCES_AVX2` than upstream build scripts.
- Rust build flags: `libopus-sys/build.rs:169-173`
- Upstream CMake AVX2 flags for same source group: `libopus-sys/opus/CMakeLists.txt:528`, `libopus-sys/opus/CMakeLists.txt:531`
- Detail: Upstream applies `-mavx2 -mfma -mavx` to `silk_sources_avx2`; `build.rs` applies only `-mavx2` for `SILK_SOURCES_AVX2` (while using `-mavx2 -mfma` for `SILK_SOURCES_FLOAT_AVX2`). This is a compile-flag parity drift in SIMD build configuration.

233. [RESOLVED][SIMD Coverage][SILK VAD] Rust now exposes the upstream-style full-function SSE4.1 override surface.
- Rust: `src/silk/VAD.rs`, `src/silk/simd/mod.rs`, `src/silk/simd/x86.rs`
- Upstream SIMD override: `libopus-sys/opus/silk/x86/main_sse.h:250-269`, `libopus-sys/opus/silk/x86/VAD_sse4_1.c:45-260`
- Detail: Rust now dispatches `silk_VAD_GetSA_Q8` through an x86 SSE4.1 full-function entrypoint selected by runtime arch tier, rather than bypassing RTCD at call sites.

234. [RESOLVED][SIMD Coverage][DNN x86] Rust DNN runtime dispatch no longer AVX2-only and now matches upstream tiering.
- Rust dispatch: `src/dnn/simd/mod.rs`
- Rust x86 kernels: `src/dnn/simd/x86.rs`
- Upstream x86 RTCD tables/macros: `libopus-sys/opus/dnn/x86/x86_dnn_map.c:39-49`, `libopus-sys/opus/dnn/x86/x86_dnn_map.c:51-62`, `libopus-sys/opus/dnn/x86/x86_dnn_map.c:64-78`, `libopus-sys/opus/dnn/x86/dnn_x86.h:82-114`
- Detail: Rust now dispatches x86 DNN kernels across `AVX2 -> SSE4.1 -> SSE2 -> scalar` for the activation EXP path and uses SSE2/SSE4.1-aware behavior across DNN linear paths, removing the prior non-AVX2 scalar fallback gap.

235. [RESOLVED][Arch Dispatch Semantics][DNN] DNN RTCD selection semantics now mirror upstream arch-tier behavior.
- Rust: `src/dnn/simd/mod.rs`, `src/dnn/vec.rs`, `tests/osce_nndsp.rs`, `libopus-sys/src/osce_test_harness.c`
- Upstream: `libopus-sys/opus/dnn/x86/dnn_x86.h`, `libopus-sys/opus/dnn/arm/dnn_arm.h`, `libopus-sys/opus/dnn/vec_neon.h`, `libopus-sys/opus/dnn/vec_avx.h`
- Detail: Rust DNN dispatch uses threaded `arch` tier, matches upstream aarch64 compile-time-NEON low-tier semantics, and has explicit forced-tier C-vs-Rust regression coverage (`osce_test_compute_linear_int8_arch` / `test_compute_linear_int8_arch_tiers_match_c`).

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
