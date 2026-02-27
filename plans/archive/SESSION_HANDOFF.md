# Session Handoff (2026-02-23)

## Repo State
- Branch: `main`
- HEAD when this handoff was updated: `6599dfa` (`celt: match x86 float2int rounding semantics`)
- Worktree at handoff time: dirty (WIP parity-debug changes, see staged commit below)

## CI Snapshot
- Latest main CI run: `22314740430`
  - Commit: `6599dfa`
  - URL: https://github.com/dignifiedquire/opurs/actions/runs/22314740430
  - Status: `completed`
  - Conclusion: `failure`
- High-signal failures observed in this run:
  - `Clippy`
  - `Clippy tools-dnn`
  - `Clippy DNN (osce)`
  - `Clippy DNN (deep-plc)`
  - `Clippy DNN (dnn,builtin-weights)`
  - `Tools tests smoke (linux-x86_64)`

## x86 Vector/Parity Investigation Completed In This Session
- Reproduced deterministic encode mismatch on x86 classic vectors:
  - `run_vectors2 ... testvector11 ... ENC 020kbps app=audio frame=20ms`
  - First mismatch around packet `235` (`len 67/68`)
- Verified isolation:
  - `--no-default-features --features tools` (SIMD off) makes the failing encode case pass.
  - Confirms SIMD involvement.
- Probed CELT/SILK dispatch paths by selectively disabling/re-enabling SIMD callsites:
  - Single-function CELT toggles did not remove `pkt=235` mismatch.
  - Disabling CELT `op_pvq_search` changed failure location to earlier packet (~`72`), indicating material effect but not full root cause.
  - SILK SIMD toggles tested had no effect on the `pkt=235` case.
- Added/ran internal parity checks in `src/celt/simd/x86.rs` against upstream C RTCD function-pointer tables:
  - `op_pvq_search`, `xcorr_kernel`, `celt_inner_prod`, `dual_inner_prod`, `comb_filter_const`, `celt_pitch_xcorr`
  - Found and fixed one real mismatch: SSE comb filter tail behavior.
  - After fix, kernel-level parity checks pass against upstream C implementations.
- Added a windowed debug test to localize stateful divergence build-up:
  - `tests/audio_20k_window_debug.rs` (`#[ignore]`)
  - Mismatch appears only after enough history accumulates, consistent with stateful drift.

## Key Architecture Finding
- Upstream C arch selection on current x86 machine reports `c_arch=3` (SSE4.1 tier).
- Rust currently still returns `0` in:
  - `src/opus/opus_encoder.rs` (`cpu_support_h::opus_select_arch`)
  - `src/opus/analysis.rs` (`cpu_support_h::opus_select_arch`)
- Decoder validation still assumes scalar-only arch:
  - `src/celt/celt_decoder.rs` has `assert!(st.arch <= 0)`
- This is still the most likely systemic source of SIMD/arch semantic drift.

## Current WIP Files (to be committed)
- `src/celt/float_cast.rs`
- `src/celt/simd/mod.rs`
- `src/celt/simd/x86.rs`
- `libopus-sys/src/lib.rs`
- `tests/audio_20k_window_debug.rs`
- `tests/pitch_c_parity.rs`
- `tests/pvq_sse2_parity.rs`
- `tests/rld_prediction_debug.rs`
- `SESSION_HANDOFF.md`

## Suggested Next Fix Path (No-Drift)
1. Implement proper Rust arch plumbing to mirror upstream C selection semantics.
2. Remove/relax scalar-only arch assumption in CELT decoder validation.
3. Re-run failing x86 vectors and compare first divergence packet.
4. Keep only parity-meaningful tests; drop temporary diagnostics once root cause is fixed.

## Commands To Resume
- Inspect a run:
  - `gh run view 22314740430 --json status,conclusion,jobs,url`
- Dump failed logs for a job:
  - `gh run view 22314740430 --job <job_id> --log-failed`
- Repro local failing vector case:
  - `CARGO_TARGET_DIR=target-local-x86 cargo run --release --features tools --example run_vectors2 -- --suite classic --matrix full --jobs 1 --vector-name testvector11 --kind-name "ENC 020kbps app=audio frame=20ms" opus_newvectors`
