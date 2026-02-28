# Plan: Remove Unnecessary Unsafe (Main Audit)

Last updated: 2026-02-28  
Branch: `main`  
Commit: `5312c9f9`

## Goal

Eliminate unnecessary `unsafe` usage and confine required `unsafe` to explicit, justified boundaries:

- SIMD intrinsic implementations
- FFI boundaries (tooling/tests only where unavoidable)

Target policy after cleanup:

- No `unsafe` in non-SIMD production paths.
- No call-site `unsafe` in dispatch layers.
- All remaining `unsafe` blocks have explicit `// SAFETY:` justification.
- CI guard prevents unsafe drift.

## Current inventory (full)

Counts come from `rg -n "\\bunsafe\\b" src` at `5312c9f9`.

| File | unsafe total | unsafe fn | unsafe blocks | unsafe extern | Category | Priority |
|---|---:|---:|---:|---:|---|---|
| `src/arch.rs` | 3 | 0 | 3 | 0 | Core non-SIMD | P0 |
| `src/celt/common.rs` | 2 | 0 | 2 | 0 | Core non-SIMD | P0 |
| `src/celt/float_cast.rs` | 3 | 0 | 3 | 0 | Core non-SIMD (arch intrinsics) | P1 |
| `src/silk/NSQ.rs` | 1 | 0 | 1 | 0 | Core non-SIMD (SIMD dispatch callsite) | P0 |
| `src/silk/NSQ_del_dec.rs` | 4 | 0 | 4 | 0 | Core non-SIMD (SIMD dispatch callsite) | P0 |
| `src/celt/simd/mod.rs` | 9 | 0 | 9 | 0 | SIMD dispatch | P1 |
| `src/dnn/simd/mod.rs` | 30 | 0 | 30 | 0 | SIMD dispatch | P1 |
| `src/silk/simd/mod.rs` | 17 | 5 | 12 | 0 | SIMD dispatch | P1 |
| `src/celt/simd/aarch64.rs` | 4 | 4 | 0 | 0 | SIMD intrinsics | P2 |
| `src/celt/simd/x86.rs` | 35 | 8 | 20 | 7 | SIMD intrinsics + test interop | P2 |
| `src/dnn/simd/aarch64.rs` | 21 | 19 | 2 | 0 | SIMD intrinsics | P2 |
| `src/dnn/simd/x86.rs` | 31 | 29 | 2 | 0 | SIMD intrinsics | P2 |
| `src/silk/simd/aarch64.rs` | 21 | 20 | 1 | 0 | SIMD intrinsics | P2 |
| `src/silk/simd/x86.rs` | 39 | 38 | 1 | 0 | SIMD intrinsics | P2 |
| `src/opus/analysis.rs` | 2 | 0 | 1 | 1 | Test interop FFI (`cfg(test, tools)`) | P3 |
| `src/silk/float/residual_energy_FLP.rs` | 3 | 0 | 2 | 1 | Test interop FFI (`cfg(test, tools)`) | P3 |
| `src/tools/demo/backend.rs` | 54 | 0 | 53 | 0 | Tooling FFI backend | P3 |

Totals:

- 17 files
- 279 `unsafe` matches
- 123 `unsafe fn`
- 146 `unsafe {}` blocks
- 9 `unsafe extern` declarations

Category rollup:

- Core non-SIMD: 13
- SIMD dispatch: 56
- SIMD intrinsics: 151
- Test interop FFI: 5
- Tooling FFI backend: 54

## Hotspots and concrete fix directions

### P0 — Core non-SIMD unsafe (must go first)

Completed:

- `src/celt/bands.rs`: 0 `unsafe` (raw-pointer qext context removed; snapshot/restore added for theta-rdo rollback).

1. `src/celt/common.rs`
- Current issue: in-place comb filter uses `from_raw_parts` + `from_raw_parts_mut` alias construction.
- Fix direction:
  - Introduce safe in-place SIMD entrypoint (single `&mut [f32]` + indices) in `src/celt/simd/mod.rs`.
  - Keep any required pointer aliasing internal to SIMD backend only.
- Exit criteria: 0 `unsafe` in `src/celt/common.rs`.

2. `src/silk/NSQ.rs`, `src/silk/NSQ_del_dec.rs`
- Current issue: call-site `unsafe` to invoke SIMD routines.
- Fix direction:
  - Make SIMD dispatch wrappers safe, moving unsafe to backend internals.
- Exit criteria: 0 `unsafe` in both files.

3. `src/arch.rs`
- Current issue: direct `unsafe` CPUID intrinsics.
- Fix direction:
  - Replace with safe feature-detection strategy where possible (`is_x86_feature_detected!` + equivalent logic).
  - If direct CPUID remains necessary for strict upstream parity, confine to one helper with full `// SAFETY:` and no duplicated call-site unsafe.
- Exit criteria: either 0 unsafe, or one tightly-scoped justified helper.

### P1 — SIMD dispatch layer cleanup

Files:

- `src/celt/simd/mod.rs`
- `src/dnn/simd/mod.rs`
- `src/silk/simd/mod.rs`
- `src/celt/float_cast.rs` (depends on chosen boundary)

Fix direction:

- Remove call-site `unsafe` from dispatch functions.
- Make public dispatch APIs safe.
- Push all intrinsic-unsafe code behind private/internal backend boundaries.
- For `float_cast`, choose one of:
  - Keep intrinsic impl and isolate unsafe in one private helper.
  - Move to safe fallback only if bit-exactness parity is preserved (must be proven by vectors).

Exit criteria:

- `simd/mod.rs` layers have no `unsafe` usage.
- Any remaining `unsafe` in `float_cast` is isolated and documented.

### P2 — SIMD intrinsic modules (retain only necessary unsafe)

Files:

- `src/celt/simd/{x86,aarch64}.rs`
- `src/dnn/simd/{x86,aarch64}.rs`
- `src/silk/simd/{x86,aarch64}.rs`

Fix direction:

- Minimize `pub unsafe fn` surface:
  - Prefer safe public wrappers with feature-gated dispatch.
  - Keep unsafe in private kernels when target-feature/ptr invariants require it.
- Add/standardize `// SAFETY:` comments for each remaining `unsafe` block.
- Ensure no unchecked memory invariants are silently assumed.

Exit criteria:

- No unnecessary public `unsafe fn` exports in SIMD modules.
- Remaining unsafe sites are intentional, documented, and tested.

### P3 — Tool/test FFI consolidation

Files:

- `src/tools/demo/backend.rs`
- `src/opus/analysis.rs` (`cfg(test, feature = "tools")`)
- `src/silk/float/residual_energy_FLP.rs` (`cfg(test, feature = "tools")`)
- Test interop in `src/celt/simd/x86.rs`

Fix direction:

- Consolidate repeated FFI call-site unsafe into small wrapper helpers.
- Keep `unsafe extern` declarations in one place per module/test helper.
- Explicitly mark these as tooling/test-only in policy.

Exit criteria:

- Tooling/test FFI unsafe is centralized and minimal.

## Execution plan (commit order)

1. `plan: refresh unsafe inventory and remediation phases`
- Update this file and add `scripts/unsafe_inventory.sh` for deterministic counts.

2. `refactor(celt): remove aliasing unsafe from common`
- `src/celt/common.rs`

3. `refactor(silk): remove NSQ call-site unsafe via safe SIMD wrappers`
- `src/silk/NSQ.rs`
- `src/silk/NSQ_del_dec.rs`
- `src/silk/simd/mod.rs`

4. `refactor(simd): eliminate dispatch-layer unsafe`
- `src/celt/simd/mod.rs`
- `src/dnn/simd/mod.rs`
- `src/silk/simd/mod.rs`
- `src/celt/float_cast.rs` (if in scope)

5. `refactor(tools/tests): centralize FFI unsafe`
- `src/tools/demo/backend.rs`
- tool/test interop modules

6. `ci(safety): enforce unsafe policy`
- Add CI check script/step for unsafe drift.
- Add lint gates (`unsafe_op_in_unsafe_fn`, `undocumented_unsafe_blocks`).

## Verification requirements after each phase

Always run:

- `cargo fmt --all`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo nextest run --all-features`

Plus parity checks:

- Required vector jobs (all major platforms).
- Bit-exact comparison suites against upstream C where already wired.

Unsafe policy checks:

- `rg -n "\\bunsafe\\b" src` count must not increase.
- `rg -n "\\bunsafe\\b" src/celt src/silk src/opus` should trend down in P0/P1.

## Definition of done

- Non-SIMD production code is free of `unsafe`.
- SIMD dispatch layers are safe.
- Remaining `unsafe` is only in justified SIMD kernels and unavoidable FFI boundaries.
- CI blocks new unsafe drift.
