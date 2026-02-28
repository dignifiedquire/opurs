//! CPU architecture detection and SIMD dispatch support.
//!
//! Mirrors upstream C RTCD (Run-Time Code Dispatch) architecture levels.
//! Detection happens once at encoder/decoder init via [`opus_select_arch()`],
//! and the result is stored in state and threaded through all dispatch calls.
//!
//! Upstream C: `celt/cpu_support.h`, `celt/x86/x86cpu.c`, `celt/arm/armcpu.c`

/// CPU architecture level for SIMD dispatch, mirroring upstream C RTCD.
///
/// Variants are platform-gated: x86 ISA variants only exist on x86/x86_64,
/// ARM variants only on aarch64. [`Scalar`](Arch::Scalar) is always available.
///
/// C RTCD arch levels for reference:
/// - x86:     0=none, 1=SSE, 2=SSE2, 3=SSE4.1, 4=AVX2
/// - aarch64: 0=none, ..., 3=NEON, 4=DOTPROD
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arch {
    /// No SIMD — scalar fallback. All platforms.
    #[default]
    Scalar,

    // -- x86 / x86_64 --
    /// x86 SSE (128-bit float SIMD). C arch level 1.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse,
    /// x86 SSE2 (128-bit integer SIMD). C arch level 2.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2,
    /// x86 SSE4.1 (extended integer SIMD). C arch level 3.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse4_1,
    /// x86 AVX2 + FMA (256-bit SIMD). C arch level 4.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,

    // -- aarch64 --
    /// ARM NEON (128-bit SIMD, always available on aarch64). C arch level 3.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// ARM NEON + DOTPROD (dot product instructions). C arch level 4.
    #[cfg(target_arch = "aarch64")]
    DotProd,

    // -- arm (32-bit) --
    /// ARM EDSP. C arch level 1.
    #[cfg(target_arch = "arm")]
    Edsp,
    /// ARM MEDIA. C arch level 2.
    #[cfg(target_arch = "arm")]
    Media,
    /// ARM NEON. C arch level 3.
    #[cfg(target_arch = "arm")]
    Neon,
}

impl Arch {
    // -- x86 / x86_64 helpers --

    /// True for any x86 SIMD level (SSE or higher).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn has_sse(self) -> bool {
        matches!(self, Self::Sse | Self::Sse2 | Self::Sse4_1 | Self::Avx2)
    }

    /// True for SSE2 or higher (x86).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn has_sse2(self) -> bool {
        matches!(self, Self::Sse2 | Self::Sse4_1 | Self::Avx2)
    }

    /// True for SSE4.1 or higher (x86).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn has_sse4_1(self) -> bool {
        matches!(self, Self::Sse4_1 | Self::Avx2)
    }

    /// True for AVX2 (x86).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn has_avx2(self) -> bool {
        matches!(self, Self::Avx2)
    }

    // -- aarch64 helpers --

    /// True for NEON or higher (ARM/aarch64).
    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    #[inline]
    pub fn has_neon(self) -> bool {
        #[cfg(target_arch = "aarch64")]
        {
            matches!(self, Self::Neon | Self::DotProd)
        }
        #[cfg(target_arch = "arm")]
        {
            matches!(self, Self::Neon)
        }
    }

    /// True for DOTPROD (aarch64).
    #[cfg(target_arch = "aarch64")]
    #[inline]
    pub fn has_dotprod(self) -> bool {
        matches!(self, Self::DotProd)
    }
}

#[cfg(all(feature = "simd", feature = "fuzzing"))]
fn fuzz_random_u32() -> u32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static SEED: AtomicU64 = AtomicU64::new(0);

    let mut seed = SEED.load(Ordering::Relaxed);
    if seed == 0 {
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        seed = t ^ ((&SEED as *const AtomicU64 as usize) as u64);
    }

    loop {
        let next = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        match SEED.compare_exchange_weak(seed, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return (next >> 32) as u32,
            Err(cur) => seed = cur,
        }
    }
}

#[cfg(all(
    feature = "simd",
    feature = "fuzzing",
    any(target_arch = "x86", target_arch = "x86_64")
))]
fn fuzz_downgrade_arch(arch: Arch) -> Arch {
    let max = match arch {
        Arch::Scalar => 0,
        Arch::Sse => 1,
        Arch::Sse2 => 2,
        Arch::Sse4_1 => 3,
        Arch::Avx2 => 4,
    };
    let arch = fuzz_random_u32() % (max + 1);
    match arch {
        0 => Arch::Scalar,
        1 => Arch::Sse,
        2 => Arch::Sse2,
        3 => Arch::Sse4_1,
        4 => Arch::Avx2,
        _ => Arch::Scalar,
    }
}

#[cfg(all(feature = "simd", feature = "fuzzing", target_arch = "aarch64"))]
fn fuzz_downgrade_arch(arch: Arch) -> Arch {
    let max = match arch {
        Arch::Scalar => 0,
        Arch::Neon => 3,
        Arch::DotProd => 4,
    };
    let arch = fuzz_random_u32() % (max + 1);
    match arch {
        4 => Arch::DotProd,
        3 => Arch::Neon,
        _ => Arch::Scalar,
    }
}

#[cfg(all(feature = "simd", feature = "fuzzing", target_arch = "arm"))]
fn fuzz_downgrade_arch(arch: Arch) -> Arch {
    let max = match arch {
        Arch::Scalar => 0,
        Arch::Edsp => 1,
        Arch::Media => 2,
        Arch::Neon => 3,
    };
    let arch = fuzz_random_u32() % (max + 1);
    match arch {
        0 => Arch::Scalar,
        1 => Arch::Edsp,
        2 => Arch::Media,
        3 => Arch::Neon,
        _ => Arch::Scalar,
    }
}

/// Detect the highest supported SIMD architecture at runtime.
///
/// Mirrors upstream C `opus_select_arch()` from `celt/x86/x86cpu.c` and
/// `celt/arm/armcpu.c`. Called once at encoder/decoder initialization;
/// the result is stored in state and passed through dispatch calls.
///
/// When the `simd` feature is disabled, always returns [`Arch::Scalar`].
#[cfg(feature = "simd")]
pub fn opus_select_arch() -> Arch {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // Use Rust's safe x86 runtime feature detection.
        let hw_sse = std::is_x86_feature_detected!("sse");
        let hw_sse2 = std::is_x86_feature_detected!("sse2");
        let hw_sse4_1 = std::is_x86_feature_detected!("sse4.1");
        let hw_avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");

        let mut arch = Arch::Scalar;
        if hw_sse {
            arch = Arch::Sse;
        } else {
            return arch;
        }

        if hw_sse2 {
            arch = Arch::Sse2;
        } else {
            return arch;
        }

        if hw_sse4_1 {
            arch = Arch::Sse4_1;
        } else {
            return arch;
        }

        if hw_avx2 {
            arch = Arch::Avx2;
        }

        #[cfg(feature = "fuzzing")]
        {
            arch = fuzz_downgrade_arch(arch);
        }
        return arch;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Mirror upstream armcpu.c:opus_select_arch_impl():
        //   NEON is always available on aarch64.
        //   DOTPROD detected via platform-specific methods:
        //     Linux:   /proc/cpuinfo "asimddp" flag
        //     macOS:   sysctlbyname("hw.optional.arm.FEAT_DotProd")
        //     FreeBSD: elf_aux_info AT_HWCAP / HWCAP_ASIMDDP
        //
        // Rust's std::arch mirrors these platform-specific detection methods.
        let arch = if std::arch::is_aarch64_feature_detected!("dotprod") {
            Arch::DotProd
        } else {
            Arch::Neon
        };
        #[cfg(feature = "fuzzing")]
        let arch = fuzz_downgrade_arch(arch);
        return arch;
    }

    #[cfg(target_arch = "arm")]
    {
        // Mirror upstream armcpu.c Linux logic for non-aarch64 ARM:
        // - Parse /proc/cpuinfo "Features" line for edsp/neon/asimd.
        // - Parse "CPU architecture:" and set MEDIA for version >= 6.
        let mut has_edsp = false;
        let mut has_media = false;
        let mut has_neon = false;

        #[cfg(target_os = "linux")]
        {
            if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in cpuinfo.lines() {
                    if let Some(features) = line.strip_prefix("Features") {
                        let f = features;
                        if f.contains(" edsp") {
                            has_edsp = true;
                        }
                        if f.contains(" neon") {
                            has_neon = true;
                        }
                        // Keep parity with upstream's aarch64 token handling.
                        if f.contains(" asimd") {
                            has_edsp = true;
                            has_media = true;
                            has_neon = true;
                        }
                    } else if let Some(v) = line.strip_prefix("CPU architecture:") {
                        if v.trim().parse::<i32>().map(|x| x >= 6).unwrap_or(false) {
                            has_media = true;
                        }
                    }
                }
            }
        }

        let mut arch = Arch::Scalar;
        if has_edsp {
            arch = Arch::Edsp;
        } else {
            return arch;
        }

        if has_media {
            arch = Arch::Media;
        } else {
            return arch;
        }

        if has_neon {
            arch = Arch::Neon;
        } else {
            return arch;
        }

        #[cfg(feature = "fuzzing")]
        let arch = fuzz_downgrade_arch(arch);

        return arch;
    }

    #[allow(unreachable_code)]
    Arch::Scalar
}

/// When `simd` feature is disabled, always returns [`Arch::Scalar`].
#[cfg(not(feature = "simd"))]
pub fn opus_select_arch() -> Arch {
    Arch::Scalar
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "aarch64")]
    use super::Arch;

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn dotprod_implies_neon() {
        assert!(Arch::DotProd.has_dotprod());
        assert!(Arch::DotProd.has_neon());
        assert!(Arch::Neon.has_neon());
        assert!(!Arch::Neon.has_dotprod());
    }
}
