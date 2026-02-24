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

    /// True for NEON or higher (aarch64).
    #[cfg(target_arch = "aarch64")]
    #[inline]
    pub fn has_neon(self) -> bool {
        matches!(self, Self::Neon | Self::DotProd)
    }

    /// True for DOTPROD (aarch64).
    #[cfg(target_arch = "aarch64")]
    #[inline]
    pub fn has_dotprod(self) -> bool {
        matches!(self, Self::DotProd)
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
        // Mirror upstream x86cpu.c:opus_select_arch_impl():
        //   check SSE → SSE2 → SSE4.1 → AVX2+FMA, return highest.
        cpufeatures::new!(detect_avx2_fma, "avx2", "fma");
        cpufeatures::new!(detect_sse4_1, "sse4.1");
        cpufeatures::new!(detect_sse2, "sse2");
        cpufeatures::new!(detect_sse, "sse");

        if detect_avx2_fma::get() {
            return Arch::Avx2;
        }
        if detect_sse4_1::get() {
            return Arch::Sse4_1;
        }
        if detect_sse2::get() {
            return Arch::Sse2;
        }
        if detect_sse::get() {
            return Arch::Sse;
        }
        return Arch::Scalar;
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
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return Arch::DotProd;
        }
        return Arch::Neon;
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
