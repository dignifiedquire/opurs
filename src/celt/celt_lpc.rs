//! Linear prediction coefficient computation.
//!
//! Upstream C: `celt/celt_lpc.c`

use crate::arch::Arch;
use crate::celt::pitch::{celt_pitch_xcorr, xcorr_kernel};

pub const LPC_ORDER: usize = 24;

/// Levinson-Durbin LPC analysis.
///
/// Computes `lpc.len()` LPC coefficients from autocorrelation values `ac`.
/// `ac` must have at least `lpc.len() + 1` elements.
///
/// Upstream C: celt/celt_lpc.c:_celt_lpc
#[inline]
pub fn _celt_lpc(lpc: &mut [f32], ac: &[f32]) {
    let p = lpc.len();
    debug_assert!(ac.len() > p);
    lpc.fill(0.0);
    if ac[0] > 1e-10f32 {
        let mut error = ac[0];
        for i in 0..p {
            let mut rr = 0.0f32;
            for j in 0..i {
                unsafe {
                    rr += *lpc.get_unchecked(j) * *ac.get_unchecked(i - j);
                }
            }
            rr += ac[i + 1];
            let r = -(rr / error);
            lpc[i] = r;
            for j in 0..((i + 1) >> 1) {
                unsafe {
                    let tmp1 = *lpc.get_unchecked(j);
                    let tmp2 = *lpc.get_unchecked(i - 1 - j);
                    *lpc.get_unchecked_mut(j) = tmp1 + r * tmp2;
                    *lpc.get_unchecked_mut(i - 1 - j) = tmp2 + r * tmp1;
                }
            }
            error -= r * r * error;
            if error <= 0.001f32 * ac[0] {
                break;
            }
        }
    }
}

/// FIR filter.
///
/// `x` must contain at least `N + ord` elements, where the first `ord`
/// elements are history (accessed as negative offsets in the C version).
/// The filter reads `x[0..N+ord]` and writes `N` samples to `y`.
/// `num` has `ord` coefficients.
///
/// Upstream C: celt/celt_lpc.c:celt_fir_c
#[inline]
pub fn celt_fir_c(x: &[f32], num: &[f32], y: &mut [f32], ord: usize, arch: Arch) {
    let n = y.len();
    debug_assert!(x.len() >= n + ord);
    debug_assert!(num.len() >= ord);

    // Reverse the numerator coefficients
    let mut rnum = [0.0f32; LPC_ORDER];
    for i in 0..ord {
        unsafe {
            *rnum.get_unchecked_mut(i) = *num.get_unchecked(ord - i - 1);
        }
    }

    let mut i = 0i32;
    while i < n as i32 - 3 {
        let ix = i as usize;
        let mut sum = [0.0f32; 4];
        // x is indexed with ord offset: x[ord + ix] is the "current" sample
        unsafe {
            sum[0] = *x.get_unchecked(ord + ix);
            sum[1] = *x.get_unchecked(ord + ix + 1);
            sum[2] = *x.get_unchecked(ord + ix + 2);
            sum[3] = *x.get_unchecked(ord + ix + 3);
        }
        xcorr_kernel(&rnum, &x[ix..], &mut sum, ord, arch);
        unsafe {
            *y.get_unchecked_mut(ix) = sum[0];
            *y.get_unchecked_mut(ix + 1) = sum[1];
            *y.get_unchecked_mut(ix + 2) = sum[2];
            *y.get_unchecked_mut(ix + 3) = sum[3];
        }
        i += 4;
    }
    while (i as usize) < n {
        let ix = i as usize;
        let mut sum = unsafe { *x.get_unchecked(ord + ix) };
        for j in 0..ord {
            unsafe {
                sum += *rnum.get_unchecked(j) * *x.get_unchecked(ix + j);
            }
        }
        unsafe {
            *y.get_unchecked_mut(ix) = sum;
        }
        i += 1;
    }
}

/// IIR filter (in-place).
///
/// Filters `buf[0..n]` in-place using `den` (length `ord`) denominator
/// coefficients. `mem` (length `ord`) holds filter state and is updated
/// on return.
///
/// Upstream C: celt/celt_lpc.c:celt_iir
#[inline]
pub fn celt_iir(buf: &mut [f32], n: usize, den: &[f32], ord: usize, mem: &mut [f32], arch: Arch) {
    debug_assert!(buf.len() >= n);
    debug_assert!(den.len() >= ord);
    debug_assert!(mem.len() >= ord);
    debug_assert!(ord & 3 == 0);

    // Reverse the denominator coefficients
    let mut rden = [0.0f32; LPC_ORDER];
    for i in 0..ord {
        unsafe {
            *rden.get_unchecked_mut(i) = *den.get_unchecked(ord - i - 1);
        }
    }

    // Upstream C allocates `N + ord` here (`celt/celt_lpc.c:celt_iir`).
    // Use the same sizing to avoid fixed-size truncation on malformed inputs.
    let mut yy = vec![0.0f32; n + ord];
    for i in 0..ord {
        unsafe {
            *yy.get_unchecked_mut(i) = -*mem.get_unchecked(ord - i - 1);
        }
    }

    let mut i = 0i32;
    while i < n as i32 - 3 {
        let ix = i as usize;
        let mut sum = [0.0f32; 4];
        unsafe {
            sum[0] = *buf.get_unchecked(ix);
            sum[1] = *buf.get_unchecked(ix + 1);
            sum[2] = *buf.get_unchecked(ix + 2);
            sum[3] = *buf.get_unchecked(ix + 3);
        }
        xcorr_kernel(&rden, &yy[ix..], &mut sum, ord, arch);
        unsafe {
            *yy.get_unchecked_mut(ix + ord) = -sum[0];
            *buf.get_unchecked_mut(ix) = sum[0];
            sum[1] += *yy.get_unchecked(ix + ord) * den[0];
            *yy.get_unchecked_mut(ix + ord + 1) = -sum[1];
            *buf.get_unchecked_mut(ix + 1) = sum[1];
            sum[2] += *yy.get_unchecked(ix + ord + 1) * den[0];
            sum[2] += *yy.get_unchecked(ix + ord) * den[1];
            *yy.get_unchecked_mut(ix + ord + 2) = -sum[2];
            *buf.get_unchecked_mut(ix + 2) = sum[2];
            sum[3] += *yy.get_unchecked(ix + ord + 2) * den[0];
            sum[3] += *yy.get_unchecked(ix + ord + 1) * den[1];
            sum[3] += *yy.get_unchecked(ix + ord) * den[2];
            *yy.get_unchecked_mut(ix + ord + 3) = -sum[3];
            *buf.get_unchecked_mut(ix + 3) = sum[3];
        }
        i += 4;
    }
    while (i as usize) < n {
        let ix = i as usize;
        let mut sum = unsafe { *buf.get_unchecked(ix) };
        for j in 0..ord {
            unsafe {
                sum -= *rden.get_unchecked(j) * *yy.get_unchecked(ix + j);
            }
        }
        unsafe {
            *yy.get_unchecked_mut(ix + ord) = sum;
            *buf.get_unchecked_mut(ix) = sum;
        }
        i += 1;
    }
    for i in 0..ord {
        unsafe {
            *mem.get_unchecked_mut(i) = *buf.get_unchecked(n - i - 1);
        }
    }
}

/// Autocorrelation.
///
/// Computes `lag + 1` autocorrelation values from `x` (length `n`),
/// optionally applying `window` (length `overlap`) at both ends.
/// Results are written to `ac[0..=lag]`.
///
/// Upstream C: celt/celt_lpc.c:_celt_autocorr
#[inline]
pub fn _celt_autocorr(
    x: &[f32],
    ac: &mut [f32],
    window: Option<&[f32]>,
    overlap: usize,
    lag: usize,
    arch: Arch,
) -> i32 {
    let n = x.len();
    let fast_n = n - lag;
    debug_assert!(n > 0);
    debug_assert!(ac.len() > lag);

    let mut xx: Vec<f32>;
    let xptr: &[f32];

    if let Some(win) = window {
        debug_assert!(win.len() >= overlap);
        xx = x.to_vec();
        for i in 0..overlap {
            unsafe {
                *xx.get_unchecked_mut(i) = *x.get_unchecked(i) * *win.get_unchecked(i);
                *xx.get_unchecked_mut(n - i - 1) =
                    *x.get_unchecked(n - i - 1) * *win.get_unchecked(i);
            }
        }
        xptr = &xx;
    } else {
        // Use x directly — allocate xx as empty to satisfy borrow checker
        xx = Vec::new();
        let _ = &xx; // suppress unused warning
        xptr = x;
    }

    celt_pitch_xcorr(xptr, xptr, &mut ac[..lag + 1], fast_n, arch);

    for (k, ac_k) in ac[..=lag].iter_mut().enumerate() {
        let mut d = 0.0f32;
        for i in (k + fast_n)..n {
            unsafe {
                d += *xptr.get_unchecked(i) * *xptr.get_unchecked(i - k);
            }
        }
        *ac_k += d;
    }

    0 // shift (always 0 for float)
}
