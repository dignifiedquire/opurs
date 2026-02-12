//! Linear prediction coefficient computation.
//!
//! Upstream C: `celt/celt_lpc.c`

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
    assert!(ac.len() > p);
    lpc.fill(0.0);
    if ac[0] > 1e-10f32 {
        let mut error = ac[0];
        for i in 0..p {
            let mut rr = 0.0f32;
            for j in 0..i {
                rr += lpc[j] * ac[i - j];
            }
            rr += ac[i + 1];
            let r = -(rr / error);
            lpc[i] = r;
            for j in 0..((i + 1) >> 1) {
                let tmp1 = lpc[j];
                let tmp2 = lpc[i - 1 - j];
                lpc[j] = tmp1 + r * tmp2;
                lpc[i - 1 - j] = tmp2 + r * tmp1;
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
pub fn celt_fir_c(x: &[f32], num: &[f32], y: &mut [f32], ord: usize) {
    let n = y.len();
    assert!(x.len() >= n + ord);
    assert!(num.len() >= ord);

    // Reverse the numerator coefficients
    let mut rnum = [0.0f32; LPC_ORDER];
    for i in 0..ord {
        rnum[i] = num[ord - i - 1];
    }

    let mut i = 0i32;
    while i < n as i32 - 3 {
        let ix = i as usize;
        let mut sum = [0.0f32; 4];
        // x is indexed with ord offset: x[ord + ix] is the "current" sample
        sum[0] = x[ord + ix];
        sum[1] = x[ord + ix + 1];
        sum[2] = x[ord + ix + 2];
        sum[3] = x[ord + ix + 3];
        xcorr_kernel(&rnum, &x[ix..], &mut sum, ord);
        y[ix] = sum[0];
        y[ix + 1] = sum[1];
        y[ix + 2] = sum[2];
        y[ix + 3] = sum[3];
        i += 4;
    }
    while (i as usize) < n {
        let ix = i as usize;
        let mut sum = x[ord + ix];
        for j in 0..ord {
            sum += rnum[j] * x[ix + j];
        }
        y[ix] = sum;
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
pub fn celt_iir(buf: &mut [f32], n: usize, den: &[f32], ord: usize, mem: &mut [f32]) {
    assert!(buf.len() >= n);
    assert!(den.len() >= ord);
    assert!(mem.len() >= ord);
    assert!(ord & 3 == 0);

    // Reverse the denominator coefficients
    let mut rden = [0.0f32; LPC_ORDER];
    for i in 0..ord {
        rden[i] = den[ord - i - 1];
    }

    // Internal y buffer with ord prefix for history
    // Max: n=1080 (960+120 for extrapolation) + ord=24 = 1104
    let mut yy = [0.0f32; 1080 + LPC_ORDER];
    for i in 0..ord {
        yy[i] = -mem[ord - i - 1];
    }

    let mut i = 0i32;
    while i < n as i32 - 3 {
        let ix = i as usize;
        let mut sum = [0.0f32; 4];
        sum[0] = buf[ix];
        sum[1] = buf[ix + 1];
        sum[2] = buf[ix + 2];
        sum[3] = buf[ix + 3];
        xcorr_kernel(&rden, &yy[ix..], &mut sum, ord);
        yy[ix + ord] = -sum[0];
        buf[ix] = sum[0];
        sum[1] += yy[ix + ord] * den[0];
        yy[ix + ord + 1] = -sum[1];
        buf[ix + 1] = sum[1];
        sum[2] += yy[ix + ord + 1] * den[0];
        sum[2] += yy[ix + ord] * den[1];
        yy[ix + ord + 2] = -sum[2];
        buf[ix + 2] = sum[2];
        sum[3] += yy[ix + ord + 2] * den[0];
        sum[3] += yy[ix + ord + 1] * den[1];
        sum[3] += yy[ix + ord] * den[2];
        yy[ix + ord + 3] = -sum[3];
        buf[ix + 3] = sum[3];
        i += 4;
    }
    while (i as usize) < n {
        let ix = i as usize;
        let mut sum = buf[ix];
        for j in 0..ord {
            sum -= rden[j] * yy[ix + j];
        }
        yy[ix + ord] = sum;
        buf[ix] = sum;
        i += 1;
    }
    for i in 0..ord {
        mem[i] = buf[n - i - 1];
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
) -> i32 {
    let n = x.len();
    let fast_n = n - lag;
    assert!(n > 0);
    assert!(ac.len() > lag);

    let mut xx: Vec<f32>;
    let xptr: &[f32];

    if let Some(win) = window {
        assert!(win.len() >= overlap);
        xx = x.to_vec();
        for i in 0..overlap {
            xx[i] = x[i] * win[i];
            xx[n - i - 1] = x[n - i - 1] * win[i];
        }
        xptr = &xx;
    } else {
        // Use x directly — allocate xx as empty to satisfy borrow checker
        xx = Vec::new();
        let _ = &xx; // suppress unused warning
        xptr = x;
    }

    celt_pitch_xcorr(xptr, xptr, &mut ac[..lag + 1], fast_n);

    for k in 0..=lag {
        let mut d = 0.0f32;
        for i in (k + fast_n)..n {
            d += xptr[i] * xptr[i - k];
        }
        ac[k] += d;
    }

    0 // shift (always 0 for float)
}
