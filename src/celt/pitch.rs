//! Pitch analysis and cross-correlation.
//!
//! Upstream C: `celt/pitch.c`

use crate::celt::celt_lpc::{_celt_autocorr, _celt_lpc};
use crate::celt::entcode::celt_udiv;
use crate::celt::mathops::celt_sqrt;

/// Upstream C: celt/pitch.c:dual_inner_prod_c
///
/// Computes two inner products simultaneously: `(x . y01, x . y02)`.
/// All slices must have at least `n` elements.
#[inline]
pub fn dual_inner_prod(x: &[f32], y01: &[f32], y02: &[f32], n: usize) -> (f32, f32) {
    let mut xy01: f32 = 0.0;
    let mut xy02: f32 = 0.0;
    for i in 0..n {
        xy01 += x[i] * y01[i];
        xy02 += x[i] * y02[i];
    }
    (xy01, xy02)
}

/// Upstream C: celt/pitch.c:xcorr_kernel_c
///
/// 4-way cross-correlation kernel. Computes 4 consecutive correlation values.
/// `x` must have at least `len` elements.
/// `y` must have at least `len + 3` elements.
/// Results are accumulated into `sum[0..4]`.
#[inline]
pub fn xcorr_kernel(x: &[f32], y: &[f32], sum: &mut [f32; 4], len: usize) {
    assert!(len >= 3);
    let mut y_0: f32;
    let mut y_1: f32;
    let mut y_2: f32;
    let mut y_3: f32 = 0.0;
    let mut xi = 0usize;
    let mut yi = 0usize;
    y_0 = y[yi];
    yi += 1;
    y_1 = y[yi];
    yi += 1;
    y_2 = y[yi];
    yi += 1;
    let mut j = 0usize;
    while j < len - 3 {
        let tmp = x[xi];
        xi += 1;
        y_3 = y[yi];
        yi += 1;
        sum[0] += tmp * y_0;
        sum[1] += tmp * y_1;
        sum[2] += tmp * y_2;
        sum[3] += tmp * y_3;

        let tmp = x[xi];
        xi += 1;
        y_0 = y[yi];
        yi += 1;
        sum[0] += tmp * y_1;
        sum[1] += tmp * y_2;
        sum[2] += tmp * y_3;
        sum[3] += tmp * y_0;

        let tmp = x[xi];
        xi += 1;
        y_1 = y[yi];
        yi += 1;
        sum[0] += tmp * y_2;
        sum[1] += tmp * y_3;
        sum[2] += tmp * y_0;
        sum[3] += tmp * y_1;

        let tmp = x[xi];
        xi += 1;
        y_2 = y[yi];
        yi += 1;
        sum[0] += tmp * y_3;
        sum[1] += tmp * y_0;
        sum[2] += tmp * y_1;
        sum[3] += tmp * y_2;

        j += 4;
    }
    if j < len {
        let tmp = x[xi];
        xi += 1;
        y_3 = y[yi];
        yi += 1;
        sum[0] += tmp * y_0;
        sum[1] += tmp * y_1;
        sum[2] += tmp * y_2;
        sum[3] += tmp * y_3;
        j += 1;
    }
    if j < len {
        let tmp = x[xi];
        xi += 1;
        y_0 = y[yi];
        yi += 1;
        sum[0] += tmp * y_1;
        sum[1] += tmp * y_2;
        sum[2] += tmp * y_3;
        sum[3] += tmp * y_0;
        j += 1;
    }
    if j < len {
        let tmp = x[xi];
        y_1 = y[yi];
        sum[0] += tmp * y_2;
        sum[1] += tmp * y_3;
        sum[2] += tmp * y_0;
        sum[3] += tmp * y_1;
    }
}

/// Upstream C: celt/pitch.c:celt_inner_prod_c
///
/// Computes the inner product (dot product) of `x` and `y`.
/// Both slices must have at least `N` elements.
#[inline]
pub fn celt_inner_prod(x: &[f32], y: &[f32], N: usize) -> f32 {
    let mut xy: f32 = 0.0;
    for i in 0..N {
        xy += x[i] * y[i];
    }
    xy
}

/// Upstream C: celt/pitch.c:find_best_pitch
///
/// Finds the two best pitch candidates from cross-correlation values.
/// Returns `[best_pitch_0, best_pitch_1]`.
fn find_best_pitch(xcorr: &[f32], y: &[f32], len: usize, max_pitch: usize) -> [i32; 2] {
    let mut Syy: f32 = 1.0;
    let mut best_num: [f32; 2] = [-1.0, -1.0];
    let mut best_den: [f32; 2] = [0.0, 0.0];
    let mut best_pitch: [i32; 2] = [0, 1];
    for yj in &y[..len] {
        Syy += yj * yj;
    }
    for i in 0..max_pitch {
        if xcorr[i] > 0.0 {
            let xcorr16 = xcorr[i] * 1e-12f32;
            let num = xcorr16 * xcorr16;
            if num * best_den[1] > best_num[1] * Syy {
                if num * best_den[0] > best_num[0] * Syy {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = Syy;
                    best_pitch[0] = i as i32;
                } else {
                    best_num[1] = num;
                    best_den[1] = Syy;
                    best_pitch[1] = i as i32;
                }
            }
        }
        Syy += y[i + len] * y[i + len] - y[i] * y[i];
        Syy = 1.0f32.max(Syy);
    }
    best_pitch
}

/// Upstream C: celt/pitch.c:celt_fir5
///
/// 5-tap FIR filter applied in-place.
fn celt_fir5(x: &mut [f32], num: &[f32; 5]) {
    let num0 = num[0];
    let num1 = num[1];
    let num2 = num[2];
    let num3 = num[3];
    let num4 = num[4];
    let mut mem0: f32 = 0.0;
    let mut mem1: f32 = 0.0;
    let mut mem2: f32 = 0.0;
    let mut mem3: f32 = 0.0;
    let mut mem4: f32 = 0.0;
    for xi in x.iter_mut() {
        let mut sum = *xi;
        sum += num0 * mem0;
        sum += num1 * mem1;
        sum += num2 * mem2;
        sum += num3 * mem3;
        sum += num4 * mem4;
        mem4 = mem3;
        mem3 = mem2;
        mem2 = mem1;
        mem1 = mem0;
        mem0 = *xi;
        *xi = sum;
    }
}

/// Upstream C: celt/pitch.c:pitch_downsample
///
/// Downsamples and LPC-filters audio for pitch analysis.
/// `x` contains 1 or 2 channel slices of length `len`.
/// `x_lp` receives the downsampled output of length `len/2`.
pub fn pitch_downsample(x: &[&[f32]], x_lp: &mut [f32], len: usize) {
    let C = x.len();
    assert!(C == 1 || C == 2);
    assert!(x[0].len() >= len);
    let half = len >> 1;
    assert!(x_lp.len() >= half);

    let mut ac: [f32; 5] = [0.0; 5];
    let mut tmp: f32 = 1.0;
    let mut lpc: [f32; 4] = [0.0; 4];
    let mut lpc2: [f32; 5] = [0.0; 5];
    let c1: f32 = 0.8f32;

    for i in 1..half {
        x_lp[i] = 0.5f32 * (0.5f32 * (x[0][2 * i - 1] + x[0][2 * i + 1]) + x[0][2 * i]);
    }
    x_lp[0] = 0.5f32 * (0.5f32 * x[0][1] + x[0][0]);

    if C == 2 {
        for i in 1..half {
            x_lp[i] += 0.5f32 * (0.5f32 * (x[1][2 * i - 1] + x[1][2 * i + 1]) + x[1][2 * i]);
        }
        x_lp[0] += 0.5f32 * (0.5f32 * x[1][1] + x[1][0]);
    }

    _celt_autocorr(&x_lp[..half], &mut ac, None, 0, 4);

    ac[0] *= 1.0001f32;
    #[allow(clippy::needless_range_loop)]
    for i in 1..=4 {
        ac[i] -= ac[i] * (0.008f32 * i as f32) * (0.008f32 * i as f32);
    }
    _celt_lpc(&mut lpc, &ac);
    for lpc_val in lpc.iter_mut() {
        tmp *= 0.9f32;
        *lpc_val *= tmp;
    }
    lpc2[0] = lpc[0] + 0.8f32;
    lpc2[1] = lpc[1] + c1 * lpc[0];
    lpc2[2] = lpc[2] + c1 * lpc[1];
    lpc2[3] = lpc[3] + c1 * lpc[2];
    lpc2[4] = c1 * lpc[3];
    celt_fir5(&mut x_lp[..half], &lpc2);
}

/// Upstream C: celt/pitch.c:celt_pitch_xcorr_c
///
/// Cross-correlation for pitch analysis.
/// `x` must have at least `len` elements.
/// `y` must have at least `len + max_pitch` elements (where `max_pitch = xcorr.len()`).
/// Results are written to `xcorr[0..max_pitch]`.
pub fn celt_pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize) {
    let max_pitch = xcorr.len();
    assert!(max_pitch > 0);
    let mut i = 0i32;
    while i < max_pitch as i32 - 3 {
        let mut sum: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
        xcorr_kernel(&x[..len], &y[i as usize..], &mut sum, len);
        xcorr[i as usize] = sum[0];
        xcorr[i as usize + 1] = sum[1];
        xcorr[i as usize + 2] = sum[2];
        xcorr[i as usize + 3] = sum[3];
        i += 4;
    }
    while (i as usize) < max_pitch {
        xcorr[i as usize] = celt_inner_prod(x, &y[i as usize..], len);
        i += 1;
    }
}

/// Upstream C: celt/pitch.c:pitch_search
///
/// Pitch search: finds the best pitch period.
/// Returns the pitch index.
pub fn pitch_search(x_lp: &[f32], y: &[f32], len: i32, max_pitch: i32) -> i32 {
    assert!(len > 0);
    assert!(max_pitch > 0);
    let lag: i32 = len + max_pitch;

    let mut x_lp4: Vec<f32> = vec![0.0; (len >> 2) as usize];
    let mut y_lp4: Vec<f32> = vec![0.0; (lag >> 2) as usize];
    let mut xcorr: Vec<f32> = vec![0.0; (max_pitch >> 1) as usize];

    for j in 0..(len >> 2) as usize {
        x_lp4[j] = x_lp[2 * j];
    }
    for j in 0..(lag >> 2) as usize {
        y_lp4[j] = y[2 * j];
    }

    celt_pitch_xcorr(
        &x_lp4,
        &y_lp4,
        &mut xcorr[..(max_pitch >> 2) as usize],
        (len >> 2) as usize,
    );

    let best_pitch = find_best_pitch(
        &xcorr[..(max_pitch >> 2) as usize],
        &y_lp4,
        (len >> 2) as usize,
        (max_pitch >> 2) as usize,
    );

    for i in 0..(max_pitch >> 1) as usize {
        xcorr[i] = 0.0;
        if !((i as i32 - 2 * best_pitch[0]).abs() > 2 && (i as i32 - 2 * best_pitch[1]).abs() > 2) {
            let sum = celt_inner_prod(x_lp, &y[i..], (len >> 1) as usize);
            xcorr[i] = (-1.0f32).max(sum);
        }
    }

    let best_pitch = find_best_pitch(
        &xcorr[..(max_pitch >> 1) as usize],
        y,
        (len >> 1) as usize,
        (max_pitch >> 1) as usize,
    );

    let offset;
    if best_pitch[0] > 0 && best_pitch[0] < (max_pitch >> 1) - 1 {
        let a = xcorr[(best_pitch[0] - 1) as usize];
        let b = xcorr[best_pitch[0] as usize];
        let c = xcorr[(best_pitch[0] + 1) as usize];
        if c - a > 0.7f32 * (b - a) {
            offset = 1;
        } else if a - c > 0.7f32 * (b - c) {
            offset = -1;
        } else {
            offset = 0;
        }
    } else {
        offset = 0;
    }
    2 * best_pitch[0] - offset
}

/// Upstream C: celt/pitch.c:compute_pitch_gain
fn compute_pitch_gain(xy: f32, xx: f32, yy: f32) -> f32 {
    xy / celt_sqrt(1.0f32 + xx * yy)
}

const SECOND_CHECK: [i32; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];

/// Upstream C: celt/pitch.c:remove_doubling
///
/// Removes pitch period doubling/tripling.
/// `x` is the full pitch buffer; `maxperiod` is the offset into `x` where
/// the current frame starts (corresponding to `x += maxperiod` in the C code).
/// `T0` is the initial pitch period (modified in place).
/// Returns the pitch gain.
pub fn remove_doubling(
    x: &[f32],
    mut maxperiod: i32,
    mut minperiod: i32,
    mut N: i32,
    T0_: &mut i32,
    mut prev_period: i32,
    prev_gain: f32,
) -> f32 {
    let mut T: i32;

    let mut g: f32;

    let mut pg: f32;
    let mut xy: f32;

    let mut yy: f32;
    let mut xcorr: [f32; 3] = [0.0; 3];
    let mut best_xy: f32;
    let mut best_yy: f32;
    let offset: i32;
    let minperiod0: i32 = minperiod;
    maxperiod /= 2;
    minperiod /= 2;
    *T0_ /= 2;
    prev_period /= 2;
    N /= 2;
    // x_off is the offset into the buffer (equivalent to x += maxperiod in C)
    let x_off = maxperiod as usize;
    if *T0_ >= maxperiod {
        *T0_ = maxperiod - 1;
    }
    let T0: i32 = *T0_;
    T = T0;
    let mut yy_lookup: Vec<f32> = vec![0.0; (maxperiod + 1) as usize];
    let (xx_val, xy_val) = dual_inner_prod(
        &x[x_off..],
        &x[x_off..],
        &x[x_off - T0 as usize..],
        N as usize,
    );
    let xx: f32 = xx_val;
    xy = xy_val;
    yy_lookup[0] = xx;
    yy = xx;
    for i in 1..=maxperiod as usize {
        yy += x[x_off - i] * x[x_off - i] - x[x_off + N as usize - i] * x[x_off + N as usize - i];
        yy_lookup[i] = 0.0f32.max(yy);
    }
    yy = yy_lookup[T0 as usize];
    best_xy = xy;
    best_yy = yy;
    let g0: f32 = compute_pitch_gain(xy, xx, yy);
    g = g0;
    for k in 2..=15 {
        let T1 = celt_udiv((2 * T0 + k) as u32, (2 * k) as u32) as i32;
        if T1 < minperiod {
            break;
        }
        let T1b;
        if k == 2 {
            if T1 + T0 > maxperiod {
                T1b = T0;
            } else {
                T1b = T0 + T1;
            }
        } else {
            T1b = celt_udiv(
                (2 * SECOND_CHECK[k as usize] * T0 + k) as u32,
                (2 * k) as u32,
            ) as i32;
        }
        let (xy_new, xy2_new) = dual_inner_prod(
            &x[x_off..],
            &x[x_off - T1 as usize..],
            &x[x_off - T1b as usize..],
            N as usize,
        );
        xy = 0.5f32 * (xy_new + xy2_new);
        yy = 0.5f32 * (yy_lookup[T1 as usize] + yy_lookup[T1b as usize]);
        let g1 = compute_pitch_gain(xy, xx, yy);
        let mut cont: f32 = 0.0;
        if (T1 - prev_period).abs() <= 1 {
            cont = prev_gain;
        } else if (T1 - prev_period).abs() <= 2 && 5 * k * k < T0 {
            cont = 0.5f32 * prev_gain;
        }
        let mut thresh = 0.3f32.max(0.7f32 * g0 - cont);
        if T1 < 3 * minperiod {
            thresh = 0.4f32.max(0.85f32 * g0 - cont);
        } else if T1 < 2 * minperiod {
            thresh = 0.5f32.max(0.9f32 * g0 - cont);
        }
        if g1 > thresh {
            best_xy = xy;
            best_yy = yy;
            T = T1;
            g = g1;
        }
    }
    best_xy = 0.0f32.max(best_xy);
    if best_yy <= best_xy {
        pg = 1.0;
    } else {
        pg = best_xy / (best_yy + 1.0);
    }
    for k in 0..3i32 {
        xcorr[k as usize] =
            celt_inner_prod(&x[x_off..], &x[x_off - (T + k - 1) as usize..], N as usize);
    }
    if xcorr[2] - xcorr[0] > 0.7f32 * (xcorr[1] - xcorr[0]) {
        offset = 1;
    } else if xcorr[0] - xcorr[2] > 0.7f32 * (xcorr[1] - xcorr[2]) {
        offset = -1;
    } else {
        offset = 0;
    }
    if pg > g {
        pg = g;
    }
    *T0_ = 2 * T + offset;
    if *T0_ < minperiod0 {
        *T0_ = minperiod0;
    }
    pg
}
