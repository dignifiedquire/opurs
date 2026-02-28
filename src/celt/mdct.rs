//! Modified Discrete Cosine Transform.
//!
//! Upstream C: `celt/mdct.c`

#![forbid(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unused_assignments
)]

use crate::celt::kiss_fft::{kiss_fft_state, opus_fft_impl};
use num_complex::Complex;
use num_traits::Zero as _;
use std::ops::Neg as _;

/// Upstream C: celt/mdct.h:mdct_lookup
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MdctLookup<'a> {
    pub n: usize,
    pub maxshift: i32,
    pub kfft: [&'a kiss_fft_state<'a>; 4],
    pub trig: &'a [&'a [f32]; 4],
}

/// Upstream C: celt/mdct.c:clt_mdct_forward_c
#[inline]
pub fn mdct_forward(
    l: &MdctLookup,
    input: &[f32],
    out: &mut [f32],
    window: &[f32],
    overlap: usize,
    shift: usize,
    output_stride: usize,
) {
    let st: &kiss_fft_state = l.kfft[shift];
    let scale = st.scale;
    let trig = l.trig[shift];
    let n = l.n >> shift;
    let n2 = n / 2;
    let n4 = n / 4;

    let o = overlap;
    let o2 = overlap / 2;
    let o4 = overlap / 4;

    debug_assert_eq!(window.len(), o);
    debug_assert_eq!(trig.len(), n2);

    // TODO: make sure all callers pass the exactly-sized slice
    debug_assert!(input.len() >= n2 + o);
    let input = &input[..n2 + o];

    debug_assert!(out.len() >= n2 * output_stride);

    debug_assert_eq!(o % 4, 0);

    let trig_real = &trig[..n4];
    let trig_imag = &trig[n4..];

    // n4 = mdct_size / 4; max 960 (QEXT 96kHz).
    const MAX_N4: usize = 960;
    debug_assert!(n4 <= MAX_N4);
    let mut f = [Complex::zero(); MAX_N4];

    /* Consider the input to be composed of four blocks: [a, b, c, d] */
    /* Window, shuffle, fold */
    {
        /* Head chunk (fh): indices 0..o4 */
        for i in 0..o4 {
            // SAFETY: i < o4 = o/4; o2 + 2*i < o2 + 2*(o4-1)+1 = o2 + o/2 - 1 = o - 1 < o = window.len()
            // o2 - 1 - 2*i >= 0 when i < o4. input indices bounded by n2+o = input.len().
            unsafe {
                let w1 = *window.get_unchecked(o2 + 2 * i); /* wtf */
                let w2 = *window.get_unchecked(o2 - 1 - 2 * i); /* whb */
                let x1_n2 = *input.get_unchecked(n2 + o2 + 2 * i); /* xt2f */
                let x2 = *input.get_unchecked(n2 + o2 - 1 - 2 * i); /* xt1b */
                f.get_unchecked_mut(i).re = w2 * x1_n2 + w1 * x2;

                let x1 = *input.get_unchecked(o2 + 2 * i); /* xh2f */
                let x2_n2 = *input.get_unchecked(o2 - 1 - 2 * i); /* xh1b */
                f.get_unchecked_mut(i).im = w1 * x1 - w2 * x2_n2;
            }
        }

        /* Middle chunk (fmid): indices o4..(n4-o4) */
        let mid_len = n4 - o2; /* = n4 - 2*o4 */
        for i in 0..mid_len {
            // SAFETY: o4 + i < o4 + mid_len = n4 <= MAX_N4 = f.len().
            // n2 - 1 - 2*i >= 0 and o + 2*i < n2 + o = input.len().
            unsafe {
                f.get_unchecked_mut(o4 + i).re = *input.get_unchecked(n2 - 1 - 2 * i); /* xmidb */
                f.get_unchecked_mut(o4 + i).im = *input.get_unchecked(o + 2 * i); /* xmidf */
            }
        }

        /* Tail chunk (ft): indices (n4-o4)..n4 */
        for i in 0..o4 {
            // SAFETY: 2*i < 2*o4 = o/2 = o2 < o = window.len().
            // o - 1 - 2*i >= 0 when i < o4. n4 - o4 + i < n4 <= MAX_N4.
            // input indices bounded by n2 + o - 1 < n2 + o = input.len().
            unsafe {
                let w1 = *window.get_unchecked(2 * i); /* whf */
                let w2 = *window.get_unchecked(o - 1 - 2 * i); /* wtb */
                let x1_n2 = *input.get_unchecked(2 * i); /* xh1f */
                let x2 = *input.get_unchecked(o - 1 - 2 * i); /* xh2b */
                f.get_unchecked_mut(n4 - o4 + i).re = -(w1 * x1_n2) + w2 * x2;

                let x1 = *input.get_unchecked(n2 + 2 * i); /* xt1f */
                let x2_n2 = *input.get_unchecked(n2 + o - 1 - 2 * i); /* xt2b */
                f.get_unchecked_mut(n4 - o4 + i).im = w2 * x1 + w1 * x2_n2;
            }
        }
    }

    let mut f2 = [Complex::zero(); MAX_N4];

    /* Pre-rotation */
    {
        for i in 0..n4 {
            // SAFETY: i < n4 <= MAX_N4; trig_real.len() == trig_imag.len() == n4;
            // st.bitrev[i] < n4 (FFT bit-reversal permutation within n4 elements).
            unsafe {
                let t = Complex::new(*trig_real.get_unchecked(i), *trig_imag.get_unchecked(i));
                let yc = scale * (*f.get_unchecked(i) * t);
                *f2.get_unchecked_mut(*st.bitrev.get_unchecked(i) as usize) = yc;
            }
        }
    }

    /* N/4 complex FFT, does not downscale anymore */
    opus_fft_impl(st, &mut f2[..n4]);

    /* Post-rotate */
    {
        for i in 0..n4 {
            // SAFETY: i < n4; trig_real/trig_imag have len n4; f2 has len MAX_N4 >= n4.
            // out indices: max(2*i, n2-1-2*i) * output_stride < n2 * output_stride <= out.len().
            unsafe {
                let t = Complex::new(*trig_real.get_unchecked(i), *trig_imag.get_unchecked(i));
                let yc = (*f2.get_unchecked(i) * t).conj().neg();
                *out.get_unchecked_mut(2 * i * output_stride) = yc.re;
                *out.get_unchecked_mut((n2 - 1 - 2 * i) * output_stride) = yc.im;
            }
        }
    }
}

/// Upstream C: celt/mdct.c:clt_mdct_backward_c
#[inline]
pub fn mdct_backward(
    l: &MdctLookup,
    input: &[f32],
    out: &mut [f32],
    window: &[f32],
    overlap: usize,
    shift: usize,
    input_stride: usize,
) {
    let trig = l.trig[shift];
    let n = l.n >> shift;
    let n2 = n / 2;
    let n4 = n / 4;

    let o = overlap;
    let o2 = overlap / 2;

    debug_assert_eq!(l.kfft[shift].nfft, n4);

    debug_assert_eq!(window.len(), o);
    debug_assert_eq!(trig.len(), n2);

    debug_assert_eq!(input.len(), n2 * input_stride);

    debug_assert_eq!(out.len(), n2 + o);
    let out = &mut out[..n2 + o];

    let trig_real = &trig[..n4];
    let trig_imag = &trig[n4..];

    let outmid_scalar = &mut out[o2..][..n2];

    // use the output space temporarily to compute fft there
    let outmid: &mut [Complex<f32>] = bytemuck::cast_slice_mut(outmid_scalar);

    /* Pre-rotate */
    {
        for i in 0..n4 {
            // SAFETY: i < n4; input indices bounded by n2 * input_stride = input.len();
            // trig_real/trig_imag have len n4; bitrev[i] < n4 = outmid.len().
            unsafe {
                let xr = *input.get_unchecked(2 * i * input_stride); /* xf: even indices */
                let xi = *input.get_unchecked((n2 - 1 - 2 * i) * input_stride); /* xb: odd indices reversed */
                let t = Complex::new(*trig_real.get_unchecked(i), *trig_imag.get_unchecked(i));
                let x = Complex::new(xr, xi);
                let y = x * t;
                *outmid.get_unchecked_mut(*l.kfft[shift].bitrev.get_unchecked(i) as usize) = y;
            }
        }
    }
    opus_fft_impl(l.kfft[shift], outmid);

    /* Post-rotate and de-shuffle from both ends of the buffer at once to make
    it in-place. */
    /* Loop to (N4+1)>>1 to handle odd N4. When N4 is odd, the
    middle pair will be computed twice. */
    // additional asserts to maybe help the optimizer remove bounds checks
    debug_assert_eq!(outmid.len(), n4);
    debug_assert_eq!(trig_real.len(), n4);
    debug_assert_eq!(trig_imag.len(), n4);
    for i in 0..n4.div_ceil(2) {
        // NB: unlike the loops in ctl_mdct_forward_c, the yp0 and yp1 "pointers" are NOT disjoint because they are stepped only by 1
        // so yp0 and yp1 can alias, especially when N4 is odd
        let yp0 = i;
        let yp1 = n4 - i - 1;

        fn swap(Complex { re, im }: Complex<f32>) -> Complex<f32> {
            Complex { re: im, im: re }
        }

        // SAFETY: yp0 = i < n4.div_ceil(2) <= n4 = outmid.len();
        // yp1 = n4 - i - 1 < n4; trig_real/trig_imag have len n4.
        unsafe {
            /* We swap real and imag because we're using an FFT instead of an IFFT. */
            let x = swap(*outmid.get_unchecked(yp0));
            let t = swap(Complex::new(*trig_real.get_unchecked(i), *trig_imag.get_unchecked(i)));
            /* We'd scale up by 2 here, but instead it's done when mixing the windows */
            let y = swap(x * t);

            /* We swap real and imag because we're using an FFT instead of an IFFT. */
            let x = swap(*outmid.get_unchecked(yp1));
            outmid.get_unchecked_mut(yp0).re = y.re;
            outmid.get_unchecked_mut(yp1).im = y.im;

            let t = swap(Complex::new(*trig_real.get_unchecked(n4 - i - 1), *trig_imag.get_unchecked(n4 - i - 1)));
            /* We'd scale up by 2 here, but instead it's done when mixing the windows */
            let y = swap(x * t);
            outmid.get_unchecked_mut(yp1).re = y.re;
            outmid.get_unchecked_mut(yp0).im = y.im;
        }
    }

    /* Mirror on both sides for TDAC */
    {
        for i in 0..o2 {
            let j = o - 1 - i;
            // SAFETY: i < o2 = o/2 < o = window.len(); j = o-1-i >= o2 and j < o.
            // out has len n2 + o; i < o2 < o < n2 + o; j < o < n2 + o.
            unsafe {
                let x1 = *out.get_unchecked(j);
                let x2 = *out.get_unchecked(i);
                let wf = *window.get_unchecked(i);
                let wb = *window.get_unchecked(j);
                *out.get_unchecked_mut(i) = wb * x2 - wf * x1;
                *out.get_unchecked_mut(j) = wf * x2 + wb * x1;
            }
        }
    }
}
