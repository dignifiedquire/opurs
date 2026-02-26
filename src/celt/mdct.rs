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

    // TODO: allocate from a custom per-frame allocator?
    let mut f = std::vec::from_elem(Complex::zero(), n4);

    /* Consider the input to be composed of four blocks: [a, b, c, d] */
    /* Window, shuffle, fold */
    {
        /* Head chunk (fh): indices 0..o4 */
        for i in 0..o4 {
            let w1 = window[o2 + 2 * i]; /* wtf */
            let w2 = window[o2 - 1 - 2 * i]; /* whb */
            let x1_n2 = input[n2 + o2 + 2 * i]; /* xt2f */
            let x2 = input[n2 + o2 - 1 - 2 * i]; /* xt1b */
            f[i].re = w2 * x1_n2 + w1 * x2;

            let x1 = input[o2 + 2 * i]; /* xh2f */
            let x2_n2 = input[o2 - 1 - 2 * i]; /* xh1b */
            f[i].im = w1 * x1 - w2 * x2_n2;
        }

        /* Middle chunk (fmid): indices o4..(n4-o4) */
        let mid_len = n4 - o2; /* = n4 - 2*o4 */
        for i in 0..mid_len {
            f[o4 + i].re = input[n2 - 1 - 2 * i]; /* xmidb */
            f[o4 + i].im = input[o + 2 * i]; /* xmidf */
        }

        /* Tail chunk (ft): indices (n4-o4)..n4 */
        for i in 0..o4 {
            let w1 = window[2 * i]; /* whf */
            let w2 = window[o - 1 - 2 * i]; /* wtb */
            let x1_n2 = input[2 * i]; /* xh1f */
            let x2 = input[o - 1 - 2 * i]; /* xh2b */
            f[n4 - o4 + i].re = -(w1 * x1_n2) + w2 * x2;

            let x1 = input[n2 + 2 * i]; /* xt1f */
            let x2_n2 = input[n2 + o - 1 - 2 * i]; /* xt2b */
            f[n4 - o4 + i].im = w2 * x1 + w1 * x2_n2;
        }
    }

    // TODO: allocate from a custom per-frame allocator?
    let mut f2 = std::vec::from_elem(Complex::zero(), n4);

    /* Pre-rotation */
    {
        for i in 0..n4 {
            let t = Complex::new(trig_real[i], trig_imag[i]);
            let yc = scale * (f[i] * t);
            f2[st.bitrev[i] as usize] = yc;
        }
    }

    /* N/4 complex FFT, does not downscale anymore */
    opus_fft_impl(st, &mut f2);

    /* Post-rotate */
    {
        for i in 0..n4 {
            let t = Complex::new(trig_real[i], trig_imag[i]);
            let yc = (f2[i] * t).conj().neg();
            out[2 * i * output_stride] = yc.re;
            out[(n2 - 1 - 2 * i) * output_stride] = yc.im;
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
            let xr = input[2 * i * input_stride]; /* xf: even indices */
            let xi = input[(n2 - 1 - 2 * i) * input_stride]; /* xb: odd indices reversed */
            let t = Complex::new(trig_real[i], trig_imag[i]);
            let x = Complex::new(xr, xi);
            let y = x * t;
            outmid[l.kfft[shift].bitrev[i] as usize] = y;
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

        /* We swap real and imag because we're using an FFT instead of an IFFT. */
        let x = swap(outmid[yp0]);
        let t = swap(Complex::new(trig_real[i], trig_imag[i]));
        /* We'd scale up by 2 here, but instead it's done when mixing the windows */
        let y = swap(x * t);

        /* We swap real and imag because we're using an FFT instead of an IFFT. */
        let x = swap(outmid[yp1]);
        outmid[yp0].re = y.re;
        outmid[yp1].im = y.im;

        let t = swap(Complex::new(trig_real[n4 - i - 1], trig_imag[n4 - i - 1]));
        /* We'd scale up by 2 here, but instead it's done when mixing the windows */
        let y = swap(x * t);
        outmid[yp1].re = y.re;
        outmid[yp0].im = y.im;
    }

    /* Mirror on both sides for TDAC */
    {
        for i in 0..o2 {
            let j = o - 1 - i;
            let x1 = out[j];
            let x2 = out[i];
            let wf = window[i];
            let wb = window[j];
            out[i] = wb * x2 - wf * x1;
            out[j] = wf * x2 + wb * x1;
        }
    }
}
