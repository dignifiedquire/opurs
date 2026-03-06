//! Split-radix FFT implementation.
//!
//! Upstream C: `celt/kiss_fft.c`

#![forbid(unsafe_code)]

use num_traits::Zero;
/// Upstream C: celt/kiss_fft.h:kiss_fft_cpx
pub type KissFftCpx = num_complex::Complex32;
/// Upstream C: celt/kiss_fft.h:kiss_twiddle_cpx
pub type KissTwiddleCpx = num_complex::Complex32;

/// Upstream C: celt/kiss_fft.h:kiss_fft_state
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct KissFftState<'a> {
    pub nfft: usize,
    pub scale: f32,
    pub shift: i32,
    pub factors: [(i32, i32); 8],
    pub bitrev: &'a [i16],
    pub twiddles: &'a [KissTwiddleCpx],
}

/// Upstream C: celt/kiss_fft.c:kf_bfly2
#[inline]
fn kf_bfly2(fout: &mut [KissFftCpx], m: i32, n: i32) {
    let tw: f32 = std::f32::consts::FRAC_1_SQRT_2;
    /* We know that m==4 here because the radix-2 is just after a radix-4 */
    debug_assert_eq!(m, 4);
    debug_assert_eq!(fout.len(), n as usize * 8);
    for chunk in fout.chunks_exact_mut(8) {
        let (fout, fout2) = chunk.split_at_mut(4);

        let t = fout2[0];
        fout2[0] = fout[0] - t;
        fout[0] += t;

        let t = KissFftCpx::new(
            (fout2[1].re + fout2[1].im) * tw,
            (fout2[1].im - fout2[1].re) * tw,
        );
        fout2[1] = fout[1] - t;
        fout[1] += t;

        let t = KissFftCpx::new(fout2[2].im, -fout2[2].re);
        fout2[2] = fout[2] - t;
        fout[2] += t;

        let t = KissFftCpx::new(
            (fout2[3].im - fout2[3].re) * tw,
            -(fout2[3].im + fout2[3].re) * tw,
        );
        fout2[3] = fout[3] - t;
        fout[3] += t;
    }
}
/// Upstream C: celt/kiss_fft.c:kf_bfly4
#[inline]
fn kf_bfly4(fout: &mut [KissFftCpx], fstride: usize, st: &KissFftState, m: i32, n: i32, mm: i32) {
    if m == 1 {
        /* Degenerate case where all the twiddles are 1. */
        debug_assert_eq!(fout.len(), n as usize * 4);
        for chunk in fout.chunks_exact_mut(4) {
            let scratch0 = chunk[0] - chunk[2];
            chunk[0] += chunk[2];
            let scratch1 = chunk[1] + chunk[3];
            chunk[2] = chunk[0] - scratch1;
            chunk[0] += scratch1;
            let scratch1 = chunk[1] - chunk[3];

            chunk[1].re = scratch0.re + scratch1.im;
            chunk[1].im = scratch0.im - scratch1.re;
            chunk[3].re = scratch0.re - scratch1.im;
            chunk[3].im = scratch0.im + scratch1.re;
        }
    } else {
        let mut scratch: [KissFftCpx; 6] = [KissFftCpx::zero(); 6];
        let m = m as usize;
        let m2 = 2 * m;
        let m3 = 3 * m;
        let tw = st.twiddles;

        for i in 0..n {
            let base = (i * mm) as usize;
            /* m is guaranteed to be a multiple of 4. */
            for j in 0..m {
                scratch[0] = fout[base + j + m] * tw[j * fstride];
                scratch[1] = fout[base + j + m2] * tw[j * fstride * 2];
                scratch[2] = fout[base + j + m3] * tw[j * fstride * 3];

                scratch[5] = fout[base + j] - scratch[1];
                fout[base + j] += scratch[1];
                scratch[3] = scratch[0] + scratch[2];
                scratch[4] = scratch[0] - scratch[2];
                fout[base + j + m2] = fout[base + j] - scratch[3];
                fout[base + j] += scratch[3];

                fout[base + j + m].re = scratch[5].re + scratch[4].im;
                fout[base + j + m].im = scratch[5].im - scratch[4].re;
                fout[base + j + m3].re = scratch[5].re - scratch[4].im;
                fout[base + j + m3].im = scratch[5].im + scratch[4].re;
            }
        }
    };
}
/// Upstream C: celt/kiss_fft.c:kf_bfly3
#[inline]
fn kf_bfly3(fout: &mut [KissFftCpx], fstride: usize, st: &KissFftState, m: i32, n: i32, mm: i32) {
    let m = m as usize;
    let m2 = 2 * m;
    let mut scratch: [KissFftCpx; 5] = [KissFftCpx::zero(); 5];
    let epi3 = st.twiddles[fstride * m];
    let tw = st.twiddles;
    for i in 0..n {
        let base = (i * mm) as usize;
        /* For non-custom modes, m is guaranteed to be a multiple of 4. */
        for j in 0..m {
            scratch[1] = fout[base + j + m] * tw[j * fstride];
            scratch[2] = fout[base + j + m2] * tw[j * fstride * 2];

            scratch[3] = scratch[1] + scratch[2];
            scratch[0] = scratch[1] - scratch[2];

            fout[base + j + m] = fout[base + j] - scratch[3] * 0.5f32;

            scratch[0] *= epi3.im;

            fout[base + j] += scratch[3];

            fout[base + j + m2].re = fout[base + j + m].re + scratch[0].im;
            fout[base + j + m2].im = fout[base + j + m].im - scratch[0].re;

            fout[base + j + m].re -= scratch[0].im;
            fout[base + j + m].im += scratch[0].re;
        }
    }
}
/// Upstream C: celt/kiss_fft.c:kf_bfly5
#[inline]
fn kf_bfly5(fout: &mut [KissFftCpx], fstride: usize, st: &KissFftState, m: i32, n: i32, mm: i32) {
    let mut scratch: [KissFftCpx; 13] = [KissFftCpx::zero(); 13];
    let ya = st.twiddles[fstride * m as usize];
    let yb = st.twiddles[fstride * m as usize * 2];
    let tw = st.twiddles;
    let m = m as usize;
    let m2 = 2 * m;
    let m3 = 3 * m;
    let m4 = 4 * m;
    for i in 0..n {
        let base = (i * mm) as usize;

        /* For non-custom modes, m is guaranteed to be a multiple of 4. */
        for u in 0..m {
            scratch[0] = fout[base + u];

            scratch[1] = fout[base + m + u] * tw[u * fstride];
            scratch[2] = fout[base + m2 + u] * tw[2 * u * fstride];
            scratch[3] = fout[base + m3 + u] * tw[3 * u * fstride];
            scratch[4] = fout[base + m4 + u] * tw[4 * u * fstride];

            scratch[7] = scratch[1] + scratch[4];
            scratch[10] = scratch[1] - scratch[4];
            scratch[8] = scratch[2] + scratch[3];
            scratch[9] = scratch[2] - scratch[3];

            fout[base + u] += scratch[7] + scratch[8];

            scratch[5].re = scratch[0].re + (scratch[7].re * ya.re + scratch[8].re * yb.re);
            scratch[5].im = scratch[0].im + (scratch[7].im * ya.re + scratch[8].im * yb.re);

            scratch[6].re = scratch[10].im * ya.im + scratch[9].im * yb.im;
            scratch[6].im = -(scratch[10].re * ya.im + scratch[9].re * yb.im);

            fout[base + m + u] = scratch[5] - scratch[6];
            fout[base + m4 + u] = scratch[5] + scratch[6];

            scratch[11].re = scratch[0].re + (scratch[7].re * yb.re + scratch[8].re * ya.re);
            scratch[11].im = scratch[0].im + (scratch[7].im * yb.re + scratch[8].im * ya.re);
            scratch[12].re = scratch[9].im * ya.im - scratch[10].im * yb.im;
            scratch[12].im = scratch[10].re * yb.im - scratch[9].re * ya.im;

            fout[base + m2 + u] = scratch[11] + scratch[12];
            fout[base + m3 + u] = scratch[11] - scratch[12];
        }
    }
}

/// Upstream C: celt/kiss_fft.c:opus_fft_impl
#[inline]
pub fn opus_fft_impl(st: &KissFftState, fout: &mut [KissFftCpx]) {
    debug_assert_eq!(st.nfft, fout.len());
    let shift = st.shift.max(0);

    let mut fstride: [i32; 8] = [0; 8];
    fstride[0] = 1;

    let mut levels = 0_usize;
    loop {
        let (p, m) = st.factors[levels];
        fstride[levels + 1] = fstride[levels] * p;
        levels += 1;
        if m == 1 {
            break;
        }
    }

    let mut m = st.factors[levels - 1].1;
    for i in (0..levels).rev() {
        let m2 = if i > 0 { st.factors[i - 1].1 } else { 1 };
        match st.factors[i].0 {
            2 => kf_bfly2(fout, m, fstride[i]),
            4 => kf_bfly4(fout, (fstride[i] << shift) as usize, st, m, fstride[i], m2),
            3 => kf_bfly3(fout, (fstride[i] << shift) as usize, st, m, fstride[i], m2),
            5 => kf_bfly5(fout, (fstride[i] << shift) as usize, st, m, fstride[i], m2),
            _ => {}
        }
        m = m2;
    }
}

/// Upstream C: celt/kiss_fft.c:opus_fft_c
#[inline]
pub fn opus_fft_c(st: &KissFftState, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
    let scale: f32 = st.scale;
    debug_assert_eq!(fin.len(), st.nfft);
    debug_assert_eq!(fout.len(), st.nfft);
    for (&x, &i) in fin.iter().zip(st.bitrev) {
        fout[i as usize] = scale * x;
    }
    opus_fft_impl(st, fout);
}
