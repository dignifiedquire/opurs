/// Tests for CELT FFT (DFT) matching upstream `celt/tests/test_unit_dft.c`.
///
/// Verifies forward and inverse FFT by comparing against a reference DFT
/// computation. SNR must be >= 60 dB.
///
/// The Rust port does not have `opus_fft_alloc` for custom sizes, so we only
/// test sizes available from the standard mode: 480, 240, 120, 60.
///
/// Upstream C: celt/tests/test_unit_dft.c
mod test_common;

use num_complex::Complex32;
use opurs::internals::{opus_fft_c, opus_fft_impl};
use test_common::TestRng;

/// Compute reference DFT and compare with FFT output, returning SNR in dB.
#[allow(clippy::needless_range_loop)]
fn check_fft(input: &[Complex32], output: &[Complex32], nfft: usize, isinverse: bool) -> f64 {
    let mut errpow = 0.0f64;
    let mut sigpow = 0.0f64;

    for bin in 0..nfft {
        let mut ansr = 0.0f64;
        let mut ansi = 0.0f64;

        for k in 0..nfft {
            let phase = -2.0 * std::f64::consts::PI * bin as f64 * k as f64 / nfft as f64;
            let mut re = phase.cos();
            let mut im = phase.sin();

            if isinverse {
                im = -im;
            }
            if !isinverse {
                re /= nfft as f64;
                im /= nfft as f64;
            }

            ansr += input[k].re as f64 * re - input[k].im as f64 * im;
            ansi += input[k].re as f64 * im + input[k].im as f64 * re;
        }

        let difr = ansr - output[bin].re as f64;
        let difi = ansi - output[bin].im as f64;
        errpow += difr * difr + difi * difi;
        sigpow += ansr * ansr + ansi * ansi;
    }

    10.0 * (sigpow / errpow).log10()
}

/// Inline implementation of opus_ifft_c (conjugate → FFT → conjugate).
fn opus_ifft_c(st: &opurs::internals::kiss_fft_state, fin: &[Complex32], fout: &mut [Complex32]) {
    // Bit-reverse copy from input
    for (&x, &br) in fin.iter().zip(st.bitrev.iter()) {
        fout[br as usize] = x;
    }
    // Negate imaginary
    for x in fout.iter_mut().take(st.nfft) {
        x.im = -x.im;
    }
    opus_fft_impl(st, fout);
    // Negate imaginary again
    for x in fout.iter_mut().take(st.nfft) {
        x.im = -x.im;
    }
}

fn test_fft(nfft: usize, isinverse: bool) {
    // Get FFT state from mode
    let mode = opurs::opus_custom_mode_create(48000, 960, None).expect("Failed to create mode");

    let shift = match nfft {
        480 => 0,
        240 => 1,
        120 => 2,
        60 => 3,
        _ => panic!("Unsupported FFT size {nfft} (no mode available)"),
    };
    let cfg = mode.mdct.kfft[shift];

    let mut rng = TestRng::new(42);
    let mut input = vec![Complex32::new(0.0, 0.0); nfft];
    let mut output = vec![Complex32::new(0.0, 0.0); nfft];

    // Generate random input
    for elem in input.iter_mut() {
        elem.re = (rng.next_u32() % 32767) as f32 - 16384.0;
        elem.im = (rng.next_u32() % 32767) as f32 - 16384.0;
    }

    // Scale by 32768
    for elem in input.iter_mut() {
        elem.re *= 32768.0;
        elem.im *= 32768.0;
    }

    // For inverse: divide by nfft
    if isinverse {
        for elem in input.iter_mut() {
            elem.re /= nfft as f32;
            elem.im /= nfft as f32;
        }
    }

    if isinverse {
        opus_ifft_c(cfg, &input, &mut output);
    } else {
        opus_fft_c(cfg, &input, &mut output);
    }

    let snr = check_fft(&input, &output, nfft, isinverse);
    assert!(
        snr >= 60.0,
        "FFT nfft={nfft} inverse={isinverse}: SNR {snr:.1} dB < 60 dB"
    );
}

#[test]
fn test_fft_forward_480() {
    test_fft(480, false);
}

#[test]
fn test_fft_inverse_480() {
    test_fft(480, true);
}

#[test]
fn test_fft_forward_240() {
    test_fft(240, false);
}

#[test]
fn test_fft_inverse_240() {
    test_fft(240, true);
}

#[test]
fn test_fft_forward_120() {
    test_fft(120, false);
}

#[test]
fn test_fft_inverse_120() {
    test_fft(120, true);
}

#[test]
fn test_fft_forward_60() {
    test_fft(60, false);
}

#[test]
fn test_fft_inverse_60() {
    test_fft(60, true);
}
