/// Tests for CELT MDCT matching upstream `celt/tests/test_unit_mdct.c`.
///
/// Verifies forward and inverse MDCT by comparing against a reference DCT
/// computation. SNR must be >= 60 dB.
///
/// Without CUSTOM_MODES, only sizes 1920, 960, 480, 240 are available from
/// the standard mode.
///
/// Upstream C: celt/tests/test_unit_mdct.c
mod test_common;

use test_common::TestRng;
use unsafe_libopus::internals::{mdct_backward, mdct_forward};

/// Reference forward MDCT check: compare actual output with DCT-IV formula.
fn check_forward(input: &[f32], output: &[f32], nfft: usize) -> f64 {
    let mut errpow = 0.0f64;
    let mut sigpow = 0.0f64;

    for bin in 0..nfft / 2 {
        let mut ansr = 0.0f64;
        for k in 0..nfft {
            let phase = 2.0
                * std::f64::consts::PI
                * (k as f64 + 0.5 + 0.25 * nfft as f64)
                * (bin as f64 + 0.5)
                / nfft as f64;
            let re = phase.cos() / (nfft as f64 / 4.0);
            ansr += input[k] as f64 * re;
        }
        let difr = ansr - output[bin] as f64;
        errpow += difr * difr;
        sigpow += ansr * ansr;
    }

    10.0 * (sigpow / errpow).log10()
}

/// Reference inverse MDCT check: reconstruct time-domain from MDCT coefficients.
fn check_inverse(input: &[f32], output: &[f32], nfft: usize) -> f64 {
    let mut errpow = 0.0f64;
    let mut sigpow = 0.0f64;

    for bin in 0..nfft {
        let mut ansr = 0.0f64;
        for k in 0..nfft / 2 {
            let phase = 2.0
                * std::f64::consts::PI
                * (bin as f64 + 0.5 + 0.25 * nfft as f64)
                * (k as f64 + 0.5)
                / nfft as f64;
            let re = phase.cos();
            ansr += input[k] as f64 * re;
        }
        let difr = ansr - output[bin] as f64;
        errpow += difr * difr;
        sigpow += ansr * ansr;
    }

    10.0 * (sigpow / errpow).log10()
}

fn test_mdct_forward(nfft: usize) {
    let mode =
        unsafe_libopus::opus_custom_mode_create(48000, 960, None).expect("Failed to create mode");

    let shift = match nfft {
        1920 => 0,
        960 => 1,
        480 => 2,
        240 => 3,
        _ => panic!("Unsupported MDCT size {nfft}"),
    };

    let overlap = nfft / 2;
    let n2 = nfft / 2;
    let mut rng = TestRng::new(42);

    // mdct_forward expects input.len() >= n2 + overlap = nfft
    let mut input = vec![0.0f32; nfft];
    // output needs n2 elements
    let mut output = vec![0.0f32; n2];
    let window = vec![1.0f32; overlap]; // Q15ONE = 1.0 in float

    for k in 0..nfft {
        input[k] = ((rng.next_u32() % 32768) as f32 - 16384.0) * 32768.0;
    }

    let input_copy = input.clone();
    mdct_forward(&mode.mdct, &input, &mut output, &window, overlap, shift, 1);
    let snr = check_forward(&input_copy, &output, nfft);
    assert!(
        snr >= 60.0,
        "MDCT forward nfft={nfft}: SNR {snr:.1} dB < 60 dB"
    );
}

fn test_mdct_inverse(nfft: usize) {
    let mode =
        unsafe_libopus::opus_custom_mode_create(48000, 960, None).expect("Failed to create mode");

    let shift = match nfft {
        1920 => 0,
        960 => 1,
        480 => 2,
        240 => 3,
        _ => panic!("Unsupported MDCT size {nfft}"),
    };

    let overlap = nfft / 2;
    let n2 = nfft / 2;
    let mut rng = TestRng::new(42);

    // For inverse: input is n2 MDCT coefficients
    let mut input = vec![0.0f32; nfft];
    for k in 0..nfft {
        input[k] = ((rng.next_u32() % 32768) as f32 - 16384.0) * 32768.0 / nfft as f32;
    }

    // mdct_backward: input.len() == n2, out.len() == n2 + overlap = nfft
    let mut output = vec![0.0f32; nfft];
    let window = vec![1.0f32; overlap];

    mdct_backward(
        &mode.mdct,
        &input[..n2],
        &mut output,
        &window,
        overlap,
        shift,
        1,
    );
    // Apply TDAC (time-domain aliasing cancellation)
    for k in 0..nfft / 4 {
        output[nfft - k - 1] = output[nfft / 2 + k];
    }
    let snr = check_inverse(&input[..n2], &output, nfft);
    assert!(
        snr >= 60.0,
        "MDCT inverse nfft={nfft}: SNR {snr:.1} dB < 60 dB"
    );
}

#[test]
fn test_mdct_forward_1920() {
    test_mdct_forward(1920);
}

#[test]
fn test_mdct_inverse_1920() {
    test_mdct_inverse(1920);
}

#[test]
fn test_mdct_forward_960() {
    test_mdct_forward(960);
}

#[test]
fn test_mdct_inverse_960() {
    test_mdct_inverse(960);
}

#[test]
fn test_mdct_forward_480() {
    test_mdct_forward(480);
}

#[test]
fn test_mdct_inverse_480() {
    test_mdct_inverse(480);
}

#[test]
fn test_mdct_forward_240() {
    test_mdct_forward(240);
}

#[test]
fn test_mdct_inverse_240() {
    test_mdct_inverse(240);
}
