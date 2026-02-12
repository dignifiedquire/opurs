//! Frequency-domain DSP for the DNN subsystem.
//!
//! Provides FFT, band energy computation, DCT, LPC analysis, and windowing
//! used by the LPCNet encoder and PLC modules.
//!
//! Upstream C: `dnn/freq.c`, `dnn/freq.h`

use super::burg::silk_burg_analysis;
use super::lpcnet_tables::{kfft, DCT_TABLE, HALF_WINDOW};
use crate::celt::kiss_fft::{kiss_fft_cpx, opus_fft_c};

// --- Constants ---

/// LPC order for DNN frequency analysis.
pub const LPC_ORDER: usize = 16;

/// Pre-emphasis coefficient.
pub const PREEMPHASIS: f32 = 0.85;

/// Frame size in 5ms units.
pub const FRAME_SIZE_5MS: usize = 2;
/// Overlap size in 5ms units.
pub const OVERLAP_SIZE_5MS: usize = 2;

/// Samples per 5ms unit at 16 kHz.
const SAMPLES_PER_5MS: usize = 80;

/// Window size in 5ms units.
pub const WINDOW_SIZE_5MS: usize = FRAME_SIZE_5MS + OVERLAP_SIZE_5MS;

/// Frame size in samples (160 at 16 kHz).
pub const FRAME_SIZE: usize = SAMPLES_PER_5MS * FRAME_SIZE_5MS;
/// Overlap size in samples (160 at 16 kHz).
pub const OVERLAP_SIZE: usize = SAMPLES_PER_5MS * OVERLAP_SIZE_5MS;
/// Window size in samples (320 at 16 kHz).
pub const WINDOW_SIZE: usize = FRAME_SIZE + OVERLAP_SIZE;
/// Frequency bins (half window + 1 = 161).
pub const FREQ_SIZE: usize = WINDOW_SIZE / 2 + 1;

/// Number of frequency bands.
pub const NB_BANDS: usize = 18;

/// Number of features per frame.
pub const NB_FEATURES: usize = 20;

/// Band edges in 5ms-unit frequency bins.
///
/// Upstream C: dnn/freq.c:eband5ms
static EBAND5MS: [usize; NB_BANDS] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40,
];

/// Compensation factors per band.
///
/// Upstream C: dnn/freq.c:compensation
static COMPENSATION: [f32; NB_BANDS] = [
    0.8, 1., 1., 1., 1., 1., 1., 1., 0.666667, 0.5, 0.5, 0.5, 0.333333, 0.25, 0.25, 0.2, 0.166667,
    0.173913,
];

// --- FFT ---

/// Forward FFT: real input (WINDOW_SIZE) -> complex output (FREQ_SIZE).
///
/// Upstream C: dnn/freq.c:forward_transform
pub fn forward_transform(out: &mut [kiss_fft_cpx], input: &[f32]) {
    let fft_state = kfft();
    let mut x = vec![kiss_fft_cpx::default(); WINDOW_SIZE];
    let mut y = vec![kiss_fft_cpx::default(); WINDOW_SIZE];
    for i in 0..WINDOW_SIZE {
        x[i].re = input[i];
        x[i].im = 0.0;
    }
    opus_fft_c(&fft_state, &x, &mut y);
    out[..FREQ_SIZE].copy_from_slice(&y[..FREQ_SIZE]);
}

/// Inverse FFT: complex input (FREQ_SIZE) -> real output (WINDOW_SIZE).
///
/// Upstream C: dnn/freq.c:inverse_transform
pub fn inverse_transform(out: &mut [f32], input: &[kiss_fft_cpx]) {
    let fft_state = kfft();
    let mut x = vec![kiss_fft_cpx::default(); WINDOW_SIZE];
    let mut y = vec![kiss_fft_cpx::default(); WINDOW_SIZE];
    x[..FREQ_SIZE].copy_from_slice(&input[..FREQ_SIZE]);
    // Mirror conjugate for real-valued IFFT
    for i in FREQ_SIZE..WINDOW_SIZE {
        x[i].re = x[WINDOW_SIZE - i].re;
        x[i].im = -x[WINDOW_SIZE - i].im;
    }
    opus_fft_c(&fft_state, &x, &mut y);
    // Output in reverse order for IFFT
    out[0] = WINDOW_SIZE as f32 * y[0].re;
    for i in 1..WINDOW_SIZE {
        out[i] = WINDOW_SIZE as f32 * y[WINDOW_SIZE - i].re;
    }
}

// --- Band energy ---

/// Compute band energy from FFT magnitudes using triangular overlap.
///
/// Upstream C: dnn/freq.c:lpcn_compute_band_energy
pub fn lpcn_compute_band_energy(band_e: &mut [f32], x: &[kiss_fft_cpx]) {
    let mut sum = [0.0f32; NB_BANDS];
    for i in 0..NB_BANDS - 1 {
        let band_size = (EBAND5MS[i + 1] - EBAND5MS[i]) * WINDOW_SIZE_5MS;
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let idx = EBAND5MS[i] * WINDOW_SIZE_5MS + j;
            let tmp = x[idx].re * x[idx].re + x[idx].im * x[idx].im;
            sum[i] += (1.0 - frac) * tmp;
            sum[i + 1] += frac * tmp;
        }
    }
    sum[0] *= 2.0;
    sum[NB_BANDS - 1] *= 2.0;
    band_e[..NB_BANDS].copy_from_slice(&sum);
}

/// Compute inverse band energy from FFT magnitudes.
///
/// Upstream C: dnn/freq.c:compute_band_energy_inverse
fn compute_band_energy_inverse(band_e: &mut [f32], x: &[kiss_fft_cpx]) {
    let mut sum = [0.0f32; NB_BANDS];
    for i in 0..NB_BANDS - 1 {
        let band_size = (EBAND5MS[i + 1] - EBAND5MS[i]) * WINDOW_SIZE_5MS;
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let idx = EBAND5MS[i] * WINDOW_SIZE_5MS + j;
            let tmp = x[idx].re * x[idx].re + x[idx].im * x[idx].im;
            let tmp = 1.0 / (tmp + 1e-9);
            sum[i] += (1.0 - frac) * tmp;
            sum[i + 1] += frac * tmp;
        }
    }
    sum[0] *= 2.0;
    sum[NB_BANDS - 1] *= 2.0;
    band_e[..NB_BANDS].copy_from_slice(&sum);
}

/// Interpolate band gains to per-bin gains.
///
/// Upstream C: dnn/freq.c:interp_band_gain
pub fn interp_band_gain(g: &mut [f32], band_e: &[f32]) {
    for x in g[..FREQ_SIZE].iter_mut() {
        *x = 0.0;
    }
    for i in 0..NB_BANDS - 1 {
        let band_size = (EBAND5MS[i + 1] - EBAND5MS[i]) * WINDOW_SIZE_5MS;
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            g[EBAND5MS[i] * WINDOW_SIZE_5MS + j] = (1.0 - frac) * band_e[i] + frac * band_e[i + 1];
        }
    }
}

// --- DCT ---

/// Type-II DCT over NB_BANDS values.
///
/// Upstream C: dnn/freq.c:dct
pub fn dct(out: &mut [f32], input: &[f32]) {
    // C: sum * sqrt(2./NB_BANDS) — sqrt in double, multiply in double
    let scale = (2.0f64 / NB_BANDS as f64).sqrt();
    for i in 0..NB_BANDS {
        let mut sum = 0.0f32;
        for j in 0..NB_BANDS {
            sum += input[j] * DCT_TABLE[j * NB_BANDS + i];
        }
        out[i] = (sum as f64 * scale) as f32;
    }
}

/// Inverse DCT (type-III) over NB_BANDS values.
///
/// Upstream C: dnn/freq.c:idct
fn idct(out: &mut [f32], input: &[f32]) {
    // C: sum * sqrt(2./NB_BANDS) — sqrt in double, multiply in double
    let scale = (2.0f64 / NB_BANDS as f64).sqrt();
    for i in 0..NB_BANDS {
        let mut sum = 0.0f32;
        for j in 0..NB_BANDS {
            sum += input[j] * DCT_TABLE[i * NB_BANDS + j];
        }
        out[i] = (sum as f64 * scale) as f32;
    }
}

// --- LPC ---

/// Levinson-Durbin LPC from autocorrelation.
///
/// Returns prediction error.
///
/// Upstream C: dnn/freq.c:lpcn_lpc
fn lpcn_lpc(lpc: &mut [f32], rc: &mut [f32], ac: &[f32], p: usize) -> f32 {
    for i in 0..p {
        lpc[i] = 0.0;
        rc[i] = 0.0;
    }
    let mut error = ac[0];
    if ac[0] == 0.0 {
        return error;
    }
    for i in 0..p {
        // Sum up this iteration's reflection coefficient
        let mut rr: f32 = 0.0;
        for j in 0..i {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1] / 8.0; // SHR32(ac[i+1], 3)
        let r = -(rr * 8.0) / error; // -SHL32(rr, 3) / error
        rc[i] = r;
        // Update LPC coefficients and total error
        lpc[i] = r / 8.0; // SHR32(r, 3)
        for j in 0..(i + 1) >> 1 {
            let tmp1 = lpc[j];
            let tmp2 = lpc[i - 1 - j];
            // MULT32_32_Q31 in float is just multiply
            lpc[j] = tmp1 + r * tmp2;
            lpc[i - 1 - j] = tmp2 + r * tmp1;
        }

        error -= r * r * error;
        // Bail out once we get 30 dB gain
        if error < 0.001 * ac[0] {
            break;
        }
    }
    error
}

/// Compute LPC coefficients from band energies.
///
/// Upstream C: dnn/freq.c:lpc_from_bands
fn lpc_from_bands(lpc: &mut [f32], ex: &[f32]) -> f32 {
    let mut xr = [0.0f32; FREQ_SIZE];
    interp_band_gain(&mut xr, ex);
    xr[FREQ_SIZE - 1] = 0.0;
    let mut x_auto_cpx = vec![kiss_fft_cpx::default(); FREQ_SIZE];
    for i in 0..FREQ_SIZE {
        x_auto_cpx[i].re = xr[i];
        x_auto_cpx[i].im = 0.0;
    }
    let mut x_auto = [0.0f32; WINDOW_SIZE];
    inverse_transform(&mut x_auto, &x_auto_cpx);
    let mut ac = [0.0f32; LPC_ORDER + 1];
    ac[..LPC_ORDER + 1].copy_from_slice(&x_auto[..LPC_ORDER + 1]);

    // -40 dB noise floor
    ac[0] += ac[0] * 1e-4 + 320.0 / 12.0 / 38.0;
    // Lag windowing
    for i in 1..LPC_ORDER + 1 {
        ac[i] *= 1.0 - 6e-5 * (i * i) as f32;
    }
    let mut rc = [0.0f32; LPC_ORDER];
    lpcn_lpc(lpc, &mut rc, &ac, LPC_ORDER)
}

/// Apply LPC weighting (bandwidth expansion).
///
/// Upstream C: dnn/freq.c:lpc_weighting
pub fn lpc_weighting(lpc: &mut [f32], gamma: f32) {
    let mut gamma_i = gamma;
    for i in 0..LPC_ORDER {
        lpc[i] *= gamma_i;
        gamma_i *= gamma;
    }
}

/// Compute LPC from cepstral coefficients.
///
/// Upstream C: dnn/freq.c:lpc_from_cepstrum
pub fn lpc_from_cepstrum(lpc: &mut [f32], cepstrum: &[f32]) -> f32 {
    let mut tmp = [0.0f32; NB_BANDS];
    tmp[..NB_BANDS].copy_from_slice(&cepstrum[..NB_BANDS]);
    tmp[0] += 4.0;
    let mut ex = [0.0f32; NB_BANDS];
    idct(&mut ex, &tmp);
    for i in 0..NB_BANDS {
        ex[i] = (10.0f32).powf(ex[i]) * COMPENSATION[i];
    }
    lpc_from_bands(lpc, &ex)
}

// --- Windowing ---

/// Apply analysis/synthesis window (in-place).
///
/// Upstream C: dnn/freq.c:apply_window
pub fn apply_window(x: &mut [f32]) {
    for i in 0..OVERLAP_SIZE {
        x[i] *= HALF_WINDOW[i];
        x[WINDOW_SIZE - 1 - i] *= HALF_WINDOW[i];
    }
}

// --- Burg cepstral analysis ---

/// Compute burg cepstrum from a PCM segment.
///
/// Upstream C: dnn/freq.c:compute_burg_cepstrum
fn compute_burg_cepstrum(pcm: &[f32], burg_cepstrum: &mut [f32], len: usize, order: usize) {
    assert!(order <= LPC_ORDER);
    assert!(len <= FRAME_SIZE);

    // Pre-emphasis
    let mut burg_in = [0.0f32; FRAME_SIZE];
    for i in 0..len - 1 {
        burg_in[i] = pcm[i + 1] - PREEMPHASIS * pcm[i];
    }

    // Burg LPC analysis
    let mut burg_lpc = [0.0f32; LPC_ORDER];
    let g = silk_burg_analysis(&mut burg_lpc, &burg_in[..len - 1], 1e-3, len - 1, 1, order);
    let g = g / (len as f32 - 2.0 * (order as f32 - 1.0));

    // Build LPC impulse response with decay
    let mut x = [0.0f32; WINDOW_SIZE];
    x[0] = 1.0;
    for i in 0..order {
        x[i + 1] = -burg_lpc[i] * (0.995f32).powi(i as i32 + 1);
    }

    // FFT of LPC impulse response
    let mut lpc_fft = vec![kiss_fft_cpx::default(); FREQ_SIZE];
    forward_transform(&mut lpc_fft, &x);

    // Inverse band energy
    let mut eburg = [0.0f32; NB_BANDS];
    compute_band_energy_inverse(&mut eburg, &lpc_fft);

    // Scale by gain and window normalization
    let scale = 0.45 * g * (1.0 / (WINDOW_SIZE as f32 * WINDOW_SIZE as f32 * WINDOW_SIZE as f32));
    for i in 0..NB_BANDS {
        eburg[i] *= scale;
    }

    // Log and clamp
    let mut ly = [0.0f32; NB_BANDS];
    let mut log_max: f32 = -2.0;
    let mut follow: f32 = -2.0;
    for i in 0..NB_BANDS {
        ly[i] = (1e-2 + eburg[i]).log10();
        ly[i] = ly[i].max(log_max - 8.0).max(follow - 2.5);
        log_max = log_max.max(ly[i]);
        follow = (follow - 2.5).max(ly[i]);
    }

    // DCT
    dct(burg_cepstrum, &ly);
    burg_cepstrum[0] += -4.0;
}

/// Compute burg cepstral features from a full frame of PCM.
///
/// Produces 2 * NB_BANDS cepstral coefficients: first half-frame average
/// and second half-frame difference.
///
/// Upstream C: dnn/freq.c:burg_cepstral_analysis
pub fn burg_cepstral_analysis(ceps: &mut [f32], x: &[f32]) {
    let half = FRAME_SIZE / 2;
    compute_burg_cepstrum(x, &mut ceps[..NB_BANDS], half, LPC_ORDER);
    compute_burg_cepstrum(&x[half..], &mut ceps[NB_BANDS..], half, LPC_ORDER);
    for i in 0..NB_BANDS {
        let c0 = ceps[i];
        let c1 = ceps[NB_BANDS + i];
        ceps[i] = 0.5 * (c0 + c1);
        ceps[NB_BANDS + i] = c0 - c1;
    }
}
