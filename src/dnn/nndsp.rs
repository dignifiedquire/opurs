//! Adaptive convolution, comb filter, and shaping primitives for OSCE.
//!
//! Upstream C: `dnn/nndsp.c`, `dnn/nndsp.h`

use crate::arch::Arch;
use crate::celt::mathops::celt_log;
use crate::celt::pitch::celt_pitch_xcorr;
use crate::dnn::nnet::*;

pub const ADACONV_MAX_KERNEL_SIZE: usize = 32;
pub const ADACONV_MAX_INPUT_CHANNELS: usize = 3;
pub const ADACONV_MAX_OUTPUT_CHANNELS: usize = 3;
pub const ADACONV_MAX_FRAME_SIZE: usize = 240;
pub const ADACONV_MAX_OVERLAP_SIZE: usize = 120;

pub const ADACOMB_MAX_LAG: usize = 300;
pub const ADACOMB_MAX_KERNEL_SIZE: usize = 16;
pub const ADACOMB_MAX_FRAME_SIZE: usize = 80;
pub const ADACOMB_MAX_OVERLAP_SIZE: usize = 40;

pub const ADASHAPE_MAX_INPUT_DIM: usize = 512;
pub const ADASHAPE_MAX_FRAME_SIZE: usize = 240;

/// Upstream C: dnn/nndsp.h:AdaConvState
#[derive(Clone)]
pub struct AdaConvState {
    pub history: Vec<f32>,
    pub last_kernel: Vec<f32>,
    pub last_gain: f32,
}

impl Default for AdaConvState {
    fn default() -> Self {
        AdaConvState {
            history: vec![0.0; ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS],
            last_kernel: vec![
                0.0;
                ADACONV_MAX_KERNEL_SIZE
                    * ADACONV_MAX_INPUT_CHANNELS
                    * ADACONV_MAX_OUTPUT_CHANNELS
            ],
            last_gain: 0.0,
        }
    }
}

/// Upstream C: dnn/nndsp.h:AdaCombState
#[derive(Clone)]
pub struct AdaCombState {
    pub history: Vec<f32>,
    pub last_kernel: Vec<f32>,
    pub last_global_gain: f32,
    pub last_pitch_lag: i32,
}

impl Default for AdaCombState {
    fn default() -> Self {
        AdaCombState {
            history: vec![0.0; ADACOMB_MAX_KERNEL_SIZE + ADACOMB_MAX_LAG],
            last_kernel: vec![0.0; ADACOMB_MAX_KERNEL_SIZE],
            last_global_gain: 0.0,
            last_pitch_lag: 0,
        }
    }
}

/// Upstream C: dnn/nndsp.h:AdaShapeState
#[derive(Clone)]
pub struct AdaShapeState {
    pub conv_alpha1f_state: Vec<f32>,
    pub conv_alpha1t_state: Vec<f32>,
    pub conv_alpha2_state: Vec<f32>,
    pub interpolate_state: [f32; 1],
}

impl Default for AdaShapeState {
    fn default() -> Self {
        AdaShapeState {
            conv_alpha1f_state: vec![0.0; ADASHAPE_MAX_INPUT_DIM],
            conv_alpha1t_state: vec![0.0; ADASHAPE_MAX_INPUT_DIM],
            conv_alpha2_state: vec![0.0; ADASHAPE_MAX_FRAME_SIZE],
            interpolate_state: [0.0],
        }
    }
}

/// Compute overlap window (raised cosine).
///
/// Upstream C: dnn/nndsp.c:compute_overlap_window
pub fn compute_overlap_window(window: &mut [f32], overlap_size: usize) {
    for i in 0..overlap_size {
        let angle = std::f64::consts::PI * (i as f64 + 0.5) / overlap_size as f64;
        window[i] = (0.5 + 0.5 * angle.cos()) as f32;
    }
}

macro_rules! kernel_index {
    ($i_out:expr, $i_in:expr, $i_k:expr, $in_channels:expr, $kernel_size:expr) => {
        (($i_out) * ($in_channels) + ($i_in)) * ($kernel_size) + ($i_k)
    };
}

/// Normalize kernel over input channel and kernel dimension.
///
/// Upstream C: dnn/nndsp.c:scale_kernel
pub fn scale_kernel(
    kernel: &mut [f32],
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    gain: &[f32],
) {
    for i_out in 0..out_channels {
        let mut norm = 0.0f32;
        for i_in in 0..in_channels {
            for i_k in 0..kernel_size {
                let idx = kernel_index!(i_out, i_in, i_k, in_channels, kernel_size);
                norm += kernel[idx] * kernel[idx];
            }
        }
        // C uses double-precision sqrt()
        norm = (1.0 / (1e-6f64 + (norm as f64).sqrt())) as f32;
        for i_in in 0..in_channels {
            for i_k in 0..kernel_size {
                let idx = kernel_index!(i_out, i_in, i_k, in_channels, kernel_size);
                kernel[idx] *= norm * gain[i_out];
            }
        }
    }
}

/// Transform gains with exp(a*x + b).
///
/// Upstream C: dnn/nndsp.c:transform_gains
pub fn transform_gains(
    gains: &mut [f32],
    num_gains: usize,
    filter_gain_a: f32,
    filter_gain_b: f32,
) {
    for i in 0..num_gains {
        // C uses double-precision exp(): float promotes to double, exp in double, truncate back
        // black_box prevents LLVM auto-vectorization of exp() (see compute_overlap_window comment).
        let val = (filter_gain_a * gains[i] + filter_gain_b) as f64;
        gains[i] = std::hint::black_box(val).exp() as f32;
    }
}

/// Adaptive convolution filter.
///
/// Upstream C: dnn/nndsp.c:adaconv_process_frame
#[allow(clippy::too_many_arguments)]
pub fn adaconv_process_frame(
    state: &mut AdaConvState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    kernel_layer: &LinearLayer,
    gain_layer: &LinearLayer,
    _feature_dim: usize,
    frame_size: usize,
    overlap_size: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    left_padding: usize,
    filter_gain_a: f32,
    filter_gain_b: f32,
    _shape_gain: f32,
    window: &[f32],
    arch: Arch,
) {
    assert_eq!(left_padding, kernel_size - 1); // only causal version supported
    assert!(kernel_size < frame_size);

    let mut output_buffer = vec![0.0f32; ADACONV_MAX_FRAME_SIZE * ADACONV_MAX_OUTPUT_CHANNELS];
    let mut kernel_buffer =
        vec![
            0.0f32;
            ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS * ADACONV_MAX_OUTPUT_CHANNELS
        ];
    let mut input_buffer = vec![
        0.0f32;
        ADACONV_MAX_INPUT_CHANNELS
            * (ADACONV_MAX_FRAME_SIZE + ADACONV_MAX_KERNEL_SIZE)
    ];
    let mut gain_buffer = vec![0.0f32; ADACONV_MAX_OUTPUT_CHANNELS];

    // Prepare input: history + new samples per channel
    for i_in in 0..in_channels {
        let base = i_in * (kernel_size + frame_size);
        input_buffer[base..base + kernel_size]
            .copy_from_slice(&state.history[i_in * kernel_size..(i_in + 1) * kernel_size]);
        input_buffer[base + kernel_size..base + kernel_size + frame_size]
            .copy_from_slice(&x_in[frame_size * i_in..frame_size * (i_in + 1)]);
    }

    // Calculate new kernel and gain
    compute_generic_dense(
        kernel_layer,
        &mut kernel_buffer,
        features,
        ACTIVATION_LINEAR,
    );
    compute_generic_dense(gain_layer, &mut gain_buffer, features, ACTIVATION_TANH);
    transform_gains(&mut gain_buffer, out_channels, filter_gain_a, filter_gain_b);
    scale_kernel(
        &mut kernel_buffer,
        in_channels,
        out_channels,
        kernel_size,
        &gain_buffer,
    );

    // Calculate overlapping part using kernel from last frame + crossfade with new kernel
    for i_out in 0..out_channels {
        for i_in in 0..in_channels {
            let mut kernel0 = [0.0f32; ADACONV_MAX_KERNEL_SIZE];
            let mut kernel1 = [0.0f32; ADACONV_MAX_KERNEL_SIZE];
            let mut channel_buffer0 = [0.0f32; ADACONV_MAX_OVERLAP_SIZE];
            let mut channel_buffer1 = vec![0.0f32; ADACONV_MAX_FRAME_SIZE];

            let k0_start = kernel_index!(i_out, i_in, 0, in_channels, kernel_size);
            kernel0[..kernel_size]
                .copy_from_slice(&state.last_kernel[k0_start..k0_start + kernel_size]);
            kernel1[..kernel_size]
                .copy_from_slice(&kernel_buffer[k0_start..k0_start + kernel_size]);

            let p_base = i_in * (frame_size + kernel_size) + kernel_size;
            let p_start = p_base - left_padding;
            celt_pitch_xcorr(
                &kernel0[..ADACONV_MAX_KERNEL_SIZE],
                &input_buffer[p_start..],
                &mut channel_buffer0[..overlap_size],
                ADACONV_MAX_KERNEL_SIZE,
                arch,
            );
            celt_pitch_xcorr(
                &kernel1[..ADACONV_MAX_KERNEL_SIZE],
                &input_buffer[p_start..],
                &mut channel_buffer1[..frame_size],
                ADACONV_MAX_KERNEL_SIZE,
                arch,
            );

            for i in 0..overlap_size {
                // C uses two separate += to match FP rounding order
                output_buffer[i + i_out * frame_size] += window[i] * channel_buffer0[i];
                output_buffer[i + i_out * frame_size] += (1.0 - window[i]) * channel_buffer1[i];
            }
            for i in overlap_size..frame_size {
                output_buffer[i + i_out * frame_size] += channel_buffer1[i];
            }
        }
    }

    x_out[..out_channels * frame_size].copy_from_slice(&output_buffer[..out_channels * frame_size]);

    // Buffer update
    for i_in in 0..in_channels {
        let p_base = i_in * (frame_size + kernel_size) + kernel_size;
        let src_start = p_base + frame_size - kernel_size;
        let hist_start = i_in * kernel_size;
        let saved: Vec<f32> = input_buffer[src_start..src_start + kernel_size].to_vec();
        state.history[hist_start..hist_start + kernel_size].copy_from_slice(&saved);
    }
    state.last_kernel[..kernel_size * in_channels * out_channels]
        .copy_from_slice(&kernel_buffer[..kernel_size * in_channels * out_channels]);
}

/// Adaptive comb filter.
///
/// Upstream C: dnn/nndsp.c:adacomb_process_frame
#[allow(clippy::too_many_arguments)]
pub fn adacomb_process_frame(
    state: &mut AdaCombState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    kernel_layer: &LinearLayer,
    gain_layer: &LinearLayer,
    global_gain_layer: &LinearLayer,
    pitch_lag: i32,
    _feature_dim: usize,
    frame_size: usize,
    overlap_size: usize,
    kernel_size: usize,
    left_padding: usize,
    filter_gain_a: f32,
    filter_gain_b: f32,
    log_gain_limit: f32,
    window: &[f32],
    arch: Arch,
) {
    let mut output_buffer = vec![0.0f32; ADACOMB_MAX_FRAME_SIZE];
    let mut output_buffer_last = vec![0.0f32; ADACOMB_MAX_FRAME_SIZE];
    let mut kernel_buffer = vec![0.0f32; ADACOMB_MAX_KERNEL_SIZE];
    let mut input_buffer =
        vec![0.0f32; ADACOMB_MAX_FRAME_SIZE + ADACOMB_MAX_LAG + ADACOMB_MAX_KERNEL_SIZE];

    // Prepare input buffer
    let hist_len = kernel_size + ADACOMB_MAX_LAG;
    input_buffer[..hist_len].copy_from_slice(&state.history[..hist_len]);
    input_buffer[hist_len..hist_len + frame_size].copy_from_slice(&x_in[..frame_size]);
    let p_offset = kernel_size + ADACOMB_MAX_LAG; // p_input = input_buffer + p_offset

    // Calculate new kernel and gain
    compute_generic_dense(
        kernel_layer,
        &mut kernel_buffer,
        features,
        ACTIVATION_LINEAR,
    );
    let mut gain = 0.0f32;
    compute_generic_dense(
        gain_layer,
        std::slice::from_mut(&mut gain),
        features,
        ACTIVATION_RELU,
    );
    let mut global_gain = 0.0f32;
    compute_generic_dense(
        global_gain_layer,
        std::slice::from_mut(&mut global_gain),
        features,
        ACTIVATION_TANH,
    );

    // C uses double-precision exp() and log()
    gain = ((log_gain_limit - gain) as f64).exp() as f32;
    global_gain = ((filter_gain_a * global_gain + filter_gain_b) as f64).exp() as f32;
    scale_kernel(&mut kernel_buffer, 1, 1, kernel_size, &[gain]);

    let mut kernel = [0.0f32; ADACOMB_MAX_KERNEL_SIZE];
    let mut last_kernel = [0.0f32; ADACOMB_MAX_KERNEL_SIZE];
    kernel[..kernel_size].copy_from_slice(&kernel_buffer[..kernel_size]);
    last_kernel[..kernel_size].copy_from_slice(&state.last_kernel[..kernel_size]);

    // Xcorr with last kernel (for overlap)
    let last_lag = state.last_pitch_lag as usize;
    let xcorr_start_last = p_offset - left_padding - last_lag;
    celt_pitch_xcorr(
        &last_kernel[..ADACOMB_MAX_KERNEL_SIZE],
        &input_buffer[xcorr_start_last..],
        &mut output_buffer_last[..overlap_size],
        ADACOMB_MAX_KERNEL_SIZE,
        arch,
    );

    // Xcorr with new kernel (full frame)
    let xcorr_start = p_offset - left_padding - pitch_lag as usize;
    celt_pitch_xcorr(
        &kernel[..ADACOMB_MAX_KERNEL_SIZE],
        &input_buffer[xcorr_start..],
        &mut output_buffer[..frame_size],
        ADACOMB_MAX_KERNEL_SIZE,
        arch,
    );

    // Overlap-add crossfade
    for i in 0..overlap_size {
        output_buffer[i] = state.last_global_gain * window[i] * output_buffer_last[i]
            + global_gain * (1.0 - window[i]) * output_buffer[i];
    }

    // Add direct signal (overlap region)
    for i in 0..overlap_size {
        output_buffer[i] += (window[i] * state.last_global_gain + (1.0 - window[i]) * global_gain)
            * input_buffer[p_offset + i];
    }

    // Add direct signal (non-overlap region)
    for i in overlap_size..frame_size {
        output_buffer[i] = global_gain * (output_buffer[i] + input_buffer[p_offset + i]);
    }

    x_out[..frame_size].copy_from_slice(&output_buffer[..frame_size]);

    // Buffer update
    state.last_kernel[..kernel_size].copy_from_slice(&kernel_buffer[..kernel_size]);
    let hist_src = p_offset + frame_size - kernel_size - ADACOMB_MAX_LAG;
    state.history[..hist_len].copy_from_slice(&input_buffer[hist_src..hist_src + hist_len]);
    state.last_pitch_lag = pitch_lag;
    state.last_global_gain = global_gain;
}

/// Adaptive shaping filter.
///
/// Upstream C: dnn/nndsp.c:adashape_process_frame
#[allow(clippy::too_many_arguments)]
pub fn adashape_process_frame(
    state: &mut AdaShapeState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    alpha1f: &LinearLayer,
    alpha1t: &LinearLayer,
    alpha2: &LinearLayer,
    feature_dim: usize,
    frame_size: usize,
    avg_pool_k: usize,
    interpolate_k: usize,
) {
    assert!(frame_size.is_multiple_of(avg_pool_k));
    assert!(frame_size.is_multiple_of(interpolate_k));
    let hidden_dim = frame_size / interpolate_k;
    let tenv_size = frame_size / avg_pool_k;
    let f = 1.0f32 / avg_pool_k as f32;
    assert!(feature_dim + tenv_size + 1 < ADASHAPE_MAX_INPUT_DIM);

    let mut in_buffer = vec![0.0f32; ADASHAPE_MAX_INPUT_DIM + ADASHAPE_MAX_FRAME_SIZE];
    let mut out_buffer = vec![0.0f32; ADASHAPE_MAX_FRAME_SIZE];
    let mut tmp_buffer = vec![0.0f32; ADASHAPE_MAX_FRAME_SIZE];

    in_buffer[..feature_dim].copy_from_slice(&features[..feature_dim]);

    // Calculate temporal envelope
    let tenv = &mut in_buffer[feature_dim..];
    tenv[..tenv_size + 1].fill(0.0);
    let mut mean = 0.0f32;
    for i in 0..tenv_size {
        for k in 0..avg_pool_k {
            tenv[i] += x_in[i * avg_pool_k + k].abs();
        }
        tenv[i] = celt_log(tenv[i] * f + 1.52587890625e-05f32);
        mean += tenv[i];
    }
    mean /= tenv_size as f32;
    for i in 0..tenv_size {
        tenv[i] -= mean;
    }
    tenv[tenv_size] = mean;

    // Calculate temporal weights
    compute_generic_conv1d(
        alpha1f,
        &mut out_buffer,
        &mut state.conv_alpha1f_state,
        &in_buffer,
        feature_dim,
        ACTIVATION_LINEAR,
    );
    compute_generic_conv1d(
        alpha1t,
        &mut tmp_buffer,
        &mut state.conv_alpha1t_state,
        &in_buffer[feature_dim..],
        tenv_size + 1,
        ACTIVATION_LINEAR,
    );

    // Leaky ReLU
    for i in 0..hidden_dim {
        let tmp = out_buffer[i] + tmp_buffer[i];
        in_buffer[i] = if tmp >= 0.0 {
            tmp
        } else {
            (0.2 * tmp as f64) as f32
        };
    }

    compute_generic_conv1d(
        alpha2,
        &mut tmp_buffer,
        &mut state.conv_alpha2_state,
        &in_buffer,
        hidden_dim,
        ACTIVATION_LINEAR,
    );

    // Interpolation stage (new in 1.6.1)
    // When interpolate_k=1, hidden_dim=frame_size, inner loop runs once with alpha=1.0 → no-op
    for i in 0..hidden_dim {
        for k in 0..interpolate_k {
            let alpha = (k + 1) as f32 / interpolate_k as f32;
            out_buffer[i * interpolate_k + k] =
                alpha * tmp_buffer[i] + (1.0 - alpha) * state.interpolate_state[0];
        }
        state.interpolate_state[0] = tmp_buffer[i];
    }

    // Shape signal
    let exp_in = out_buffer.clone();
    compute_activation(&mut out_buffer, &exp_in, frame_size, ACTIVATION_EXP);
    for i in 0..frame_size {
        x_out[i] = out_buffer[i] * x_in[i];
    }
}
