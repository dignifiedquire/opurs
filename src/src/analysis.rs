use std::f64::consts::PI;

use num_traits::Zero;

use crate::celt::float_cast::float2int;
use crate::celt::kiss_fft::{kiss_fft_cpx, opus_fft_c};
use crate::celt::mathops::{celt_log, celt_log10, celt_sqrt, fast_atan2f};
use crate::celt::modes::OpusCustomMode;
use crate::opus_private::opus_select_arch;
use crate::src::mlp::analysis_mlp::run_analysis_mlp;
use crate::src::opus_encoder::is_digital_silence;

/// 30 ms at 24 kHz
pub const ANALYSIS_BUF_SIZE: i32 = 720;
pub const DETECT_SIZE: i32 = 100;
pub const NB_FRAMES: i32 = 8;
pub const NB_TBANDS: i32 = 18;

pub const NB_TONAL_SKIP_BANDS: i32 = 9;
pub const LEAKAGE_OFFSET: f32 = 2.5f32;
pub const LEAKAGE_SLOPE: f32 = 2.0f32;

const STD_FEATURE_BIAS: [f32; 9] = [
    5.684947f32,
    3.475288f32,
    1.770634f32,
    1.599784f32,
    3.773215f32,
    2.163313f32,
    1.260756f32,
    1.116868f32,
    1.918795f32,
];

const DCT_TABLE: [f32; 128] = [
    0.250000f32,
    0.250000f32,
    0.250000f32,
    0.250000f32,
    0.250000f32,
    0.250000f32,
    0.250000f32,
    0.250000f32,
    0.250000f32,
    0.250000f32,
    0.250000f32,
    0.250000f32,
    0.250000f32,
    0.250000f32,
    0.250000f32,
    0.250000f32,
    0.351851f32,
    0.338330f32,
    0.311806f32,
    0.273300f32,
    0.224292f32,
    0.166664f32,
    0.102631f32,
    0.034654f32,
    -0.034654f32,
    -0.102631f32,
    -0.166664f32,
    -0.224292f32,
    -0.273300f32,
    -0.311806f32,
    -0.338330f32,
    -0.351851f32,
    0.346760f32,
    0.293969f32,
    0.196424f32,
    0.068975f32,
    -0.068975f32,
    -0.196424f32,
    -0.293969f32,
    -0.346760f32,
    -0.346760f32,
    -0.293969f32,
    -0.196424f32,
    -0.068975f32,
    0.068975f32,
    0.196424f32,
    0.293969f32,
    0.346760f32,
    0.338330f32,
    0.224292f32,
    0.034654f32,
    -0.166664f32,
    -0.311806f32,
    -0.351851f32,
    -0.273300f32,
    -0.102631f32,
    0.102631f32,
    0.273300f32,
    0.351851f32,
    0.311806f32,
    0.166664f32,
    -0.034654f32,
    -0.224292f32,
    -0.338330f32,
    0.326641f32,
    0.135299f32,
    -0.135299f32,
    -0.326641f32,
    -0.326641f32,
    -0.135299f32,
    0.135299f32,
    0.326641f32,
    0.326641f32,
    0.135299f32,
    -0.135299f32,
    -0.326641f32,
    -0.326641f32,
    -0.135299f32,
    0.135299f32,
    0.326641f32,
    0.311806f32,
    0.034654f32,
    -0.273300f32,
    -0.338330f32,
    -0.102631f32,
    0.224292f32,
    0.351851f32,
    0.166664f32,
    -0.166664f32,
    -0.351851f32,
    -0.224292f32,
    0.102631f32,
    0.338330f32,
    0.273300f32,
    -0.034654f32,
    -0.311806f32,
    0.293969f32,
    -0.068975f32,
    -0.346760f32,
    -0.196424f32,
    0.196424f32,
    0.346760f32,
    0.068975f32,
    -0.293969f32,
    -0.293969f32,
    0.068975f32,
    0.346760f32,
    0.196424f32,
    -0.196424f32,
    -0.346760f32,
    -0.068975f32,
    0.293969f32,
    0.273300f32,
    -0.166664f32,
    -0.338330f32,
    0.034654f32,
    0.351851f32,
    0.102631f32,
    -0.311806f32,
    -0.224292f32,
    0.224292f32,
    0.311806f32,
    -0.102631f32,
    -0.351851f32,
    -0.034654f32,
    0.338330f32,
    0.166664f32,
    -0.273300f32,
];
const ANALYSIS_WINDOW: [f32; 240] = [
    0.000043f32,
    0.000171f32,
    0.000385f32,
    0.000685f32,
    0.001071f32,
    0.001541f32,
    0.002098f32,
    0.002739f32,
    0.003466f32,
    0.004278f32,
    0.005174f32,
    0.006156f32,
    0.007222f32,
    0.008373f32,
    0.009607f32,
    0.010926f32,
    0.012329f32,
    0.013815f32,
    0.015385f32,
    0.017037f32,
    0.018772f32,
    0.020590f32,
    0.022490f32,
    0.024472f32,
    0.026535f32,
    0.028679f32,
    0.030904f32,
    0.033210f32,
    0.035595f32,
    0.038060f32,
    0.040604f32,
    0.043227f32,
    0.045928f32,
    0.048707f32,
    0.051564f32,
    0.054497f32,
    0.057506f32,
    0.060591f32,
    0.063752f32,
    0.066987f32,
    0.070297f32,
    0.073680f32,
    0.077136f32,
    0.080665f32,
    0.084265f32,
    0.087937f32,
    0.091679f32,
    0.095492f32,
    0.099373f32,
    0.103323f32,
    0.107342f32,
    0.111427f32,
    0.115579f32,
    0.119797f32,
    0.124080f32,
    0.128428f32,
    0.132839f32,
    0.137313f32,
    0.141849f32,
    0.146447f32,
    0.151105f32,
    0.155823f32,
    0.160600f32,
    0.165435f32,
    0.170327f32,
    0.175276f32,
    0.180280f32,
    0.185340f32,
    0.190453f32,
    0.195619f32,
    0.200838f32,
    0.206107f32,
    0.211427f32,
    0.216797f32,
    0.222215f32,
    0.227680f32,
    0.233193f32,
    0.238751f32,
    0.244353f32,
    0.250000f32,
    0.255689f32,
    0.261421f32,
    0.267193f32,
    0.273005f32,
    0.278856f32,
    0.284744f32,
    0.290670f32,
    0.296632f32,
    0.302628f32,
    0.308658f32,
    0.314721f32,
    0.320816f32,
    0.326941f32,
    0.333097f32,
    0.339280f32,
    0.345492f32,
    0.351729f32,
    0.357992f32,
    0.364280f32,
    0.370590f32,
    0.376923f32,
    0.383277f32,
    0.389651f32,
    0.396044f32,
    0.402455f32,
    0.408882f32,
    0.415325f32,
    0.421783f32,
    0.428254f32,
    0.434737f32,
    0.441231f32,
    0.447736f32,
    0.454249f32,
    0.460770f32,
    0.467298f32,
    0.473832f32,
    0.480370f32,
    0.486912f32,
    0.493455f32,
    0.500000f32,
    0.506545f32,
    0.513088f32,
    0.519630f32,
    0.526168f32,
    0.532702f32,
    0.539230f32,
    0.545751f32,
    0.552264f32,
    0.558769f32,
    0.565263f32,
    0.571746f32,
    0.578217f32,
    0.584675f32,
    0.591118f32,
    0.597545f32,
    0.603956f32,
    0.610349f32,
    0.616723f32,
    0.623077f32,
    0.629410f32,
    0.635720f32,
    0.642008f32,
    0.648271f32,
    0.654508f32,
    0.660720f32,
    0.666903f32,
    0.673059f32,
    0.679184f32,
    0.685279f32,
    0.691342f32,
    0.697372f32,
    0.703368f32,
    0.709330f32,
    0.715256f32,
    0.721144f32,
    0.726995f32,
    0.732807f32,
    0.738579f32,
    0.744311f32,
    0.750000f32,
    0.755647f32,
    0.761249f32,
    0.766807f32,
    0.772320f32,
    0.777785f32,
    0.783203f32,
    0.788573f32,
    0.793893f32,
    0.799162f32,
    0.804381f32,
    0.809547f32,
    0.814660f32,
    0.819720f32,
    0.824724f32,
    0.829673f32,
    0.834565f32,
    0.839400f32,
    0.844177f32,
    0.848895f32,
    0.853553f32,
    0.858151f32,
    0.862687f32,
    0.867161f32,
    0.871572f32,
    0.875920f32,
    0.880203f32,
    0.884421f32,
    0.888573f32,
    0.892658f32,
    0.896677f32,
    0.900627f32,
    0.904508f32,
    0.908321f32,
    0.912063f32,
    0.915735f32,
    0.919335f32,
    0.922864f32,
    0.926320f32,
    0.929703f32,
    0.933013f32,
    0.936248f32,
    0.939409f32,
    0.942494f32,
    0.945503f32,
    0.948436f32,
    0.951293f32,
    0.954072f32,
    0.956773f32,
    0.959396f32,
    0.961940f32,
    0.964405f32,
    0.966790f32,
    0.969096f32,
    0.971321f32,
    0.973465f32,
    0.975528f32,
    0.977510f32,
    0.979410f32,
    0.981228f32,
    0.982963f32,
    0.984615f32,
    0.986185f32,
    0.987671f32,
    0.989074f32,
    0.990393f32,
    0.991627f32,
    0.992778f32,
    0.993844f32,
    0.994826f32,
    0.995722f32,
    0.996534f32,
    0.997261f32,
    0.997902f32,
    0.998459f32,
    0.998929f32,
    0.999315f32,
    0.999615f32,
    0.999829f32,
    0.999957f32,
    1.000000f32,
];
const TBANDS: [i32; 19] = [
    4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160, 192, 240,
];

#[derive(Copy, Clone, Default)]
pub struct AnalysisInfo {
    pub valid: i32,
    pub tonality: f32,
    pub tonality_slope: f32,
    pub noisiness: f32,
    pub activity: f32,
    pub music_prob: f32,
    pub music_prob_min: f32,
    pub music_prob_max: f32,
    pub bandwidth: i32,
    pub activity_probability: f32,
    pub max_pitch_ratio: f32,
    pub leak_boost: [u8; 19],
}

pub const LEAK_BANDS: i32 = 19;

pub type DownmixFn<T> = Option<fn(&[T], &mut [f32], i32, i32, i32, i32, i32) -> ()>;

#[derive(Copy, Clone)]
pub struct TonalityAnalysisState {
    #[allow(dead_code)]
    pub arch: i32,
    pub application: i32,
    pub Fs: i32,
    pub angle: [f32; 240],
    pub d_angle: [f32; 240],
    pub d2_angle: [f32; 240],
    pub inmem: [f32; 720],
    /// number of usable samples in the buffer
    pub mem_fill: i32,
    pub prev_band_tonality: [f32; 18],
    pub prev_tonality: f32,
    pub prev_bandwidth: i32,
    pub E: [[f32; 18]; 8],
    pub logE: [[f32; 18]; 8],
    pub lowE: [f32; 18],
    pub highE: [f32; 18],
    pub meanE: [f32; 19],
    pub mem: [f32; 32],
    pub cmean: [f32; 8],
    pub std: [f32; 9],
    pub Etracker: f32,
    pub lowECount: f32,
    pub E_count: i32,
    pub count: i32,
    pub analysis_offset: i32,
    pub write_pos: i32,
    pub read_pos: i32,
    pub read_subframe: i32,
    pub hp_ener_accum: f32,
    pub initialized: i32,
    pub rnn_state: [f32; 24],
    pub downmix_state: [f32; 3],
    pub info: [AnalysisInfo; 100],
}

/// Inputs
/// - state:  State vector [ 2 ]
/// - out:    Output signal [ floor(len/2) ]
/// - input:  Input signal [ len ]
/// - in_len: Number of input samples
fn silk_resampler_down2_hp(
    state: &mut [f32],
    out: &mut [f32],
    input: &[f32],
    in_len: usize,
) -> f32 {
    let len2 = in_len / 2;

    let mut hp_ener: f32 = 0.;

    // Internal variables and state are in Q10 format
    for k in 0..len2 {
        // Convert to Q10
        let in32 = input[2 * k];

        // All-pass section for even input sample
        let mut Y = in32 - state[0];
        let mut X = 0.6074371 * Y;
        let mut out32 = state[0] + X;
        state[0] = in32 + X;
        let mut out32_hp = out32;

        // Convert to Q10
        let in32 = input[2 * k + 1];

        // All-pass section for odd input sample, and add to output of previous section
        Y = in32 - state[1];
        X = 0.15063 * Y;
        out32 += state[1];
        out32 += X;
        state[1] = in32 + X;

        Y = -in32 - state[2];
        X = 0.15063 * Y;
        out32_hp += state[2];
        out32_hp = out32_hp + X;
        state[2] = -in32 + X;

        hp_ener += out32_hp * out32_hp;

        // Add, convert back to int16 and store to output
        out[k as usize] = 0.5f32 * out32;
    }

    hp_ener
}

fn downmix_and_resample<T>(
    downmix: DownmixFn<T>,
    _x: &[T],
    y: &mut [f32],
    S: &mut [f32],
    mut subframe: usize,
    mut offset: i32,
    c1: i32,
    c2: i32,
    C: i32,
    fs: i32,
) -> f32 {
    if subframe == 0 {
        return 0.;
    }

    let mut scale: f32 = 0.;
    let mut ret: f32 = 0.;
    if fs == 48_000 {
        subframe *= 2;
        offset *= 2;
    } else if fs == 16_000 {
        subframe = subframe * 2 / 3;
        offset = offset * 2 / 3;
    }
    let mut tmp: Vec<f32> = vec![0.; subframe];

    downmix.expect("non-null function pointer")(_x, &mut tmp, subframe as _, offset, c1, c2, C);

    scale = 1.0 / 32_768.;

    if c2 == -2 {
        scale /= C as f32;
    } else if c2 > -1 {
        scale /= 2 as f32;
    }
    for j in 0..subframe {
        tmp[j] *= scale;
    }
    if fs == 48_000 {
        ret = silk_resampler_down2_hp(S, y, &tmp, subframe);
    } else if fs == 24_000 {
        // OPUS_COPY(y, tmp, subframe)
        y[..subframe].copy_from_slice(&tmp[..subframe]);
    } else if fs == 16_000 {
        let mut tmp3x: Vec<f32> = vec![0.; 3 * subframe];

        // Don't do this at home! This resampler is horrible and it's only (barely)
        // usable for the purpose of the analysis because we don't care about all
        // the aliasing between 8 kHz and 12 kHz. */
        for j in 0..subframe {
            tmp3x[3 * j] = tmp[j];
            tmp3x[3 * j + 1] = tmp[j];
            tmp3x[3 * j + 2] = tmp[j];
        }
        silk_resampler_down2_hp(S, y, &tmp3x, 3 * subframe);
    }

    ret
}

impl TonalityAnalysisState {
    pub fn new(fs: i32) -> Self {
        Self {
            arch: opus_select_arch(),
            Fs: fs,
            application: 0,
            angle: [0.; 240],
            d_angle: [0.; 240],
            d2_angle: [0.; 240],
            inmem: [0.; 720],
            mem_fill: 0,
            prev_band_tonality: [0.; 18],
            prev_tonality: 0.,
            prev_bandwidth: 0,
            E: [[0.; 18]; 8],
            logE: [[0.; 18]; 8],
            lowE: [0.; 18],
            highE: [0.; 18],
            meanE: [0.; 19],
            mem: [0.; 32],
            cmean: [0.; 8],
            std: [0.; 9],
            Etracker: 0.,
            lowECount: 0.,
            E_count: 0,
            count: 0,
            analysis_offset: 0,
            write_pos: 0,
            read_pos: 0,
            read_subframe: 0,
            hp_ener_accum: 0.,
            initialized: 0,
            rnn_state: [0.; 24],
            downmix_state: [0.; 3],
            info: [AnalysisInfo::default(); 100],
        }
    }

    pub fn get_info(&mut self, len: i32) -> AnalysisInfo {
        let mut pos = self.read_pos;
        let mut curr_lookahead = self.write_pos - self.read_pos;
        if curr_lookahead < 0 {
            curr_lookahead += DETECT_SIZE;
        }
        self.read_subframe += len / (self.Fs / 400);
        while self.read_subframe >= 8 {
            self.read_subframe -= 8;
            self.read_pos += 1;
        }
        if self.read_pos >= DETECT_SIZE {
            self.read_pos -= DETECT_SIZE;
        }
        // On long frames, look at the second analysis window rather than the first.
        if len > self.Fs / 50 && pos != self.write_pos {
            pos += 1;
            if pos == DETECT_SIZE {
                pos = 0;
            }
        }
        if pos == self.write_pos {
            pos -= 1;
        }
        if pos < 0 {
            pos = DETECT_SIZE - 1;
        }
        let pos0 = pos;

        let mut info = self.info[pos as usize].clone();
        if info.valid == 0 {
            return info;
        }
        let mut tonality_avg = info.tonality;
        let mut tonality_max = tonality_avg;
        let mut tonality_count = 1;

        // Look at the neighbouring frames and pick largest bandwidth found (to be safe).
        let mut bandwidth_span = 6;
        // If possible, look ahead for a tone to compensate for the delay in the tone detector.
        for _i in 0..3 {
            pos += 1;
            if pos == DETECT_SIZE {
                pos = 0;
            }
            if pos == self.write_pos {
                break;
            }
            tonality_max = if tonality_max > self.info[pos as usize].tonality {
                tonality_max
            } else {
                self.info[pos as usize].tonality
            };
            tonality_avg += self.info[pos as usize].tonality;
            tonality_count += 1;
            info.bandwidth = if info.bandwidth > self.info[pos as usize].bandwidth {
                info.bandwidth
            } else {
                self.info[pos as usize].bandwidth
            };
            bandwidth_span -= 1;
        }
        pos = pos0;

        // Look back in time to see if any has a wider bandwidth than the current frame.
        for _i in 0..bandwidth_span {
            pos -= 1;
            if pos < 0 {
                pos = DETECT_SIZE - 1;
            }
            if pos == self.write_pos {
                break;
            }
            info.bandwidth = if info.bandwidth > self.info[pos as usize].bandwidth {
                info.bandwidth
            } else {
                self.info[pos as usize].bandwidth
            };
        }
        // If we have enough look-ahead, compensate for the ~5-frame delay in the music prob and
        // ~1 frame delay in the VAD prob.
        info.tonality = if tonality_avg / tonality_count as f32 > tonality_max - 0.2 {
            tonality_avg / tonality_count as f32
        } else {
            tonality_max - 0.2
        };
        let mut vpos = pos0;
        let mut mpos = vpos;
        if curr_lookahead > 15 {
            mpos += 5;
            if mpos >= DETECT_SIZE {
                mpos -= DETECT_SIZE;
            }
            vpos += 1;
            if vpos >= DETECT_SIZE {
                vpos -= DETECT_SIZE;
            }
        }

        // The following calculations attempt to minimize a "badness function"
        // for the transition. When switching from speech to music, the badness
        // of switching at frame k is
        // b_k = S*v_k + \sum_{i=0}^{k-1} v_i*(p_i - T)
        // where
        // v_i is the activity probability (VAD) at frame i,
        // p_i is the music probability at frame i
        // T is the probability threshold for switching
        // S is the penalty for switching during active audio rather than silence
        // the current frame has index i=0
        //
        // Rather than apply badness to directly decide when to switch, what we compute
        // instead is the threshold for which the optimal switching point is now. When
        // considering whether to switch now (frame 0) or at frame k, we have:
        // S*v_0 = S*v_k + \sum_{i=0}^{k-1} v_i*(p_i - T)
        // which gives us:
        // T = ( \sum_{i=0}^{k-1} v_i*p_i + S*(v_k-v_0) ) / ( \sum_{i=0}^{k-1} v_i )
        // We take the min threshold across all positive values of k (up to the maximum
        // amount of lookahead we have) to give us the threshold for which the current
        // frame is the optimal switch point.
        //
        // The last step is that we need to consider whether we want to switch at all.
        // For that we use the average of the music probability over the entire window.
        // If the threshold is higher than that average we're not going to
        // switch, so we compute a min with the average as well. The result of all these
        // min operations is music_prob_min, which gives the threshold for switching to music
        // if we're currently encoding for speech.
        //
        // We do the exact opposite to compute music_prob_max which is used for switching
        // from music to speech.

        let mut prob_min = 1.0;
        let mut prob_max = 0.0;
        let vad_prob = self.info[vpos as usize].activity_probability;
        let mut prob_count = if 0.1 > vad_prob { 0.1 } else { vad_prob };
        let mut prob_avg =
            (if 0.1 > vad_prob { 0.1 } else { vad_prob }) * self.info[mpos as usize].music_prob;
        loop {
            let mut pos_vad: f32 = 0.;
            mpos += 1;
            if mpos == DETECT_SIZE {
                mpos = 0;
            }
            if mpos == self.write_pos {
                break;
            }
            vpos += 1;
            if vpos == DETECT_SIZE {
                vpos = 0;
            }
            if vpos == self.write_pos {
                break;
            }
            pos_vad = self.info[vpos as usize].activity_probability;
            prob_min = if (prob_avg - 10 as f32 * (vad_prob - pos_vad)) / prob_count < prob_min {
                (prob_avg - 10 as f32 * (vad_prob - pos_vad)) / prob_count
            } else {
                prob_min
            };
            prob_max = if (prob_avg + 10 as f32 * (vad_prob - pos_vad)) / prob_count > prob_max {
                (prob_avg + 10 as f32 * (vad_prob - pos_vad)) / prob_count
            } else {
                prob_max
            };
            prob_count += if 0.1f32 > pos_vad { 0.1f32 } else { pos_vad };
            prob_avg += (if 0.1f32 > pos_vad { 0.1f32 } else { pos_vad })
                * self.info[mpos as usize].music_prob;
        }
        info.music_prob = prob_avg / prob_count;
        prob_min = if prob_avg / prob_count < prob_min {
            prob_avg / prob_count
        } else {
            prob_min
        };
        prob_max = if prob_avg / prob_count > prob_max {
            prob_avg / prob_count
        } else {
            prob_max
        };
        prob_min = if prob_min > 0.0f32 { prob_min } else { 0.0f32 };
        prob_max = if prob_max < 1.0f32 { prob_max } else { 1.0f32 };

        // If we don't have enough look-ahead, do our best to make a decent decision.
        if curr_lookahead < 10 {
            let mut pmin = prob_min;
            let mut pmax = prob_max;
            pos = pos0;

            // Look for min/max in the past.
            let mut i = 0;
            while i
                < (if (self.count - 1) < 15 {
                    self.count - 1
                } else {
                    15
                })
            {
                pos -= 1;
                if pos < 0 {
                    pos = DETECT_SIZE - 1;
                }
                pmin = if pmin < self.info[pos as usize].music_prob {
                    pmin
                } else {
                    self.info[pos as usize].music_prob
                };
                pmax = if pmax > self.info[pos as usize].music_prob {
                    pmax
                } else {
                    self.info[pos as usize].music_prob
                };
                i += 1;
            }
            // Bias against switching on active audio.
            pmin = if 0.0 > pmin - 0.1 * vad_prob {
                0.0
            } else {
                pmin - 0.1 * vad_prob
            };
            pmax = if 1.0 < pmax + 0.1 * vad_prob {
                1.0
            } else {
                pmax + 0.1 * vad_prob
            };
            prob_min += (1.0f32 - 0.1f32 * curr_lookahead as f32) * (pmin - prob_min);
            prob_max += (1.0f32 - 0.1f32 * curr_lookahead as f32) * (pmax - prob_max);
        }
        info.music_prob_min = prob_min;
        info.music_prob_max = prob_max;

        info
    }

    fn analysis<T>(
        &mut self,
        celt_mode: &OpusCustomMode,
        x: &[T],
        mut len: i32,
        mut offset: i32,
        c1: i32,
        c2: i32,
        C: i32,
        lsb_depth: i32,
        downmix: DownmixFn<T>,
    ) {
        let mut i: i32 = 0;
        let mut b: i32 = 0;
        let N: i32 = 480;
        let N2: i32 = 240;
        let A = &mut self.angle;
        let dA = &mut self.d_angle;
        let d2A = &mut self.d2_angle;
        let mut band_tonality: [f32; 18] = [0.; 18];
        let mut logE: [f32; 18] = [0.; 18];
        let mut BFCC: [f32; 8] = [0.; 8];
        let mut features: [f32; 25] = [0.; 25];
        let mut frame_tonality: f32 = 0.;
        let mut max_frame_tonality: f32 = 0.;
        let mut frame_noisiness: f32 = 0.;
        let pi4: f32 = (PI * PI * PI * PI) as f32;
        let mut slope: f32 = 0 as f32;
        let mut frame_stationarity: f32 = 0.;
        let mut relativeE: f32 = 0.;
        let mut alpha: f32 = 0.;
        let mut alphaE: f32 = 0.;
        let mut alphaE2: f32 = 0.;
        let mut frame_loudness: f32 = 0.;
        let mut bandwidth_mask: f32 = 0.;
        let mut is_masked: [i32; 19] = [0; 19];
        let mut bandwidth: i32 = 0;
        let mut maxE: f32 = 0 as f32;
        let mut noise_floor: f32 = 0.;
        let mut remaining: i32 = 0;
        let mut hp_ener: f32 = 0.;
        let mut tonality2: [f32; 240] = [0.; 240];
        let mut midE: [f32; 8] = [0.; 8];
        let mut spec_variability: f32 = 0 as f32;
        let mut band_log2: [f32; 19] = [0.; 19];
        let mut leakage_from: [f32; 19] = [0.; 19];
        let mut leakage_to: [f32; 19] = [0.; 19];
        let mut below_max_pitch: f32 = 0.;
        let mut above_max_pitch: f32 = 0.;
        let mut is_silence: i32 = 0;

        if self.initialized == 0 {
            self.mem_fill = 240;
            self.initialized = 1;
        }
        alpha = 1.0f32
            / (if (10) < 1 + self.count {
                10
            } else {
                1 + self.count
            }) as f32;
        alphaE = 1.0f32
            / (if (25) < 1 + self.count {
                25
            } else {
                1 + self.count
            }) as f32;
        alphaE2 = 1.0f32
            / (if (100) < 1 + self.count {
                100
            } else {
                1 + self.count
            }) as f32;
        if self.count <= 1 {
            alphaE2 = 1 as f32;
        }
        if self.Fs == 48000 {
            len /= 2;
            offset /= 2;
        } else if self.Fs == 16000 {
            len = 3 * len / 2;
            offset = 3 * offset / 2;
        }
        let kfft = (*celt_mode).mdct.kfft[0];
        self.hp_ener_accum += downmix_and_resample(
            downmix,
            x,
            &mut self.inmem[self.mem_fill as usize..],
            &mut self.downmix_state,
            if len < 720 - self.mem_fill {
                len as usize
            } else {
                720 - self.mem_fill as usize
            },
            offset,
            c1,
            c2,
            C,
            self.Fs,
        );
        if self.mem_fill + len < ANALYSIS_BUF_SIZE {
            self.mem_fill += len;
            return;
        }
        hp_ener = self.hp_ener_accum;
        let info_pos = self.write_pos as usize;
        self.write_pos = self.write_pos + 1;
        if self.write_pos >= DETECT_SIZE {
            self.write_pos -= DETECT_SIZE;
        }
        is_silence = is_digital_silence(&self.inmem, 720, 1, lsb_depth);
        let mut in_0: [kiss_fft_cpx; 480] = [kiss_fft_cpx::zero(); 480];
        let mut out: [kiss_fft_cpx; 480] = [kiss_fft_cpx::zero(); 480];
        let mut tonality: [f32; 240] = [0.; 240];
        let mut noisiness: [f32; 240] = [0.; 240];
        i = 0;
        while i < N2 {
            let w: f32 = ANALYSIS_WINDOW[i as usize];
            in_0[i as usize].re = w * self.inmem[i as usize];
            in_0[i as usize].im = w * self.inmem[(N2 + i) as usize];
            in_0[(N - i - 1) as usize].re = w * self.inmem[(N - i - 1) as usize];
            in_0[(N - i - 1) as usize].im = w * self.inmem[(N + N2 - i - 1) as usize];
            i += 1;
        }
        // OPUS_MOVE(tonal->inmem, tonal->inmem+ANALYSIS_BUF_SIZE-240, 240);
        let start = ANALYSIS_BUF_SIZE as usize - 240;
        self.inmem.copy_within(start..start + 240, 0);
        remaining = len - (ANALYSIS_BUF_SIZE - self.mem_fill);
        self.hp_ener_accum = downmix_and_resample(
            downmix,
            x,
            &mut self.inmem[240..],
            &mut self.downmix_state,
            remaining as usize,
            offset + ANALYSIS_BUF_SIZE - self.mem_fill,
            c1,
            c2,
            C,
            self.Fs,
        );
        self.mem_fill = 240 + remaining;
        if is_silence != 0 {
            let mut prev_pos: i32 = self.write_pos - 2;
            if prev_pos < 0 {
                prev_pos += DETECT_SIZE;
            }
            self.info[info_pos] = self.info[prev_pos as usize].clone();

            return;
        }
        opus_fft_c(kfft, &in_0, &mut out);

        let info = &mut self.info[info_pos];
        if out[0].re != out[0].re {
            info.valid = 0;
            return;
        }
        i = 1;
        while i < N2 {
            let mut X1r: f32 = 0.;
            let mut X2r: f32 = 0.;
            let mut X1i: f32 = 0.;
            let mut X2i: f32 = 0.;
            let mut angle: f32 = 0.;
            let mut d_angle: f32 = 0.;
            let mut d2_angle: f32 = 0.;
            let mut angle2: f32 = 0.;
            let mut d_angle2: f32 = 0.;
            let mut d2_angle2: f32 = 0.;
            let mut mod1: f32 = 0.;
            let mut mod2: f32 = 0.;
            let mut avg_mod: f32 = 0.;
            X1r = out[i as usize].re + out[(N - i) as usize].re;
            X1i = out[i as usize].im - out[(N - i) as usize].im;
            X2r = out[i as usize].im + out[(N - i) as usize].im;
            X2i = out[(N - i) as usize].re - out[i as usize].re;
            angle = (0.5f32 as f64 / PI) as f32 * fast_atan2f(X1i, X1r);
            d_angle = angle - A[i as usize];
            d2_angle = d_angle - dA[i as usize];
            angle2 = (0.5f32 as f64 / PI) as f32 * fast_atan2f(X2i, X2r);
            d_angle2 = angle2 - angle;
            d2_angle2 = d_angle2 - d_angle;
            mod1 = d2_angle - float2int(d2_angle) as f32;
            noisiness[i as usize] = (mod1).abs();
            mod1 *= mod1;
            mod1 *= mod1;
            mod2 = d2_angle2 - float2int(d2_angle2) as f32;
            noisiness[i as usize] += (mod2).abs();
            mod2 *= mod2;
            mod2 *= mod2;
            avg_mod = 0.25f32 * (d2A[i as usize] + mod1 + 2 as f32 * mod2);
            tonality[i as usize] = 1.0f32 / (1.0f32 + 40.0f32 * 16.0f32 * pi4 * avg_mod) - 0.015f32;
            tonality2[i as usize] = 1.0f32 / (1.0f32 + 40.0f32 * 16.0f32 * pi4 * mod2) - 0.015f32;
            A[i as usize] = angle2;
            dA[i as usize] = d_angle2;
            d2A[i as usize] = mod2;
            i += 1;
        }
        i = 2;
        while i < N2 - 1 {
            let tt: f32 = if tonality2[i as usize]
                < (if tonality2[(i - 1) as usize] > tonality2[(i + 1) as usize] {
                    tonality2[(i - 1) as usize]
                } else {
                    tonality2[(i + 1) as usize]
                }) {
                tonality2[i as usize]
            } else if tonality2[(i - 1) as usize] > tonality2[(i + 1) as usize] {
                tonality2[(i - 1) as usize]
            } else {
                tonality2[(i + 1) as usize]
            };
            tonality[i as usize] = 0.9f32
                * (if tonality[i as usize] > tt - 0.1f32 {
                    tonality[i as usize]
                } else {
                    tt - 0.1f32
                });
            i += 1;
        }
        frame_tonality = 0 as f32;
        max_frame_tonality = 0 as f32;
        (*info).activity = 0 as f32;
        frame_noisiness = 0 as f32;
        frame_stationarity = 0 as f32;
        if self.count == 0 {
            b = 0;
            while b < NB_TBANDS {
                self.lowE[b as usize] = 1e10f64 as f32;
                self.highE[b as usize] = -1e10f64 as f32;
                b += 1;
            }
        }
        relativeE = 0 as f32;
        frame_loudness = 0 as f32;
        let mut E: f32 = 0 as f32;
        let mut X1r_0: f32 = 0.;
        let mut X2r_0: f32 = 0.;
        X1r_0 = 2 as f32 * out[0 as usize].re;
        X2r_0 = 2 as f32 * out[0 as usize].im;
        E = X1r_0 * X1r_0 + X2r_0 * X2r_0;
        i = 1;
        while i < 4 {
            let binE: f32 = out[i as usize].re * out[i as usize].re
                + out[(N - i) as usize].re * out[(N - i) as usize].re
                + out[i as usize].im * out[i as usize].im
                + out[(N - i) as usize].im * out[(N - i) as usize].im;
            E += binE;
            i += 1;
        }
        // E = E;
        band_log2[0 as usize] = 0.5f32 * std::f32::consts::LOG2_E * celt_log(E + 1e-10f32);
        b = 0;
        while b < NB_TBANDS {
            let mut E_0: f32 = 0 as f32;
            let mut tE: f32 = 0 as f32;
            let mut nE: f32 = 0 as f32;
            let mut L1: f32 = 0.;
            let mut L2: f32 = 0.;
            let mut stationarity: f32 = 0.;
            i = TBANDS[b as usize];
            while i < TBANDS[(b + 1) as usize] {
                let binE_0: f32 = out[i as usize].re * out[i as usize].re
                    + out[(N - i) as usize].re * out[(N - i) as usize].re
                    + out[i as usize].im * out[i as usize].im
                    + out[(N - i) as usize].im * out[(N - i) as usize].im;
                // binE_0 = binE_0;
                E_0 += binE_0;
                tE += binE_0
                    * (if 0 as f32 > tonality[i as usize] {
                        0 as f32
                    } else {
                        tonality[i as usize]
                    });
                nE += binE_0 * 2.0f32 * (0.5f32 - noisiness[i as usize]);
                i += 1;
            }
            if !(E_0 < 1e9f32) || E_0 != E_0 {
                (*info).valid = 0;
                return;
            }
            self.E[self.E_count as usize][b as usize] = E_0;
            frame_noisiness += nE / (1e-15f32 + E_0);
            frame_loudness += celt_sqrt(E_0 + 1e-10f32);
            logE[b as usize] = celt_log(E_0 + 1e-10f32);
            band_log2[(b + 1) as usize] =
                0.5f32 * std::f32::consts::LOG2_E * celt_log(E_0 + 1e-10f32);
            self.logE[self.E_count as usize][b as usize] = logE[b as usize];
            if self.count == 0 {
                self.lowE[b as usize] = logE[b as usize];
                self.highE[b as usize] = self.lowE[b as usize];
            }
            if self.highE[b as usize] as f64 > self.lowE[b as usize] as f64 + 7.5f64 {
                if self.highE[b as usize] - logE[b as usize]
                    > logE[b as usize] - self.lowE[b as usize]
                {
                    self.highE[b as usize] -= 0.01f32;
                } else {
                    self.lowE[b as usize] += 0.01f32;
                }
            }
            if logE[b as usize] > self.highE[b as usize] {
                self.highE[b as usize] = logE[b as usize];
                self.lowE[b as usize] =
                    if self.highE[b as usize] - 15 as f32 > self.lowE[b as usize] {
                        self.highE[b as usize] - 15 as f32
                    } else {
                        self.lowE[b as usize]
                    };
            } else if logE[b as usize] < self.lowE[b as usize] {
                self.lowE[b as usize] = logE[b as usize];
                self.highE[b as usize] =
                    if (self.lowE[b as usize] + 15 as f32) < self.highE[b as usize] {
                        self.lowE[b as usize] + 15 as f32
                    } else {
                        self.highE[b as usize]
                    };
            }
            relativeE += (logE[b as usize] - self.lowE[b as usize])
                / (1e-5f32 + (self.highE[b as usize] - self.lowE[b as usize]));
            L2 = 0 as f32;
            L1 = L2;
            i = 0;
            while i < NB_FRAMES {
                L1 += celt_sqrt(self.E[i as usize][b as usize]);
                L2 += self.E[i as usize][b as usize];
                i += 1;
            }
            // NB:
            //  because `1e-15` is specified without a suffix in the upstream,
            //   this addition is performed as f64 and `celt_sqrt` can't be used
            stationarity =
                0.99f32.min(L1 / (1e-15 + ((NB_FRAMES as f32 * L2) as f64)).sqrt() as f32);
            stationarity *= stationarity;
            stationarity *= stationarity;
            frame_stationarity += stationarity;
            band_tonality[b as usize] =
                if tE / (1e-15f32 + E_0) > stationarity * self.prev_band_tonality[b as usize] {
                    tE / (1e-15f32 + E_0)
                } else {
                    stationarity * self.prev_band_tonality[b as usize]
                };
            frame_tonality += band_tonality[b as usize];
            if b >= NB_TBANDS - NB_TONAL_SKIP_BANDS {
                frame_tonality -= band_tonality[(b - NB_TBANDS + NB_TONAL_SKIP_BANDS) as usize];
            }
            max_frame_tonality =
                if max_frame_tonality > (1.0f32 + 0.03f32 * (b - 18) as f32) * frame_tonality {
                    max_frame_tonality
                } else {
                    (1.0f32 + 0.03f32 * (b - 18) as f32) * frame_tonality
                };
            slope += band_tonality[b as usize] * (b - 8) as f32;
            self.prev_band_tonality[b as usize] = band_tonality[b as usize];
            b += 1;
        }
        leakage_from[0 as usize] = band_log2[0 as usize];
        leakage_to[0 as usize] = band_log2[0 as usize] - LEAKAGE_OFFSET;
        b = 1;
        while b < NB_TBANDS + 1 {
            let leak_slope: f32 =
                LEAKAGE_SLOPE * (TBANDS[b as usize] - TBANDS[(b - 1) as usize]) as f32 / 4 as f32;
            leakage_from[b as usize] =
                if leakage_from[(b - 1) as usize] + leak_slope < band_log2[b as usize] {
                    leakage_from[(b - 1) as usize] + leak_slope
                } else {
                    band_log2[b as usize]
                };
            leakage_to[b as usize] =
                if leakage_to[(b - 1) as usize] - leak_slope > band_log2[b as usize] - 2.5f32 {
                    leakage_to[(b - 1) as usize] - leak_slope
                } else {
                    band_log2[b as usize] - 2.5f32
                };
            b += 1;
        }
        b = NB_TBANDS - 2;
        while b >= 0 {
            let leak_slope_0: f32 =
                LEAKAGE_SLOPE * (TBANDS[(b + 1) as usize] - TBANDS[b as usize]) as f32 / 4 as f32;
            leakage_from[b as usize] =
                if leakage_from[(b + 1) as usize] + leak_slope_0 < leakage_from[b as usize] {
                    leakage_from[(b + 1) as usize] + leak_slope_0
                } else {
                    leakage_from[b as usize]
                };
            leakage_to[b as usize] =
                if leakage_to[(b + 1) as usize] - leak_slope_0 > leakage_to[b as usize] {
                    leakage_to[(b + 1) as usize] - leak_slope_0
                } else {
                    leakage_to[b as usize]
                };
            b -= 1;
        }
        assert!(18 + 1 <= 19);
        b = 0;
        while b < NB_TBANDS + 1 {
            let boost: f32 =
                (if 0 as f32 > leakage_to[b as usize] - band_log2[b as usize] {
                    0 as f32
                } else {
                    leakage_to[b as usize] - band_log2[b as usize]
                }) + (if 0 as f32 > band_log2[b as usize] - (leakage_from[b as usize] + 2.5f32) {
                    0 as f32
                } else {
                    band_log2[b as usize] - (leakage_from[b as usize] + 2.5f32)
                });
            (*info).leak_boost[b as usize] = (if (255) < (0.5 + (64.0 * boost)).floor() as i32 {
                255
            } else {
                (0.5 + (64.0 * boost)).floor() as i32
            }) as u8;
            b += 1;
        }
        while b < LEAK_BANDS {
            (*info).leak_boost[b as usize] = 0;
            b += 1;
        }
        i = 0;
        while i < NB_FRAMES {
            let mut j: i32 = 0;
            let mut mindist: f32 = 1e15f32;
            j = 0;
            while j < NB_FRAMES {
                let mut k: i32 = 0;
                let mut dist: f32 = 0 as f32;
                k = 0;
                while k < NB_TBANDS {
                    let mut tmp: f32 = 0.;
                    tmp = self.logE[i as usize][k as usize] - self.logE[j as usize][k as usize];
                    dist += tmp * tmp;
                    k += 1;
                }
                if j != i {
                    mindist = if mindist < dist { mindist } else { dist };
                }
                j += 1;
            }
            spec_variability += mindist;
            i += 1;
        }
        spec_variability = celt_sqrt(spec_variability / NB_FRAMES as f32 / NB_TBANDS as f32);
        bandwidth_mask = 0 as f32;
        bandwidth = 0;
        maxE = 0 as f32;
        noise_floor =
            5.7e-4f32 / ((1) << (if 0 > lsb_depth - 8 { 0 } else { lsb_depth - 8 })) as f32;
        noise_floor *= noise_floor;
        below_max_pitch = 0 as f32;
        above_max_pitch = 0 as f32;
        b = 0;
        while b < NB_TBANDS {
            let mut E_1: f32 = 0 as f32;
            let mut Em: f32 = 0.;
            let mut band_start: i32 = 0;
            let mut band_end: i32 = 0;
            band_start = TBANDS[b as usize];
            band_end = TBANDS[(b + 1) as usize];
            i = band_start;
            while i < band_end {
                let binE_1: f32 = out[i as usize].re * out[i as usize].re
                    + out[(N - i) as usize].re * out[(N - i) as usize].re
                    + out[i as usize].im * out[i as usize].im
                    + out[(N - i) as usize].im * out[(N - i) as usize].im;
                E_1 += binE_1;
                i += 1;
            }
            // E_1 = E_1;
            maxE = if maxE > E_1 { maxE } else { E_1 };
            if band_start < 64 {
                below_max_pitch += E_1;
            } else {
                above_max_pitch += E_1;
            }
            self.meanE[b as usize] = if (1 as f32 - alphaE2) * self.meanE[b as usize] > E_1 {
                (1 as f32 - alphaE2) * self.meanE[b as usize]
            } else {
                E_1
            };
            Em = if E_1 > self.meanE[b as usize] {
                E_1
            } else {
                self.meanE[b as usize]
            };
            if E_1 * 1e9f32 > maxE
                && (Em > 3 as f32 * noise_floor * (band_end - band_start) as f32
                    || E_1 > noise_floor * (band_end - band_start) as f32)
            {
                bandwidth = b + 1;
            }
            is_masked[b as usize] = (E_1
                < (if self.prev_bandwidth >= b + 1 {
                    0.01f32
                } else {
                    0.05f32
                }) * bandwidth_mask) as i32;
            bandwidth_mask = if 0.05f32 * bandwidth_mask > E_1 {
                0.05f32 * bandwidth_mask
            } else {
                E_1
            };
            b += 1;
        }
        if self.Fs == 48000 {
            let mut noise_ratio: f32 = 0.;
            let mut Em_0: f32 = 0.;
            let E_2: f32 = hp_ener * (1.0f32 / (60 * 60) as f32);
            noise_ratio = if self.prev_bandwidth == 20 {
                10.0f32
            } else {
                30.0f32
            };
            above_max_pitch += E_2;
            self.meanE[b as usize] = if (1 as f32 - alphaE2) * self.meanE[b as usize] > E_2 {
                (1 as f32 - alphaE2) * self.meanE[b as usize]
            } else {
                E_2
            };
            Em_0 = if E_2 > self.meanE[b as usize] {
                E_2
            } else {
                self.meanE[b as usize]
            };
            if Em_0 > 3 as f32 * noise_ratio * noise_floor * 160 as f32
                || E_2 > noise_ratio * noise_floor * 160 as f32
            {
                bandwidth = 20;
            }
            is_masked[b as usize] = (E_2
                < (if self.prev_bandwidth == 20 {
                    0.01f32
                } else {
                    0.05f32
                }) * bandwidth_mask) as i32;
        }
        if above_max_pitch > below_max_pitch {
            (*info).max_pitch_ratio = below_max_pitch / above_max_pitch;
        } else {
            (*info).max_pitch_ratio = 1 as f32;
        }
        if bandwidth == 20 && is_masked[NB_TBANDS as usize] != 0 {
            bandwidth -= 2;
        } else if bandwidth > 0
            && bandwidth <= NB_TBANDS
            && is_masked[(bandwidth - 1) as usize] != 0
        {
            bandwidth -= 1;
        }
        if self.count <= 2 {
            bandwidth = 20;
        }
        frame_loudness = 20f32 * celt_log10(frame_loudness);
        self.Etracker = if self.Etracker - 0.003f32 > frame_loudness {
            self.Etracker - 0.003f32
        } else {
            frame_loudness
        };
        self.lowECount *= 1 as f32 - alphaE;
        if frame_loudness < self.Etracker - 30 as f32 {
            self.lowECount += alphaE;
        }
        i = 0;
        while i < 8 {
            let mut sum: f32 = 0 as f32;
            b = 0;
            while b < 16 {
                sum += DCT_TABLE[(i * 16 + b) as usize] * logE[b as usize];
                b += 1;
            }
            BFCC[i as usize] = sum;
            i += 1;
        }
        i = 0;
        while i < 8 {
            let mut sum_0: f32 = 0 as f32;
            b = 0;
            while b < 16 {
                sum_0 += DCT_TABLE[(i * 16 + b) as usize]
                    * 0.5f32
                    * (self.highE[b as usize] + self.lowE[b as usize]);
                b += 1;
            }
            midE[i as usize] = sum_0;
            i += 1;
        }
        frame_stationarity /= NB_TBANDS as f32;
        relativeE /= NB_TBANDS as f32;
        if self.count < 10 {
            relativeE = 0.5f32;
        }
        frame_noisiness /= NB_TBANDS as f32;
        (*info).activity = frame_noisiness + (1 as f32 - frame_noisiness) * relativeE;
        frame_tonality = max_frame_tonality / (NB_TBANDS - NB_TONAL_SKIP_BANDS) as f32;
        frame_tonality = if frame_tonality > self.prev_tonality * 0.8f32 {
            frame_tonality
        } else {
            self.prev_tonality * 0.8f32
        };
        self.prev_tonality = frame_tonality;
        slope /= (8 * 8) as f32;
        (*info).tonality_slope = slope;
        self.E_count = (self.E_count + 1) % NB_FRAMES;
        self.count = if (self.count + 1) < 10000 {
            self.count + 1
        } else {
            10000
        };
        (*info).tonality = frame_tonality;
        i = 0;
        while i < 4 {
            features[i as usize] = -0.12299f32 * (BFCC[i as usize] + self.mem[(i + 24) as usize])
                + 0.49195f32 * (self.mem[i as usize] + self.mem[(i + 16) as usize])
                + 0.69693f32 * self.mem[(i + 8) as usize]
                - 1.4349f32 * self.cmean[i as usize];
            i += 1;
        }
        i = 0;
        while i < 4 {
            self.cmean[i as usize] =
                (1 as f32 - alpha) * self.cmean[i as usize] + alpha * BFCC[i as usize];
            i += 1;
        }
        i = 0;
        while i < 4 {
            features[(4 + i) as usize] = 0.63246f32
                * (BFCC[i as usize] - self.mem[(i + 24) as usize])
                + 0.31623f32 * (self.mem[i as usize] - self.mem[(i + 16) as usize]);
            i += 1;
        }
        i = 0;
        while i < 3 {
            features[(8 + i) as usize] = 0.53452f32
                * (BFCC[i as usize] + self.mem[(i + 24) as usize])
                - 0.26726f32 * (self.mem[i as usize] + self.mem[(i + 16) as usize])
                - 0.53452f32 * self.mem[(i + 8) as usize];
            i += 1;
        }
        if self.count > 5 {
            i = 0;
            while i < 9 {
                self.std[i as usize] = (1 as f32 - alpha) * self.std[i as usize]
                    + alpha * features[i as usize] * features[i as usize];
                i += 1;
            }
        }
        i = 0;
        while i < 4 {
            features[i as usize] = BFCC[i as usize] - midE[i as usize];
            i += 1;
        }
        i = 0;
        while i < 8 {
            self.mem[(i + 24) as usize] = self.mem[(i + 16) as usize];
            self.mem[(i + 16) as usize] = self.mem[(i + 8) as usize];
            self.mem[(i + 8) as usize] = self.mem[i as usize];
            self.mem[i as usize] = BFCC[i as usize];
            i += 1;
        }
        i = 0;
        while i < 9 {
            features[(11 + i) as usize] =
                celt_sqrt(self.std[i as usize]) - STD_FEATURE_BIAS[i as usize];
            i += 1;
        }
        features[18] = spec_variability - 0.78f32;
        features[20] = (*info).tonality - 0.154723f32;
        features[21] = (*info).activity - 0.724643f32;
        features[22] = frame_stationarity - 0.743717f32;
        features[23] = (*info).tonality_slope + 0.069216f32;
        features[24] = self.lowECount - 0.067930f32;

        let frame_probs = run_analysis_mlp(&features, &mut self.rnn_state);

        // compute_dense(&layer0, &mut layer_out, &features);
        // compute_gru(&layer1, &mut self.rnn_state, &layer_out);
        // compute_dense(&layer2, &mut frame_probs, &self.rnn_state);
        (*info).activity_probability = frame_probs[1];
        (*info).music_prob = frame_probs[0];
        (*info).bandwidth = bandwidth;
        self.prev_bandwidth = bandwidth;
        (*info).noisiness = frame_noisiness;
        (*info).valid = 1;

        #[cfg(feature = "ent-dump")]
        eprintln!(
            "tonality_analysis: \
    tonality=0x{:x} tonality_slope=0x{:x} noisiness=0x{:x} activity=0x{:x} music_prob=0x{:x} \
    music_prob_min=0x{:x} music_prob_max=0x{:x} activity_probability=0x{:x} max_pitch_ratio=0x{:x}",
            (*info).tonality.to_bits(),
            (*info).tonality_slope.to_bits(),
            (*info).noisiness.to_bits(),
            (*info).activity.to_bits(),
            (*info).music_prob.to_bits(),
            (*info).music_prob_min.to_bits(),
            (*info).music_prob_max.to_bits(),
            (*info).activity_probability.to_bits(),
            (*info).max_pitch_ratio.to_bits()
        );
    }

    pub fn run_analysis<T>(
        &mut self,
        celt_mode: &OpusCustomMode,
        analysis_pcm: Option<&[T]>,
        mut analysis_frame_size: i32,
        frame_size: i32,
        c1: i32,
        c2: i32,
        C: i32,
        Fs: i32,
        lsb_depth: i32,
        downmix: DownmixFn<T>,
    ) -> AnalysisInfo {
        let mut offset: i32 = 0;
        let mut pcm_len: i32 = 0;
        analysis_frame_size -= analysis_frame_size & 1;
        if let Some(analysis_pcm) = analysis_pcm {
            analysis_frame_size = if ((100 - 5) * Fs / 50) < analysis_frame_size {
                (100 - 5) * Fs / 50
            } else {
                analysis_frame_size
            };
            pcm_len = analysis_frame_size - self.analysis_offset;
            offset = self.analysis_offset;
            while pcm_len > 0 {
                self.analysis(
                    celt_mode,
                    analysis_pcm,
                    if (Fs / 50) < pcm_len {
                        Fs / 50
                    } else {
                        pcm_len
                    },
                    offset,
                    c1,
                    c2,
                    C,
                    lsb_depth,
                    downmix,
                );
                offset += Fs / 50;
                pcm_len -= Fs / 50;
            }
            self.analysis_offset = analysis_frame_size;
            self.analysis_offset -= frame_size;
        }
        self.get_info(frame_size)
    }
}
