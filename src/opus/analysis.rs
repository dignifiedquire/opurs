//! Tonality and music/speech detection analysis.
//!
//! Upstream C: `src/analysis.c`

use num_traits::Zero;

pub mod arch_h {
    pub type opus_val32 = f32;
    pub type opus_val64 = f32;
}

#[derive(Copy, Clone, Default)]
#[repr(C)]
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

/// Safe representation of interleaved PCM input for downmixing.
pub enum DownmixInput<'a> {
    Float(&'a [f32]),
    Int(&'a [i16]),
    Int24(&'a [i32]),
}

impl<'a> DownmixInput<'a> {
    /// Downmix interleaved input into a mono output buffer.
    ///
    /// - `y`: output buffer (length >= `subframe`)
    /// - `subframe`: number of output samples to produce
    /// - `offset`: sample offset into the interleaved input
    /// - `c1`: first channel index
    /// - `c2`: second channel index (-1 = none, -2 = all channels)
    /// - `C`: total number of channels
    pub fn downmix(
        &self,
        y: &mut [opus_val32],
        subframe: i32,
        offset: i32,
        c1: i32,
        c2: i32,
        C: i32,
    ) {
        match self {
            DownmixInput::Float(x) => {
                let mut j = 0;
                while j < subframe {
                    y[j as usize] = x[((j + offset) * C + c1) as usize] * CELT_SIG_SCALE;
                    j += 1;
                }
                if c2 > -1 {
                    j = 0;
                    while j < subframe {
                        y[j as usize] += x[((j + offset) * C + c2) as usize] * CELT_SIG_SCALE;
                        j += 1;
                    }
                } else if c2 == -2 {
                    let mut c = 1;
                    while c < C {
                        j = 0;
                        while j < subframe {
                            y[j as usize] += x[((j + offset) * C + c) as usize] * CELT_SIG_SCALE;
                            j += 1;
                        }
                        c += 1;
                    }
                }
                // Saturate and replace NaN to prevent downstream issues
                for sample in y.iter_mut().take(subframe as usize) {
                    *sample = sample.clamp(-65536.0, 65536.0);
                    if sample.is_nan() {
                        *sample = 0.0;
                    }
                }
            }
            DownmixInput::Int(x) => {
                let mut j = 0;
                while j < subframe {
                    y[j as usize] = x[((j + offset) * C + c1) as usize] as opus_val32;
                    j += 1;
                }
                if c2 > -1 {
                    j = 0;
                    while j < subframe {
                        y[j as usize] += x[((j + offset) * C + c2) as usize] as i32 as f32;
                        j += 1;
                    }
                } else if c2 == -2 {
                    let mut c = 1;
                    while c < C {
                        j = 0;
                        while j < subframe {
                            y[j as usize] += x[((j + offset) * C + c) as usize] as i32 as f32;
                            j += 1;
                        }
                        c += 1;
                    }
                }
            }
            DownmixInput::Int24(x) => {
                let mut j = 0;
                while j < subframe {
                    y[j as usize] = x[((j + offset) * C + c1) as usize] as opus_val32 / 256.0;
                    j += 1;
                }
                if c2 > -1 {
                    j = 0;
                    while j < subframe {
                        y[j as usize] += x[((j + offset) * C + c2) as usize] as opus_val32 / 256.0;
                        j += 1;
                    }
                } else if c2 == -2 {
                    let mut c = 1;
                    while c < C {
                        j = 0;
                        while j < subframe {
                            y[j as usize] +=
                                x[((j + offset) * C + c) as usize] as opus_val32 / 256.0;
                            j += 1;
                        }
                        c += 1;
                    }
                }
            }
        }
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct TonalityAnalysisState {
    pub arch: Arch,
    pub application: i32,
    pub Fs: i32,
    pub angle: [f32; 240],
    pub d_angle: [f32; 240],
    pub d2_angle: [f32; 240],
    pub inmem: [opus_val32; 720],
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
    pub downmix_state: [opus_val32; 3],
    pub info: [AnalysisInfo; 100],
}
impl Default for TonalityAnalysisState {
    fn default() -> Self {
        Self {
            arch: Arch::default(),
            application: 0,
            Fs: 0,
            angle: [0.0; 240],
            d_angle: [0.0; 240],
            d2_angle: [0.0; 240],
            inmem: [0.0; 720],
            mem_fill: 0,
            prev_band_tonality: [0.0; 18],
            prev_tonality: 0.0,
            prev_bandwidth: 0,
            E: [[0.0; 18]; 8],
            logE: [[0.0; 18]; 8],
            lowE: [0.0; 18],
            highE: [0.0; 18],
            meanE: [0.0; 19],
            mem: [0.0; 32],
            cmean: [0.0; 8],
            std: [0.0; 9],
            Etracker: 0.0,
            lowECount: 0.0,
            E_count: 0,
            count: 0,
            analysis_offset: 0,
            write_pos: 0,
            read_pos: 0,
            read_subframe: 0,
            hp_ener_accum: 0.0,
            initialized: 0,
            rnn_state: [0.0; 24],
            downmix_state: [0.0; 3],
            info: [AnalysisInfo::default(); 100],
        }
    }
}
pub const ANALYSIS_BUF_SIZE: i32 = 720;
pub const DETECT_SIZE: i32 = 100;
pub const NB_FRAMES: i32 = 8;
pub const NB_TBANDS: i32 = 18;
pub mod math_h {
    pub const M_PI: f64 = std::f64::consts::PI;
}
pub use self::arch_h::{opus_val32, opus_val64};
pub use self::math_h::M_PI;
use crate::arch::{opus_select_arch, Arch};
use crate::celt::float_cast::{float2int, CELT_SIG_SCALE};
use crate::celt::kiss_fft::{kiss_fft_cpx, opus_fft_c};
use crate::celt::mathops::{celt_log10, celt_sqrt, fast_atan2f};
use crate::celt::modes::OpusCustomMode;

use crate::opus::mlp::analysis_mlp::run_analysis_mlp;

#[allow(clippy::approx_constant)]
const LOG2_E_UPSTREAM: f32 = 1.442695f32;
use crate::opus::opus_encoder::is_digital_silence;

static dct_table: [f32; 128] = [
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
static analysis_window: [f32; 240] = [
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
static tbands: [i32; 19] = [
    4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160, 192, 240,
];
pub const NB_TONAL_SKIP_BANDS: i32 = 9;
fn silk_resampler_down2_hp(
    S: &mut [opus_val32; 3],
    out: &mut [opus_val32],
    in_0: &[opus_val32],
    inLen: i32,
) -> opus_val32 {
    let len2: i32 = inLen / 2;
    let mut in32: opus_val32;
    let mut out32: opus_val32;
    let mut out32_hp: opus_val32;
    let mut Y: opus_val32;
    let mut X: opus_val32;
    let mut hp_ener: opus_val64 = 0 as opus_val64;
    let mut k = 0;
    while k < len2 {
        in32 = in_0[(2 * k) as usize];
        Y = in32 - S[0];
        X = 0.6074371f32 * Y;
        out32 = S[0] + X;
        S[0] = in32 + X;
        out32_hp = out32;
        in32 = in_0[(2 * k + 1) as usize];
        Y = in32 - S[1];
        X = 0.15063f32 * Y;
        out32 += S[1];
        out32 += X;
        S[1] = in32 + X;
        Y = -in32 - S[2];
        X = 0.15063f32 * Y;
        out32_hp += S[2];
        out32_hp += X;
        S[2] = -in32 + X;
        hp_ener += out32_hp * out32_hp;
        out[k as usize] = 0.5f32 * out32;
        k += 1;
    }
    hp_ener
}
fn downmix_and_resample(
    input: &DownmixInput,
    y: &mut [opus_val32],
    S: &mut [opus_val32; 3],
    mut subframe: i32,
    mut offset: i32,
    c1: i32,
    c2: i32,
    C: i32,
    Fs: i32,
) -> opus_val32 {
    let mut j: i32;
    let mut ret: opus_val32 = 0 as opus_val32;
    if subframe == 0 {
        return 0 as opus_val32;
    }
    debug_assert!(
        matches!(Fs, 16000 | 24000 | 48000),
        "libopus: assert(Fs == 48000 || Fs == 24000 || Fs == 16000) called"
    );
    if Fs == 48000 {
        subframe *= 2;
        offset *= 2;
    } else if Fs == 16000 {
        subframe = subframe * 2 / 3;
        offset = offset * 2 / 3;
    }
    let mut tmp: Vec<opus_val32> = vec![0.0; subframe as usize];
    input.downmix(&mut tmp, subframe, offset, c1, c2, C);
    if (c2 == -2 && C == 2) || c2 > -1 {
        j = 0;
        while j < subframe {
            tmp[j as usize] *= 0.5f32;
            j += 1;
        }
    }
    if Fs == 48000 {
        ret = silk_resampler_down2_hp(S, y, &tmp, subframe);
    } else if Fs == 24000 {
        y[..subframe as usize].copy_from_slice(&tmp[..subframe as usize]);
    } else if Fs == 16000 {
        let mut tmp3x: Vec<opus_val32> = vec![0.0; (3 * subframe) as usize];
        j = 0;
        while j < subframe {
            tmp3x[(3 * j) as usize] = tmp[j as usize];
            tmp3x[(3 * j + 1) as usize] = tmp[j as usize];
            tmp3x[(3 * j + 2) as usize] = tmp[j as usize];
            j += 1;
        }
        silk_resampler_down2_hp(S, y, &tmp3x, 3 * subframe);
    }
    ret *= 1.0f32 / 32768.0 / 32768.0;
    ret
}
pub fn tonality_analysis_init(tonal: &mut TonalityAnalysisState, Fs: i32) {
    tonal.arch = opus_select_arch();
    tonal.Fs = Fs;
    tonality_analysis_reset(tonal);
}
pub fn tonality_analysis_reset(tonal: &mut TonalityAnalysisState) {
    // Zero everything from `angle` onwards, preserving arch, application, Fs
    tonal.angle = [0.0; 240];
    tonal.d_angle = [0.0; 240];
    tonal.d2_angle = [0.0; 240];
    tonal.inmem = [0.0; 720];
    tonal.mem_fill = 0;
    tonal.prev_band_tonality = [0.0; 18];
    tonal.prev_tonality = 0.0;
    tonal.prev_bandwidth = 0;
    tonal.E = [[0.0; 18]; 8];
    tonal.logE = [[0.0; 18]; 8];
    tonal.lowE = [0.0; 18];
    tonal.highE = [0.0; 18];
    tonal.meanE = [0.0; 19];
    tonal.mem = [0.0; 32];
    tonal.cmean = [0.0; 8];
    tonal.std = [0.0; 9];
    tonal.Etracker = 0.0;
    tonal.lowECount = 0.0;
    tonal.E_count = 0;
    tonal.count = 0;
    tonal.analysis_offset = 0;
    tonal.write_pos = 0;
    tonal.read_pos = 0;
    tonal.read_subframe = 0;
    tonal.hp_ener_accum = 0.0;
    tonal.initialized = 0;
    tonal.rnn_state = [0.0; 24];
    tonal.downmix_state = [0.0; 3];
    tonal.info = [AnalysisInfo {
        valid: 0,
        tonality: 0.0,
        tonality_slope: 0.0,
        noisiness: 0.0,
        activity: 0.0,
        music_prob: 0.0,
        music_prob_min: 0.0,
        music_prob_max: 0.0,
        bandwidth: 0,
        activity_probability: 0.0,
        max_pitch_ratio: 0.0,
        leak_boost: [0; 19],
    }; 100];
}
pub fn tonality_get_info(tonal: &mut TonalityAnalysisState, info_out: &mut AnalysisInfo, len: i32) {
    let mut pos: i32 = 0;
    let mut curr_lookahead: i32 = 0;
    let mut tonality_max: f32 = 0.;
    let mut tonality_avg: f32 = 0.;
    let mut tonality_count: i32 = 0;
    let mut i: i32 = 0;
    let mut pos0: i32 = 0;
    let mut prob_avg: f32 = 0.;
    let mut prob_count: f32 = 0.;
    let mut prob_min: f32 = 0.;
    let mut prob_max: f32 = 0.;
    let mut vad_prob: f32 = 0.;
    let mut mpos: i32 = 0;
    let mut vpos: i32 = 0;
    let mut bandwidth_span: i32 = 0;
    pos = tonal.read_pos;
    curr_lookahead = tonal.write_pos - tonal.read_pos;
    if curr_lookahead < 0 {
        curr_lookahead += DETECT_SIZE;
    }
    tonal.read_subframe += len / (tonal.Fs / 400);
    while tonal.read_subframe >= 8 {
        tonal.read_subframe -= 8;
        tonal.read_pos += 1;
    }
    if tonal.read_pos >= DETECT_SIZE {
        tonal.read_pos -= DETECT_SIZE;
    }
    if len > tonal.Fs / 50 && pos != tonal.write_pos {
        pos += 1;
        if pos == DETECT_SIZE {
            pos = 0;
        }
    }
    if pos == tonal.write_pos {
        pos -= 1;
    }
    if pos < 0 {
        pos = DETECT_SIZE - 1;
    }
    pos0 = pos;
    *info_out = tonal.info[pos as usize];
    if info_out.valid == 0 {
        return;
    }
    tonality_avg = info_out.tonality;
    tonality_max = tonality_avg;
    tonality_count = 1;
    bandwidth_span = 6;
    i = 0;
    while i < 3 {
        pos += 1;
        if pos == DETECT_SIZE {
            pos = 0;
        }
        if pos == tonal.write_pos {
            break;
        }
        tonality_max = if tonality_max > tonal.info[pos as usize].tonality {
            tonality_max
        } else {
            tonal.info[pos as usize].tonality
        };
        tonality_avg += tonal.info[pos as usize].tonality;
        tonality_count += 1;
        info_out.bandwidth = if info_out.bandwidth > tonal.info[pos as usize].bandwidth {
            info_out.bandwidth
        } else {
            tonal.info[pos as usize].bandwidth
        };
        bandwidth_span -= 1;
        i += 1;
    }
    pos = pos0;
    i = 0;
    while i < bandwidth_span {
        pos -= 1;
        if pos < 0 {
            pos = DETECT_SIZE - 1;
        }
        if pos == tonal.write_pos {
            break;
        }
        info_out.bandwidth = if info_out.bandwidth > tonal.info[pos as usize].bandwidth {
            info_out.bandwidth
        } else {
            tonal.info[pos as usize].bandwidth
        };
        i += 1;
    }
    info_out.tonality = if tonality_avg / tonality_count as f32 > tonality_max - 0.2f32 {
        tonality_avg / tonality_count as f32
    } else {
        tonality_max - 0.2f32
    };
    vpos = pos0;
    mpos = vpos;
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
    prob_min = 1.0f32;
    prob_max = 0.0f32;
    vad_prob = tonal.info[vpos as usize].activity_probability;
    prob_count = if 0.1f32 > vad_prob { 0.1f32 } else { vad_prob };
    prob_avg =
        (if 0.1f32 > vad_prob { 0.1f32 } else { vad_prob }) * tonal.info[mpos as usize].music_prob;
    loop {
        let mut pos_vad: f32 = 0.;
        mpos += 1;
        if mpos == DETECT_SIZE {
            mpos = 0;
        }
        if mpos == tonal.write_pos {
            break;
        }
        vpos += 1;
        if vpos == DETECT_SIZE {
            vpos = 0;
        }
        if vpos == tonal.write_pos {
            break;
        }
        pos_vad = tonal.info[vpos as usize].activity_probability;
        prob_min = if (prob_avg - 10_f32 * (vad_prob - pos_vad)) / prob_count < prob_min {
            (prob_avg - 10_f32 * (vad_prob - pos_vad)) / prob_count
        } else {
            prob_min
        };
        prob_max = if (prob_avg + 10_f32 * (vad_prob - pos_vad)) / prob_count > prob_max {
            (prob_avg + 10_f32 * (vad_prob - pos_vad)) / prob_count
        } else {
            prob_max
        };
        prob_count += if 0.1f32 > pos_vad { 0.1f32 } else { pos_vad };
        prob_avg += (if 0.1f32 > pos_vad { 0.1f32 } else { pos_vad })
            * tonal.info[mpos as usize].music_prob;
    }
    info_out.music_prob = prob_avg / prob_count;
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
    if curr_lookahead < 10 {
        let mut pmin: f32 = 0.;
        let mut pmax: f32 = 0.;
        pmin = prob_min;
        pmax = prob_max;
        pos = pos0;
        i = 0;
        while i
            < (if (tonal.count - 1) < 15 {
                tonal.count - 1
            } else {
                15
            })
        {
            pos -= 1;
            if pos < 0 {
                pos = DETECT_SIZE - 1;
            }
            pmin = if pmin < tonal.info[pos as usize].music_prob {
                pmin
            } else {
                tonal.info[pos as usize].music_prob
            };
            pmax = if pmax > tonal.info[pos as usize].music_prob {
                pmax
            } else {
                tonal.info[pos as usize].music_prob
            };
            i += 1;
        }
        pmin = if 0.0f32 > pmin - 0.1f32 * vad_prob {
            0.0f32
        } else {
            pmin - 0.1f32 * vad_prob
        };
        pmax = if 1.0f32 < pmax + 0.1f32 * vad_prob {
            1.0f32
        } else {
            pmax + 0.1f32 * vad_prob
        };
        prob_min += (1.0f32 - 0.1f32 * curr_lookahead as f32) * (pmin - prob_min);
        prob_max += (1.0f32 - 0.1f32 * curr_lookahead as f32) * (pmax - prob_max);
    }
    info_out.music_prob_min = prob_min;
    info_out.music_prob_max = prob_max;
}
static std_feature_bias: [f32; 9] = [
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
pub const LEAKAGE_OFFSET: f32 = 2.5f32;
pub const LEAKAGE_SLOPE: f32 = 2.0f32;
fn tonality_analysis(
    tonal: &mut TonalityAnalysisState,
    celt_mode: &OpusCustomMode,
    input: &DownmixInput,
    mut len: i32,
    mut offset: i32,
    c1: i32,
    c2: i32,
    C: i32,
    lsb_depth: i32,
) {
    let mut i: i32 = 0;
    let mut b: i32 = 0;
    let N: i32 = 480;
    let N2: i32 = 240;
    let mut band_tonality: [f32; 18] = [0.; 18];
    let mut logE: [f32; 18] = [0.; 18];
    let mut BFCC: [f32; 8] = [0.; 8];
    let mut features: [f32; 25] = [0.; 25];
    let mut frame_tonality: f32 = 0.;
    let mut max_frame_tonality: f32 = 0.;
    let mut frame_noisiness: f32 = 0.;
    let pi4: f32 = (M_PI * M_PI * M_PI * M_PI) as f32;
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
    let mut info_idx: usize = 0;
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
    if tonal.initialized == 0 {
        tonal.mem_fill = 240;
        tonal.initialized = 1;
    }
    alpha = 1.0f32
        / (if (10) < 1 + tonal.count {
            10
        } else {
            1 + tonal.count
        }) as f32;
    alphaE = 1.0f32
        / (if (25) < 1 + tonal.count {
            25
        } else {
            1 + tonal.count
        }) as f32;
    alphaE2 = 1.0f32
        / (if (100) < 1 + tonal.count {
            100
        } else {
            1 + tonal.count
        }) as f32;
    if tonal.count <= 1 {
        alphaE2 = 1_f32;
    }
    if tonal.Fs == 48000 {
        len /= 2;
        offset /= 2;
    } else if tonal.Fs == 16000 {
        len = 3 * len / 2;
        offset = 3 * offset / 2;
    }
    let kfft = celt_mode.mdct.kfft[0];
    {
        let fill = tonal.mem_fill as usize;
        let fs = tonal.Fs;
        let sub = if len < 720 - tonal.mem_fill {
            len
        } else {
            720 - tonal.mem_fill
        };
        let ret = downmix_and_resample(
            input,
            &mut tonal.inmem[fill..],
            &mut tonal.downmix_state,
            sub,
            offset,
            c1,
            c2,
            C,
            fs,
        );
        tonal.hp_ener_accum += ret;
    }
    if tonal.mem_fill + len < ANALYSIS_BUF_SIZE {
        tonal.mem_fill += len;
        return;
    }
    hp_ener = tonal.hp_ener_accum;
    info_idx = tonal.write_pos as usize;
    tonal.write_pos += 1;
    if tonal.write_pos >= DETECT_SIZE {
        tonal.write_pos -= DETECT_SIZE;
    }
    is_silence = is_digital_silence(&tonal.inmem, 720, 1, lsb_depth);
    let mut in_0: [kiss_fft_cpx; 480] = [kiss_fft_cpx::zero(); 480];
    let mut out: [kiss_fft_cpx; 480] = [kiss_fft_cpx::zero(); 480];
    let mut tonality: [f32; 240] = [0.; 240];
    let mut noisiness: [f32; 240] = [0.; 240];
    i = 0;
    while i < N2 {
        let w: f32 = analysis_window[i as usize];
        in_0[i as usize].re = w * tonal.inmem[i as usize];
        in_0[i as usize].im = w * tonal.inmem[(N2 + i) as usize];
        in_0[(N - i - 1) as usize].re = w * tonal.inmem[(N - i - 1) as usize];
        in_0[(N - i - 1) as usize].im = w * tonal.inmem[(N + N2 - i - 1) as usize];
        i += 1;
    }
    // memmove: copy last 240 samples (inmem[480..720]) to start (inmem[0..240])
    tonal.inmem.copy_within(480..720, 0);
    remaining = len - (ANALYSIS_BUF_SIZE - tonal.mem_fill);
    {
        let fs = tonal.Fs;
        let off = offset + ANALYSIS_BUF_SIZE - tonal.mem_fill;
        tonal.hp_ener_accum = downmix_and_resample(
            input,
            &mut tonal.inmem[240..],
            &mut tonal.downmix_state,
            remaining,
            off,
            c1,
            c2,
            C,
            fs,
        );
    }
    tonal.mem_fill = 240 + remaining;
    if is_silence != 0 {
        let mut prev_pos: i32 = tonal.write_pos - 2;
        if prev_pos < 0 {
            prev_pos += DETECT_SIZE;
        }
        tonal.info[info_idx] = tonal.info[prev_pos as usize];
        return;
    }
    opus_fft_c(kfft, &in_0, &mut out);
    if out[0].re.is_nan() {
        tonal.info[info_idx].valid = 0;
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
        angle = (0.5f32 as f64 / M_PI) as f32 * fast_atan2f(X1i, X1r);
        d_angle = angle - tonal.angle[i as usize];
        d2_angle = d_angle - tonal.d_angle[i as usize];
        angle2 = (0.5f32 as f64 / M_PI) as f32 * fast_atan2f(X2i, X2r);
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
        avg_mod = 0.25f32 * (tonal.d2_angle[i as usize] + mod1 + 2_f32 * mod2);
        tonality[i as usize] = 1.0f32 / (1.0f32 + 40.0f32 * 16.0f32 * pi4 * avg_mod) - 0.015f32;
        tonality2[i as usize] = 1.0f32 / (1.0f32 + 40.0f32 * 16.0f32 * pi4 * mod2) - 0.015f32;
        tonal.angle[i as usize] = angle2;
        tonal.d_angle[i as usize] = d_angle2;
        tonal.d2_angle[i as usize] = mod2;
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
    tonal.info[info_idx].activity = 0 as f32;
    frame_noisiness = 0 as f32;
    frame_stationarity = 0 as f32;
    if tonal.count == 0 {
        b = 0;
        while b < NB_TBANDS {
            tonal.lowE[b as usize] = 1e10f64 as f32;
            tonal.highE[b as usize] = -1e10f64 as f32;
            b += 1;
        }
    }
    relativeE = 0 as f32;
    frame_loudness = 0 as f32;
    let mut E: f32 = 0 as f32;
    let mut X1r_0: f32 = 0.;
    let mut X2r_0: f32 = 0.;
    X1r_0 = 2_f32 * out[0_usize].re;
    X2r_0 = 2_f32 * out[0_usize].im;
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
    E *= 1.0f32 / 32768.0 / 32768.0;
    let mut loge0: f32 = ((E + 1e-10f32) as f64).ln() as f32;
    band_log2[0_usize] = 0.5f32 * LOG2_E_UPSTREAM * loge0;
    b = 0;
    while b < NB_TBANDS {
        let mut E_0: f32 = 0 as f32;
        let mut tE: f32 = 0 as f32;
        let mut nE: f32 = 0 as f32;
        let mut L1: f32 = 0.;
        let mut L2: f32 = 0.;
        let mut stationarity: f32 = 0.;
        i = tbands[b as usize];
        while i < tbands[(b + 1) as usize] {
            let binE_0: f32 = out[i as usize].re * out[i as usize].re
                + out[(N - i) as usize].re * out[(N - i) as usize].re
                + out[i as usize].im * out[i as usize].im
                + out[(N - i) as usize].im * out[(N - i) as usize].im;
            let binE_0 = binE_0 * (1.0f32 / 32768.0 / 32768.0);
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
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(E_0 < 1e9f32) || E_0.is_nan() {
            tonal.info[info_idx].valid = 0;
            return;
        }
        tonal.E[tonal.E_count as usize][b as usize] = E_0;
        frame_noisiness += nE / (1e-15f32 + E_0);
        frame_loudness += celt_sqrt(E_0 + 1e-10f32);
        loge0 = ((E_0 + 1e-10f32) as f64).ln() as f32;
        logE[b as usize] = loge0;
        band_log2[(b + 1) as usize] = 0.5f32 * LOG2_E_UPSTREAM * loge0;
        tonal.logE[tonal.E_count as usize][b as usize] = logE[b as usize];
        if tonal.count == 0 {
            tonal.lowE[b as usize] = logE[b as usize];
            tonal.highE[b as usize] = tonal.lowE[b as usize];
        }
        if tonal.highE[b as usize] as f64 > tonal.lowE[b as usize] as f64 + 7.5f64 {
            if tonal.highE[b as usize] - logE[b as usize]
                > logE[b as usize] - tonal.lowE[b as usize]
            {
                tonal.highE[b as usize] -= 0.01f32;
            } else {
                tonal.lowE[b as usize] += 0.01f32;
            }
        }
        if logE[b as usize] > tonal.highE[b as usize] {
            tonal.highE[b as usize] = logE[b as usize];
            tonal.lowE[b as usize] = if tonal.highE[b as usize] - 15_f32 > tonal.lowE[b as usize] {
                tonal.highE[b as usize] - 15_f32
            } else {
                tonal.lowE[b as usize]
            };
        } else if logE[b as usize] < tonal.lowE[b as usize] {
            tonal.lowE[b as usize] = logE[b as usize];
            tonal.highE[b as usize] = if (tonal.lowE[b as usize] + 15_f32) < tonal.highE[b as usize]
            {
                tonal.lowE[b as usize] + 15_f32
            } else {
                tonal.highE[b as usize]
            };
        }
        relativeE += (logE[b as usize] - tonal.lowE[b as usize])
            / (1e-5f32 + (tonal.highE[b as usize] - tonal.lowE[b as usize]));
        L2 = 0 as f32;
        L1 = L2;
        i = 0;
        while i < NB_FRAMES {
            L1 += celt_sqrt(tonal.E[i as usize][b as usize]);
            L2 += tonal.E[i as usize][b as usize];
            i += 1;
        }
        // NB:
        //  because `1e-15` is specified without a suffix in the upstream,
        //   this addition is performed as f64 and `celt_sqrt` can't be used
        stationarity = 0.99f32.min(L1 / (1e-15 + ((NB_FRAMES as f32 * L2) as f64)).sqrt() as f32);
        stationarity *= stationarity;
        stationarity *= stationarity;
        frame_stationarity += stationarity;
        band_tonality[b as usize] =
            if tE / (1e-15f32 + E_0) > stationarity * tonal.prev_band_tonality[b as usize] {
                tE / (1e-15f32 + E_0)
            } else {
                stationarity * tonal.prev_band_tonality[b as usize]
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
        tonal.prev_band_tonality[b as usize] = band_tonality[b as usize];
        b += 1;
    }
    leakage_from[0_usize] = band_log2[0_usize];
    leakage_to[0_usize] = band_log2[0_usize] - LEAKAGE_OFFSET;
    b = 1;
    while b < NB_TBANDS + 1 {
        let leak_slope: f32 =
            LEAKAGE_SLOPE * (tbands[b as usize] - tbands[(b - 1) as usize]) as f32 / 4_f32;
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
            LEAKAGE_SLOPE * (tbands[(b + 1) as usize] - tbands[b as usize]) as f32 / 4_f32;
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
        tonal.info[info_idx].leak_boost[b as usize] =
            (if (255) < (0.5f64 + (64.0f32 * boost) as f64).floor() as i32 {
                255
            } else {
                (0.5f64 + (64.0f32 * boost) as f64).floor() as i32
            }) as u8;
        b += 1;
    }
    while b < LEAK_BANDS {
        tonal.info[info_idx].leak_boost[b as usize] = 0;
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
                tmp = tonal.logE[i as usize][k as usize] - tonal.logE[j as usize][k as usize];
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
    noise_floor = 5.7e-4f32 / ((1) << (if 0 > lsb_depth - 8 { 0 } else { lsb_depth - 8 })) as f32;
    noise_floor *= noise_floor;
    below_max_pitch = 0 as f32;
    above_max_pitch = 0 as f32;
    b = 0;
    while b < NB_TBANDS {
        let mut E_1: f32 = 0 as f32;
        let mut Em: f32 = 0.;
        let mut band_start: i32 = 0;
        let mut band_end: i32 = 0;
        band_start = tbands[b as usize];
        band_end = tbands[(b + 1) as usize];
        i = band_start;
        while i < band_end {
            let binE_1: f32 = out[i as usize].re * out[i as usize].re
                + out[(N - i) as usize].re * out[(N - i) as usize].re
                + out[i as usize].im * out[i as usize].im
                + out[(N - i) as usize].im * out[(N - i) as usize].im;
            E_1 += binE_1;
            i += 1;
        }
        E_1 *= 1.0f32 / 32768.0 / 32768.0;
        maxE = if maxE > E_1 { maxE } else { E_1 };
        if band_start < 64 {
            below_max_pitch += E_1;
        } else {
            above_max_pitch += E_1;
        }
        tonal.meanE[b as usize] = if (1_f32 - alphaE2) * tonal.meanE[b as usize] > E_1 {
            (1_f32 - alphaE2) * tonal.meanE[b as usize]
        } else {
            E_1
        };
        Em = if E_1 > tonal.meanE[b as usize] {
            E_1
        } else {
            tonal.meanE[b as usize]
        };
        if E_1 * 1e9f32 > maxE
            && (Em > 3_f32 * noise_floor * (band_end - band_start) as f32
                || E_1 > noise_floor * (band_end - band_start) as f32)
        {
            bandwidth = b + 1;
        }
        is_masked[b as usize] = (E_1
            < (if tonal.prev_bandwidth > b {
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
    if tonal.Fs == 48000 {
        let mut noise_ratio: f32 = 0.;
        let mut Em_0: f32 = 0.;
        let E_2: f32 = hp_ener * (1.0f32 / (60 * 60) as f32);
        noise_ratio = if tonal.prev_bandwidth == 20 {
            10.0f32
        } else {
            30.0f32
        };
        above_max_pitch += E_2;
        tonal.meanE[b as usize] = if (1_f32 - alphaE2) * tonal.meanE[b as usize] > E_2 {
            (1_f32 - alphaE2) * tonal.meanE[b as usize]
        } else {
            E_2
        };
        Em_0 = if E_2 > tonal.meanE[b as usize] {
            E_2
        } else {
            tonal.meanE[b as usize]
        };
        if Em_0 > 3_f32 * noise_ratio * noise_floor * 160_f32
            || E_2 > noise_ratio * noise_floor * 160_f32
        {
            bandwidth = 20;
        }
        is_masked[b as usize] = (E_2
            < (if tonal.prev_bandwidth == 20 {
                0.01f32
            } else {
                0.05f32
            }) * bandwidth_mask) as i32;
    }
    if above_max_pitch > below_max_pitch {
        tonal.info[info_idx].max_pitch_ratio = below_max_pitch / above_max_pitch;
    } else {
        tonal.info[info_idx].max_pitch_ratio = 1_f32;
    }
    if bandwidth == 20 && is_masked[NB_TBANDS as usize] != 0 {
        bandwidth -= 2;
    } else if bandwidth > 0 && bandwidth <= NB_TBANDS && is_masked[(bandwidth - 1) as usize] != 0 {
        bandwidth -= 1;
    }
    if tonal.count <= 2 {
        bandwidth = 20;
    }
    frame_loudness = 20f32 * celt_log10(frame_loudness);
    tonal.Etracker = if tonal.Etracker - 0.003f32 > frame_loudness {
        tonal.Etracker - 0.003f32
    } else {
        frame_loudness
    };
    tonal.lowECount *= 1_f32 - alphaE;
    if frame_loudness < tonal.Etracker - 30_f32 {
        tonal.lowECount += alphaE;
    }
    i = 0;
    while i < 8 {
        let mut sum: f32 = 0 as f32;
        b = 0;
        while b < 16 {
            sum += dct_table[(i * 16 + b) as usize] * logE[b as usize];
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
            sum_0 += dct_table[(i * 16 + b) as usize]
                * 0.5f32
                * (tonal.highE[b as usize] + tonal.lowE[b as usize]);
            b += 1;
        }
        midE[i as usize] = sum_0;
        i += 1;
    }
    frame_stationarity /= NB_TBANDS as f32;
    relativeE /= NB_TBANDS as f32;
    if tonal.count < 10 {
        relativeE = 0.5f32;
    }
    frame_noisiness /= NB_TBANDS as f32;
    tonal.info[info_idx].activity = frame_noisiness + (1_f32 - frame_noisiness) * relativeE;
    frame_tonality = max_frame_tonality / (NB_TBANDS - NB_TONAL_SKIP_BANDS) as f32;
    frame_tonality = if frame_tonality > tonal.prev_tonality * 0.8f32 {
        frame_tonality
    } else {
        tonal.prev_tonality * 0.8f32
    };
    tonal.prev_tonality = frame_tonality;
    slope /= (8 * 8) as f32;
    tonal.info[info_idx].tonality_slope = slope;
    tonal.E_count = (tonal.E_count + 1) % NB_FRAMES;
    tonal.count = if (tonal.count + 1) < 10000 {
        tonal.count + 1
    } else {
        10000
    };
    tonal.info[info_idx].tonality = frame_tonality;
    i = 0;
    while i < 4 {
        features[i as usize] = -0.12299f32 * (BFCC[i as usize] + tonal.mem[(i + 24) as usize])
            + 0.49195f32 * (tonal.mem[i as usize] + tonal.mem[(i + 16) as usize])
            + 0.69693f32 * tonal.mem[(i + 8) as usize]
            - 1.4349f32 * tonal.cmean[i as usize];
        i += 1;
    }
    i = 0;
    while i < 4 {
        tonal.cmean[i as usize] =
            (1_f32 - alpha) * tonal.cmean[i as usize] + alpha * BFCC[i as usize];
        i += 1;
    }
    i = 0;
    while i < 4 {
        features[(4 + i) as usize] = 0.63246f32 * (BFCC[i as usize] - tonal.mem[(i + 24) as usize])
            + 0.31623f32 * (tonal.mem[i as usize] - tonal.mem[(i + 16) as usize]);
        i += 1;
    }
    i = 0;
    while i < 3 {
        features[(8 + i) as usize] = 0.53452f32 * (BFCC[i as usize] + tonal.mem[(i + 24) as usize])
            - 0.26726f32 * (tonal.mem[i as usize] + tonal.mem[(i + 16) as usize])
            - 0.53452f32 * tonal.mem[(i + 8) as usize];
        i += 1;
    }
    if tonal.count > 5 {
        i = 0;
        while i < 9 {
            tonal.std[i as usize] = (1_f32 - alpha) * tonal.std[i as usize]
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
        tonal.mem[(i + 24) as usize] = tonal.mem[(i + 16) as usize];
        tonal.mem[(i + 16) as usize] = tonal.mem[(i + 8) as usize];
        tonal.mem[(i + 8) as usize] = tonal.mem[i as usize];
        tonal.mem[i as usize] = BFCC[i as usize];
        i += 1;
    }
    i = 0;
    while i < 9 {
        features[(11 + i) as usize] =
            celt_sqrt(tonal.std[i as usize]) - std_feature_bias[i as usize];
        i += 1;
    }
    features[18] = spec_variability - 0.78f32;
    features[20] = tonal.info[info_idx].tonality - 0.154723f32;
    features[21] = tonal.info[info_idx].activity - 0.724643f32;
    features[22] = frame_stationarity - 0.743717f32;
    features[23] = tonal.info[info_idx].tonality_slope + 0.069216f32;
    features[24] = tonal.lowECount - 0.067930f32;

    let frame_probs = run_analysis_mlp(&features, &mut tonal.rnn_state);

    // compute_dense(&layer0, &mut layer_out, &features);
    // compute_gru(&layer1, &mut tonal.rnn_state, &layer_out);
    // compute_dense(&layer2, &mut frame_probs, &tonal.rnn_state);
    tonal.info[info_idx].activity_probability = frame_probs[1];
    tonal.info[info_idx].music_prob = frame_probs[0];
    tonal.info[info_idx].bandwidth = bandwidth;
    tonal.prev_bandwidth = bandwidth;
    tonal.info[info_idx].noisiness = frame_noisiness;
    tonal.info[info_idx].valid = 1;

    #[cfg(feature = "ent-dump")]
    eprintln!(
        "tonality_analysis: \
    tonality=0x{:x} tonality_slope=0x{:x} noisiness=0x{:x} activity=0x{:x} music_prob=0x{:x} \
    music_prob_min=0x{:x} music_prob_max=0x{:x} activity_probability=0x{:x} max_pitch_ratio=0x{:x}",
        tonal.info[info_idx].tonality.to_bits(),
        tonal.info[info_idx].tonality_slope.to_bits(),
        tonal.info[info_idx].noisiness.to_bits(),
        tonal.info[info_idx].activity.to_bits(),
        tonal.info[info_idx].music_prob.to_bits(),
        tonal.info[info_idx].music_prob_min.to_bits(),
        tonal.info[info_idx].music_prob_max.to_bits(),
        tonal.info[info_idx].activity_probability.to_bits(),
        tonal.info[info_idx].max_pitch_ratio.to_bits()
    );
}
pub fn run_analysis(
    analysis: &mut TonalityAnalysisState,
    celt_mode: &OpusCustomMode,
    input: Option<&DownmixInput>,
    mut analysis_frame_size: i32,
    frame_size: i32,
    c1: i32,
    c2: i32,
    C: i32,
    Fs: i32,
    lsb_depth: i32,
    analysis_info: &mut AnalysisInfo,
) {
    let mut offset: i32 = 0;
    let mut pcm_len: i32 = 0;
    analysis_frame_size -= analysis_frame_size & 1;
    if let Some(input) = input {
        analysis_frame_size = if ((100 - 5) * Fs / 50) < analysis_frame_size {
            (100 - 5) * Fs / 50
        } else {
            analysis_frame_size
        };
        pcm_len = analysis_frame_size - analysis.analysis_offset;
        offset = analysis.analysis_offset;
        while pcm_len > 0 {
            tonality_analysis(
                analysis,
                celt_mode,
                input,
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
            );
            offset += Fs / 50;
            pcm_len -= Fs / 50;
        }
        analysis.analysis_offset = analysis_frame_size;
        analysis.analysis_offset -= frame_size;
    }
    tonality_get_info(analysis, analysis_info, frame_size);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "tools")]
    use core::ffi::c_void;

    #[cfg(feature = "tools")]
    unsafe extern "C" {
        fn downmix_float(
            x: *const c_void,
            y: *mut opus_val32,
            subframe: i32,
            offset: i32,
            c1: i32,
            c2: i32,
            C: i32,
        );
    }

    #[test]
    fn downmix_and_resample_24k_passthrough_matches_downmix_scale() {
        let input_pcm = vec![0.25f32; 16];
        let input = DownmixInput::Float(&input_pcm);
        let mut y = vec![0.0f32; 8];
        let mut s = [0.0f32; 3];

        let hp = downmix_and_resample(&input, &mut y, &mut s, 8, 0, 0, -1, 1, 24000);
        assert_eq!(hp, 0.0);
        for &v in &y {
            assert!((v - 8192.0).abs() < 1e-3, "unexpected sample {v}");
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn downmix_and_resample_unsupported_fs_triggers_debug_assert() {
        let input_pcm = vec![0.25f32; 8];
        let input = DownmixInput::Float(&input_pcm);
        let mut y = vec![0.0f32; 4];
        let mut s = [0.0f32; 3];
        let _ = downmix_and_resample(&input, &mut y, &mut s, 4, 0, 0, -1, 1, 44100);
    }

    #[cfg(feature = "tools")]
    #[test]
    fn downmix_float_matches_upstream_c_nan_and_clamp_behavior() {
        let c = 3i32;
        let subframe = 8i32;
        let offset = 2i32;
        let mut pcm = vec![0.0f32; ((offset + subframe) * c) as usize];
        for (i, sample) in pcm.iter_mut().enumerate() {
            *sample = match i % 7 {
                0 => 0.25,
                1 => 3.0,
                2 => -3.5,
                3 => f32::INFINITY,
                4 => f32::NEG_INFINITY,
                5 => f32::NAN,
                _ => 1e20,
            };
        }

        for &(c1, c2) in &[(0, -1), (0, 1), (0, -2)] {
            let input = DownmixInput::Float(&pcm);
            let mut rust_out = vec![0.0f32; subframe as usize];
            input.downmix(&mut rust_out, subframe, offset, c1, c2, c);

            let mut c_out = vec![0.0f32; subframe as usize];
            unsafe {
                downmix_float(
                    pcm.as_ptr().cast::<c_void>(),
                    c_out.as_mut_ptr(),
                    subframe,
                    offset,
                    c1,
                    c2,
                    c,
                );
            }

            for i in 0..subframe as usize {
                assert_eq!(
                    rust_out[i].to_bits(),
                    c_out[i].to_bits(),
                    "sample mismatch c1={c1} c2={c2} i={i}"
                );
            }
        }
    }
}
