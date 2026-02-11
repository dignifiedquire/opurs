use crate::celt::modes::OpusCustomMode;

pub mod arch_h {
    pub type opus_val16 = f32;
    pub type opus_val32 = f32;
}

pub use self::arch_h::opus_val16;

pub static trim_icdf: [u8; 11] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];
pub static spread_icdf: [u8; 4] = [25, 23, 2, 0];
pub static tapset_icdf: [u8; 3] = [2, 1, 0];
pub static tf_select_table: [[i8; 8]; 4] = [
    [0, -1, 0, -1, 0, -1, 0, -1],
    [0, -1, 0, -2, 1, 0, 1, -1],
    [0, -2, 0, -3, 2, 0, 1, -1],
    [0, -2, 0, -3, 3, 0, 1, -1],
];

pub const COMBFILTER_MAXPERIOD: i32 = 1024;
pub const COMBFILTER_MINPERIOD: i32 = 15;

const GAINS: [[opus_val16; 3]; 3] = [
    [0.3066406250f32, 0.2170410156f32, 0.1296386719f32],
    [0.4638671875f32, 0.2680664062f32, 0.0f32],
    [0.7998046875f32, 0.1000976562f32, 0.0f32],
];

pub fn resampling_factor(rate: i32) -> i32 {
    match rate {
        48000 => 1,
        24000 => 2,
        16000 => 3,
        12000 => 4,
        8000 => 6,
        _ => panic!("Unsupported sampling rate: {}", rate),
    }
}

/// Upstream C: celt/celt.c:comb_filter_const_c
///
/// Constant-coefficient comb filter inner loop.
/// `x` must contain at least `T+2` samples before `x_start` for lookback.
fn comb_filter_const_c(
    y: &mut [f32],
    y_start: usize,
    x: &[f32],
    x_start: usize,
    T: i32,
    N: i32,
    g10: opus_val16,
    g11: opus_val16,
    g12: opus_val16,
) {
    let t = T as usize;
    let mut x4 = x[x_start - t - 2];
    let mut x3 = x[x_start - t - 1];
    let mut x2 = x[x_start - t];
    let mut x1 = x[x_start - t + 1];
    for i in 0..N as usize {
        let x0 = x[x_start + i - t + 2];
        y[y_start + i] = x[x_start + i] + g10 * x2 + g11 * (x1 + x3) + g12 * (x0 + x4);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }
}

/// Upstream C: celt/celt.c:comb_filter
///
/// Comb filter with separate input/output buffers. The input buffer `x`
/// must have lookback data: `x[x_start - T - 2]` through `x[x_start + N - 1]`
/// must be valid. Output is written to `y[y_start..y_start + N]`.
/// `window` may be empty if `overlap` is 0.
pub fn comb_filter(
    y: &mut [f32],
    y_start: usize,
    x: &[f32],
    x_start: usize,
    mut T0: i32,
    mut T1: i32,
    N: i32,
    g0: opus_val16,
    g1: opus_val16,
    tapset0: i32,
    tapset1: i32,
    window: &[f32],
    mut overlap: i32,
    _arch: i32,
) {
    if g0 == 0.0f32 && g1 == 0.0f32 {
        y[y_start..y_start + N as usize].copy_from_slice(&x[x_start..x_start + N as usize]);
        return;
    }
    T0 = if T0 > 15 { T0 } else { 15 };
    T1 = if T1 > 15 { T1 } else { 15 };
    let g00 = g0 * GAINS[tapset0 as usize][0];
    let g01 = g0 * GAINS[tapset0 as usize][1];
    let g02 = g0 * GAINS[tapset0 as usize][2];
    let g10 = g1 * GAINS[tapset1 as usize][0];
    let g11 = g1 * GAINS[tapset1 as usize][1];
    let g12 = g1 * GAINS[tapset1 as usize][2];
    let mut x1 = x[x_start - T1 as usize + 1];
    let mut x2 = x[x_start - T1 as usize];
    let mut x3 = x[x_start - T1 as usize - 1];
    let mut x4 = x[x_start - T1 as usize - 2];
    if g0 == g1 && T0 == T1 && tapset0 == tapset1 {
        overlap = 0;
    }
    let mut i = 0;
    while i < overlap {
        let iu = i as usize;
        let x0 = x[x_start + iu - T1 as usize + 2];
        let f = window[iu] * window[iu];
        y[y_start + iu] = x[x_start + iu]
            + (1.0f32 - f) * g00 * x[x_start + iu - T0 as usize]
            + (1.0f32 - f)
                * g01
                * (x[x_start + iu - T0 as usize + 1] + x[x_start + iu - T0 as usize - 1])
            + (1.0f32 - f)
                * g02
                * (x[x_start + iu - T0 as usize + 2] + x[x_start + iu - T0 as usize - 2])
            + f * g10 * x2
            + f * g11 * (x1 + x3)
            + f * g12 * (x0 + x4);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
        i += 1;
    }
    if g1 == 0.0f32 {
        let ov = overlap as usize;
        y[y_start + ov..y_start + N as usize]
            .copy_from_slice(&x[x_start + ov..x_start + N as usize]);
        return;
    }
    comb_filter_const_c(
        y,
        y_start + i as usize,
        x,
        x_start + i as usize,
        T1,
        N - i,
        g10,
        g11,
        g12,
    );
}

/// Upstream C: celt/celt.c:comb_filter (in-place variant)
///
/// In-place comb filter where input and output are in the same buffer.
/// The buffer must have lookback data before `start`. Reads from
/// `buf[start - T - 2..]` and writes to `buf[start..start + N]`.
pub fn comb_filter_inplace(
    buf: &mut [f32],
    start: usize,
    mut T0: i32,
    mut T1: i32,
    N: i32,
    g0: opus_val16,
    g1: opus_val16,
    tapset0: i32,
    tapset1: i32,
    window: &[f32],
    mut overlap: i32,
    _arch: i32,
) {
    if g0 == 0.0f32 && g1 == 0.0f32 {
        // In-place with no filtering: nothing to do
        return;
    }
    T0 = if T0 > 15 { T0 } else { 15 };
    T1 = if T1 > 15 { T1 } else { 15 };
    let g00 = g0 * GAINS[tapset0 as usize][0];
    let g01 = g0 * GAINS[tapset0 as usize][1];
    let g02 = g0 * GAINS[tapset0 as usize][2];
    let g10 = g1 * GAINS[tapset1 as usize][0];
    let g11 = g1 * GAINS[tapset1 as usize][1];
    let g12 = g1 * GAINS[tapset1 as usize][2];
    let mut x1 = buf[start - T1 as usize + 1];
    let mut x2 = buf[start - T1 as usize];
    let mut x3 = buf[start - T1 as usize - 1];
    let mut x4 = buf[start - T1 as usize - 2];
    if g0 == g1 && T0 == T1 && tapset0 == tapset1 {
        overlap = 0;
    }
    let mut i: usize = 0;
    while i < overlap as usize {
        let x0 = buf[start + i - T1 as usize + 2];
        let f = window[i] * window[i];
        // Since T >= 15, reads at [start + i - T] are always before writes at [start + i]
        buf[start + i] = buf[start + i]
            + (1.0f32 - f) * g00 * buf[start + i - T0 as usize]
            + (1.0f32 - f)
                * g01
                * (buf[start + i - T0 as usize + 1] + buf[start + i - T0 as usize - 1])
            + (1.0f32 - f)
                * g02
                * (buf[start + i - T0 as usize + 2] + buf[start + i - T0 as usize - 2])
            + f * g10 * x2
            + f * g11 * (x1 + x3)
            + f * g12 * (x0 + x4);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
        i += 1;
    }
    if g1 == 0.0f32 {
        // In-place with no g1: nothing left to do after overlap
        return;
    }
    // Constant-coefficient section: T >= 15 guarantees reads are always behind writes
    let t = T1 as usize;
    let pos = start + i;
    let remain = (N as usize) - i;
    let mut xv4 = buf[pos - t - 2];
    let mut xv3 = buf[pos - t - 1];
    let mut xv2 = buf[pos - t];
    let mut xv1 = buf[pos - t + 1];
    for j in 0..remain {
        let xv0 = buf[pos + j - t + 2];
        buf[pos + j] = buf[pos + j] + g10 * xv2 + g11 * (xv1 + xv3) + g12 * (xv0 + xv4);
        xv4 = xv3;
        xv3 = xv2;
        xv2 = xv1;
        xv1 = xv0;
    }
}

/// Upstream C: celt/celt.c:init_caps
pub fn init_caps(m: &OpusCustomMode, cap: &mut [i32], LM: i32, C: i32) {
    for i in 0..m.nbEBands as usize {
        let N = (m.eBands[i + 1] as i32 - m.eBands[i] as i32) << LM;
        cap[i] = (m.cache.caps[m.nbEBands as usize * (2 * LM as usize + C as usize - 1) + i]
            as i32
            + 64)
            * C
            * N
            >> 2;
    }
}

pub fn opus_strerror(error: i32) -> &'static str {
    static error_strings: [&str; 8] = [
        "success (0)",
        "invalid argument (-1)",
        "buffer too small (-2)",
        "internal error (-3)",
        "corrupted stream (-4)",
        "request not implemented (-5)",
        "invalid state (-6)",
        "memory allocation failed (-7)",
    ];
    if error > 0 || error < -7 {
        "unknown error"
    } else {
        error_strings[-error as usize]
    }
}

pub fn opus_get_version_string() -> &'static str {
    "unsafe-libopus (rust port) 1.3.1"
}
