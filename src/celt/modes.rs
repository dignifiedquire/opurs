//! Codec mode configuration and static mode tables.
//!
//! Upstream C: `celt/modes.c`, `celt/modes.h`

/// Upstream C: celt/modes.h:OpusCustomMode
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OpusCustomMode {
    pub(crate) Fs: i32,
    pub overlap: usize,
    pub(crate) nbEBands: usize,
    pub(crate) effEBands: i32,
    pub(crate) preemph: [f32; 4],
    pub(crate) eBands: &'static [i16],
    pub(crate) maxLM: i32,
    pub(crate) nbShortMdcts: i32,
    pub shortMdctSize: i32,
    pub(crate) nbAllocVectors: i32,
    pub(crate) allocVectors: &'static [u8],
    pub(crate) logN: &'static [i16],
    pub window: &'static [f32],
    pub mdct: MdctLookup<'static>,
    pub(crate) cache: PulseCache,
    #[cfg(feature = "qext")]
    pub(crate) qext_cache: PulseCache,
}
/// Upstream C: celt/modes.h:PulseCache
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PulseCache {
    pub size: i32,
    pub index: &'static [i16],
    pub bits: &'static [u8],
    pub caps: &'static [u8],
}
pub const MAX_PERIOD: i32 = 1024;

#[cfg(feature = "qext")]
pub mod data_96000;
pub mod static_modes_float_h;

#[cfg(not(feature = "qext"))]
pub use self::static_modes_float_h::static_mode_list;
#[cfg(feature = "qext")]
pub static static_mode_list: [&OpusCustomMode; 2] = [
    &static_modes_float_h::mode48000_960_120,
    &data_96000::mode96000_1920_240,
];
use crate::celt::mdct::MdctLookup;
use crate::opus::opus_defines::{OPUS_BAD_ARG, OPUS_OK};

#[cfg(feature = "qext")]
use self::data_96000::{
    NB_QEXT_BANDS, QEXT_EBANDS_180, QEXT_EBANDS_240, QEXT_LOGN_180, QEXT_LOGN_240,
};

static eband5ms: [i16; 22] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];
static band_allocation: [u8; 231] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 80, 75, 69, 63, 56, 49, 40,
    34, 29, 20, 18, 10, 0, 0, 0, 0, 0, 0, 0, 0, 110, 100, 90, 84, 78, 71, 65, 58, 51, 45, 39, 32,
    26, 20, 12, 0, 0, 0, 0, 0, 0, 118, 110, 103, 93, 86, 80, 75, 70, 65, 59, 53, 47, 40, 31, 23,
    15, 4, 0, 0, 0, 0, 126, 119, 112, 104, 95, 89, 83, 78, 72, 66, 60, 54, 47, 39, 32, 25, 17, 12,
    1, 0, 0, 134, 127, 120, 114, 103, 97, 91, 85, 78, 72, 66, 60, 54, 47, 41, 35, 29, 23, 16, 10,
    1, 144, 137, 130, 124, 113, 107, 101, 95, 88, 82, 76, 70, 64, 57, 51, 45, 39, 33, 26, 15, 1,
    152, 145, 138, 132, 123, 117, 111, 105, 98, 92, 86, 80, 74, 67, 61, 55, 49, 43, 36, 20, 1, 162,
    155, 148, 142, 133, 127, 121, 115, 108, 102, 96, 90, 84, 77, 71, 65, 59, 53, 46, 30, 1, 172,
    165, 158, 152, 143, 137, 131, 125, 118, 112, 106, 100, 94, 87, 81, 75, 69, 63, 56, 45, 20, 200,
    200, 200, 200, 200, 200, 200, 200, 198, 193, 188, 183, 178, 173, 168, 163, 158, 153, 148, 129,
    104,
];

/// Upstream C: celt/modes.c:opus_custom_mode_create
pub fn opus_custom_mode_create(
    Fs: i32,
    frame_size: i32,
    error: Option<&mut i32>,
) -> Option<&'static OpusCustomMode> {
    // TODO: maybe return Result instead of error code?
    for mode in static_mode_list {
        for j in 0..4 {
            if Fs == mode.Fs && frame_size << j == mode.shortMdctSize * mode.nbShortMdcts {
                if let Some(error) = error {
                    *error = OPUS_OK;
                }
                return Some(mode);
            }
        }
    }
    if let Some(error) = error {
        *error = OPUS_BAD_ARG;
    }
    None
}

/// Build a temporary QEXT mode from the base mode.
///
/// Selects the appropriate eBands/logN tables based on frame size,
/// sets nbEBands/effEBands to NB_QEXT_BANDS, and copies the
/// pre-computed pulse cache from the base mode's qext_cache.
///
/// Upstream C: celt/modes.c:compute_qext_mode
#[cfg(feature = "qext")]
#[allow(dead_code)]
pub fn compute_qext_mode(m: &OpusCustomMode) -> OpusCustomMode {
    let mut qext = *m;
    if m.shortMdctSize * 48000 == 120 * m.Fs {
        qext.eBands = &QEXT_EBANDS_240;
        qext.logN = &QEXT_LOGN_240;
    } else if m.shortMdctSize * 48000 == 90 * m.Fs {
        qext.eBands = &QEXT_EBANDS_180;
        qext.logN = &QEXT_LOGN_180;
    } else {
        panic!("compute_qext_mode: unsupported shortMdctSize/Fs combination");
    }
    qext.nbEBands = NB_QEXT_BANDS;
    qext.effEBands = NB_QEXT_BANDS as i32;
    // Trim effEBands if last eBand exceeds shortMdctSize
    while qext.eBands[qext.effEBands as usize] > qext.shortMdctSize as i16 {
        qext.effEBands -= 1;
    }
    qext.nbAllocVectors = 0;
    qext.allocVectors = &[];
    qext.cache = m.qext_cache;
    qext
}
