//! aarch64 NEON SIMD implementations for SILK functions.
//!
//! NEON is always available on aarch64, so these are selected at compile time.

use core::arch::aarch64::*;

use crate::silk::define::MAX_PREDICTION_POWER_GAIN;
use crate::silk::define::{
    DECISION_DELAY, HARM_SHAPE_FIR_TAPS, LTP_ORDER, MAX_LPC_ORDER, MAX_SHAPE_LPC_ORDER,
    MAX_SUB_FRAME_LENGTH, NSQ_LPC_BUF_LENGTH, QUANT_LEVEL_ADJUST_Q10, TYPE_VOICED,
};
use crate::silk::macros::silk_CLZ32;
use crate::silk::macros::{silk_SMLAWB, silk_SMULWB, silk_SMULWW};
use crate::silk::structs::{silk_nsq_state, NsqConfig, SideInfoIndices};
use crate::silk::tables_other::silk_Quantization_Offsets_Q10;
use crate::silk::Inlines::{silk_DIV32_varQ, silk_INVERSE32_varQ};
use crate::silk::LPC_analysis_filter::silk_LPC_analysis_filter;
use crate::silk::NSQ_del_dec::silk_int32_MAX;
use crate::silk::SigProc_FIX::silk_RSHIFT_ROUND64;
use crate::silk::SigProc_FIX::{
    silk_SMMUL, silk_min_int, RAND_INCREMENT, RAND_MULTIPLIER, SILK_FIX_CONST, SILK_MAX_ORDER_LPC,
};

/// NEON implementation of `silk_noise_shape_quantizer_short_prediction`.
/// Port of `silk/arm/NSQ_neon.h`.
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn silk_noise_shape_quantizer_short_prediction_neon(
    buf32: &[i32],
    coef16: &[i16],
    order: i32,
) -> i32 {
    let b = buf32.len();
    debug_assert!(b >= order as usize);
    debug_assert!(coef16.len() >= order as usize);
    debug_assert!(order == 10 || order == 16);

    let mut out: i32 = order >> 1;

    // Process elements in groups of 4 using NEON
    // buf32 is indexed backwards from end, coef16 forwards
    let mut acc = vdupq_n_s64(0);

    // Process first 8 elements (order 10 has 10, order 16 has 16)
    let iterations = if order == 16 { 4 } else { 2 };

    for k in 0..iterations {
        let base = b - (k * 4 + 4);
        let coef_base = k * 4;

        let buf = vld1q_s32(buf32.as_ptr().add(base));

        // Sign-extend 4 x i16 to 4 x i32, then reverse order so that
        // buf[base+i] pairs with coef16[coef_base + 3 - i] (matching the
        // scalar code where buf32[b-1-j] pairs with coef16[j]).
        let c16 = vld1_s16(coef16.as_ptr().add(coef_base));
        let coef_fwd = vmovl_s16(c16);
        // Reverse 4 x i32: vrev64q reverses within 64-bit halves, then swap halves
        let coef = vextq_s32(vrev64q_s32(coef_fwd), vrev64q_s32(coef_fwd), 2);

        // We need (buf[i] * coef[i]) >> 16 for each element
        // Use widening multiply: i32 * i32 -> i64, then >> 16
        let prod_lo = vmull_s32(vget_low_s32(buf), vget_low_s32(coef));
        let prod_hi = vmull_s32(vget_high_s32(buf), vget_high_s32(coef));

        // Shift right by 16 and accumulate
        acc = vaddq_s64(acc, vshrq_n_s64(prod_lo, 16));
        acc = vaddq_s64(acc, vshrq_n_s64(prod_hi, 16));
    }

    // Horizontal sum of i64 accumulator
    out += (vgetq_lane_s64(acc, 0) + vgetq_lane_s64(acc, 1)) as i32;

    // For order 10, handle remaining 2 elements scalar
    if order == 10 {
        out = (out as i64 + ((buf32[b - 9] as i64 * coef16[8] as i64) >> 16)) as i32;
        out = (out as i64 + ((buf32[b - 10] as i64 * coef16[9] as i64) >> 16)) as i32;
    }

    out
}

/// NEON implementation of `silk_NSQ_noise_shape_feedback_loop`.
/// Port of `silk/arm/NSQ_neon.c:silk_NSQ_noise_shape_feedback_loop_neon`.
///
/// Only handles order == 8 with NEON. Other orders fall through to scalar.
/// NOTE: This is intentionally NOT bit-exact with the scalar C version. As
/// the C code comments: "we do not drop the lower 16 bits of each multiply,
/// but wait until the end to truncate precision". This is encoder-only and
/// does not affect decoder output.
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn silk_NSQ_noise_shape_feedback_loop_neon(
    data0: i32,
    data1: &mut [i32],
    coef: &[i16],
    order: i32,
) -> i32 {
    if order == 8 {
        // a00 = [data0, data0, data0, data0]
        let a00 = vdupq_n_s32(data0);
        // a01 = [data1[0], data1[1], data1[2], data1[3]]
        let a01 = vld1q_s32(data1.as_ptr());

        // a0 = [data0, data1[0], data1[1], data1[2]] (insert data0 as first element)
        let a0 = vextq_s32(a00, a01, 3);
        // a1 = [data1[3], data1[4], data1[5], data1[6]]
        let a1 = vld1q_s32(data1.as_ptr().add(3));

        // Load and widen coefficients from i16 to i32
        let coef16 = vld1q_s16(coef.as_ptr());
        let coef0 = vmovl_s16(vget_low_s16(coef16));
        let coef1 = vmovl_s16(vget_high_s16(coef16));

        // Widening multiply and accumulate: i32 * i32 -> i64
        let b0 = vmull_s32(vget_low_s32(a0), vget_low_s32(coef0));
        let b1 = vmlal_s32(b0, vget_high_s32(a0), vget_high_s32(coef0));
        let b2 = vmlal_s32(b1, vget_low_s32(a1), vget_low_s32(coef1));
        let b3 = vmlal_s32(b2, vget_high_s32(a1), vget_high_s32(coef1));

        // Horizontal sum of i64, then round-shift right by 15
        let c = vadd_s64(vget_low_s64(b3), vget_high_s64(b3));
        let c_shifted = vrshr_n_s64::<15>(c);
        let d = vreinterpret_s32_s64(c_shifted);

        let out = vget_lane_s32::<0>(d);

        // Shift data1: store a0 and a1 back
        vst1q_s32(data1.as_mut_ptr(), a0);
        vst1q_s32(data1.as_mut_ptr().add(4), a1);

        return out;
    }

    // Fall through to scalar for other orders
    crate::silk::NSQ::silk_NSQ_noise_shape_feedback_loop_c(data0, data1, coef, order)
}

/// NEON implementation of `silk_inner_product_FLP`.
/// f32→f64 inner product using NEON widening conversion and FMA.
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn silk_inner_product_flp_neon(data1: &[f32], data2: &[f32]) -> f64 {
    let n = data1.len().min(data2.len());
    let mut acc1 = vdupq_n_f64(0.0);
    let mut acc2 = vdupq_n_f64(0.0);
    let mut i = 0usize;

    // Main loop: 4 f32s per iteration → 2 pairs of f64s
    while i + 3 < n {
        let x = vld1q_f32(data1.as_ptr().add(i));
        let y = vld1q_f32(data2.as_ptr().add(i));

        // Low 2 elements: f32 → f64
        let x_lo = vcvt_f64_f32(vget_low_f32(x));
        let y_lo = vcvt_f64_f32(vget_low_f32(y));
        acc1 = vfmaq_f64(acc1, x_lo, y_lo);

        // High 2 elements: f32 → f64
        let x_hi = vcvt_f64_f32(vget_high_f32(x));
        let y_hi = vcvt_f64_f32(vget_high_f32(y));
        acc2 = vfmaq_f64(acc2, x_hi, y_hi);

        i += 4;
    }

    // Combine accumulators and horizontal sum
    acc1 = vaddq_f64(acc1, acc2);
    let mut result = vgetq_lane_f64(acc1, 0) + vgetq_lane_f64(acc1, 1);

    // Scalar tail for remaining 0-3 elements
    while i < n {
        result += *data1.get_unchecked(i) as f64 * *data2.get_unchecked(i) as f64;
        i += 1;
    }

    result
}

const QA: i32 = 24;
const A_LIMIT: i32 = SILK_FIX_CONST!(0.99975, QA);

fn MUL32_FRAC_Q(a32: i32, b32: i32, Q: i32) -> i32 {
    silk_RSHIFT_ROUND64(a32 as i64 * b32 as i64, Q) as i32
}

/// NEON-accelerated inner function for LPC inverse prediction gain.
/// Port of `silk/arm/LPC_inv_pred_gain_neon_intr.c:LPC_inverse_pred_gain_QA_neon`.
///
/// The NEON version processes 4 AR coefficients at a time and uses a clever
/// overflow detection scheme based on narrowing shifts.
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
unsafe fn LPC_inverse_pred_gain_QA_neon(A_QA: &mut [i32; SILK_MAX_ORDER_LPC], order: usize) -> i32 {
    let mut max_s32x4 = vdupq_n_s32(i32::MIN);
    let mut min_s32x4 = vdupq_n_s32(i32::MAX);
    let mut invGain_Q30 = SILK_FIX_CONST!(1.0, 30);

    let mut k = order - 1;
    while k > 0 {
        // Check for stability
        if A_QA[k] > A_LIMIT || A_QA[k] < -A_LIMIT {
            return 0;
        }

        // Set RC equal to negated AR coef
        let rc_Q31 = -(A_QA[k] << (31 - QA));

        // rc_mult1_Q30 range: [ 1 : 2^30 ]
        let rc_mult1_Q30 = SILK_FIX_CONST!(1, 30) - silk_SMMUL(rc_Q31, rc_Q31);
        debug_assert!(rc_mult1_Q30 > (1 << 15));
        debug_assert!(rc_mult1_Q30 <= (1 << 30));

        // Update inverse gain
        invGain_Q30 = silk_SMMUL(invGain_Q30, rc_mult1_Q30) << 2;
        debug_assert!(invGain_Q30 >= 0);
        debug_assert!(invGain_Q30 <= (1 << 30));
        if invGain_Q30 < SILK_FIX_CONST!(1.0 / MAX_PREDICTION_POWER_GAIN, 30) {
            return 0;
        }

        // rc_mult2 range: [ 2^30 : silk_int32_MAX ]
        let mult2Q = 32 - silk_CLZ32(rc_mult1_Q30.abs());
        let rc_mult2 = silk_INVERSE32_varQ(rc_mult1_Q30, mult2Q + 30);

        // NEON: Update AR coefficients 4 at a time
        let rc_Q31_s32x2 = vdup_n_s32(rc_Q31);
        let mult2Q_s64x2 = vdupq_n_s64(-(mult2Q as i64));
        let rc_mult2_s32x2 = vdup_n_s32(rc_mult2);

        let half = (k + 1) >> 1;
        let mut n = 0usize;
        while n + 4 <= half {
            // Load forward and backward elements
            let tmp1_s32x4 = vld1q_s32(A_QA.as_ptr().add(n));
            let tmp2_raw = vld1q_s32(A_QA.as_ptr().add(k - n - 4));
            // Reverse tmp2: vrev64q reverses within 64-bit halves, then swap halves
            let tmp2_s32x4 = vcombine_s32(
                vget_high_s32(vrev64q_s32(tmp2_raw)),
                vget_low_s32(vrev64q_s32(tmp2_raw)),
            );

            // vqrdmulhq_lane_s32: saturating rounding doubling multiply high
            // Equivalent to MUL32_FRAC_Q(x, rc_Q31, 31)
            let t0_s32x4 = vqrdmulhq_lane_s32::<0>(tmp2_s32x4, rc_Q31_s32x2);
            let t1_s32x4 = vqrdmulhq_lane_s32::<0>(tmp1_s32x4, rc_Q31_s32x2);
            let t_QA0_s32x4 = vqsubq_s32(tmp1_s32x4, t0_s32x4);
            let t_QA1_s32x4 = vqsubq_s32(tmp2_s32x4, t1_s32x4);

            // Widening multiply and variable shift right (rounding)
            let t0_s64x2 = vmull_s32(vget_low_s32(t_QA0_s32x4), rc_mult2_s32x2);
            let t1_s64x2 = vmull_s32(vget_high_s32(t_QA0_s32x4), rc_mult2_s32x2);
            let t2_s64x2 = vmull_s32(vget_low_s32(t_QA1_s32x4), rc_mult2_s32x2);
            let t3_s64x2 = vmull_s32(vget_high_s32(t_QA1_s32x4), rc_mult2_s32x2);

            let t0_s64x2 = vrshlq_s64(t0_s64x2, mult2Q_s64x2);
            let t1_s64x2 = vrshlq_s64(t1_s64x2, mult2Q_s64x2);
            let t2_s64x2 = vrshlq_s64(t2_s64x2, mult2Q_s64x2);
            let t3_s64x2 = vrshlq_s64(t3_s64x2, mult2Q_s64x2);

            // Narrow to i32 for result
            let r0_s32x4 = vcombine_s32(vmovn_s64(t0_s64x2), vmovn_s64(t1_s64x2));
            let r1_s32x4 = vcombine_s32(vmovn_s64(t2_s64x2), vmovn_s64(t3_s64x2));

            // Overflow detection: narrowing shift right by 31
            // If tmp64 fits in 32 bits, then (tmp64 << 1) >> 32 is 0 or -1
            let s0_s32x4 = vcombine_s32(vshrn_n_s64(t0_s64x2, 31), vshrn_n_s64(t1_s64x2, 31));
            let s1_s32x4 = vcombine_s32(vshrn_n_s64(t2_s64x2, 31), vshrn_n_s64(t3_s64x2, 31));
            max_s32x4 = vmaxq_s32(max_s32x4, s0_s32x4);
            min_s32x4 = vminq_s32(min_s32x4, s0_s32x4);
            max_s32x4 = vmaxq_s32(max_s32x4, s1_s32x4);
            min_s32x4 = vminq_s32(min_s32x4, s1_s32x4);

            // Store results: r1 reversed back
            let r1_rev = vrev64q_s32(r1_s32x4);
            let r1_final = vcombine_s32(vget_high_s32(r1_rev), vget_low_s32(r1_rev));
            vst1q_s32(A_QA.as_mut_ptr().add(n), r0_s32x4);
            vst1q_s32(A_QA.as_mut_ptr().add(k - n - 4), r1_final);

            n += 4;
        }

        // Scalar tail for remaining elements
        while n < half {
            let tmp1 = A_QA[n];
            let tmp2 = A_QA[k - n - 1];
            let tmp64 = silk_RSHIFT_ROUND64(
                tmp1.saturating_sub(MUL32_FRAC_Q(tmp2, rc_Q31, 31)) as i64 * rc_mult2 as i64,
                mult2Q,
            );
            if tmp64 > i32::MAX as i64 || tmp64 < i32::MIN as i64 {
                return 0;
            }
            A_QA[n] = tmp64 as i32;
            let tmp64 = silk_RSHIFT_ROUND64(
                tmp2.saturating_sub(MUL32_FRAC_Q(tmp1, rc_Q31, 31)) as i64 * rc_mult2 as i64,
                mult2Q,
            );
            if tmp64 > i32::MAX as i64 || tmp64 < i32::MIN as i64 {
                return 0;
            }
            A_QA[k - n - 1] = tmp64 as i32;
            n += 1;
        }
        k -= 1;
    }

    // Check for stability
    if A_QA[0] > A_LIMIT || A_QA[0] < -A_LIMIT {
        return 0;
    }

    // Check NEON overflow detection results
    let max_s32x2 = vmax_s32(vget_low_s32(max_s32x4), vget_high_s32(max_s32x4));
    let min_s32x2 = vmin_s32(vget_low_s32(min_s32x4), vget_high_s32(min_s32x4));
    let max_s32x2 = vmax_s32(
        max_s32x2,
        vreinterpret_s32_s64(vshr_n_s64::<32>(vreinterpret_s64_s32(max_s32x2))),
    );
    let min_s32x2 = vmin_s32(
        min_s32x2,
        vreinterpret_s32_s64(vshr_n_s64::<32>(vreinterpret_s64_s32(min_s32x2))),
    );
    let max_val = vget_lane_s32::<0>(max_s32x2);
    let min_val = vget_lane_s32::<0>(min_s32x2);
    if max_val > 0 || min_val < -1 {
        return 0;
    }

    // Set RC equal to negated AR coef
    let rc_Q31 = -(A_QA[0] << (31 - QA));

    // Range: [ 1 : 2^30 ]
    let rc_mult1_Q30 = SILK_FIX_CONST!(1, 30) - silk_SMMUL(rc_Q31, rc_Q31);

    // Update inverse gain
    invGain_Q30 = silk_SMMUL(invGain_Q30, rc_mult1_Q30) << 2;
    debug_assert!(invGain_Q30 >= 0);
    debug_assert!(invGain_Q30 <= (1 << 30));
    if invGain_Q30 < SILK_FIX_CONST!(1.0 / MAX_PREDICTION_POWER_GAIN, 30) {
        0
    } else {
        invGain_Q30
    }
}

/// NEON implementation of `silk_LPC_inverse_pred_gain`.
/// Port of `silk/arm/LPC_inv_pred_gain_neon_intr.c:silk_LPC_inverse_pred_gain_neon`.
///
/// Uses NEON widening shifts for Q12→QA conversion and NEON pairwise adds
/// for DC response accumulation.
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn silk_LPC_inverse_pred_gain_neon(A_Q12: &[i16]) -> i32 {
    let order = A_Q12.len();

    if SILK_MAX_ORDER_LPC != 24 || (order & 1) != 0 {
        return crate::silk::LPC_inv_pred_gain::silk_LPC_inverse_pred_gain_c(A_Q12);
    }

    let mut Atmp_QA = [0i32; SILK_MAX_ORDER_LPC];

    // Use NEON to widen A_Q12 from i16 to i32 and shift left by (QA - 12) = 12
    // Also compute DC response (sum of all A_Q12)
    let groups_of_8 = order / 8;
    let leftover = order & 7;
    let mut dc_acc = vdupq_n_s32(0);

    for g in 0..groups_of_8 {
        let off = g * 8;
        let t = vld1q_s16(A_Q12.as_ptr().add(off));
        dc_acc = vpadalq_s16(dc_acc, t);
        vst1q_s32(
            Atmp_QA.as_mut_ptr().add(off),
            vshll_n_s16::<12>(vget_low_s16(t)),
        );
        vst1q_s32(
            Atmp_QA.as_mut_ptr().add(off + 4),
            vshll_n_s16::<12>(vget_high_s16(t)),
        );
    }

    // Horizontal sum of dc_acc
    let dc_pair = vpadd_s32(vget_low_s32(dc_acc), vget_high_s32(dc_acc));
    let dc_s64 = vpaddl_s32(dc_pair);
    let mut DC_resp = vget_lane_s32::<0>(vreinterpret_s32_s64(dc_s64));

    // Handle leftover elements scalar
    let base = groups_of_8 * 8;
    for i in 0..leftover {
        DC_resp += A_Q12[base + i] as i32;
        Atmp_QA[base + i] = (A_Q12[base + i] as i32) << (QA - 12);
    }

    if DC_resp >= 4096 {
        0
    } else {
        LPC_inverse_pred_gain_QA_neon(&mut Atmp_QA, order)
    }
}

// ---------------------------------------------------------------------------
// NEON NSQ del_dec — Structure-of-Arrays (SoA) delayed decision quantizer
// Port of silk/arm/NSQ_del_dec_neon_intr.c
// ---------------------------------------------------------------------------

/// Maximum number of delayed-decision states that the NEON path supports.
const NEON_MAX_DEL_DEC_STATES: usize = 4;

/// SoA delayed-decision state for up to 4 states processed in parallel via NEON.
/// Each `[i32; 4]` holds one value per state and is loaded/stored as `int32x4_t`.
#[derive(Clone)]
#[repr(C, align(16))]
struct NeonDelDecStates {
    sLPC_Q14: [[i32; NEON_MAX_DEL_DEC_STATES]; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
    RandState: [[i32; NEON_MAX_DEL_DEC_STATES]; DECISION_DELAY as usize],
    Q_Q10: [[i32; NEON_MAX_DEL_DEC_STATES]; DECISION_DELAY as usize],
    Xq_Q14: [[i32; NEON_MAX_DEL_DEC_STATES]; DECISION_DELAY as usize],
    Pred_Q15: [[i32; NEON_MAX_DEL_DEC_STATES]; DECISION_DELAY as usize],
    Shape_Q14: [[i32; NEON_MAX_DEL_DEC_STATES]; DECISION_DELAY as usize],
    sAR2_Q14: [[i32; NEON_MAX_DEL_DEC_STATES]; MAX_SHAPE_LPC_ORDER as usize],
    LF_AR_Q14: [i32; NEON_MAX_DEL_DEC_STATES],
    Diff_Q14: [i32; NEON_MAX_DEL_DEC_STATES],
    Seed: [i32; NEON_MAX_DEL_DEC_STATES],
    SeedInit: [i32; NEON_MAX_DEL_DEC_STATES],
    RD_Q10: [i32; NEON_MAX_DEL_DEC_STATES],
}

impl NeonDelDecStates {
    fn new_zeroed() -> Box<Self> {
        // Use a boxed allocation to avoid stack overflow from the large struct
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Self;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(ptr)
        }
    }
}

/// Per-sample candidate state for quantization decisions.
#[derive(Copy, Clone, Default)]
#[repr(C, align(16))]
struct NeonSampleState {
    Q_Q10: [i32; NEON_MAX_DEL_DEC_STATES],
    RD_Q10: [i32; NEON_MAX_DEL_DEC_STATES],
    xq_Q14: [i32; NEON_MAX_DEL_DEC_STATES],
    LF_AR_Q14: [i32; NEON_MAX_DEL_DEC_STATES],
    Diff_Q14: [i32; NEON_MAX_DEL_DEC_STATES],
    sLTP_shp_Q14: [i32; NEON_MAX_DEL_DEC_STATES],
    LPC_exc_Q14: [i32; NEON_MAX_DEL_DEC_STATES],
}

/// Helper: saturating round-shift for xq output: silk_RSHIFT_ROUND + silk_SAT16
#[inline]
fn neon_rshift_round_sat16(val: i32, shift: i32) -> i16 {
    let rounded = if shift == 1 {
        (val >> 1) + (val & 1)
    } else {
        ((val >> (shift - 1)) + 1) >> 1
    };
    if rounded > i16::MAX as i32 {
        i16::MAX
    } else if rounded < i16::MIN as i32 {
        i16::MIN
    } else {
        rounded as i16
    }
}

/// SMLAWB on NEON vectors: out + vqdmulhq_lane_s32(in, coef, 0)
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_smlawb_lane0(
    out_s32x4: int32x4_t,
    in_s32x4: int32x4_t,
    coef_s32x2: int32x2_t,
) -> int32x4_t {
    vaddq_s32(out_s32x4, vqdmulhq_lane_s32::<0>(in_s32x4, coef_s32x2))
}

#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_smlawb_lane1(
    out_s32x4: int32x4_t,
    in_s32x4: int32x4_t,
    coef_s32x2: int32x2_t,
) -> int32x4_t {
    vaddq_s32(out_s32x4, vqdmulhq_lane_s32::<1>(in_s32x4, coef_s32x2))
}

/// Create arch-specific coefficients for short-term prediction (reversed, widened to i32 << 15).
#[target_feature(enable = "neon")]
unsafe fn neon_short_prediction_create_arch_coef(
    out: &mut [i32; MAX_LPC_ORDER],
    coefs: &[i16],
    order: i32,
) {
    debug_assert!(order == 10 || order == 16);

    let t_s16x8 = vld1q_s16(coefs.as_ptr());
    let t_s16x8 = vrev64q_s16(t_s16x8);
    let t2 = vshll_n_s16::<15>(vget_high_s16(t_s16x8));
    let t3 = vshll_n_s16::<15>(vget_low_s16(t_s16x8));

    let (t0, t1);
    if order == 16 {
        let t_s16x8b = vld1q_s16(coefs.as_ptr().add(8));
        let t_s16x8b = vrev64q_s16(t_s16x8b);
        t0 = vshll_n_s16::<15>(vget_high_s16(t_s16x8b));
        t1 = vshll_n_s16::<15>(vget_low_s16(t_s16x8b));
    } else {
        let zero = vdupq_n_s32(0);
        let t_s16x4 = vld1_s16(coefs.as_ptr().add(6));
        let t_s16x4 = vrev64_s16(t_s16x4);
        let tmp = vshll_n_s16::<15>(t_s16x4);
        t1 = vcombine_s32(vget_low_s32(zero), vget_low_s32(tmp));
        t0 = zero;
    }

    vst1q_s32(out.as_mut_ptr(), t0);
    vst1q_s32(out.as_mut_ptr().add(4), t1);
    vst1q_s32(out.as_mut_ptr().add(8), t2);
    vst1q_s32(out.as_mut_ptr().add(12), t3);
}

/// NEON short-term prediction for the SoA del_dec quantizer.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_short_prediction_local(
    buf32: *const [i32; NEON_MAX_DEL_DEC_STATES],
    a_Q12_arch: &[i32; MAX_LPC_ORDER],
    order: i32,
) -> int32x4_t {
    let a0 = vld1q_s32(a_Q12_arch.as_ptr());
    let a1 = vld1q_s32(a_Q12_arch.as_ptr().add(4));
    let a2 = vld1q_s32(a_Q12_arch.as_ptr().add(8));
    let a3 = vld1q_s32(a_Q12_arch.as_ptr().add(12));

    let mut pred = vdupq_n_s32(order >> 1);

    pred = neon_smlawb_lane0(pred, vld1q_s32((*buf32.add(0)).as_ptr()), vget_low_s32(a0));
    pred = neon_smlawb_lane1(pred, vld1q_s32((*buf32.add(1)).as_ptr()), vget_low_s32(a0));
    pred = neon_smlawb_lane0(pred, vld1q_s32((*buf32.add(2)).as_ptr()), vget_high_s32(a0));
    pred = neon_smlawb_lane1(pred, vld1q_s32((*buf32.add(3)).as_ptr()), vget_high_s32(a0));
    pred = neon_smlawb_lane0(pred, vld1q_s32((*buf32.add(4)).as_ptr()), vget_low_s32(a1));
    pred = neon_smlawb_lane1(pred, vld1q_s32((*buf32.add(5)).as_ptr()), vget_low_s32(a1));
    pred = neon_smlawb_lane0(pred, vld1q_s32((*buf32.add(6)).as_ptr()), vget_high_s32(a1));
    pred = neon_smlawb_lane1(pred, vld1q_s32((*buf32.add(7)).as_ptr()), vget_high_s32(a1));
    pred = neon_smlawb_lane0(pred, vld1q_s32((*buf32.add(8)).as_ptr()), vget_low_s32(a2));
    pred = neon_smlawb_lane1(pred, vld1q_s32((*buf32.add(9)).as_ptr()), vget_low_s32(a2));
    pred = neon_smlawb_lane0(
        pred,
        vld1q_s32((*buf32.add(10)).as_ptr()),
        vget_high_s32(a2),
    );
    pred = neon_smlawb_lane1(
        pred,
        vld1q_s32((*buf32.add(11)).as_ptr()),
        vget_high_s32(a2),
    );
    pred = neon_smlawb_lane0(pred, vld1q_s32((*buf32.add(12)).as_ptr()), vget_low_s32(a3));
    pred = neon_smlawb_lane1(pred, vld1q_s32((*buf32.add(13)).as_ptr()), vget_low_s32(a3));
    pred = neon_smlawb_lane0(
        pred,
        vld1q_s32((*buf32.add(14)).as_ptr()),
        vget_high_s32(a3),
    );
    pred = neon_smlawb_lane1(
        pred,
        vld1q_s32((*buf32.add(15)).as_ptr()),
        vget_high_s32(a3),
    );

    pred
}

/// SMULWB for 8 i16 elements producing 8 i32 results.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_smulwb_8(a: *const i16, b: int32x2_t, o: *mut i32) {
    let a_s16x8 = vld1q_s16(a);
    let o0 = vqdmulhq_lane_s32::<0>(vshll_n_s16::<15>(vget_low_s16(a_s16x8)), b);
    let o1 = vqdmulhq_lane_s32::<0>(vshll_n_s16::<15>(vget_high_s16(a_s16x8)), b);
    vst1q_s32(o, o0);
    vst1q_s32(o.add(4), o1);
}

/// SMULWW with small b (|b| < 65536) applied to 4 elements in-place.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_smulww_small_4(a: *mut i32, b: int32x2_t) {
    let v = vld1q_s32(a);
    vst1q_s32(a, vqdmulhq_lane_s32::<0>(v, b));
}

/// SMULWW with small b applied to 8 elements in-place.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_smulww_small_8(a: *mut i32, b: int32x2_t) {
    let v0 = vld1q_s32(a);
    let v1 = vld1q_s32(a.add(4));
    vst1q_s32(a, vqdmulhq_lane_s32::<0>(v0, b));
    vst1q_s32(a.add(4), vqdmulhq_lane_s32::<0>(v1, b));
}

/// Full SMULWW applied to 4 elements in-place (for large b values).
/// b lane 0 = (gain & 0xFFFF) << 15, lane 1 = gain >> 16
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_smulww_4(a: *mut i32, b: int32x2_t) {
    let v = vld1q_s32(a);
    let mut o = vqdmulhq_lane_s32::<0>(v, b);
    o = vmlaq_lane_s32::<1>(o, v, b);
    vst1q_s32(a, o);
}

/// Full SMULWW applied to 8 elements in-place.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn neon_smulww_8(a: *mut i32, b: int32x2_t) {
    let v0 = vld1q_s32(a);
    let v1 = vld1q_s32(a.add(4));
    let mut o0 = vqdmulhq_lane_s32::<0>(v0, b);
    let mut o1 = vqdmulhq_lane_s32::<0>(v1, b);
    o0 = vmlaq_lane_s32::<1>(o0, v0, b);
    o1 = vmlaq_lane_s32::<1>(o1, v1, b);
    vst1q_s32(a, o0);
    vst1q_s32(a.add(4), o1);
}

/// SMULWW loop: compute o[i] = silk_SMULWW(a[i], b) using NEON.
#[target_feature(enable = "neon")]
unsafe fn neon_smulww_loop(a: *const i16, b: i32, o: *mut i32, count: i32) {
    let b_v = vdup_n_s32(b);
    let mut i = 0i32;
    while i < count - 7 {
        neon_smulwb_8(a.offset(i as isize), b_v, o.offset(i as isize));
        i += 8;
    }
    while i < count {
        *o.offset(i as isize) = silk_SMULWW(*a.offset(i as isize) as i32, b);
        i += 1;
    }
}

/// Copy winner state data to output buffers (8 samples at a time).
#[target_feature(enable = "neon")]
unsafe fn neon_copy_winner_kernel(
    dd: &NeonDelDecStates,
    offset: i32,
    last_idx: usize,
    winner: usize,
    gain_lo: int32x2_t,
    gain_hi: int32x2_t,
    shift: int32x4_t,
    mut t0: int32x4_t,
    mut t1: int32x4_t,
    pulses: &mut [i8],
    p_off: usize,
    xq_off: usize,
    nsq: &mut silk_nsq_state,
) {
    // Load Q_Q10 from winner state
    t0 = vld1q_lane_s32::<0>(&dd.Q_Q10[last_idx][winner], t0);
    t0 = vld1q_lane_s32::<1>(&dd.Q_Q10[last_idx - 1][winner], t0);
    t0 = vld1q_lane_s32::<2>(&dd.Q_Q10[last_idx - 2][winner], t0);
    t0 = vld1q_lane_s32::<3>(&dd.Q_Q10[last_idx - 3][winner], t0);
    t1 = vld1q_lane_s32::<0>(&dd.Q_Q10[last_idx - 4][winner], t1);
    t1 = vld1q_lane_s32::<1>(&dd.Q_Q10[last_idx - 5][winner], t1);
    t1 = vld1q_lane_s32::<2>(&dd.Q_Q10[last_idx - 6][winner], t1);
    t1 = vld1q_lane_s32::<3>(&dd.Q_Q10[last_idx - 7][winner], t1);
    let t_s16x8 = vcombine_s16(vrshrn_n_s32::<10>(t0), vrshrn_n_s32::<10>(t1));
    let dst = (p_off as isize + offset as isize) as usize;
    vst1_s8(pulses.as_mut_ptr().add(dst), vmovn_s16(t_s16x8));

    // Load Xq_Q14, apply gain and shift
    t0 = vld1q_lane_s32::<0>(&dd.Xq_Q14[last_idx][winner], t0);
    t0 = vld1q_lane_s32::<1>(&dd.Xq_Q14[last_idx - 1][winner], t0);
    t0 = vld1q_lane_s32::<2>(&dd.Xq_Q14[last_idx - 2][winner], t0);
    t0 = vld1q_lane_s32::<3>(&dd.Xq_Q14[last_idx - 3][winner], t0);
    t1 = vld1q_lane_s32::<0>(&dd.Xq_Q14[last_idx - 4][winner], t1);
    t1 = vld1q_lane_s32::<1>(&dd.Xq_Q14[last_idx - 5][winner], t1);
    t1 = vld1q_lane_s32::<2>(&dd.Xq_Q14[last_idx - 6][winner], t1);
    t1 = vld1q_lane_s32::<3>(&dd.Xq_Q14[last_idx - 7][winner], t1);
    let mut o0 = vqdmulhq_lane_s32::<0>(t0, gain_lo);
    let mut o1 = vqdmulhq_lane_s32::<0>(t1, gain_lo);
    o0 = vmlaq_lane_s32::<0>(o0, t0, gain_hi);
    o1 = vmlaq_lane_s32::<0>(o1, t1, gain_hi);
    o0 = vrshlq_s32(o0, shift);
    o1 = vrshlq_s32(o1, shift);
    let xdst = (xq_off as isize + offset as isize) as usize;
    vst1_s16(nsq.xq.as_mut_ptr().add(xdst), vqmovn_s32(o0));
    vst1_s16(nsq.xq.as_mut_ptr().add(xdst + 4), vqmovn_s32(o1));

    // Load Shape_Q14
    t0 = vld1q_lane_s32::<0>(&dd.Shape_Q14[last_idx][winner], t0);
    t0 = vld1q_lane_s32::<1>(&dd.Shape_Q14[last_idx - 1][winner], t0);
    t0 = vld1q_lane_s32::<2>(&dd.Shape_Q14[last_idx - 2][winner], t0);
    t0 = vld1q_lane_s32::<3>(&dd.Shape_Q14[last_idx - 3][winner], t0);
    t1 = vld1q_lane_s32::<0>(&dd.Shape_Q14[last_idx - 4][winner], t1);
    t1 = vld1q_lane_s32::<1>(&dd.Shape_Q14[last_idx - 5][winner], t1);
    t1 = vld1q_lane_s32::<2>(&dd.Shape_Q14[last_idx - 6][winner], t1);
    t1 = vld1q_lane_s32::<3>(&dd.Shape_Q14[last_idx - 7][winner], t1);
    let shp_base = (nsq.sLTP_shp_buf_idx + offset) as usize;
    vst1q_s32(nsq.sLTP_shp_Q14.as_mut_ptr().add(shp_base), t0);
    vst1q_s32(nsq.sLTP_shp_Q14.as_mut_ptr().add(shp_base + 4), t1);
}

/// Copy winner state outputs using the vectorized kernel plus scalar tail.
#[target_feature(enable = "neon")]
unsafe fn neon_copy_winner_state(
    dd: &NeonDelDecStates,
    decision_delay: i32,
    smpl_buf_idx: i32,
    winner: usize,
    gain: i32,
    shift: i32,
    pulses: &mut [i8],
    p_off: usize,
    xq_off: usize,
    nsq: &mut silk_nsq_state,
) {
    let gain_lo = vdup_n_s32(((gain & 0x0000FFFF) as u32 as i32) << 15);
    let gain_hi = vdup_n_s32(gain >> 16);
    let shift_v = vdupq_n_s32(-shift);
    let t0 = vdupq_n_s32(0);
    let t1 = vdupq_n_s32(0);

    let mut last_idx = (smpl_buf_idx + decision_delay - 1 + DECISION_DELAY) as usize;
    if last_idx >= DECISION_DELAY as usize {
        last_idx -= DECISION_DELAY as usize;
    }
    if last_idx >= DECISION_DELAY as usize {
        last_idx -= DECISION_DELAY as usize;
    }

    let mut i = 0i32;
    // First vectorized loop
    while i < (decision_delay - 7) && last_idx >= 7 {
        neon_copy_winner_kernel(
            dd,
            i - decision_delay,
            last_idx,
            winner,
            gain_lo,
            gain_hi,
            shift_v,
            t0,
            t1,
            pulses,
            p_off,
            xq_off,
            nsq,
        );
        i += 8;
        last_idx -= 8;
    }
    // Scalar tail for wrap-around
    while i < decision_delay && last_idx < DECISION_DELAY as usize {
        let pidx = (p_off as isize + (i - decision_delay) as isize) as usize;
        pulses[pidx] = (((dd.Q_Q10[last_idx][winner] >> 9) + 1) >> 1) as i8;
        let xq_val = (dd.Xq_Q14[last_idx][winner] as i64 * gain as i64) >> 16;
        let xidx = (xq_off as isize + (i - decision_delay) as isize) as usize;
        nsq.xq[xidx] = neon_rshift_round_sat16(xq_val as i32, shift);
        nsq.sLTP_shp_Q14[(nsq.sLTP_shp_buf_idx - decision_delay + i) as usize] =
            dd.Shape_Q14[last_idx][winner];
        i += 1;
        if last_idx == 0 {
            break;
        }
        last_idx -= 1;
    }

    // After wrap-around
    if i < decision_delay {
        last_idx = last_idx.wrapping_add(DECISION_DELAY as usize);
        while i < (decision_delay - 7) {
            neon_copy_winner_kernel(
                dd,
                i - decision_delay,
                last_idx,
                winner,
                gain_lo,
                gain_hi,
                shift_v,
                t0,
                t1,
                pulses,
                p_off,
                xq_off,
                nsq,
            );
            i += 8;
            last_idx -= 8;
        }
        while i < decision_delay {
            let pidx = (p_off as isize + (i - decision_delay) as isize) as usize;
            pulses[pidx] = (((dd.Q_Q10[last_idx][winner] >> 9) + 1) >> 1) as i8;
            let xq_val = (dd.Xq_Q14[last_idx][winner] as i64 * gain as i64) >> 16;
            let xidx = (xq_off as isize + (i - decision_delay) as isize) as usize;
            nsq.xq[xidx] = neon_rshift_round_sat16(xq_val as i32, shift);
            nsq.sLTP_shp_Q14[(nsq.sLTP_shp_buf_idx - decision_delay + i) as usize] =
                dd.Shape_Q14[last_idx][winner];
            i += 1;
            last_idx -= 1;
        }
    }
}

/// NEON noise shape quantizer for one subframe (del_dec variant).
/// `xq_off` is the offset into `nsq.xq` for output.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn neon_noise_shape_quantizer_del_dec(
    nsq: &mut silk_nsq_state,
    dd: &mut NeonDelDecStates,
    signal_type: i32,
    x_Q10: &[i32],
    pulses: &mut [i8],
    p_off: usize,
    xq_off: usize,
    sLTP_Q15: &mut [i32],
    delayed_gain_Q10: &mut [i32; DECISION_DELAY as usize],
    a_Q12: &[i16],
    b_Q14: &[i16],
    AR_shp_Q13: &[i16],
    lag: i32,
    harm_shape_packed: i32,
    tilt_Q14: i32,
    lf_shp_Q14: i32,
    gain_Q16: i32,
    lambda_Q10: i32,
    offset_Q10: i32,
    length: i32,
    subfr: i32,
    shaping_order: i32,
    predict_order: i32,
    warping_Q16: i32,
    n_states: i32,
    smpl_buf_idx: &mut i32,
    decision_delay: i32,
) {
    let warping_v = vdup_n_s32(((warping_Q16 as u32) << 16) as i32 >> 1);
    let lf_shp_Q29 = ((lf_shp_Q14 as u32) << 16) as i32 >> 1;
    let mut ar_shp_Q28 = [0i32; MAX_SHAPE_LPC_ORDER as usize];
    let rand_mul = vdupq_n_u32(RAND_MULTIPLIER as u32);
    let rand_inc = vdupq_n_u32(RAND_INCREMENT as u32);

    let mut ss = [NeonSampleState::default(); 2];

    let mut shp_lag_idx = (nsq.sLTP_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS / 2) as usize;
    let mut pred_lag_idx = (nsq.sLTP_buf_idx - lag + LTP_ORDER as i32 / 2) as usize;
    let gain_Q10 = gain_Q16 >> 6;

    // Pre-shift AR_shp_Q13 to Q28
    {
        let mut j = 0i32;
        while j < MAX_SHAPE_LPC_ORDER - 7 {
            let t = vld1q_s16(AR_shp_Q13.as_ptr().add(j as usize));
            vst1q_s32(
                ar_shp_Q28.as_mut_ptr().add(j as usize),
                vshll_n_s16::<15>(vget_low_s16(t)),
            );
            vst1q_s32(
                ar_shp_Q28.as_mut_ptr().add(j as usize + 4),
                vshll_n_s16::<15>(vget_high_s16(t)),
            );
            j += 8;
        }
        while j < MAX_SHAPE_LPC_ORDER {
            ar_shp_Q28[j as usize] = (AR_shp_Q13[j as usize] as i32) << 15;
            j += 1;
        }
    }

    let mut a_Q12_arch = [0i32; MAX_LPC_ORDER];
    neon_short_prediction_create_arch_coef(&mut a_Q12_arch, a_Q12, predict_order);

    #[allow(clippy::needless_range_loop)]
    for i in 0..length as usize {
        // Long-term prediction (shared)
        let ltp_pred;
        if signal_type == TYPE_VOICED {
            let mut p = 2i32;
            p = silk_SMLAWB(p, sLTP_Q15[pred_lag_idx], b_Q14[0] as i32);
            p = silk_SMLAWB(p, sLTP_Q15[pred_lag_idx - 1], b_Q14[1] as i32);
            p = silk_SMLAWB(p, sLTP_Q15[pred_lag_idx - 2], b_Q14[2] as i32);
            p = silk_SMLAWB(p, sLTP_Q15[pred_lag_idx - 3], b_Q14[3] as i32);
            p = silk_SMLAWB(p, sLTP_Q15[pred_lag_idx - 4], b_Q14[4] as i32);
            ltp_pred = ((p as u32) << 1) as i32;
            pred_lag_idx += 1;
        } else {
            ltp_pred = 0;
        }

        // Harmonic noise shaping (shared)
        let n_ltp;
        if lag > 0 {
            let sum = nsq.sLTP_shp_Q14[shp_lag_idx].wrapping_add(nsq.sLTP_shp_Q14[shp_lag_idx - 2]);
            let mut v = silk_SMULWB(sum, harm_shape_packed);
            v = (v as i64
                + ((nsq.sLTP_shp_Q14[shp_lag_idx - 1] as i64 * (harm_shape_packed as i64 >> 16))
                    >> 16)) as i32;
            n_ltp = ltp_pred - ((v as u32) << 2) as i32;
            shp_lag_idx += 1;
        } else {
            n_ltp = 0;
        }

        // PRNG
        let mut seed_v = vld1q_s32(dd.Seed.as_ptr());
        seed_v =
            vreinterpretq_s32_u32(vmlaq_u32(rand_inc, vreinterpretq_u32_s32(seed_v), rand_mul));
        vst1q_s32(dd.Seed.as_mut_ptr(), seed_v);

        // Short-term prediction (NEON)
        let buf_ptr = dd.sLPC_Q14[NSQ_LPC_BUF_LENGTH - 16 + i..].as_ptr();
        let mut lpc_v = neon_short_prediction_local(buf_ptr, &a_Q12_arch, predict_order);
        lpc_v = vshlq_n_s32::<4>(lpc_v);

        // Noise shape feedback
        let mut tmp2_v = neon_smlawb_lane0(
            vld1q_s32(dd.Diff_Q14.as_ptr()),
            vld1q_s32(dd.sAR2_Q14[0].as_ptr()),
            warping_v,
        );
        let mut tmp1_v = vsubq_s32(vld1q_s32(dd.sAR2_Q14[1].as_ptr()), tmp2_v);
        tmp1_v = neon_smlawb_lane0(vld1q_s32(dd.sAR2_Q14[0].as_ptr()), tmp1_v, warping_v);
        vst1q_s32(dd.sAR2_Q14[0].as_mut_ptr(), tmp2_v);

        let mut ar_v2 = vld1_s32(ar_shp_Q28.as_ptr());
        let mut n_ar_v = vaddq_s32(
            vdupq_n_s32(shaping_order >> 1),
            vqdmulhq_lane_s32::<0>(tmp2_v, ar_v2),
        );

        let mut j = 2i32;
        while j < shaping_order {
            tmp2_v = vsubq_s32(vld1q_s32(dd.sAR2_Q14[j as usize].as_ptr()), tmp1_v);
            tmp2_v = neon_smlawb_lane0(
                vld1q_s32(dd.sAR2_Q14[(j - 1) as usize].as_ptr()),
                tmp2_v,
                warping_v,
            );
            vst1q_s32(dd.sAR2_Q14[(j - 1) as usize].as_mut_ptr(), tmp1_v);
            n_ar_v = vaddq_s32(n_ar_v, vqdmulhq_lane_s32::<1>(tmp1_v, ar_v2));

            tmp1_v = vsubq_s32(vld1q_s32(dd.sAR2_Q14[(j + 1) as usize].as_ptr()), tmp2_v);
            tmp1_v = neon_smlawb_lane0(
                vld1q_s32(dd.sAR2_Q14[j as usize].as_ptr()),
                tmp1_v,
                warping_v,
            );
            vst1q_s32(dd.sAR2_Q14[j as usize].as_mut_ptr(), tmp2_v);
            ar_v2 = vld1_s32(ar_shp_Q28.as_ptr().add(j as usize));
            n_ar_v = vaddq_s32(n_ar_v, vqdmulhq_lane_s32::<0>(tmp2_v, ar_v2));

            j += 2;
        }
        vst1q_s32(
            dd.sAR2_Q14[(shaping_order - 1) as usize].as_mut_ptr(),
            tmp1_v,
        );
        n_ar_v = vaddq_s32(n_ar_v, vqdmulhq_lane_s32::<1>(tmp1_v, ar_v2));
        n_ar_v = vshlq_n_s32::<1>(n_ar_v); // Q11 -> Q12
        n_ar_v = vaddq_s32(
            n_ar_v,
            vqdmulhq_n_s32(
                vld1q_s32(dd.LF_AR_Q14.as_ptr()),
                ((tilt_Q14 as u32) << 16) as i32 >> 1,
            ),
        );
        n_ar_v = vshlq_n_s32::<2>(n_ar_v); // Q12 -> Q14

        let mut n_lf_v = vqdmulhq_n_s32(
            vld1q_s32(dd.Shape_Q14[*smpl_buf_idx as usize].as_ptr()),
            lf_shp_Q29,
        );
        n_lf_v = vaddq_s32(
            n_lf_v,
            vqdmulhq_n_s32(
                vld1q_s32(dd.LF_AR_Q14.as_ptr()),
                ((lf_shp_Q14 >> 16) as u32 as i32) << 15,
            ),
        );
        n_lf_v = vshlq_n_s32::<2>(n_lf_v); // Q12 -> Q14

        // Residual: r = x[i] - (LTP + LPC - n_AR - n_LF)
        let mut r_v = vaddq_s32(n_ar_v, n_lf_v);
        let pred_sum = vaddq_s32(vdupq_n_s32(n_ltp), lpc_v);
        r_v = vsubq_s32(pred_sum, r_v);
        r_v = vrshrq_n_s32::<4>(r_v);
        r_v = vsubq_s32(vdupq_n_s32(x_Q10[i]), r_v);

        // Flip sign depending on dither
        let sign_v = vreinterpretq_s32_u32(vcltq_s32(seed_v, vdupq_n_s32(0)));
        r_v = veorq_s32(r_v, sign_v);
        r_v = vsubq_s32(r_v, sign_v);
        r_v = vmaxq_s32(r_v, vdupq_n_s32(-(31 << 10)));
        r_v = vminq_s32(r_v, vdupq_n_s32(30 << 10));
        let r_s16 = vmovn_s32(r_v);

        // Quantization
        let (q_best, q_second);
        {
            let mut q1_s16 = vsub_s16(r_s16, vdup_n_s16(offset_Q10 as i16));
            let mut q1_Q0 = vshr_n_s16::<10>(q1_s16);

            if lambda_Q10 > 2048 {
                let rdo_off = (lambda_Q10 / 2 - 512) as i16;
                let gt = vcgt_s16(q1_s16, vdup_n_s16(rdo_off));
                let lt = vclt_s16(q1_s16, vdup_n_s16(-rdo_off));
                let mut soff = vbsl_s16(gt, vdup_n_s16(-rdo_off), vdup_n_s16(0));
                soff = vbsl_s16(lt, vdup_n_s16(rdo_off), soff);
                q1_Q0 = vreinterpret_s16_u16(vclt_s16(q1_s16, vdup_n_s16(0)));
                q1_Q0 = vbsl_s16(vorr_u16(gt, lt), vadd_s16(q1_s16, soff), q1_Q0);
                q1_Q0 = vshr_n_s16::<10>(q1_Q0);
            }

            let eq0 = vceq_s16(q1_Q0, vdup_n_s16(0));
            let eq_m1 = vceq_s16(q1_Q0, vdup_n_s16(-1));
            let lt_m1 = vclt_s16(q1_Q0, vdup_n_s16(-1));

            q1_s16 = vshl_n_s16::<10>(q1_Q0);
            let mut ts = vand_s16(
                vreinterpret_s16_u16(vcge_s16(q1_Q0, vdup_n_s16(0))),
                vdup_n_s16((offset_Q10 - QUANT_LEVEL_ADJUST_Q10) as i16),
            );
            let mut t1s = vadd_s16(q1_s16, ts);
            ts = vbsl_s16(
                lt_m1,
                vdup_n_s16((offset_Q10 + QUANT_LEVEL_ADJUST_Q10) as i16),
                vdup_n_s16(0),
            );
            q1_s16 = vadd_s16(q1_s16, ts);
            q1_s16 = vbsl_s16(lt_m1, q1_s16, t1s);
            q1_s16 = vbsl_s16(eq0, vdup_n_s16(offset_Q10 as i16), q1_s16);
            q1_s16 = vbsl_s16(
                eq_m1,
                vdup_n_s16((offset_Q10 - (1024 - QUANT_LEVEL_ADJUST_Q10)) as i16),
                q1_s16,
            );

            let mut q2_s16 = vadd_s16(q1_s16, vdup_n_s16(1024));
            q2_s16 = vbsl_s16(
                eq0,
                vdup_n_s16((offset_Q10 + 1024 - QUANT_LEVEL_ADJUST_Q10) as i16),
                q2_s16,
            );
            q2_s16 = vbsl_s16(eq_m1, vdup_n_s16(offset_Q10 as i16), q2_s16);

            // Rate terms
            t1s = q1_s16;
            let mut t2s = q2_s16;
            t1s = vbsl_s16(vorr_u16(eq_m1, lt_m1), vneg_s16(t1s), t1s);
            t2s = vbsl_s16(lt_m1, vneg_s16(t2s), t2s);

            let mut rd1 = vmull_s16(t1s, vdup_n_s16(lambda_Q10 as i16));
            let mut rd2 = vmull_s16(t2s, vdup_n_s16(lambda_Q10 as i16));

            let rr1 = vsub_s16(r_s16, q1_s16);
            rd1 = vmlal_s16(rd1, rr1, rr1);
            rd1 = vshrq_n_s32::<10>(rd1);

            let rr2 = vsub_s16(r_s16, q2_s16);
            rd2 = vmlal_s16(rd2, rr2, rr2);
            rd2 = vshrq_n_s32::<10>(rd2);

            let rd_v = vld1q_s32(dd.RD_Q10.as_ptr());
            let best_rd = vaddq_s32(rd_v, vminq_s32(rd1, rd2));
            let second_rd = vaddq_s32(rd_v, vmaxq_s32(rd1, rd2));
            vst1q_s32(ss[0].RD_Q10.as_mut_ptr(), best_rd);
            vst1q_s32(ss[1].RD_Q10.as_mut_ptr(), second_rd);

            let cmp = vcltq_s32(rd1, rd2);
            let q1w = vmovl_s16(q1_s16);
            let q2w = vmovl_s16(q2_s16);
            q_best = vbslq_s32(cmp, q1w, q2w);
            q_second = vbslq_s32(cmp, q2w, q1w);
            vst1q_s32(ss[0].Q_Q10.as_mut_ptr(), q_best);
            vst1q_s32(ss[1].Q_Q10.as_mut_ptr(), q_second);
        }

        // Update states for best (sample 0)
        {
            let exc = vsubq_s32(veorq_s32(vshlq_n_s32::<4>(q_best), sign_v), sign_v);
            let lpc_exc = vaddq_s32(exc, vdupq_n_s32(ltp_pred));
            let xq_v = vaddq_s32(lpc_exc, lpc_v);
            let diff = vsubq_s32(xq_v, vshlq_n_s32::<4>(vdupq_n_s32(x_Q10[i])));
            vst1q_s32(ss[0].Diff_Q14.as_mut_ptr(), diff);
            let slf = vsubq_s32(diff, n_ar_v);
            vst1q_s32(ss[0].sLTP_shp_Q14.as_mut_ptr(), vsubq_s32(slf, n_lf_v));
            vst1q_s32(ss[0].LF_AR_Q14.as_mut_ptr(), slf);
            vst1q_s32(ss[0].LPC_exc_Q14.as_mut_ptr(), lpc_exc);
            vst1q_s32(ss[0].xq_Q14.as_mut_ptr(), xq_v);
        }
        // Update states for second-best (sample 1)
        {
            let exc = vsubq_s32(veorq_s32(vshlq_n_s32::<4>(q_second), sign_v), sign_v);
            let lpc_exc = vaddq_s32(exc, vdupq_n_s32(ltp_pred));
            let xq_v = vaddq_s32(lpc_exc, lpc_v);
            let diff = vsubq_s32(xq_v, vshlq_n_s32::<4>(vdupq_n_s32(x_Q10[i])));
            vst1q_s32(ss[1].Diff_Q14.as_mut_ptr(), diff);
            let slf = vsubq_s32(diff, n_ar_v);
            vst1q_s32(ss[1].sLTP_shp_Q14.as_mut_ptr(), vsubq_s32(slf, n_lf_v));
            vst1q_s32(ss[1].LF_AR_Q14.as_mut_ptr(), slf);
            vst1q_s32(ss[1].LPC_exc_Q14.as_mut_ptr(), lpc_exc);
            vst1q_s32(ss[1].xq_Q14.as_mut_ptr(), xq_v);
        }

        *smpl_buf_idx = if *smpl_buf_idx != 0 {
            *smpl_buf_idx - 1
        } else {
            DECISION_DELAY - 1
        };
        let mut last_idx = (*smpl_buf_idx + decision_delay + DECISION_DELAY) as usize;
        if last_idx >= DECISION_DELAY as usize {
            last_idx -= DECISION_DELAY as usize;
        }
        if last_idx >= DECISION_DELAY as usize {
            last_idx -= DECISION_DELAY as usize;
        }

        // Find winner
        let mut rd_min = ss[0].RD_Q10[0];
        let mut winner = 0usize;
        for k in 1..n_states as usize {
            if ss[0].RD_Q10[k] < rd_min {
                rd_min = ss[0].RD_Q10[k];
                winner = k;
            }
        }

        // Clear unused states
        if (n_states as usize) < NEON_MAX_DEL_DEC_STATES {
            for k in n_states as usize..NEON_MAX_DEL_DEC_STATES {
                ss[0].RD_Q10[k] = 0;
                ss[1].RD_Q10[k] = 0;
            }
        }

        // Increase RD of expired states
        let winner_rand = dd.RandState[last_idx][winner];
        {
            let t = vmvnq_u32(vceqq_s32(
                vld1q_s32(dd.RandState[last_idx].as_ptr()),
                vdupq_n_s32(winner_rand),
            ));
            let penalty = vshrq_n_u32::<5>(t);
            let mut s0r = vld1q_s32(ss[0].RD_Q10.as_ptr());
            let mut s1r = vld1q_s32(ss[1].RD_Q10.as_ptr());
            s0r = vaddq_s32(s0r, vreinterpretq_s32_u32(penalty));
            s1r = vaddq_s32(s1r, vreinterpretq_s32_u32(penalty));
            vst1q_s32(ss[0].RD_Q10.as_mut_ptr(), s0r);
            vst1q_s32(ss[1].RD_Q10.as_mut_ptr(), s1r);

            let mut rd_max = ss[0].RD_Q10[0];
            let mut rd_min2 = ss[1].RD_Q10[0];
            let mut max_ind = 0usize;
            let mut min_ind = 0usize;
            for k in 1..n_states as usize {
                if ss[0].RD_Q10[k] > rd_max {
                    rd_max = ss[0].RD_Q10[k];
                    max_ind = k;
                }
                if ss[1].RD_Q10[k] < rd_min2 {
                    rd_min2 = ss[1].RD_Q10[k];
                    min_ind = k;
                }
            }

            if rd_min2 < rd_max {
                for jj in (i + 1)..(i + NSQ_LPC_BUF_LENGTH) {
                    dd.sLPC_Q14[jj][max_ind] = dd.sLPC_Q14[jj][min_ind];
                }
                for slot in 0..DECISION_DELAY as usize {
                    dd.RandState[slot][max_ind] = dd.RandState[slot][min_ind];
                    dd.Q_Q10[slot][max_ind] = dd.Q_Q10[slot][min_ind];
                    dd.Xq_Q14[slot][max_ind] = dd.Xq_Q14[slot][min_ind];
                    dd.Pred_Q15[slot][max_ind] = dd.Pred_Q15[slot][min_ind];
                    dd.Shape_Q14[slot][max_ind] = dd.Shape_Q14[slot][min_ind];
                }
                for slot in 0..MAX_SHAPE_LPC_ORDER as usize {
                    dd.sAR2_Q14[slot][max_ind] = dd.sAR2_Q14[slot][min_ind];
                }
                dd.LF_AR_Q14[max_ind] = dd.LF_AR_Q14[min_ind];
                dd.Diff_Q14[max_ind] = dd.Diff_Q14[min_ind];
                dd.Seed[max_ind] = dd.Seed[min_ind];
                dd.SeedInit[max_ind] = dd.SeedInit[min_ind];
                dd.RD_Q10[max_ind] = dd.RD_Q10[min_ind];

                ss[0].Q_Q10[max_ind] = ss[1].Q_Q10[min_ind];
                ss[0].RD_Q10[max_ind] = ss[1].RD_Q10[min_ind];
                ss[0].xq_Q14[max_ind] = ss[1].xq_Q14[min_ind];
                ss[0].LF_AR_Q14[max_ind] = ss[1].LF_AR_Q14[min_ind];
                ss[0].Diff_Q14[max_ind] = ss[1].Diff_Q14[min_ind];
                ss[0].sLTP_shp_Q14[max_ind] = ss[1].sLTP_shp_Q14[min_ind];
                ss[0].LPC_exc_Q14[max_ind] = ss[1].LPC_exc_Q14[min_ind];
            }
        }

        // Write delayed samples
        if subfr > 0 || i as i32 >= decision_delay {
            let oidx = (p_off as isize + i as isize - decision_delay as isize) as usize;
            pulses[oidx] = (((dd.Q_Q10[last_idx][winner] >> 9) + 1) >> 1) as i8;
            let xq_val =
                (dd.Xq_Q14[last_idx][winner] as i64 * delayed_gain_Q10[last_idx] as i64) >> 16;
            let xidx = (xq_off as isize + i as isize - decision_delay as isize) as usize;
            nsq.xq[xidx] = neon_rshift_round_sat16(xq_val as i32, 8);
            nsq.sLTP_shp_Q14[(nsq.sLTP_shp_buf_idx - decision_delay) as usize] =
                dd.Shape_Q14[last_idx][winner];
            sLTP_Q15[(nsq.sLTP_buf_idx - decision_delay) as usize] = dd.Pred_Q15[last_idx][winner];
        }
        nsq.sLTP_shp_buf_idx += 1;
        nsq.sLTP_buf_idx += 1;

        // Update all states with best candidate
        vst1q_s32(
            dd.LF_AR_Q14.as_mut_ptr(),
            vld1q_s32(ss[0].LF_AR_Q14.as_ptr()),
        );
        vst1q_s32(dd.Diff_Q14.as_mut_ptr(), vld1q_s32(ss[0].Diff_Q14.as_ptr()));
        vst1q_s32(
            dd.sLPC_Q14[NSQ_LPC_BUF_LENGTH + i].as_mut_ptr(),
            vld1q_s32(ss[0].xq_Q14.as_ptr()),
        );
        vst1q_s32(
            dd.Xq_Q14[*smpl_buf_idx as usize].as_mut_ptr(),
            vld1q_s32(ss[0].xq_Q14.as_ptr()),
        );
        let q_v = vld1q_s32(ss[0].Q_Q10.as_ptr());
        vst1q_s32(dd.Q_Q10[*smpl_buf_idx as usize].as_mut_ptr(), q_v);
        vst1q_s32(
            dd.Pred_Q15[*smpl_buf_idx as usize].as_mut_ptr(),
            vshlq_n_s32::<1>(vld1q_s32(ss[0].LPC_exc_Q14.as_ptr())),
        );
        vst1q_s32(
            dd.Shape_Q14[*smpl_buf_idx as usize].as_mut_ptr(),
            vld1q_s32(ss[0].sLTP_shp_Q14.as_ptr()),
        );

        let q_rounded = vrshrq_n_s32::<10>(q_v);
        let seed_upd = vreinterpretq_s32_u32(vaddq_u32(
            vreinterpretq_u32_s32(vld1q_s32(dd.Seed.as_ptr())),
            vreinterpretq_u32_s32(q_rounded),
        ));
        vst1q_s32(dd.Seed.as_mut_ptr(), seed_upd);
        vst1q_s32(dd.RandState[*smpl_buf_idx as usize].as_mut_ptr(), seed_upd);
        vst1q_s32(dd.RD_Q10.as_mut_ptr(), vld1q_s32(ss[0].RD_Q10.as_ptr()));
        delayed_gain_Q10[*smpl_buf_idx as usize] = gain_Q10;
    }

    // Copy LPC state for next subframe
    std::ptr::copy(
        dd.sLPC_Q14[length as usize..].as_ptr(),
        dd.sLPC_Q14.as_mut_ptr(),
        NSQ_LPC_BUF_LENGTH,
    );
}

/// NEON scale_states for del_dec quantizer.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn neon_scale_states(
    cfg: &NsqConfig,
    nsq: &mut silk_nsq_state,
    dd: &mut NeonDelDecStates,
    x16: &[i16],
    x_sc_Q10: &mut [i32],
    sLTP: &[i16],
    sLTP_Q15: &mut [i32],
    subfr: i32,
    ltp_scale_Q14: i32,
    gains_Q16: &[i32],
    pitchL: &[i32],
    signal_type: i32,
    decision_delay: i32,
) {
    let lag = pitchL[subfr as usize];
    let mut inv_gain_Q31 = silk_INVERSE32_varQ(gains_Q16[subfr as usize].max(1), 47);

    let inv_gain_Q26 = ((inv_gain_Q31 >> 4) + 1) >> 1;
    neon_smulww_loop(
        x16.as_ptr(),
        inv_gain_Q26,
        x_sc_Q10.as_mut_ptr(),
        cfg.subfr_length as i32,
    );

    if nsq.rewhite_flag != 0 {
        if subfr == 0 {
            inv_gain_Q31 = ((silk_SMULWB(inv_gain_Q31, ltp_scale_Q14) as u32) << 2) as i32;
        }
        let start = (nsq.sLTP_buf_idx - lag - LTP_ORDER as i32 / 2) as usize;
        let count = lag + LTP_ORDER as i32 / 2;
        neon_smulww_loop(
            sLTP.as_ptr().add(start),
            inv_gain_Q31,
            sLTP_Q15.as_mut_ptr().add(start),
            count,
        );
    }

    if gains_Q16[subfr as usize] != nsq.prev_gain_Q16 {
        let gain_adj = silk_DIV32_varQ(nsq.prev_gain_Q16, gains_Q16[subfr as usize], 16);

        if (-65536..65536).contains(&gain_adj) {
            let gv = vdup_n_s32((gain_adj as u32 as i32) << 15);

            let shp_start = (nsq.sLTP_shp_buf_idx - cfg.ltp_mem_length as i32) as usize;
            let shp_end = nsq.sLTP_shp_buf_idx as usize;
            let mut ii = shp_start;
            while ii + 7 < shp_end {
                neon_smulww_small_8(nsq.sLTP_shp_Q14.as_mut_ptr().add(ii), gv);
                ii += 8;
            }
            while ii < shp_end {
                nsq.sLTP_shp_Q14[ii] = silk_SMULWW(gain_adj, nsq.sLTP_shp_Q14[ii]);
                ii += 1;
            }

            if signal_type == TYPE_VOICED && nsq.rewhite_flag == 0 {
                let ps = (nsq.sLTP_buf_idx - lag - LTP_ORDER as i32 / 2) as usize;
                let pe = (nsq.sLTP_buf_idx - decision_delay) as usize;
                let mut ii = ps;
                while ii + 7 < pe {
                    neon_smulww_small_8(sLTP_Q15.as_mut_ptr().add(ii), gv);
                    ii += 8;
                }
                while ii < pe {
                    sLTP_Q15[ii] = silk_SMULWW(gain_adj, sLTP_Q15[ii]);
                    ii += 1;
                }
            }

            neon_smulww_small_4(dd.LF_AR_Q14.as_mut_ptr(), gv);
            neon_smulww_small_4(dd.Diff_Q14.as_mut_ptr(), gv);
            for ii in 0..NSQ_LPC_BUF_LENGTH {
                neon_smulww_small_4(dd.sLPC_Q14[ii].as_mut_ptr(), gv);
            }
            for ii in 0..MAX_SHAPE_LPC_ORDER as usize {
                neon_smulww_small_4(dd.sAR2_Q14[ii].as_mut_ptr(), gv);
            }
            for ii in 0..DECISION_DELAY as usize {
                neon_smulww_small_4(dd.Pred_Q15[ii].as_mut_ptr(), gv);
                neon_smulww_small_4(dd.Shape_Q14[ii].as_mut_ptr(), gv);
            }
        } else {
            let mut gv = vdup_n_s32(((gain_adj & 0x0000FFFF) as u32 as i32) << 15);
            gv = vset_lane_s32::<1>(gain_adj >> 16, gv);

            let shp_start = (nsq.sLTP_shp_buf_idx - cfg.ltp_mem_length as i32) as usize;
            let shp_end = nsq.sLTP_shp_buf_idx as usize;
            let mut ii = shp_start;
            while ii + 7 < shp_end {
                neon_smulww_8(nsq.sLTP_shp_Q14.as_mut_ptr().add(ii), gv);
                ii += 8;
            }
            while ii < shp_end {
                nsq.sLTP_shp_Q14[ii] = silk_SMULWW(gain_adj, nsq.sLTP_shp_Q14[ii]);
                ii += 1;
            }

            if signal_type == TYPE_VOICED && nsq.rewhite_flag == 0 {
                let ps = (nsq.sLTP_buf_idx - lag - LTP_ORDER as i32 / 2) as usize;
                let pe = (nsq.sLTP_buf_idx - decision_delay) as usize;
                let mut ii = ps;
                while ii + 7 < pe {
                    neon_smulww_8(sLTP_Q15.as_mut_ptr().add(ii), gv);
                    ii += 8;
                }
                while ii < pe {
                    sLTP_Q15[ii] = silk_SMULWW(gain_adj, sLTP_Q15[ii]);
                    ii += 1;
                }
            }

            neon_smulww_4(dd.LF_AR_Q14.as_mut_ptr(), gv);
            neon_smulww_4(dd.Diff_Q14.as_mut_ptr(), gv);
            for ii in 0..NSQ_LPC_BUF_LENGTH {
                neon_smulww_4(dd.sLPC_Q14[ii].as_mut_ptr(), gv);
            }
            for ii in 0..MAX_SHAPE_LPC_ORDER as usize {
                neon_smulww_4(dd.sAR2_Q14[ii].as_mut_ptr(), gv);
            }
            for ii in 0..DECISION_DELAY as usize {
                neon_smulww_4(dd.Pred_Q15[ii].as_mut_ptr(), gv);
                neon_smulww_4(dd.Shape_Q14[ii].as_mut_ptr(), gv);
            }
        }

        nsq.prev_gain_Q16 = gains_Q16[subfr as usize];
    }
}

/// NEON NSQ del_dec: complete outer function replacing `silk_NSQ_del_dec_c`
/// when 2 < nStatesDelayedDecision <= 4 on aarch64.
///
/// # Safety
/// Requires aarch64 NEON (always available on aarch64).
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn silk_NSQ_del_dec_neon(
    psEncC: &NsqConfig,
    nsq: &mut silk_nsq_state,
    psIndices: &mut SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    PredCoef_Q12: &[i16],
    LTPCoef_Q14: &[i16],
    AR_Q13: &[i16],
    HarmShapeGain_Q14: &[i32],
    Tilt_Q14: &[i32],
    LF_shp_Q14: &[i32],
    Gains_Q16: &[i32],
    pitchL: &[i32],
    Lambda_Q10: i32,
    LTP_scale_Q14: i32,
) {
    let ltp_mem_len = psEncC.ltp_mem_length;
    let frame_len = psEncC.frame_length;
    let subfr_len = psEncC.subfr_length;
    let n_states = psEncC.nStatesDelayedDecision;

    let mut lag = nsq.lagPrev;
    debug_assert!(nsq.prev_gain_Q16 != 0);

    let mut dd = NeonDelDecStates::new_zeroed();

    for k in 0..n_states as usize {
        dd.Seed[k] = (k as i32 + psIndices.Seed as i32) & 3;
        dd.SeedInit[k] = dd.Seed[k];
    }
    vst1q_s32(
        dd.LF_AR_Q14.as_mut_ptr(),
        vld1q_dup_s32(&nsq.sLF_AR_shp_Q14),
    );
    vst1q_s32(dd.Diff_Q14.as_mut_ptr(), vld1q_dup_s32(&nsq.sDiff_shp_Q14));
    vst1q_s32(
        dd.Shape_Q14[0].as_mut_ptr(),
        vld1q_dup_s32(&nsq.sLTP_shp_Q14[ltp_mem_len - 1]),
    );
    for ii in 0..NSQ_LPC_BUF_LENGTH {
        vst1q_s32(
            dd.sLPC_Q14[ii].as_mut_ptr(),
            vld1q_dup_s32(&nsq.sLPC_Q14[ii]),
        );
    }
    for ii in 0..nsq.sAR2_Q14.len() {
        vst1q_s32(
            dd.sAR2_Q14[ii].as_mut_ptr(),
            vld1q_dup_s32(&nsq.sAR2_Q14[ii]),
        );
    }

    let offset_Q10 = silk_Quantization_Offsets_Q10[(psIndices.signalType as i32 >> 1) as usize]
        [psIndices.quantOffsetType as usize] as i32;
    let mut smpl_buf_idx = 0i32;

    let mut ddly = silk_min_int(DECISION_DELAY, subfr_len as i32);
    if psIndices.signalType as i32 == TYPE_VOICED {
        let mut pitch_min = pitchL[0];
        for p in pitchL.iter().take(psEncC.nb_subfr).skip(1) {
            pitch_min = silk_min_int(pitch_min, *p);
        }
        ddly = silk_min_int(ddly, pitch_min - LTP_ORDER as i32 / 2 - 1);
    } else if lag > 0 {
        ddly = silk_min_int(ddly, lag - LTP_ORDER as i32 / 2 - 1);
    }

    let lsf_interp: i32 = if psIndices.NLSFInterpCoef_Q2 as i32 == 4 {
        0
    } else {
        1
    };

    let mut sLTP_Q15 = vec![0i32; ltp_mem_len + frame_len];
    let mut sLTP = vec![0i16; ltp_mem_len + frame_len];
    let mut x_sc_Q10 = vec![0i32; subfr_len];
    let mut delayed_gain_Q10 = [0i32; DECISION_DELAY as usize];

    let mut pxq_off = ltp_mem_len;
    nsq.sLTP_shp_buf_idx = ltp_mem_len as i32;
    nsq.sLTP_buf_idx = ltp_mem_len as i32;
    let mut subfr_ctr = 0i32;
    let mut x16_off = 0usize;
    let mut pulses_off = 0usize;

    for k in 0..psEncC.nb_subfr as i32 {
        let a_off = (((k >> 1) | (1 - lsf_interp)) * MAX_LPC_ORDER as i32) as usize;
        let a_Q12 = &PredCoef_Q12[a_off..a_off + psEncC.predictLPCOrder as usize];
        let b_off = (k * LTP_ORDER as i32) as usize;
        let b_Q14 = &LTPCoef_Q14[b_off..b_off + LTP_ORDER];
        let ar_off = (k * MAX_SHAPE_LPC_ORDER) as usize;
        let ar_Q13 = &AR_Q13[ar_off..ar_off + psEncC.shapingLPCOrder as usize];

        debug_assert!(HarmShapeGain_Q14[k as usize] >= 0);
        let mut hsp = HarmShapeGain_Q14[k as usize] >> 2;
        hsp |= (((HarmShapeGain_Q14[k as usize] >> 1) as u32) << 16) as i32;

        nsq.rewhite_flag = 0;
        if psIndices.signalType as i32 == TYPE_VOICED {
            lag = pitchL[k as usize];
            if k & (3 - ((lsf_interp as u32) << 1) as i32) == 0 {
                if k == 2 {
                    let mut rd_min = dd.RD_Q10[0];
                    let mut winner = 0usize;
                    for ii in 1..n_states as usize {
                        if dd.RD_Q10[ii] < rd_min {
                            rd_min = dd.RD_Q10[ii];
                            winner = ii;
                        }
                    }
                    dd.RD_Q10[winner] -= silk_int32_MAX >> 4;
                    let mut rv = vld1q_s32(dd.RD_Q10.as_ptr());
                    rv = vaddq_s32(rv, vdupq_n_s32(silk_int32_MAX >> 4));
                    vst1q_s32(dd.RD_Q10.as_mut_ptr(), rv);

                    neon_copy_winner_state(
                        &dd,
                        ddly,
                        smpl_buf_idx,
                        winner,
                        Gains_Q16[1],
                        14,
                        pulses,
                        pulses_off,
                        pxq_off,
                        nsq,
                    );
                    subfr_ctr = 0;
                }

                let start_idx =
                    ltp_mem_len as i32 - lag - psEncC.predictLPCOrder - LTP_ORDER as i32 / 2;
                debug_assert!(start_idx > 0);
                silk_LPC_analysis_filter(
                    &mut sLTP[start_idx as usize..ltp_mem_len],
                    &nsq.xq[(start_idx + k * subfr_len as i32) as usize..]
                        [..ltp_mem_len - start_idx as usize],
                    a_Q12,
                );
                nsq.sLTP_buf_idx = ltp_mem_len as i32;
                nsq.rewhite_flag = 1;
            }
        }

        neon_scale_states(
            psEncC,
            nsq,
            &mut dd,
            &x16[x16_off..x16_off + subfr_len],
            &mut x_sc_Q10,
            &sLTP,
            &mut sLTP_Q15,
            k,
            LTP_scale_Q14,
            Gains_Q16,
            pitchL,
            psIndices.signalType as i32,
            ddly,
        );

        let fresh_subfr = subfr_ctr;
        subfr_ctr += 1;

        neon_noise_shape_quantizer_del_dec(
            nsq,
            &mut dd,
            psIndices.signalType as i32,
            &x_sc_Q10,
            pulses,
            pulses_off,
            pxq_off,
            &mut sLTP_Q15,
            &mut delayed_gain_Q10,
            a_Q12,
            b_Q14,
            ar_Q13,
            lag,
            hsp,
            Tilt_Q14[k as usize],
            LF_shp_Q14[k as usize],
            Gains_Q16[k as usize],
            Lambda_Q10,
            offset_Q10,
            subfr_len as i32,
            fresh_subfr,
            psEncC.shapingLPCOrder,
            psEncC.predictLPCOrder,
            psEncC.warping_Q16,
            n_states,
            &mut smpl_buf_idx,
            ddly,
        );

        x16_off += subfr_len;
        pulses_off += subfr_len;
        pxq_off += subfr_len;
    }

    // Find final winner
    let mut rd_min = dd.RD_Q10[0];
    let mut winner = 0usize;
    for k in 1..n_states as usize {
        if dd.RD_Q10[k] < rd_min {
            rd_min = dd.RD_Q10[k];
            winner = k;
        }
    }

    psIndices.Seed = dd.SeedInit[winner] as i8;
    let gain_Q10 = Gains_Q16[psEncC.nb_subfr - 1] >> 6;
    neon_copy_winner_state(
        &dd,
        ddly,
        smpl_buf_idx,
        winner,
        gain_Q10,
        8,
        pulses,
        pulses_off,
        pxq_off,
        nsq,
    );

    // Copy winner's LPC state
    let mut t_v = vdupq_n_s32(0);
    let mut ii = 0usize;
    while ii + 3 < NSQ_LPC_BUF_LENGTH {
        t_v = vld1q_lane_s32::<0>(&dd.sLPC_Q14[ii][winner], t_v);
        t_v = vld1q_lane_s32::<1>(&dd.sLPC_Q14[ii + 1][winner], t_v);
        t_v = vld1q_lane_s32::<2>(&dd.sLPC_Q14[ii + 2][winner], t_v);
        t_v = vld1q_lane_s32::<3>(&dd.sLPC_Q14[ii + 3][winner], t_v);
        vst1q_s32(nsq.sLPC_Q14.as_mut_ptr().add(ii), t_v);
        ii += 4;
    }
    while ii < NSQ_LPC_BUF_LENGTH {
        nsq.sLPC_Q14[ii] = dd.sLPC_Q14[ii][winner];
        ii += 1;
    }

    // Copy winner's AR2 state
    let sar2_len = nsq.sAR2_Q14.len();
    ii = 0;
    while ii + 3 < sar2_len {
        t_v = vld1q_lane_s32::<0>(&dd.sAR2_Q14[ii][winner], t_v);
        t_v = vld1q_lane_s32::<1>(&dd.sAR2_Q14[ii + 1][winner], t_v);
        t_v = vld1q_lane_s32::<2>(&dd.sAR2_Q14[ii + 2][winner], t_v);
        t_v = vld1q_lane_s32::<3>(&dd.sAR2_Q14[ii + 3][winner], t_v);
        vst1q_s32(nsq.sAR2_Q14.as_mut_ptr().add(ii), t_v);
        ii += 4;
    }
    while ii < sar2_len {
        nsq.sAR2_Q14[ii] = dd.sAR2_Q14[ii][winner];
        ii += 1;
    }

    nsq.sLF_AR_shp_Q14 = dd.LF_AR_Q14[winner];
    nsq.sDiff_shp_Q14 = dd.Diff_Q14[winner];
    nsq.lagPrev = pitchL[psEncC.nb_subfr - 1];

    nsq.xq.copy_within(frame_len..frame_len + ltp_mem_len, 0);
    nsq.sLTP_shp_Q14
        .copy_within(frame_len..frame_len + ltp_mem_len, 0);
}
