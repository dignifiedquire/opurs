use crate::silk::bwexpander_32::silk_bwexpander_32;
use crate::silk::SigProc_FIX::silk_min_32;

pub mod typedef_h {
    pub const silk_int16_MAX: i32 = i16::MAX as i32;
}
pub use self::typedef_h::silk_int16_MAX;
use crate::silk::define::LSF_COS_TAB_SZ_FIX;
use crate::silk::table_LSF_cos::silk_LSFCosTab_FIX_Q12;

pub const BIN_DIV_STEPS_A2NLSF_FIX: i32 = 3;
pub const MAX_ITERATIONS_A2NLSF_FIX: i32 = 16;
#[inline]
fn silk_A2NLSF_trans_poly(p: &mut [i32], dd: i32) {
    let mut k: i32 = 0;
    let mut n: i32 = 0;
    k = 2;
    while k <= dd {
        n = dd;
        while n > k {
            p[(n - 2) as usize] -= p[n as usize];
            n -= 1;
        }
        p[(k - 2) as usize] -= ((p[k as usize] as u32) << 1) as i32;
        k += 1;
    }
}
#[inline]
fn silk_A2NLSF_eval_poly(p: &[i32], x: i32, dd: i32) -> i32 {
    let mut n: i32 = 0;
    let mut x_Q16: i32 = 0;
    let mut y32: i32 = 0;
    y32 = p[dd as usize];
    x_Q16 = ((x as u32) << 4) as i32;
    if (8 == dd) as i32 as i64 != 0 {
        y32 = (p[7] as i64 + (y32 as i64 * x_Q16 as i64 >> 16)) as i32;
        y32 = (p[6] as i64 + (y32 as i64 * x_Q16 as i64 >> 16)) as i32;
        y32 = (p[5] as i64 + (y32 as i64 * x_Q16 as i64 >> 16)) as i32;
        y32 = (p[4] as i64 + (y32 as i64 * x_Q16 as i64 >> 16)) as i32;
        y32 = (p[3] as i64 + (y32 as i64 * x_Q16 as i64 >> 16)) as i32;
        y32 = (p[2] as i64 + (y32 as i64 * x_Q16 as i64 >> 16)) as i32;
        y32 = (p[1] as i64 + (y32 as i64 * x_Q16 as i64 >> 16)) as i32;
        y32 = (p[0] as i64 + (y32 as i64 * x_Q16 as i64 >> 16)) as i32;
    } else {
        n = dd - 1;
        while n >= 0 {
            y32 = (p[n as usize] as i64 + (y32 as i64 * x_Q16 as i64 >> 16)) as i32;
            n -= 1;
        }
    }
    return y32;
}
#[inline]
fn silk_A2NLSF_init(a_Q16: &[i32], P: &mut [i32], Q: &mut [i32], dd: i32) {
    let mut k: i32 = 0;
    P[dd as usize] = ((1) << 16) as i32;
    Q[dd as usize] = ((1) << 16) as i32;
    k = 0;
    while k < dd {
        P[k as usize] = -a_Q16[(dd - k - 1) as usize] - a_Q16[(dd + k) as usize];
        Q[k as usize] = -a_Q16[(dd - k - 1) as usize] + a_Q16[(dd + k) as usize];
        k += 1;
    }
    k = dd;
    while k > 0 {
        P[(k - 1) as usize] -= P[k as usize];
        Q[(k - 1) as usize] += Q[k as usize];
        k -= 1;
    }
    silk_A2NLSF_trans_poly(P, dd);
    silk_A2NLSF_trans_poly(Q, dd);
}
/// Upstream C: silk/A2NLSF.c:silk_A2NLSF
pub fn silk_A2NLSF(NLSF: &mut [i16], a_Q16: &mut [i32], d: i32) {
    let mut i: i32 = 0;
    let mut k: i32 = 0;
    let mut m: i32 = 0;
    let mut dd: i32 = 0;
    let mut root_ix: i32 = 0;
    let mut ffrac: i32 = 0;
    let mut xlo: i32 = 0;
    let mut xhi: i32 = 0;
    let mut xmid: i32 = 0;
    let mut ylo: i32 = 0;
    let mut yhi: i32 = 0;
    let mut ymid: i32 = 0;
    let mut thr: i32 = 0;
    let mut nom: i32 = 0;
    let mut den: i32 = 0;
    let mut P: [i32; 13] = [0; 13];
    let mut Q: [i32; 13] = [0; 13];
    // use_Q: false = use P, true = use Q
    let mut use_Q: bool;
    dd = d >> 1;
    silk_A2NLSF_init(a_Q16, &mut P, &mut Q, dd);
    use_Q = false;
    xlo = silk_LSFCosTab_FIX_Q12[0 as usize] as i32;
    ylo = silk_A2NLSF_eval_poly(&P, xlo, dd);
    if ylo < 0 {
        NLSF[0] = 0;
        use_Q = true;
        ylo = silk_A2NLSF_eval_poly(&Q, xlo, dd);
        root_ix = 1;
    } else {
        root_ix = 0;
    }
    k = 1;
    i = 0;
    thr = 0;
    loop {
        xhi = silk_LSFCosTab_FIX_Q12[k as usize] as i32;
        yhi = silk_A2NLSF_eval_poly(if use_Q { &Q } else { &P }, xhi, dd);
        if ylo <= 0 && yhi >= thr || ylo >= 0 && yhi <= -thr {
            if yhi == 0 {
                thr = 1;
            } else {
                thr = 0;
            }
            ffrac = -(256);
            m = 0;
            while m < BIN_DIV_STEPS_A2NLSF_FIX {
                xmid = (xlo + xhi >> 1) + (xlo + xhi & 1);
                ymid = silk_A2NLSF_eval_poly(if use_Q { &Q } else { &P }, xmid, dd);
                if ylo <= 0 && ymid >= 0 || ylo >= 0 && ymid <= 0 {
                    xhi = xmid;
                    yhi = ymid;
                } else {
                    xlo = xmid;
                    ylo = ymid;
                    ffrac = ffrac + (128 >> m);
                }
                m += 1;
            }
            if (if ylo > 0 { ylo } else { -ylo }) < 65536 {
                den = ylo - yhi;
                nom = ((ylo as u32) << 8 - 3) as i32 + (den >> 1);
                if den != 0 {
                    ffrac += nom / den;
                }
            } else {
                ffrac += ylo / (ylo - yhi >> 8 - 3);
            }
            NLSF[root_ix as usize] =
                silk_min_32(((k as u32) << 8) as i32 + ffrac, silk_int16_MAX) as i16;
            root_ix += 1;
            if root_ix >= d {
                break;
            }
            use_Q = (root_ix & 1) != 0;
            xlo = silk_LSFCosTab_FIX_Q12[(k - 1) as usize] as i32;
            ylo = (((1 - (root_ix & 2)) as u32) << 12) as i32;
        } else {
            k += 1;
            xlo = xhi;
            ylo = yhi;
            thr = 0;
            if k > LSF_COS_TAB_SZ_FIX {
                i += 1;
                if i > MAX_ITERATIONS_A2NLSF_FIX {
                    NLSF[0] = (((1) << 15) / (d + 1)) as i16;
                    k = 1;
                    while k < d {
                        NLSF[k as usize] = (NLSF[(k - 1) as usize] as i32 + NLSF[0] as i32) as i16;
                        k += 1;
                    }
                    return;
                }
                silk_bwexpander_32(&mut a_Q16[..d as usize], 65536 - ((1) << i) as i32);
                silk_A2NLSF_init(a_Q16, &mut P, &mut Q, dd);
                use_Q = false;
                xlo = silk_LSFCosTab_FIX_Q12[0 as usize] as i32;
                ylo = silk_A2NLSF_eval_poly(&P, xlo, dd);
                if ylo < 0 {
                    NLSF[0] = 0;
                    use_Q = true;
                    ylo = silk_A2NLSF_eval_poly(&Q, xlo, dd);
                    root_ix = 1;
                } else {
                    root_ix = 0;
                }
                k = 1;
            }
        }
    }
}
