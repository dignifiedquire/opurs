//! Floating-point LTP coefficient search.
//!
//! Upstream c: `silk/float/find_LTP_FLP.c`

use crate::silk::float::corr_matrix_flp::{silk_corr_matrix_flp, silk_corr_vector_flp};
use crate::silk::float::energy_flp::silk_energy_flp;
use crate::silk::tuning_parameters::LTP_CORR_INV_MAX;
use crate::util::nalgebra::MatrixViewRMut;
use nalgebra::{Const, Dim, DimMul, DimProd, Dyn, VectorView};

const LTP_ORDER: usize = crate::silk::define::LTP_ORDER;
type LtpOrder = Const<{ LTP_ORDER }>;

///
/// LTP analysis
///
/// ```text
/// xx[ MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER ]  /* O    Weight for LTP quantization
/// x_x[ MAX_NB_SUBFR * LTP_ORDER ]              /* O    Weight for LTP quantization
/// r_ptr[]                                     /* I    LPC residual
/// lag[ MAX_NB_SUBFR ]                         /* I    LTP lags
/// subfr_length                                /* I    Subframe length
/// nb_subfr                                    /* I    number of subframes
/// ```
/// Upstream c: silk/float/find_LTP_FLP.c:silk_find_ltp_flp
pub fn silk_find_ltp_flp<NbSubfr>(
    xx: &mut MatrixViewRMut<f32, DimProd<NbSubfr, LtpOrder>, LtpOrder>,
    x_x: &mut MatrixViewRMut<f32, NbSubfr, LtpOrder>,
    r: &[f32],
    mut r_ptr: usize,
    lag: &VectorView<i32, NbSubfr>,
    subfr_length: usize,
) where
    NbSubfr: Dim,
    NbSubfr: DimMul<LtpOrder>,
{
    let (nb_subfr_x_order, _) = xx.shape_generic();
    let (nb_subfr, _) = x_x.shape_generic();

    assert_eq!(nb_subfr_x_order.value(), nb_subfr.value() * LTP_ORDER);
    assert_eq!(lag.shape().0, nb_subfr.value());

    for k in 0..nb_subfr.value() {
        let r_frame = VectorView::<f32, Dyn>::from_slice(&r[r_ptr..], subfr_length);
        let lag_frame = VectorView::<f32, Dyn>::from_slice(
            &r[r_ptr - lag[k] as usize - LTP_ORDER / 2..],
            subfr_length + LTP_ORDER - 1,
        );

        let mut xx_ptr = xx.fixed_view_mut::<{ LTP_ORDER }, { LTP_ORDER }>(k * LTP_ORDER, 0);
        let mut x_x_ptr = x_x.fixed_view_mut::<1, { LTP_ORDER }>(k, 0);

        silk_corr_matrix_flp(&lag_frame, Dyn(subfr_length), &mut xx_ptr);
        silk_corr_vector_flp(&lag_frame, &r_frame, &mut x_x_ptr);

        let xx = silk_energy_flp(&r[r_ptr..][..subfr_length + LTP_ORDER]) as f32;
        let temp = 1.0 / xx.max(LTP_CORR_INV_MAX * 0.5 * (xx_ptr[(0, 0)] + xx_ptr[(4, 4)]) + 1.0);
        xx_ptr *= temp;
        x_x_ptr *= temp;

        r_ptr += subfr_length;
    }
}
