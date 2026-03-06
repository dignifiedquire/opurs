//! Floating-point correlation matrix computation.
//!
//! Upstream c: `silk/float/corrMatrix_FLP.c`

use crate::silk::float::inner_product_flp::silk_inner_product2_flp;
use crate::util::nalgebra::MatrixViewRMut;
use nalgebra::{Dim, DimAdd, DimDiff, DimSub, DimSum, VectorView, U1};

// Correlation matrix computations for LS estimate.

///
/// Calculates correlation vector x'*t
///
/// ```text
/// x       _i    x vector [L+Order-1] used to create x
/// t       _i    Target vector [L]
/// L       _i    Length of vectors
/// Order   _i    Max lag for correlation
/// xt      O    x'*t correlation vector [Order]
/// ```
/// Upstream c: silk/float/corrMatrix_FLP.c:silk_corr_vector_flp
pub fn silk_corr_vector_flp<Len, OrderDim>(
    x: &VectorView<f32, DimDiff<DimSum<Len, OrderDim>, U1>>,
    t: &VectorView<f32, Len>,
    // accept a row vector because it's more convenient
    xt: &mut MatrixViewRMut<f32, U1, OrderDim>,
) where
    Len: Dim,
    OrderDim: Dim,
    Len: DimAdd<OrderDim>,
    <Len as DimAdd<OrderDim>>::Output: DimSub<U1>,
{
    let (x_len, _) = x.shape_generic();
    let (len, _) = t.shape_generic();
    let (_, order_dim) = xt.shape_generic();
    assert_eq!(x_len.value(), len.add(order_dim).sub(U1).value());

    for lag in 0..order_dim.value() {
        let ptr1 = x.generic_view::<Len, U1>((order_dim.value() - 1 - lag, 0), (len, U1));
        xt[lag] = silk_inner_product2_flp(&ptr1, t) as f32;
    }
}

///
/// Calculates correlation matrix x'*x
///
/// ```text
/// x       _i   x vector [ L+Order-1 ] used to create x
/// L       _i   Length of vectors
/// Order   _i   Max lag for correlation
/// xx      O   x'*x correlation matrix [Order x Order]
/// ```
/// Upstream c: silk/float/corrMatrix_FLP.c:silk_corr_matrix_flp
pub fn silk_corr_matrix_flp<Dx, Len, OrderDim>(
    x: &VectorView<f32, Dx>,
    len: Len,
    xx: &mut MatrixViewRMut<f32, OrderDim, OrderDim, OrderDim, U1>,
) where
    Dx: Dim,
    Len: Dim,
    OrderDim: Dim,
{
    let (order_dim, _) = xx.shape_generic();
    assert_eq!(x.shape().0, len.value() + order_dim.value() - 1);

    let window_at = |lag: usize| x.generic_view((order_dim.value() - 1 - lag, 0), (len, U1));
    let hvalue_at = |lag: usize| x[order_dim.value() - 1 - lag];
    let tvalue_at = |lag: usize| x[order_dim.value() + len.value() - 1 - lag];

    let order = order_dim.value();

    // calculate the diagonal by using a sliding window
    for lag in 0..order {
        // use a sliding window
        let mut energy = silk_inner_product2_flp(&window_at(0), &window_at(lag));
        xx[(lag, 0)] = energy as f32;
        xx[(0, lag)] = energy as f32;

        for j in 1..(order - lag) {
            energy +=
                // yes, this is how it's done in the c impl: the sliding window diff is calculated as an f32
                (hvalue_at(j) * hvalue_at(lag + j) - tvalue_at(j) * tvalue_at(lag + j)) as f64;
            xx[(lag + j, j)] = energy as f32;
            xx[(j, lag + j)] = energy as f32;
        }
    }
}
