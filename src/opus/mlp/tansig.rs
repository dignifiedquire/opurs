//! Activation function approximations.
//!
//! Upstream C: `src/mlp.c`

/// Rational approximation of tanh(x), matching upstream v1.6.1.
///
/// Upstream C: `src/mlp.c:tansig_approx`
#[inline]
#[allow(clippy::excessive_precision)]
pub fn tansig_approx(x: f32) -> f32 {
    const N0: f32 = 952.528_015_14;
    const N1: f32 = 96.392_356_87;
    const N2: f32 = 0.608_630_4;
    const D0: f32 = 952.723_999_02;
    const D1: f32 = 413.368_011_47;
    const D2: f32 = 11.886_009_22;

    let x2 = x * x;
    // fmadd(a, b, c) = a*b + c
    let num = (N2 * x2 + N1) * x2 + N0;
    let den = (D2 * x2 + D1) * x2 + D0;
    let num = num * x / den;
    (-1.0f32).max(1.0f32.min(num))
}

/// Upstream C: `src/mlp.c:sigmoid_approx`
#[inline]
pub fn sigmoid_approx(x: f32) -> f32 {
    0.5 + 0.5 * tansig_approx(0.5 * x)
}
