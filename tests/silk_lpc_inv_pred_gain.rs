/// Tests for SILK LPC inverse prediction gain matching upstream
/// `silk/tests/test_unit_LPC_inv_pred_gain.c`.
///
/// Tests that silk_LPC_inverse_pred_gain_c() never reports a filter as stable
/// when it is definitely unstable (verified by impulse response simulation).
///
/// Upstream C: silk/tests/test_unit_LPC_inv_pred_gain.c
mod test_common;

use test_common::TestRng;
use unsafe_libopus::internals::{silk_LPC_inverse_pred_gain_c, SILK_MAX_ORDER_LPC};

/// Check filter stability via impulse response simulation.
///
/// Returns true if the filter appears stable, false if definitely unstable.
/// Some unstable filters may be classified as stable, but not the other way.
fn check_stability(a_q12: &[i16], order: usize) -> bool {
    let mut sum_a: i32 = 0;
    let mut sum_abs_a: i32 = 0;
    for j in 0..order {
        sum_a += a_q12[j] as i32;
        sum_abs_a += (a_q12[j] as i32).abs();
    }

    // Check DC stability
    if sum_a >= 4096 {
        return false;
    }

    // If sum of absolute values < 1 (in Q12), filter is definitely stable
    if sum_abs_a < 4096 {
        return true;
    }

    // Simulate impulse response
    let mut y = [0.0f64; SILK_MAX_ORDER_LPC];
    y[0] = 1.0;

    for i in 0..10_000 {
        let mut sum = 0.0;
        for j in 0..order {
            sum += y[j] * a_q12[j] as f64;
        }
        for j in (1..order).rev() {
            y[j] = y[j - 1];
        }
        y[0] = sum / 4096.0;

        // If impulse response reaches +/- 10000, definitely unstable
        if !(y[0] < 10000.0 && y[0] > -10000.0) {
            return false;
        }

        // Test every 8 samples for low amplitude
        if (i & 0x7) == 0 {
            let amp: f64 = y[..order].iter().map(|v| v.abs()).sum();
            if amp < 0.00001 {
                return true;
            }
        }
    }

    true
}

/// Run 10,000 iterations of random filter stability testing.
///
/// For each iteration, test all even orders 2..=SILK_MAX_ORDER_LPC with
/// 16 dynamic range shifts. Verify that when silk_LPC_inverse_pred_gain_c()
/// reports stable (gain != 0), the filter is not definitely unstable.
#[test]
fn test_lpc_inverse_pred_gain() {
    let mut rng = TestRng::new(0); // srand(0) in upstream

    for count in 0..10_000 {
        for order in (2..=SILK_MAX_ORDER_LPC).step_by(2) {
            for shift in 0..16u32 {
                let mut a_q12 = [0i16; SILK_MAX_ORDER_LPC];
                for i in 0..SILK_MAX_ORDER_LPC {
                    a_q12[i] = (rng.next_i32() as i16) >> shift;
                }

                let gain = silk_LPC_inverse_pred_gain_c(&a_q12[..order]);

                if gain != 0 && !check_stability(&a_q12, order) {
                    panic!(
                        "Loop {count} failed: gain={gain} but filter is unstable \
                         (order={order}, shift={shift}, A_Q12={:?})",
                        &a_q12[..order]
                    );
                }
            }
        }
    }
}
