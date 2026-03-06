//!  nlsf stabilizer:
//!
//!  - Moves NLSFs further apart if they are too close
//!  - Moves NLSFs away from borders if they are too close
//!  - High effort to achieve a modification with minimum
//!    Euclidean distance to input vector
//!  - Output are sorted nlsf coefficients
//!

use crate::silk::sigproc_fix::{silk_max_int, silk_min_int};
use crate::silk::sort::silk_insertion_sort_increasing_all_values_int16;
use crate::silk::typedefs::{SILK_INT16_MAX, SILK_INT16_MIN};

pub const MAX_LOOPS: i32 = 20;

///
/// nlsf stabilizer, for a single input data vector
/// Upstream c: silk/NLSF_stabilize.c:silk_NLSF_stabilize
pub fn silk_nlsf_stabilize(nlsf_q15: &mut [i16], ndelta_min_q15: &[i16]) {
    let mut i: usize;
    let mut i_min: usize;
    let mut k: usize;
    let mut loops: i32;
    let mut center_freq_q15: i16;
    let mut diff_q15: i32;
    let mut min_diff_q15: i32;
    let mut min_center_q15: i32;
    let mut max_center_q15: i32;

    let l = nlsf_q15.len();

    /* This is necessary to ensure an output within range of a opus_int16 */
    debug_assert!(ndelta_min_q15[l] >= 1);

    loops = 0;
    while loops < MAX_LOOPS {
        /**************************/
        /* Find smallest distance */
        /**************************/
        /* First element */
        min_diff_q15 = nlsf_q15[0] as i32 - ndelta_min_q15[0] as i32;
        i_min = 0;
        /* Middle elements */
        i = 1;
        while i < l {
            diff_q15 = nlsf_q15[i] as i32 - (nlsf_q15[i - 1] as i32 + ndelta_min_q15[i] as i32);
            if diff_q15 < min_diff_q15 {
                min_diff_q15 = diff_q15;
                i_min = i;
            }
            i += 1;
        }
        /* Last element */
        diff_q15 = ((1) << 15) - (nlsf_q15[l - 1] as i32 + ndelta_min_q15[l] as i32);
        if diff_q15 < min_diff_q15 {
            min_diff_q15 = diff_q15;
            i_min = l;
        }

        /***************************************************/
        /* Now check if the smallest distance non-negative */
        /***************************************************/
        if min_diff_q15 >= 0 {
            return;
        }
        if i_min == 0 {
            /* Move away from lower limit */
            nlsf_q15[0] = ndelta_min_q15[0];
        } else if i_min == l {
            /* Move away from higher limit */
            nlsf_q15[l - 1] = (((1) << 15) - ndelta_min_q15[l] as i32) as i16;
        } else {
            /* Find the lower extreme for the location of the current center frequency */
            min_center_q15 = 0;
            k = 0;
            while k < i_min {
                min_center_q15 += ndelta_min_q15[k] as i32;
                k += 1;
            }
            min_center_q15 += ndelta_min_q15[i_min] as i32 >> 1;

            /* Find the upper extreme for the location of the current center frequency */
            max_center_q15 = (1) << 15;
            k = l;
            while k > i_min {
                max_center_q15 -= ndelta_min_q15[k] as i32;
                k -= 1;
            }
            max_center_q15 -= ndelta_min_q15[i_min] as i32 >> 1;

            /* Move apart, sorted by value, keeping the same center frequency */
            let avg_q15 = ((nlsf_q15[i_min - 1] as i32 + nlsf_q15[i_min] as i32) >> 1)
                + ((nlsf_q15[i_min - 1] as i32 + nlsf_q15[i_min] as i32) & 1);
            center_freq_q15 = (if min_center_q15 > max_center_q15 {
                avg_q15.clamp(max_center_q15, min_center_q15)
            } else {
                avg_q15.clamp(min_center_q15, max_center_q15)
            }) as i16;
            nlsf_q15[i_min - 1] =
                (center_freq_q15 as i32 - (ndelta_min_q15[i_min] as i32 >> 1)) as i16;
            nlsf_q15[i_min] = (nlsf_q15[i_min - 1] as i32 + ndelta_min_q15[i_min] as i32) as i16;
        }
        loops += 1;
    }

    /* Safe and simple fall back method, which is less ideal than the above */
    if loops == MAX_LOOPS {
        /* Insertion sort (fast for already almost sorted arrays):   */
        /* Best case:  O(n)   for an already sorted array            */
        /* Worst case: O(n^2) for an inversely sorted array          */
        silk_insertion_sort_increasing_all_values_int16(nlsf_q15);

        /* First nlsf should be no less than NDeltaMin[0] */
        nlsf_q15[0] = silk_max_int(nlsf_q15[0] as i32, ndelta_min_q15[0_usize] as i32) as i16;

        /* Keep delta_min distance between the NLSFs */
        i = 1;
        while i < l {
            nlsf_q15[i] = silk_max_int(
                nlsf_q15[i] as i32,
                (nlsf_q15[i - 1] as i32 + ndelta_min_q15[i] as i32)
                    .clamp(SILK_INT16_MIN, SILK_INT16_MAX),
            ) as i16;
            i += 1;
        }

        /* Last nlsf should be no higher than 1 - NDeltaMin[L] */
        nlsf_q15[l - 1] = silk_min_int(
            nlsf_q15[l - 1] as i32,
            ((1) << 15) - ndelta_min_q15[l] as i32,
        ) as i16;

        /* Keep NDeltaMin distance between the NLSFs */

        for i in (0..=l - 2).rev() {
            nlsf_q15[i] = silk_min_int(
                nlsf_q15[i] as i32,
                nlsf_q15[i + 1] as i32 - ndelta_min_q15[i + 1] as i32,
            ) as i16;
        }
    }
}
