//! Floating-point sorting utilities.
//!
//! Upstream c: `silk/float/sort_FLP.c`

/// Upstream c: silk/float/sort_FLP.c:silk_insertion_sort_decreasing_FLP
pub fn silk_insertion_sort_decreasing_flp(a: &mut [f32], idx: &mut [i32], l: i32, k: i32) {
    let mut value: f32;
    let mut _i: i32;
    let mut j: i32;
    debug_assert!(k > 0);
    debug_assert!(l > 0);
    debug_assert!(l >= k);
    _i = 0;
    while _i < k {
        idx[_i as usize] = _i;
        _i += 1;
    }
    _i = 1;
    while _i < k {
        value = a[_i as usize];
        j = _i - 1;
        while j >= 0 && value > a[j as usize] {
            a[(j + 1) as usize] = a[j as usize];
            idx[(j + 1) as usize] = idx[j as usize];
            j -= 1;
        }
        a[(j + 1) as usize] = value;
        idx[(j + 1) as usize] = _i;
        _i += 1;
    }
    _i = k;
    while _i < l {
        value = a[_i as usize];
        if value > a[(k - 1) as usize] {
            j = k - 2;
            while j >= 0 && value > a[j as usize] {
                a[(j + 1) as usize] = a[j as usize];
                idx[(j + 1) as usize] = idx[j as usize];
                j -= 1;
            }
            a[(j + 1) as usize] = value;
            idx[(j + 1) as usize] = _i;
        }
        _i += 1;
    }
}
