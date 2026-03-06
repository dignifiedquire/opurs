//! Sorting utilities.
//!
//! Upstream c: `silk/sort.c`

/// Upstream c: silk/sort.c:silk_insertion_sort_increasing
pub fn silk_insertion_sort_increasing(a: &mut [i32], idx: &mut [i32], l: i32, k: i32) {
    let mut value: i32;
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
        while j >= 0 && value < a[j as usize] {
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
        if value < a[(k - 1) as usize] {
            j = k - 2;
            while j >= 0 && value < a[j as usize] {
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

pub fn silk_insertion_sort_increasing_all_values_int16(a: &mut [i16]) {
    for _i in 1..a.len() {
        let mut j = _i;
        while j > 0 && a[j] < a[j - 1] {
            a.swap(j, j - 1);
            j -= 1;
        }
    }
}
