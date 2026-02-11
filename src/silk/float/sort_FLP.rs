//! Floating-point sorting utilities.
//!
//! Upstream C: `silk/float/sort_FLP.c`

/// Upstream C: silk/float/sort_FLP.c:silk_insertion_sort_decreasing_FLP
pub fn silk_insertion_sort_decreasing_FLP(a: &mut [f32], idx: &mut [i32], L: i32, K: i32) {
    let mut value: f32 = 0.;
    let mut i: i32 = 0;
    let mut j: i32 = 0;
    assert!(K > 0);
    assert!(L > 0);
    assert!(L >= K);
    i = 0;
    while i < K {
        idx[i as usize] = i;
        i += 1;
    }
    i = 1;
    while i < K {
        value = a[i as usize];
        j = i - 1;
        while j >= 0 && value > a[j as usize] {
            a[(j + 1) as usize] = a[j as usize];
            idx[(j + 1) as usize] = idx[j as usize];
            j -= 1;
        }
        a[(j + 1) as usize] = value;
        idx[(j + 1) as usize] = i;
        i += 1;
    }
    i = K;
    while i < L {
        value = a[i as usize];
        if value > a[(K - 1) as usize] {
            j = K - 2;
            while j >= 0 && value > a[j as usize] {
                a[(j + 1) as usize] = a[j as usize];
                idx[(j + 1) as usize] = idx[j as usize];
                j -= 1;
            }
            a[(j + 1) as usize] = value;
            idx[(j + 1) as usize] = i;
        }
        i += 1;
    }
}
