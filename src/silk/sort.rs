//! Sorting utilities.
//!
//! Upstream C: `silk/sort.c`

/// Upstream C: silk/sort.c:silk_insertion_sort_increasing
pub fn silk_insertion_sort_increasing(a: &mut [i32], idx: &mut [i32], L: i32, K: i32) {
    let mut value: i32 = 0;
    let mut i: i32 = 0;
    let mut j: i32 = 0;
    debug_assert!(K > 0);
    debug_assert!(L > 0);
    debug_assert!(L >= K);
    i = 0;
    while i < K {
        unsafe { *idx.get_unchecked_mut(i as usize) = i; }
        i += 1;
    }
    i = 1;
    while i < K {
        value = unsafe { *a.get_unchecked(i as usize) };
        j = i - 1;
        while j >= 0 && value < unsafe { *a.get_unchecked(j as usize) } {
            unsafe {
                *a.get_unchecked_mut((j + 1) as usize) = *a.get_unchecked(j as usize);
                *idx.get_unchecked_mut((j + 1) as usize) = *idx.get_unchecked(j as usize);
            }
            j -= 1;
        }
        unsafe {
            *a.get_unchecked_mut((j + 1) as usize) = value;
            *idx.get_unchecked_mut((j + 1) as usize) = i;
        }
        i += 1;
    }
    i = K;
    while i < L {
        value = unsafe { *a.get_unchecked(i as usize) };
        if value < unsafe { *a.get_unchecked((K - 1) as usize) } {
            j = K - 2;
            while j >= 0 && value < unsafe { *a.get_unchecked(j as usize) } {
                unsafe {
                    *a.get_unchecked_mut((j + 1) as usize) = *a.get_unchecked(j as usize);
                    *idx.get_unchecked_mut((j + 1) as usize) = *idx.get_unchecked(j as usize);
                }
                j -= 1;
            }
            unsafe {
                *a.get_unchecked_mut((j + 1) as usize) = value;
                *idx.get_unchecked_mut((j + 1) as usize) = i;
            }
        }
        i += 1;
    }
}

pub fn silk_insertion_sort_increasing_all_values_int16(a: &mut [i16]) {
    for i in 1..a.len() {
        let mut j = i;
        while j > 0 && a[j] < a[j - 1] {
            a.swap(j, j - 1);
            j -= 1;
        }
    }
}
