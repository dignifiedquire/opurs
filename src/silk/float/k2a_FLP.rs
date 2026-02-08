pub fn silk_k2a_FLP(A: &mut [f32], rc: &[f32], order: i32) {
    let mut k: i32 = 0;
    let mut n: i32 = 0;
    let mut rck: f32 = 0.;
    let mut tmp1: f32 = 0.;
    let mut tmp2: f32 = 0.;
    k = 0;
    while k < order {
        rck = rc[k as usize];
        n = 0;
        while n < k + 1 >> 1 {
            tmp1 = A[n as usize];
            tmp2 = A[(k - n - 1) as usize];
            A[n as usize] = tmp1 + tmp2 * rck;
            A[(k - n - 1) as usize] = tmp2 + tmp1 * rck;
            n += 1;
        }
        A[k as usize] = -rck;
        k += 1;
    }
}
