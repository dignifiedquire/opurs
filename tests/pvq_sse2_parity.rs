#![cfg(all(
    feature = "tools",
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]

use opurs::celt::simd::x86::op_pvq_search_sse2 as rust_op_pvq_search_sse2;

unsafe extern "C" {
    fn op_pvq_search_sse2(X: *mut f32, iy: *mut i32, K: i32, N: i32, arch: i32) -> f32;
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        let v = self.next_u32() as f32 / (u32::MAX as f32);
        2.0 * v - 1.0
    }
}

#[test]
fn pvq_sse2_matches_c() {
    let mut rng = Rng::new(0x1234_9876_dead_beef);

    for n in 4..=64 {
        for _ in 0..250 {
            let k = 1 + (rng.next_u32() % 30) as i32;
            let mut x_c = vec![0.0f32; n];
            for v in &mut x_c {
                *v = rng.next_f32();
            }
            let mut x_r = x_c.clone();
            let mut iy_c = vec![0i32; n + 4];
            let mut iy_r = vec![0i32; n + 4];

            let yy_c =
                unsafe { op_pvq_search_sse2(x_c.as_mut_ptr(), iy_c.as_mut_ptr(), k, n as i32, 2) };
            let yy_r = unsafe { rust_op_pvq_search_sse2(&mut x_r, &mut iy_r, k, n as i32) };

            if yy_c.to_bits() != yy_r.to_bits() || iy_c[..n] != iy_r[..n] {
                panic!(
                    "mismatch n={n} k={k} yy_c={:#010x} yy_r={:#010x} first_iy_diff={:?}",
                    yy_c.to_bits(),
                    yy_r.to_bits(),
                    iy_c[..n]
                        .iter()
                        .zip(iy_r[..n].iter())
                        .position(|(a, b)| a != b)
                );
            }
        }
    }
}
