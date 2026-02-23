#![cfg(feature = "tools")]

use opurs::celt::pitch::{
    pitch_downsample as rust_pitch_downsample, pitch_search as rust_pitch_search,
    remove_doubling as rust_remove_doubling,
};

unsafe extern "C" {
    fn pitch_downsample(x: *mut *mut f32, x_lp: *mut f32, len: i32, c: i32, factor: i32, arch: i32);
    fn pitch_search(
        x_lp: *const f32,
        y: *mut f32,
        len: i32,
        max_pitch: i32,
        pitch: *mut i32,
        arch: i32,
    );
    fn remove_doubling(
        x: *mut f32,
        maxperiod: i32,
        minperiod: i32,
        n: i32,
        t0: *mut i32,
        prev_period: i32,
        prev_gain: f32,
        arch: i32,
    ) -> f32;
    fn comb_filter(
        y: *mut f32,
        x: *mut f32,
        t0: i32,
        t1: i32,
        n: i32,
        g0: f32,
        g1: f32,
        tapset0: i32,
        tapset1: i32,
        window: *const f32,
        overlap: i32,
        arch: i32,
    );
}

#[inline]
fn test_arch_level() -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return 4;
        }
        if std::is_x86_feature_detected!("sse4.1") {
            return 3;
        }
        if std::is_x86_feature_detected!("sse2") {
            return 2;
        }
        if std::is_x86_feature_detected!("sse") {
            return 1;
        }
    }
    0
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
fn pitch_primitives_match_c_scalar() {
    let mut rng = Rng::new(0xdecafbad_u64);
    let c_arch = test_arch_level();

    for _ in 0..200 {
        let c = if (rng.next_u32() & 1) == 0 {
            1usize
        } else {
            2usize
        };
        let len = 2048usize;
        let half = len / 2;
        let maxperiod = 1024i32;
        let minperiod = 15i32;
        let n = 960i32;
        let lag = (n + maxperiod) as usize / 2;
        let max_pitch = maxperiod - 3 * minperiod;

        let mut ch0 = vec![0.0f32; len];
        let mut ch1 = vec![0.0f32; len];
        for i in 0..len {
            ch0[i] = rng.next_f32();
            ch1[i] = rng.next_f32();
        }

        let mut x_lp_r = vec![0.0f32; half];
        if c == 2 {
            rust_pitch_downsample(&[&ch0, &ch1], &mut x_lp_r, len);
        } else {
            rust_pitch_downsample(&[&ch0], &mut x_lp_r, len);
        }

        let mut x_lp_c = vec![0.0f32; half];
        unsafe {
            let mut ptrs = [ch0.as_mut_ptr(), ch1.as_mut_ptr()];
            pitch_downsample(
                ptrs.as_mut_ptr(),
                x_lp_c.as_mut_ptr(),
                (len >> 1) as i32,
                c as i32,
                2,
                c_arch,
            );
        }

        for i in 0..half {
            assert!(
                x_lp_r[i].to_bits() == x_lp_c[i].to_bits(),
                "pitch_downsample mismatch at {i}: rust={} (0x{:08x}) c={} (0x{:08x})",
                x_lp_r[i],
                x_lp_r[i].to_bits(),
                x_lp_c[i],
                x_lp_c[i].to_bits()
            );
        }

        let mut y_r = x_lp_r.clone();
        let mut y_c = x_lp_c.clone();
        let mut pitch_c = 0i32;
        let pitch_r = rust_pitch_search(
            &x_lp_r[(maxperiod as usize / 2)..(maxperiod as usize / 2 + (n / 2) as usize)],
            &y_r[..lag],
            n,
            max_pitch,
        );
        unsafe {
            pitch_search(
                x_lp_c[(maxperiod as usize / 2)..].as_ptr(),
                y_c.as_mut_ptr(),
                n,
                max_pitch,
                &mut pitch_c,
                c_arch,
            );
        }
        assert_eq!(
            pitch_r, pitch_c,
            "pitch_search mismatch rust={pitch_r} c={pitch_c}"
        );

        let mut t0_r = maxperiod - pitch_r;
        let mut t0_c = t0_r;
        let prev_period = 120 + (rng.next_u32() % 200) as i32;
        let prev_gain = (rng.next_u32() as f32 / u32::MAX as f32).min(1.0);

        let g_r = rust_remove_doubling(
            &x_lp_r[..lag],
            maxperiod,
            minperiod,
            n,
            &mut t0_r,
            prev_period,
            prev_gain,
        );
        let g_c = unsafe {
            remove_doubling(
                y_c.as_mut_ptr(),
                maxperiod,
                minperiod,
                n,
                &mut t0_c,
                prev_period,
                prev_gain,
                c_arch,
            )
        };

        assert_eq!(
            g_r.to_bits(),
            g_c.to_bits(),
            "remove_doubling gain mismatch rust={} (0x{:08x}) c={} (0x{:08x})",
            g_r,
            g_r.to_bits(),
            g_c,
            g_c.to_bits()
        );
        assert_eq!(
            t0_r, t0_c,
            "remove_doubling T0 mismatch rust={t0_r} c={t0_c}"
        );
    }
}

#[test]
fn comb_filter_matches_c_scalar() {
    use opurs::celt::common::comb_filter as rust_comb_filter;

    let mut rng = Rng::new(0x1234_5678_9abc_def0);
    let c_arch = test_arch_level();
    for _ in 0..500 {
        let n = 60 + (rng.next_u32() % 120) as i32;
        let overlap = (rng.next_u32() % 60) as i32;
        let t0 = 15 + (rng.next_u32() % 120) as i32;
        let t1 = 15 + (rng.next_u32() % 120) as i32;
        let tapset0 = (rng.next_u32() % 3) as i32;
        let tapset1 = (rng.next_u32() % 3) as i32;
        let g0 = 0.95 * rng.next_f32();
        let g1 = 0.95 * rng.next_f32();

        // Provide enough lookback for x[-T-2] reads.
        let lookback = (t0.max(t1) + 4) as usize;
        let x_len = lookback + n as usize + 8;
        let x_start = lookback;
        let y_start = 0usize;

        let mut x = vec![0.0f32; x_len];
        for v in &mut x {
            *v = rng.next_f32();
        }
        let mut y_r = vec![0.0f32; n as usize];
        let mut y_c = vec![0.0f32; n as usize];
        let mut window = vec![0.0f32; overlap as usize];
        for w in &mut window {
            *w = (rng.next_u32() as f32 / u32::MAX as f32).clamp(0.0, 1.0);
        }

        rust_comb_filter(
            &mut y_r, y_start, &x, x_start, t0, t1, n, g0, g1, tapset0, tapset1, &window, overlap,
            0,
        );

        unsafe {
            comb_filter(
                y_c.as_mut_ptr().add(y_start),
                x.as_mut_ptr().add(x_start),
                t0,
                t1,
                n,
                g0,
                g1,
                tapset0,
                tapset1,
                if overlap > 0 {
                    window.as_ptr()
                } else {
                    std::ptr::null()
                },
                overlap,
                c_arch,
            );
        }

        for i in 0..n as usize {
            assert!(
                y_r[i].to_bits() == y_c[i].to_bits(),
                "comb_filter mismatch at {i}: rust={} (0x{:08x}) c={} (0x{:08x}) params n={n} overlap={overlap} t0={t0} t1={t1} g0={g0} g1={g1} tap0={tapset0} tap1={tapset1}",
                y_r[i],
                y_r[i].to_bits(),
                y_c[i],
                y_c[i].to_bits()
            );
        }
    }
}
