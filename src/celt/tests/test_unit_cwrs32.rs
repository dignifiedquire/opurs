use crate::celt::cwrs::{cwrsi, icwrs, pvq_v};
use crate::celt::rate::get_pulses;

const PN: [i32; 22] = [
    2, 3, 4, 6, 8, 9, 11, 12, 16, 18, 22, 24, 32, 36, 44, 48, 64, 72, 88, 96, 144, 176,
];
const PKMAX: [i32; 22] = [
    128, 128, 128, 88, 36, 26, 18, 16, 12, 11, 9, 9, 7, 7, 6, 6, 5, 5, 5, 5, 4, 4,
];

#[test]
fn test_cwrs32() {
    for t in 0..22 {
        let n = PN[t as usize];

        for pseudo in 1..41 {
            let mut k: i32 = 0;
            let mut inc: u32 = 0;
            let mut nc: u32 = 0;
            k = get_pulses(pseudo);
            if k > PKMAX[t as usize] {
                break;
            }
            println!("Testing CWRS with N={}, K={}...", n, k,);
            nc = pvq_v(n as _, k as _);
            inc = nc.wrapping_div(20000);
            if inc < 1 {
                inc = 1;
            }
            for i in (0..nc).step_by(inc as _) {
                let mut y: [i32; 240] = [0; 240];
                let mut sy: i32 = 0;
                let mut v: u32 = 0;
                let mut ii: u32 = 0;
                unsafe { cwrsi(n, k, i, y.as_mut_ptr()) };
                sy = 0;
                for j in 0..n {
                    sy += (y[j as usize]).abs();
                }
                if sy != k {
                    panic!("N={} Pulse count mismatch in cwrsi ({}!={}).", n, sy, k,);
                }
                ii = unsafe { icwrs(n, y.as_mut_ptr()) };
                v = pvq_v(n as _, k as _);
                if ii != i {
                    panic!("Combination-index mismatch ({}!={}).", ii as i64, i as i64,);
                }
                if v != nc {
                    panic!("Combination count mismatch ({}!={}).", v as i64, nc as i64,);
                }
            }
        }
    }
}
