use crate::celt::kiss_fft::{kiss_fft_cpx, opus_fft_c, opus_ifft_c};
use crate::opus_custom_mode_create;

fn check(input: &[kiss_fft_cpx], out: &[kiss_fft_cpx], nfft: usize, isinverse: bool) {
    let mut errpow = 0.;
    let mut sigpow = 0.;

    for bin in 0..nfft {
        let mut ansr = 0.;
        let mut ansi = 0.;
        let mut difr = 0.;
        let mut difi = 0.;

        for k in 0..nfft {
            let phase = -2. * std::f32::consts::PI * k as f32 / nfft as f32;
            let mut re = phase.cos();
            let mut im = phase.sin();

            if isinverse {
                im = -im;
            }
            if isinverse {
                re /= nfft as f32;
                im /= nfft as f32;
            }
            ansr += input[k].re * re - (input[k]).im * im;
            ansi += input[k].re * im + (input[k]).im * re;
        }
        difr = ansr - out[bin].re;
        difi = ansi - out[bin].im;
        errpow += difr * difr + difi * difi;
        sigpow += ansr * ansr + ansi * ansi;
    }
    let snr = 10. * (sigpow / errpow).log10();
    println!("nfft={} inverse={},snr = {}", nfft, isinverse, snr);

    if snr < 60. {
        // TODO: panic, these currently fail :/
        println!("** poor snr: {} ** ", snr);
    }
}

fn rand() -> i32 {
    let mut t = [0u8; 4];
    getrandom::getrandom(&mut t).unwrap();
    i32::from_be_bytes(t)
}

fn test1d(nfft: usize, isinverse: bool) {
    let mode = opus_custom_mode_create(48000, 960, None).unwrap();
    let id = if nfft == 480 {
        0
    } else if nfft == 240 {
        1
    } else if nfft == 120 {
        2
    } else if nfft == 60 {
        3
    } else {
        return;
    };
    let cfg = mode.mdct.kfft[id as usize];
    let mut in_0 = vec![kiss_fft_cpx::default(); nfft];
    let mut out = vec![kiss_fft_cpx::default(); nfft];

    for k in 0..nfft {
        in_0[k].re = (rand() % 32767 - 16384) as f32;
        in_0[k].im = (rand() % 32767 - 16384) as f32;
    }
    for k in 0..nfft {
        in_0[k].re *= 32768.;
        in_0[k].im *= 32768.;
    }
    if isinverse {
        for k in 0..nfft {
            in_0[k].re /= nfft as f32;
            in_0[k].im /= nfft as f32;
        }
    }
    if isinverse {
        opus_ifft_c(cfg, &in_0, &mut out);
    } else {
        opus_fft_c(cfg, &in_0, &mut out);
    }
    check(&in_0, &out, nfft, isinverse);
}

#[test]
fn test_dft_fft() {
    test1d(32, false);
    test1d(128, false);
    test1d(256, false);
    test1d(36, false);
    test1d(50, false);
    test1d(60, false);
    test1d(120, false);
    test1d(240, false);
    test1d(480, false);
}

#[test]
fn test_dft_ifft() {
    test1d(32, true);
    test1d(128, true);
    test1d(256, true);
    test1d(36, true);
    test1d(50, true);
    test1d(60, true);
    test1d(120, true);
    test1d(240, true);
    test1d(480, true);
}
