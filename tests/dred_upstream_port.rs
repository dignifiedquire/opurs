//! Upstream DRED random parse/process port.
//!
//! Upstream C: `tests/test_opus_dred.c:test_random_dred`

#![cfg(feature = "tools-dnn")]

mod test_common;

use opurs::{
    opus_dred_alloc, opus_dred_decoder_create, opus_dred_parse, opus_dred_process, OPUS_OK,
};
use test_common::TestRng;

const DEFAULT_ITERS: usize = 20_000;
const MAX_EXTENSION_SIZE: usize = 200;

fn iterations_from_env() -> usize {
    std::env::var("OPURS_DRED_RANDOM_ITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(DEFAULT_ITERS)
}

/// Upstream C: tests/test_opus_dred.c:test_random_dred
#[test]
fn dred_random_parse_process() {
    let mut rng = TestRng::from_iseed(0x1357_9BDF);
    let iters = iterations_from_env();

    let mut dred_dec = opus_dred_decoder_create().expect("opus_dred_decoder_create failed");
    let mut dred = opus_dred_alloc();

    for _ in 0..iters {
        let len = (rng.next_u32() as usize) % (MAX_EXTENSION_SIZE + 1);
        let mut payload = vec![0u8; len];
        for b in &mut payload {
            *b = (rng.next_u32() & 0xFF) as u8;
        }

        let mut dred_end = 0i32;
        let defer = (rng.next_u32() & 1) != 0;
        let res1 = opus_dred_parse(
            &mut dred_dec,
            &mut dred,
            &payload,
            48_000,
            48_000,
            Some(&mut dred_end),
            defer,
        );

        if res1 > 0 {
            let src = dred.clone();
            let res2 = opus_dred_process(&dred_dec, &src, &mut dred);
            assert_eq!(res2, OPUS_OK, "process should succeed if parse succeeds");
            assert!(res1 >= dred_end, "end before beginning");
        }
    }
}
