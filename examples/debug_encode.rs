//! Temporary debug tool for comparing Rust vs C decode/encode outputs
use opurs::tools::demo::{
    opus_demo_decode, opus_demo_encode, Application, Channels, DecodeArgs, DnnOptions, EncodeArgs,
    EncoderOptions, OpusBackend, SampleRate,
};

fn main() {
    let mode = std::env::var("DEBUG_MODE").unwrap_or_else(|_| "decode".to_string());

    match mode.as_str() {
        "decode" => run_decode_compare(),
        "encode" => run_encode_compare(),
        _ => eprintln!("Unknown DEBUG_MODE: {}", mode),
    }
}

fn run_decode_compare() {
    let tv = std::env::var("TV").unwrap_or_else(|_| "testvector01".to_string());
    let sr = std::env::var("SR").unwrap_or_else(|_| "48000".to_string());
    let ch = std::env::var("CH").unwrap_or_else(|_| "stereo".to_string());

    let encoded = std::fs::read(format!("opus_newvectors/{}.bit", tv)).unwrap();

    let sample_rate = match sr.as_str() {
        "48000" => SampleRate::R48000,
        "24000" => SampleRate::R24000,
        "16000" => SampleRate::R16000,
        "12000" => SampleRate::R12000,
        "8000" => SampleRate::R8000,
        _ => panic!("Unknown sample rate: {}", sr),
    };
    let channels = match ch.as_str() {
        "mono" => Channels::Mono,
        "stereo" => Channels::Stereo,
        _ => panic!("Unknown channels: {}", ch),
    };

    let decode_args = DecodeArgs {
        sample_rate,
        channels,
        options: Default::default(),
        complexity: None,
    };
    let no_dnn = DnnOptions::default();

    eprintln!("=== Decoding {} at {} {} ===", tv, sr, ch);

    let rust_decoded = opus_demo_decode(OpusBackend::Rust, &encoded, decode_args, &no_dnn);
    let c_decoded = opus_demo_decode(OpusBackend::Upstream, &encoded, decode_args, &no_dnn);

    eprintln!(
        "Rust decoded: {} bytes, C decoded: {} bytes",
        rust_decoded.len(),
        c_decoded.len()
    );

    if rust_decoded == c_decoded {
        eprintln!("MATCH: decoded outputs are identical");
        return;
    }

    // Compare as i16 samples
    let rust_samples: Vec<i16> = rust_decoded
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let c_samples: Vec<i16> = c_decoded
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    let n = rust_samples.len().min(c_samples.len());
    let mut diff_count = 0;
    let mut first_diff_sample = None;
    let mut max_diff: i32 = 0;
    for i in 0..n {
        let diff = (rust_samples[i] as i32 - c_samples[i] as i32).abs();
        if diff != 0 {
            diff_count += 1;
            max_diff = max_diff.max(diff);
            if first_diff_sample.is_none() {
                first_diff_sample = Some(i);
            }
            if diff_count <= 20 {
                let ch_count = if matches!(channels, Channels::Stereo) {
                    2
                } else {
                    1
                };
                let frame_idx = i / (960 * ch_count);
                let within = i % (960 * ch_count);
                eprintln!(
                    "  Sample {:6} (frame {:3} pos {:4}): Rust={:6} C={:6} diff={:4}",
                    i, frame_idx, within, rust_samples[i], c_samples[i], diff
                );
            }
        }
    }
    eprintln!(
        "DIFF: {}/{} samples differ, max_diff={}, first_diff_at={}",
        diff_count,
        n,
        max_diff,
        first_diff_sample.unwrap_or(0)
    );
}

fn run_encode_compare() {
    let tv = std::env::var("TV").unwrap_or_else(|_| "testvector01".to_string());
    let br: u32 = std::env::var("BR")
        .unwrap_or_else(|_| "60000".to_string())
        .parse()
        .unwrap();

    let data = std::fs::read(format!("opus_newvectors/{}.dec", tv)).unwrap();

    let encode_args = EncodeArgs {
        application: Application::Audio,
        sample_rate: SampleRate::R48000,
        channels: Channels::Stereo,
        bitrate: br,
        options: EncoderOptions::default(),
    };
    let no_dnn = DnnOptions::default();

    eprintln!("=== Encoding {} at {}bps ===", tv, br);
    let (rust_encoded, _) = opus_demo_encode(OpusBackend::Rust, &data, encode_args, &no_dnn);
    let (c_encoded, _) = opus_demo_encode(OpusBackend::Upstream, &data, encode_args, &no_dnn);

    // Parse packets
    let rust_pkts = parse_packets(&rust_encoded);
    let c_pkts = parse_packets(&c_encoded);

    let n = rust_pkts.len().min(c_pkts.len());
    let mut diff_count = 0;
    for i in 0..n {
        let (r_len, r_range, ref r_data) = rust_pkts[i];
        let (c_len, c_range, ref c_data) = c_pkts[i];
        if r_data != c_data {
            diff_count += 1;
            let r_toc = r_data[0];
            let r_config = (r_toc >> 3) & 0x1f;
            let mode_str = if r_config <= 11 {
                "SILK"
            } else if r_config <= 15 {
                "HYBRID"
            } else {
                "CELT"
            };
            let min_len = r_data.len().min(c_data.len());
            let first_diff = (0..min_len)
                .find(|&j| r_data[j] != c_data[j])
                .unwrap_or(min_len);
            eprintln!(
                "Pkt {:3}: len=R{}/C{} range=R{:08x}/C{:08x} mode={} first_diff_byte={}",
                i, r_len, c_len, r_range, c_range, mode_str, first_diff
            );
        }
    }
    eprintln!(
        "{} packets differ out of {} total",
        diff_count,
        n.max(rust_pkts.len().max(c_pkts.len()))
    );
}

fn parse_packets(encoded: &[u8]) -> Vec<(i32, u32, Vec<u8>)> {
    let mut packets = Vec::new();
    let mut pos = 0;
    while pos + 8 <= encoded.len() {
        let pkt_len = i32::from_be_bytes(encoded[pos..pos + 4].try_into().unwrap());
        let final_range = u32::from_be_bytes(encoded[pos + 4..pos + 8].try_into().unwrap());
        if pkt_len <= 0 || pos + 8 + pkt_len as usize > encoded.len() {
            break;
        }
        let pkt_data = encoded[pos + 8..pos + 8 + pkt_len as usize].to_vec();
        packets.push((pkt_len, final_range, pkt_data));
        pos += 8 + pkt_len as usize;
    }
    packets
}
