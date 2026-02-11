#![forbid(unsafe_code)]

use std::fs::File;
use std::io::{Read, Seek, Write};

use clap::Parser;
use opurs::{opus_strerror, OpusRepacketizer};

#[derive(Parser)]
#[command(about = "Opus repacketizer demo")]
struct Args {
    /// Number of packets to merge (1-48)
    #[arg(long = "merge", default_value_t = 1)]
    merge: usize,

    /// Split packets into individual frames
    #[arg(long = "split")]
    split: bool,

    /// Input file
    input: std::path::PathBuf,

    /// Output file
    output: std::path::PathBuf,
}

fn is_eof(file: &mut File) -> bool {
    let position = file.stream_position().expect("seeking");
    let end = file.seek(std::io::SeekFrom::End(0)).expect("seeking");
    file.seek(std::io::SeekFrom::Start(position))
        .expect("seeking");
    position == end
}

fn main() {
    let args = Args::parse();

    assert!(
        (1..=48).contains(&args.merge),
        "-merge parameter must be between 1 and 48"
    );

    let mut packets: [[u8; 1500]; 48] = [[0; 1500]; 48];
    let mut len: [i32; 48] = [0; 48];
    let mut rng: [i32; 48] = [0; 48];
    let mut output_packet: [u8; 32000] = [0; 32000];

    let mut fin = File::open(&args.input).expect("opening input file");
    let mut fout = File::create(&args.output).expect("opening output file");

    let mut rp = OpusRepacketizer::default();
    let mut eof = false;
    while !eof {
        #[allow(unused_assignments)]
        let mut err: i32 = 0;
        let mut nb_packets: usize = args.merge;
        rp.init();
        let mut i: usize = 0;
        while i < nb_packets {
            let mut ch: [u8; 4] = [0; 4];
            fin.read_exact(&mut ch).unwrap();
            len[i] = i32::from_be_bytes(ch);
            if len[i] > 1500 || len[i] < 0 {
                if is_eof(&mut fin) {
                    eof = true;
                } else {
                    eprintln!("Invalid payload length");
                    std::process::exit(1);
                }
                break;
            } else {
                fin.read_exact(&mut ch).unwrap();
                rng[i] = i32::from_be_bytes(ch);
                fin.read_exact(&mut packets[i][..len[i] as usize]).unwrap();
                if is_eof(&mut fin) {
                    eof = true;
                    break;
                } else {
                    err = rp.cat(&packets[i][..len[i] as usize]);
                    if err != 0 {
                        eprintln!("opus_repacketizer_cat() failed: {}", opus_strerror(err));
                        break;
                    } else {
                        i += 1;
                    }
                }
            }
        }
        nb_packets = i;
        if eof {
            break;
        }
        if !args.split {
            err = rp.out(&mut output_packet);
            if err > 0 {
                let int_field: [u8; 4] = err.to_be_bytes();
                fout.write_all(&int_field).unwrap();

                let int_field = rng[nb_packets - 1].to_be_bytes();
                fout.write_all(&int_field).unwrap();

                fout.write_all(&output_packet[..err as usize]).unwrap();
            } else {
                eprintln!("opus_repacketizer_out() failed: {}", opus_strerror(err));
            }
        } else {
            let nb_frames = rp.get_nb_frames();
            let mut i = 0;
            while i < nb_frames {
                err = rp.out_range(i, i + 1, &mut output_packet);
                if err > 0 {
                    let int_field: [u8; 4] = err.to_be_bytes();
                    fout.write_all(&int_field).unwrap();
                    let int_field = if i == nb_frames - 1 {
                        rng[nb_packets - 1].to_be_bytes()
                    } else {
                        0i32.to_be_bytes()
                    };
                    fout.write_all(&int_field).unwrap();

                    fout.write_all(&output_packet[..err as usize]).unwrap();
                } else {
                    eprintln!(
                        "opus_repacketizer_out_range() failed: {}",
                        opus_strerror(err)
                    );
                }
                i += 1;
            }
        }
    }
}
