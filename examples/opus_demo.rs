#![forbid(unsafe_code)]

use clap::{Parser, Subcommand};
use opurs::opus_get_version_string;
use opurs::tools::demo::{
    opus_demo_adjust_length, opus_demo_decode, opus_demo_encode, Application, Bandwidth, Channels,
    CommonOptions, Complexity, DecodeArgs, DnnOptions, EncodeArgs, EncoderOptions, FrameSize,
    OpusBackend, SampleRate, MAX_PACKET,
};

#[derive(Parser, Debug, Clone)]
#[command(about = "Opus encoder/decoder demo")]
struct Cli {
    /// Backend to use: unsafe (Rust) or upstream (C)
    #[arg(short, long, default_value = "unsafe")]
    backend: OpusBackend,

    #[command(subcommand)]
    mode: Mode,

    /// Input file
    input: std::path::PathBuf,

    /// Output file
    output: std::path::PathBuf,
}

#[derive(Subcommand, Debug, Clone)]
enum Mode {
    /// Encode and decode (roundtrip)
    #[command(name = "enc-dec")]
    EncodeDecode(EncodeCliArgs),

    /// Encode only (output bitstream)
    #[command(name = "enc")]
    EncodeOnly(EncodeCliArgs),

    /// Decode only (read bitstream)
    #[command(name = "dec")]
    DecodeOnly(DecodeCliArgs),
}

#[derive(Parser, Debug, Clone)]
struct EncodeCliArgs {
    /// Application: voip, audio, or restricted-lowdelay
    application: Application,

    /// Sampling rate in Hz
    sample_rate: SampleRate,

    /// Number of channels (1 or 2)
    channels: Channels,

    /// Bitrate in bits per second
    bitrate: u32,

    /// Use constant bit-rate
    #[arg(long)]
    cbr: bool,

    /// Enable constrained variable bitrate
    #[arg(long)]
    cvbr: bool,

    /// Audio bandwidth (NB, MB, WB, SWB, FB)
    #[arg(long)]
    bandwidth: Option<Bandwidth>,

    /// Frame size in ms (2.5, 5, 10, 20, 40, 60, 80, 100, 120)
    #[arg(long, default_value = "20")]
    framesize: FrameSize,

    /// Maximum payload size in bytes
    #[arg(long, default_value_t = 1024)]
    max_payload: usize,

    /// Complexity (0-10)
    #[arg(long, default_value = "10")]
    complexity: Complexity,

    /// Force mono encoding
    #[arg(long)]
    forcemono: bool,

    /// Enable SILK DTX
    #[arg(long)]
    dtx: bool,

    /// Use look-ahead for speech/music detection
    #[arg(long)]
    delayed_decision: bool,

    /// Enable SILK inband FEC
    #[arg(long)]
    inbandfec: bool,

    /// Simulate packet loss percentage (0-100)
    #[arg(long, default_value_t = 0)]
    loss: u32,

    /// DRED duration in frames (0 = disabled, max 100). Requires 'dred' feature.
    #[arg(long, default_value_t = 0)]
    dred_duration: i32,

    /// Path to external DNN weights blob file (uses compiled-in weights if omitted)
    #[arg(long)]
    weights: Option<std::path::PathBuf>,
}

impl EncodeCliArgs {
    fn into_encode_args(self) -> EncodeArgs {
        assert!(
            self.max_payload <= MAX_PACKET,
            "max_payload must be <= {MAX_PACKET}"
        );
        assert!(self.loss <= 100, "loss must be between 0 and 100");

        EncodeArgs {
            application: self.application,
            sample_rate: self.sample_rate,
            channels: self.channels,
            bitrate: self.bitrate,
            options: EncoderOptions {
                cbr: self.cbr,
                cvbr: self.cvbr,
                bandwidth: self.bandwidth,
                framesize: self.framesize,
                max_payload: self.max_payload,
                complexity: self.complexity,
                forcemono: self.forcemono,
                dtx: self.dtx,
                delayed_decision: self.delayed_decision,
                common: CommonOptions {
                    inbandfec: self.inbandfec,
                    loss: self.loss,
                },
                dred_duration: self.dred_duration,
            },
        }
    }

    fn to_dnn_options(&self) -> DnnOptions {
        DnnOptions {
            weights_file: self.weights.clone(),
        }
    }
}

#[derive(Parser, Debug, Clone)]
struct DecodeCliArgs {
    /// Sampling rate in Hz
    sample_rate: SampleRate,

    /// Number of channels (1 or 2)
    channels: Channels,

    /// Enable SILK inband FEC
    #[arg(long)]
    inbandfec: bool,

    /// Simulate packet loss percentage (0-100)
    #[arg(long, default_value_t = 0)]
    loss: u32,

    /// Decoder complexity (0-10). Controls Deep PLC (>=5) and OSCE (>=6) activation.
    #[arg(long)]
    decoder_complexity: Option<Complexity>,

    /// Path to external DNN weights blob file (uses compiled-in weights if omitted)
    #[arg(long)]
    weights: Option<std::path::PathBuf>,
}

impl DecodeCliArgs {
    fn into_decode_args(self) -> DecodeArgs {
        assert!(self.loss <= 100, "loss must be between 0 and 100");

        DecodeArgs {
            sample_rate: self.sample_rate,
            channels: self.channels,
            options: CommonOptions {
                inbandfec: self.inbandfec,
                loss: self.loss,
            },
            complexity: self.decoder_complexity,
        }
    }

    fn to_dnn_options(&self) -> DnnOptions {
        DnnOptions {
            weights_file: self.weights.clone(),
        }
    }
}

fn main() {
    let cli = Cli::parse();

    eprintln!("{}", opus_get_version_string());

    let backend = cli.backend;

    match cli.mode {
        Mode::EncodeDecode(ref enc_cli) => {
            let dnn = enc_cli.to_dnn_options();
            let args = enc_cli.clone().into_encode_args();
            let fin = std::fs::read(&cli.input).expect("failed to read input file");
            let (encoded, pre_skip) = opus_demo_encode(backend, &fin, args, &dnn);
            let mut decoded = opus_demo_decode(
                backend,
                &encoded,
                DecodeArgs {
                    sample_rate: args.sample_rate,
                    channels: args.channels,
                    options: args.options.common,
                    complexity: None,
                },
                &dnn,
            );
            opus_demo_adjust_length(
                &mut decoded,
                pre_skip,
                fin.len(),
                args.sample_rate,
                args.channels,
            );
            std::fs::write(&cli.output, &decoded).expect("failed to write output file");
        }
        Mode::EncodeOnly(ref enc_cli) => {
            let dnn = enc_cli.to_dnn_options();
            let args = enc_cli.clone().into_encode_args();
            let fin = std::fs::read(&cli.input).expect("failed to read input file");
            let (output, _pre_skip) = opus_demo_encode(backend, &fin, args, &dnn);
            std::fs::write(&cli.output, &output).expect("failed to write output file");
        }
        Mode::DecodeOnly(ref dec_cli) => {
            let dnn = dec_cli.to_dnn_options();
            let args = dec_cli.clone().into_decode_args();
            let fin = std::fs::read(&cli.input).expect("failed to read input file");
            let output = opus_demo_decode(backend, &fin, args, &dnn);
            std::fs::write(&cli.output, &output).expect("failed to write output file");
        }
    }
}
