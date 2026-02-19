//! A stronger version of the upstream vector scripts.
//!
//! This tool compares Rust and upstream libopus through shared `opus_demo` helpers.
//! It can run strict parity checks, upstream-style compliance checks, or both.

use clap::{Parser, ValueEnum};
use itertools::iproduct;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use indicatif::ParallelProgressIterator;
use opurs::tools::demo::{
    opus_demo_decode, opus_demo_encode, Application, Channels, CommonOptions, Complexity,
    DecodeArgs, DnnOptions, EncodeArgs, EncoderOptions, FrameSize, OpusBackend, SampleRate,
};
use opurs::tools::{opus_compare, CompareParams};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
enum RunMode {
    Parity,
    Compliance,
    Both,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
enum MatrixMode {
    Quick,
    Full,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, ValueEnum)]
enum SuiteArg {
    Classic,
    Qext,
    QextFuzz,
    DredOpus,
    All,
}

#[derive(Parser, Debug)]
struct Cli {
    /// Path to Opus test vectors.
    vector_path: PathBuf,

    /// Directory to save intermediate files to.
    #[arg(long)]
    dump_dir: Option<PathBuf>,

    /// Run parity checks, compliance checks, or both.
    #[arg(long, value_enum, default_value = "parity")]
    mode: RunMode,

    /// Matrix size for parity encode configurations.
    #[arg(long, value_enum, default_value = "quick")]
    matrix: MatrixMode,

    /// Restrict suites. Repeat `--suite` or pass none for auto-discovery.
    #[arg(long = "suite", value_enum)]
    suites: Vec<SuiteArg>,

    /// Maximum number of vectors per suite (deterministic first-N by name).
    #[arg(long)]
    vector_limit: Option<usize>,

    /// Write a machine-readable JSON report.
    #[arg(long)]
    report_json: Option<PathBuf>,

    /// Mirror upstream `-ignore_extensions` for decode paths.
    #[arg(long)]
    ignore_extensions: bool,

    /// Also run DNN comparison tests (DRED encode, Deep PLC / OSCE decode).
    /// Requires the tools-dnn feature.
    #[arg(long)]
    dnn: bool,

    /// Run only DNN tests (skip standard encode/decode tests).
    #[arg(long)]
    dnn_only: bool,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
enum VectorSuite {
    Classic,
    Qext,
    QextFuzz,
    DredOpus,
}

impl VectorSuite {
    fn as_str(self) -> &'static str {
        match self {
            VectorSuite::Classic => "classic",
            VectorSuite::Qext => "qext",
            VectorSuite::QextFuzz => "qext-fuzz",
            VectorSuite::DredOpus => "dred-opus",
        }
    }
}

#[derive(Debug, Clone)]
struct TestVector {
    suite: VectorSuite,
    name: String,
    encoded: Vec<u8>,
    decoded_stereo: Option<Vec<u8>>,
    decoded_mono: Option<Vec<u8>>,
}

#[derive(Debug, Copy, Clone)]
#[allow(clippy::enum_variant_names)]
enum TestKind {
    ParityDecode {
        channels: Channels,
        sample_rate: SampleRate,
        ignore_extensions: bool,
    },
    ComplianceDecode {
        channels: Channels,
        sample_rate: SampleRate,
        ignore_extensions: bool,
    },
    ParityEncode {
        bitrate: u32,
        application: Application,
        frame_size: FrameSize,
    },
    /// Encode with DRED redundancy, compare Rust vs C bitstreams.
    ParityEncodeDred { bitrate: u32, dred_duration: i32 },
    /// Decode at elevated complexity to exercise Deep PLC / OSCE, compare Rust vs C.
    ParityDecodeDnn {
        channels: Channels,
        sample_rate: SampleRate,
        complexity: Complexity,
        ignore_extensions: bool,
    },
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum TestStatus {
    Pass,
    Fail,
    Skip,
}

#[derive(Debug, Clone)]
struct TestResult {
    status: TestStatus,
    detail: String,
}

impl TestResult {
    fn pass(detail: impl Into<String>) -> Self {
        Self {
            status: TestStatus::Pass,
            detail: detail.into(),
        }
    }

    fn fail(detail: impl Into<String>) -> Self {
        Self {
            status: TestStatus::Fail,
            detail: detail.into(),
        }
    }

    fn skip(detail: impl Into<String>) -> Self {
        Self {
            status: TestStatus::Skip,
            detail: detail.into(),
        }
    }

    fn is_fail(&self) -> bool {
        self.status == TestStatus::Fail
    }

    fn status_str(&self) -> &'static str {
        match self.status {
            TestStatus::Pass => "PASS",
            TestStatus::Fail => "FAIL",
            TestStatus::Skip => "SKIP",
        }
    }
}

#[derive(Debug, Clone)]
struct TestExecution {
    suite: VectorSuite,
    vector: String,
    kind: TestKind,
    result: TestResult,
}

fn app_label(application: Application) -> &'static str {
    match application {
        Application::Voip => "voip",
        Application::Audio => "audio",
        Application::RestrictedLowDelay => "rld",
    }
}

fn frame_size_label(frame_size: FrameSize) -> &'static str {
    match frame_size {
        FrameSize::Ms2_5 => "2.5",
        FrameSize::Ms5 => "5",
        FrameSize::Ms10 => "10",
        FrameSize::Ms20 => "20",
        FrameSize::Ms40 => "40",
        FrameSize::Ms60 => "60",
        FrameSize::Ms80 => "80",
        FrameSize::Ms100 => "100",
        FrameSize::Ms120 => "120",
    }
}

fn kind_label(kind: TestKind) -> String {
    match kind {
        TestKind::ParityDecode {
            channels,
            sample_rate,
            ignore_extensions,
        } => {
            let channels = match channels {
                Channels::Mono => "M",
                Channels::Stereo => "S",
            };
            let sample_rate = format!("{:02}k", usize::from(sample_rate) / 1000);
            if ignore_extensions {
                format!("DEC {} {} (parity, ignore-ext)", channels, sample_rate)
            } else {
                format!("DEC {} {} (parity)", channels, sample_rate)
            }
        }
        TestKind::ComplianceDecode {
            channels,
            sample_rate,
            ignore_extensions,
        } => {
            let channels = match channels {
                Channels::Mono => "M",
                Channels::Stereo => "S",
            };
            let sample_rate = format!("{:02}k", usize::from(sample_rate) / 1000);
            if ignore_extensions {
                format!("DEC {} {} (compliance, ignore-ext)", channels, sample_rate)
            } else {
                format!("DEC {} {} (compliance)", channels, sample_rate)
            }
        }
        TestKind::ParityEncode {
            bitrate,
            application,
            frame_size,
        } => format!(
            "ENC {:03}kbps app={} frame={}ms",
            bitrate / 1000,
            app_label(application),
            frame_size_label(frame_size),
        ),
        TestKind::ParityEncodeDred {
            bitrate,
            dred_duration,
        } => {
            format!("DRED ENC {:03}kbps d={}", bitrate / 1000, dred_duration)
        }
        TestKind::ParityDecodeDnn {
            channels,
            sample_rate,
            complexity,
            ignore_extensions,
        } => {
            let channels = match channels {
                Channels::Mono => "M",
                Channels::Stereo => "S",
            };
            let sample_rate = format!("{:02}k", usize::from(sample_rate) / 1000);
            if ignore_extensions {
                format!(
                    "DNN DEC {} {} c={} (ignore-ext)",
                    channels,
                    sample_rate,
                    i32::from(complexity)
                )
            } else {
                format!(
                    "DNN DEC {} {} c={}",
                    channels,
                    sample_rate,
                    i32::from(complexity)
                )
            }
        }
    }
}

fn is_ascii_digits(value: &str) -> bool {
    !value.is_empty() && value.as_bytes().iter().all(u8::is_ascii_digit)
}

fn load_classic_vectors(vector_dir: &Path) -> Vec<TestVector> {
    let mut stems = BTreeSet::new();

    for entry in std::fs::read_dir(vector_dir).expect("reading test vectors directory") {
        let entry = entry.expect("reading test vectors directory");
        if entry
            .file_type()
            .expect("reading test vectors directory")
            .is_dir()
        {
            continue;
        }

        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();

        if !file_name.starts_with("testvector") || !file_name.ends_with(".bit") {
            continue;
        }

        let stem = file_name.trim_end_matches(".bit");
        let suffix = stem.trim_start_matches("testvector");
        if !is_ascii_digits(suffix) {
            continue;
        }

        stems.insert(stem.to_string());
    }

    let mut vectors = Vec::new();
    for stem in stems {
        let encoded_path = vector_dir.join(format!("{stem}.bit"));
        let decoded_stereo_path = vector_dir.join(format!("{stem}.dec"));
        let decoded_mono_path = vector_dir.join(format!("{stem}m.dec"));

        if !encoded_path.exists() || !decoded_stereo_path.exists() || !decoded_mono_path.exists() {
            continue;
        }

        vectors.push(TestVector {
            suite: VectorSuite::Classic,
            name: stem,
            encoded: std::fs::read(encoded_path).expect("reading encoded file"),
            decoded_stereo: Some(
                std::fs::read(decoded_stereo_path).expect("reading decoded stereo file"),
            ),
            decoded_mono: Some(
                std::fs::read(decoded_mono_path).expect("reading decoded mono file"),
            ),
        });
    }

    vectors
}

fn load_qext_vectors(vector_dir: &Path, fuzz: bool) -> Vec<TestVector> {
    let mut stems = BTreeSet::new();

    for entry in std::fs::read_dir(vector_dir).expect("reading test vectors directory") {
        let entry = entry.expect("reading test vectors directory");
        if entry
            .file_type()
            .expect("reading test vectors directory")
            .is_dir()
        {
            continue;
        }

        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();

        if fuzz {
            if !file_name.starts_with("qext_vector") || !file_name.ends_with("fuzz.bit") {
                continue;
            }
            let middle = file_name
                .trim_start_matches("qext_vector")
                .trim_end_matches("fuzz.bit");
            if !is_ascii_digits(middle) {
                continue;
            }
            stems.insert(file_name.trim_end_matches(".bit").to_string());
        } else {
            if !file_name.starts_with("qext_vector")
                || !file_name.ends_with(".bit")
                || file_name.ends_with("fuzz.bit")
            {
                continue;
            }
            let middle = file_name
                .trim_start_matches("qext_vector")
                .trim_end_matches(".bit");
            if !is_ascii_digits(middle) {
                continue;
            }
            stems.insert(file_name.trim_end_matches(".bit").to_string());
        }
    }

    let suite = if fuzz {
        VectorSuite::QextFuzz
    } else {
        VectorSuite::Qext
    };

    let mut vectors = Vec::new();
    for stem in stems {
        let encoded_path = vector_dir.join(format!("{stem}.bit"));
        if !encoded_path.exists() {
            continue;
        }

        vectors.push(TestVector {
            suite,
            name: stem,
            encoded: std::fs::read(encoded_path).expect("reading qext bitstream"),
            decoded_stereo: None,
            decoded_mono: None,
        });
    }

    vectors
}

fn load_dred_opus_vectors(vector_dir: &Path) -> Vec<TestVector> {
    let mut stems = BTreeSet::new();

    for entry in std::fs::read_dir(vector_dir).expect("reading test vectors directory") {
        let entry = entry.expect("reading test vectors directory");
        if entry
            .file_type()
            .expect("reading test vectors directory")
            .is_dir()
        {
            continue;
        }

        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if !file_name.starts_with("vector") || !file_name.ends_with("_opus.bit") {
            continue;
        }

        let middle = file_name
            .trim_start_matches("vector")
            .trim_end_matches("_opus.bit");
        if !is_ascii_digits(middle) {
            continue;
        }

        stems.insert(file_name.trim_end_matches(".bit").to_string());
    }

    let mut vectors = Vec::new();
    for stem in stems {
        let encoded_path = vector_dir.join(format!("{stem}.bit"));
        if !encoded_path.exists() {
            continue;
        }

        vectors.push(TestVector {
            suite: VectorSuite::DredOpus,
            name: stem,
            encoded: std::fs::read(encoded_path).expect("reading dred-opus bitstream"),
            decoded_stereo: None,
            decoded_mono: None,
        });
    }

    vectors
}

fn load_vectors_by_suite(vector_dir: &Path) -> BTreeMap<VectorSuite, Vec<TestVector>> {
    let mut suites = BTreeMap::new();

    let classic = load_classic_vectors(vector_dir);
    if !classic.is_empty() {
        suites.insert(VectorSuite::Classic, classic);
    }

    let qext = load_qext_vectors(vector_dir, false);
    if !qext.is_empty() {
        suites.insert(VectorSuite::Qext, qext);
    }

    let qext_fuzz = load_qext_vectors(vector_dir, true);
    if !qext_fuzz.is_empty() {
        suites.insert(VectorSuite::QextFuzz, qext_fuzz);
    }

    let dred_opus = load_dred_opus_vectors(vector_dir);
    if !dred_opus.is_empty() {
        suites.insert(VectorSuite::DredOpus, dred_opus);
    }

    suites
}

fn resolve_selected_suites(
    requested: &[SuiteArg],
    discovered: &BTreeMap<VectorSuite, Vec<TestVector>>,
) -> Vec<VectorSuite> {
    let discovered_all = discovered.keys().copied().collect::<Vec<_>>();

    if requested.is_empty() || requested.contains(&SuiteArg::All) {
        return discovered_all;
    }

    let mut output = Vec::new();
    for &suite in requested {
        let suite = match suite {
            SuiteArg::Classic => Some(VectorSuite::Classic),
            SuiteArg::Qext => Some(VectorSuite::Qext),
            SuiteArg::QextFuzz => Some(VectorSuite::QextFuzz),
            SuiteArg::DredOpus => Some(VectorSuite::DredOpus),
            SuiteArg::All => None,
        };
        if let Some(suite) = suite {
            output.push(suite);
        }
    }

    output.sort();
    output.dedup();
    output
}

fn build_classic_standard_kinds(
    mode: RunMode,
    matrix: MatrixMode,
    ignore_extensions: bool,
) -> Vec<TestKind> {
    let mut kinds = Vec::new();

    let parity_enabled = matches!(mode, RunMode::Parity | RunMode::Both);
    let compliance_enabled = matches!(mode, RunMode::Compliance | RunMode::Both);

    let decode_sample_rates = [
        SampleRate::R48000,
        SampleRate::R24000,
        SampleRate::R16000,
        SampleRate::R12000,
        SampleRate::R8000,
    ];

    if parity_enabled {
        kinds.extend(
            iproduct!(
                decode_sample_rates.iter(),
                [Channels::Mono, Channels::Stereo].iter()
            )
            .map(|(&sample_rate, &channels)| TestKind::ParityDecode {
                sample_rate,
                channels,
                ignore_extensions,
            }),
        );
    }

    if compliance_enabled {
        kinds.extend(
            iproduct!(
                decode_sample_rates.iter(),
                [Channels::Mono, Channels::Stereo].iter()
            )
            .map(|(&sample_rate, &channels)| TestKind::ComplianceDecode {
                sample_rate,
                channels,
                ignore_extensions,
            }),
        );
    }

    if parity_enabled {
        let bitrates = [
            10_000u32, 20_000, 30_000, 45_000, 60_000, 90_000, 120_000, 180_000, 240_000,
        ];
        let applications: &[Application] = match matrix {
            MatrixMode::Quick => &[Application::Audio],
            MatrixMode::Full => &[
                Application::Audio,
                Application::Voip,
                Application::RestrictedLowDelay,
            ],
        };
        let frame_sizes: &[FrameSize] = match matrix {
            MatrixMode::Quick => &[FrameSize::Ms20],
            MatrixMode::Full => &[
                FrameSize::Ms10,
                FrameSize::Ms20,
                FrameSize::Ms40,
                FrameSize::Ms60,
            ],
        };

        kinds.extend(
            iproduct!(bitrates.iter(), applications.iter(), frame_sizes.iter()).map(
                |(&bitrate, &application, &frame_size)| TestKind::ParityEncode {
                    bitrate,
                    application,
                    frame_size,
                },
            ),
        );
    }

    kinds
}

fn build_dnn_kinds(ignore_extensions: bool) -> Vec<TestKind> {
    let mut kinds = Vec::new();

    kinds.extend(
        iproduct!([32_000u32, 64_000, 128_000].iter(), [5i32, 10].iter()).map(
            |(&bitrate, &dred_duration)| TestKind::ParityEncodeDred {
                bitrate,
                dred_duration,
            },
        ),
    );

    kinds.extend(
        iproduct!(
            [SampleRate::R48000, SampleRate::R16000].iter(),
            [Channels::Mono, Channels::Stereo].iter(),
            [
                Complexity::C5,
                Complexity::C6,
                Complexity::C7,
                Complexity::C10
            ]
            .iter()
        )
        .map(
            |(&sample_rate, &channels, &complexity)| TestKind::ParityDecodeDnn {
                sample_rate,
                channels,
                complexity,
                ignore_extensions,
            },
        ),
    );

    kinds
}

fn run_test(
    test_vector: &TestVector,
    test_kind: TestKind,
    dump_directory: Option<&Path>,
) -> TestResult {
    match test_kind {
        TestKind::ParityDecode {
            sample_rate,
            channels,
            ignore_extensions,
        } => {
            let decode_args = DecodeArgs {
                sample_rate,
                channels,
                options: CommonOptions {
                    ignore_extensions,
                    ..Default::default()
                },
                complexity: None,
            };
            let no_dnn = DnnOptions::default();

            let upstream_decoded = opus_demo_decode(
                OpusBackend::Upstream,
                &test_vector.encoded,
                decode_args,
                &no_dnn,
            );
            let rust_decoded = opus_demo_decode(
                OpusBackend::Rust,
                &test_vector.encoded,
                decode_args,
                &no_dnn,
            );

            if let Some(dump_directory) = dump_directory {
                let name_base = format!(
                    "{}_dec_{}_{}_{}",
                    test_vector.suite.as_str(),
                    test_vector.name,
                    usize::from(sample_rate),
                    match channels {
                        Channels::Mono => "mono",
                        Channels::Stereo => "stereo",
                    }
                );

                std::fs::write(
                    dump_directory.join(format!("{}_upstream.dec", &name_base)),
                    &upstream_decoded,
                )
                .expect("writing upstream decode dump");
                std::fs::write(
                    dump_directory.join(format!("{}_rust.dec", &name_base)),
                    &rust_decoded,
                )
                .expect("writing rust decode dump");
            }

            if upstream_decoded == rust_decoded {
                TestResult::pass("exact decoded audio")
            } else {
                TestResult::fail("different decoded audio")
            }
        }
        TestKind::ComplianceDecode {
            sample_rate,
            channels,
            ignore_extensions,
        } => {
            let Some(true_stereo) = test_vector.decoded_stereo.as_deref() else {
                return TestResult::skip("missing compliance stereo reference");
            };
            let Some(true_mono) = test_vector.decoded_mono.as_deref() else {
                return TestResult::skip("missing compliance mono reference");
            };

            let decode_args = DecodeArgs {
                sample_rate,
                channels,
                options: CommonOptions {
                    ignore_extensions,
                    ..Default::default()
                },
                complexity: None,
            };
            let no_dnn = DnnOptions::default();

            let rust_decoded = opus_demo_decode(
                OpusBackend::Rust,
                &test_vector.encoded,
                decode_args,
                &no_dnn,
            );

            if let Some(dump_directory) = dump_directory {
                let name_base = format!(
                    "{}_cmp_{}_{}_{}",
                    test_vector.suite.as_str(),
                    test_vector.name,
                    usize::from(sample_rate),
                    match channels {
                        Channels::Mono => "mono",
                        Channels::Stereo => "stereo",
                    }
                );

                std::fs::write(
                    dump_directory.join(format!("{}_rust.dec", &name_base)),
                    &rust_decoded,
                )
                .expect("writing compliance decode dump");
            }

            let params = CompareParams {
                sample_rate,
                channels,
            };
            let stereo_cmp = opus_compare(params, true_stereo, &rust_decoded);
            let mono_cmp = opus_compare(params, true_mono, &rust_decoded);
            let pass = stereo_cmp.is_success() || mono_cmp.is_success();

            let detail = format!(
                "quality stereo={:.2}% mono={:.2}%",
                stereo_cmp.quality, mono_cmp.quality
            );

            if pass {
                TestResult::pass(detail)
            } else {
                TestResult::fail(detail)
            }
        }
        TestKind::ParityEncode {
            bitrate,
            application,
            frame_size,
        } => {
            let Some(true_decoded) = test_vector.decoded_stereo.as_deref() else {
                return TestResult::skip("missing stereo reference PCM for encode parity");
            };

            let encode_args = EncodeArgs {
                application,
                sample_rate: SampleRate::R48000,
                channels: Channels::Stereo,
                bitrate,
                options: EncoderOptions {
                    framesize: frame_size,
                    ..Default::default()
                },
            };
            let decode_args = DecodeArgs {
                sample_rate: SampleRate::R48000,
                channels: Channels::Stereo,
                options: Default::default(),
                complexity: None,
            };
            let no_dnn = DnnOptions::default();

            let (upstream_encoded, pre_skip) =
                opus_demo_encode(OpusBackend::Upstream, true_decoded, encode_args, &no_dnn);
            let (rust_encoded, rust_pre_skip) =
                opus_demo_encode(OpusBackend::Rust, true_decoded, encode_args, &no_dnn);

            if rust_pre_skip != pre_skip {
                return TestResult::fail(format!(
                    "different pre-skip: rust={} upstream={}",
                    rust_pre_skip, pre_skip
                ));
            }

            if let Some(dump_directory) = dump_directory {
                let name_base = format!(
                    "{}_enc_{}_{}_{}_{}",
                    test_vector.suite.as_str(),
                    test_vector.name,
                    bitrate,
                    app_label(application),
                    frame_size_label(frame_size)
                );

                std::fs::write(
                    dump_directory.join(format!("{}_upstream.enc", name_base)),
                    &upstream_encoded,
                )
                .expect("writing upstream encode dump");
                std::fs::write(
                    dump_directory.join(format!("{}_rust.enc", name_base)),
                    &rust_encoded,
                )
                .expect("writing rust encode dump");

                let upstream_decoded = opus_demo_decode(
                    OpusBackend::Upstream,
                    &upstream_encoded,
                    decode_args,
                    &no_dnn,
                );
                let rust_decoded =
                    opus_demo_decode(OpusBackend::Upstream, &rust_encoded, decode_args, &no_dnn);

                std::fs::write(
                    dump_directory.join(format!("{}_upstream.dec", name_base)),
                    &upstream_decoded,
                )
                .expect("writing upstream re-decode dump");
                std::fs::write(
                    dump_directory.join(format!("{}_rust.dec", name_base)),
                    &rust_decoded,
                )
                .expect("writing rust re-decode dump");
            }

            if upstream_encoded == rust_encoded {
                TestResult::pass("exact bitstream")
            } else {
                TestResult::fail("different bitstream")
            }
        }
        TestKind::ParityEncodeDred {
            bitrate,
            dred_duration,
        } => {
            let Some(true_decoded) = test_vector.decoded_stereo.as_deref() else {
                return TestResult::skip("missing stereo reference PCM for DRED encode parity");
            };

            let encode_args = EncodeArgs {
                application: Application::Audio,
                sample_rate: SampleRate::R48000,
                channels: Channels::Stereo,
                bitrate,
                options: EncoderOptions {
                    dred_duration,
                    ..Default::default()
                },
            };
            let dnn = DnnOptions::default();

            let (upstream_encoded, pre_skip) =
                opus_demo_encode(OpusBackend::Upstream, true_decoded, encode_args, &dnn);
            let (rust_encoded, rust_pre_skip) =
                opus_demo_encode(OpusBackend::Rust, true_decoded, encode_args, &dnn);

            if rust_pre_skip != pre_skip {
                return TestResult::fail(format!(
                    "different pre-skip: rust={} upstream={}",
                    rust_pre_skip, pre_skip
                ));
            }

            if let Some(dump_directory) = dump_directory {
                let name_base = format!(
                    "{}_dred_enc_{}_{}_d{}",
                    test_vector.suite.as_str(),
                    test_vector.name,
                    bitrate,
                    dred_duration
                );
                std::fs::write(
                    dump_directory.join(format!("{}_upstream.enc", name_base)),
                    &upstream_encoded,
                )
                .expect("writing upstream dred encode dump");
                std::fs::write(
                    dump_directory.join(format!("{}_rust.enc", name_base)),
                    &rust_encoded,
                )
                .expect("writing rust dred encode dump");
            }

            if upstream_encoded == rust_encoded {
                TestResult::pass("exact bitstream")
            } else {
                TestResult::fail("different bitstream")
            }
        }
        TestKind::ParityDecodeDnn {
            sample_rate,
            channels,
            complexity,
            ignore_extensions,
        } => {
            let decode_args = DecodeArgs {
                sample_rate,
                channels,
                options: CommonOptions {
                    ignore_extensions,
                    ..Default::default()
                },
                complexity: Some(complexity),
            };
            let dnn = DnnOptions::default();

            let upstream_decoded = opus_demo_decode(
                OpusBackend::Upstream,
                &test_vector.encoded,
                decode_args,
                &dnn,
            );
            let rust_decoded =
                opus_demo_decode(OpusBackend::Rust, &test_vector.encoded, decode_args, &dnn);

            if let Some(dump_directory) = dump_directory {
                let name_base = format!(
                    "{}_dnn_dec_{}_{}_{}_c{}",
                    test_vector.suite.as_str(),
                    test_vector.name,
                    usize::from(sample_rate),
                    match channels {
                        Channels::Mono => "mono",
                        Channels::Stereo => "stereo",
                    },
                    i32::from(complexity)
                );

                std::fs::write(
                    dump_directory.join(format!("{}_upstream.dec", &name_base)),
                    &upstream_decoded,
                )
                .expect("writing upstream dnn decode dump");
                std::fs::write(
                    dump_directory.join(format!("{}_rust.dec", &name_base)),
                    &rust_decoded,
                )
                .expect("writing rust dnn decode dump");
            }

            if upstream_decoded == rust_decoded {
                TestResult::pass("exact decoded audio")
            } else {
                TestResult::fail("different decoded audio")
            }
        }
    }
}

fn json_escape(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(output, "\\u{:04x}", c as u32);
            }
            c => output.push(c),
        }
    }
    output
}

fn write_json_report(path: &Path, elapsed: Duration, results: &[TestExecution]) {
    let failed = results
        .iter()
        .filter(|r| r.result.status == TestStatus::Fail)
        .count();
    let skipped = results
        .iter()
        .filter(|r| r.result.status == TestStatus::Skip)
        .count();
    let passed = results.len() - failed - skipped;

    let mut json = String::new();
    let _ = write!(
        json,
        "{{\"elapsed_ms\":{},\"summary\":{{\"passed\":{},\"failed\":{},\"skipped\":{},\"total\":{}}},\"results\":[",
        elapsed.as_millis(),
        passed,
        failed,
        skipped,
        results.len()
    );

    for (index, result) in results.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }

        let _ = write!(
            json,
            "{{\"suite\":\"{}\",\"vector\":\"{}\",\"kind\":\"{}\",\"status\":\"{}\",\"detail\":\"{}\"}}",
            json_escape(result.suite.as_str()),
            json_escape(&result.vector),
            json_escape(&kind_label(result.kind)),
            result.result.status_str(),
            json_escape(&result.result.detail),
        );
    }

    json.push_str("]}");
    std::fs::write(path, json).expect("writing JSON report");
}

fn main() {
    let args = Cli::parse();

    let mut vectors_by_suite = load_vectors_by_suite(&args.vector_path);

    if let Some(limit) = args.vector_limit {
        for vectors in vectors_by_suite.values_mut() {
            vectors.truncate(limit);
        }
    }

    if vectors_by_suite.is_empty() {
        eprintln!("No vectors found in {}", args.vector_path.display());
        std::process::exit(1);
    }

    let selected_suites = resolve_selected_suites(&args.suites, &vectors_by_suite);
    if selected_suites.is_empty() {
        eprintln!("No suites selected");
        std::process::exit(1);
    }

    if let Some(ref dump_dir) = args.dump_dir {
        std::fs::remove_dir_all(dump_dir)
            .or_else(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Ok(())
                } else {
                    Err(e)
                }
            })
            .expect("removing dump directory");
        std::fs::create_dir(dump_dir).expect("creating dump directory");
    }

    let run_standard = !args.dnn_only;
    let classic_standard_kinds =
        build_classic_standard_kinds(args.mode, args.matrix, args.ignore_extensions);
    let dnn_kinds = build_dnn_kinds(args.ignore_extensions);

    let mut tests = Vec::<(&TestVector, TestKind)>::new();

    for &suite in &selected_suites {
        let Some(vectors) = vectors_by_suite.get(&suite) else {
            eprintln!("Requested suite '{}' was not discovered", suite.as_str());
            continue;
        };

        let mut suite_kinds = Vec::new();
        if run_standard {
            match suite {
                VectorSuite::Classic => {
                    suite_kinds.extend(classic_standard_kinds.iter().copied());
                }
                VectorSuite::Qext | VectorSuite::QextFuzz => {
                    if matches!(args.mode, RunMode::Parity | RunMode::Both) {
                        suite_kinds.push(TestKind::ParityDecode {
                            sample_rate: SampleRate::R96000,
                            channels: Channels::Stereo,
                            ignore_extensions: args.ignore_extensions,
                        });
                    }
                }
                VectorSuite::DredOpus => {
                    if matches!(args.mode, RunMode::Parity | RunMode::Both) {
                        suite_kinds.push(TestKind::ParityDecode {
                            sample_rate: SampleRate::R16000,
                            channels: Channels::Mono,
                            ignore_extensions: args.ignore_extensions,
                        });
                    }
                }
            }
        }

        if (args.dnn || args.dnn_only) && suite == VectorSuite::Classic {
            suite_kinds.extend(dnn_kinds.iter().copied());
        }

        if suite_kinds.is_empty() {
            continue;
        }

        for vector in vectors {
            for &kind in &suite_kinds {
                tests.push((vector, kind));
            }
        }
    }

    if tests.is_empty() {
        eprintln!("No tests selected after suite/mode filtering");
        std::process::exit(1);
    }

    println!("Running {} tests in parallel", tests.len());
    for &suite in &selected_suites {
        if let Some(vectors) = vectors_by_suite.get(&suite) {
            println!("  suite={} vectors={}", suite.as_str(), vectors.len());
        }
    }

    let start_time = Instant::now();

    let results = tests
        .into_par_iter()
        .progress()
        .map(|(test_vector, test_kind)| {
            let test_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_test(test_vector, test_kind, args.dump_dir.as_deref())
            }))
            .unwrap_or_else(|_| TestResult::fail("panic while running test"));

            TestExecution {
                suite: test_vector.suite,
                vector: test_vector.name.clone(),
                kind: test_kind,
                result: test_result,
            }
        })
        .collect::<Vec<_>>();

    let elapsed = start_time.elapsed();
    println!("Ran {} tests in {:?}", results.len(), elapsed);

    for result in &results {
        println!(
            "[{}/{}] {}: {} -> {}",
            result.suite.as_str(),
            result.vector,
            kind_label(result.kind),
            result.result.status_str(),
            result.result.detail
        );
    }

    let failed = results
        .iter()
        .filter(|r| r.result.status == TestStatus::Fail)
        .count();
    let skipped = results
        .iter()
        .filter(|r| r.result.status == TestStatus::Skip)
        .count();
    let passed = results.len() - failed - skipped;

    println!(
        "summary: passed={} failed={} skipped={} total={}",
        passed,
        failed,
        skipped,
        results.len()
    );

    if let Some(report_path) = args.report_json.as_deref() {
        write_json_report(report_path, elapsed, &results);
        println!("wrote report to {}", report_path.display());
    }

    if results.iter().any(|r| r.result.is_fail()) {
        std::process::exit(1);
    }
}
