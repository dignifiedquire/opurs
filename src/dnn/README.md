# DNN Subsystem

Neural network inference engine for Opus 1.5.2, ported from the upstream C
implementation in `dnn/`. Provides three codec enhancements:

- **Deep PLC** — Neural packet loss concealment using LPCNet/FARGAN vocoder
- **DRED** — Deep REDundancy coding via RDOVAE for loss recovery
- **OSCE** — Online Speech Coding Enhancement (LACE and NoLACE adaptive filtering)

## References

- J.-M. Valin, J. Skoglund, [LPCNet: Improving Neural Speech Synthesis Through Linear Prediction](https://jmvalin.ca/papers/lpcnet_icassp2019.pdf), *Proc. ICASSP*, 2019.
- J.-M. Valin et al., [Neural Speech Synthesis on a Shoestring](https://jmvalin.ca/papers/improved_lpcnet.pdf), *Proc. ICASSP*, 2022.
- J.-M. Valin et al., [Real-Time Packet Loss Concealment With Mixed Generative and Predictive Model](https://jmvalin.ca/papers/lpcnet_plc.pdf), *Proc. INTERSPEECH*, 2022.
- J.-M. Valin et al., [Low-Bitrate Redundancy Coding of Speech Using a Rate-Distortion-Optimized Variational Autoencoder](https://jmvalin.ca/papers/valin_dred.pdf), *Proc. ICASSP*, 2023.

## Module Layout

```
dnn/
├── nnet.rs              # Core: WeightArray, parse_weights, write_weights, linear_init, conv2d_init
├── weights.rs           # Weight loading API: compiled_weights(), load_weights()
├── vec.rs               # SIMD-free vector math (dot products, activations)
├── freq.rs              # Frequency-domain utilities (bark scale, DCT)
├── burg.rs              # Burg's method for LPC estimation
├── lpcnet.rs            # LPCNet encoder + PLC state (orchestrates PitchDNN + FARGAN)
├── lpcnet_tables.rs     # LPC codec tables
├── pitchdnn.rs          # PitchDNN: neural pitch estimator
├── fargan.rs            # FARGAN: neural vocoder for speech synthesis
├── pitchdnn_data.rs     # [generated] PitchDNN weights (28 arrays)
├── fargan_data.rs       # [generated] FARGAN weights (66 arrays)
├── plc_data.rs          # [generated] PLC model weights (20 arrays)
├── dred/                # DRED subsystem (feature: dred)
│   ├── encoder.rs       # DRED encoder (DREDEnc)
│   ├── decoder.rs       # DRED decoder (OpusDREDDecoder)
│   ├── rdovae_enc.rs    # RDOVAE encoder network
│   ├── rdovae_dec.rs    # RDOVAE decoder network
│   ├── rdovae_enc_data.rs  # [generated] RDOVAE encoder weights (79 arrays)
│   ├── rdovae_dec_data.rs  # [generated] RDOVAE decoder weights (97 arrays)
│   ├── stats.rs         # Quantization statistics
│   ├── coding.rs        # Entropy coding for DRED latents
│   └── config.rs        # DRED constants
├── osce.rs              # OSCE: LACE + NoLACE models and inference (feature: osce)
├── osce_lace_data.rs    # [generated] LACE weights (42 arrays)
├── osce_nolace_data.rs  # [generated] NoLACE weights (104 arrays)
└── nndsp.rs             # Neural DSP primitives for OSCE (feature: osce)
```

## Feature Flags

| Feature    | What it enables                             | Depends on |
|------------|---------------------------------------------|------------|
| `deep-plc` | Deep PLC (nnet, pitchdnn, fargan, lpcnet)  | —          |
| `dred`     | DRED encoder/decoder + RDOVAE              | `deep-plc` |
| `osce`     | LACE/NoLACE speech enhancement             | `deep-plc` |
| `dnn`      | All of the above                            | `deep-plc`, `dred`, `osce` |

## Weight Loading

Models require weight data to function. Two loading methods are available:

### Compiled-in weights (recommended)

When DNN features are enabled, weight data is compiled directly into the binary
from the `*_data.rs` files (~4.7 MB total across all models):

```rust
// Encoder (requires "dred" feature)
let mut encoder = OpusEncoder::new(48000, 1, OPUS_APPLICATION_VOIP)?;
encoder.load_dnn_weights()?;

// Decoder (requires "deep-plc" feature)
let mut decoder = OpusDecoder::new(48000, 1)?;
decoder.load_dnn_weights()?;
```

### Runtime loading from blob

Alternatively, load weights from an external binary file at runtime:

```rust
let blob = std::fs::read("weights_blob.bin")?;
decoder.set_dnn_blob(&blob)?;
```

The blob format uses 64-byte headers per array with the "DNNw" magic. This is the
same format used by upstream libopus with `USE_WEIGHTS_FILE`.

## Activation Thresholds

DNN features activate based on decoder complexity:

| Complexity | Feature activated |
|------------|-------------------|
| >= 5       | Deep PLC          |
| >= 8       | OSCE (LACE)       |
| >= 9       | OSCE (NoLACE)     |

DRED encoding is controlled separately via `encoder.set_dred_duration()`.

## Regenerating Weight Data

The `*_data.rs` files are auto-generated from the upstream C weight arrays.
To regenerate after updating upstream libopus:

```bash
cargo run --features tools-dnn --example dnn_codegen
```

This extracts the compiled-in C weights via `libopus-sys` and emits Rust source.

## Training / Export

Model training and weight export use Python/PyTorch and live in the upstream
libopus repository only. They are not part of this Rust port. The ported Rust
code handles inference only.
