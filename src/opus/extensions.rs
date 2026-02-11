//! Opus packet extension parsing and generation.
//!
//! Extensions are carried in the padding area of Opus packets.
//! Each extension has an ID (2..127), a frame number, and variable-length data.
//!
//! Upstream C: `src/extensions.c`

// These functions are used by the DNN subsystem (DRED), not yet integrated.
#![allow(dead_code)]

use crate::opus::opus_defines::{OPUS_BAD_ARG, OPUS_BUFFER_TOO_SMALL, OPUS_INVALID_PACKET};

/// Extension data associated with a specific frame in an Opus packet.
///
/// Upstream C: src/opus_private.h:opus_extension_data
#[derive(Clone, Debug)]
pub struct OpusExtensionData {
    pub id: i32,
    pub frame: i32,
    pub data: Vec<u8>,
}

/// Given an extension payload, advance past the current extension and return
/// the length of the remaining data. Returns `(remaining_len, header_size)` or
/// negative on error.
///
/// Upstream C: src/extensions.c:skip_extension
fn skip_extension(data: &[u8], mut pos: usize, len: usize) -> Result<(usize, usize), i32> {
    if pos >= len {
        return Ok((0, 0));
    }
    let id = data[pos] >> 1;
    let l = data[pos] & 1;
    if id == 0 && l == 1 {
        // Padding byte
        let header_size = 1;
        pos += 1;
        if pos > len {
            return Err(-1);
        }
        Ok((len - pos, header_size))
    } else if id > 0 && id < 32 {
        // Short extension: 1-byte header + optional 1-byte payload
        if pos + 1 + l as usize > len {
            return Err(-1);
        }
        pos += 1 + l as usize;
        Ok((len - pos, 1))
    } else {
        // Long extension (id >= 32)
        if l == 0 {
            // Last extension: consumes rest of data
            Ok((0, 1))
        } else {
            // Not last: length encoded as sum of bytes until non-255
            let mut bytes: usize = 0;
            let mut header_size = 1;
            loop {
                pos += 1;
                if pos >= len {
                    return Err(-1);
                }
                bytes += data[pos] as usize;
                header_size += 1;
                if data[pos] != 255 {
                    break;
                }
            }
            pos += 1;
            if bytes <= len - pos {
                pos += bytes;
                Ok((len - pos, header_size))
            } else {
                Err(-1)
            }
        }
    }
}

/// Count the number of extensions (excluding padding and separators).
///
/// Upstream C: src/extensions.c:opus_packet_extensions_count
pub fn opus_packet_extensions_count(data: &[u8]) -> Result<i32, i32> {
    let len = data.len();
    let mut count: i32 = 0;
    let mut pos: usize = 0;
    let mut remaining = len;

    while remaining > 0 {
        let id = data[pos] >> 1;
        let (new_remaining, _header_size) = skip_extension(data, pos, pos + remaining)?;
        let consumed = remaining - new_remaining;
        pos += consumed;
        remaining = new_remaining;
        if id > 1 {
            count += 1;
        }
    }
    Ok(count)
}

/// Parse extensions from Opus padding data.
///
/// Returns a vector of `OpusExtensionData` entries (excluding padding and separators).
///
/// Upstream C: src/extensions.c:opus_packet_extensions_parse
pub fn opus_packet_extensions_parse(
    data: &[u8],
    max_extensions: i32,
) -> Result<Vec<OpusExtensionData>, i32> {
    let len = data.len();
    let mut extensions = Vec::new();
    let mut curr_frame: i32 = 0;
    let mut pos: usize = 0;
    let mut remaining = len;

    while remaining > 0 {
        let id = data[pos] >> 1;

        let ext_data_start = pos;

        if id > 1 {
            // Real extension — will record after skip
        } else if id == 1 {
            // Frame separator
            let l = data[pos] & 1;
            if l == 0 {
                curr_frame += 1;
            } else if remaining >= 2 {
                curr_frame += data[pos + 1] as i32;
            }
            if curr_frame >= 48 {
                return Err(OPUS_INVALID_PACKET);
            }
        }

        let (new_remaining, header_size) =
            skip_extension(data, pos, pos + remaining).map_err(|_| OPUS_INVALID_PACKET)?;
        let consumed = remaining - new_remaining;
        pos += consumed;
        remaining = new_remaining;

        if id > 1 {
            if extensions.len() as i32 == max_extensions {
                return Err(OPUS_BUFFER_TOO_SMALL);
            }
            let data_start = ext_data_start + header_size;
            let data_end = pos;
            extensions.push(OpusExtensionData {
                id: id as i32,
                frame: curr_frame,
                data: data[data_start..data_end].to_vec(),
            });
        }
    }
    Ok(extensions)
}

/// Generate extension padding data.
///
/// If `pad` is true, the output is padded to fill `len` bytes.
/// Returns the number of bytes written, or a negative error code.
///
/// Upstream C: src/extensions.c:opus_packet_extensions_generate
pub fn opus_packet_extensions_generate(
    output: &mut [u8],
    extensions: &[OpusExtensionData],
    pad: bool,
) -> Result<usize, i32> {
    let len = output.len();
    let nb_extensions = extensions.len();

    let mut max_frame: i32 = 0;
    for ext in extensions {
        max_frame = max_frame.max(ext.frame);
        if ext.id < 2 || ext.id > 127 {
            return Err(OPUS_BAD_ARG);
        }
    }
    if max_frame >= 48 {
        return Err(OPUS_BAD_ARG);
    }

    let mut pos: usize = 0;
    let mut written: usize = 0;
    let mut curr_frame: i32 = 0;

    for frame in 0..=max_frame {
        for (i, ext) in extensions.iter().enumerate() {
            if ext.frame == frame {
                // Insert separator when needed
                if frame != curr_frame {
                    let diff = frame - curr_frame;
                    if len - pos < 2 {
                        return Err(OPUS_BUFFER_TOO_SMALL);
                    }
                    if diff == 1 {
                        output[pos] = 0x02;
                        pos += 1;
                    } else {
                        output[pos] = 0x03;
                        pos += 1;
                        output[pos] = diff as u8;
                        pos += 1;
                    }
                    curr_frame = frame;
                }
                let ext_len = ext.data.len() as i32;
                if ext.id < 32 {
                    if !(0..=1).contains(&ext_len) {
                        return Err(OPUS_BAD_ARG);
                    }
                    if len - pos < ext_len as usize + 1 {
                        return Err(OPUS_BUFFER_TOO_SMALL);
                    }
                    output[pos] = ((ext.id << 1) + ext_len) as u8;
                    pos += 1;
                    if ext_len > 0 {
                        output[pos] = ext.data[0];
                        pos += 1;
                    }
                } else {
                    let last = written == nb_extensions - 1;
                    let length_bytes = if last { 0 } else { 1 + ext_len as usize / 255 };
                    if ext_len < 0 {
                        return Err(OPUS_BAD_ARG);
                    }
                    if len - pos < 1 + length_bytes + ext_len as usize {
                        return Err(OPUS_BUFFER_TOO_SMALL);
                    }
                    output[pos] = ((ext.id << 1) + !last as i32) as u8;
                    pos += 1;
                    if !last {
                        for _ in 0..ext_len / 255 {
                            output[pos] = 255;
                            pos += 1;
                        }
                        output[pos] = (ext_len % 255) as u8;
                        pos += 1;
                    }
                    output[pos..pos + ext_len as usize]
                        .copy_from_slice(&ext.data[..ext_len as usize]);
                    pos += ext_len as usize;
                }
                written += 1;
                let _ = i; // suppress unused warning
            }
        }
    }

    // Pad with 0x01 separator bytes if requested
    if pad && pos < len {
        let padding = len - pos;
        output.copy_within(0..pos, padding);
        for b in &mut output[..padding] {
            *b = 0x01;
        }
        pos += padding;
    }

    Ok(pos)
}

/// Generate extension data, returning the required size without writing.
///
/// This is equivalent to calling generate with a None output to measure the size.
///
/// Upstream C: src/extensions.c:opus_packet_extensions_generate (with data=NULL)
pub fn opus_packet_extensions_generate_size(
    extensions: &[OpusExtensionData],
) -> Result<usize, i32> {
    let nb_extensions = extensions.len();

    let mut max_frame: i32 = 0;
    for ext in extensions {
        max_frame = max_frame.max(ext.frame);
        if ext.id < 2 || ext.id > 127 {
            return Err(OPUS_BAD_ARG);
        }
    }
    if max_frame >= 48 {
        return Err(OPUS_BAD_ARG);
    }

    let mut pos: usize = 0;
    let mut written: usize = 0;
    let mut curr_frame: i32 = 0;

    for frame in 0..=max_frame {
        for ext in extensions {
            if ext.frame == frame {
                if frame != curr_frame {
                    let diff = frame - curr_frame;
                    if diff == 1 {
                        pos += 1;
                    } else {
                        pos += 2;
                    }
                    curr_frame = frame;
                }
                let ext_len = ext.data.len();
                if ext.id < 32 {
                    pos += 1 + ext_len;
                } else {
                    let last = written == nb_extensions - 1;
                    let length_bytes = if last { 0 } else { 1 + ext_len / 255 };
                    pos += 1 + length_bytes + ext_len;
                }
                written += 1;
            }
        }
    }

    Ok(pos)
}
