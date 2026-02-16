//! Opus packet extension parsing and generation.
//!
//! Extensions are carried in the padding area of Opus packets.
//! Each extension has an ID (3..127), a frame number, and variable-length data.
//! IDs 0-1 are reserved (padding/separator), ID 2 is "Repeat These Extensions".
//!
//! Upstream C: `src/extensions.c`

// These functions are used by the DNN subsystem (DRED) and QEXT.
#![allow(dead_code)]

use crate::opus::opus_defines::{OPUS_BAD_ARG, OPUS_BUFFER_TOO_SMALL, OPUS_INVALID_PACKET};

/// Extension ID 2: "Repeat These Extensions" (reserved, not yet implemented).
pub const EXTENSION_ID_REPEAT: i32 = 2;

/// Minimum valid user extension ID.
pub const EXTENSION_ID_MIN: i32 = 3;

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

/// Count the number of extensions (excluding padding, separators, and repeat markers).
///
/// Upstream C: src/extensions.c:opus_packet_extensions_count
pub fn opus_packet_extensions_count(data: &[u8], nb_frames: i32) -> Result<i32, i32> {
    let _ = nb_frames; // reserved for future repeat mechanism
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
        if id >= EXTENSION_ID_MIN as u8 {
            count += 1;
        }
    }
    Ok(count)
}

/// Parse extensions from Opus padding data.
///
/// Returns a vector of `OpusExtensionData` entries (excluding padding, separators,
/// and repeat markers).
///
/// Upstream C: src/extensions.c:opus_packet_extensions_parse
pub fn opus_packet_extensions_parse(
    data: &[u8],
    max_extensions: i32,
    nb_frames: i32,
) -> Result<Vec<OpusExtensionData>, i32> {
    let _ = nb_frames; // reserved for future repeat mechanism
    let len = data.len();
    let mut extensions = Vec::new();
    let mut curr_frame: i32 = 0;
    let mut pos: usize = 0;
    let mut remaining = len;

    while remaining > 0 {
        let id = data[pos] >> 1;

        let ext_data_start = pos;

        if id >= EXTENSION_ID_MIN as u8 {
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

        if id >= EXTENSION_ID_MIN as u8 {
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
        if ext.id < EXTENSION_ID_MIN || ext.id > 127 {
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

/// Find the first extension with the given ID for a frame in `[0, frame_max)`.
///
/// Returns `Some(OpusExtensionData)` if found, `None` otherwise.
/// This is a simplified version of the C iterator's `opus_extension_iterator_find`.
pub fn opus_packet_extension_find(
    data: &[u8],
    target_id: i32,
    frame_max: i32,
    nb_frames: i32,
) -> Option<OpusExtensionData> {
    let _ = nb_frames; // reserved for future repeat mechanism
    let len = data.len();
    let mut curr_frame: i32 = 0;
    let mut pos: usize = 0;
    let mut remaining = len;

    while remaining > 0 {
        let id = data[pos] >> 1;
        let ext_data_start = pos;

        if id == 1 {
            let l = data[pos] & 1;
            if l == 0 {
                curr_frame += 1;
            } else if remaining >= 2 {
                curr_frame += data[pos + 1] as i32;
            }
            if curr_frame >= frame_max {
                return None;
            }
        }

        let (new_remaining, header_size) = skip_extension(data, pos, pos + remaining).ok()?;
        let consumed = remaining - new_remaining;
        pos += consumed;
        remaining = new_remaining;

        if id as i32 == target_id && curr_frame < frame_max {
            let data_start = ext_data_start + header_size;
            let data_end = pos;
            return Some(OpusExtensionData {
                id: id as i32,
                frame: curr_frame,
                data: data[data_start..data_end].to_vec(),
            });
        }
    }
    None
}

/// Extension data referencing a slice of the original padding buffer.
///
/// Like `OpusExtensionData` but borrows the payload instead of copying it.
/// Upstream C: `opus_extension_data` (pointer-based).
#[derive(Clone, Debug)]
pub struct ExtensionRef {
    pub id: i32,
    pub frame: i32,
    /// Byte offset into the original padding data where this extension's
    /// payload begins.
    pub data_offset: usize,
    /// Length of the extension payload in bytes.
    pub len: usize,
}

/// Skip an extension payload (excluding the initial ID byte which the caller
/// has already read). Returns updated `(remaining_len, header_size)` or
/// negative on error.
///
/// This is the inner helper matching C `skip_extension_payload`.
fn skip_extension_payload(
    data: &[u8],
    mut pos: usize,
    mut len: i32,
    id_byte: u8,
    trailing_short_len: i32,
) -> Result<(usize, i32, i32), i32> {
    let id = id_byte >> 1;
    let l = id_byte & 1;
    let mut header_size: i32 = 0;

    if (id == 0 && l == 1) || id == 2 {
        // Padding byte or RTE indicator — nothing to skip
    } else if id > 0 && id < 32 {
        if len < l as i32 {
            return Err(-1);
        }
        pos += l as usize;
        len -= l as i32;
    } else {
        // Long extension (id >= 32)
        if l == 0 {
            if len < trailing_short_len {
                return Err(-1);
            }
            pos += (len - trailing_short_len) as usize;
            len = trailing_short_len;
        } else {
            let mut bytes: i32 = 0;
            loop {
                if len < 1 {
                    return Err(-1);
                }
                let lacing = data[pos] as i32;
                pos += 1;
                bytes += lacing;
                header_size += 1;
                len -= lacing + 1;
                if lacing != 255 {
                    break;
                }
            }
            if len < 0 {
                return Err(-1);
            }
            pos += bytes as usize;
        }
    }
    Ok((pos, len, header_size))
}

/// Skip a full extension (ID byte + payload). Returns
/// `(new_pos, remaining_len, header_size)` or negative on error.
///
/// Matches C `skip_extension` (the outer one that includes the ID byte).
fn skip_extension_full(data: &[u8], pos: usize, len: i32) -> Result<(usize, i32, i32), i32> {
    if len == 0 {
        return Ok((pos, 0, 0));
    }
    if len < 1 {
        return Err(-1);
    }
    let id_byte = data[pos];
    let new_pos = pos + 1;
    let new_len = len - 1;
    let (final_pos, remaining, mut header_size) =
        skip_extension_payload(data, new_pos, new_len, id_byte, 0)?;
    header_size += 1; // account for the ID byte
    Ok((final_pos, remaining, header_size))
}

/// Stateful iterator over extensions in Opus packet padding.
///
/// Supports the "Repeat These Extensions" (RTE, ID=2) mechanism for
/// multi-frame packets. Uses byte offsets into the original data slice
/// rather than raw pointers.
///
/// Upstream C: `OpusExtensionIterator` in `opus_private.h`
#[derive(Clone)]
pub struct OpusExtensionIterator<'a> {
    data: &'a [u8],
    /// Current read position (byte offset into `data`).
    curr_pos: usize,
    /// Remaining bytes from curr_pos.
    curr_len: i32,
    /// Total length of `data`.
    len: i32,
    /// Start of the region that will be repeated by RTE.
    repeat_pos: usize,
    /// Position of the last long extension seen (for L=0 fixup during repeat).
    last_long_pos: Option<usize>,
    /// Source position for repeat iteration.
    src_pos: usize,
    /// Remaining bytes in the repeat source.
    src_len: i32,
    /// Trailing short extension payload bytes after last long extension.
    trailing_short_len: i32,
    /// Total number of frames in the packet.
    nb_frames: i32,
    /// Maximum frame index to return (exclusive).
    frame_max: i32,
    /// Current frame index.
    curr_frame: i32,
    /// Frame index during repeat iteration (0 = not repeating).
    repeat_frame: i32,
    /// The L bit of the RTE extension.
    repeat_l: u8,
    /// Length of the repeat source region.
    repeat_len: i32,
}

impl<'a> OpusExtensionIterator<'a> {
    /// Create a new extension iterator over the given padding data.
    ///
    /// Upstream C: `opus_extension_iterator_init`
    pub fn new(data: &'a [u8], nb_frames: i32) -> Self {
        debug_assert!((0..=48).contains(&nb_frames));
        let len = data.len() as i32;
        OpusExtensionIterator {
            data,
            curr_pos: 0,
            curr_len: len,
            len,
            repeat_pos: 0,
            last_long_pos: None,
            src_pos: 0,
            src_len: 0,
            trailing_short_len: 0,
            nb_frames,
            frame_max: nb_frames,
            curr_frame: 0,
            repeat_frame: 0,
            repeat_l: 0,
            repeat_len: 0,
        }
    }

    /// Return the next repeated extension, or 0 if repeat is finished,
    /// or negative on error.
    ///
    /// Upstream C: `opus_extension_iterator_next_repeat`
    fn next_repeat(&mut self) -> Result<Option<ExtensionRef>, i32> {
        debug_assert!(self.repeat_frame > 0);
        while self.repeat_frame < self.nb_frames {
            while self.src_len > 0 {
                let repeat_id_byte_raw = self.data[self.src_pos];
                let mut repeat_id_byte = repeat_id_byte_raw;

                // Skip in src
                let (new_src_pos, new_src_len, _header_size) =
                    skip_extension_full(self.data, self.src_pos, self.src_len)?;
                self.src_pos = new_src_pos;
                self.src_len = new_src_len;

                // Don't repeat padding or frame separators with a 0 increment
                if repeat_id_byte <= 3 {
                    continue;
                }

                // If RTE had L==0 and this is the last repeated long extension
                // in the last frame, force L=0.
                if self.repeat_l == 0
                    && self.repeat_frame + 1 >= self.nb_frames
                    && self.last_long_pos == Some(self.src_pos)
                {
                    repeat_id_byte &= !1;
                }

                let curr_data0 = self.curr_pos;
                let (new_pos, new_len, header_size) = skip_extension_payload(
                    self.data,
                    self.curr_pos,
                    self.curr_len,
                    repeat_id_byte,
                    self.trailing_short_len,
                )?;
                self.curr_pos = new_pos;
                self.curr_len = new_len;

                // Skip extensions for frames past frame_max
                if self.repeat_frame >= self.frame_max {
                    continue;
                }

                let ext_data_start = curr_data0 + header_size as usize;
                let ext_data_len = self.curr_pos - ext_data_start;
                return Ok(Some(ExtensionRef {
                    id: (repeat_id_byte >> 1) as i32,
                    frame: self.repeat_frame,
                    data_offset: ext_data_start,
                    len: ext_data_len,
                }));
            }
            // Finished repeating extensions for this frame
            self.src_pos = self.repeat_pos;
            self.src_len = self.repeat_len;
            self.repeat_frame += 1;
        }

        // Finished repeating entirely
        self.repeat_pos = self.curr_pos;
        self.last_long_pos = None;

        // If L==0, advance frame for unfinished L=0 long extension
        if self.repeat_l == 0 {
            self.curr_frame += 1;
            if self.curr_frame >= self.nb_frames {
                self.curr_len = 0;
            }
        }
        self.repeat_frame = 0;
        Ok(None)
    }

    /// Return the next extension.
    ///
    /// Returns `Ok(Some(ext))` for each extension found,
    /// `Ok(None)` when iteration is complete,
    /// or `Err(OPUS_INVALID_PACKET)` on parse error.
    ///
    /// Upstream C: `opus_extension_iterator_next`
    pub fn next(&mut self) -> Result<Option<ExtensionRef>, i32> {
        if self.curr_len < 0 {
            return Err(OPUS_INVALID_PACKET);
        }
        // If we're in the middle of repeating extensions
        if self.repeat_frame > 0 {
            let ret = self.next_repeat()?;
            if ret.is_some() {
                return Ok(ret);
            }
        }
        // Check frame_max
        if self.curr_frame >= self.frame_max {
            return Ok(None);
        }

        while self.curr_len > 0 {
            let curr_data0 = self.curr_pos;
            let id_byte = self.data[curr_data0];
            let id = id_byte >> 1;
            let l = id_byte & 1;

            let (new_pos, new_len, header_size) =
                skip_extension_full(self.data, self.curr_pos, self.curr_len)
                    .map_err(|_| OPUS_INVALID_PACKET)?;
            self.curr_pos = new_pos;
            self.curr_len = new_len;

            if id == 1 {
                // Frame separator
                if l == 0 {
                    self.curr_frame += 1;
                } else {
                    let inc = self.data[curr_data0 + 1];
                    if inc == 0 {
                        continue;
                    }
                    self.curr_frame += inc as i32;
                }
                if self.curr_frame >= self.nb_frames {
                    self.curr_len = -1;
                    return Err(OPUS_INVALID_PACKET);
                }
                if self.curr_frame >= self.frame_max {
                    self.curr_len = 0;
                }
                self.repeat_pos = self.curr_pos;
                self.last_long_pos = None;
                self.trailing_short_len = 0;
            } else if id == 2 {
                // Repeat These Extensions
                self.repeat_l = l;
                self.repeat_frame = self.curr_frame + 1;
                self.repeat_len = (curr_data0 - self.repeat_pos) as i32;
                self.src_pos = self.repeat_pos;
                self.src_len = self.repeat_len;
                let ret = self.next_repeat()?;
                if ret.is_some() {
                    return Ok(ret);
                }
            } else if id > 2 {
                // Track last long extension position
                if id >= 32 {
                    self.last_long_pos = Some(self.curr_pos);
                    self.trailing_short_len = 0;
                } else {
                    self.trailing_short_len += l as i32;
                }

                let ext_data_start = curr_data0 + header_size as usize;
                let ext_data_len = self.curr_pos - ext_data_start;
                return Ok(Some(ExtensionRef {
                    id: id as i32,
                    frame: self.curr_frame,
                    data_offset: ext_data_start,
                    len: ext_data_len,
                }));
            }
        }
        Ok(None)
    }

    /// Find the next extension with the given ID.
    ///
    /// Upstream C: `opus_extension_iterator_find`
    pub fn find(&mut self, target_id: i32) -> Result<Option<ExtensionRef>, i32> {
        loop {
            match self.next()? {
                None => return Ok(None),
                Some(ext) if ext.id == target_id => return Ok(Some(ext)),
                Some(_) => continue,
            }
        }
    }
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
        if ext.id < EXTENSION_ID_MIN || ext.id > 127 {
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
