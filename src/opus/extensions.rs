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

/// Count the number of extensions (excluding real padding, separators, and
/// repeat indicators, but including repeated extensions).
///
/// Upstream C: src/extensions.c:opus_packet_extensions_count
pub fn opus_packet_extensions_count(data: &[u8], nb_frames: i32) -> Result<i32, i32> {
    let mut iter = OpusExtensionIterator::new(data, nb_frames);
    let mut count = 0;
    loop {
        match iter.next()? {
            Some(_) => count += 1,
            None => return Ok(count),
        }
    }
}

/// Count the number of extensions for each frame.
///
/// Upstream C: src/extensions.c:opus_packet_extensions_count_ext
pub fn opus_packet_extensions_count_ext(
    data: &[u8],
    nb_frame_exts: &mut [i32],
    nb_frames: i32,
) -> Result<i32, i32> {
    if nb_frames < 0 || nb_frames as usize > nb_frame_exts.len() {
        return Err(OPUS_BAD_ARG);
    }
    for item in nb_frame_exts.iter_mut().take(nb_frames as usize) {
        *item = 0;
    }

    let mut iter = OpusExtensionIterator::new(data, nb_frames);
    let mut count = 0;
    while let Some(ext) = iter.next()? {
        nb_frame_exts[ext.frame as usize] += 1;
        count += 1;
    }
    Ok(count)
}

/// Parse extensions from Opus padding data in bitstream order.
///
/// Upstream C: src/extensions.c:opus_packet_extensions_parse
pub fn opus_packet_extensions_parse(
    data: &[u8],
    max_extensions: i32,
    nb_frames: i32,
) -> Result<Vec<OpusExtensionData>, i32> {
    let mut iter = OpusExtensionIterator::new(data, nb_frames);
    let mut extensions = Vec::new();
    while let Some(ext) = iter.next()? {
        if extensions.len() as i32 == max_extensions {
            return Err(OPUS_BUFFER_TOO_SMALL);
        }
        let start = ext.data_offset;
        let end = start + ext.len;
        extensions.push(OpusExtensionData {
            id: ext.id,
            frame: ext.frame,
            data: data[start..end].to_vec(),
        });
    }
    Ok(extensions)
}

/// Parse extensions from Opus padding data in frame order.
///
/// `nb_frame_exts` must contain the output of
/// [`opus_packet_extensions_count_ext`].
///
/// Upstream C: src/extensions.c:opus_packet_extensions_parse_ext
pub fn opus_packet_extensions_parse_ext(
    data: &[u8],
    max_extensions: i32,
    nb_frame_exts: &[i32],
    nb_frames: i32,
) -> Result<Vec<OpusExtensionData>, i32> {
    if !(0..=48).contains(&nb_frames) || nb_frame_exts.len() < nb_frames as usize {
        return Err(OPUS_BAD_ARG);
    }

    let mut frame_offsets = [0i32; 49];
    let mut running = 0i32;
    for frame in 0..nb_frames as usize {
        frame_offsets[frame] = running;
        running += nb_frame_exts[frame];
    }
    frame_offsets[nb_frames as usize] = running;

    let mut out = vec![
        OpusExtensionData {
            id: 0,
            frame: 0,
            data: Vec::new(),
        };
        max_extensions.max(0) as usize
    ];

    let mut iter = OpusExtensionIterator::new(data, nb_frames);
    let mut count = 0i32;
    while let Some(ext) = iter.next()? {
        let idx = frame_offsets[ext.frame as usize];
        if idx >= max_extensions {
            return Err(OPUS_BUFFER_TOO_SMALL);
        }
        frame_offsets[ext.frame as usize] += 1;
        let start = ext.data_offset;
        let end = start + ext.len;
        out[idx as usize] = OpusExtensionData {
            id: ext.id,
            frame: ext.frame,
            data: data[start..end].to_vec(),
        };
        count += 1;
    }
    out.truncate(count as usize);
    Ok(out)
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
    nb_frames: i32,
    pad: bool,
) -> Result<usize, i32> {
    generate_extensions_internal(Some(output), extensions, nb_frames, pad)
}

struct ExtWriter<'a> {
    out: Option<&'a mut [u8]>,
    len: usize,
    pos: usize,
}

impl<'a> ExtWriter<'a> {
    /// Construct a writer over an optional output buffer.
    ///
    /// Upstream C: src/extensions.c:opus_packet_extensions_generate
    fn new(out: Option<&'a mut [u8]>) -> Self {
        let len = out.as_ref().map_or(0, |b| b.len());
        Self { out, len, pos: 0 }
    }

    /// Return the number of bytes emitted so far.
    ///
    /// Upstream C: src/extensions.c:opus_packet_extensions_generate
    fn pos(&self) -> usize {
        self.pos
    }

    /// Write a single byte and advance the output cursor.
    ///
    /// Upstream C: src/extensions.c:opus_packet_extensions_generate
    fn put(&mut self, byte: u8) -> Result<(), i32> {
        if self.out.is_some() && self.pos >= self.len {
            return Err(OPUS_BUFFER_TOO_SMALL);
        }
        if let Some(buf) = self.out.as_deref_mut() {
            buf[self.pos] = byte;
        }
        self.pos += 1;
        Ok(())
    }

    /// Write a byte slice and advance the output cursor.
    ///
    /// Upstream C: src/extensions.c:opus_packet_extensions_generate
    fn put_slice(&mut self, data: &[u8]) -> Result<(), i32> {
        if self.out.is_some() && self.len.saturating_sub(self.pos) < data.len() {
            return Err(OPUS_BUFFER_TOO_SMALL);
        }
        if let Some(buf) = self.out.as_deref_mut() {
            buf[self.pos..self.pos + data.len()].copy_from_slice(data);
        }
        self.pos += data.len();
        Ok(())
    }
}

/// Emit only the payload bytes for one extension.
///
/// Upstream C: src/extensions.c:opus_packet_extensions_generate
fn write_extension_payload(
    writer: &mut ExtWriter<'_>,
    ext: &OpusExtensionData,
    last: bool,
) -> Result<(), i32> {
    debug_assert!((EXTENSION_ID_MIN..=127).contains(&ext.id));
    if ext.id < 32 {
        if ext.data.len() > 1 {
            return Err(OPUS_BAD_ARG);
        }
        if let Some(first) = ext.data.first() {
            writer.put(*first)?;
        }
    } else {
        let mut length_bytes = 1 + ext.data.len() / 255;
        if last {
            length_bytes = 0;
        }
        if !last {
            for _ in 0..ext.data.len() / 255 {
                writer.put(255)?;
            }
            writer.put((ext.data.len() % 255) as u8)?;
        }
        if length_bytes > 0 && ext.data.is_empty() {
            // Keep behavior explicit for clippy/readability.
        }
        writer.put_slice(&ext.data)?;
    }
    Ok(())
}

/// Emit one extension header and payload.
///
/// Upstream C: src/extensions.c:opus_packet_extensions_generate
fn write_extension(
    writer: &mut ExtWriter<'_>,
    ext: &OpusExtensionData,
    last: bool,
) -> Result<(), i32> {
    debug_assert!((EXTENSION_ID_MIN..=127).contains(&ext.id));
    let header = ((ext.id << 1)
        + if ext.id < 32 {
            ext.data.len() as i32
        } else {
            !last as i32
        }) as u8;
    writer.put(header)?;
    write_extension_payload(writer, ext, last)
}

/// Internal generator used for both size-only and write modes.
///
/// Upstream C: `src/extensions.c:opus_packet_extensions_generate`.
fn generate_extensions_internal(
    out: Option<&mut [u8]>,
    extensions: &[OpusExtensionData],
    nb_frames: i32,
    pad: bool,
) -> Result<usize, i32> {
    if !(0..=48).contains(&nb_frames) {
        return Err(OPUS_BAD_ARG);
    }

    let nb_extensions = extensions.len();
    let mut writer = ExtWriter::new(out);

    let mut frame_min_idx = [0i32; 48];
    let mut frame_max_idx = [0i32; 48];
    let mut frame_repeat_idx = [0i32; 48];

    for item in frame_min_idx.iter_mut().take(nb_frames as usize) {
        *item = nb_extensions as i32;
    }
    for (idx, ext) in extensions.iter().enumerate() {
        let frame = ext.frame;
        if !(0..nb_frames).contains(&frame) {
            return Err(OPUS_BAD_ARG);
        }
        if !(EXTENSION_ID_MIN..=127).contains(&ext.id) {
            return Err(OPUS_BAD_ARG);
        }
        frame_min_idx[frame as usize] = frame_min_idx[frame as usize].min(idx as i32);
        frame_max_idx[frame as usize] = frame_max_idx[frame as usize].max(idx as i32 + 1);
    }
    frame_repeat_idx[..nb_frames as usize].copy_from_slice(&frame_min_idx[..nb_frames as usize]);

    let mut curr_frame = 0i32;
    let mut written = 0usize;

    for f in 0..nb_frames {
        let mut repeat_count = 0i32;
        let mut last_long_idx = -1i32;

        if f + 1 < nb_frames {
            let mut i = frame_min_idx[f as usize];
            while i < frame_max_idx[f as usize] {
                let ext_i = &extensions[i as usize];
                if ext_i.frame == f {
                    let mut g = f + 1;
                    while g < nb_frames {
                        if frame_repeat_idx[g as usize] >= frame_max_idx[g as usize] {
                            break;
                        }
                        let candidate = &extensions[frame_repeat_idx[g as usize] as usize];
                        debug_assert_eq!(candidate.frame, g);
                        if candidate.id != ext_i.id {
                            break;
                        }
                        if candidate.id < 32 && candidate.data.len() != ext_i.data.len() {
                            break;
                        }
                        g += 1;
                    }
                    if g < nb_frames {
                        break;
                    }
                    if ext_i.id >= 32 {
                        last_long_idx = frame_repeat_idx[(nb_frames - 1) as usize];
                    }

                    for g in f + 1..nb_frames {
                        let mut j = frame_repeat_idx[g as usize] + 1;
                        while j < frame_max_idx[g as usize] && extensions[j as usize].frame != g {
                            j += 1;
                        }
                        frame_repeat_idx[g as usize] = j;
                    }

                    repeat_count += 1;
                    frame_repeat_idx[f as usize] = i;
                }
                i += 1;
            }
        }

        let mut i = frame_min_idx[f as usize];
        while i < frame_max_idx[f as usize] {
            if extensions[i as usize].frame == f {
                if f != curr_frame {
                    let diff = f - curr_frame;
                    if diff == 1 {
                        writer.put(0x02)?;
                    } else {
                        writer.put(0x03)?;
                        writer.put(diff as u8)?;
                    }
                    curr_frame = f;
                }

                let ext = &extensions[i as usize];
                write_extension(&mut writer, ext, written == nb_extensions - 1)?;
                written += 1;

                if repeat_count > 0 && frame_repeat_idx[f as usize] == i {
                    let nb_repeated = repeat_count * (nb_frames - (f + 1));
                    let last = written + nb_repeated as usize == nb_extensions
                        || (last_long_idx < 0 && i + 1 >= frame_max_idx[f as usize]);
                    writer.put((0x04 + !last as i32) as u8)?;

                    for g in f + 1..nb_frames {
                        let mut j = frame_min_idx[g as usize];
                        while j < frame_repeat_idx[g as usize] {
                            if extensions[j as usize].frame == g {
                                let ext = &extensions[j as usize];
                                write_extension_payload(
                                    &mut writer,
                                    ext,
                                    last && j == last_long_idx,
                                )?;
                                written += 1;
                            }
                            j += 1;
                        }
                        frame_min_idx[g as usize] = j;
                    }
                    if last {
                        curr_frame += 1;
                    }
                }
            }
            i += 1;
        }
    }

    debug_assert_eq!(written, nb_extensions);
    if pad {
        if let Some(buf) = writer.out.as_deref_mut() {
            if writer.pos < buf.len() {
                let padding = buf.len() - writer.pos;
                buf.copy_within(0..writer.pos, padding);
                for byte in &mut buf[..padding] {
                    *byte = 0x01;
                }
                writer.pos += padding;
            }
        }
    }

    Ok(writer.pos())
}

/// Find the first extension with the given ID for a frame in `[0, frame_max)`.
///
/// Returns `Some(OpusExtensionData)` if found, `None` otherwise.
/// Upstream C: src/extensions.c:opus_extension_iterator_find
pub fn opus_packet_extension_find(
    data: &[u8],
    target_id: i32,
    frame_max: i32,
    nb_frames: i32,
) -> Option<OpusExtensionData> {
    let mut iter = OpusExtensionIterator::new(data, nb_frames);
    iter.set_frame_max(frame_max);
    match iter.find(target_id) {
        Ok(Some(ext)) => {
            let start = ext.data_offset;
            let end = start + ext.len;
            Some(OpusExtensionData {
                id: ext.id,
                frame: ext.frame,
                data: data[start..end].to_vec(),
            })
        }
        _ => None,
    }
}

/// Extension data referencing a slice of the original padding buffer.
///
/// Like `OpusExtensionData` but borrows the payload instead of copying it.
/// Upstream C: src/opus_private.h:opus_extension_data
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
///
/// Upstream C: src/extensions.c:skip_extension_payload
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
///
/// Upstream C: src/extensions.c:skip_extension
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
/// Upstream C: src/opus_private.h:OpusExtensionIterator
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
    /// Upstream C: src/extensions.c:opus_extension_iterator_init
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

    /// Reset iterator state to the first extension.
    ///
    /// Upstream C: src/extensions.c:opus_extension_iterator_reset
    pub fn reset(&mut self) {
        self.repeat_pos = 0;
        self.curr_pos = 0;
        self.last_long_pos = None;
        self.curr_len = self.len;
        self.repeat_frame = 0;
        self.curr_frame = 0;
        self.trailing_short_len = 0;
    }

    /// Set an exclusive maximum frame index to return extensions for.
    ///
    /// Upstream C: src/extensions.c:opus_extension_iterator_set_frame_max
    pub fn set_frame_max(&mut self, frame_max: i32) {
        self.frame_max = frame_max;
    }

    /// Return the next repeated extension, or 0 if repeat is finished,
    /// or negative on error.
    ///
    /// Upstream C: src/extensions.c:opus_extension_iterator_next_repeat
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
    /// Upstream C: src/extensions.c:opus_extension_iterator_next
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
    /// Upstream C: src/extensions.c:opus_extension_iterator_find
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
    nb_frames: i32,
) -> Result<usize, i32> {
    generate_extensions_internal(None, extensions, nb_frames, false)
}
