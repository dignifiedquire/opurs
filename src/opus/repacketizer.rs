//! Opus packet merging and splitting.
//!
//! Upstream C: `src/repacketizer.c`

use crate::opus::extensions::OpusExtensionData;
use crate::opus::opus_defines::{
    OPUS_BAD_ARG, OPUS_BUFFER_TOO_SMALL, OPUS_INVALID_PACKET, OPUS_OK,
};
use crate::opus::packet::{encode_size, opus_packet_parse_impl};
use crate::{opus_packet_get_nb_frames, opus_packet_get_samples_per_frame};

/// The repacketizer can be used to merge multiple Opus packets into a single
/// packet or alternatively to split Opus packets that have previously been
/// merged. Splitting valid Opus packets is always guaranteed to succeed,
/// whereas merging valid packets only succeeds if all frames have the same
/// mode, bandwidth, and frame size, and when the total duration of the merged
/// packet is no more than 120 ms. The 120 ms limit comes from the
/// specification and limits decoder memory requirements at a point where
/// framing overhead becomes negligible.
///
/// The repacketizer currently only operates on elementary Opus
/// streams. It will not manipualte multistream packets successfully, except in
/// the degenerate case where they consist of data from a single stream.
///
/// The repacketizing process starts with creating a repacketizer state.
///
/// Then the application should submit packets with [`OpusRepacketizer::cat`],
/// extract new packets with `opus_repacketizer_out` or
/// `opus_repacketizer_out_range`, and then reset the state for the next set of
/// input packets via [`OpusRepacketizer::init`].
///
/// An alternate way of merging packets is to simply call [`OpusRepacketizer::cat`]
/// unconditionally until it fails. At that point, the merged packet can be
/// obtained with `opus_repacketizer_out` and the input packet for which
/// [`OpusRepacketizer::cat`] needs to be re-added to a newly reinitialized
/// repacketizer state.
#[derive(Debug, Clone)]
pub struct OpusRepacketizer {
    toc: u8,
    nb_frames: i32,
    frames: [usize; 48],
    len: [i16; 48],
    framesize: i32,
}

impl Default for OpusRepacketizer {
    fn default() -> Self {
        Self {
            toc: 0,
            nb_frames: 0,
            frames: [0; 48],
            len: [0; 48],
            framesize: 0,
        }
    }
}

impl OpusRepacketizer {
    /// (Re)initializes a previously allocated repacketizer state.
    ///
    /// It must also be called to reset the queue of packets waiting to be
    /// repacketized, which is necessary if the maximum packet duration of 120 ms
    /// is reached or if you wish to submit packets with a different Opus
    /// configuration (coding mode, audio bandwidth, frame size, or channel count).
    /// Failure to do so will prevent a new packet from being added with
    /// [`OpusRepacketizer::cat`].
    pub fn init(&mut self) {
        self.nb_frames = 0;
    }

    /// Add a packet to the current repacketizer state.
    /// This packet must match the configuration of any packets already submitted
    /// for repacketization since the last call to `OpusRepacketizer::init`.
    /// This means that it must have the same coding mode, audio bandwidth, frame
    /// size, and channel count.
    /// This can be checked in advance by examining the top 6 bits of the first
    /// byte of the packet, and ensuring they match the top 6 bits of the first
    /// byte of any previously submitted packet.
    /// The total duration of audio in the repacketizer state also must not exceed
    /// 120 ms, the maximum duration of a single packet, after adding this packet.
    ///
    /// The contents of the current repacketizer state can be extracted into new
    /// packets using `opus_repacketizer_out` or `opus_repacketizer_out_range`.
    ///
    /// In order to add a packet with a different configuration or to add more
    /// audio beyond 120 ms, you must clear the repacketizer state by calling
    /// `OpusRepacketizer::init`.
    /// If a packet is too large to add to the current repacketizer state, no part
    /// of it is added, even if it contains multiple frames, some of which might
    /// fit.
    /// If you wish to be able to add parts of such packets, you should first use
    /// another repacketizer to split the packet into pieces and add them
    /// individually.
    ///
    /// The parameter `data` contains the actual packet data.
    ///
    /// If `OPUS_INVALID_PACKET` is returned: The packet did not have a valid TOC sequence,
    /// the packet's TOC sequence was not compatible
    /// with previously submitted packets (because the coding mode, audio bandwidth, frame size,
    /// or channel count did not match), or adding this packet would increase the total amount of
    /// audio stored in the repacketizer state to more than 120 ms.
    pub fn cat(&mut self, data: &[u8]) -> i32 {
        self.cat_impl(data, false)
    }

    fn cat_impl(&mut self, data: &[u8], self_delimited: bool) -> i32 {
        // Set of check ToC
        if data.is_empty() {
            return OPUS_INVALID_PACKET;
        }
        if self.nb_frames == 0 {
            self.toc = data[0];
            self.framesize = opus_packet_get_samples_per_frame(data[0], 8000);
        } else if self.toc & 0xfc != data[0] & 0xfc {
            return OPUS_INVALID_PACKET;
        }
        let curr_nb_frames = opus_packet_get_nb_frames(data);
        if curr_nb_frames < 1 {
            return OPUS_INVALID_PACKET;
        }

        // Check the 120 ms maximum packet size
        if (curr_nb_frames + self.nb_frames) * self.framesize > 960 {
            return OPUS_INVALID_PACKET;
        }

        let mut tmp_toc: u8 = 0;
        let num_frames = opus_packet_parse_impl(
            data,
            self_delimited,
            Some(&mut tmp_toc),
            Some(&mut self.frames[self.nb_frames as usize..]),
            &mut self.len[self.nb_frames as usize..],
            None,
            None,
        );
        if num_frames < 1 {
            return num_frames;
        }

        self.nb_frames += curr_nb_frames;

        OPUS_OK
    }

    /// Return the total number of frames contained in packet data submitted to
    /// the repacketizer state so far via `OpusRepacketizer::cat` since the last
    /// call to `OpusRepacketizer::init` or creation.
    /// This defines the valid range of packets that can be extracted with
    /// `OpusRrepacketizer::out_range` or `OpusRepacketizer::out`.
    ///
    /// Returns the total number of frames contained in the packet data submitted
    /// to the repacketizer state.
    pub fn get_nb_frames(&self) -> i32 {
        self.nb_frames
    }

    /// Construct a new packet from data previously submitted to the repacketizer
    /// state via [`OpusRepacketizer::cat`].
    ///
    /// - `begin`: The index of the first frame in the current repacketizer state to include in the output.
    /// - `end`: One past the index of the last frame in the current repacketizer state to include in the output.
    /// - `data`: The buffer in which to store the output packet.
    /// - `maxlen`: The maximum number of bytes to store in the output buffer. In order to guarantee
    ///   success, this should be at least `1276` for a single frame, or for multiple frames,
    ///   `1277*(end-begin)`. However, `1*(end-begin)` plus the size of all packet data submitted to the repacketizer since the last call to
    ///   `OpusRepacketizer::init` or creation is also sufficient, and possibly much smaller.
    ///
    /// Returns the total size of the output packet on success, or an error code on failure.
    /// - `OPUS_BAD_ARG`: `[begin,end)` was an invalid range of frames (begin < 0, begin >= end, or end >
    ///   `OpusRepacketizer::get_nb_frames`).
    /// - `OPUS_BUFFER_TOO_SMALL`: `maxlen` was insufficient to contain the complete output packet.
    pub fn out_range(&mut self, begin: i32, end: i32, data: &mut [u8]) -> i32 {
        self.out_range_impl(
            begin,
            end,
            data,
            false,
            false,
            FrameSource::Data { offset: 0 },
        )
    }

    /// Construct a new packet from data previously submitted to the repacketizer
    /// state via opus_repacketizer_cat().
    /// This is a convenience routine that returns all the data submitted so far
    /// in a single packet.
    /// It is equivalent to calling `rp.out_range(rp, 0, rp.get_nb_frames(rp), data, maxlen)`.
    ///
    /// - `data`: The buffer in which to store the output packet.
    /// - `maxlen`: The maximum number of bytes to store in the output buffer. In order to guarantee
    ///   success, this should be at least `1276` for a single frame, or for multiple frames,
    ///   `1277*(end-begin)`. However, `1*(end-begin)` plus the size of all packet data submitted to the repacketizer since the last call to
    ///   `OpusRepacketizer::init` or creation is also sufficient, and possibly much smaller.
    ///
    /// Returns the total size of the output packet on success, or an error code
    ///          on failure.
    /// - `OPUS_BUFFER_TOO_SMALL`: `maxlen` was insufficient to contain the complete output packet.
    pub fn out(&mut self, data: &mut [u8]) -> i32 {
        self.out_range_impl(
            0,
            self.nb_frames,
            data,
            false,
            false,
            FrameSource::Data { offset: 0 },
        )
    }

    pub(crate) fn out_range_impl(
        &mut self,
        begin: i32,
        end: i32,
        data: &mut [u8],
        self_delimited: bool,
        pad: bool,
        frame_source: FrameSource<'_>,
    ) -> i32 {
        self.out_range_impl_ext(begin, end, data, self_delimited, pad, frame_source, &[])
    }

    /// Like `out_range_impl` but also embeds extension data into the packet padding.
    ///
    /// Upstream C: src/repacketizer.c:opus_repacketizer_out_range_impl
    pub(crate) fn out_range_impl_ext(
        &mut self,
        begin: i32,
        end: i32,
        data: &mut [u8],
        self_delimited: bool,
        pad: bool,
        frame_source: FrameSource<'_>,
        extensions: &[OpusExtensionData],
    ) -> i32 {
        use crate::opus::extensions::{
            opus_packet_extensions_generate, opus_packet_extensions_generate_size,
        };

        let ext_count = extensions.len();

        if begin < 0 || begin >= end || end > self.nb_frames {
            return OPUS_BAD_ARG;
        }

        let maxlen = data.len() as i32;
        let count = end - begin;
        let len = &self.len[begin as usize..];
        let frames = &self.frames[begin as usize..];

        let mut tot_size = 0;
        if self_delimited {
            tot_size = 1 + (len[(count - 1) as usize] >= 252) as i32;
        }

        let mut ptr = 0;
        if count == 1 {
            // Code 0
            tot_size += len[0] as i32 + 1;
            if tot_size > maxlen {
                return OPUS_BUFFER_TOO_SMALL;
            }
            data[ptr] = self.toc & 0xfc;
            ptr += 1;
        } else if count == 2 {
            if len[1] == len[0] {
                // Code 1
                tot_size += 2 * len[0] as i32 + 1;
                if tot_size > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                data[ptr] = self.toc & 0xfc | 0x1;
                ptr += 1;
            } else {
                // Code 2
                tot_size += len[0] as i32 + len[1] as i32 + 2 + (len[0] >= 252) as i32;
                if tot_size > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                data[ptr] = self.toc & 0xfc | 0x2;
                ptr += 1;
                ptr += encode_size(len[0] as i32, &mut data[ptr..]) as usize;
            }
        }
        if count > 2 || (pad && tot_size < maxlen) || ext_count > 0 {
            // Code 3
            let mut vbr: i32 = 0;
            let mut pad_amount: i32;
            let mut ext_len: i32 = 0;
            let mut ext_begin: usize = 0;
            let mut ones_begin: usize = 0;
            let mut ones_end: usize = 0;

            // Restart the process for the padding case
            ptr = 0;
            if self_delimited {
                tot_size = 1 + (len[(count - 1) as usize] >= 252) as i32;
            } else {
                tot_size = 0;
            }
            vbr = 0;
            for i in 1..count {
                if len[i as usize] != len[0] {
                    vbr = 1;
                    break;
                }
            }
            if vbr != 0 {
                tot_size += 2;
                for i in 0..count - 1 {
                    tot_size += 1 + (len[i as usize] >= 252) as i32 + len[i as usize] as i32;
                }
                tot_size += len[(count - 1) as usize] as i32;
                if tot_size > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                data[ptr] = self.toc & 0xfc | 0x3;
                ptr += 1;
                data[ptr] = (count | 0x80) as u8;
                ptr += 1;
            } else {
                tot_size += count * len[0] as i32 + 2;
                if tot_size > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                data[ptr] = (self.toc as i32 & 0xfc | 0x3) as u8;
                ptr += 1;
                data[ptr] = count as u8;
                ptr += 1;
            }
            pad_amount = if pad { maxlen - tot_size } else { 0 };
            if ext_count > 0 {
                // Figure out how much space we need for the extensions
                match opus_packet_extensions_generate_size(extensions) {
                    Ok(size) => ext_len = size as i32,
                    Err(e) => return e,
                }
                if !pad {
                    pad_amount = ext_len + ext_len / 254 + 1;
                }
            }
            if pad_amount != 0 {
                let nb_255s = (pad_amount - 1) / 255;
                data[1] |= 0x40;
                if tot_size + ext_len + nb_255s + 1 > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                ext_begin = (tot_size + pad_amount - ext_len) as usize;
                // Prepend 0x01 padding
                ones_begin = (tot_size + nb_255s + 1) as usize;
                ones_end = (tot_size + pad_amount - ext_len) as usize;
                for _i in 0..nb_255s {
                    data[ptr] = 255;
                    ptr += 1;
                }
                data[ptr] = (pad_amount - 255 * nb_255s - 1) as u8;
                ptr += 1;
                tot_size += pad_amount;
            }
            if vbr != 0 {
                for i in 0..count - 1 {
                    ptr += encode_size(len[i as usize] as i32, &mut data[ptr..]) as usize;
                }
            }

            // Copy the actual data
            if self_delimited {
                let sdlen =
                    encode_size(len[(count - 1) as usize] as i32, &mut data[ptr..]) as usize;
                ptr += sdlen;
            }
            for (i, (len, frame)) in len.iter().zip(frames).enumerate().take(count as _) {
                let len = *len as usize;
                let frame = *frame;
                match frame_source {
                    FrameSource::Data { offset } => {
                        let frame = frame + offset;
                        data.copy_within(frame..frame + len, ptr);
                    }
                    FrameSource::Slice {
                        data: ref frame_sources,
                    } => {
                        let source = frame_sources[i];
                        data[ptr..ptr + len].copy_from_slice(&source[frame..frame + len]);
                    }
                }
                ptr += len;
            }

            // Write extensions into the padding area
            if ext_len > 0 {
                let _ = opus_packet_extensions_generate(
                    &mut data[ext_begin..ext_begin + ext_len as usize],
                    extensions,
                    false,
                );
            }
            // Fill 0x01 padding bytes between header and extensions
            data[ones_begin..ones_end].fill(0x01);
            // Fill remaining padding with zeros (only when pad and no extensions)
            if pad && ext_count == 0 {
                while ptr < data.len() {
                    data[ptr] = 0;
                    ptr += 1;
                }
            }

            return tot_size;
        }
        if self_delimited {
            let sdlen = encode_size(len[(count - 1) as usize] as i32, &mut data[ptr..]) as usize;
            ptr += sdlen;
        }

        // Copy the actual data (non-Code 3 path)
        for (i, (len, frame)) in len.iter().zip(frames).enumerate().take(count as _) {
            let len = *len as usize;
            let frame = *frame;
            match frame_source {
                FrameSource::Data { offset } => {
                    // The source frames are inside the output buffer
                    let frame = frame + offset;
                    data.copy_within(frame..frame + len, ptr);
                }
                FrameSource::Slice {
                    data: ref frame_sources,
                } => {
                    // The source frames are inside the provided slices
                    let source = frame_sources[i];
                    data[ptr..ptr + len].copy_from_slice(&source[frame..frame + len]);
                }
            }
            ptr += len;
        }
        if pad {
            // Fill padding with zeros.
            data[ptr..].fill(0);
        }

        tot_size
    }
}

pub(crate) enum FrameSource<'a> {
    /// Frames are pointers directly into `data`, with a potential `offset`.
    Data { offset: usize },
    /// Frames are pointers into the given slice
    Slice { data: Vec<&'a [u8]> },
}

/// Pad a packet and optionally embed extension data.
///
/// - `data`: The buffer containing the packet to pad. Must be large enough for `new_len`.
/// - `len`: The current size of the packet.
/// - `new_len`: The desired size after padding.
/// - `pad`: If true, fill remaining space with zero padding (CBR). If false, only add
///   enough padding for the extensions.
/// - `extensions`: Extension data to embed in the packet padding.
///
/// Returns the new packet size on success, or a negative error code.
///
/// Upstream C: src/repacketizer.c:opus_packet_pad_impl
pub fn opus_packet_pad_impl(
    data: &mut [u8],
    len: i32,
    new_len: i32,
    pad: bool,
    extensions: &[OpusExtensionData],
) -> i32 {
    if len < 1 {
        return OPUS_BAD_ARG;
    }
    if len == new_len {
        return OPUS_OK;
    } else if len > new_len {
        return OPUS_BAD_ARG;
    }

    // Copy the original packet data so we can rewrite in place
    let copy = data[..len as usize].to_vec();

    let mut rp = OpusRepacketizer::default();
    let ret = rp.cat(&copy);
    if ret != OPUS_OK {
        return ret;
    }

    rp.out_range_impl_ext(
        0,
        rp.nb_frames,
        &mut data[..new_len as usize],
        false,
        pad,
        FrameSource::Slice {
            data: vec![&copy; rp.nb_frames as usize],
        },
        extensions,
    )
}

/// Pads a given Opus packet to a larger size (possibly changing the TOC sequence).
///
/// - `data`: The buffer containing the packet to pad.
/// - `len`: The size of the packet. This must be at least 1.
/// - `new_len`: The desired size of the packet after padding. This must be at least as large as len.
///
/// Returns
/// - `OPUS_OK`: on success.
/// - `OPUS_BAD_ARG`:  len was less than 1 or new_len was less than len.
/// - `OPUS_INVALID_PACKET`:  data did not contain a valid Opus packet.
pub fn opus_packet_pad(data: &mut [u8], len: i32, new_len: i32) -> i32 {
    let ret = opus_packet_pad_impl(data, len, new_len, true, &[]);
    if ret > 0 {
        OPUS_OK
    } else {
        ret
    }
}

/// Remove all padding from a given Opus packet and rewrite the TOC sequence to
/// minimize space usage.
///
/// - `data`: The buffer containing the packet to strip.
/// - `len`: The size of the packet. This must be at least 1.
///
/// Returns the new size of the output packet on success, or an error code on failure.
/// - `OPUS_BAD_ARG`: len was less than 1.
/// - `OPUS_INVALID_PACKET`: data did not contain a valid Opus packet.
pub fn opus_packet_unpad(data: &mut [u8]) -> i32 {
    if data.is_empty() {
        return OPUS_BAD_ARG;
    }
    let mut rp = OpusRepacketizer::default();
    let ret = rp.cat(data);
    if ret < 0 {
        return ret;
    }
    let ret = rp.out_range_impl(
        0,
        rp.nb_frames,
        data,
        false,
        false,
        FrameSource::Data { offset: 0 },
    );
    assert!(ret > 0 && ret <= data.len() as _);
    ret
}
