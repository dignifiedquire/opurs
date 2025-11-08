use crate::externs::memmove;

use crate::src::opus::{encode_size, opus_packet_parse_impl};
use crate::src::opus_defines::{OPUS_BAD_ARG, OPUS_BUFFER_TOO_SMALL, OPUS_INVALID_PACKET, OPUS_OK};
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
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct OpusRepacketizer {
    pub(crate) toc: u8,
    pub(crate) nb_frames: i32,
    pub(crate) frames: [*const u8; 48],
    pub(crate) len: [i16; 48],
    pub(crate) framesize: i32,
}

impl Default for OpusRepacketizer {
    fn default() -> Self {
        Self {
            toc: 0,
            nb_frames: 0,
            frames: [std::ptr::null(); 48],
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
    /// []`OpusRepacketizer::cat`].
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
        } else if self.toc as i32 & 0xfc != data[0] as i32 & 0xfc {
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
        let ret = opus_packet_parse_impl(
            data,
            data.len() as _,
            self_delimited,
            Some(&mut tmp_toc),
            Some(&mut self.frames[self.nb_frames as usize..]),
            &mut self.len[self.nb_frames as usize..],
            None,
            None,
        );
        if ret < 1 {
            return ret;
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
    pub unsafe fn out_range(&mut self, begin: i32, end: i32, data: *mut u8, maxlen: i32) -> i32 {
        self.out_range_impl(begin, end, data, maxlen, 0, 0)
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
    pub unsafe fn out(&mut self, data: *mut u8, maxlen: i32) -> i32 {
        self.out_range_impl(0, self.nb_frames, data, maxlen, 0, 0)
    }

    pub(crate) unsafe fn out_range_impl(
        &mut self,
        begin: i32,
        end: i32,
        data: *mut u8,
        maxlen: i32,
        self_delimited: i32,
        pad: i32,
    ) -> i32 {
        let mut i: i32 = 0;
        let mut count: i32 = 0;
        let mut tot_size: i32 = 0;
        let mut len: *mut i16 = 0 as *mut i16;
        let mut frames: *mut *const u8 = 0 as *mut *const u8;
        let mut ptr: *mut u8 = 0 as *mut u8;
        if begin < 0 || begin >= end || end > self.nb_frames {
            return OPUS_BAD_ARG;
        }
        count = end - begin;
        len = (self.len).as_mut_ptr().offset(begin as isize);
        frames = (self.frames).as_mut_ptr().offset(begin as isize);
        if self_delimited != 0 {
            tot_size = 1 + (*len.offset((count - 1) as isize) as i32 >= 252) as i32;
        } else {
            tot_size = 0;
        }
        ptr = data;
        if count == 1 {
            tot_size += *len.offset(0 as isize) as i32 + 1;
            if tot_size > maxlen {
                return OPUS_BUFFER_TOO_SMALL;
            }
            let fresh0 = ptr;
            ptr = ptr.offset(1);
            *fresh0 = (self.toc as i32 & 0xfc) as u8;
        } else if count == 2 {
            if *len.offset(1 as isize) as i32 == *len.offset(0 as isize) as i32 {
                tot_size += 2 * *len.offset(0 as isize) as i32 + 1;
                if tot_size > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                let fresh1 = ptr;
                ptr = ptr.offset(1);
                *fresh1 = (self.toc as i32 & 0xfc | 0x1) as u8;
            } else {
                tot_size += *len.offset(0 as isize) as i32
                    + *len.offset(1 as isize) as i32
                    + 2
                    + (*len.offset(0 as isize) as i32 >= 252) as i32;
                if tot_size > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                let fresh2 = ptr;
                ptr = ptr.offset(1);
                *fresh2 = (self.toc as i32 & 0xfc | 0x2) as u8;
                ptr = ptr.offset(encode_size(*len.offset(0 as isize) as i32, ptr) as isize);
            }
        }
        if count > 2 || pad != 0 && tot_size < maxlen {
            let mut vbr: i32 = 0;
            let mut pad_amount: i32 = 0;
            ptr = data;
            if self_delimited != 0 {
                tot_size = 1 + (*len.offset((count - 1) as isize) as i32 >= 252) as i32;
            } else {
                tot_size = 0;
            }
            vbr = 0;
            i = 1;
            while i < count {
                if *len.offset(i as isize) as i32 != *len.offset(0 as isize) as i32 {
                    vbr = 1;
                    break;
                } else {
                    i += 1;
                }
            }
            if vbr != 0 {
                tot_size += 2;
                i = 0;
                while i < count - 1 {
                    tot_size += 1
                        + (*len.offset(i as isize) as i32 >= 252) as i32
                        + *len.offset(i as isize) as i32;
                    i += 1;
                }
                tot_size += *len.offset((count - 1) as isize) as i32;
                if tot_size > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                let fresh3 = ptr;
                ptr = ptr.offset(1);
                *fresh3 = (self.toc as i32 & 0xfc | 0x3) as u8;
                let fresh4 = ptr;
                ptr = ptr.offset(1);
                *fresh4 = (count | 0x80) as u8;
            } else {
                tot_size += count * *len.offset(0 as isize) as i32 + 2;
                if tot_size > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                let fresh5 = ptr;
                ptr = ptr.offset(1);
                *fresh5 = (self.toc as i32 & 0xfc | 0x3) as u8;
                let fresh6 = ptr;
                ptr = ptr.offset(1);
                *fresh6 = count as u8;
            }
            pad_amount = if pad != 0 { maxlen - tot_size } else { 0 };
            if pad_amount != 0 {
                let mut nb_255s: i32 = 0;
                let ref mut fresh7 = *data.offset(1 as isize);
                *fresh7 = (*fresh7 as i32 | 0x40) as u8;
                nb_255s = (pad_amount - 1) / 255;
                i = 0;
                while i < nb_255s {
                    let fresh8 = ptr;
                    ptr = ptr.offset(1);
                    *fresh8 = 255;
                    i += 1;
                }
                let fresh9 = ptr;
                ptr = ptr.offset(1);
                *fresh9 = (pad_amount - 255 * nb_255s - 1) as u8;
                tot_size += pad_amount;
            }
            if vbr != 0 {
                i = 0;
                while i < count - 1 {
                    ptr = ptr.offset(encode_size(*len.offset(i as isize) as i32, ptr) as isize);
                    i += 1;
                }
            }
        }
        if self_delimited != 0 {
            let sdlen: i32 = encode_size(*len.offset((count - 1) as isize) as i32, ptr);
            ptr = ptr.offset(sdlen as isize);
        }
        i = 0;
        while i < count {
            memmove(
                ptr as *mut core::ffi::c_void,
                *frames.offset(i as isize) as *const core::ffi::c_void,
                (*len.offset(i as isize) as u64)
                    .wrapping_mul(::core::mem::size_of::<u8>() as u64)
                    .wrapping_add((0 * ptr.offset_from(*frames.offset(i as isize)) as i64) as u64),
            );
            ptr = ptr.offset(*len.offset(i as isize) as i32 as isize);
            i += 1;
        }
        if pad != 0 {
            while ptr < data.offset(maxlen as isize) {
                let fresh10 = ptr;
                ptr = ptr.offset(1);
                *fresh10 = 0;
            }
        }

        tot_size
    }
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
pub unsafe fn opus_packet_pad(data: *mut u8, len: i32, new_len: i32) -> i32 {
    let mut rp = OpusRepacketizer::default();
    if len < 1 {
        return OPUS_BAD_ARG;
    }
    if len == new_len {
        return OPUS_OK;
    } else {
        if len > new_len {
            return OPUS_BAD_ARG;
        }
    }
    // Moving payload to the end of the packet so we can do in-place padding
    memmove(
        data.offset(new_len as isize).offset(-(len as isize)) as *mut core::ffi::c_void,
        data as *const core::ffi::c_void,
        (len as u64)
            .wrapping_mul(::core::mem::size_of::<u8>() as u64)
            .wrapping_add(
                (0 * data
                    .offset(new_len as isize)
                    .offset(-(len as isize))
                    .offset_from(data) as i64) as u64,
            ),
    );
    let ret = rp.cat(std::slice::from_raw_parts(
        data.offset(new_len as isize).offset(-(len as isize)),
        len as _,
    ));
    if ret != OPUS_OK {
        return ret;
    }
    let ret = rp.out_range_impl(0, rp.nb_frames, data, new_len, 0, 1);
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
pub unsafe fn opus_packet_unpad(data: *mut u8, len: i32) -> i32 {
    if len < 1 {
        return OPUS_BAD_ARG;
    }
    let mut rp = OpusRepacketizer::default();
    let mut ret = rp.cat(std::slice::from_raw_parts(data, len as _));
    if ret < 0 {
        return ret;
    }
    ret = rp.out_range_impl(0, rp.nb_frames, data, len, 0, 0);
    assert!(ret > 0 && ret <= len);
    ret
}
