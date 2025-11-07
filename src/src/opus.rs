pub mod stddef_h {
    pub const NULL: i32 = 0;
}
pub use self::stddef_h::NULL;
use crate::src::opus_defines::{OPUS_BAD_ARG, OPUS_INVALID_PACKET};

/// Applies soft-clipping to bring a float signal within the [-1,1] range. If
/// the signal is already in that range, nothing is done. If there are values
/// outside of [-1,1], then the signal is clipped as smoothly as possible to
/// both fit in the range and avoid creating excessive distortion in the
/// process.
/// - `pcm`: Input PCM and modified PCM
/// - `frame_size`: Number of samples per channel to process
/// - `channels`: Number of channels
/// - `softclip_mem`: State memory for the soft clipping process (one float per channel, initialized to zero)
pub fn opus_pcm_soft_clip(
    pcm: &mut [f32],
    frame_size: usize,
    channels: usize,
    softclip_mem: &mut [f32],
) {
    if channels < 1 || frame_size < 1 {
        return;
    }

    // First thing: saturate everything to +/- 2 which is the highest level our
    // non-linearity can handle. At the point where the signal reaches +/-2,
    // the derivative will be zero anyway, so this doesn't introduce any
    // discontinuity in the derivative.
    for i in 0..frame_size * channels {
        pcm[i] = (-2.0f32).max((2.0f32).min(pcm[i]));
    }

    for c in 0..channels {
        let x = &mut pcm[c..];
        let mut a = softclip_mem[c];

        // Continue applying the non-linearity from the previous frame to avoid
        // any discontinuity.
        for i in 0..frame_size {
            if x[i * channels] * a >= 0. {
                break;
            }
            x[i * channels] = x[i * channels] + a * x[i * channels] * x[i * channels];
        }

        let mut curr = 0;
        let x0 = x[0];
        loop {
            let mut i = curr;
            while i < frame_size {
                if x[i * channels] > 1. || x[i * channels] < -1. {
                    break;
                }
                i += 1;
            }
            if i == frame_size {
                a = 0.;
                break;
            }
            let mut peak_pos = i;
            let mut end = i;
            let mut start = end;
            let mut maxval = x[i * channels].abs();

            // Look for first zero crossing before clipping
            while start > 0 && x[i * channels] * x[(start - 1) * channels] >= 0. {
                start -= 1;
            }
            // Look for first zero crossing after clipping
            while end < frame_size && x[i * channels] * x[end * channels] >= 0. {
                if x[end * channels].abs() > maxval {
                    maxval = x[end * channels].abs();
                    peak_pos = end;
                }
                end += 1;
            }
            // Detect the special case where we clip before the first zero crossing
            let special = (start == 0 && x[i * channels] * x[0] >= 0.) as i32;
            // Compute a such that maxval + a*maxval^2 = 1
            a = (maxval - 1.) / (maxval * maxval);
            // Slightly boost "a" by 2^-22. This is just enough to ensure -ffast-math
            // does not cause output values larger than +/-1, but small enough not
            // to matter even for 24-bit output.
            a += a * 2.4e-7f32;
            if x[i * channels] > 0. {
                a = -a;
            }
            // Apply soft clipping
            for i in start..end {
                x[i * channels] = x[i * channels] + a * x[i * channels] * x[i * channels];
            }

            if special != 0 && peak_pos >= 2 {
                // Add a linear ramp from the first sample to the signal peak.
                // This avoids a discontinuity at the beginning of the frame.
                let mut delta = 0.;
                let mut offset = x0 - x[0];
                delta = offset / peak_pos as f32;

                for i in curr..peak_pos {
                    offset -= delta;
                    x[i * channels] += offset;
                    x[i * channels] = (-1.0f32).max((1.0f32).min(x[i * channels]));
                }
            }
            curr = end;
            if curr == frame_size {
                break;
            }
        }
        softclip_mem[c] = a;
    }
}

pub unsafe fn encode_size(size: i32, data: *mut u8) -> i32 {
    if size < 252 {
        *data.offset(0 as isize) = size as u8;
        return 1;
    } else {
        *data.offset(0 as isize) = (252 + (size & 0x3)) as u8;
        *data.offset(1 as isize) = (size - *data.offset(0 as isize) as i32 >> 2) as u8;
        return 2;
    };
}
unsafe fn parse_size(data: *const u8, len: i32, size: *mut i16) -> i32 {
    if len < 1 {
        *size = -1 as i16;
        return -1;
    } else if (*data.offset(0 as isize) as i32) < 252 {
        *size = *data.offset(0 as isize) as i16;
        return 1;
    } else if len < 2 {
        *size = -1 as i16;
        return -1;
    } else {
        *size = (4 * *data.offset(1 as isize) as i32 + *data.offset(0 as isize) as i32) as i16;
        return 2;
    };
}

/// Gets the number of samples per frame from an Opus packet.
///
/// - `data`: The first byte of an opus packet.
/// - `fs`: Sampling rate in Hz.
///   This must be a multiple of 400, or inaccurate results will be returned.
/// Returns the number of samples per frame.
pub fn opus_packet_get_samples_per_frame(data: u8, fs: i32) -> i32 {
    if data & 0x80 != 0 {
        let audiosize = data as i32 >> 3 & 0x3;
        (fs << audiosize) / 400
    } else if data as i32 & 0x60 == 0x60 {
        if data as i32 & 0x8 != 0 {
            fs / 50
        } else {
            fs / 100
        }
    } else {
        let audiosize = data as i32 >> 3 & 0x3;
        if audiosize == 3 {
            fs * 60 / 1000
        } else {
            (fs << audiosize) / 100
        }
    }
}

pub unsafe fn opus_packet_parse_impl(
    mut data: *const u8,
    mut len: i32,
    self_delimited: i32,
    out_toc: *mut u8,
    frames: *mut *const u8,
    size: *mut i16,
    payload_offset: *mut i32,
    packet_offset: *mut i32,
) -> i32 {
    let mut i: i32 = 0;
    let mut bytes: i32 = 0;
    let mut count: i32 = 0;
    let mut cbr: i32 = 0;
    let mut ch: u8 = 0;
    let mut toc: u8 = 0;
    let mut framesize: i32 = 0;
    let mut last_size: i32 = 0;
    let mut pad: i32 = 0;
    let data0: *const u8 = data;
    if size.is_null() || len < 0 {
        return OPUS_BAD_ARG;
    }
    if len == 0 {
        return OPUS_INVALID_PACKET;
    }
    framesize = opus_packet_get_samples_per_frame(*data.offset(0), 48000);
    cbr = 0;
    let fresh0 = data;
    data = data.offset(1);
    toc = *fresh0;
    len -= 1;
    last_size = len;
    match toc as i32 & 0x3 {
        0 => {
            count = 1;
        }
        1 => {
            count = 2;
            cbr = 1;
            if self_delimited == 0 {
                if len & 0x1 != 0 {
                    return OPUS_INVALID_PACKET;
                }
                last_size = len / 2;
                *size.offset(0 as isize) = last_size as i16;
            }
        }
        2 => {
            count = 2;
            bytes = parse_size(data, len, size);
            len -= bytes;
            if (*size.offset(0 as isize) as i32) < 0 || *size.offset(0 as isize) as i32 > len {
                return OPUS_INVALID_PACKET;
            }
            data = data.offset(bytes as isize);
            last_size = len - *size.offset(0 as isize) as i32;
        }
        _ => {
            if len < 1 {
                return OPUS_INVALID_PACKET;
            }
            let fresh1 = data;
            data = data.offset(1);
            ch = *fresh1;
            count = ch as i32 & 0x3f;
            if count <= 0 || framesize * count > 5760 {
                return OPUS_INVALID_PACKET;
            }
            len -= 1;
            if ch as i32 & 0x40 != 0 {
                let mut p: i32 = 0;
                loop {
                    let mut tmp: i32 = 0;
                    if len <= 0 {
                        return OPUS_INVALID_PACKET;
                    }
                    let fresh2 = data;
                    data = data.offset(1);
                    p = *fresh2 as i32;
                    len -= 1;
                    tmp = if p == 255 { 254 } else { p };
                    len -= tmp;
                    pad += tmp;
                    if !(p == 255) {
                        break;
                    }
                }
            }
            if len < 0 {
                return OPUS_INVALID_PACKET;
            }
            cbr = (ch as i32 & 0x80 == 0) as i32;
            if cbr == 0 {
                last_size = len;
                i = 0;
                while i < count - 1 {
                    bytes = parse_size(data, len, size.offset(i as isize));
                    len -= bytes;
                    if (*size.offset(i as isize) as i32) < 0
                        || *size.offset(i as isize) as i32 > len
                    {
                        return OPUS_INVALID_PACKET;
                    }
                    data = data.offset(bytes as isize);
                    last_size -= bytes + *size.offset(i as isize) as i32;
                    i += 1;
                }
                if last_size < 0 {
                    return OPUS_INVALID_PACKET;
                }
            } else if self_delimited == 0 {
                last_size = len / count;
                if last_size * count != len {
                    return OPUS_INVALID_PACKET;
                }
                i = 0;
                while i < count - 1 {
                    *size.offset(i as isize) = last_size as i16;
                    i += 1;
                }
            }
        }
    }
    if self_delimited != 0 {
        bytes = parse_size(data, len, size.offset(count as isize).offset(-(1 as isize)));
        len -= bytes;
        if (*size.offset((count - 1) as isize) as i32) < 0
            || *size.offset((count - 1) as isize) as i32 > len
        {
            return OPUS_INVALID_PACKET;
        }
        data = data.offset(bytes as isize);
        if cbr != 0 {
            if *size.offset((count - 1) as isize) as i32 * count > len {
                return OPUS_INVALID_PACKET;
            }
            i = 0;
            while i < count - 1 {
                *size.offset(i as isize) = *size.offset((count - 1) as isize);
                i += 1;
            }
        } else if bytes + *size.offset((count - 1) as isize) as i32 > last_size {
            return OPUS_INVALID_PACKET;
        }
    } else {
        if last_size > 1275 {
            return OPUS_INVALID_PACKET;
        }
        *size.offset((count - 1) as isize) = last_size as i16;
    }
    if !payload_offset.is_null() {
        *payload_offset = data.offset_from(data0) as i64 as i32;
    }
    i = 0;
    while i < count {
        if !frames.is_null() {
            let ref mut fresh3 = *frames.offset(i as isize);
            *fresh3 = data;
        }
        data = data.offset(*size.offset(i as isize) as i32 as isize);
        i += 1;
    }
    if !packet_offset.is_null() {
        *packet_offset = pad + data.offset_from(data0) as i64 as i32;
    }
    if !out_toc.is_null() {
        *out_toc = toc;
    }
    return count;
}
pub unsafe fn opus_packet_parse(
    data: *const u8,
    len: i32,
    out_toc: *mut u8,
    frames: *mut *const u8,
    size: *mut i16,
    payload_offset: *mut i32,
) -> i32 {
    return opus_packet_parse_impl(
        data,
        len,
        0,
        out_toc,
        frames,
        size,
        payload_offset,
        NULL as *mut i32,
    );
}
