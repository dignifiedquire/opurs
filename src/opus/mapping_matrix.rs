//! Mapping matrix utilities used by projection/ambisonics paths.
//!
//! Upstream C: `src/mapping_matrix.c`

use crate::celt::float_cast::{float2int, FLOAT2INT16};
use crate::opus::opus_defines::OPUS_BAD_ARG;
use crate::opus::opus_private::align;

#[inline]
fn matrix_index(rows: usize, row: usize, col: usize) -> usize {
    rows * col + row
}

#[derive(Debug, Clone)]
pub struct MappingMatrix {
    rows: usize,
    cols: usize,
    gain: i32,
    // Col-wise ordering: rows*col + row.
    data: Vec<i16>,
}

impl MappingMatrix {
    /// Upstream C: src/mapping_matrix.c:mapping_matrix_get_size
    pub fn get_size(rows: i32, cols: i32) -> i32 {
        if rows < 0 || cols < 0 || rows > 255 || cols > 255 {
            return 0;
        }
        let cell_size = rows.saturating_mul(cols).saturating_mul(2);
        if cell_size > 65004 {
            return 0;
        }
        // C uses align(sizeof(MappingMatrix)) + align(rows*cols*sizeof(opus_int16)).
        // MappingMatrix stores 3 ints in the header.
        align((core::mem::size_of::<i32>() * 3) as i32) + align(cell_size)
    }

    /// Upstream C: src/mapping_matrix.c:mapping_matrix_init
    pub fn new(rows: i32, cols: i32, gain: i32, data: &[i16]) -> Result<Self, i32> {
        if Self::get_size(rows, cols) == 0 {
            return Err(OPUS_BAD_ARG);
        }
        let rows = rows as usize;
        let cols = cols as usize;
        if data.len() != rows * cols {
            return Err(OPUS_BAD_ARG);
        }
        Ok(Self {
            rows,
            cols,
            gain,
            data: data.to_vec(),
        })
    }

    pub fn from_bytes_le(rows: i32, cols: i32, gain: i32, data: &[u8]) -> Result<Self, i32> {
        if !data.len().is_multiple_of(2) {
            return Err(OPUS_BAD_ARG);
        }
        let mut values = Vec::with_capacity(data.len() / 2);
        for chunk in data.chunks_exact(2) {
            values.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
        Self::new(rows, cols, gain, &values)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn gain(&self) -> i32 {
        self.gain
    }

    pub fn data(&self) -> &[i16] {
        &self.data
    }

    pub fn data_as_bytes_le(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.data.len() * 2);
        for value in &self.data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn validate_multiply_dims(
        &self,
        input_rows: usize,
        output_rows: usize,
        frame_size: usize,
        output_row: usize,
    ) -> Result<(), i32> {
        if input_rows == 0 || output_rows == 0 || frame_size == 0 {
            return Err(OPUS_BAD_ARG);
        }
        if input_rows > self.cols || output_rows > self.rows || output_row >= self.rows {
            return Err(OPUS_BAD_ARG);
        }
        Ok(())
    }

    /// Upstream C: src/mapping_matrix.c:mapping_matrix_multiply_channel_in_float
    pub fn multiply_channel_in_float(
        &self,
        input: &[f32],
        input_rows: usize,
        output: &mut [f32],
        output_row: usize,
        output_rows: usize,
        frame_size: usize,
    ) -> Result<(), i32> {
        self.validate_multiply_dims(input_rows, output_rows, frame_size, output_row)?;
        if input.len() < input_rows * frame_size
            || output.len() <= output_rows.saturating_mul(frame_size.saturating_sub(1))
        {
            return Err(OPUS_BAD_ARG);
        }

        for i in 0..frame_size {
            let mut tmp = 0f32;
            for col in 0..input_rows {
                let coeff = self.data[matrix_index(self.rows, output_row, col)] as f32;
                tmp += coeff * input[matrix_index(input_rows, col, i)];
            }
            output[output_rows * i] = (1.0 / 32768.0) * tmp;
        }
        Ok(())
    }

    /// Upstream C: src/mapping_matrix.c:mapping_matrix_multiply_channel_out_float
    pub fn multiply_channel_out_float(
        &self,
        input: &[f32],
        input_row: usize,
        input_rows: usize,
        output: &mut [f32],
        output_rows: usize,
        frame_size: usize,
    ) -> Result<(), i32> {
        if input_row >= self.cols {
            return Err(OPUS_BAD_ARG);
        }
        self.validate_multiply_dims(input_rows, output_rows, frame_size, 0)?;
        if input.len() <= input_rows.saturating_mul(frame_size.saturating_sub(1))
            || output.len() < output_rows * frame_size
        {
            return Err(OPUS_BAD_ARG);
        }

        for i in 0..frame_size {
            let input_sample = input[input_rows * i];
            for row in 0..output_rows {
                let coeff = self.data[matrix_index(self.rows, row, input_row)] as f32;
                output[matrix_index(output_rows, row, i)] += (1.0 / 32768.0) * coeff * input_sample;
            }
        }
        Ok(())
    }

    /// Upstream C: src/mapping_matrix.c:mapping_matrix_multiply_channel_in_short
    pub fn multiply_channel_in_short(
        &self,
        input: &[i16],
        input_rows: usize,
        output: &mut [f32],
        output_row: usize,
        output_rows: usize,
        frame_size: usize,
    ) -> Result<(), i32> {
        self.validate_multiply_dims(input_rows, output_rows, frame_size, output_row)?;
        if input.len() < input_rows * frame_size
            || output.len() <= output_rows.saturating_mul(frame_size.saturating_sub(1))
        {
            return Err(OPUS_BAD_ARG);
        }

        for i in 0..frame_size {
            let mut tmp = 0f32;
            for col in 0..input_rows {
                let coeff = self.data[matrix_index(self.rows, output_row, col)] as f32;
                let sample = input[matrix_index(input_rows, col, i)] as f32;
                tmp += coeff * sample;
            }
            output[output_rows * i] = (1.0 / (32768.0 * 32768.0)) * tmp;
        }
        Ok(())
    }

    /// Upstream C: src/mapping_matrix.c:mapping_matrix_multiply_channel_out_short
    pub fn multiply_channel_out_short(
        &self,
        input: &[f32],
        input_row: usize,
        input_rows: usize,
        output: &mut [i16],
        output_rows: usize,
        frame_size: usize,
    ) -> Result<(), i32> {
        if input_row >= self.cols {
            return Err(OPUS_BAD_ARG);
        }
        self.validate_multiply_dims(input_rows, output_rows, frame_size, 0)?;
        if input.len() <= input_rows.saturating_mul(frame_size.saturating_sub(1))
            || output.len() < output_rows * frame_size
        {
            return Err(OPUS_BAD_ARG);
        }

        for i in 0..frame_size {
            let input_sample = FLOAT2INT16(input[input_rows * i]) as i32;
            for row in 0..output_rows {
                let coeff = self.data[matrix_index(self.rows, row, input_row)] as i32;
                let tmp = coeff * input_sample;
                let add = (tmp + 16384) >> 15;
                let idx = matrix_index(output_rows, row, i);
                output[idx] = output[idx].wrapping_add(add as i16);
            }
        }
        Ok(())
    }

    /// Same math as `multiply_channel_out_short`, but with pre-quantized
    /// `opus_int16`-equivalent input samples.
    pub fn multiply_channel_out_short_i16(
        &self,
        input: &[i16],
        input_row: usize,
        input_rows: usize,
        output: &mut [i16],
        output_rows: usize,
        frame_size: usize,
    ) -> Result<(), i32> {
        if input_row >= self.cols {
            return Err(OPUS_BAD_ARG);
        }
        self.validate_multiply_dims(input_rows, output_rows, frame_size, 0)?;
        if input.len() <= input_rows.saturating_mul(frame_size.saturating_sub(1))
            || output.len() < output_rows * frame_size
        {
            return Err(OPUS_BAD_ARG);
        }

        for i in 0..frame_size {
            let input_sample = input[input_rows * i] as i32;
            for row in 0..output_rows {
                let coeff = self.data[matrix_index(self.rows, row, input_row)] as i32;
                let tmp = coeff * input_sample;
                let add = (tmp + 16384) >> 15;
                let idx = matrix_index(output_rows, row, i);
                output[idx] = output[idx].wrapping_add(add as i16);
            }
        }
        Ok(())
    }

    /// Upstream C: src/mapping_matrix.c:mapping_matrix_multiply_channel_in_int24
    pub fn multiply_channel_in_int24(
        &self,
        input: &[i32],
        input_rows: usize,
        output: &mut [f32],
        output_row: usize,
        output_rows: usize,
        frame_size: usize,
    ) -> Result<(), i32> {
        self.validate_multiply_dims(input_rows, output_rows, frame_size, output_row)?;
        if input.len() < input_rows * frame_size
            || output.len() <= output_rows.saturating_mul(frame_size.saturating_sub(1))
        {
            return Err(OPUS_BAD_ARG);
        }

        for i in 0..frame_size {
            let mut tmp = 0f32;
            for col in 0..input_rows {
                let coeff = self.data[matrix_index(self.rows, output_row, col)] as f32;
                let sample = input[matrix_index(input_rows, col, i)] as f32;
                tmp += coeff * sample;
            }
            output[output_rows * i] = (1.0 / (32768.0 * 32768.0 * 256.0)) * tmp;
        }
        Ok(())
    }

    /// Upstream C: src/mapping_matrix.c:mapping_matrix_multiply_channel_out_int24
    pub fn multiply_channel_out_int24(
        &self,
        input: &[f32],
        input_row: usize,
        input_rows: usize,
        output: &mut [i32],
        output_rows: usize,
        frame_size: usize,
    ) -> Result<(), i32> {
        if input_row >= self.cols {
            return Err(OPUS_BAD_ARG);
        }
        self.validate_multiply_dims(input_rows, output_rows, frame_size, 0)?;
        if input.len() <= input_rows.saturating_mul(frame_size.saturating_sub(1))
            || output.len() < output_rows * frame_size
        {
            return Err(OPUS_BAD_ARG);
        }

        for i in 0..frame_size {
            let input_sample = float2int(32768.0 * 256.0 * input[input_rows * i]) as i64;
            for row in 0..output_rows {
                let coeff = self.data[matrix_index(self.rows, row, input_row)] as i64;
                let tmp = coeff * input_sample;
                let add = ((tmp + 16384) >> 15) as i32;
                let idx = matrix_index(output_rows, row, i);
                output[idx] = output[idx].wrapping_add(add);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_MATRIX_ROWS: i32 = 4;
    const SIMPLE_MATRIX_COLS: i32 = 3;
    const SIMPLE_MATRIX_DATA: [i16; 12] = [0, 32767, 0, 0, 32767, 0, 0, 0, 0, 0, 0, 32767];
    const SIMPLE_MATRIX_FRAME_SIZE: usize = 10;
    const SIMPLE_INPUT: [i16; 30] = [
        32767, 0, -32768, 29491, -3277, -29491, 26214, -6554, -26214, 22938, -9830, -22938, 19661,
        -13107, -19661, 16384, -16384, -16384, 13107, -19661, -13107, 9830, -22938, -9830, 6554,
        -26214, -6554, 3277, -29491, -3277,
    ];
    const SIMPLE_EXPECTED: [i16; 40] = [
        0, 32767, 0, -32768, -3277, 29491, 0, -29491, -6554, 26214, 0, -26214, -9830, 22938, 0,
        -22938, -13107, 19661, 0, -19661, -16384, 16384, 0, -16384, -19661, 13107, 0, -13107,
        -22938, 9830, 0, -9830, -26214, 6554, 0, -6554, -29491, 3277, 0, -3277,
    ];

    #[test]
    fn mapping_matrix_get_size_limits() {
        assert!(MappingMatrix::get_size(4, 3) > 0);
        assert!(MappingMatrix::get_size(0, 3) > 0);
        assert!(MappingMatrix::get_size(3, 0) > 0);
        assert_eq!(MappingMatrix::get_size(-1, 3), 0);
        assert_eq!(MappingMatrix::get_size(3, -1), 0);
        assert_eq!(MappingMatrix::get_size(256, 3), 0);
        assert_eq!(MappingMatrix::get_size(3, 256), 0);
        // 181*181*2 > 65004
        assert_eq!(MappingMatrix::get_size(181, 181), 0);
    }

    #[test]
    fn mapping_matrix_simple_in_short_matches_upstream_expected() {
        let matrix = MappingMatrix::new(
            SIMPLE_MATRIX_ROWS,
            SIMPLE_MATRIX_COLS,
            0,
            &SIMPLE_MATRIX_DATA,
        )
        .expect("matrix create");
        let mut out = vec![0f32; SIMPLE_MATRIX_ROWS as usize * SIMPLE_MATRIX_FRAME_SIZE];
        for row in 0..SIMPLE_MATRIX_ROWS as usize {
            matrix
                .multiply_channel_in_short(
                    &SIMPLE_INPUT,
                    SIMPLE_MATRIX_COLS as usize,
                    &mut out[row..],
                    row,
                    SIMPLE_MATRIX_ROWS as usize,
                    SIMPLE_MATRIX_FRAME_SIZE,
                )
                .expect("in_short");
        }

        let got = out.iter().map(|x| FLOAT2INT16(*x)).collect::<Vec<_>>();
        for (index, (&g, &e)) in got.iter().zip(SIMPLE_EXPECTED.iter()).enumerate() {
            assert!(
                (g as i32 - e as i32).abs() <= 1,
                "in_short mismatch at index {index}: got {g}, expected {e}"
            );
        }
    }

    #[test]
    fn mapping_matrix_simple_out_short_matches_upstream_expected() {
        let matrix = MappingMatrix::new(
            SIMPLE_MATRIX_ROWS,
            SIMPLE_MATRIX_COLS,
            0,
            &SIMPLE_MATRIX_DATA,
        )
        .expect("matrix create");
        let input_res = SIMPLE_INPUT
            .iter()
            .map(|&x| x as f32 / 32768.0)
            .collect::<Vec<_>>();
        let mut out = vec![0i16; SIMPLE_MATRIX_ROWS as usize * SIMPLE_MATRIX_FRAME_SIZE];
        for col in 0..SIMPLE_MATRIX_COLS as usize {
            matrix
                .multiply_channel_out_short(
                    &input_res[col..],
                    col,
                    SIMPLE_MATRIX_COLS as usize,
                    &mut out,
                    SIMPLE_MATRIX_ROWS as usize,
                    SIMPLE_MATRIX_FRAME_SIZE,
                )
                .expect("out_short");
        }

        for (index, (&g, &e)) in out.iter().zip(SIMPLE_EXPECTED.iter()).enumerate() {
            assert!(
                (g as i32 - e as i32).abs() <= 1,
                "out_short mismatch at index {index}: got {g}, expected {e}"
            );
        }
    }

    #[test]
    fn mapping_matrix_simple_float_paths_match_upstream_expected() {
        let matrix = MappingMatrix::new(
            SIMPLE_MATRIX_ROWS,
            SIMPLE_MATRIX_COLS,
            0,
            &SIMPLE_MATRIX_DATA,
        )
        .expect("matrix create");
        let input_res = SIMPLE_INPUT
            .iter()
            .map(|&x| x as f32 / 32768.0)
            .collect::<Vec<_>>();

        let mut out_in = vec![0f32; SIMPLE_MATRIX_ROWS as usize * SIMPLE_MATRIX_FRAME_SIZE];
        for row in 0..SIMPLE_MATRIX_ROWS as usize {
            matrix
                .multiply_channel_in_float(
                    &input_res,
                    SIMPLE_MATRIX_COLS as usize,
                    &mut out_in[row..],
                    row,
                    SIMPLE_MATRIX_ROWS as usize,
                    SIMPLE_MATRIX_FRAME_SIZE,
                )
                .expect("in_float");
        }
        let got_in = out_in.iter().map(|x| FLOAT2INT16(*x)).collect::<Vec<_>>();
        for (index, (&g, &e)) in got_in.iter().zip(SIMPLE_EXPECTED.iter()).enumerate() {
            assert!(
                (g as i32 - e as i32).abs() <= 1,
                "in_float mismatch at index {index}: got {g}, expected {e}"
            );
        }

        let mut out_out = vec![0f32; SIMPLE_MATRIX_ROWS as usize * SIMPLE_MATRIX_FRAME_SIZE];
        for col in 0..SIMPLE_MATRIX_COLS as usize {
            matrix
                .multiply_channel_out_float(
                    &input_res[col..],
                    col,
                    SIMPLE_MATRIX_COLS as usize,
                    &mut out_out,
                    SIMPLE_MATRIX_ROWS as usize,
                    SIMPLE_MATRIX_FRAME_SIZE,
                )
                .expect("out_float");
        }
        let got_out = out_out.iter().map(|x| FLOAT2INT16(*x)).collect::<Vec<_>>();
        for (index, (&g, &e)) in got_out.iter().zip(SIMPLE_EXPECTED.iter()).enumerate() {
            assert!(
                (g as i32 - e as i32).abs() <= 1,
                "out_float mismatch at index {index}: got {g}, expected {e}"
            );
        }
    }
}
