//! MLP layer implementations.
//!
//! Upstream C: `src/mlp.c`

use crate::opus::mlp::tansig::{sigmoid_approx, tansig_approx};

pub const WEIGHTS_SCALE: f32 = 1.0f32 / 128f32;

/// Matrix-vector multiply-accumulate: out[i] += sum_j(weights[i + j * col_stride] * x[j])
///
/// The weight matrix is stored in interleaved layout with `col_stride` between columns,
/// where rows are contiguous starting at offset `row` (stride 1 between rows).
///
/// Note: accumulates directly into `out[i]` (not via a temporary), matching the upstream C
/// accumulation order for bit-exact results.
fn gemm_accum(out: &mut [f32], weights: &[i8], n: usize, m: usize, col_stride: usize, x: &[f32]) {
    debug_assert_eq!(out.len(), n);
    debug_assert_eq!(x.len(), m);
    for i in 0..n {
        for j in 0..m {
            out[i] += weights[i + j * col_stride] as f32 * x[j];
        }
    }
}

pub enum ActivationFunction {
    Tansig,
    Sigmoid,
}

pub struct AnalysisDenseLayer {
    pub bias: &'static [i8],
    pub input_weights: &'static [i8],
    pub activation: ActivationFunction,
}

impl AnalysisDenseLayer {
    pub fn nb_inputs(&self) -> usize {
        self.input_weights.len() / self.nb_neurons()
    }

    pub fn nb_neurons(&self) -> usize {
        self.bias.len()
    }

    #[inline]
    pub fn compute(&self, out: &mut [f32], input: &[f32]) {
        let n = self.nb_neurons();
        let m = self.nb_inputs();
        assert_eq!(n, out.len());
        assert_eq!(m, input.len());

        for (out, &bias) in out.iter_mut().zip(self.bias.iter()) {
            *out = bias as f32;
        }

        gemm_accum(out, self.input_weights, n, m, n, input);

        for out in out.iter_mut() {
            *out *= WEIGHTS_SCALE;
        }

        match self.activation {
            ActivationFunction::Tansig => {
                for out in out.iter_mut() {
                    *out = tansig_approx(*out);
                }
            }
            ActivationFunction::Sigmoid => {
                for out in out.iter_mut() {
                    *out = sigmoid_approx(*out);
                }
            }
        }
    }
}

pub struct AnalysisGRULayer {
    // `bias`, `input_weights` and `recurrent_weights` are three concatenated matrices for update, reset and output components
    pub bias: &'static [i8],
    pub input_weights: &'static [i8],
    pub recurrent_weights: &'static [i8],
}

impl AnalysisGRULayer {
    pub fn nb_inputs(&self) -> usize {
        self.input_weights.len() / self.nb_neurons() / 3
    }

    pub fn nb_neurons(&self) -> usize {
        self.bias.len() / 3
    }

    #[inline]
    pub fn compute(&self, state: &mut [f32], input: &[f32]) {
        const MAX_NEURONS: usize = 32;

        let n = self.nb_neurons();
        let m = self.nb_inputs();
        assert_eq!(n, state.len());
        assert_eq!(m, input.len());

        assert!(n <= MAX_NEURONS);

        let col_stride = 3 * n;

        // Bias sub-slices (3 concatenated vectors of length n)
        let bias0 = &self.bias[..n];
        let bias1 = &self.bias[n..2 * n];
        let bias2 = &self.bias[2 * n..3 * n];

        // Weight sub-slices: each sub-matrix starts at offset 0, n, or 2*n within the interleaved layout
        let iw0 = self.input_weights;
        let iw1 = &self.input_weights[n..];
        let iw2 = &self.input_weights[2 * n..];
        let rw0 = self.recurrent_weights;
        let rw1 = &self.recurrent_weights[n..];
        let rw2 = &self.recurrent_weights[2 * n..];

        let mut z: [f32; MAX_NEURONS] = [0.; MAX_NEURONS];
        let mut r: [f32; MAX_NEURONS] = [0.; MAX_NEURONS];
        let mut h: [f32; MAX_NEURONS] = [0.; MAX_NEURONS];
        let z = &mut z[..n];
        let r = &mut r[..n];
        let h = &mut h[..n];

        /* Compute update gate. */
        for (z, &bias) in z.iter_mut().zip(bias0.iter()) {
            *z = bias as f32;
        }
        gemm_accum(z, iw0, n, m, col_stride, input);
        gemm_accum(z, rw0, n, n, col_stride, state);
        for z in z.iter_mut() {
            *z = sigmoid_approx(WEIGHTS_SCALE * *z);
        }

        /* Compute reset gate. */
        for (r, &bias) in r.iter_mut().zip(bias1.iter()) {
            *r = bias as f32;
        }
        gemm_accum(r, iw1, n, m, col_stride, input);
        gemm_accum(r, rw1, n, n, col_stride, state);
        for r in r.iter_mut() {
            *r = sigmoid_approx(WEIGHTS_SCALE * *r);
        }

        /* Compute output. */
        for (h, &bias) in h.iter_mut().zip(bias2.iter()) {
            *h = bias as f32;
        }
        let mut tmp: [f32; MAX_NEURONS] = [0.; MAX_NEURONS];
        let tmp = &mut tmp[..n];
        for ((tmp, &s), &r) in tmp.iter_mut().zip(state.iter()).zip(r.iter()) {
            *tmp = s * r;
        }
        gemm_accum(h, iw2, n, m, col_stride, input);
        gemm_accum(h, rw2, n, n, col_stride, tmp);
        for ((h, &s), &z) in h.iter_mut().zip(state.iter()).zip(z.iter()) {
            *h = z * s + (1.0 - z) * tansig_approx(WEIGHTS_SCALE * *h);
        }
        for (state, &h) in state.iter_mut().zip(h.iter()) {
            *state = h;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analysis_gru_allows_max_neurons() {
        const N: usize = 32;
        const M: usize = 1;

        let bias = Box::leak(vec![0i8; 3 * N].into_boxed_slice());
        let input_weights = Box::leak(vec![0i8; 3 * N * M].into_boxed_slice());
        let recurrent_weights = Box::leak(vec![0i8; 3 * N * N].into_boxed_slice());

        let layer = AnalysisGRULayer {
            bias,
            input_weights,
            recurrent_weights,
        };
        let mut state = vec![0.0f32; N];
        let input = vec![0.0f32; M];

        layer.compute(&mut state, &input);
        assert_eq!(state, vec![0.0; N]);
    }
}
