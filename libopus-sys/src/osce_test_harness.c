/* Self-contained OSCE test harness.
 *
 * Generates deterministic test inputs, runs each nndsp building block
 * (adaconv, adacomb, adashape) with the compiled-in LACE/NoLACE weights,
 * and writes the float output arrays to caller-provided buffers.
 *
 * This allows Rust tests to call these functions and compare against
 * the equivalent Rust implementation for bit-exactness. */

#include "config.h"

#ifdef ENABLE_OSCE
#include "osce.h"
#include "nndsp.h"
#include "vec.h"
#include "pitch.h"
#include "cpu_support.h"
#include "os_support.h"
#include "lace_data.h"
#include "nolace_data.h"

#include <string.h>
#include <math.h>

/* Diagnostic: return C libm cos/exp/ln results for cross-platform comparison.
 * out[0..40] = cos values for overlap window (PI * (i+0.5) / 40)
 * out[40..48] = exp values
 * out[48..56] = ln values */
void osce_test_libm_values(float *out)
{
    int i;
    /* overlap window cos values */
    for (i = 0; i < 40; i++) {
        double angle = M_PI * (i + 0.5) / 40.0;
        out[i] = (float)(0.5 + 0.5 * cos(angle));
    }
    /* exp values */
    {
        double test_exp[] = {-0.5, -1.0, -2.0, -3.5, 0.1, 0.5, 1.0, 2.0};
        for (i = 0; i < 8; i++) {
            out[40 + i] = (float)exp(test_exp[i]);
        }
    }
    /* ln values */
    {
        double test_ln[] = {0.001, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 0.000015258789};
        for (i = 0; i < 8; i++) {
            out[48 + i] = (float)log(test_ln[i]);
        }
    }
}

/* Simple deterministic PRNG (xorshift32) for reproducible test data */
static unsigned int prng_state = 12345;

static float prng_float(void) {
    prng_state ^= prng_state << 13;
    prng_state ^= prng_state >> 17;
    prng_state ^= prng_state << 5;
    /* Map to [-1, 1] */
    return ((float)(prng_state & 0xFFFF) / 32768.0f) - 1.0f;
}

static void prng_reset(unsigned int seed) {
    prng_state = seed;
}

extern const WeightArray lacelayers_arrays[];
extern const WeightArray nolacelayers_arrays[];

/* Run adaconv for num_frames with deterministic inputs.
 * out must have room for num_frames * frame_size * out_channels floats.
 * Returns 0 on success. */
int osce_test_adaconv(
    float *out,
    int use_nolace,   /* 0 = LACE af1, 1 = NoLACE af1, 2 = NoLACE af2, 3 = NoLACE af4 */
    int num_frames,
    unsigned int seed
)
{
    LACELayers hLACE;
    NOLACELayers hNoLACE;
    AdaConvState hAdaConv;

    LinearLayer *kernel_layer;
    LinearLayer *gain_layer;
    int feature_dim, frame_size, overlap_size;
    int in_channels, out_channels, kernel_size, left_padding;
    float filter_gain_a, filter_gain_b, shape_gain;
    float window[ADACONV_MAX_OVERLAP_SIZE];
    float features[512];
    float x_in[512];
    float x_out[512];
    int i_frame, i;

    if (use_nolace == 0) {
        init_lacelayers(&hLACE, lacelayers_arrays);
        kernel_layer = &hLACE.lace_af1_kernel;
        gain_layer = &hLACE.lace_af1_gain;
        feature_dim = LACE_AF1_FEATURE_DIM;
        frame_size = LACE_AF1_FRAME_SIZE;
        overlap_size = LACE_AF1_OVERLAP_SIZE;
        in_channels = LACE_AF1_IN_CHANNELS;
        out_channels = LACE_AF1_OUT_CHANNELS;
        kernel_size = LACE_AF1_KERNEL_SIZE;
        left_padding = LACE_AF1_LEFT_PADDING;
        filter_gain_a = LACE_AF1_FILTER_GAIN_A;
        filter_gain_b = LACE_AF1_FILTER_GAIN_B;
        shape_gain = LACE_AF1_SHAPE_GAIN;
    } else {
        init_nolacelayers(&hNoLACE, nolacelayers_arrays);
        if (use_nolace == 1) {
            kernel_layer = &hNoLACE.nolace_af1_kernel;
            gain_layer = &hNoLACE.nolace_af1_gain;
            feature_dim = NOLACE_AF1_FEATURE_DIM;
            frame_size = NOLACE_AF1_FRAME_SIZE;
            overlap_size = NOLACE_AF1_OVERLAP_SIZE;
            in_channels = NOLACE_AF1_IN_CHANNELS;
            out_channels = NOLACE_AF1_OUT_CHANNELS;
            kernel_size = NOLACE_AF1_KERNEL_SIZE;
            left_padding = NOLACE_AF1_LEFT_PADDING;
            filter_gain_a = NOLACE_AF1_FILTER_GAIN_A;
            filter_gain_b = NOLACE_AF1_FILTER_GAIN_B;
            shape_gain = NOLACE_AF1_SHAPE_GAIN;
        } else if (use_nolace == 2) {
            kernel_layer = &hNoLACE.nolace_af2_kernel;
            gain_layer = &hNoLACE.nolace_af2_gain;
            feature_dim = NOLACE_AF2_FEATURE_DIM;
            frame_size = NOLACE_AF2_FRAME_SIZE;
            overlap_size = NOLACE_AF2_OVERLAP_SIZE;
            in_channels = NOLACE_AF2_IN_CHANNELS;
            out_channels = NOLACE_AF2_OUT_CHANNELS;
            kernel_size = NOLACE_AF2_KERNEL_SIZE;
            left_padding = NOLACE_AF2_LEFT_PADDING;
            filter_gain_a = NOLACE_AF2_FILTER_GAIN_A;
            filter_gain_b = NOLACE_AF2_FILTER_GAIN_B;
            shape_gain = NOLACE_AF2_SHAPE_GAIN;
        } else {
            kernel_layer = &hNoLACE.nolace_af4_kernel;
            gain_layer = &hNoLACE.nolace_af4_gain;
            feature_dim = NOLACE_AF4_FEATURE_DIM;
            frame_size = NOLACE_AF4_FRAME_SIZE;
            overlap_size = NOLACE_AF4_OVERLAP_SIZE;
            in_channels = NOLACE_AF4_IN_CHANNELS;
            out_channels = NOLACE_AF4_OUT_CHANNELS;
            kernel_size = NOLACE_AF4_KERNEL_SIZE;
            left_padding = NOLACE_AF4_LEFT_PADDING;
            filter_gain_a = NOLACE_AF4_FILTER_GAIN_A;
            filter_gain_b = NOLACE_AF4_FILTER_GAIN_B;
            shape_gain = NOLACE_AF4_SHAPE_GAIN;
        }
    }

    init_adaconv_state(&hAdaConv);
    compute_overlap_window(window, overlap_size);

    prng_reset(seed);

    for (i_frame = 0; i_frame < num_frames; i_frame++) {
        for (i = 0; i < feature_dim; i++) {
            features[i] = prng_float() * 0.1f;
        }
        for (i = 0; i < frame_size * in_channels; i++) {
            x_in[i] = prng_float() * 0.5f;
        }

        adaconv_process_frame(&hAdaConv, x_out, x_in, features,
            kernel_layer, gain_layer, feature_dim,
            frame_size, overlap_size, in_channels, out_channels,
            kernel_size, left_padding,
            filter_gain_a, filter_gain_b, shape_gain,
            window, opus_select_arch());

        memcpy(out + i_frame * frame_size * out_channels,
               x_out, sizeof(float) * frame_size * out_channels);
    }

    return 0;
}

/* Run adacomb for num_frames with deterministic inputs. */
int osce_test_adacomb(
    float *out,
    int use_nolace,   /* 0 = LACE cf1 */
    int num_frames,
    unsigned int seed
)
{
    LACELayers hLACE;
    AdaCombState hAdaComb;

    LinearLayer *kernel_layer;
    LinearLayer *gain_layer;
    LinearLayer *global_gain_layer;
    int feature_dim, frame_size, overlap_size;
    int kernel_size, left_padding;
    float filter_gain_a, filter_gain_b, log_gain_limit;
    float window[ADACOMB_MAX_OVERLAP_SIZE];
    float features[512];
    float x_in[512];
    float x_out[512];
    int i_frame, i;
    int pitch_lag;

    (void)use_nolace;
    init_lacelayers(&hLACE, lacelayers_arrays);
    kernel_layer = &hLACE.lace_cf1_kernel;
    gain_layer = &hLACE.lace_cf1_gain;
    global_gain_layer = &hLACE.lace_cf1_global_gain;
    feature_dim = LACE_CF1_FEATURE_DIM;
    frame_size = LACE_CF1_FRAME_SIZE;
    overlap_size = LACE_CF1_OVERLAP_SIZE;
    kernel_size = LACE_CF1_KERNEL_SIZE;
    left_padding = LACE_CF1_LEFT_PADDING;
    filter_gain_a = LACE_CF1_FILTER_GAIN_A;
    filter_gain_b = LACE_CF1_FILTER_GAIN_B;
    log_gain_limit = LACE_CF1_LOG_GAIN_LIMIT;

    init_adacomb_state(&hAdaComb);
    compute_overlap_window(window, overlap_size);

    prng_reset(seed);

    for (i_frame = 0; i_frame < num_frames; i_frame++) {
        for (i = 0; i < feature_dim; i++) {
            features[i] = prng_float() * 0.1f;
        }
        for (i = 0; i < frame_size; i++) {
            x_in[i] = prng_float() * 0.5f;
        }
        /* pitch lag between kernel_size and 250 */
        pitch_lag = kernel_size + (int)((unsigned int)(prng_float() * 32768.0f + 32768.0f) % (250 - kernel_size));

        adacomb_process_frame(&hAdaComb, x_out, x_in, features,
            kernel_layer, gain_layer, global_gain_layer,
            pitch_lag, feature_dim, frame_size, overlap_size,
            kernel_size, left_padding,
            filter_gain_a, filter_gain_b, log_gain_limit,
            window, opus_select_arch());

        memcpy(out + i_frame * frame_size, x_out, sizeof(float) * frame_size);
    }

    return 0;
}

/* Run adashape for num_frames with deterministic inputs. */
int osce_test_adashape(
    float *out,
    int num_frames,
    unsigned int seed
)
{
    NOLACELayers hNoLACE;
    AdaShapeState hAdaShape;

    int feature_dim, frame_size, avg_pool_k;
    float features[512];
    float x_in[512];
    float x_out[512];
    int i_frame, i;

    init_nolacelayers(&hNoLACE, nolacelayers_arrays);
    feature_dim = NOLACE_TDSHAPE1_FEATURE_DIM;
    frame_size = NOLACE_TDSHAPE1_FRAME_SIZE;
    avg_pool_k = NOLACE_TDSHAPE1_AVG_POOL_K;

    init_adashape_state(&hAdaShape);

    prng_reset(seed);

    for (i_frame = 0; i_frame < num_frames; i_frame++) {
        for (i = 0; i < feature_dim; i++) {
            features[i] = prng_float() * 0.1f;
        }
        for (i = 0; i < frame_size; i++) {
            x_in[i] = prng_float() * 0.5f;
        }

        adashape_process_frame(&hAdaShape, x_out, x_in, features,
            &hNoLACE.nolace_tdshape1_alpha1_f,
            &hNoLACE.nolace_tdshape1_alpha1_t,
            &hNoLACE.nolace_tdshape1_alpha2,
            feature_dim, frame_size, avg_pool_k, opus_select_arch());

        memcpy(out + i_frame * frame_size, x_out, sizeof(float) * frame_size);
    }

    return 0;
}

/* Test compute_linear on the LACE af1_kernel layer with deterministic input.
 * This isolates the core matrix-vector multiply from the rest of adaconv.
 * out must have room for nb_outputs (16) floats. */
int osce_test_compute_linear(
    float *out,
    unsigned int seed
)
{
    LACELayers hLACE;
    float input[512];
    int i;

    init_lacelayers(&hLACE, lacelayers_arrays);

    prng_reset(seed);
    for (i = 0; i < hLACE.lace_af1_kernel.nb_inputs; i++) {
        input[i] = prng_float() * 0.1f;
    }

    compute_linear(&hLACE.lace_af1_kernel, out, input, opus_select_arch());
    return hLACE.lace_af1_kernel.nb_inputs;
}

/* Test compute_generic_dense with ACTIVATION_TANH on the LACE af1_gain layer.
 * Returns nb_inputs via return value. out gets nb_outputs floats. */
int osce_test_dense_tanh(
    float *out,
    unsigned int seed
)
{
    LACELayers hLACE;
    float input[512];
    int i;

    init_lacelayers(&hLACE, lacelayers_arrays);

    prng_reset(seed);
    for (i = 0; i < hLACE.lace_af1_gain.nb_inputs; i++) {
        input[i] = prng_float() * 0.1f;
    }

    compute_generic_dense(&hLACE.lace_af1_gain, out, input, ACTIVATION_TANH, opus_select_arch());
    return hLACE.lace_af1_gain.nb_inputs;
}

/* Test compute_linear on the LACE af1_gain layer (float weights, 128->1).
 * out gets 1 float (the raw linear output before activation). */
int osce_test_compute_linear_gain(
    float *out,
    unsigned int seed
)
{
    LACELayers hLACE;
    float input[512];
    int i;

    init_lacelayers(&hLACE, lacelayers_arrays);

    prng_reset(seed);
    for (i = 0; i < hLACE.lace_af1_gain.nb_inputs; i++) {
        input[i] = prng_float() * 0.1f;
    }

    compute_linear(&hLACE.lace_af1_gain, out, input, opus_select_arch());
    return hLACE.lace_af1_gain.nb_inputs;
}

/* Test tanh_approx on a specific value.
 * out[0] = tanh_approx(value)
 * out[1] = value (echoed back for verification) */
int osce_test_tanh_approx(
    float *out,
    float value
)
{
    out[0] = tanh_approx(value);
    out[1] = value;
    return 0;
}

/* Run 1 frame of adashape and return out_buffer (before exp) for debugging.
 * out[0..frame_size-1] = out_buffer values.
 * out[frame_size..2*frame_size-1] = final x_out values. */
int osce_test_adashape_intermediates(
    float *out,
    unsigned int seed
)
{
    NOLACELayers hNoLACE;
    AdaShapeState hAdaShape;

    int feature_dim, frame_size, avg_pool_k;
    float features[512];
    float x_in[512];
    float x_out[512];
    int i, k;
    int tenv_size;

    float in_buffer[ADASHAPE_MAX_INPUT_DIM + ADASHAPE_MAX_FRAME_SIZE];
    float out_buffer[ADASHAPE_MAX_FRAME_SIZE];
    float tmp_buffer[ADASHAPE_MAX_FRAME_SIZE];
    float mean;
    float *tenv;

    init_nolacelayers(&hNoLACE, nolacelayers_arrays);
    feature_dim = NOLACE_TDSHAPE1_FEATURE_DIM;
    frame_size = NOLACE_TDSHAPE1_FRAME_SIZE;
    avg_pool_k = NOLACE_TDSHAPE1_AVG_POOL_K;

    init_adashape_state(&hAdaShape);

    prng_reset(seed);

    /* Generate one frame of inputs */
    for (i = 0; i < feature_dim; i++) {
        features[i] = prng_float() * 0.1f;
    }
    for (i = 0; i < frame_size; i++) {
        x_in[i] = prng_float() * 0.5f;
    }

    /* Replicate adashape_process_frame logic manually */
    tenv_size = frame_size / avg_pool_k;
    tenv = in_buffer + feature_dim;
    OPUS_CLEAR(tenv, tenv_size + 1);
    OPUS_COPY(in_buffer, features, feature_dim);

    mean = 0;
    for (i = 0; i < tenv_size; i++) {
        for (k = 0; k < avg_pool_k; k++) {
            tenv[i] += fabs(x_in[i * avg_pool_k + k]);
        }
        tenv[i] = log(tenv[i] / avg_pool_k + 1.52587890625e-05f);
        mean += tenv[i];
    }
    mean /= tenv_size;
    for (i = 0; i < tenv_size; i++) {
        tenv[i] -= mean;
    }
    tenv[tenv_size] = mean;

    compute_generic_conv1d(&hNoLACE.nolace_tdshape1_alpha1_f, out_buffer,
        hAdaShape.conv_alpha1f_state, in_buffer, feature_dim, ACTIVATION_LINEAR, opus_select_arch());
    compute_generic_conv1d(&hNoLACE.nolace_tdshape1_alpha1_t, tmp_buffer,
        hAdaShape.conv_alpha1t_state, tenv, tenv_size + 1, ACTIVATION_LINEAR, opus_select_arch());

    for (i = 0; i < frame_size; i++) {
        float tmp = out_buffer[i] + tmp_buffer[i];
        in_buffer[i] = tmp >= 0 ? tmp : 0.2 * tmp;
    }

    compute_generic_conv1d(&hNoLACE.nolace_tdshape1_alpha2, out_buffer,
        hAdaShape.conv_alpha2_state, in_buffer, frame_size, ACTIVATION_LINEAR, opus_select_arch());

    /* Copy out_buffer to first half of output */
    memcpy(out, out_buffer, sizeof(float) * frame_size);

    /* Compute final and copy to second half */
    for (i = 0; i < frame_size; i++) {
        x_out[i] = exp(out_buffer[i]) * x_in[i];
    }
    memcpy(out + frame_size, x_out, sizeof(float) * frame_size);

    return frame_size;
}

/* Test compute_linear on a NoLACE layer (tdshape1_alpha1_f).
 * This isolates whether the divergence starts at the dense layer level.
 * out must have room for nb_outputs floats.
 * Returns nb_inputs. */
int osce_test_compute_linear_nolace_tdshape(
    float *out,
    unsigned int seed
)
{
    NOLACELayers hNoLACE;
    float input[2048];
    int i;

    init_nolacelayers(&hNoLACE, nolacelayers_arrays);

    prng_reset(seed);
    for (i = 0; i < hNoLACE.nolace_tdshape1_alpha1_f.nb_inputs; i++) {
        input[i] = prng_float() * 0.1f;
    }

    compute_linear(&hNoLACE.nolace_tdshape1_alpha1_f, out, input, opus_select_arch());
    return hNoLACE.nolace_tdshape1_alpha1_f.nb_inputs;
}

/* Test compute_linear on NoLACE af2_kernel layer.
 * out must have room for nb_outputs floats.
 * Returns nb_inputs. */
int osce_test_compute_linear_nolace_af2(
    float *out,
    unsigned int seed
)
{
    NOLACELayers hNoLACE;
    float input[2048];
    int i;

    init_nolacelayers(&hNoLACE, nolacelayers_arrays);

    prng_reset(seed);
    for (i = 0; i < hNoLACE.nolace_af2_kernel.nb_inputs; i++) {
        input[i] = prng_float() * 0.1f;
    }

    compute_linear(&hNoLACE.nolace_af2_kernel, out, input, opus_select_arch());
    return hNoLACE.nolace_af2_kernel.nb_inputs;
}

/* Test celt_pitch_xcorr with deterministic inputs.
 * Mimics the exact call pattern from adaconv_process_frame:
 * kernel_size=16 (ADACONV_MAX_KERNEL_SIZE), max_pitch=overlap_size or frame_size.
 *
 * out[0..max_pitch-1] = xcorr results.
 * Returns max_pitch. */
int osce_test_celt_pitch_xcorr(
    float *out,
    int max_pitch,
    unsigned int seed
)
{
    float kernel[ADACONV_MAX_KERNEL_SIZE];
    float signal[ADACONV_MAX_FRAME_SIZE + ADACONV_MAX_KERNEL_SIZE + 4];
    int i;
    int len = ADACONV_MAX_KERNEL_SIZE;

    prng_reset(seed);
    for (i = 0; i < len; i++) {
        kernel[i] = prng_float() * 0.1f;
    }
    for (i = 0; i < max_pitch + len; i++) {
        signal[i] = prng_float() * 0.5f;
    }

    celt_pitch_xcorr(kernel, signal, out, len, max_pitch, opus_select_arch());
    return max_pitch;
}

/* Test compute_linear on the LACE fnet_conv2 layer (int8 weights, 768->128).
 * This exercises cgemv8x4 specifically. out gets nb_outputs (128) floats. */
int osce_test_compute_linear_int8(
    float *out,
    unsigned int seed
)
{
    LACELayers hLACE;
    float input[1024];
    int i;

    init_lacelayers(&hLACE, lacelayers_arrays);

    prng_reset(seed);
    for (i = 0; i < hLACE.lace_fnet_conv2.nb_inputs; i++) {
        input[i] = prng_float() * 0.1f;
    }

    compute_linear(&hLACE.lace_fnet_conv2, out, input, opus_select_arch());
    return hLACE.lace_fnet_conv2.nb_inputs;
}

/* Test compute_generic_gru on LACE fnet GRU layers (int8 weights).
 * Runs 2 GRU steps. out gets 2*LACE_COND_DIM floats (state after each step). */
int osce_test_gru_lace_fnet(
    float *out,
    unsigned int seed
)
{
    LACELayers hLACE;
    float state[128]; /* LACE_COND_DIM */
    float input[128];
    int i;

    init_lacelayers(&hLACE, lacelayers_arrays);

    /* Zero initial state */
    OPUS_CLEAR(state, 128);

    prng_reset(seed);

    /* Step 1 */
    for (i = 0; i < 128; i++) {
        input[i] = prng_float() * 0.1f;
    }
    compute_generic_gru(
        &hLACE.lace_fnet_gru_input,
        &hLACE.lace_fnet_gru_recurrent,
        state, input, opus_select_arch());
    OPUS_COPY(out, state, 128);

    /* Step 2 */
    for (i = 0; i < 128; i++) {
        input[i] = prng_float() * 0.1f;
    }
    compute_generic_gru(
        &hLACE.lace_fnet_gru_input,
        &hLACE.lace_fnet_gru_recurrent,
        state, input, opus_select_arch());
    OPUS_COPY(out + 128, state, 128);

    return 2 * 128;
}

/* Test compute_generic_dense on LACE fnet_tconv layer (int8 weights, 128->512, tanh).
 * out gets 512 floats. */
int osce_test_dense_tanh_lace_tconv(
    float *out,
    unsigned int seed
)
{
    LACELayers hLACE;
    float input[512];
    int i;

    init_lacelayers(&hLACE, lacelayers_arrays);

    prng_reset(seed);
    for (i = 0; i < hLACE.lace_fnet_tconv.nb_inputs; i++) {
        input[i] = prng_float() * 0.1f;
    }

    compute_generic_dense(&hLACE.lace_fnet_tconv, out, input, ACTIVATION_TANH, opus_select_arch());
    return hLACE.lace_fnet_tconv.nb_inputs;
}

/* Dump adacomb intermediates for one frame, including xcorr output.
 *
 * Replicates the logic of adacomb_process_frame step-by-step so we can
 * capture intermediates that are normally local variables.
 *
 * Layout of out buffer (total 452 floats):
 *   [0..15]     = kernel_buffer (16 floats from compute_generic_dense)
 *   [16]        = gain (after RELU)
 *   [17]        = global_gain (after TANH)
 *   [18]        = gain after exp transform
 *   [19]        = global_gain after exp transform
 *   [20..35]    = scaled kernel
 *   [36..115]   = xcorr output_buffer (frame_size=80, after celt_pitch_xcorr, before overlap-add)
 *   [116..155]  = window (overlap_size=40)
 *   [156..235]  = x_in (frame_size=80)
 *   [236..315]  = overlap-add result (frame_size=80, after all three overlap loops)
 *   [316..395]  = full adacomb_process_frame output (frame_size=80, for cross-check)
 *   [396]       = pitch_lag
 *   [397]       = last_global_gain (=0 for frame 0)
 */
int osce_test_adacomb_intermediates(
    float *out,
    unsigned int seed
)
{
    LACELayers hLACE;
    AdaCombState hAdaComb;

    int feature_dim, frame_size, overlap_size;
    int kernel_size, left_padding;
    float filter_gain_a, filter_gain_b, log_gain_limit;
    float window[ADACOMB_MAX_OVERLAP_SIZE];
    float features[512];
    float x_in[512];
    float x_out[512];
    int i;
    int pitch_lag;

    float kernel_buffer[ADACOMB_MAX_KERNEL_SIZE];
    float gain, global_gain;

    /* For manual adacomb replication */
    float input_buffer[ADACOMB_MAX_FRAME_SIZE + ADACOMB_MAX_LAG + ADACOMB_MAX_KERNEL_SIZE];
    float output_buffer[ADACOMB_MAX_FRAME_SIZE];
    float kernel_padded[ADACOMB_MAX_KERNEL_SIZE];
    float *p_input;

    init_lacelayers(&hLACE, lacelayers_arrays);
    feature_dim = LACE_CF1_FEATURE_DIM;
    frame_size = LACE_CF1_FRAME_SIZE;
    overlap_size = LACE_CF1_OVERLAP_SIZE;
    kernel_size = LACE_CF1_KERNEL_SIZE;
    left_padding = LACE_CF1_LEFT_PADDING;
    filter_gain_a = LACE_CF1_FILTER_GAIN_A;
    filter_gain_b = LACE_CF1_FILTER_GAIN_B;
    log_gain_limit = LACE_CF1_LOG_GAIN_LIMIT;

    init_adacomb_state(&hAdaComb);
    compute_overlap_window(window, overlap_size);

    prng_reset(seed);

    /* Generate one frame of inputs */
    for (i = 0; i < feature_dim; i++) {
        features[i] = prng_float() * 0.1f;
    }
    for (i = 0; i < frame_size; i++) {
        x_in[i] = prng_float() * 0.5f;
    }
    pitch_lag = kernel_size + (int)((unsigned int)(prng_float() * 32768.0f + 32768.0f) % (250 - kernel_size));

    /* Step 1: compute_generic_dense on kernel */
    compute_generic_dense(&hLACE.lace_cf1_kernel, kernel_buffer, features, ACTIVATION_LINEAR, opus_select_arch());
    memcpy(out, kernel_buffer, sizeof(float) * 16);

    /* Step 2: gain (RELU) */
    compute_generic_dense(&hLACE.lace_cf1_gain, &gain, features, ACTIVATION_RELU, opus_select_arch());
    out[16] = gain;

    /* Step 3: global_gain (TANH) */
    compute_generic_dense(&hLACE.lace_cf1_global_gain, &global_gain, features, ACTIVATION_TANH, opus_select_arch());
    out[17] = global_gain;

    /* Step 4: transform gains */
    gain = exp(log_gain_limit - gain);
    global_gain = exp(filter_gain_a * global_gain + filter_gain_b);
    out[18] = gain;
    out[19] = global_gain;

    /* Step 5: scale_kernel (inlined since static in nndsp.c) */
    {
        float norm = 0;
        int k;
        for (k = 0; k < kernel_size; k++) {
            norm += kernel_buffer[k] * kernel_buffer[k];
        }
        norm = (float)(1.0 / (1e-6 + sqrt(norm)));
        for (k = 0; k < kernel_size; k++) {
            kernel_buffer[k] *= norm * gain;
        }
    }
    memcpy(out + 20, kernel_buffer, sizeof(float) * 16);

    /* Step 6: Replicate adacomb_process_frame internals to capture xcorr output.
     *
     * For frame 0: last_kernel = zeros, last_global_gain = 0, last_pitch_lag = 0
     * So output_buffer_last contribution is zero.
     */
    OPUS_CLEAR(input_buffer, ADACOMB_MAX_FRAME_SIZE + ADACOMB_MAX_LAG + ADACOMB_MAX_KERNEL_SIZE);
    OPUS_COPY(input_buffer, hAdaComb.history, kernel_size + ADACOMB_MAX_LAG);
    OPUS_COPY(input_buffer + kernel_size + ADACOMB_MAX_LAG, x_in, frame_size);
    p_input = input_buffer + kernel_size + ADACOMB_MAX_LAG;

    OPUS_CLEAR(kernel_padded, ADACOMB_MAX_KERNEL_SIZE);
    OPUS_COPY(kernel_padded, kernel_buffer, kernel_size);

    OPUS_CLEAR(output_buffer, ADACOMB_MAX_FRAME_SIZE);
    celt_pitch_xcorr(kernel_padded, &p_input[-left_padding - pitch_lag],
                     output_buffer, ADACOMB_MAX_KERNEL_SIZE, frame_size, opus_select_arch());

    /* Dump xcorr output (before overlap-add) */
    memcpy(out + 36, output_buffer, sizeof(float) * frame_size);

    /* Dump window */
    memcpy(out + 116, window, sizeof(float) * overlap_size);

    /* Dump x_in */
    memcpy(out + 156, x_in, sizeof(float) * frame_size);

    /* Step 7: Overlap-add (same logic as adacomb_process_frame).
     * For frame 0: last_global_gain=0, so first term vanishes. */
    for (i = 0; i < overlap_size; i++) {
        output_buffer[i] = hAdaComb.last_global_gain * window[i] * 0.0f
            + global_gain * (1.0f - window[i]) * output_buffer[i];
    }
    for (i = 0; i < overlap_size; i++) {
        output_buffer[i] += (window[i] * hAdaComb.last_global_gain
            + (1.0f - window[i]) * global_gain) * p_input[i];
    }
    for (i = overlap_size; i < frame_size; i++) {
        output_buffer[i] = global_gain * (output_buffer[i] + p_input[i]);
    }

    /* Dump overlap-add result */
    memcpy(out + 236, output_buffer, sizeof(float) * frame_size);

    /* Step 8: Run actual adacomb_process_frame for cross-check */
    init_adacomb_state(&hAdaComb);
    prng_reset(seed);
    for (i = 0; i < feature_dim; i++) {
        features[i] = prng_float() * 0.1f;
    }
    for (i = 0; i < frame_size; i++) {
        x_in[i] = prng_float() * 0.5f;
    }
    pitch_lag = kernel_size + (int)((unsigned int)(prng_float() * 32768.0f + 32768.0f) % (250 - kernel_size));

    adacomb_process_frame(&hAdaComb, x_out, x_in, features,
        &hLACE.lace_cf1_kernel, &hLACE.lace_cf1_gain, &hLACE.lace_cf1_global_gain,
        pitch_lag, feature_dim, frame_size, overlap_size,
        kernel_size, left_padding,
        filter_gain_a, filter_gain_b, log_gain_limit,
        window, opus_select_arch());

    memcpy(out + 316, x_out, sizeof(float) * frame_size);

    out[396] = (float)pitch_lag;
    out[397] = 0.0f; /* last_global_gain for frame 0 */

    return 0;
}

#endif /* ENABLE_OSCE */
