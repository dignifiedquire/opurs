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
#include "lace_data.h"
#include "nolace_data.h"

#include <string.h>
#include <math.h>

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
            window, 0);

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
            window, 0);

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
            feature_dim, frame_size, avg_pool_k, 0);

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

    compute_linear(&hLACE.lace_af1_kernel, out, input, 0);
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

    compute_generic_dense(&hLACE.lace_af1_gain, out, input, ACTIVATION_TANH, 0);
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

    compute_linear(&hLACE.lace_af1_gain, out, input, 0);
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

#endif /* ENABLE_OSCE */
