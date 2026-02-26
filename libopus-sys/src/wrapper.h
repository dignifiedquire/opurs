#include "../opus/include/opus_multistream.h"
#include "../opus/include/opus_custom.h"

#ifdef ENABLE_DEEP_PLC
int opus_dnn_weights_blob_size(void);
int opus_dnn_write_weights_blob(unsigned char *buf);
int opus_dnn_pitchdnn_blob_size(void);
int opus_dnn_pitchdnn_write(unsigned char *buf);
int opus_dnn_fargan_blob_size(void);
int opus_dnn_fargan_write(unsigned char *buf);
int opus_dnn_plcmodel_blob_size(void);
int opus_dnn_plcmodel_write(unsigned char *buf);
#endif

#ifdef ENABLE_DRED
int opus_dnn_rdovaeenc_blob_size(void);
int opus_dnn_rdovaeenc_write(unsigned char *buf);
int opus_dnn_rdovaedec_blob_size(void);
int opus_dnn_rdovaedec_write(unsigned char *buf);
#endif

#ifdef ENABLE_OSCE
int opus_dnn_lace_blob_size(void);
int opus_dnn_lace_write(unsigned char *buf);
int opus_dnn_nolace_blob_size(void);
int opus_dnn_nolace_write(unsigned char *buf);
int opus_dnn_bbwenet_blob_size(void);
int opus_dnn_bbwenet_write(unsigned char *buf);

void osce_test_libm_values(float *out);
int osce_test_adaconv(float *out, int use_nolace, int num_frames, unsigned int seed);
int osce_test_adacomb(float *out, int use_nolace, int num_frames, unsigned int seed);
int osce_test_adashape(float *out, int num_frames, unsigned int seed);
int osce_test_compute_linear(float *out, unsigned int seed);
int osce_test_dense_tanh(float *out, unsigned int seed);
int osce_test_compute_linear_gain(float *out, unsigned int seed);
int osce_test_tanh_approx(float *out, float value);
int osce_test_adashape_intermediates(float *out, unsigned int seed);
int osce_test_compute_linear_nolace_tdshape(float *out, unsigned int seed);
int osce_test_compute_linear_nolace_af2(float *out, unsigned int seed);
int osce_test_celt_pitch_xcorr(float *out, int max_pitch, unsigned int seed);
int osce_test_compute_linear_int8(float *out, unsigned int seed);
int osce_test_compute_linear_int8_arch(float *out, unsigned int seed, int arch);
int osce_test_gru_lace_fnet(float *out, unsigned int seed);
int osce_test_dense_tanh_lace_tconv(float *out, unsigned int seed);
int osce_test_compute_conv2d_3x3(float *out, unsigned int seed);
int osce_test_adacomb_intermediates(float *out, unsigned int seed);
#endif

#ifdef OPUS_X86_MAY_HAVE_SSE
int opus_select_arch(void);
float celt_inner_prod_sse(const float *x, const float *y, int N);
void dual_inner_prod_sse(const float *x, const float *y01, const float *y02, int N, float *xy1, float *xy2);
void xcorr_kernel_sse(const float *x, const float *y, float *sum, int len);
void comb_filter_const_sse(float *y, float *x, int T, int N, float g10, float g11, float g12);
float op_pvq_search_sse2(float *X, int *iy, int K, int N, int arch);
void celt_pitch_xcorr_avx2(const float *_x, const float *_y, float *xcorr, int len, int max_pitch, int arch);
#endif
