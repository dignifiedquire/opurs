/* Helper to serialize compiled-in DNN weight arrays to the binary blob format.
 *
 * Exposes functions to get the weight blob — both the full combined blob
 * and per-model blobs for codegen tools.
 *
 * These are used by the opurs Rust crate's codegen tools and verification tests
 * to obtain the reference weight blob without needing to compile/run
 * write_lpcnet_weights.c as a standalone binary.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <string.h>
#include "nnet.h"

#ifdef ENABLE_OSCE
extern const WeightArray lacelayers_arrays[];
extern const WeightArray nolacelayers_arrays[];
#endif

/* Count bytes needed for one WeightArray list (NULL-terminated). */
static int weight_list_blob_size(const WeightArray *list)
{
    int total = 0;
    int i = 0;
    while (list[i].name != NULL) {
        int block_size = (list[i].size + WEIGHT_BLOCK_SIZE - 1) / WEIGHT_BLOCK_SIZE * WEIGHT_BLOCK_SIZE;
        total += WEIGHT_BLOCK_SIZE + block_size;
        i++;
    }
    return total;
}

/* Count entries in one WeightArray list (NULL-terminated). */
static int weight_list_count(const WeightArray *list)
{
    int i = 0;
    while (list[i].name != NULL) i++;
    return i;
}

/* Write one WeightArray list into buf, return bytes written. */
static int write_weight_list(const WeightArray *list, unsigned char *buf)
{
    int pos = 0;
    int i = 0;

    while (list[i].name != NULL) {
        WeightHead h;
        memcpy(h.head, "DNNw", 4);
        h.version = WEIGHT_BLOB_VERSION;
        h.type = list[i].type;
        h.size = list[i].size;
        h.block_size = (h.size + WEIGHT_BLOCK_SIZE - 1) / WEIGHT_BLOCK_SIZE * WEIGHT_BLOCK_SIZE;
        memset(h.name, 0, sizeof(h.name));
        strncpy(h.name, list[i].name, sizeof(h.name));
        h.name[sizeof(h.name) - 1] = 0;

        memcpy(buf + pos, &h, WEIGHT_BLOCK_SIZE);
        pos += WEIGHT_BLOCK_SIZE;
        memcpy(buf + pos, list[i].data, h.size);
        if (h.block_size > h.size) {
            memset(buf + pos + h.size, 0, h.block_size - h.size);
        }
        pos += h.block_size;
        i++;
    }
    return pos;
}

/* Returns the total size in bytes of the serialized weight blob. */
int opus_dnn_weights_blob_size(void)
{
    int total = 0;
    total += weight_list_blob_size(pitchdnn_arrays);
    total += weight_list_blob_size(fargan_arrays);
    total += weight_list_blob_size(plcmodel_arrays);
#ifdef ENABLE_DRED
    total += weight_list_blob_size(rdovaeenc_arrays);
    total += weight_list_blob_size(rdovaedec_arrays);
#endif
#ifdef ENABLE_OSCE
    total += weight_list_blob_size(lacelayers_arrays);
    total += weight_list_blob_size(nolacelayers_arrays);
#endif
    return total;
}

/* Writes the serialized weight blob into `buf`.
 * `buf` must have at least `opus_dnn_weights_blob_size()` bytes available.
 * Returns the number of bytes written.
 */
int opus_dnn_write_weights_blob(unsigned char *buf)
{
    int pos = 0;
    pos += write_weight_list(pitchdnn_arrays, buf + pos);
    pos += write_weight_list(fargan_arrays, buf + pos);
    pos += write_weight_list(plcmodel_arrays, buf + pos);
#ifdef ENABLE_DRED
    pos += write_weight_list(rdovaeenc_arrays, buf + pos);
    pos += write_weight_list(rdovaedec_arrays, buf + pos);
#endif
#ifdef ENABLE_OSCE
    pos += write_weight_list(lacelayers_arrays, buf + pos);
    pos += write_weight_list(nolacelayers_arrays, buf + pos);
#endif
    return pos;
}

/* -----------------------------------------------------------------------
 * Per-model blob accessors for codegen tools.
 * Each model has _size, _count, and _write functions.
 * ----------------------------------------------------------------------- */

/* PitchDNN */
int opus_dnn_pitchdnn_blob_size(void)  { return weight_list_blob_size(pitchdnn_arrays); }
int opus_dnn_pitchdnn_count(void)      { return weight_list_count(pitchdnn_arrays); }
int opus_dnn_pitchdnn_write(unsigned char *buf) { return write_weight_list(pitchdnn_arrays, buf); }

/* FARGAN */
int opus_dnn_fargan_blob_size(void)    { return weight_list_blob_size(fargan_arrays); }
int opus_dnn_fargan_count(void)        { return weight_list_count(fargan_arrays); }
int opus_dnn_fargan_write(unsigned char *buf)   { return write_weight_list(fargan_arrays, buf); }

/* PLCModel */
int opus_dnn_plcmodel_blob_size(void)  { return weight_list_blob_size(plcmodel_arrays); }
int opus_dnn_plcmodel_count(void)      { return weight_list_count(plcmodel_arrays); }
int opus_dnn_plcmodel_write(unsigned char *buf) { return write_weight_list(plcmodel_arrays, buf); }

#ifdef ENABLE_DRED
/* RDOVAE Encoder */
int opus_dnn_rdovaeenc_blob_size(void) { return weight_list_blob_size(rdovaeenc_arrays); }
int opus_dnn_rdovaeenc_count(void)     { return weight_list_count(rdovaeenc_arrays); }
int opus_dnn_rdovaeenc_write(unsigned char *buf) { return write_weight_list(rdovaeenc_arrays, buf); }

/* RDOVAE Decoder */
int opus_dnn_rdovaedec_blob_size(void) { return weight_list_blob_size(rdovaedec_arrays); }
int opus_dnn_rdovaedec_count(void)     { return weight_list_count(rdovaedec_arrays); }
int opus_dnn_rdovaedec_write(unsigned char *buf) { return write_weight_list(rdovaedec_arrays, buf); }
#endif

#ifdef ENABLE_OSCE
/* LACE */
int opus_dnn_lace_blob_size(void)      { return weight_list_blob_size(lacelayers_arrays); }
int opus_dnn_lace_count(void)          { return weight_list_count(lacelayers_arrays); }
int opus_dnn_lace_write(unsigned char *buf)     { return write_weight_list(lacelayers_arrays, buf); }

/* NoLACE */
int opus_dnn_nolace_blob_size(void)    { return weight_list_blob_size(nolacelayers_arrays); }
int opus_dnn_nolace_count(void)        { return weight_list_count(nolacelayers_arrays); }
int opus_dnn_nolace_write(unsigned char *buf)   { return write_weight_list(nolacelayers_arrays, buf); }
#endif
