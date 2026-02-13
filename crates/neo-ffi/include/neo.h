/* neo.h — C API for the NEO audio format
 *
 * Generated for NEO v0.1.0
 * License: Unlicense (public domain)
 *
 * Usage:
 *   #include "neo.h"
 *   NeoHandle *h = neo_open("track.neo");
 *   NeoFileInfo info;
 *   neo_get_info(h, &info);
 *   printf("Stems: %d\n", info.stem_count);
 *   neo_close(h);
 */

#ifndef NEO_H
#define NEO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a NEO file reader. */
typedef struct NeoHandle NeoHandle;

/* Result codes. */
enum NeoResult {
    NEO_OK = 0,
    NEO_NULL_POINTER = -1,
    NEO_IO_ERROR = -2,
    NEO_FORMAT_ERROR = -3,
    NEO_STEM_NOT_FOUND = -4,
    NEO_UTF8_ERROR = -5,
    NEO_INTERNAL_ERROR = -99,
};

/* Stem information. */
typedef struct {
    uint8_t  stem_id;
    uint8_t  codec_id;
    uint8_t  channels;
    uint32_t sample_rate;
    uint8_t  bit_depth;
    uint32_t bitrate_kbps;
    uint64_t sample_count;
} NeoStemInfo;

/* File header information. */
typedef struct {
    uint16_t version;
    uint64_t feature_flags;
    uint8_t  stem_count;
    uint32_t sample_rate;
    uint64_t duration_us;
    uint64_t chunk_count;
} NeoFileInfo;

/* Library version string (static, do not free). */
const char* neo_version(void);

/* Open a NEO file. Returns NULL on failure. */
NeoHandle* neo_open(const char* path);

/* Close a NEO file handle. */
void neo_close(NeoHandle* handle);

/* Get file header information. */
int neo_get_info(const NeoHandle* handle, NeoFileInfo* info);

/* Get number of stems (-1 on error). */
int neo_stem_count(const NeoHandle* handle);

/* Get stem info by index. */
int neo_get_stem_info(const NeoHandle* handle, uint32_t index, NeoStemInfo* info);

/* Get stem label (caller must free with neo_free_string). */
char* neo_get_stem_label(const NeoHandle* handle, uint32_t index);

/* Read JSON-LD metadata (caller must free with neo_free_string). */
char* neo_read_metadata(NeoHandle* handle);

/* Read spatial audio JSON (caller must free with neo_free_string). */
char* neo_read_spatial(NeoHandle* handle);

/* Read edit history JSON (caller must free with neo_free_string). */
char* neo_read_edit_history(NeoHandle* handle);

/* Read raw stem data (caller must free with neo_free_buffer). */
uint8_t* neo_read_stem_data(NeoHandle* handle, uint8_t stem_id, uint64_t* out_len);

/* Free a string returned by neo_read_* or neo_get_* functions. */
void neo_free_string(char* ptr);

/* Free a buffer returned by neo_read_stem_data. */
void neo_free_buffer(uint8_t* ptr, uint64_t len);

#ifdef __cplusplus
}
#endif

#endif /* NEO_H */
