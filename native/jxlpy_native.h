#ifndef JXLPY_NATIVE_H_
#define JXLPY_NATIVE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define JXLPY_EXPORT __declspec(dllexport)
#else
#define JXLPY_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
  JXLPY_DTYPE_UINT8 = 1,
  JXLPY_DTYPE_UINT16 = 2,
  JXLPY_DTYPE_FLOAT16 = 3,
  JXLPY_DTYPE_FLOAT32 = 4,
};

typedef struct {
  int32_t id;
  int32_t is_float;
  int64_t int_value;
  float float_value;
} jxlpy_encoder_setting;

typedef struct {
  int lossless;
  float distance;
  float alpha_distance;
  int effort;
  int modular;
  int level;
  int threads;
  int use_container;
  int jpeg_store_metadata;
  uint32_t tps_numerator;
  uint32_t tps_denominator;
  int lossless_jpeg;
  int allow_expert_options;
  int compress_boxes;
  int brotli_effort;
  int keep_invisible;
  int patches;
  int dots;
  int noise;
  int gaborish;
  int group_order;
  int center_x;
  int center_y;
  int progressive_ac;
  int qprogressive_ac;
  int progressive_dc;
  int responsive;
  int epf;
  int faster_decoding;
  int resampling;
  int ec_resampling;
  int already_downsampled;
  int upsampling_mode;
  float photon_noise_iso;
  float intensity_target;
  int premultiply;
  int override_bitdepth;
  int buffering;
  int jpeg_reconstruction_cfl;
  int disable_perceptual_optimizations;
  int modular_group_size;
  int modular_predictor;
  int modular_colorspace;
  float modular_ma_tree_learning_percent;
  int modular_nb_prev_channels;
  int modular_palette_colors;
  int modular_lossy_palette;
  float modular_channel_colors_global_percent;
  float modular_channel_colors_group_percent;
  const jxlpy_encoder_setting* extra_encoder_settings;
  size_t num_extra_encoder_settings;
} jxlpy_encode_options;

typedef struct {
  const void* pixels;
  size_t size;
  uint32_t xsize;
  uint32_t ysize;
  uint32_t dtype;
  uint32_t bits_per_sample;
  uint32_t type;
  const char* name;
  size_t name_size;
} jxlpy_extra_channel;

typedef struct {
  const void* pixels;
  size_t size;
  uint32_t xsize;
  uint32_t ysize;
  uint32_t have_crop;
  int32_t crop_x0;
  int32_t crop_y0;
  uint32_t duration;
  uint32_t source_ref;
  uint32_t save_as_ref;
} jxlpy_frame;

typedef struct {
  int ok;
  char* error;
  uint8_t* data;
  size_t size;
  uint32_t xsize;
  uint32_t ysize;
  uint32_t num_channels;
  uint32_t dtype;
  uint32_t bits_per_sample;
  uint32_t exponent_bits_per_sample;
  uint32_t num_frames;
  uint32_t frame_index;
  uint32_t have_animation;
  uint32_t layer_have_crop;
  int32_t crop_x0;
  int32_t crop_y0;
  uint32_t layer_xsize;
  uint32_t layer_ysize;
  uint32_t duration;
  uint32_t num_extra_channels;
  uint32_t extra_channel_index;
  uint32_t extra_channel_type;
  char* extra_channel_name;
} jxlpy_result;

JXLPY_EXPORT const char* jxlpy_version(void);
JXLPY_EXPORT int jxlpy_supports_frame_settings_passthrough(void);
JXLPY_EXPORT void jxlpy_free_result(jxlpy_result* result);

JXLPY_EXPORT jxlpy_result jxlpy_encode_pixels(
    const void* pixels, size_t size, uint32_t xsize, uint32_t ysize,
    uint32_t num_channels, uint32_t dtype, uint32_t bits_per_sample,
    const jxlpy_encode_options* options);

JXLPY_EXPORT jxlpy_result jxlpy_encode_pixels_ex(
    const void* pixels, size_t size, uint32_t xsize, uint32_t ysize,
    uint32_t num_channels, uint32_t dtype, uint32_t bits_per_sample,
    const jxlpy_extra_channel* extra_channels, size_t num_extra_channels,
    const jxlpy_encode_options* options);

JXLPY_EXPORT jxlpy_result jxlpy_encode_image_bytes(
    const uint8_t* bytes, size_t size, const jxlpy_encode_options* options);

JXLPY_EXPORT jxlpy_result jxlpy_encode_multiframe(
    const jxlpy_frame* frames, size_t num_frames, uint32_t canvas_xsize,
    uint32_t canvas_ysize, uint32_t num_channels, uint32_t dtype,
    uint32_t bits_per_sample, const jxlpy_encode_options* options);

JXLPY_EXPORT jxlpy_result jxlpy_encode_multiframe_ex(
    const jxlpy_frame* frames, size_t num_frames, uint32_t canvas_xsize,
    uint32_t canvas_ysize, uint32_t num_channels, uint32_t dtype,
    uint32_t bits_per_sample, const jxlpy_extra_channel* extra_channels,
    size_t num_extra_channels_per_frame,
    const jxlpy_encode_options* options);

JXLPY_EXPORT jxlpy_result jxlpy_decode_jxl(
    const uint8_t* bytes, size_t size, int frame_index, int coalesced,
    uint32_t requested_channels, uint32_t requested_dtype);

JXLPY_EXPORT jxlpy_result jxlpy_decode_extra_channel_jxl(
    const uint8_t* bytes, size_t size, int frame_index, int coalesced,
    uint32_t extra_channel_index, uint32_t requested_dtype);

typedef struct {
  uint32_t extra_channel_index;
  uint32_t extra_channel_type;
  uint32_t bits_per_sample;
  uint32_t exponent_bits_per_sample;
  uint32_t dtype;
  char* extra_channel_name;
  uint8_t* data;
  size_t size;
} jxlpy_extra_channel_result;

typedef struct {
  int ok;
  char* error;
  uint8_t* color_data;
  size_t color_size;
  uint32_t xsize;
  uint32_t ysize;
  uint32_t num_channels;
  uint32_t dtype;
  uint32_t bits_per_sample;
  uint32_t exponent_bits_per_sample;
  uint32_t num_frames;
  uint32_t frame_index;
  uint32_t have_animation;
  uint32_t layer_have_crop;
  int32_t crop_x0;
  int32_t crop_y0;
  uint32_t layer_xsize;
  uint32_t layer_ysize;
  uint32_t duration;
  uint32_t num_extra_channels;
  jxlpy_extra_channel_result* extra_channels;
} jxlpy_decode_all_result;

JXLPY_EXPORT jxlpy_decode_all_result jxlpy_decode_all_jxl(
    const uint8_t* bytes, size_t size, int frame_index, int coalesced,
    uint32_t requested_channels, uint32_t requested_dtype);

JXLPY_EXPORT void jxlpy_free_decode_all_result(jxlpy_decode_all_result* result);

JXLPY_EXPORT jxlpy_result jxlpy_decode_image_bytes(
    const uint8_t* bytes, size_t size, int frame_index);

JXLPY_EXPORT jxlpy_result jxlpy_info(const uint8_t* bytes, size_t size);

JXLPY_EXPORT jxlpy_result jxlpy_reconstruct_jpeg(
    const uint8_t* bytes, size_t size);

JXLPY_EXPORT jxlpy_result jxlpy_decode_to_format(
    const uint8_t* bytes, size_t size, const char* extension, int quality);

#ifdef __cplusplus
}
#endif

#endif  // JXLPY_NATIVE_H_
