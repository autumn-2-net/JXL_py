#include "native/jxlpy_native.h"

#include <jxl/color_encoding.h>
#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/thread_parallel_runner.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "lib/extras/dec/decode.h"
#include "lib/extras/dec/jxl.h"
#include "lib/extras/enc/encode.h"
#include "lib/extras/enc/jxl.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"

namespace {

char* DupString(const std::string& text) {
  char* out = static_cast<char*>(std::malloc(text.size() + 1));
  if (out == nullptr) return nullptr;
  std::memcpy(out, text.c_str(), text.size() + 1);
  return out;
}

jxlpy_result ErrorResult(const std::string& message) {
  jxlpy_result result = {};
  result.ok = 0;
  result.error = DupString(message);
  return result;
}

jxlpy_result BytesResult(const std::vector<uint8_t>& bytes) {
  jxlpy_result result = {};
  result.ok = 1;
  result.size = bytes.size();
  if (!bytes.empty()) {
    result.data = static_cast<uint8_t*>(std::malloc(bytes.size()));
    if (result.data == nullptr) return ErrorResult("out of memory");
    std::memcpy(result.data, bytes.data(), bytes.size());
  }
  return result;
}

size_t DataTypeBytes(uint32_t dtype) {
  switch (dtype) {
    case JXLPY_DTYPE_UINT8:
      return 1;
    case JXLPY_DTYPE_UINT16:
    case JXLPY_DTYPE_FLOAT16:
      return 2;
    case JXLPY_DTYPE_FLOAT32:
      return 4;
    default:
      return 0;
  }
}

JxlDataType ToJxlDataType(uint32_t dtype) {
  switch (dtype) {
    case JXLPY_DTYPE_UINT8:
      return JXL_TYPE_UINT8;
    case JXLPY_DTYPE_UINT16:
      return JXL_TYPE_UINT16;
    case JXLPY_DTYPE_FLOAT16:
      return JXL_TYPE_FLOAT16;
    case JXLPY_DTYPE_FLOAT32:
      return JXL_TYPE_FLOAT;
    default:
      return JXL_TYPE_UINT8;
  }
}

uint32_t FromJxlDataType(JxlDataType dtype) {
  switch (dtype) {
    case JXL_TYPE_UINT8:
      return JXLPY_DTYPE_UINT8;
    case JXL_TYPE_UINT16:
      return JXLPY_DTYPE_UINT16;
    case JXL_TYPE_FLOAT16:
      return JXLPY_DTYPE_FLOAT16;
    case JXL_TYPE_FLOAT:
      return JXLPY_DTYPE_FLOAT32;
    default:
      return 0;
  }
}

uint32_t DefaultBitsPerSample(uint32_t dtype) {
  switch (dtype) {
    case JXLPY_DTYPE_UINT8:
      return 8;
    case JXLPY_DTYPE_UINT16:
      return 16;
    case JXLPY_DTYPE_FLOAT16:
      return 16;
    case JXLPY_DTYPE_FLOAT32:
      return 32;
    default:
      return 0;
  }
}

uint32_t ExponentBits(uint32_t dtype) {
  switch (dtype) {
    case JXLPY_DTYPE_FLOAT16:
      return 5;
    case JXLPY_DTYPE_FLOAT32:
      return 8;
    default:
      return 0;
  }
}

bool ValidatePixelInput(uint32_t xsize, uint32_t ysize, uint32_t channels,
                        uint32_t dtype, size_t size, std::string* error) {
  const size_t bytes_per_sample = DataTypeBytes(dtype);
  if (bytes_per_sample == 0) {
    *error = "unsupported dtype";
    return false;
  }
  if (channels < 1 || channels > 4) {
    *error = "num_channels must be 1, 2, 3 or 4";
    return false;
  }
  const uint64_t expected = static_cast<uint64_t>(xsize) * ysize * channels *
                            bytes_per_sample;
  if (expected != size) {
    *error = "pixel buffer size does not match dimensions";
    return false;
  }
  return true;
}

jxlpy_encode_options NormalizeOptions(const jxlpy_encode_options* options) {
  jxlpy_encode_options out = {};
  out.lossless = 1;
  out.distance = 0.0f;
  out.alpha_distance = 0.0f;
  out.effort = 7;
  out.modular = -1;
  out.level = -1;
  out.threads = 0;
  out.use_container = 0;
  out.jpeg_store_metadata = 1;
  out.tps_numerator = 1000;
  out.tps_denominator = 1;
  if (options != nullptr) out = *options;
  if (out.tps_numerator == 0) out.tps_numerator = 1000;
  if (out.tps_denominator == 0) out.tps_denominator = 1;
  return out;
}

using RunnerPtr = std::unique_ptr<void, decltype(&JxlThreadParallelRunnerDestroy)>;

RunnerPtr MakeRunner(const jxlpy_encode_options& options) {
  const size_t threads =
      options.threads > 0
          ? static_cast<size_t>(options.threads)
          : JxlThreadParallelRunnerDefaultNumWorkerThreads();
  return RunnerPtr(JxlThreadParallelRunnerCreate(nullptr, threads),
                   JxlThreadParallelRunnerDestroy);
}

jxl::extras::JXLCompressParams MakeCompressParams(
    const jxlpy_encode_options& options, void* runner) {
  jxl::extras::JXLCompressParams params;
  params.distance = options.lossless ? 0.0f
                                     : (options.distance > 0.0f ? options.distance
                                                               : 1.0f);
  params.alpha_distance = options.alpha_distance;
  params.codestream_level = options.level;
  params.use_container = options.use_container != 0;
  params.jpeg_store_metadata = options.jpeg_store_metadata != 0;
  params.runner_opaque = runner;
  if (options.effort >= 0) {
    params.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, options.effort);
  }
  if (options.modular >= 0) {
    params.AddOption(JXL_ENC_FRAME_SETTING_MODULAR, options.modular);
  }
  return params;
}

JxlPixelFormat PixelFormat(uint32_t channels, uint32_t dtype) {
  JxlPixelFormat format = {};
  format.num_channels = channels;
  format.data_type = ToJxlDataType(dtype);
  format.endianness = JXL_NATIVE_ENDIAN;
  format.align = 0;
  return format;
}

void FillBasicInfo(uint32_t xsize, uint32_t ysize, uint32_t channels,
                   uint32_t dtype, uint32_t bits_per_sample,
                   uint32_t user_extra_channels,
                   jxl::extras::PackedPixelFile* ppf) {
  JxlEncoderInitBasicInfo(&ppf->info);
  ppf->info.xsize = xsize;
  ppf->info.ysize = ysize;
  ppf->info.bits_per_sample =
      bits_per_sample != 0 ? bits_per_sample : DefaultBitsPerSample(dtype);
  ppf->info.exponent_bits_per_sample = ExponentBits(dtype);
  ppf->info.num_color_channels = channels <= 2 ? 1 : 3;
  const bool has_alpha = channels == 2 || channels == 4;
  ppf->info.num_extra_channels = user_extra_channels + (has_alpha ? 1 : 0);
  if (has_alpha) {
    ppf->info.alpha_bits = ppf->info.bits_per_sample;
    ppf->info.alpha_exponent_bits = ppf->info.exponent_bits_per_sample;
  }
  ppf->input_bitdepth = {JXL_BIT_DEPTH_FROM_PIXEL_FORMAT, 0, 0};
  JxlColorEncodingSetToSRGB(&ppf->color_encoding,
                            channels <= 2 ? JXL_TRUE : JXL_FALSE);
  ppf->primary_color_representation =
      jxl::extras::PackedPixelFile::kColorEncodingIsPrimary;
}

JxlExtraChannelType ToExtraChannelType(uint32_t type) {
  if (type <= static_cast<uint32_t>(JXL_CHANNEL_OPTIONAL)) {
    return static_cast<JxlExtraChannelType>(type);
  }
  return JXL_CHANNEL_UNKNOWN;
}

bool ConfigureExtraChannels(jxl::extras::PackedPixelFile* ppf,
                            const jxlpy_extra_channel* extra_channels,
                            size_t num_extra_channels) {
  ppf->extra_channels_info.clear();
  for (size_t i = 0; i < num_extra_channels; ++i) {
    jxl::extras::PackedExtraChannel extra_info = {};
    const JxlExtraChannelType type = ToExtraChannelType(extra_channels[i].type);
    JxlEncoderInitExtraChannelInfo(type, &extra_info.ec_info);
    extra_info.ec_info.bits_per_sample =
        extra_channels[i].bits_per_sample != 0
            ? extra_channels[i].bits_per_sample
            : DefaultBitsPerSample(extra_channels[i].dtype);
    extra_info.ec_info.exponent_bits_per_sample =
        ExponentBits(extra_channels[i].dtype);
    extra_info.index = i;
    if (extra_channels[i].name != nullptr && extra_channels[i].name_size > 0) {
      extra_info.name.assign(extra_channels[i].name,
                             extra_channels[i].name + extra_channels[i].name_size);
    }
    ppf->extra_channels_info.push_back(std::move(extra_info));
  }
  return true;
}

jxl::Status AddPackedFrame(jxl::extras::PackedPixelFile* ppf,
                           const void* pixels, size_t size, uint32_t xsize,
                           uint32_t ysize, uint32_t channels, uint32_t dtype,
                           const jxlpy_extra_channel* extra_channels,
                           size_t num_extra_channels,
                           const JxlFrameHeader& header) {
  JXL_ASSIGN_OR_RETURN(
      jxl::extras::PackedFrame frame,
      jxl::extras::PackedFrame::Create(xsize, ysize, PixelFormat(channels, dtype)));
  if (frame.color.pixels_size != size) {
    return JXL_FAILURE("unexpected packed image size");
  }
  std::memcpy(frame.color.pixels(), pixels, size);
  for (size_t i = 0; i < num_extra_channels; ++i) {
    if (extra_channels[i].pixels == nullptr) {
      return JXL_FAILURE("extra channel pixels are null");
    }
    if (extra_channels[i].xsize != xsize || extra_channels[i].ysize != ysize) {
      return JXL_FAILURE("extra channel dimensions must match frame dimensions");
    }
    JXL_ASSIGN_OR_RETURN(
        jxl::extras::PackedImage extra,
        jxl::extras::PackedImage::Create(
            xsize, ysize, PixelFormat(1, extra_channels[i].dtype)));
    if (extra.pixels_size != extra_channels[i].size) {
      return JXL_FAILURE("extra channel buffer size does not match dimensions");
    }
    std::memcpy(extra.pixels(), extra_channels[i].pixels, extra.pixels_size);
    frame.extra_channels.push_back(std::move(extra));
  }
  frame.frame_info = header;
  ppf->frames.push_back(std::move(frame));
  return true;
}

JxlFrameHeader MakeFrameHeader(uint32_t canvas_xsize, uint32_t canvas_ysize,
                               const jxlpy_frame* frame, bool animation) {
  JxlFrameHeader header;
  JxlEncoderInitFrameHeader(&header);
  header.duration = frame != nullptr ? frame->duration : 0;
  if (animation && header.duration == 0) header.duration = 1;
  if (frame != nullptr && frame->have_crop) {
    header.layer_info.have_crop = JXL_TRUE;
    header.layer_info.crop_x0 = frame->crop_x0;
    header.layer_info.crop_y0 = frame->crop_y0;
    header.layer_info.xsize = frame->xsize;
    header.layer_info.ysize = frame->ysize;
  } else {
    header.layer_info.have_crop = JXL_FALSE;
    header.layer_info.crop_x0 = 0;
    header.layer_info.crop_y0 = 0;
    header.layer_info.xsize = canvas_xsize;
    header.layer_info.ysize = canvas_ysize;
  }
  header.layer_info.blend_info.blendmode = JXL_BLEND_REPLACE;
  header.layer_info.blend_info.source = frame != nullptr ? frame->source_ref : 0;
  header.layer_info.save_as_reference =
      frame != nullptr ? frame->save_as_ref : (animation ? 1 : 0);
  return header;
}

jxlpy_result EncodePackedPixelFile(jxl::extras::PackedPixelFile* ppf,
                                   const jxlpy_encode_options& options,
                                   const std::vector<uint8_t>* jpeg_bytes) {
  auto runner = MakeRunner(options);
  if (!runner) return ErrorResult("failed to create JPEG XL thread runner");
  auto params = MakeCompressParams(options, runner.get());
  std::vector<uint8_t> compressed;
  if (!jxl::extras::EncodeImageJXL(params, *ppf, jpeg_bytes, &compressed)) {
    return ErrorResult("JPEG XL encoding failed");
  }
  return BytesResult(compressed);
}

jxlpy_result CopyPackedFrame(const jxl::extras::PackedPixelFile& ppf,
                             size_t frame_index) {
  if (frame_index >= ppf.frames.size()) {
    return ErrorResult("frame index out of range");
  }
  const auto& frame = ppf.frames[frame_index];
  const auto& image = frame.color;
  jxlpy_result result = {};
  result.ok = 1;
  result.size = image.pixels_size;
  result.xsize = static_cast<uint32_t>(image.xsize);
  result.ysize = static_cast<uint32_t>(image.ysize);
  result.num_channels = image.format.num_channels;
  result.dtype = FromJxlDataType(image.format.data_type);
  result.bits_per_sample = ppf.info.bits_per_sample;
  result.exponent_bits_per_sample = ppf.info.exponent_bits_per_sample;
  result.num_frames = static_cast<uint32_t>(ppf.frames.size());
  result.frame_index = static_cast<uint32_t>(frame_index);
  result.have_animation = ppf.info.have_animation;
  result.layer_have_crop = frame.frame_info.layer_info.have_crop;
  result.crop_x0 = frame.frame_info.layer_info.crop_x0;
  result.crop_y0 = frame.frame_info.layer_info.crop_y0;
  result.layer_xsize = frame.frame_info.layer_info.xsize;
  result.layer_ysize = frame.frame_info.layer_info.ysize;
  result.duration = frame.frame_info.duration;
  result.num_extra_channels = ppf.info.num_extra_channels;
  if (result.size != 0) {
    result.data = static_cast<uint8_t*>(std::malloc(result.size));
    if (result.data == nullptr) return ErrorResult("out of memory");
    std::memcpy(result.data, image.pixels(), result.size);
  }
  return result;
}

bool IsJxlBytes(const uint8_t* bytes, size_t size) {
  if (size >= 2 && bytes[0] == 0xff && bytes[1] == 0x0a) return true;
  static const uint8_t kContainerSig[12] = {0x00, 0x00, 0x00, 0x0c, 0x4a, 0x58,
                                           0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a};
  return size >= sizeof(kContainerSig) &&
         std::memcmp(bytes, kContainerSig, sizeof(kContainerSig)) == 0;
}

uint32_t DefaultDecodeDtype(const JxlBasicInfo& info) {
  if (info.exponent_bits_per_sample != 0 || info.bits_per_sample > 16) {
    return JXLPY_DTYPE_FLOAT32;
  }
  return info.bits_per_sample > 8 ? JXLPY_DTYPE_UINT16 : JXLPY_DTYPE_UINT8;
}

}  // namespace

extern "C" {

const char* jxlpy_version(void) { return "jxlpy_native/0.1"; }

void jxlpy_free_result(jxlpy_result* result) {
  if (result == nullptr) return;
  std::free(result->error);
  std::free(result->data);
  std::free(result->extra_channel_name);
  std::memset(result, 0, sizeof(*result));
}

jxlpy_result jxlpy_encode_pixels(const void* pixels, size_t size,
                                 uint32_t xsize, uint32_t ysize,
                                 uint32_t num_channels, uint32_t dtype,
                                 uint32_t bits_per_sample,
                                 const jxlpy_encode_options* options) {
  return jxlpy_encode_pixels_ex(pixels, size, xsize, ysize, num_channels, dtype,
                                bits_per_sample, nullptr, 0, options);
}

jxlpy_result jxlpy_encode_pixels_ex(
    const void* pixels, size_t size, uint32_t xsize, uint32_t ysize,
    uint32_t num_channels, uint32_t dtype, uint32_t bits_per_sample,
    const jxlpy_extra_channel* extra_channels, size_t num_extra_channels,
    const jxlpy_encode_options* options) {
  std::string error;
  if (pixels == nullptr) return ErrorResult("pixels is null");
  if (!ValidatePixelInput(xsize, ysize, num_channels, dtype, size, &error)) {
    return ErrorResult(error);
  }
  const auto opts = NormalizeOptions(options);
  jxl::extras::PackedPixelFile ppf;
  FillBasicInfo(xsize, ysize, num_channels, dtype, bits_per_sample,
                static_cast<uint32_t>(num_extra_channels), &ppf);
  ConfigureExtraChannels(&ppf, extra_channels, num_extra_channels);
  JxlFrameHeader header = MakeFrameHeader(xsize, ysize, nullptr, false);
  if (!AddPackedFrame(&ppf, pixels, size, xsize, ysize, num_channels, dtype,
                      extra_channels, num_extra_channels, header)) {
    return ErrorResult("failed to create packed frame");
  }
  return EncodePackedPixelFile(&ppf, opts, nullptr);
}

jxlpy_result jxlpy_encode_image_bytes(const uint8_t* bytes, size_t size,
                                      const jxlpy_encode_options* options) {
  if (bytes == nullptr || size == 0) return ErrorResult("input bytes are empty");
  const auto opts = NormalizeOptions(options);
  jxl::extras::PackedPixelFile ppf;
  jxl::extras::Codec codec = jxl::extras::Codec::kUnknown;
  if (!jxl::extras::DecodeBytes(jxl::Bytes(bytes, size), jxl::extras::ColorHints(),
                                &ppf, nullptr, &codec)) {
    return ErrorResult("failed to decode input image bytes");
  }
  std::vector<uint8_t> jpeg_bytes;
  const std::vector<uint8_t>* jpeg_ptr = nullptr;
  if (codec == jxl::extras::Codec::kJPG && opts.lossless) {
    jpeg_bytes.assign(bytes, bytes + size);
    jpeg_ptr = &jpeg_bytes;
  }
  return EncodePackedPixelFile(&ppf, opts, jpeg_ptr);
}

jxlpy_result jxlpy_encode_multiframe(const jxlpy_frame* frames,
                                     size_t num_frames,
                                     uint32_t canvas_xsize,
                                     uint32_t canvas_ysize,
                                     uint32_t num_channels, uint32_t dtype,
                                     uint32_t bits_per_sample,
                                     const jxlpy_encode_options* options) {
  return jxlpy_encode_multiframe_ex(frames, num_frames, canvas_xsize,
                                    canvas_ysize, num_channels, dtype,
                                    bits_per_sample, nullptr, 0, options);
}

jxlpy_result jxlpy_encode_multiframe_ex(
    const jxlpy_frame* frames, size_t num_frames, uint32_t canvas_xsize,
    uint32_t canvas_ysize, uint32_t num_channels, uint32_t dtype,
    uint32_t bits_per_sample, const jxlpy_extra_channel* extra_channels,
    size_t num_extra_channels_per_frame,
    const jxlpy_encode_options* options) {
  if (frames == nullptr || num_frames == 0) {
    return ErrorResult("frames are empty");
  }
  const auto opts = NormalizeOptions(options);
  jxl::extras::PackedPixelFile ppf;
  FillBasicInfo(canvas_xsize, canvas_ysize, num_channels, dtype, bits_per_sample,
                static_cast<uint32_t>(num_extra_channels_per_frame), &ppf);
  ConfigureExtraChannels(&ppf, extra_channels, num_extra_channels_per_frame);
  ppf.info.have_animation = TO_JXL_BOOL(num_frames > 1);
  if (num_frames > 1) {
    ppf.info.animation.tps_numerator = opts.tps_numerator;
    ppf.info.animation.tps_denominator = opts.tps_denominator;
    ppf.info.animation.num_loops = 0;
    ppf.info.animation.have_timecodes = JXL_FALSE;
  }

  for (size_t i = 0; i < num_frames; ++i) {
    std::string error;
    if (frames[i].pixels == nullptr) return ErrorResult("frame pixels are null");
    if (!ValidatePixelInput(frames[i].xsize, frames[i].ysize, num_channels, dtype,
                            frames[i].size, &error)) {
      return ErrorResult(error);
    }
    JxlFrameHeader header =
        MakeFrameHeader(canvas_xsize, canvas_ysize, &frames[i], num_frames > 1);
    const jxlpy_extra_channel* frame_extras =
        extra_channels == nullptr
            ? nullptr
            : extra_channels + i * num_extra_channels_per_frame;
    if (!AddPackedFrame(&ppf, frames[i].pixels, frames[i].size, frames[i].xsize,
                        frames[i].ysize, num_channels, dtype, frame_extras,
                        num_extra_channels_per_frame, header)) {
      return ErrorResult("failed to create packed animation frame");
    }
  }
  return EncodePackedPixelFile(&ppf, opts, nullptr);
}

jxlpy_result jxlpy_decode_jxl(const uint8_t* bytes, size_t size,
                              int frame_index, int coalesced,
                              uint32_t requested_channels,
                              uint32_t requested_dtype) {
  if (bytes == nullptr || size == 0) return ErrorResult("input bytes are empty");
  if (!IsJxlBytes(bytes, size)) return ErrorResult("input is not JPEG XL");

  JxlDecoder* dec = JxlDecoderCreate(nullptr);
  if (dec == nullptr) return ErrorResult("failed to create JPEG XL decoder");
  std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> dec_ptr(
      dec, JxlDecoderDestroy);
  void* runner = JxlThreadParallelRunnerCreate(
      nullptr, JxlThreadParallelRunnerDefaultNumWorkerThreads());
  RunnerPtr runner_ptr(runner, JxlThreadParallelRunnerDestroy);
  if (runner == nullptr) return ErrorResult("failed to create JPEG XL runner");
  if (JxlDecoderSetParallelRunner(dec, JxlThreadParallelRunner, runner) !=
      JXL_DEC_SUCCESS) {
    return ErrorResult("failed to set JPEG XL runner");
  }
  if (JxlDecoderSetCoalescing(dec, coalesced ? JXL_TRUE : JXL_FALSE) !=
      JXL_DEC_SUCCESS) {
    return ErrorResult("failed to set JPEG XL coalescing mode");
  }
  if (JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_FRAME |
                                         JXL_DEC_FULL_IMAGE) !=
      JXL_DEC_SUCCESS) {
    return ErrorResult("failed to subscribe decoder events");
  }
  JxlDecoderSetInput(dec, bytes, size);
  JxlDecoderCloseInput(dec);

  const uint32_t target_frame =
      frame_index < 0 ? 0u : static_cast<uint32_t>(frame_index);
  bool have_info = false;
  bool have_current_header = false;
  bool got_target = false;
  JxlBasicInfo info = {};
  JxlFrameHeader current_header = {};
  uint32_t current_frame = 0;
  uint32_t total_frames = 0;
  uint32_t output_channels = 0;
  uint32_t output_dtype = 0;
  JxlPixelFormat format = {};
  std::vector<uint8_t> current_buffer;
  std::vector<uint8_t> final_buffer;
  jxlpy_result result = {};

  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec);
    if (status == JXL_DEC_ERROR) {
      return ErrorResult("failed to decode JPEG XL bytes");
    }
    if (status == JXL_DEC_SUCCESS) {
      break;
    }
    if (status == JXL_DEC_NEED_MORE_INPUT) {
      return ErrorResult("truncated JPEG XL input");
    }
    if (status == JXL_DEC_BASIC_INFO) {
      if (JxlDecoderGetBasicInfo(dec, &info) != JXL_DEC_SUCCESS) {
        return ErrorResult("failed to get JPEG XL basic info");
      }
      have_info = true;
      output_channels =
          (requested_channels >= 1 && requested_channels <= 4)
              ? requested_channels
              : info.num_color_channels + (info.alpha_bits ? 1u : 0u);
      output_dtype = DataTypeBytes(requested_dtype) != 0
                         ? requested_dtype
                         : DefaultDecodeDtype(info);
      format = PixelFormat(output_channels, output_dtype);
      continue;
    }
    if (status == JXL_DEC_FRAME) {
      if (!have_info) return ErrorResult("frame seen before basic info");
      if (JxlDecoderGetFrameHeader(dec, &current_header) != JXL_DEC_SUCCESS) {
        return ErrorResult("failed to get JPEG XL frame header");
      }
      have_current_header = true;
      continue;
    }
    if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      if (!have_info || !have_current_header) {
        return ErrorResult("image buffer requested before frame header");
      }
      size_t out_size = 0;
      if (JxlDecoderImageOutBufferSize(dec, &format, &out_size) !=
          JXL_DEC_SUCCESS) {
        return ErrorResult("failed to get JPEG XL output buffer size");
      }
      current_buffer.assign(out_size, 0);
      if (JxlDecoderSetImageOutBuffer(dec, &format, current_buffer.data(),
                                      current_buffer.size()) !=
          JXL_DEC_SUCCESS) {
        return ErrorResult("failed to set JPEG XL output buffer");
      }
      if (output_dtype == JXLPY_DTYPE_UINT8 ||
          output_dtype == JXLPY_DTYPE_UINT16) {
        JxlBitDepth bit_depth = {JXL_BIT_DEPTH_FROM_CODESTREAM, 0, 0};
        JxlDecoderSetImageOutBitDepth(dec, &bit_depth);
      }
      continue;
    }
    if (status == JXL_DEC_FULL_IMAGE) {
      if (current_frame == target_frame) {
        final_buffer = current_buffer;
        result.ok = 1;
        result.xsize =
            coalesced ? info.xsize : current_header.layer_info.xsize;
        result.ysize =
            coalesced ? info.ysize : current_header.layer_info.ysize;
        result.num_channels = output_channels;
        result.dtype = output_dtype;
        result.bits_per_sample = info.bits_per_sample;
        result.exponent_bits_per_sample = info.exponent_bits_per_sample;
        result.frame_index = current_frame;
        result.have_animation = info.have_animation;
        result.num_extra_channels = info.num_extra_channels;
        result.layer_have_crop = coalesced ? 0 : current_header.layer_info.have_crop;
        result.crop_x0 = coalesced ? 0 : current_header.layer_info.crop_x0;
        result.crop_y0 = coalesced ? 0 : current_header.layer_info.crop_y0;
        result.layer_xsize = current_header.layer_info.xsize;
        result.layer_ysize = current_header.layer_info.ysize;
        result.duration = current_header.duration;
        got_target = true;
      }
      ++current_frame;
      total_frames = current_frame;
      current_buffer.clear();
      have_current_header = false;
      continue;
    }
  }

  if (!got_target) return ErrorResult("frame index out of range");
  result.size = final_buffer.size();
  if (!final_buffer.empty()) {
    result.data = static_cast<uint8_t*>(std::malloc(final_buffer.size()));
    if (result.data == nullptr) return ErrorResult("out of memory");
    std::memcpy(result.data, final_buffer.data(), final_buffer.size());
  }
  result.num_frames = total_frames;
  return result;
}

jxlpy_result jxlpy_decode_extra_channel_jxl(
    const uint8_t* bytes, size_t size, int frame_index, int coalesced,
    uint32_t extra_channel_index, uint32_t requested_dtype) {
  if (bytes == nullptr || size == 0) return ErrorResult("input bytes are empty");
  if (!IsJxlBytes(bytes, size)) return ErrorResult("input is not JPEG XL");

  JxlDecoder* dec = JxlDecoderCreate(nullptr);
  if (dec == nullptr) return ErrorResult("failed to create JPEG XL decoder");
  std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> dec_ptr(
      dec, JxlDecoderDestroy);
  void* runner = JxlThreadParallelRunnerCreate(
      nullptr, JxlThreadParallelRunnerDefaultNumWorkerThreads());
  RunnerPtr runner_ptr(runner, JxlThreadParallelRunnerDestroy);
  if (runner == nullptr) return ErrorResult("failed to create JPEG XL runner");
  if (JxlDecoderSetParallelRunner(dec, JxlThreadParallelRunner, runner) !=
      JXL_DEC_SUCCESS) {
    return ErrorResult("failed to set JPEG XL runner");
  }
  if (JxlDecoderSetCoalescing(dec, coalesced ? JXL_TRUE : JXL_FALSE) !=
      JXL_DEC_SUCCESS) {
    return ErrorResult("failed to set JPEG XL coalescing mode");
  }
  if (JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_FRAME |
                                         JXL_DEC_FULL_IMAGE) !=
      JXL_DEC_SUCCESS) {
    return ErrorResult("failed to subscribe decoder events");
  }
  JxlDecoderSetInput(dec, bytes, size);
  JxlDecoderCloseInput(dec);

  const uint32_t target_frame =
      frame_index < 0 ? 0u : static_cast<uint32_t>(frame_index);
  bool have_info = false;
  bool have_current_header = false;
  bool got_target = false;
  JxlBasicInfo info = {};
  JxlExtraChannelInfo extra_info = {};
  std::string extra_name;
  JxlFrameHeader current_header = {};
  uint32_t current_frame = 0;
  uint32_t total_frames = 0;
  uint32_t image_dtype = 0;
  uint32_t extra_dtype = 0;
  JxlPixelFormat image_format = {};
  JxlPixelFormat extra_format = {};
  std::vector<uint8_t> image_buffer;
  std::vector<uint8_t> current_extra_buffer;
  std::vector<uint8_t> final_extra_buffer;
  jxlpy_result result = {};

  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec);
    if (status == JXL_DEC_ERROR) {
      return ErrorResult("failed to decode JPEG XL extra channel");
    }
    if (status == JXL_DEC_SUCCESS) {
      break;
    }
    if (status == JXL_DEC_NEED_MORE_INPUT) {
      return ErrorResult("truncated JPEG XL input");
    }
    if (status == JXL_DEC_BASIC_INFO) {
      if (JxlDecoderGetBasicInfo(dec, &info) != JXL_DEC_SUCCESS) {
        return ErrorResult("failed to get JPEG XL basic info");
      }
      if (extra_channel_index >= info.num_extra_channels) {
        return ErrorResult("extra channel index out of range");
      }
      if (JxlDecoderGetExtraChannelInfo(dec, extra_channel_index, &extra_info) !=
          JXL_DEC_SUCCESS) {
        return ErrorResult("failed to get JPEG XL extra channel info");
      }
      if (extra_info.name_length > 0) {
        std::vector<char> name_buffer(extra_info.name_length + 1, '\0');
        if (JxlDecoderGetExtraChannelName(dec, extra_channel_index,
                                          name_buffer.data(),
                                          name_buffer.size()) !=
            JXL_DEC_SUCCESS) {
          return ErrorResult("failed to get JPEG XL extra channel name");
        }
        extra_name.assign(name_buffer.data(), extra_info.name_length);
      }
      have_info = true;
      image_dtype = DefaultDecodeDtype(info);
      image_format =
          PixelFormat(info.num_color_channels + (info.alpha_bits ? 1u : 0u),
                      image_dtype);
      if (DataTypeBytes(requested_dtype) != 0) {
        extra_dtype = requested_dtype;
      } else if (extra_info.exponent_bits_per_sample != 0 ||
                 extra_info.bits_per_sample > 16) {
        extra_dtype = JXLPY_DTYPE_FLOAT32;
      } else {
        extra_dtype =
            extra_info.bits_per_sample > 8 ? JXLPY_DTYPE_UINT16
                                           : JXLPY_DTYPE_UINT8;
      }
      extra_format = PixelFormat(1, extra_dtype);
      continue;
    }
    if (status == JXL_DEC_FRAME) {
      if (!have_info) return ErrorResult("frame seen before basic info");
      if (JxlDecoderGetFrameHeader(dec, &current_header) != JXL_DEC_SUCCESS) {
        return ErrorResult("failed to get JPEG XL frame header");
      }
      have_current_header = true;
      continue;
    }
    if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      if (!have_info || !have_current_header) {
        return ErrorResult("image buffer requested before frame header");
      }
      size_t image_size = 0;
      if (JxlDecoderImageOutBufferSize(dec, &image_format, &image_size) !=
          JXL_DEC_SUCCESS) {
        return ErrorResult("failed to get JPEG XL output buffer size");
      }
      image_buffer.assign(image_size, 0);
      if (JxlDecoderSetImageOutBuffer(dec, &image_format, image_buffer.data(),
                                      image_buffer.size()) !=
          JXL_DEC_SUCCESS) {
        return ErrorResult("failed to set JPEG XL image output buffer");
      }
      if (image_dtype == JXLPY_DTYPE_UINT8 ||
          image_dtype == JXLPY_DTYPE_UINT16) {
        JxlBitDepth bit_depth = {JXL_BIT_DEPTH_FROM_CODESTREAM, 0, 0};
        JxlDecoderSetImageOutBitDepth(dec, &bit_depth);
      }

      size_t extra_size = 0;
      if (JxlDecoderExtraChannelBufferSize(dec, &extra_format, &extra_size,
                                           extra_channel_index) !=
          JXL_DEC_SUCCESS) {
        return ErrorResult("failed to get JPEG XL extra channel buffer size");
      }
      current_extra_buffer.assign(extra_size, 0);
      if (JxlDecoderSetExtraChannelBuffer(
              dec, &extra_format, current_extra_buffer.data(),
              current_extra_buffer.size(), extra_channel_index) !=
          JXL_DEC_SUCCESS) {
        return ErrorResult("failed to set JPEG XL extra channel output buffer");
      }
      continue;
    }
    if (status == JXL_DEC_FULL_IMAGE) {
      if (current_frame == target_frame) {
        final_extra_buffer = current_extra_buffer;
        result.ok = 1;
        result.xsize =
            coalesced ? info.xsize : current_header.layer_info.xsize;
        result.ysize =
            coalesced ? info.ysize : current_header.layer_info.ysize;
        result.num_channels = 1;
        result.dtype = extra_dtype;
        result.bits_per_sample = extra_info.bits_per_sample;
        result.exponent_bits_per_sample = extra_info.exponent_bits_per_sample;
        result.frame_index = current_frame;
        result.have_animation = info.have_animation;
        result.num_extra_channels = info.num_extra_channels;
        result.extra_channel_index = extra_channel_index;
        result.extra_channel_type = static_cast<uint32_t>(extra_info.type);
        result.layer_have_crop = coalesced ? 0 : current_header.layer_info.have_crop;
        result.crop_x0 = coalesced ? 0 : current_header.layer_info.crop_x0;
        result.crop_y0 = coalesced ? 0 : current_header.layer_info.crop_y0;
        result.layer_xsize = current_header.layer_info.xsize;
        result.layer_ysize = current_header.layer_info.ysize;
        result.duration = current_header.duration;
        got_target = true;
      }
      ++current_frame;
      total_frames = current_frame;
      image_buffer.clear();
      current_extra_buffer.clear();
      have_current_header = false;
      continue;
    }
  }

  if (!got_target) return ErrorResult("frame index out of range");
  result.size = final_extra_buffer.size();
  if (!final_extra_buffer.empty()) {
    result.data = static_cast<uint8_t*>(std::malloc(final_extra_buffer.size()));
    if (result.data == nullptr) return ErrorResult("out of memory");
    std::memcpy(result.data, final_extra_buffer.data(), final_extra_buffer.size());
  }
  if (!extra_name.empty()) {
    result.extra_channel_name = DupString(extra_name);
  }
  result.num_frames = total_frames;
  return result;
}

jxlpy_decode_all_result jxlpy_decode_all_jxl(
    const uint8_t* bytes, size_t size, int frame_index, int coalesced,
    uint32_t requested_channels, uint32_t requested_dtype) {
  jxlpy_decode_all_result all_result = {};
  if (bytes == nullptr || size == 0) {
    all_result.error = DupString("input bytes are empty");
    return all_result;
  }
  if (!IsJxlBytes(bytes, size)) {
    all_result.error = DupString("input is not JPEG XL");
    return all_result;
  }

  JxlDecoder* dec = JxlDecoderCreate(nullptr);
  if (dec == nullptr) {
    all_result.error = DupString("failed to create JPEG XL decoder");
    return all_result;
  }
  std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> dec_ptr(
      dec, JxlDecoderDestroy);
  void* runner = JxlThreadParallelRunnerCreate(
      nullptr, JxlThreadParallelRunnerDefaultNumWorkerThreads());
  RunnerPtr runner_ptr(runner, JxlThreadParallelRunnerDestroy);
  if (runner == nullptr) {
    all_result.error = DupString("failed to create JPEG XL runner");
    return all_result;
  }
  if (JxlDecoderSetParallelRunner(dec, JxlThreadParallelRunner, runner) !=
      JXL_DEC_SUCCESS) {
    all_result.error = DupString("failed to set JPEG XL runner");
    return all_result;
  }
  if (JxlDecoderSetCoalescing(dec, coalesced ? JXL_TRUE : JXL_FALSE) !=
      JXL_DEC_SUCCESS) {
    all_result.error = DupString("failed to set JPEG XL coalescing mode");
    return all_result;
  }
  if (JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_FRAME |
                                         JXL_DEC_FULL_IMAGE) !=
      JXL_DEC_SUCCESS) {
    all_result.error = DupString("failed to subscribe decoder events");
    return all_result;
  }
  JxlDecoderSetInput(dec, bytes, size);
  JxlDecoderCloseInput(dec);

  const uint32_t target_frame =
      frame_index < 0 ? 0u : static_cast<uint32_t>(frame_index);
  bool have_info = false;
  bool have_current_header = false;
  bool got_target = false;
  JxlBasicInfo info = {};
  JxlFrameHeader current_header = {};
  uint32_t current_frame = 0;
  uint32_t total_frames = 0;
  uint32_t output_channels = 0;
  uint32_t output_dtype = 0;
  JxlPixelFormat format = {};
  std::vector<uint8_t> color_buffer;
  std::vector<uint8_t> final_color_buffer;

  struct ExtraChannelState {
    uint32_t index;
    JxlExtraChannelInfo ec_info;
    std::string name;
    uint32_t dtype;
    JxlPixelFormat format;
    std::vector<uint8_t> buffer;
    std::vector<uint8_t> final_buffer;
  };
  std::vector<ExtraChannelState> extra_states;

  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec);
    if (status == JXL_DEC_ERROR) {
      all_result.error = DupString("failed to decode JPEG XL bytes");
      return all_result;
    }
    if (status == JXL_DEC_SUCCESS) {
      break;
    }
    if (status == JXL_DEC_NEED_MORE_INPUT) {
      all_result.error = DupString("truncated JPEG XL input");
      return all_result;
    }
    if (status == JXL_DEC_BASIC_INFO) {
      if (JxlDecoderGetBasicInfo(dec, &info) != JXL_DEC_SUCCESS) {
        all_result.error = DupString("failed to get JPEG XL basic info");
        return all_result;
      }
      output_channels =
          (requested_channels >= 1 && requested_channels <= 4)
              ? requested_channels
              : info.num_color_channels + (info.alpha_bits ? 1u : 0u);
      output_dtype = DataTypeBytes(requested_dtype) != 0
                         ? requested_dtype
                         : DefaultDecodeDtype(info);
      format = PixelFormat(output_channels, output_dtype);

      extra_states.resize(info.num_extra_channels);
      for (uint32_t i = 0; i < info.num_extra_channels; ++i) {
        ExtraChannelState& ecs = extra_states[i];
        ecs.index = i;
        if (JxlDecoderGetExtraChannelInfo(dec, i, &ecs.ec_info) !=
            JXL_DEC_SUCCESS) {
          all_result.error = DupString("failed to get extra channel info");
          return all_result;
        }
        if (ecs.ec_info.name_length > 0) {
          std::vector<char> name_buf(ecs.ec_info.name_length + 1, '\0');
          if (JxlDecoderGetExtraChannelName(dec, i, name_buf.data(),
                                            name_buf.size()) !=
              JXL_DEC_SUCCESS) {
            all_result.error = DupString("failed to get extra channel name");
            return all_result;
          }
          ecs.name.assign(name_buf.data(), ecs.ec_info.name_length);
        }
        if (DataTypeBytes(requested_dtype) != 0) {
          ecs.dtype = requested_dtype;
        } else if (ecs.ec_info.exponent_bits_per_sample != 0 ||
                   ecs.ec_info.bits_per_sample > 16) {
          ecs.dtype = JXLPY_DTYPE_FLOAT32;
        } else {
          ecs.dtype = ecs.ec_info.bits_per_sample > 8
                          ? JXLPY_DTYPE_UINT16
                          : JXLPY_DTYPE_UINT8;
        }
        ecs.format = PixelFormat(1, ecs.dtype);
      }
      have_info = true;
      continue;
    }
    if (status == JXL_DEC_FRAME) {
      if (!have_info) {
        all_result.error = DupString("frame seen before basic info");
        return all_result;
      }
      if (JxlDecoderGetFrameHeader(dec, &current_header) != JXL_DEC_SUCCESS) {
        all_result.error = DupString("failed to get JPEG XL frame header");
        return all_result;
      }
      have_current_header = true;
      continue;
    }
    if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      if (!have_info || !have_current_header) {
        all_result.error =
            DupString("image buffer requested before frame header");
        return all_result;
      }
      size_t out_size = 0;
      if (JxlDecoderImageOutBufferSize(dec, &format, &out_size) !=
          JXL_DEC_SUCCESS) {
        all_result.error =
            DupString("failed to get JPEG XL output buffer size");
        return all_result;
      }
      color_buffer.assign(out_size, 0);
      if (JxlDecoderSetImageOutBuffer(dec, &format, color_buffer.data(),
                                      color_buffer.size()) !=
          JXL_DEC_SUCCESS) {
        all_result.error =
            DupString("failed to set JPEG XL output buffer");
        return all_result;
      }
      if (output_dtype == JXLPY_DTYPE_UINT8 ||
          output_dtype == JXLPY_DTYPE_UINT16) {
        JxlBitDepth bit_depth = {JXL_BIT_DEPTH_FROM_CODESTREAM, 0, 0};
        JxlDecoderSetImageOutBitDepth(dec, &bit_depth);
      }

      for (auto& ecs : extra_states) {
        size_t ec_size = 0;
        if (JxlDecoderExtraChannelBufferSize(dec, &ecs.format, &ec_size,
                                             ecs.index) != JXL_DEC_SUCCESS) {
          all_result.error =
              DupString("failed to get extra channel buffer size");
          return all_result;
        }
        ecs.buffer.assign(ec_size, 0);
        if (JxlDecoderSetExtraChannelBuffer(dec, &ecs.format,
                                            ecs.buffer.data(),
                                            ecs.buffer.size(),
                                            ecs.index) != JXL_DEC_SUCCESS) {
          all_result.error =
              DupString("failed to set extra channel output buffer");
          return all_result;
        }
      }
      continue;
    }
    if (status == JXL_DEC_FULL_IMAGE) {
      if (current_frame == target_frame) {
        final_color_buffer = color_buffer;
        for (auto& ecs : extra_states) {
          ecs.final_buffer = ecs.buffer;
        }
        all_result.ok = 1;
        all_result.xsize =
            coalesced ? info.xsize : current_header.layer_info.xsize;
        all_result.ysize =
            coalesced ? info.ysize : current_header.layer_info.ysize;
        all_result.num_channels = output_channels;
        all_result.dtype = output_dtype;
        all_result.bits_per_sample = info.bits_per_sample;
        all_result.exponent_bits_per_sample = info.exponent_bits_per_sample;
        all_result.frame_index = current_frame;
        all_result.have_animation = info.have_animation;
        all_result.num_extra_channels = info.num_extra_channels;
        all_result.layer_have_crop =
            coalesced ? 0 : current_header.layer_info.have_crop;
        all_result.crop_x0 =
            coalesced ? 0 : current_header.layer_info.crop_x0;
        all_result.crop_y0 =
            coalesced ? 0 : current_header.layer_info.crop_y0;
        all_result.layer_xsize = current_header.layer_info.xsize;
        all_result.layer_ysize = current_header.layer_info.ysize;
        all_result.duration = current_header.duration;
        got_target = true;
      }
      ++current_frame;
      total_frames = current_frame;
      color_buffer.clear();
      for (auto& ecs : extra_states) {
        ecs.buffer.clear();
      }
      have_current_header = false;
      continue;
    }
  }

  if (!got_target) {
    all_result.error = DupString("frame index out of range");
    return all_result;
  }

  all_result.color_size = final_color_buffer.size();
  if (!final_color_buffer.empty()) {
    all_result.color_data =
        static_cast<uint8_t*>(std::malloc(final_color_buffer.size()));
    if (all_result.color_data == nullptr) {
      all_result.error = DupString("out of memory");
      return all_result;
    }
    std::memcpy(all_result.color_data, final_color_buffer.data(),
                final_color_buffer.size());
  }

  if (!extra_states.empty()) {
    all_result.extra_channels = static_cast<jxlpy_extra_channel_result*>(
        std::calloc(extra_states.size(), sizeof(jxlpy_extra_channel_result)));
    if (all_result.extra_channels == nullptr) {
      all_result.error = DupString("out of memory");
      return all_result;
    }
    for (size_t i = 0; i < extra_states.size(); ++i) {
      auto& ecs = extra_states[i];
      auto& out = all_result.extra_channels[i];
      out.extra_channel_index = ecs.index;
      out.extra_channel_type = static_cast<uint32_t>(ecs.ec_info.type);
      out.bits_per_sample = ecs.ec_info.bits_per_sample;
      out.exponent_bits_per_sample = ecs.ec_info.exponent_bits_per_sample;
      out.dtype = ecs.dtype;
      if (!ecs.name.empty()) {
        out.extra_channel_name = DupString(ecs.name);
      }
      out.size = ecs.final_buffer.size();
      if (!ecs.final_buffer.empty()) {
        out.data = static_cast<uint8_t*>(
            std::malloc(ecs.final_buffer.size()));
        if (out.data == nullptr) {
          all_result.error = DupString("out of memory");
          return all_result;
        }
        std::memcpy(out.data, ecs.final_buffer.data(),
                    ecs.final_buffer.size());
      }
    }
  }

  all_result.num_frames = total_frames;
  return all_result;
}

void jxlpy_free_decode_all_result(jxlpy_decode_all_result* result) {
  if (result == nullptr) return;
  std::free(result->error);
  std::free(result->color_data);
  if (result->extra_channels != nullptr) {
    for (uint32_t i = 0; i < result->num_extra_channels; ++i) {
      std::free(result->extra_channels[i].extra_channel_name);
      std::free(result->extra_channels[i].data);
    }
    std::free(result->extra_channels);
  }
  std::memset(result, 0, sizeof(*result));
}

jxlpy_result jxlpy_decode_image_bytes(const uint8_t* bytes, size_t size,
                                      int frame_index) {
  if (bytes == nullptr || size == 0) return ErrorResult("input bytes are empty");
  jxl::extras::PackedPixelFile ppf;
  if (!jxl::extras::DecodeBytes(jxl::Bytes(bytes, size), jxl::extras::ColorHints(),
                                &ppf)) {
    return ErrorResult("failed to decode image bytes");
  }
  const size_t index = frame_index < 0 ? 0 : static_cast<size_t>(frame_index);
  return CopyPackedFrame(ppf, index);
}

jxlpy_result jxlpy_info(const uint8_t* bytes, size_t size) {
  if (bytes == nullptr || size == 0) return ErrorResult("input bytes are empty");
  if (!IsJxlBytes(bytes, size)) return ErrorResult("input is not JPEG XL");

  JxlDecoder* dec = JxlDecoderCreate(nullptr);
  if (dec == nullptr) return ErrorResult("failed to create JPEG XL decoder");
  std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> dec_ptr(
      dec, JxlDecoderDestroy);
  void* runner = JxlThreadParallelRunnerCreate(
      nullptr, JxlThreadParallelRunnerDefaultNumWorkerThreads());
  RunnerPtr runner_ptr(runner, JxlThreadParallelRunnerDestroy);
  if (runner == nullptr) return ErrorResult("failed to create JPEG XL runner");
  if (JxlDecoderSetParallelRunner(dec, JxlThreadParallelRunner, runner) !=
      JXL_DEC_SUCCESS) {
    return ErrorResult("failed to set JPEG XL runner");
  }
  JxlDecoderSetCoalescing(dec, JXL_FALSE);
  if (JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_FRAME) !=
      JXL_DEC_SUCCESS) {
    return ErrorResult("failed to subscribe decoder events");
  }
  JxlDecoderSetInput(dec, bytes, size);
  JxlDecoderCloseInput(dec);

  jxlpy_result result = {};
  result.ok = 1;
  JxlBasicInfo info = {};
  uint32_t frames = 0;
  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec);
    if (status == JXL_DEC_SUCCESS) {
      break;
    }
    if (status == JXL_DEC_ERROR) {
      return ErrorResult("failed to parse JPEG XL info");
    }
    if (status == JXL_DEC_BASIC_INFO) {
      if (JxlDecoderGetBasicInfo(dec, &info) != JXL_DEC_SUCCESS) {
        return ErrorResult("failed to get JPEG XL basic info");
      }
      result.xsize = info.xsize;
      result.ysize = info.ysize;
      result.bits_per_sample = info.bits_per_sample;
      result.exponent_bits_per_sample = info.exponent_bits_per_sample;
      result.dtype = DefaultDecodeDtype(info);
      result.have_animation = info.have_animation;
      result.num_channels = info.num_color_channels + (info.alpha_bits ? 1 : 0);
      result.num_extra_channels = info.num_extra_channels;
    } else if (status == JXL_DEC_FRAME) {
      ++frames;
    } else if (status == JXL_DEC_NEED_MORE_INPUT) {
      return ErrorResult("truncated JPEG XL input");
    }
  }
  result.num_frames = frames == 0 ? 1 : frames;
  return result;
}

jxlpy_result jxlpy_reconstruct_jpeg(const uint8_t* bytes, size_t size) {
  if (bytes == nullptr || size == 0) return ErrorResult("input bytes are empty");
  if (!IsJxlBytes(bytes, size)) return ErrorResult("input is not JPEG XL");

  JxlDecoder* dec = JxlDecoderCreate(nullptr);
  if (dec == nullptr) return ErrorResult("failed to create JPEG XL decoder");
  std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> dec_ptr(
      dec, JxlDecoderDestroy);
  void* runner = JxlThreadParallelRunnerCreate(
      nullptr, JxlThreadParallelRunnerDefaultNumWorkerThreads());
  RunnerPtr runner_ptr(runner, JxlThreadParallelRunnerDestroy);
  if (runner == nullptr) return ErrorResult("failed to create JPEG XL runner");
  if (JxlDecoderSetParallelRunner(dec, JxlThreadParallelRunner, runner) !=
      JXL_DEC_SUCCESS) {
    return ErrorResult("failed to set JPEG XL runner");
  }
  if (JxlDecoderSubscribeEvents(
          dec, JXL_DEC_JPEG_RECONSTRUCTION | JXL_DEC_FULL_IMAGE) !=
      JXL_DEC_SUCCESS) {
    return ErrorResult("failed to subscribe decoder events");
  }
  JxlDecoderSetInput(dec, bytes, size);
  JxlDecoderCloseInput(dec);

  std::vector<uint8_t> jpeg_out;
  const size_t kChunkSize = 1u << 20;  // 1 MiB chunks
  bool reconstruction_available = false;

  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec);
    if (status == JXL_DEC_ERROR) {
      return ErrorResult("failed to decode JPEG XL for JPEG reconstruction");
    }
    if (status == JXL_DEC_SUCCESS) {
      break;
    }
    if (status == JXL_DEC_NEED_MORE_INPUT) {
      return ErrorResult("truncated JPEG XL input");
    }
    if (status == JXL_DEC_JPEG_RECONSTRUCTION) {
      reconstruction_available = true;
      jpeg_out.resize(kChunkSize);
      if (JxlDecoderSetJPEGBuffer(dec, jpeg_out.data(), jpeg_out.size()) !=
          JXL_DEC_SUCCESS) {
        return ErrorResult("failed to set JPEG reconstruction buffer");
      }
      continue;
    }
    if (status == JXL_DEC_JPEG_NEED_MORE_OUTPUT) {
      size_t remaining = JxlDecoderReleaseJPEGBuffer(dec);
      size_t written = jpeg_out.size() - remaining;
      jpeg_out.resize(jpeg_out.size() + kChunkSize);
      if (JxlDecoderSetJPEGBuffer(dec, jpeg_out.data() + written,
                                   jpeg_out.size() - written) !=
          JXL_DEC_SUCCESS) {
        return ErrorResult("failed to set JPEG reconstruction buffer");
      }
      continue;
    }
    if (status == JXL_DEC_FULL_IMAGE) {
      break;
    }
    // Any other status (e.g. JXL_DEC_NEED_IMAGE_OUT_BUFFER, JXL_DEC_BASIC_INFO)
    // means the file does not contain JPEG reconstruction data.
    if (!reconstruction_available) {
      return ErrorResult("JXL file does not contain JPEG reconstruction data");
    }
    // If we already got reconstruction data but hit an unexpected event, stop.
    break;
  }

  if (!reconstruction_available) {
    return ErrorResult("JXL file does not contain JPEG reconstruction data");
  }

  size_t remaining = JxlDecoderReleaseJPEGBuffer(dec);
  jpeg_out.resize(jpeg_out.size() - remaining);
  return BytesResult(jpeg_out);
}

jxlpy_result jxlpy_decode_to_format(const uint8_t* bytes, size_t size,
                                    const char* extension, int quality) {
  if (bytes == nullptr || size == 0) return ErrorResult("input bytes are empty");
  if (extension == nullptr) return ErrorResult("extension is null");
  auto encoder = jxl::extras::Encoder::FromExtension(extension);
  if (!encoder) {
    return ErrorResult(std::string("unsupported output format: ") + extension);
  }
  if (quality >= 0 && quality <= 100) {
    encoder->SetOption("q", std::to_string(quality));
  }

  jxl::extras::PackedPixelFile ppf;

  if (IsJxlBytes(bytes, size)) {
    // For JXL input, use DecodeImageJXL with the encoder's accepted formats
    // so the decoder outputs pixels in a format the encoder can consume.
    jxl::extras::JXLDecompressParams dparams;
    dparams.accepted_formats = encoder->AcceptedFormats();
    size_t decoded_bytes = 0;
    if (!jxl::extras::DecodeImageJXL(bytes, size, dparams, &decoded_bytes,
                                     &ppf)) {
      return ErrorResult("failed to decode JXL image");
    }
  } else {
    // For PNG/JPEG/other input, DecodeBytes works fine.
    if (!jxl::extras::DecodeBytes(jxl::Bytes(bytes, size),
                                  jxl::extras::ColorHints(), &ppf)) {
      return ErrorResult("failed to decode input image");
    }
    // Fix endianness for non-JXL decoded images.
    for (auto& frame : ppf.frames) {
      if (frame.color.format.endianness == JXL_NATIVE_ENDIAN) {
        frame.color.format.endianness = JXL_LITTLE_ENDIAN;
      }
    }
  }

  jxl::extras::EncodedImage encoded;
  if (!encoder->Encode(ppf, &encoded, nullptr)) {
    return ErrorResult("failed to encode to target format");
  }
  if (encoded.bitstreams.empty() || encoded.bitstreams[0].empty()) {
    return ErrorResult("encoder produced empty output");
  }
  return BytesResult(encoded.bitstreams[0]);
}

}  // extern "C"
