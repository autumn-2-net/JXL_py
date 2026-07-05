cmake_minimum_required(VERSION 3.16)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING
    "Minimum CMake policy compatibility for bundled third-party projects.")

project(jxl_python_superbuild LANGUAGES C CXX)

set(JXL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libjxl" CACHE PATH
    "Path to the libjxl source tree.")

if(NOT EXISTS "${JXL_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
      "JXL_SOURCE_DIR must point at a complete libjxl source tree. "
      "Expected: ${JXL_SOURCE_DIR}/CMakeLists.txt")
endif()

option(JXL_SUPERBUILD_ENABLE_CLI "Build libjxl command line tools." ON)
option(JXL_SUPERBUILD_BUNDLE_JPEG "Use bundled libjpeg-turbo for JPEG I/O." ON)
option(JXL_PYTHON_ENABLE_NATIVE "Build the jxlpy C ABI native library." ON)

# Keep the first build focused: core libraries plus the user-facing CLI.
# Presets or users can override these cache entries when a broader build is
# needed.
set(BUILD_TESTING OFF CACHE BOOL "Build tests.")
set(JPEGXL_ENABLE_TOOLS ${JXL_SUPERBUILD_ENABLE_CLI} CACHE BOOL
    "Build JPEG XL user tools.")
set(JPEGXL_ENABLE_DEVTOOLS OFF CACHE BOOL "Build JPEG XL developer tools.")
set(JPEGXL_ENABLE_FUZZERS OFF CACHE BOOL "Build JPEG XL fuzzer targets.")
set(JPEGXL_ENABLE_BENCHMARK OFF CACHE BOOL "Build JPEG XL benchmark tools.")
set(JPEGXL_ENABLE_EXAMPLES OFF CACHE BOOL "Build JPEG XL examples.")
set(JPEGXL_ENABLE_DOXYGEN OFF CACHE BOOL "Generate API documentation.")
set(JPEGXL_ENABLE_MANPAGES OFF CACHE BOOL "Build command line man pages.")
set(JPEGXL_ENABLE_JNI OFF CACHE BOOL "Build JNI wrapper.")
set(JPEGXL_ENABLE_OPENEXR OFF CACHE BOOL "Build OpenEXR support.")
set(JPEGXL_ENABLE_PLUGINS OFF CACHE BOOL "Build third-party plugins.")
set(JPEGXL_ENABLE_VIEWERS OFF CACHE BOOL "Build viewer tools.")
set(JPEGXL_ENABLE_TCMALLOC OFF CACHE BOOL "Use tcmalloc allocator.")
set(JPEGXL_ENABLE_SJPEG ON CACHE BOOL "Build sjpeg support.")
set(JPEGXL_BUNDLE_LIBPNG ON CACHE BOOL "Build libpng from bundled sources.")

if(JXL_SUPERBUILD_BUNDLE_JPEG AND
   EXISTS "${JXL_SOURCE_DIR}/third_party/libjpeg-turbo/CMakeLists.txt")
  set(ENABLE_SHARED OFF CACHE BOOL "Build libjpeg-turbo shared libraries.")
  set(ENABLE_STATIC ON CACHE BOOL "Build libjpeg-turbo static libraries.")
  set(WITH_SIMD OFF CACHE BOOL "Build libjpeg-turbo SIMD code.")
  set(WITH_TURBOJPEG OFF CACHE BOOL "Build the TurboJPEG API library.")
  set(WITH_JAVA OFF CACHE BOOL "Build the TurboJPEG Java wrapper.")
  set(WITH_FUZZ OFF CACHE BOOL "Build libjpeg-turbo fuzz targets.")
  add_subdirectory("${JXL_SOURCE_DIR}/third_party/libjpeg-turbo"
                   "${CMAKE_BINARY_DIR}/libjpeg-turbo" EXCLUDE_FROM_ALL)
  if(TARGET jpeg-static)
    set(JPEGXL_BUNDLED_JPEG_INCLUDE_DIRS
        "${JXL_SOURCE_DIR}/third_party/libjpeg-turbo"
        "${CMAKE_BINARY_DIR}/libjpeg-turbo")
  endif()
endif()

add_subdirectory("${JXL_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/libjxl")

if(TARGET cjxl AND TARGET djxl)
  add_custom_target(jxl_cli DEPENDS cjxl djxl)
  if(TARGET jxlinfo)
    add_dependencies(jxl_cli jxlinfo)
  endif()
  if(TARGET jxltran)
    add_dependencies(jxl_cli jxltran)
  endif()
endif()

set(_jxl_library_targets)
foreach(_target IN ITEMS jxl jxl_dec jxl_threads jxl_cms)
  if(TARGET "${_target}")
    list(APPEND _jxl_library_targets "${_target}")
  endif()
endforeach()

if(_jxl_library_targets)
  add_custom_target(jxl_libraries DEPENDS ${_jxl_library_targets})
endif()

if(JXL_PYTHON_ENABLE_NATIVE)
  add_library(jxlpy_native SHARED
      native/jxlpy_native.cc
      native/jxlpy_native.h)
  target_compile_features(jxlpy_native PRIVATE cxx_std_17)
  target_include_directories(jxlpy_native PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}"
      "${JXL_SOURCE_DIR}"
      "${JXL_SOURCE_DIR}/lib/include"
      "${CMAKE_BINARY_DIR}/libjxl/lib/include")
  target_link_libraries(jxlpy_native PRIVATE
      jxl
      jxl_threads)
  if(TARGET jxl_extras_codec)
    target_link_libraries(jxlpy_native PRIVATE jxl_extras_codec)
  endif()
  if(TARGET jxl_cms)
    target_link_libraries(jxlpy_native PRIVATE jxl_cms)
  endif()
  if(WIN32)
    set_target_properties(jxlpy_native PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS OFF)
  endif()
endif()
