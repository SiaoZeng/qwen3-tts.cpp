cmake_minimum_required(VERSION 3.14)
project(qwen3-tts-ggml VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
endif()

# Timing instrumentation option
option(QWEN3_TTS_TIMING "Enable detailed timing instrumentation" OFF)
if(QWEN3_TTS_TIMING)
    add_compile_definitions(QWEN3_TTS_TIMING)
endif()

# CoreML code predictor bridge (macOS only)
option(QWEN3_TTS_COREML "Enable CoreML code predictor bridge on macOS" ON)
if(APPLE AND QWEN3_TTS_COREML)
    enable_language(OBJCXX)
endif()

# Find threads
find_package(Threads REQUIRED)

# GGML library paths
set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
set(GGML_BUILD_DIR "${GGML_DIR}/build")

# Detect Metal backend availability
if(APPLE AND EXISTS "${GGML_BUILD_DIR}/src/ggml-metal/libggml-metal.dylib")
    set(GGML_HAS_METAL ON)
    message(STATUS "GGML Metal backend found — GPU acceleration enabled")
else()
    set(GGML_HAS_METAL OFF)
endif()

# Text tokenizer library
add_library(text_tokenizer STATIC
    src/text_tokenizer.cpp
    src/gguf_loader.cpp
)
target_include_directories(text_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(text_tokenizer PUBLIC
    ${GGML_BUILD_DIR}/src
    ${GGML_BUILD_DIR}/src/ggml-metal
)
target_link_libraries(text_tokenizer PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
if(GGML_HAS_METAL)
    target_link_libraries(text_tokenizer PUBLIC ggml-metal "-framework Metal" "-framework MetalKit")
endif()

# TTS Transformer library (GGML-based + optional CoreML bridge)
set(TTS_TRANSFORMER_SOURCES
    src/gguf_loader.cpp
    src/tts_transformer.cpp
)
if(APPLE AND QWEN3_TTS_COREML)
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor.mm)
    set_source_files_properties(src/coreml_code_predictor.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor_stub.cpp)
endif()

add_library(tts_transformer STATIC
    ${TTS_TRANSFORMER_SOURCES}
)
target_include_directories(tts_transformer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(tts_transformer PUBLIC
    ${GGML_BUILD_DIR}/src
    ${GGML_BUILD_DIR}/src/ggml-metal
)
target_link_libraries(tts_transformer PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
if(GGML_HAS_METAL)
    target_link_libraries(tts_transformer PUBLIC ggml-metal "-framework Metal" "-framework MetalKit")
endif()
if(APPLE AND QWEN3_TTS_COREML)
    target_link_libraries(tts_transformer PUBLIC
        "-framework Foundation"
        "-framework CoreML"
    )
endif()

# Audio tokenizer encoder library (GGML-based)
add_library(audio_tokenizer_encoder STATIC
    src/audio_tokenizer_encoder.cpp
    src/gguf_loader.cpp
)
target_include_directories(audio_tokenizer_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(audio_tokenizer_encoder PUBLIC
    ${GGML_BUILD_DIR}/src
    ${GGML_BUILD_DIR}/src/ggml-metal
)
target_link_libraries(audio_tokenizer_encoder PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
if(GGML_HAS_METAL)
    target_link_libraries(audio_tokenizer_encoder PUBLIC ggml-metal "-framework Metal" "-framework MetalKit")
endif()

# Audio tokenizer decoder library (GGML-based)
add_library(audio_tokenizer_decoder STATIC
    src/audio_tokenizer_decoder.cpp
    src/gguf_loader.cpp
)
target_include_directories(audio_tokenizer_decoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(audio_tokenizer_decoder PUBLIC
    ${GGML_BUILD_DIR}/src
    ${GGML_BUILD_DIR}/src/ggml-metal
)
target_link_libraries(audio_tokenizer_decoder PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
if(GGML_HAS_METAL)
    target_link_libraries(audio_tokenizer_decoder PUBLIC ggml-metal "-framework Metal" "-framework MetalKit")
endif()

# Qwen3 TTS library (full pipeline)
add_library(qwen3_tts STATIC
    src/qwen3_tts.cpp
)
target_include_directories(qwen3_tts PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(qwen3_tts PUBLIC
    text_tokenizer
    tts_transformer
    audio_tokenizer_encoder
    audio_tokenizer_decoder
    Threads::Threads
)

set_property(TARGET
    text_tokenizer
    tts_transformer
    audio_tokenizer_encoder
    audio_tokenizer_decoder
    qwen3_tts
    PROPERTY POSITION_INDEPENDENT_CODE ON
)

# Shared library with C API (for Nim FFI)
add_library(qwen3tts_shared SHARED
    src/qwen3tts_c_api.cpp
)
set_target_properties(qwen3tts_shared PROPERTIES
    OUTPUT_NAME "qwen3tts"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)
target_include_directories(qwen3tts_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(qwen3tts_shared PRIVATE
    qwen3_tts
)

# CLI executable
add_executable(qwen3-tts-cli
    src/main.cpp
)
target_link_libraries(qwen3-tts-cli PRIVATE
    qwen3_tts
)

# Persistent server executable
add_executable(qwen3-tts-server
    src/server.cpp
)
target_link_libraries(qwen3-tts-server PRIVATE
    qwen3_tts
)

# Test executable for tokenizer
add_executable(test_tokenizer
    tests/test_tokenizer.cpp
)
target_link_libraries(test_tokenizer PRIVATE
    text_tokenizer
    Threads::Threads
)

# Test executable for audio encoder
add_executable(test_encoder
    tests/test_encoder.cpp
)
target_link_libraries(test_encoder PRIVATE
    audio_tokenizer_encoder
    Threads::Threads
)

# Test executable for transformer
add_executable(test_transformer
    tests/test_transformer.cpp
)
target_link_libraries(test_transformer PRIVATE
    tts_transformer
    Threads::Threads
)

# Test executable for audio decoder
add_executable(test_decoder
    tests/test_decoder.cpp
)
target_link_libraries(test_decoder PRIVATE
    audio_tokenizer_decoder
    Threads::Threads
)

add_executable(test_streaming_pipeline
    tests/test_streaming_pipeline.cpp
)
target_link_libraries(test_streaming_pipeline PRIVATE
    qwen3_tts
    Threads::Threads
)

# Install targets
install(TARGETS text_tokenizer tts_transformer audio_tokenizer_encoder audio_tokenizer_decoder qwen3_tts qwen3tts_shared qwen3-tts-cli
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES
    src/gguf_loader.h
    src/text_tokenizer.h
    src/tts_transformer.h
    src/coreml_code_predictor.h
    src/audio_tokenizer_encoder.h
    src/audio_tokenizer_decoder.h
    src/qwen3_tts.h
    src/qwen3tts_c_api.h
    DESTINATION include
)

# CTest support
enable_testing()

set(QWEN3_TTS_LOCAL_SHARE "$ENV{HOME}/.local/share/qwen3-tts")
set(QWEN3_TTS_TEST_MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models")
if(EXISTS "${QWEN3_TTS_LOCAL_SHARE}/models/qwen3-tts-0.6b-f16.gguf")
    set(QWEN3_TTS_TEST_MODEL_DIR "${QWEN3_TTS_LOCAL_SHARE}/models")
endif()

set(QWEN3_TTS_TEST_TTS_MODEL "${QWEN3_TTS_TEST_MODEL_DIR}/qwen3-tts-0.6b-f16.gguf")
set(QWEN3_TTS_TEST_TOKENIZER_MODEL "${QWEN3_TTS_TEST_MODEL_DIR}/qwen3-tts-tokenizer-f16.gguf")
set(QWEN3_TTS_TEST_AUDIO "${CMAKE_CURRENT_SOURCE_DIR}/clone.wav")
set(QWEN3_TTS_TEST_ENCODER_REF "${CMAKE_CURRENT_SOURCE_DIR}/reference/ref_audio_embedding.bin")
set(QWEN3_TTS_TEST_TRANSFORMER_TOKENS "${CMAKE_CURRENT_SOURCE_DIR}/reference/det_text_tokens.bin")
set(QWEN3_TTS_TEST_TRANSFORMER_SPEAKER "${CMAKE_CURRENT_SOURCE_DIR}/reference/det_speaker_embedding.bin")
set(QWEN3_TTS_TEST_DECODER_CODES "${CMAKE_CURRENT_SOURCE_DIR}/reference/speech_codes.bin")

add_test(NAME tokenizer_test
    COMMAND test_tokenizer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
if(EXISTS "${QWEN3_TTS_TEST_TTS_MODEL}" AND EXISTS "${QWEN3_TTS_TEST_AUDIO}")
    set(QWEN3_TTS_ENCODER_TEST_CMD test_encoder --tokenizer ${QWEN3_TTS_TEST_TTS_MODEL} --audio ${QWEN3_TTS_TEST_AUDIO})
    if(EXISTS "${QWEN3_TTS_TEST_ENCODER_REF}")
        list(APPEND QWEN3_TTS_ENCODER_TEST_CMD --reference ${QWEN3_TTS_TEST_ENCODER_REF})
    endif()
    add_test(NAME encoder_test
        COMMAND ${QWEN3_TTS_ENCODER_TEST_CMD}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
else()
    message(STATUS "Skipping encoder_test: required model/audio artifacts not found")
endif()

if(EXISTS "${QWEN3_TTS_TEST_TTS_MODEL}" AND EXISTS "${QWEN3_TTS_TEST_TRANSFORMER_TOKENS}" AND EXISTS "${QWEN3_TTS_TEST_TRANSFORMER_SPEAKER}")
    add_test(NAME transformer_test
        COMMAND test_transformer --model ${QWEN3_TTS_TEST_TTS_MODEL} --ref-dir ${CMAKE_CURRENT_SOURCE_DIR}/reference/
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
else()
    message(STATUS "Skipping transformer_test: required model/reference artifacts not found")
endif()

if(EXISTS "${QWEN3_TTS_TEST_TOKENIZER_MODEL}" AND EXISTS "${QWEN3_TTS_TEST_DECODER_CODES}")
    set(QWEN3_TTS_DECODER_TEST_CMD test_decoder --tokenizer ${QWEN3_TTS_TEST_TOKENIZER_MODEL} --codes ${QWEN3_TTS_TEST_DECODER_CODES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reference/decoded_audio.bin")
        list(APPEND QWEN3_TTS_DECODER_TEST_CMD --reference ${CMAKE_CURRENT_SOURCE_DIR}/reference/decoded_audio.bin)
    endif()
    add_test(NAME decoder_test
        COMMAND ${QWEN3_TTS_DECODER_TEST_CMD}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
else()
    message(STATUS "Skipping decoder_test: required model/code artifacts not found")
endif()
add_test(NAME live_abort_recovery_test
    COMMAND ${CMAKE_COMMAND} -E env QWEN3_TTS_TEST_PORT=19001 bash tests/test_live_abort_recovery.sh ${CMAKE_CURRENT_BINARY_DIR}/qwen3-tts-server
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME post_split_body_test
    COMMAND ${CMAKE_COMMAND} -E env QWEN3_TTS_TEST_PORT=19002 bash tests/test_post_split_body.sh ${CMAKE_CURRENT_BINARY_DIR}/qwen3-tts-server
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME post_incomplete_body_test
    COMMAND ${CMAKE_COMMAND} -E env QWEN3_TTS_TEST_PORT=19003 bash tests/test_post_incomplete_body.sh ${CMAKE_CURRENT_BINARY_DIR}/qwen3-tts-server
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME live_duration_matches_stream_test
    COMMAND ${CMAKE_COMMAND} -E env QWEN3_TTS_TEST_PORT=19004 bash tests/test_live_duration_matches_stream.sh ${CMAKE_CURRENT_BINARY_DIR}/qwen3-tts-server
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
if(EXISTS "${QWEN3_TTS_TEST_MODEL_DIR}" AND EXISTS "${QWEN3_TTS_LOCAL_SHARE}/voices/lina.bin")
    add_test(NAME streaming_pipeline_test
        COMMAND test_streaming_pipeline --models ${QWEN3_TTS_TEST_MODEL_DIR} --embedding ${QWEN3_TTS_LOCAL_SHARE}/voices/lina.bin
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
else()
    message(STATUS "Skipping streaming_pipeline_test: required model/embedding artifacts not found")
endif()
