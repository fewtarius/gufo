# Laguna S 2.1 model extension.
#
# The CPU-only half carries the config parser, the chat template and the
# reference implementation. The HIP half carries the GPU engine, its kernels
# and the text runner that plugs the model into the serve layer.

gufo_extension_target(
  NAME gufo_laguna
  SOURCES
    runtime/config.cpp
    chat_template.cpp
    reference.cpp
    tool_calls.cpp
    probe.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_core
    Threads::Threads
)

if(TARGET gufo_laguna)
  target_include_directories(gufo_laguna PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()

# Declared unconditionally: the registry validates that every target named
# in extension.json exists in the build, and gufo_extension_target
# early-returns for a disabled (HIP-off) target. Declaring inside
# if(ENGINE_ENABLE_HIP) left the target undefined and broke any HIP-off
# configure.
gufo_extension_target(
  NAME gufo_laguna_hip
  SOURCES
    gpu/device_model.cpp
    gpu/engine.cpp
    continuation_adapter.cpp
    runner.cpp
    drivers.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_laguna
    gufo_qwen38_flash_next
    gufo_qwen38_flash_next_mmq
    gufo_qwen_text
    gufo_cache_hip
    gufo_compare
    hip::host
    roc::hipblas
    roc::hipblaslt
)

if(TARGET gufo_laguna_hip)
  gufo_extension_hip_sources(gufo_laguna_hip
    SOURCES
      gpu/kernels/kernels.hip.cpp
  )
  target_compile_definitions(gufo_laguna_hip PRIVATE QFN_MMQ_BUILD=1)
  target_compile_options(gufo_laguna_hip PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math;-fno-finite-math-only;-Wno-unused-value;-Wno-unused-command-line-argument>)
  target_include_directories(gufo_laguna_hip SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})
  target_include_directories(gufo_laguna_hip PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${GUFO_EXTENSION_BUILD_ROOT}
  )
endif()
