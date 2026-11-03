# Laguna chat template and tool-call dialect checks. The golden cases come
# from the artifact's own Jinja template, so they need no model artifact.
add_executable(laguna_chat_template_test chat_template_test.cpp)
target_link_libraries(laguna_chat_template_test PRIVATE gufo_laguna)
target_compile_definitions(laguna_chat_template_test PRIVATE
  GUFO_LAGUNA_CHAT_CASES="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/chat_cases.json")
add_test(NAME laguna.chat_template COMMAND laguna_chat_template_test)
set_tests_properties(laguna.chat_template PROPERTIES LABELS "cpu;laguna;tokenization")

add_executable(laguna_tool_calls_test
  tool_calls_test.cpp
  ${PROJECT_SOURCE_DIR}/src/models/laguna/tool_calls.cpp)
target_link_libraries(laguna_tool_calls_test PRIVATE gufo_laguna)
add_test(NAME laguna.tool_calls COMMAND laguna_tool_calls_test)
set_tests_properties(laguna.tool_calls PROPERTIES LABELS "cpu;laguna;tokenization")

if(ENGINE_ENABLE_HIP)
  # The device q8_0 KV path has to agree with the host codec byte for byte.
  # The extension's HIP tier already carries the device kernels, so the test
  # links it rather than compiling them again.
  add_executable(laguna_kv_q8_test kv_q8_test.cpp)
  target_compile_definitions(laguna_kv_q8_test PRIVATE ENGINE_ENABLE_HIP=1)
  target_link_libraries(laguna_kv_q8_test PRIVATE
    gufo_laguna_hip hip::host)
  add_test(NAME laguna.kv_q8 COMMAND laguna_kv_q8_test)
  set_tests_properties(laguna.kv_q8 PROPERTIES LABELS "hip;laguna;kv;quant")
endif()
