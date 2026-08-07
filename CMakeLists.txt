cmake_minimum_required(VERSION 3.20)
project(PrologTsetlinMachine VERSION 0.1.0 LANGUAGES C CXX)

find_package(Threads REQUIRED)

option(PTM_BUILD_TESTS "Build PTM native tests" ON)
option(PTM_BUILD_BENCHMARKS "Build PTM microbenchmarks" OFF)
option(PTM_BUILD_EXAMPLES "Build PTM native examples" ON)
option(PTM_BUILD_RUNTIME_CLI "Build the standalone ptmrt inference CLI" ON)
option(PTM_ENABLE_X86_SIMD "Build x86 AVX2 and AVX-512 TM backends" ON)
option(PTM_ENABLE_CUDA "Build experimental CUDA TM backends" OFF)

add_library(ptm_core
    src/class_ii_persistence.cpp
    src/consolidation_registry.cpp
    src/concurrent_mapping.cpp
    src/logic_ir.cpp
    src/logic_program.cpp
    src/packed_tm.cpp
    src/packed_tm_dispatch.cpp
    src/pa_kernel.cpp
    src/scalar_tm.cpp
    src/shadow_audit.cpp
)
add_library(PTM::core ALIAS ptm_core)
set_target_properties(ptm_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(ptm_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(ptm_core PUBLIC cxx_std_20)

set(PTM_X86_NATIVE_BACKENDS OFF)
if(PTM_ENABLE_X86_SIMD AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86|X86|x86_64|X86_64|i.86)$")
    set(PTM_X86_NATIVE_BACKENDS ON)
endif()

if(PTM_X86_NATIVE_BACKENDS)
    target_sources(ptm_core PRIVATE
        src/packed_tm_avx2.cpp
        src/packed_tm_avx512.cpp
    )
    target_compile_definitions(ptm_core PRIVATE
        PTM_COMPILED_AVX2=1
        PTM_COMPILED_AVX512=1
    )
    if(MSVC)
        set_source_files_properties(src/packed_tm_avx2.cpp PROPERTIES
            COMPILE_OPTIONS "/arch:AVX2")
        set_source_files_properties(src/packed_tm_avx512.cpp PROPERTIES
            COMPILE_OPTIONS "/arch:AVX512")
    else()
        set_source_files_properties(src/packed_tm_avx2.cpp PROPERTIES
            COMPILE_OPTIONS "-mavx2")
        set_source_files_properties(src/packed_tm_avx512.cpp PROPERTIES
            COMPILE_OPTIONS "-mavx512f")
    endif()
endif()

if(MSVC)
    target_compile_options(ptm_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(ptm_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_library(ptm_runtime STATIC src/model_runtime.cpp)
add_library(PTM::runtime ALIAS ptm_runtime)
target_include_directories(ptm_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(ptm_runtime PUBLIC cxx_std_20)
set_target_properties(ptm_runtime PROPERTIES
    OUTPUT_NAME ptmrt
    POSITION_INDEPENDENT_CODE ON
)
if(MSVC)
    target_compile_options(ptm_runtime PRIVATE /W4 /permissive-)
else()
    target_compile_options(ptm_runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(PTM_BUILD_RUNTIME_CLI)
    add_executable(ptmrt tools/ptmrt.cpp)
    target_link_libraries(ptmrt PRIVATE PTM::runtime)
    target_compile_features(ptmrt PRIVATE cxx_std_20)
endif()

if(PTM_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "PTM_ENABLE_CUDA=ON, but no CUDA compiler was found")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(ptm_cuda STATIC
        src/packed_tm_cuda.cpp
        src/packed_tm_cuda_kernels.cu
    )
    add_library(PTM::cuda ALIAS ptm_cuda)
    target_link_libraries(ptm_cuda PUBLIC PTM::core PRIVATE CUDA::cudart)
    target_include_directories(ptm_cuda
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_features(ptm_cuda PUBLIC cxx_std_20)
    set_target_properties(ptm_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_definitions(ptm_cuda PUBLIC PTM_HAS_CUDA=1)
    if(MSVC)
        target_compile_options(ptm_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CXX>:/permissive->)
    else()
        target_compile_options(ptm_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
    endif()
endif()

add_library(ptm_c_api SHARED src/c_api.cpp)
add_library(PTM::c_api ALIAS ptm_c_api)
target_link_libraries(ptm_c_api PRIVATE PTM::core)
target_include_directories(ptm_c_api PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(ptm_c_api PRIVATE cxx_std_20)
target_compile_definitions(ptm_c_api PRIVATE PTM_C_API_BUILD)
set_target_properties(ptm_c_api PROPERTIES OUTPUT_NAME ptm)

if(MSVC)
    target_compile_options(ptm_c_api PRIVATE /W4 /permissive-)
else()
    target_compile_options(ptm_c_api PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(PTM_BUILD_TESTS)
    enable_testing()
    add_executable(ptm_core_tests tests/cpp/core_tests.cpp)
    target_link_libraries(ptm_core_tests PRIVATE PTM::core)
    target_compile_features(ptm_core_tests PRIVATE cxx_std_20)
    add_test(NAME ptm_core_tests COMMAND ptm_core_tests)

    add_executable(ptm_c_api_smoke tests/c/c_api_smoke.c)
    target_link_libraries(ptm_c_api_smoke PRIVATE PTM::c_api)
    target_compile_features(ptm_c_api_smoke PRIVATE c_std_11)
    add_test(NAME ptm_c_api_smoke COMMAND ptm_c_api_smoke)

    add_executable(ptm_class_ii_tests tests/cpp/class_ii_tests.cpp)
    target_link_libraries(ptm_class_ii_tests PRIVATE PTM::core Threads::Threads)
    target_compile_features(ptm_class_ii_tests PRIVATE cxx_std_20)
    add_test(NAME ptm_class_ii_tests COMMAND ptm_class_ii_tests)

    add_executable(ptm_class_ii_persistence_tests
        tests/cpp/class_ii_persistence_tests.cpp)
    target_link_libraries(ptm_class_ii_persistence_tests PRIVATE PTM::core)
    target_compile_features(ptm_class_ii_persistence_tests PRIVATE cxx_std_20)
    add_test(NAME ptm_class_ii_persistence_tests
        COMMAND ptm_class_ii_persistence_tests)

    add_executable(ptm_logic_ir_tests tests/cpp/logic_ir_tests.cpp)
    target_link_libraries(ptm_logic_ir_tests PRIVATE PTM::core)
    target_compile_features(ptm_logic_ir_tests PRIVATE cxx_std_20)
    add_test(NAME ptm_logic_ir_tests COMMAND ptm_logic_ir_tests)

    add_executable(ptm_logic_program_tests tests/cpp/logic_program_tests.cpp)
    target_link_libraries(ptm_logic_program_tests PRIVATE PTM::core)
    target_compile_features(ptm_logic_program_tests PRIVATE cxx_std_20)
    add_test(NAME ptm_logic_program_tests COMMAND ptm_logic_program_tests)

    add_executable(ptm_packed_tm_tests tests/cpp/packed_tm_tests.cpp)
    target_link_libraries(ptm_packed_tm_tests PRIVATE PTM::core)
    target_compile_features(ptm_packed_tm_tests PRIVATE cxx_std_20)
    add_test(NAME ptm_packed_tm_tests COMMAND ptm_packed_tm_tests)

    add_executable(ptm_model_runtime_tests
        tests/cpp/model_runtime_tests.cpp)
    target_link_libraries(ptm_model_runtime_tests
        PRIVATE PTM::runtime Threads::Threads)
    target_compile_features(ptm_model_runtime_tests PRIVATE cxx_std_20)
    target_compile_definitions(ptm_model_runtime_tests PRIVATE
        PTM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
        PTM_TEST_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}")
    add_test(NAME ptm_model_runtime_tests COMMAND ptm_model_runtime_tests)

    add_executable(ptm_runtime_c_smoke tests/c/runtime_smoke.c)
    target_link_libraries(ptm_runtime_c_smoke PRIVATE PTM::runtime)
    target_compile_features(ptm_runtime_c_smoke PRIVATE c_std_11)
    add_test(NAME ptm_runtime_c_smoke COMMAND ptm_runtime_c_smoke)

    if(PTM_ENABLE_CUDA)
        add_executable(ptm_packed_tm_cuda_tests
            tests/cpp/packed_tm_cuda_tests.cpp)
        target_link_libraries(ptm_packed_tm_cuda_tests
            PRIVATE PTM::core PTM::cuda)
        target_compile_features(ptm_packed_tm_cuda_tests PRIVATE cxx_std_20)
        add_test(NAME ptm_packed_tm_cuda_tests
            COMMAND ptm_packed_tm_cuda_tests)
    endif()
endif()

if(PTM_BUILD_EXAMPLES)
    add_executable(ptm_class_ii_persistence_example
        examples/class_ii_persistence.cpp)
    target_link_libraries(ptm_class_ii_persistence_example PRIVATE PTM::core)
    target_compile_features(ptm_class_ii_persistence_example PRIVATE cxx_std_20)
endif()

if(PTM_BUILD_BENCHMARKS)
    add_executable(ptm_mapping_benchmark benchmarks/mapping_benchmark.cpp)
    target_link_libraries(ptm_mapping_benchmark PRIVATE PTM::core)
    target_compile_features(ptm_mapping_benchmark PRIVATE cxx_std_20)

    add_executable(ptm_logic_benchmark benchmarks/logic_compiler_benchmark.cpp)
    target_link_libraries(ptm_logic_benchmark PRIVATE PTM::core)
    target_compile_features(ptm_logic_benchmark PRIVATE cxx_std_20)

    add_executable(ptm_packed_tm_benchmark benchmarks/packed_tm_benchmark.cpp)
    target_link_libraries(ptm_packed_tm_benchmark PRIVATE PTM::core)
    if(PTM_ENABLE_CUDA)
        target_link_libraries(ptm_packed_tm_benchmark PRIVATE PTM::cuda)
    endif()
    target_compile_features(ptm_packed_tm_benchmark PRIVATE cxx_std_20)
endif()
