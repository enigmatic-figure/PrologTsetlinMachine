cmake_minimum_required(VERSION 3.20)
file(STRINGS
    "${CMAKE_CURRENT_SOURCE_DIR}/python/prolog_tsetlin/_version.py"
    PTM_VERSION_LINE
    REGEX "^__version__ = \"[0-9]+\\.[0-9]+\\.[0-9]+\"$")
if(NOT PTM_VERSION_LINE)
    message(FATAL_ERROR "Could not read the PTM version")
endif()
string(REGEX REPLACE
    "^__version__ = \"([0-9]+\\.[0-9]+\\.[0-9]+)\"$" "\\1"
    PTM_VERSION "${PTM_VERSION_LINE}")
project(PrologTsetlinMachine VERSION ${PTM_VERSION} LANGUAGES C CXX)

find_package(Threads REQUIRED)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

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
set_target_properties(ptm_core PROPERTIES EXPORT_NAME core)

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
    EXPORT_NAME runtime
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
target_include_directories(ptm_c_api
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(ptm_c_api PRIVATE cxx_std_20)
target_compile_definitions(ptm_c_api PRIVATE PTM_C_API_BUILD)
set_target_properties(ptm_c_api PROPERTIES OUTPUT_NAME ptm)
set_target_properties(ptm_c_api PROPERTIES EXPORT_NAME c_api)

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
    target_include_directories(ptm_model_runtime_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_features(ptm_model_runtime_tests PRIVATE cxx_std_20)
    target_compile_definitions(ptm_model_runtime_tests PRIVATE
        PTM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
        PTM_TEST_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}")
    add_test(NAME ptm_model_runtime_tests COMMAND ptm_model_runtime_tests)

    if(PTM_BUILD_RUNTIME_CLI)
        set(PTM_PTA_THRESHOLD_HEX_FIXTURE
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/pta_threshold_clause_v1.hex")
        set(PTM_PTA_THRESHOLD_FIXTURE
            "${CMAKE_CURRENT_BINARY_DIR}/pta_threshold_clause_v1.ptm")
        add_executable(ptm_test_fixture_materializer
            tests/cpp/materialize_hex_fixture.cpp)
        target_compile_features(ptm_test_fixture_materializer PRIVATE cxx_std_20)
        add_test(NAME ptmrt_pta_threshold_fixture
            COMMAND ptm_test_fixture_materializer
                "${PTM_PTA_THRESHOLD_HEX_FIXTURE}"
                "${PTM_PTA_THRESHOLD_FIXTURE}")
        set_tests_properties(ptmrt_pta_threshold_fixture PROPERTIES
            FIXTURES_SETUP ptmrt_pta_threshold_artifact)
        add_test(NAME ptmrt_pta_threshold_verify
            COMMAND ptmrt verify "${PTM_PTA_THRESHOLD_FIXTURE}")
        set_tests_properties(ptmrt_pta_threshold_verify PROPERTIES
            FIXTURES_REQUIRED ptmrt_pta_threshold_artifact
            PASS_REGULAR_EXPRESSION "verified sha256:")
        add_test(NAME ptmrt_pta_threshold_below
            COMMAND ptmrt run-record "${PTM_PTA_THRESHOLD_FIXTURE}"
                "temperature:float=74.5")
        set_tests_properties(ptmrt_pta_threshold_below PROPERTIES
            FIXTURES_REQUIRED ptmrt_pta_threshold_artifact
            PASS_REGULAR_EXPRESSION "prediction.:0.*score.:0")
        add_test(NAME ptmrt_pta_threshold_boundary
            COMMAND ptmrt run-record "${PTM_PTA_THRESHOLD_FIXTURE}"
                "temperature:float=75")
        set_tests_properties(ptmrt_pta_threshold_boundary PROPERTIES
            FIXTURES_REQUIRED ptmrt_pta_threshold_artifact
            PASS_REGULAR_EXPRESSION "prediction.:1.*score.:1")

        set(PTM_TRAINED_PARENT_HEX_FIXTURE
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/trained_parent_child_v1.hex")
        set(PTM_TRAINED_PARENT_FIXTURE
            "${CMAKE_CURRENT_BINARY_DIR}/trained_parent_child_v1.ptm")
        add_test(NAME ptmrt_trained_parent_fixture
            COMMAND ptm_test_fixture_materializer
                "${PTM_TRAINED_PARENT_HEX_FIXTURE}"
                "${PTM_TRAINED_PARENT_FIXTURE}")
        set_tests_properties(ptmrt_trained_parent_fixture PROPERTIES
            FIXTURES_SETUP ptmrt_trained_parent_artifact)
        add_test(NAME ptmrt_trained_parent_verify
            COMMAND ptmrt verify "${PTM_TRAINED_PARENT_FIXTURE}")
        set_tests_properties(ptmrt_trained_parent_verify PROPERTIES
            FIXTURES_REQUIRED ptmrt_trained_parent_artifact
            PASS_REGULAR_EXPRESSION
                "verified sha256:4330436bdabed010bfd84b581fc4b0d447a6db749d1114d5094eb2ade662e3ae")
        add_test(NAME ptmrt_trained_parent_below
            COMMAND ptmrt run-record "${PTM_TRAINED_PARENT_FIXTURE}"
                "temperature:int=74"
                "mode:string=heat"
                "previous_state:bool=false")
        set_tests_properties(ptmrt_trained_parent_below PROPERTIES
            FIXTURES_REQUIRED ptmrt_trained_parent_artifact
            PASS_REGULAR_EXPRESSION "prediction.:0.*score.:-5")
        add_test(NAME ptmrt_trained_parent_above
            COMMAND ptmrt run-record "${PTM_TRAINED_PARENT_FIXTURE}"
                "temperature:int=76"
                "mode:string=heat"
                "previous_state:bool=false")
        set_tests_properties(ptmrt_trained_parent_above PROPERTIES
            FIXTURES_REQUIRED ptmrt_trained_parent_artifact
            PASS_REGULAR_EXPRESSION "prediction.:1.*score.:6")
    endif()

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

if(PTM_BUILD_TESTS OR PTM_BUILD_BENCHMARKS)
    add_executable(ptm_campaign_native_runner
        benchmarks/campaign_native_runner.cpp)
    target_link_libraries(ptm_campaign_native_runner PRIVATE PTM::core)
    target_compile_features(ptm_campaign_native_runner PRIVATE cxx_std_20)
    if(PTM_BUILD_TESTS)
        add_test(NAME ptm_campaign_native_runner_smoke
            COMMAND ptm_campaign_native_runner
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/campaign_binary_xor.txt"
                4 10 3.0 5 1 7 2 1 1
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/data/campaign_binary_xor.txt"
                "${CMAKE_CURRENT_BINARY_DIR}/campaign_binary_predictions.txt")
        set_tests_properties(ptm_campaign_native_runner_smoke PROPERTIES
            PASS_REGULAR_EXPRESSION
                "ptm.native-campaign-runner.v2.*runtime_conformance_mismatches.*0")
    endif()
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

    add_executable(ptm_mnist_ovr_benchmark benchmarks/mnist_ovr_benchmark.cpp)
    target_link_libraries(ptm_mnist_ovr_benchmark PRIVATE PTM::core)
    target_compile_features(ptm_mnist_ovr_benchmark PRIVATE cxx_std_20)
endif()

set(PTM_INSTALL_TARGETS ptm_core ptm_runtime ptm_c_api)
if(PTM_BUILD_RUNTIME_CLI)
    list(APPEND PTM_INSTALL_TARGETS ptmrt)
endif()

install(TARGETS ${PTM_INSTALL_TARGETS}
    EXPORT PTMTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

configure_package_config_file(
    cmake/PTMConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/PTMConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PTM
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/PTMConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)
install(EXPORT PTMTargets
    FILE PTMTargets.cmake
    NAMESPACE PTM::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PTM
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/PTMConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/PTMConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PTM
)
install(FILES LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/prolog-tsetlin-machine)
