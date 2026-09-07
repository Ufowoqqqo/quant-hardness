cmake_minimum_required(VERSION 3.24)

project(quant_hardness LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(QH_FAISS_EXPECTED_COMMIT
    "20f14b31a6d54e243a3d1de6ae193fc4c3ec18ed")
execute_process(
    COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss rev-parse HEAD
    OUTPUT_VARIABLE QH_FAISS_ACTUAL_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE QH_FAISS_REVISION_RESULT)
if(NOT QH_FAISS_REVISION_RESULT EQUAL 0 OR
   NOT QH_FAISS_ACTUAL_COMMIT STREQUAL QH_FAISS_EXPECTED_COMMIT)
  message(FATAL_ERROR
      "FAISS must be initialized at ${QH_FAISS_EXPECTED_COMMIT}; found "
      "'${QH_FAISS_ACTUAL_COMMIT}'. Run git submodule update --init.")
endif()

execute_process(
    COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse HEAD
    OUTPUT_VARIABLE QH_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE QH_GIT_REVISION_RESULT)
if(NOT QH_GIT_REVISION_RESULT EQUAL 0)
  message(FATAL_ERROR "Unable to record the superproject Git commit")
endif()
execute_process(
    COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} status --porcelain
    OUTPUT_VARIABLE QH_GIT_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(QH_GIT_STATUS STREQUAL "")
  set(QH_GIT_DIRTY false)
else()
  set(QH_GIT_DIRTY true)
endif()

include(ExternalProject)
set(QH_FAISS_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/faiss-build)
ExternalProject_Add(faiss_external
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss
    BINARY_DIR ${QH_FAISS_BUILD_DIR}
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DFAISS_ENABLE_GPU=OFF
        -DFAISS_ENABLE_PYTHON=OFF
        -DFAISS_ENABLE_C_API=OFF
        -DFAISS_ENABLE_EXTRAS=OFF
        -DFAISS_ENABLE_MKL=OFF
        -DFAISS_ENABLE_SVS=OFF
        -DFAISS_OPT_LEVEL=generic
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_TESTING=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target faiss
    BUILD_BYPRODUCTS ${QH_FAISS_BUILD_DIR}/faiss/libfaiss.a
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND "")

add_library(faiss_pinned STATIC IMPORTED GLOBAL)
set_target_properties(faiss_pinned PROPERTIES
    IMPORTED_LOCATION ${QH_FAISS_BUILD_DIR}/faiss/libfaiss.a
    INTERFACE_INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss)
add_dependencies(faiss_pinned faiss_external)

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenSSL REQUIRED)

add_library(qh_faiss_backend
    src/graph/faiss_shared_hnsw.cpp)
target_include_directories(qh_faiss_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(qh_faiss_backend PUBLIC
    faiss_pinned
    OpenSSL::Crypto
    OpenMP::OpenMP_CXX
    ${LAPACK_LIBRARIES}
    ${BLAS_LIBRARIES})
target_compile_options(qh_faiss_backend PRIVATE -Wall -Wextra -Wpedantic)

add_library(qh_metrics
    src/metrics/candidate_oracle.cpp
    src/metrics/ground_truth.cpp
    src/metrics/recall.cpp)
target_include_directories(qh_metrics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(qh_metrics PUBLIC faiss_pinned)
target_compile_options(qh_metrics PRIVATE -Wall -Wextra -Wpedantic)

add_library(qh_instrumentation
    src/instrumentation/faiss_distance_recorder.cpp)
target_include_directories(qh_instrumentation PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(qh_instrumentation PUBLIC faiss_pinned)
target_compile_options(qh_instrumentation PRIVATE
    -Wall -Wextra -Wpedantic)

add_executable(faiss_backend_validation
    tests/faiss_backend_validation.cpp)
target_link_libraries(faiss_backend_validation PRIVATE qh_faiss_backend)
target_compile_definitions(faiss_backend_validation PRIVATE
    QH_FAISS_COMMIT="${QH_FAISS_EXPECTED_COMMIT}")
target_compile_options(faiss_backend_validation PRIVATE
    -Wall -Wextra -Wpedantic)

add_executable(faiss_paired_recall
    src/experiments/faiss_paired_recall.cpp)
target_link_libraries(faiss_paired_recall PRIVATE
    qh_faiss_backend qh_metrics)
target_compile_definitions(faiss_paired_recall PRIVATE
    QH_FAISS_COMMIT="${QH_FAISS_EXPECTED_COMMIT}"
    QH_GIT_COMMIT="${QH_GIT_COMMIT}"
    QH_GIT_DIRTY=${QH_GIT_DIRTY})
target_compile_options(faiss_paired_recall PRIVATE
    -Wall -Wextra -Wpedantic)

add_executable(faiss_calibration
    src/experiments/faiss_calibration.cpp)
target_link_libraries(faiss_calibration PRIVATE
    qh_faiss_backend qh_metrics)
target_compile_definitions(faiss_calibration PRIVATE
    QH_FAISS_COMMIT="${QH_FAISS_EXPECTED_COMMIT}"
    QH_GIT_COMMIT="${QH_GIT_COMMIT}"
    QH_GIT_DIRTY=${QH_GIT_DIRTY})
target_compile_options(faiss_calibration PRIVATE
    -Wall -Wextra -Wpedantic)

add_executable(faiss_decomposition
    src/experiments/faiss_decomposition.cpp)
target_link_libraries(faiss_decomposition PRIVATE
    qh_faiss_backend qh_instrumentation qh_metrics)
target_compile_definitions(faiss_decomposition PRIVATE
    QH_FAISS_COMMIT="${QH_FAISS_EXPECTED_COMMIT}"
    QH_GIT_COMMIT="${QH_GIT_COMMIT}"
    QH_GIT_DIRTY=${QH_GIT_DIRTY})
target_compile_options(faiss_decomposition PRIVATE
    -Wall -Wextra -Wpedantic)

add_executable(phase1_correctness
    tests/phase1_correctness.cpp)
target_link_libraries(phase1_correctness PRIVATE
    qh_faiss_backend qh_metrics)
target_compile_options(phase1_correctness PRIVATE
    -Wall -Wextra -Wpedantic)

add_executable(decomposition_correctness
    tests/decomposition_correctness.cpp)
target_link_libraries(decomposition_correctness PRIVATE
    qh_faiss_backend qh_instrumentation qh_metrics)
target_compile_options(decomposition_correctness PRIVATE
    -Wall -Wextra -Wpedantic)

enable_testing()
add_test(
    NAME faiss_backend_validation
    COMMAND faiss_backend_validation
            ${CMAKE_CURRENT_SOURCE_DIR}/configs/indexes/faiss_hnsw_poc.conf)
add_test(NAME phase1_correctness COMMAND phase1_correctness)
add_test(NAME decomposition_correctness COMMAND decomposition_correctness)
