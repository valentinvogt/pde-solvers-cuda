cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(PDE_SOLVERS_CUDA VERSION 0.1 LANGUAGES C CXX)

option(ENABLE_CUDA "Enable CUDA Acceleration" OFF)

set(CUDA_AVAILABLE ${ENABLE_CUDA})
set(ZISA_HAS_CUDA ${ENABLE_CUDA})

message(${ENABLE_CUDA})

if(CUDA_AVAILABLE)
  enable_language(CUDA)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


include(FetchContent)

FetchContent_Declare(
  ZisaCore
  GIT_REPOSITORY https://github.com/1uc/ZisaCore.git
  GIT_TAG main
  CMAKE_ARGS -DZISA_HAS_CUDA=${ENABLE_CUDA} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(
  ZisaMemory
  GIT_REPOSITORY https://github.com/1uc/ZisaMemory.git
  GIT_TAG main
  CMAKE_ARGS -DZISA_HAS_CUDA=${ENABLE_CUDA} -DZISA_HAS_NETCDF=1 -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.13.0
  OVERRIDE_FIND_PACKAGE
)

FetchContent_MakeAvailable(ZisaCore ZisaMemory googletest)

find_package(ZisaCore REQUIRED)
find_package(ZisaMemory REQUIRED)
find_package(googletest REQUIRED)


add_executable(main)
target_include_directories(main PUBLIC include include/cuda)

if (CUDA_AVAILABLE)
    find_package(CUDAToolkit REQUIRED)
    add_library(cuda_lib OBJECT include/cuda/convolve_cuda_impl.cu include/cuda/convolve_cuda.hpp)
    target_compile_definitions(cuda_lib PUBLIC ZISA_HAS_CUDA=1)
    target_link_libraries(cuda_lib Zisa::memory Zisa::core)
    set_target_properties(cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON) 
    set_target_properties(cuda_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(main cuda_lib)
endif()

target_link_libraries(main Zisa::core Zisa::memory)

add_subdirectory(src)
