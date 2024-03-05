cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(PDE_SOLVERS_CUDA VERSION 0.1 LANGUAGES C CXX)

option(ENABLE_CUDA "Enable CUDA Acceleration" OFF)

set(CUDA_AVAILABLE ${ENABLE_CUDA})
set(ZISA_HAS_CUDA ${ENABLE_CUDA})

message(${ENABLE_CUDA})

if(CUDA_AVAILABLE)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
  ZisaCore
  GIT_REPOSITORY https://github.com/1uc/ZisaCore.git
  GIT_TAG 9243c391b70f5b1ff8c0275905351158c4f59eab
  CMAKE_ARGS -DZISA_HAS_CUDA=${ENABLE_CUDA} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(
  ZisaMemory
  GIT_REPOSITORY https://github.com/1uc/ZisaMemory.git
  GIT_TAG 321013db1830fb77589e13139400aeaea8796f79
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

if (CUDA_AVAILABLE)
  target_compile_options(main PUBLIC -DCUDA_AVAILABLE=1)
  target_compile_options(main PUBLIC $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>>)
endif()

target_include_directories(main PUBLIC include)

target_link_libraries(main Zisa::core Zisa::memory)

add_subdirectory(src)
