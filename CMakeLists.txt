cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(satt VERSION 0.1 LANGUAGES C CXX CUDA)


include(FetchContent)

FetchContent_Declare(
  ZisaCore
  GIT_REPOSITORY https://github.com/1uc/ZisaCore.git
  GIT_TAG main
  CMAKE_ARGS -DZISA_HAS_CUDA=1 -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(
  ZisaMemory
  GIT_REPOSITORY https://github.com/1uc/ZisaMemory.git
  GIT_TAG main
  CMAKE_ARGS -DZISA_HAS_CUDA=1 -DZISA_HAS_NETCDF=1 -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
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
target_include_directories(main PUBLIC include)
target_link_libraries(main Zisa::core Zisa::memory)

add_subdirectory(src)
