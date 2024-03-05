target_sources(main
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

if (ENABLE_CUDA)
  add_subdirectory(cuda)
endif()
