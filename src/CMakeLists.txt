target_sources(main
  PRIVATE main.cpp
)

if (ENABLE_CUDA)
  add_subdirectory(cuda)
endif()
