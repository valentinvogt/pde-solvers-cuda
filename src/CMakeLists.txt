target_sources(main
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_sources(run_from_netcdf
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/run_from_netcdf.cpp
)

if (ENABLE_CUDA)
  add_subdirectory(cuda)
endif()
