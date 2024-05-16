target_sources(main
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/convolve_cuda.cu
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/convolve_sigma_add_f_cuda.cu
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/periodic_bc_cuda.cu
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/neumann_bc_cuda.cu
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/add_arrays_interior_cuda.cu
)

target_sources(run_from_netcdf
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/convolve_cuda.cu
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/convolve_sigma_add_f_cuda.cu
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/periodic_bc_cuda.cu
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/neumann_bc_cuda.cu
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/add_arrays_interior_cuda.cu
)

target_sources(unit_tests
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/convolve_cuda.cu
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/convolve_sigma_add_f_cuda.cu
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/periodic_bc_cuda.cu
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/neumann_bc_cuda.cu
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/add_arrays_interior_cuda.cu
)

target_sources(benchmarks
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/convolve_cuda.cu
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/convolve_sigma_add_f_cuda.cu
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/periodic_bc_cuda.cu
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/neumann_bc_cuda.cu
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/add_arrays_interior_cuda.cu
)
