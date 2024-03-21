target_sources(main
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/convolve_cuda.cu
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/periodic_bc_cuda.cu
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/neumann_bc_cuda.cu
)
