target_sources(main
  PRIVATE convolve_cuda.cu
  PRIVATE periodic_bc_cuda.cu
  PRIVATE dirichlet_bc_cuda.cu
  PRIVATE neumann_bc_cuda.cu
)
