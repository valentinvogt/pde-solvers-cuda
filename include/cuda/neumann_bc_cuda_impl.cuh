#ifndef NEUMANN_BC_CUDA_IMPL_H_
#define NEUMANN_BC_CUDA_IMPL_H_

#include <zisa/memory/array.hpp>

template <typename Scalar>
__global__ void
neumann_bc_cuda_kernel(zisa::array_view<Scalar, 2> &data,
                     unsigned n_ghost_cells_x,
                     unsigned n_ghost_cells_y) {
  // TODO
  printf("Hello from GPU");
}

template <typename Scalar>
void neumann_bc_cuda(zisa::array<Scalar, 2> &data,
                   unsigned n_ghost_cells_x,
                   unsigned n_ghost_cells_y) {
#if CUDA_AVAILABLE
  const int thread_dims = 1024;
  const int block_dims = std::ceil((data.shape(0) * data.shape(1)) / thread_dims);
  neumann_bc_cuda_kernel<<<block_dims, thread_dims>>>(data, n_ghost_cells_x, n_ghost_cells_y);
  cudaDeviceSynchronize();
#endif // CUDA_AVAILABLE
}
#endif // NEUMANN_BC_CUDA_H_
