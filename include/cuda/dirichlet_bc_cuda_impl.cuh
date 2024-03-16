#ifndef DIRICHLET_BC_CUDA_IMPL_H_
#define DIRICHLET_BC_CUDA_IMPL_H_

#include <zisa/memory/array.hpp>

#ifndef THREAD_DIMS
#define THREAD_DIMS 1024
#endif

template <typename Scalar>
__global__ void
dirichlet_bc_cuda_kernel(zisa::array_view<Scalar, 2> data,
                         zisa::array_const_view<Scalar, 2> bc,
                         unsigned n_ghost_cells_x,
                         unsigned n_ghost_cells_y) {
  const int linear_idx = threadIdx.x + THREAD_DIMS * blockIdx.x;
  const int Nx = data.shape(0);
  const int Ny = data.shape(1);
  if (linear_idx < 2 * (Nx + Ny) - 4) {
    if (linear_idx < Nx) {
      data(linear_idx, 0) = bc(linear_idx, 0);
    } else if (linear_idx < Nx + 2 * Ny - 4) {
      const int y_idx = 1 + (linear_idx - Nx) / 2;
      if (linear_idx % 2) {
        data(0, y_idx) = bc(0, y_idx);
      } else {
        data(Nx - 1, y_idx) = bc(Nx - 1, y_idx);
      }
    } else {
      const int x_idx = 2 * (Nx + Ny) - 4 - linear_idx;
      data(x_idx, Ny - 1) = bc(x_idx, Ny - 1);
    }
    
  }
}

template <typename Scalar>
void dirichlet_bc_cuda(zisa::array_view<Scalar, 2> data,
                       zisa::array_const_view<Scalar, 2> bc,
                       unsigned n_ghost_cells_x,
                       unsigned n_ghost_cells_y) {
#if CUDA_AVAILABLE
  const int thread_dims = THREAD_DIMS;
  const int block_dims = std::ceil((double)(2 * (data.shape(0) + data.shape(1)) - 4) / thread_dims);
  dirichlet_bc_cuda_kernel<<<block_dims, thread_dims>>>(data, bc, n_ghost_cells_x, n_ghost_cells_y);
  const auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cout << "Error in convolve_cuda: " << cudaGetErrorString(error) << std::endl;
  }
#endif // CUDA_AVAILABLE
}
#endif // DIRICHLET_BC_CUDA_H_
