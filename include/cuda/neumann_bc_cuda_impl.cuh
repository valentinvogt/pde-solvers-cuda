#ifndef NEUMANN_BC_CUDA_IMPL_H_
#define NEUMANN_BC_CUDA_IMPL_H_

#include <cassert>
#include <zisa/memory/array.hpp>

// TODO: add n_coupled
template <typename Scalar, int n_coupled>
__global__ void neumann_bc_cuda_kernel(zisa::array_view<Scalar, 2> data,
                                       zisa::array_const_view<Scalar, 2> bc,
                                       Scalar dt) {

  const int linear_idx = threadIdx.x + THREAD_DIMS * blockIdx.x;
  const int Nx = data.shape(0);
  const int Ny = data.shape(1);
  if (linear_idx < 2 * (Nx + Ny) - 4) {
    if (linear_idx < Nx) {
      data(linear_idx, 0) += dt * bc(linear_idx, 0);
    } else if (linear_idx < Nx + 2 * Ny - 4) {
      const int y_idx = 1 + (linear_idx - Nx) / 2;
      if (linear_idx % 2) {
        data(0, y_idx) += dt * bc(0, y_idx);
      } else {
        data(Nx - 1, y_idx) += dt * bc(Nx - 1, y_idx);
      }
    } else {
      const int x_idx = 2 * (Nx + Ny) - 4 - linear_idx - 1;
      data(x_idx, Ny - 1) += dt * bc(x_idx, Ny - 1);
    }
  }
}

template <typename Scalar, int n_coupled>
void neumann_bc_cuda(zisa::array_view<Scalar, 2> data,
                     zisa::array_const_view<Scalar, 2> bc, Scalar dt) {
#if CUDA_AVAILABLE
  const int thread_dims = THREAD_DIMS;
  const int block_dims =
      std::ceil((double)(data.shape(0) * data.shape(1)) / thread_dims);
  neumann_bc_cuda_kernel<Scalar, n_coupled>
      <<<block_dims, thread_dims>>>(data, bc, dt);
  const auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cout << "Error in convolve_cuda: " << cudaGetErrorString(error)
              << std::endl;
  }
#endif // CUDA_AVAILABLE
}
#endif // NEUMANN_BC_CUDA_H_
