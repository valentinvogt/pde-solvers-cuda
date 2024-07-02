#ifndef NEUMANN_BC_CUDA_IMPL_H_
#define NEUMANN_BC_CUDA_IMPL_H_

#include <cassert>
#include <zisa/memory/array.hpp>

#ifndef THREAD_DIMS
#define THREAD_DIMS 1024
#endif

template <int n_coupled, typename Scalar>
__global__ void neumann_bc_cuda_kernel(zisa::array_view<Scalar, 2> data,
                                       zisa::array_const_view<Scalar, 2> bc,
                                       Scalar dt) {

  const int linear_idx = threadIdx.x + THREAD_DIMS * blockIdx.x;
  const int Nx = data.shape(0);
  const int Ny = data.shape(1);
  if (linear_idx < 2 * (Nx * n_coupled + Ny) - 4 * n_coupled) {
    if (linear_idx < Nx * n_coupled) {
      // left boundary of size n_coupled
      const int x_idx = linear_idx % Nx;
      const int y_idx = linear_idx / Nx;
      data(x_idx, y_idx) += dt * bc(x_idx, y_idx);
    } else if (linear_idx < Nx * n_coupled + 2 * Ny - 4 * n_coupled) {
      // upper and lower boundary of size 1
      const int y_idx = n_coupled + (linear_idx - Nx * n_coupled) / 2;
      if (linear_idx % 2) {
        data(0, y_idx) += dt * bc(0, y_idx);
      } else {
        data(Nx - 1, y_idx) += dt * bc(Nx - 1, y_idx);
      }
    } else {
      // right boundary of size n_coupled
      const int shifted_idx =
          linear_idx - (Nx * n_coupled + 2 * Ny - 4 * n_coupled);
      const int x_idx = shifted_idx % Nx;
      const int y_idx = Ny - 1 - (shifted_idx / Nx);
      data(x_idx, y_idx) += dt * bc(x_idx, y_idx);
    }
  }
}

template <int n_coupled, typename Scalar>
void neumann_bc_cuda(zisa::array_view<Scalar, 2> data,
                     zisa::array_const_view<Scalar, 2> bc, Scalar dt) {
#if CUDA_AVAILABLE
  const int thread_dims = THREAD_DIMS;
  const int block_dims =
      std::ceil((double)(2 * (data.shape(0) * n_coupled + data.shape(1)) -
                         4 * n_coupled) /
                THREAD_DIMS);
  neumann_bc_cuda_kernel<n_coupled, Scalar>
      <<<block_dims, thread_dims>>>(data, bc, dt);
  const auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cout << "Error in neumann_bc: " << cudaGetErrorString(error)
              << std::endl;
  }
#else
  std::cout << "Tried to call neumann_bc_cuda without cuda available!"
            << std::endl;
#endif // CUDA_AVAILABLE
}
#endif // NEUMANN_BC_CUDA_H_
