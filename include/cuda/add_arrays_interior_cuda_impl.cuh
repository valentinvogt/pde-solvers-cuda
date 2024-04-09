#ifndef ADD_ARRAYS_CUDA_IMPL_H_
#define ADD_ARRAYS_CUDA_IMPL_H_

#include <iostream>
#include <stdio.h>
#include <zisa/memory/array.hpp>

#ifndef THREAD_DIMS
#define THREAD_DIMS 1024
#endif

template <typename Scalar>
__global__ void add_arrays_interior_cuda_kernel(zisa::array_view<Scalar, 2> dst,
                                       zisa::array_const_view<Scalar, 2> src,
                                       Scalar scaling) {

  // TODO: only do it in interior
  const int linear_idx = threadIdx.x + blockIdx.x * THREAD_DIMS;
  const int Nx = src.shape(0);
  const int Ny = src.shape(1);
  if (linear_idx < Nx * Ny) {
    const int x_idx = linear_idx / Ny;
    const int y_idx = linear_idx % Ny;
    if (x_idx != 0 && y_ind != 0 && x_idx != Nx - 1 && x_idx != Ny - 1) {
      dst(x_idx, y_idx) += scaling * src(x_idx, y_idx);
    }
  }
}

template <typename Scalar>
void add_arrays_interior_cuda(zisa::array_view<Scalar, 2> dst,
                     zisa::array_const_view<Scalar, 2> src, Scalar scaling) {
#if CUDA_AVAILABLE
  const int thread_dims = THREAD_DIMS;
  const int block_dims =
      std::ceil((double)(src.shape(0) * src.shape(1)) / thread_dims);
  add_arrays_interior_cuda_kernel<<<block_dims, thread_dims>>>(dst, src, scaling);
  const auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cout << "Error in convolve_cuda: " << cudaGetErrorString(error)
              << std::endl;
  }
#endif // CUDA_AVAILABLE
}

#endif // ADD_ARRAYS_CUDA_IMPL_H_
