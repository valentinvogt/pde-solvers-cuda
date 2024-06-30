#ifndef ADD_ARRAYS_CUDA_IMPL_H_
#define ADD_ARRAYS_CUDA_IMPL_H_

#include <iostream>
#include <stdio.h>
#include <zisa/memory/array.hpp>

#ifndef NUM_THREAD_X
#define NUM_THREAD_X 32
#endif

#ifndef NUM_THREAD_Y
#define NUM_THREAD_Y 32
#endif

template <int n_coupled, typename Scalar>
__global__ void
add_arrays_interior_cuda_kernel(zisa::array_view<Scalar, 2> dst,
                                zisa::array_const_view<Scalar, 2> src,
                                Scalar scaling) {

  const int x_idx =
      threadIdx.x + 1 + blockIdx.x * NUM_THREAD_X; // cannot be on boundary
  const int y_idx = threadIdx.y + n_coupled + blockIdx.y * NUM_THREAD_Y;

  const int Nx = src.shape(0);
  const int Ny = src.shape(1);

  if (x_idx < Nx - 1 && y_idx < Ny - n_coupled) {
    dst(x_idx, y_idx) += scaling * src(x_idx, y_idx);
  }
}

template <int n_coupled, typename Scalar>
void add_arrays_interior_cuda(zisa::array_view<Scalar, 2> dst,
                              zisa::array_const_view<Scalar, 2> src,
                              Scalar scaling) {
#if CUDA_AVAILABLE
  const dim3 thread_dims(NUM_THREAD_X, NUM_THREAD_Y);
  const dim3 block_dims(
      std::ceil((src.shape(0) - 2) / (double)thread_dims.x),
      std::ceil((src.shape(1) - 2 * n_coupled) / (double)thread_dims.y));

  // std::cout << "thread dims: " << thread_dims.x << " " << thread_dims.y <<
  // std::endl; std::cout << "block dims: " << block_dims.x << " " <<
  // block_dims.y << std::endl;
  add_arrays_interior_cuda_kernel<n_coupled>
      <<<block_dims, thread_dims>>>(dst, src, scaling);
  const auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cout << "Error in add_arrays: " << cudaGetErrorString(error)
              << std::endl;
  }
#endif // CUDA_AVAILABLE
}

#endif // ADD_ARRAYS_CUDA_IMPL_H_
