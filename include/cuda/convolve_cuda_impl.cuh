#ifndef CONVOLVE_CUDA_IMPL_H_
#define CONVOLVE_CUDA_IMPL_H_

#include <zisa/memory/array.hpp>
#include <stdio.h>
#include <iostream>

#ifndef THREAD_DIMS
#define THREAD_DIMS 1024
#endif

template <typename Scalar>
__global__ void
convolve_cuda_kernel(zisa::array_view<Scalar, 2> dst,
                     const zisa::array_const_view<Scalar, 2> &src,
                     const zisa::array_const_view<Scalar, 2> &kernel) {
  // TODO
  printf("convolve_cuda_kernel reached with threadIdx = %u, blockIdx = %u\n", threadIdx.x, blockIdx.x);
}

template <typename Scalar>
void convolve_cuda(zisa::array_view<Scalar, 2> dst,
                   const zisa::array_const_view<Scalar, 2> &src,
                   const zisa::array_const_view<Scalar, 2> &kernel) {
#if CUDA_AVAILABLE
  const int thread_dims = THREAD_DIMS;
  const int block_dims = std::ceil((double)(src.shape(0) * src.shape(1)) / thread_dims);
  convolve_cuda_kernel<<<block_dims, thread_dims>>>(dst, src, kernel);
  const auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cout << "Error in convolve_cuda: " << cudaGetErrorString(error) << std::endl;
  }
#endif // CUDA_AVAILABLE
}
#endif // CONVOLVE_CUDA_IMPL_H_
