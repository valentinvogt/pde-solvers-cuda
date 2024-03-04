#ifndef CONVOLUTION_CUDA_H_
#define CONVOLUTION_CUDA_H_

#include <zisa/memory/array.hpp>

template <typename Scalar>
__global__ void
convolve_cuda_kernel(zisa::array_view<Scalar, 2> dst,
                  const zisa::array_const_view<Scalar, 2> &src,
                  const zisa::array_const_view<Scalar, 2> &kernel) {
  //TODO
  printf("Hello from GPU");
}

template<typename Scalar>
void convolve_cuda(zisa::array_view<Scalar, 2> dst,
                  const zisa::array_const_view<Scalar, 2> &src,
                  const zisa::array_const_view<Scalar, 2> &kernel){
  #if CUDA_AVAILABLE
  const int thread_dims = 1024;
  const int block_dims = std::ceil((src.shape(0) * src.shape(1)) / thread_dims);
  convolve_cuda_kernel<<<block_dims, thread_dims>>>(dst, src, kernel);
  cudaDeviceSynchronize();
  #endif // CUDA_AVAILABLE
}
#endif // CONVOLUTION_CUDA_H_

