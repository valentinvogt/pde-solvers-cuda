#ifndef CONVOLUTION_CUDA_H_
#define CONVOLUTION_CUDA_H_


template <typename Scalar>
__global__ void
convolve_cuda_kernel(zisa::array_view<Scalar, 2> dst,
                  const zisa::array_const_view<Scalar, 2> &src,
                  const zisa::array_const_view<Scalar, 2> &kernel) {
  //TODO
  printf("Hello from block %d, thread %d\n", block_dims, thread_dims)
}

template<typename Scalar>
void convolve_cuda(zisa::array_view<Scalar, 2> dst,
                  const zisa::array_const_view<Scalar, 2> &src,
                  const zisa::array_const_view<Scalar, 2> &kernel){
  #if ENABLE_CUDA
  const int thread_dims = 1024;
  const int block_dims = std::ceil((src.shape(0) * src.shape(1)) / thread_dims);
  convolve_cuda_kernel<<<block_dims, thread_dims>>>(dst, src, kernel);
  cudaDeviceSynchronize();
  #endif // ENABLE_CUDA
}
#endif // CONVOLUTION_CUDA_H_

