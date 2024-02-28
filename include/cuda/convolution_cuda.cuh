#ifndef CONVOLUTION_CUDA_H_
#define CONVOLUTION_CUDA_H_


#if ENABLE_CUDA
template <typename Scalar>
__global__ void
convolve_cuda_kernel(zisa::array_view<Scalar, 2> dst,
                  const zisa::array_const_view<Scalar, 2> &src,
                  const zisa::array_const_view<Scalar, 2> &kernel) {
  //TODO
  printf("Hello from block %d, thread %d\n", block_dims, thread_dims)
}
#endif // ENABLE_CUDA

#endif // CONVOLUTION_CUDA_H_

