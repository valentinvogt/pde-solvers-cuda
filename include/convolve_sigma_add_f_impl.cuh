#ifndef CONVOLVE_SIGMA_ADD_CUDA_IMPL_H_
#define CONVOLVE_SIGMA_ADD_CUDA_IMPL_H_

#ifndef THREAD_DIMS
#define THREAD_DIMS 1024
#endif

templaate <typename Scalar, typename Function>
__global__ void convolve_sigma_add_f_cuda_kernel(zisa::array_view<Scalar, 2> dst,
                   zisa::array_const_view<Scalar, 2> src,
                   zisa::array_const_view<Scalar, 2> sigma, Scalar del_x_2, Function f) {
  //TODO:
}

template <typename Scalar, typename Function>
void convolve_sigma_add_f_cuda(zisa::array_view<Scalar, 2> dst,
                   zisa::array_const_view<Scalar, 2> src,
                   zisa::array_const_view<Scalar, 2> sigma, Scalar del_x_2, Function f) {
#if CUDA_AVAILABLE
  const int thread_dims = THREAD_DIMS;
  const int block_dims =
      std::ceil((double)((src.shape(0) - 2 * (kernel.shape(0) / 2)) *
                         (src.shape(1) - 2 * (kernel.shape(1) / 2))) /
                thread_dims);
  convolve_sigma_add_f_cuda_kernel<<<block_dims, thread_dims>>>(dst, src, sigma, del_x_2, f);
  const auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cout << "Error in convolve_cuda: " << cudaGetErrorString(error)
              << std::endl;
  }
#endif // CUDA_AVAILABLE
}
#endif // CONVOLVE_SIGMA_ADD_CUDA_IMPL_H_
