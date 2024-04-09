#ifndef CONVOLVE_SIGMA_ADD_F_CUDA_IMPL_H_
#define CONVOLVE_SIGMA_ADD_F_CUDA_IMPL_H_

#ifndef NUM_THREAD_X
#define NUM_THREAD_X 32
#endif

#ifndef NUM_THREAD_Y
#define NUM_THREAD_Y 32
#endif


template <typename Scalar, typename Function>
__global__ void convolve_sigma_add_f_cuda_kernel(
    zisa::array_view<Scalar, 2> dst, zisa::array_const_view<Scalar, 2> src,
    zisa::array_const_view<Scalar, 2> sigma, Scalar del_x_2, Function f) {
  const int x_idx = threadIdx.x + blockIdx.x * NUM_THREAD_X + 1; // cannot be on boundary
  const int y_idx = threadIdx.y + blockIdx.y * NUM_THREAD_Y + 1;

  const int Nx = src.shape(0) - 2;
  const int Ny = src.shape(1) - 2;
  if (x_idx < Nx - 1 && y_idx < Ny - 1) {
    dst(x_idx, y_idx) =
        del_x_2 *
            (sigma(2 * x_idx - 1, y_idx - 1) * src(x_idx, y_idx - 1) +
             sigma(2 * x_idx - 1, y_idx) * src(x_idx, y_idx + 1) +
             sigma(2 * x_idx - 2, y_idx - 1) * src(x_idx - 1, y_idx) +
             sigma(2 * x_idx, y_idx - 1) * src(x_idx + 1, y_idx) -
             (sigma(2 * x_idx - 1, y_idx - 1) + sigma(2 * x_idx - 1, y_idx) +
              sigma(2 * x_idx - 2, y_idx - 1) + sigma(2 * x_idx, y_idx - 1)) *
                 src(x_idx, y_idx)) +
        f(src(x_idx, y_idx));
  }
}

template <typename Scalar, typename Function>
void convolve_sigma_add_f_cuda(zisa::array_view<Scalar, 2> dst,
                               zisa::array_const_view<Scalar, 2> src,
                               zisa::array_const_view<Scalar, 2> sigma,
                               Scalar del_x_2, Function f) {
#if CUDA_AVAILABLE
  const dim3 thread_dims(NUM_THREAD_X, NUM_THREAD_Y);
  const dim3 block_dims((src.shape(0) - 2) / thread_dims.x,
                        (src.shape(1) - 2) / thread_dims.y);
  convolve_sigma_add_f_cuda_kernel<<<block_dims, thread_dims>>>(dst, src, sigma,
                                                                del_x_2, f);
  const auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cout << "Error in convolve_cuda: " << cudaGetErrorString(error)
              << std::endl;
  }
#endif // CUDA_AVAILABLE
}
#endif // CONVOLVE_SIGMA_ADD_F_CUDA_IMPL_H_
