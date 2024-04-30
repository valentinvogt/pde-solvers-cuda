#ifndef CONVOLVE_SIGMA_ADD_F_CUDA_IMPL_H_
#define CONVOLVE_SIGMA_ADD_F_CUDA_IMPL_H_

#ifndef NUM_THREAD_X
#define NUM_THREAD_X 32
#endif

#ifndef NUM_THREAD_Y
#define NUM_THREAD_Y 32
#endif

template <int n_coupled, typename Scalar, typename Function>
__global__ void convolve_sigma_add_f_cuda_kernel(
    zisa::array_view<Scalar, 2> dst, zisa::array_const_view<Scalar, 2> src,
    zisa::array_const_view<Scalar, 2> sigma, Scalar del_x_2, Function f) {
  // printf("reached kernel\n");
  const int x_idx =
      threadIdx.x + 1 + blockIdx.x * NUM_THREAD_X; // cannot be on boundary
  const int y_idx = threadIdx.y + 1 + blockIdx.y * NUM_THREAD_Y;

  const int Nx = src.shape(0);
  const int Ny = src.shape(1) / n_coupled;
  if (x_idx < Nx - 1 && y_idx < Ny - 1) {
    zisa::array_const_view<Scalar, 1> input_1 {zisa ::shape_t<1>(n_coupled),
                                             &src(x_idx, n_coupled * y_idx),
                                             zisa ::device_type ::cuda};
    zisa::array_const_view<Scalar, 1> input = input_1;
  // if (input.memory_location() == zisa::device_type::cpu) {
  //   printf("input called with memory_location = cpu\n");
  // } else if (input.memory_location() == zisa::device_type::cuda) {
  //   printf("input called with memory_location = cuda\n");
  // }

    // printf("call function with input: ");
    for (int i = 0; i < n_coupled; i++) {
      // printf("%f ", input(i));
    }
    // printf("\n");
    Scalar result_function[n_coupled];
    f(input, result_function);
    // printf("output: ");
    for (int i = 0; i < n_coupled; i++) {
      // printf("%f ", result_function(i));
    }
   // printf("\n");
    for (int i = 0; i < n_coupled; i++) {
      dst(x_idx, n_coupled * y_idx + i) =
          del_x_2 *
              (sigma(2 * x_idx - 1, y_idx - 1) * src(x_idx, y_idx - 1) +
               sigma(2 * x_idx - 1, y_idx + 1) * src(x_idx, y_idx + 1) +
               sigma(2 * x_idx - 2, y_idx) * src(x_idx - 1, y_idx) +
               sigma(2 * x_idx, y_idx) * src(x_idx + 1, y_idx) -
               (sigma(2 * x_idx - 1, y_idx - 1) + sigma(2 * x_idx - 1, y_idx) +
                sigma(2 * x_idx - 2, y_idx - 1) + sigma(2 * x_idx, y_idx - 1)) *
                   src(x_idx, y_idx)) +
          result_function[i];
    }
  }
}

template <int n_coupled, typename Scalar, typename Function>
void convolve_sigma_add_f_cuda(zisa::array_view<Scalar, 2> dst,
                               zisa::array_const_view<Scalar, 2> src,
                               zisa::array_const_view<Scalar, 2> sigma,
                               Scalar del_x_2, Function f) {
#if CUDA_AVAILABLE
  const dim3 thread_dims(NUM_THREAD_X, NUM_THREAD_Y);
  const dim3 block_dims(std::ceil((src.shape(0) - 2) / (double)thread_dims.x),
                        std::ceil((src.shape(1) - 2) / (double)thread_dims.y));
  // std::cout << "reached convolve_sigma" << std::endl;
  convolve_sigma_add_f_cuda_kernel<n_coupled, Scalar, Function>
      <<<block_dims, thread_dims>>>(dst, src, sigma, del_x_2, f);
  const auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cout << "Error in convolve_cuda: " << cudaGetErrorString(error)
              << std::endl;
  }
#endif // CUDA_AVAILABLE
}
#endif // CONVOLVE_SIGMA_ADD_F_CUDA_IMPL_H_
