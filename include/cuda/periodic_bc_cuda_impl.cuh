#ifndef PERIODIC_BC_CUDA_IMPL_H_
#define PERIODIC_BC_CUDA_IMPL_H_

#include <zisa/memory/array.hpp>

#ifndef THREAD_DIMS
#define THREAD_DIMS 1024
#endif

template <typename Scalar>
__global__ void periodic_bc_cuda_kernel(zisa::array_view<Scalar, 2> data,
                                        unsigned data_size) {
  const unsigned idx = blockIdx.x * THREAD_DIMS + threadIdx.x;
  const unsigned x_length = data.shape(0);
  const unsigned y_length = data.shape(1);
  printf("idx: %u\n", idx);
  // access not working properly now
  if (idx < data_size) {
    if (idx < y_length) {
      if (idx == 0) {
        // upper left corner
        data(0, 0) = data(x_length - 2, y_length - 2);
      } else if (idx == y_length - 1) {
        // upper right corner
        data(0, idx) = data(x_length - 2, 1);
      } else {
        // upper boundary without corners
        data(0, idx) = data(x_length - 2, idx);
      }
      return;
    } else if (idx < y_length + (x_length - 2) * 2) {
      const int loc_idx = idx - y_length;
      // left or right boundary
      if (idx % 2 == 0) {
        // left boundary
        const int x_idx = (loc_idx) / 2 + 1;
        data(x_idx, 0) = data(x_idx, y_length - 2);
      } else {
        const int x_idx = (loc_idx + 1) / 2;
        data(x_idx, y_length - 1) = data(x_idx, 1);
      }
      return;
    } else {
      if (idx == data_size - y_length - 1) {
        // lower left corner
        data(x_length - 1, 0) = data(1, y_length - 2);
      } else if (idx == data_size - 1) {
        // lower right corner
        data(x_length - 1, y_length - 1) = data(1, 1);
      } else {
        const int loc_y_idx = data_size - idx;
        data(x_length - 1, loc_y_idx) = data(1, loc_y_idx);
      }
      return;
    }
  }
}

  template <typename Scalar>
  void periodic_bc_cuda(zisa::array_view<Scalar, 2> data) {
#if CUDA_AVAILABLE
    const unsigned thread_dims = THREAD_DIMS;
    // size of whole boundary where periodic bc has to be applied
    const unsigned data_size = (data.shape(0) + data.shape(1)) * 2 - 4;
    const unsigned block_dims = std::ceil((double)data_size / thread_dims);
    std::cout << "should reach cuda " << block_dims << " "
              << "thread_dims" << std::endl;
    periodic_bc_cuda_kernel<<<block_dims, thread_dims>>>(data, data_size);
    const auto error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
      std::cout << "Error in convolve_cuda: " << cudaGetErrorString(error)
                << std::endl;
    }
#endif // CUDA_AVAILABLE
  }
#endif // PERIODIC_BC_CUDA_H_
