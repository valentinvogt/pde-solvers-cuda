#ifndef PERIODIC_BC_CUDA_IMPL_H_
#define PERIODIC_BC_CUDA_IMPL_H_

#include <zisa/memory/array.hpp>

#ifndef THREAD_DIMS
#define THREAD_DIMS 1024
#endif

template <typename Scalar>
__global__ void
periodic_bc_cuda_kernel(zisa::array<Scalar, 2> &data, unsigned n_ghost_cells_x,
                        unsigned n_ghost_cells_y, unsigned data_size) {
  const unsigned idx = blockIdx.x * THREAD_DIMS + threadIdx.x;
    std::cout << "idx: " << idx << std::endl;
  if (idx < data_size) {
    if (idx < n_ghost_cells_x * data.shape(1)) {
      // upper boundary
      const unsigned x_idx = idx / data.shape(1);
      const unsigned y_idx = idx - data.shape(1) * x_idx;
      const unsigned x_idx_to_copy =
          data.shape(0) - 2 * n_ghost_cells_x + x_idx;
      data(x_idx, y_idx) = data(x_idx_to_copy, y_idx);
      return;
    } else if (idx < n_ghost_cells_x * data.shape(1) +
                         n_ghost_cells_y *
                             (data.shape(0) - 2 * n_ghost_cells_y) * 2) {
      // left or right boundary
      const unsigned idx_without_top = idx - n_ghost_cells_x * data.shape(1);

      const unsigned x_idx =
          (idx_without_top) / (2 * n_ghost_cells_y) + n_ghost_cells_x;
      const unsigned y_shift_idx =
          idx_without_top - (x_idx - n_ghost_cells_x) * 2 * n_ghost_cells_y;
      bool on_left_boundary = y_shift_idx < n_ghost_cells_y;
      if (on_left_boundary) {
        const unsigned y_idx = y_shift_idx;
        const unsigned y_idx_to_copy =
            y_idx + data.shape(1) - 2 * n_ghost_cells_y;
        data(x_idx, y_idx) = data(x_idx, y_idx_to_copy);
        return;
      } else {
        const unsigned y_idx =
            y_shift_idx + data.shape(1) - 2 * n_ghost_cells_y;
        const unsigned y_idx_to_copy = y_shift_idx;
        data(x_idx, y_idx) = data(x_idx, y_idx_to_copy);
        return;
      }
    } else {
      // bottom boundary
      const unsigned idx_without_top_and_boundaries =
          idx - n_ghost_cells_x * data.shape(1) -
          n_ghost_cells_y * 2 * (data.shape(0) - 2 * n_ghost_cells_x);
      const unsigned x_idx = data.shape(0) - n_ghost_cells_x +
                             idx_without_top_and_boundaries / data.shape(1);
      const unsigned y_idx =
          idx_without_top_and_boundaries - x_idx * data.shape(1);
      const unsigned x_idx_to_copy =
          x_idx + 2 * n_ghost_cells_x - data.shape(0);
      data(x_idx, y_idx) = data(x_idx_to_copy, y_idx);
      return
    }
  }
}

template <typename Scalar>
void periodic_bc_cuda(zisa::array<Scalar, 2> &data, unsigned n_ghost_cells_x,
                      unsigned n_ghost_cells_y) {
#if CUDA_AVAILABLE
  const unsigned thread_dims = THREAD_DIMS;
  // size of whole boundary where periodic bc has to be applied
  const unsigned data_size =
      data.shape(1) * n_ghost_cells_x * 2 +
      (data.shape(0) - 2 * n_ghost_cells_x) * n_ghost_cells_y * 2;
  const unsigned block_dims = std::ceil(data_size / thread_dims);
  periodic_bc_cuda_kernel<<<block_dims, thread_dims>>>(
      data, n_ghost_cells_x, n_ghost_cells_y, data_size);
  cudaDeviceSynchronize();
#endif // CUDA_AVAILABLE
}
#endif // PERIODIC_BC_CUDA_H_
