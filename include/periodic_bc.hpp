#ifndef PERIODIC_BC_HPP_
#define PERIODIC_BC_HPP_
#include <zisa/memory/array.hpp>
#if CUDA_AVAILABLE
#include <cuda/periodic_bc_cuda.hpp>
#endif

template <typename Scalar>
void periodic_bc_cpu(zisa::array<Scalar, 2> &data,unsigned n_ghost_cells_x,
                    unsigned n_ghost_cells_y) {
  unsigned x_length = data.shape(0);
  unsigned y_length = data.shape(1);
  unsigned x_shift = x_length - 2 * n_ghost_cells_x;
  unsigned y_shift = y_length - 2 * n_ghost_cells_y;


  for (int x_idx = n_ghost_cells_x; x_idx < x_length - n_ghost_cells_x; x_idx++) {
    // left cols without corners
    for (int y_idx = 0; y_idx < n_ghost_cells_y; y_idx++) {
      data(x_idx, y_idx) = data(x_idx, y_idx + y_shift);
    }
    // right cols without corners
    for (int y_idx = 0; y_idx < n_ghost_cells_y; y_idx++) {
      data(x_idx, y_length - 1 - y_idx) = data(x_idx, y_length - 1 - y_idx - y_shift);
    }
  }

  for (int y_idx = n_ghost_cells_y; y_idx < y_length - n_ghost_cells_y; y_idx++) {
    // top cols without corners
    for (int x_idx = 0; x_idx < n_ghost_cells_x; x_idx++) {
      data(x_idx, y_idx) = data(x_idx + x_shift, y_idx);
    }

    // bottom cols without corners
    for (int x_idx = 0; x_idx < n_ghost_cells_x; x_idx++) {
      data(x_length - 1 - x_idx, y_idx) = data(x_length - 1 - x_idx - x_shift, y_idx);
    }
  }

  // corners
  for (int x_idx = 0; x_idx < n_ghost_cells_x; x_idx++) {
    for (int y_idx = 0; y_idx < n_ghost_cells_y; y_idx++) {
      data(x_idx, y_idx) = data(x_idx + x_shift, y_idx + y_shift);
      data(x_length - x_idx - 1, y_idx) = data(x_length - 1 - x_idx - x_shift, y_idx + y_shift);
      data(x_idx, y_length - y_idx - 1) = data(x_idx + x_shift, y_length - 1 - y_idx - y_shift);
      data(x_length - x_idx - 1, y_length - y_idx - 1) = data(x_length - 1 - x_idx - x_shift, y_length - 1 - y_idx - y_shift);
    }
  }
}

// only implemented for f'(x) = 0 so far
template <typename Scalar>
void periodic_bc(zisa::array<Scalar, 2> &data, unsigned n_ghost_cells_x,
                unsigned n_ghost_cells_y, zisa::device_type memory_location) {
  if (memory_location == zisa::device_type::cpu) {
    periodic_bc_cpu(data, n_ghost_cells_x, n_ghost_cells_y);
  }
#if CUDA_AVAILABLE
  else if (memory_location == zisa::device_type::cuda) {
    // TODO
    periodic_bc_cuda(data, n_ghost_cells_x, n_ghost_cells_y);
  }
#endif // CUDA_AVAILABLE
  else {
    std::cerr << "periodic bc unknown device_type of inputs\n";
  }
}

#endif // PERIODIC_BC_HPP_
