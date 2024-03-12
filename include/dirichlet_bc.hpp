#ifndef DIRICHLET_BC_HPP_
#define DIRICHLET_BC_HPP_
#include <zisa/memory/array.hpp>
#if CUDA_AVAILABLE
#include <cuda/dirichlet_bc_cuda.hpp>
#endif

template <typename Scalar>
void dirichlet_bc_cpu(zisa::array<Scalar, 2> &data,
                      unsigned n_ghost_cells_x, unsigned n_ghost_cells_y, Scalar value) {

  // add boundary condition on left and right boundary
  unsigned x_length = data.shape(0);
  unsigned y_length = data.shape(1);

  for (int x_idx = 0; x_idx < x_length; x_idx++) {
    for (int y_idx = 0; y_idx < n_ghost_cells_y; y_idx++) {
      data(x_idx, y_idx) = value;
      data(x_idx, y_length - 1 - y_idx) = value;
    }
  }

  // add boundary on top and botton without corners
  for (int x_idx = 0; x_idx < n_ghost_cells_x; x_idx++) {
    for (int y_idx = n_ghost_cells_y; y_idx < y_length - n_ghost_cells_y;
         y_idx++) {
      data(x_idx, y_idx) = value;
      data(x_length - 1 - x_idx, y_idx) = value;
    }
  }
}

// Note that this only has to be done once at the beginning,
// the boundary data will not change during the algorithm
template <typename Scalar>
void dirichlet_bc(zisa::array<Scalar, 2> &data, unsigned n_ghost_cells_x,
                  unsigned n_ghost_cells_y, Scalar value,
                  zisa::device_type memory_location) {
  if (memory_location == zisa::device_type::cpu) {
    dirichlet_bc_cpu(data, n_ghost_cells_x, n_ghost_cells_y, value);
  }
  #if CUDA_AVAILABLE
  else if (memory_location == zisa::device_type::cuda) {
    // TODO
    dirichlet_bc_cuda(data, n_ghost_cells_x, n_ghost_cells_y, value);    
  }
  #endif // CUDA_AVAILABLE
  else {
    std::cerr << "dirichlet bc unknown device_type of inputs\n";
  }

}

#endif // DIRICHLET_BC_HPP_
