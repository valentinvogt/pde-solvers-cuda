#ifndef DIRICHLET_BC_HPP_
#define DIRICHLET_BC_HPP_
#include "zisa/memory/array_view_decl.hpp"
#include <zisa/memory/array.hpp>
#if CUDA_AVAILABLE
#include <cuda/dirichlet_bc_cuda.hpp>
#endif

template <typename Scalar>
void dirichlet_bc_cpu(zisa::array_view<Scalar, 2> data,
                      zisa::array_const_view<Scalar, 2> bc,
                      unsigned n_ghost_cells_x, unsigned n_ghost_cells_y) {

  // add boundary condition on left and right boundary
  unsigned x_length = data.shape(0);
  unsigned y_length = data.shape(1);

  for (int x_idx = 0; x_idx < x_length; x_idx++) {
    for (int y_idx = 0; y_idx < n_ghost_cells_y; y_idx++) {
      data(x_idx, y_idx) = bc(x_idx, y_idx);
      data(x_idx, y_length - 1 - y_idx) = bc(x_idx, y_length - 1 - y_idx);
    }
  }

  // add boundary on top and botton without corners
  for (int x_idx = 0; x_idx < n_ghost_cells_x; x_idx++) {
    for (int y_idx = n_ghost_cells_y; y_idx < y_length - n_ghost_cells_y;
         y_idx++) {
      data(x_idx, y_idx) = bc(x_idx, y_idx);
      data(x_length - 1 - x_idx, y_idx) = bc(x_length - 1 - x_idx, y_idx);
    }
  }
}

// Note that this only has to be done once at the beginning,
// the boundary data will not change during the algorithm
template <typename Scalar>
void dirichlet_bc(zisa::array_view<Scalar, 2> data,
                  zisa::array_const_view<Scalar, 2> bc, 
                  unsigned n_ghost_cells_x,
                  unsigned n_ghost_cells_y,
                  zisa::device_type memory_location) {
  if (memory_location == zisa::device_type::cpu) {
    dirichlet_bc_cpu(data, bc, n_ghost_cells_x, n_ghost_cells_y);
  }
  #if CUDA_AVAILABLE
  else if (memory_location == zisa::device_type::cuda) {
    // TODO
    dirichlet_bc_cuda(data, bc, n_ghost_cells_x, n_ghost_cells_y);
  }
  #endif // CUDA_AVAILABLE
  else {
    std::cerr << "dirichlet bc unknown device_type of inputs\n";
  }

}

#endif // DIRICHLET_BC_HPP_
