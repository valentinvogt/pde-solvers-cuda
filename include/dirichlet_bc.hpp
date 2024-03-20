#ifndef DIRICHLET_BC_HPP_
#define DIRICHLET_BC_HPP_
#include "zisa/memory/array_view_decl.hpp"
#include <zisa/memory/array.hpp>
#if CUDA_AVAILABLE
#include <cuda/dirichlet_bc_cuda.hpp>
#endif

template <typename Scalar>
void dirichlet_bc_cpu(zisa::array_view<Scalar, 2> data,
                      zisa::array_const_view<Scalar, 2> bc) {

  const int x_length = data.shape(0);
  const int y_length = data.shape(1);

  // add boundary condition on left and right boundary
  for (int x_idx = 0; x_idx < x_length; x_idx++) {
    data(x_idx, 0) = bc(x_idx, 0);
    data(x_idx, y_length - 1) = bc(x_idx, y_length - 1);
  }

  // add boundary on top and botton without corners
  for (int y_idx = 1; y_idx < y_length - 1; y_idx++) {
    data(0, y_idx) = bc(0, y_idx);
    data(x_length - 1, y_idx) = bc(x_length - 1, y_idx);
  }
}

// Note that this only has to be done once at the beginning,
// the boundary data will not change during the algorithm
template <typename Scalar>
void dirichlet_bc(zisa::array_view<Scalar, 2> data,
                  zisa::array_const_view<Scalar, 2> bc) {
  const zisa::device_type memory_location = data.memory_location();
  if (memory_location != bc.memory_location()) {
    std::cerr << "Dirichlet: Inputs must be located on the same hardware\n";
    exit(1);
  }
  if (memory_location == zisa::device_type::cpu) {
    dirichlet_bc_cpu(data, bc);
  }
#if CUDA_AVAILABLE
  else if (memory_location == zisa::device_type::cuda) {
    // TODO
    dirichlet_bc_cuda(data, bc);
  }
#endif // CUDA_AVAILABLE
  else {
    std::cerr << "dirichlet bc unknown device_type of inputs\n";
  }
}

#endif // DIRICHLET_BC_HPP_
