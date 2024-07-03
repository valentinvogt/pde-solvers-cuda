#ifndef NEUMANN_BC_HPP_
#define NEUMANN_BC_HPP_
#include <zisa/memory/array.hpp>
#if CUDA_AVAILABLE
#include <cuda/neumann_bc_cuda.hpp>
#endif

template <int n_coupled, typename Scalar>
void neumann_bc_cpu(zisa::array_view<Scalar, 2> data,
                    zisa::array_const_view<Scalar, 2> bc, Scalar dt) {

  unsigned x_length = data.shape(0);
  unsigned y_length = data.shape(1);

  // add boundary condition on left and right boundary
  for (int x_idx = 0; x_idx < x_length; x_idx++) {
    for (int i = 0; i < n_coupled; i++) {
      data(x_idx, i) += dt * bc(x_idx, i);
      data(x_idx, y_length - 1 - i) += dt * bc(x_idx, y_length - 1 - i);
    }
  }

  // add boundary on top and botton without corners
  for (int y_idx = n_coupled; y_idx < y_length - n_coupled; y_idx++) {
    data(0, y_idx) += dt * bc(0, y_idx);
    data(x_length - 1, y_idx) += dt * bc(x_length - 1, y_idx);
  }
}

// updates the outermost data with dt * values on bc
template <int n_coupled, typename Scalar>
void neumann_bc(zisa::array_view<Scalar, 2> data,
                zisa::array_const_view<Scalar, 2> bc, Scalar dt) {
  const zisa::device_type memory_location = data.memory_location();
  if (memory_location != bc.memory_location()) {
    std::cerr << "Neumann: Inputs must be located on the same hardware\n";
    exit(1);
  }
  if (memory_location == zisa::device_type::cpu) {
    neumann_bc_cpu<n_coupled, Scalar>(data, bc, dt);
  }
#if CUDA_AVAILABLE
  else if (memory_location == zisa::device_type::cuda) {
    neumann_bc_cuda<n_coupled, Scalar>(data, bc, dt);
  }
#endif // CUDA_AVAILABLE
  else {
    std::cerr << "neumann bc unknown device_type of inputs\n";
  }
}

#endif // NEUMANN_BC_HPP_
