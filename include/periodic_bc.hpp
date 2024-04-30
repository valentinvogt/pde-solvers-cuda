#ifndef PERIODIC_BC_HPP_
#define PERIODIC_BC_HPP_
#include <zisa/memory/array.hpp>
#if CUDA_AVAILABLE
#include <cuda/periodic_bc_cuda.hpp>
#endif

// TODO: add n_coupled
template <int n_coupled, typename Scalar>
void periodic_bc_cpu(zisa::array_view<Scalar, 2> data) {
  const unsigned x_length = data.shape(0);
  const unsigned y_length = data.shape(1);

  const unsigned x_shift = x_length - 2;
  const unsigned y_shift = y_length - 2 * n_coupled;

  for (int x_idx = 1; x_idx < x_length - 1; x_idx++) {
    // left cols without corners
    for (int y_idx = 0; y_idx < n_coupled; y_idx++) {
      data(x_idx, y_idx) = data(x_idx, y_idx + y_shift);
    }
    // right cols without corners
    for (int y_idx = 0; y_idx < n_coupled; y_idx++) {
      data(x_idx, y_length - 1 - y_idx) =
          data(x_idx, y_length - 1 - y_idx - y_shift);
    }
  }

  for (int y_idx = n_coupled; y_idx < y_length - n_coupled; y_idx++) {
    // top cols without corners
    data(0, y_idx) = data(0 + x_shift, y_idx);

    // bottom cols without corners
    data(x_length - 1, y_idx) = data(1, y_idx);
  }

  // corners
  for (int x_idx = 0; x_idx < 1; x_idx++) {
    for (int y_idx = 0; y_idx < n_coupled; y_idx++) {
      // top left corner
      data(0, y_idx) = data(x_shift, y_idx + y_shift);
      // bottom left corner
      data(x_length - 1, y_idx) = data(1, y_idx + y_shift);
      // top right corner
      data(0, y_length - y_idx - 1) =
          data(x_shift, y_length - 1 - y_idx - y_shift);
      // bottom right corner
      data(x_length - 1, y_length - y_idx - 1) =
          data(1, y_length - 1 - y_idx - y_shift);
    }
  }
}

template <int n_coupled, typename Scalar>
void periodic_bc(zisa::array_view<Scalar, 2> data) {

  const zisa::device_type memory_location = data.memory_location();
  if (memory_location == zisa::device_type::cpu) {
    // std::cout << "periodic bc cpu reached" << std::endl;
    periodic_bc_cpu<n_coupled, Scalar>(data);
  }
#if CUDA_AVAILABLE
  else if (memory_location == zisa::device_type::cuda) {
    // TODO
    // std::cout << "periodic bc cuda reached" << std::endl;
    periodic_bc_cuda<n_coupled, Scalar>(data);
  }
#endif // CUDA_AVAILABLE
  else {
    std::cerr << "periodic bc unknown device_type of inputs\n";
  }
}

#endif // PERIODIC_BC_HPP_
