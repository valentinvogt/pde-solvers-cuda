#ifndef PERIODIC_BC_HPP_
#define PERIODIC_BC_HPP_
#include <zisa/memory/array.hpp>
#if CUDA_AVAILABLE
#include <cuda/periodic_bc_cuda.hpp>
#endif

template <typename Scalar>
void periodic_bc_cpu(zisa::array_view<Scalar, 2> data) {
  const unsigned x_length = data.shape(0);
  const unsigned y_length = data.shape(1);

  const unsigned x_shift = x_length - 2;
  const unsigned y_shift = y_length - 2;

  for (int x_idx = 1; x_idx < x_length - 1; x_idx++) {
    // left cols without corners
    for (int y_idx = 0; y_idx < 1; y_idx++) {
      data(x_idx, y_idx) = data(x_idx, y_idx + y_shift);
    }
    // right cols without corners
    for (int y_idx = 0; y_idx < 1; y_idx++) {
      data(x_idx, y_length - 1 - y_idx) =
          data(x_idx, y_length - 1 - y_idx - y_shift);
    }
  }

  for (int y_idx = 1; y_idx < y_length - 1; y_idx++) {
    // top cols without corners
    for (int x_idx = 0; x_idx < 1; x_idx++) {
      data(x_idx, y_idx) = data(x_idx + x_shift, y_idx);
    }

    // bottom cols without corners
    for (int x_idx = 0; x_idx < 1; x_idx++) {
      data(x_length - 1 - x_idx, y_idx) =
          data(x_length - 1 - x_idx - x_shift, y_idx);
    }
  }

  // corners
  for (int x_idx = 0; x_idx < 1; x_idx++) {
    for (int y_idx = 0; y_idx < 1; y_idx++) {
      data(x_idx, y_idx) = data(x_idx + x_shift, y_idx + y_shift);
      data(x_length - x_idx - 1, y_idx) =
          data(x_length - 1 - x_idx - x_shift, y_idx + y_shift);
      data(x_idx, y_length - y_idx - 1) =
          data(x_idx + x_shift, y_length - 1 - y_idx - y_shift);
      data(x_length - x_idx - 1, y_length - y_idx - 1) =
          data(x_length - 1 - x_idx - x_shift, y_length - 1 - y_idx - y_shift);
    }
  }
}

template <typename Scalar> void periodic_bc(zisa::array_view<Scalar, 2> data) {

  const zisa::device_type memory_location = data.memory_location();
  if (memory_location == zisa::device_type::cpu) {
    std::cout << "periodic bc cpu reached" << std::endl;
    periodic_bc_cpu(data);
  }
#if CUDA_AVAILABLE
  else if (memory_location == zisa::device_type::cuda) {
    // TODO
    std::cout << "periodic bc cuda reached" << std::endl;
    periodic_bc_cuda(data);
  }
#endif // CUDA_AVAILABLE
  else {
    std::cerr << "periodic bc unknown device_type of inputs\n";
  }
}

#endif // PERIODIC_BC_HPP_
