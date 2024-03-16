#ifndef CONVOLUTION_HPP_
#define CONVOLUTION_HPP_

#include <iostream>
#include <zisa/memory/array.hpp>
#if CUDA_AVAILABLE
#include <cuda/convolve_cuda.hpp>
#endif

template <typename Scalar>
void convolve_cpu(zisa::array_view<Scalar, 2> dst,
                  const zisa::array_const_view<Scalar, 2> &src,
                  const zisa::array_const_view<Scalar, 2> &kernel) {
  // TODO: Optimize
  // IDEA: recognize at compile time which kernel entries are 0 and only multiply if not
  const int ghost_x = kernel.shape(0) / 2;
  const int ghost_y = kernel.shape(1) / 2;
  const int Nx = src.shape(0) - 2 * ghost_x;
  const int Ny = src.shape(1) - 2 * ghost_y;
  for (int i = ghost_x; i < Nx + ghost_x; ++i) {
    for (int j = ghost_y; j < Ny + ghost_y; ++j) {
      dst(i, j) = 0.;
      for (int di = -ghost_x; di <= ghost_x; ++di) {
        for (int dj = -ghost_y; dj <= ghost_y; ++dj) {
          if (kernel(ghost_x + di, ghost_y + dj) != 0) {
            dst(i, j) += kernel(ghost_x + di, ghost_y + dj) * src(i + di, j + dj);
          }
        }
      }
    }
  }
}

template <typename Scalar>
void convolve(zisa::array_view<Scalar, 2> dst,
              const zisa::array_const_view<Scalar, 2> &src,
              const zisa::array_const_view<Scalar, 2> &kernel) {
  const zisa::device_type memory_dst = dst.memory_location();
  const zisa::device_type memory_src = src.memory_location();
  const zisa::device_type memory_kernel = kernel.memory_location();

  if (!(memory_dst == memory_src && memory_src == memory_kernel)) {
    std::cerr << "Convolution: Inputs must be located on the same hardware\n";
    exit(1);
  }
  if (dst.shape() != src.shape()) {
    std::cerr
        << "Convolution: Input and output array must have the same shape\n";
    exit(1);
  }

  if (memory_dst == zisa::device_type::cpu) {
    convolve_cpu(dst, src, kernel);
  }
#if CUDA_AVAILABLE
  else if (memory_dst == zisa::device_type::cuda) {
    std::cout << "reached_cuda" << std::endl;
    convolve_cuda(dst, src, kernel);
  }
#endif // CUDA_AVAILABLE
  else {
    std::cerr << "Convolution: Unknown device_type of inputs\n";
    exit(1);
  }
}

#endif
