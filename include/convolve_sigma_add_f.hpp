#ifndef CONVOLVE_SIGMA_ADD_F_HPP_
#define CONVOLVE_SIGMA_ADD_F_HPP_

#include <zisa/memory/array.hpp>
#include <zisa/memory/device_type.hpp>
#if CUDA_AVAILABLE
#include <cuda/convolve_sigma_add_f_cuda.hpp>
#endif

template <typename Scalar, typename Function>
void convolve_sigma_add_f_cpu(zisa::array_view<Scalar, 2> dst,
                              zisa::array_const_view<Scalar, 2> src,
                              zisa::array_const_view<Scalar, 2> sigma,
                              Scalar del_x_2 /* 1/(dx^2)*/, Function f) {
  const unsigned Nx = src.shape(0);
  const unsigned Ny = src.shape(1);
  for (int x = 1; x < Nx - 1; x++) {
    for (int y = 1; y < Ny - 1; y++) {
      dst(x, y) = del_x_2 * (sigma(2 * x - 1, y - 1) * src(x, y - 1) +
                             sigma(2 * x - 1, y) * src(x, y + 1) +
                             sigma(2 * x - 2, y - 1) * src(x - 1, y) +
                             sigma(2 * x, y - 1) * src(x + 1, y) -
                             (sigma(2 * x - 1, y - 1) + sigma(2 * x - 1, y) +
                              sigma(2 * x - 2, y - 1) + sigma(2 * x, y - 1)) *
                                 src(x, y)) +
                  f(src(x, y));
    }
  }
}

// TODO: add coupled function(s)
// Function is a general function taking a Scalar returning a Scalar
template <typename Scalar, typename Function>
void convolve_sigma_add_f(zisa::array_view<Scalar, 2> dst,
                          zisa::array_const_view<Scalar, 2> src,
                          zisa::array_const_view<Scalar, 2> sigma,
                          Scalar del_x_2 /* 1/(dx^2)*/, Function f) {

  const zisa::device_type memory_dst = dst.memory_location();
  const zisa::device_type memory_src = src.memory_location();
  const zisa::device_type memory_sigma = sigma.memory_location();

  if (!(memory_dst == memory_src && memory_src == memory_sigma)) {
    std::cerr << "Convolve sigma add f: Inputs must be located on the same "
                 "hardware\n";
    exit(1);
  }
  if (dst.shape() != src.shape()) {
    std::cerr << "Convolve sigma add f: Input and output array must have the "
                 "same shape\n";
    exit(1);
  }

  if (memory_dst == zisa::device_type::cpu) {
    convolve_sigma_add_f_cpu(dst, src, sigma, del_x_2, f);
  }
#if CUDA_AVAILABLE
  else if (memory_dst == zisa::device_type::cuda) {
    // TODO
    convolve_sigma_add_f_cuda(dst, src, sigma, del_x_2, f);
  }
#endif // CUDA_AVAILABLE
  else {
    std::cerr << "Convolution: Unknown device_type of inputs\n";
    exit(1);
  }
}

#endif // CONVOLVE_SIGMA_ADD_F_HPP_