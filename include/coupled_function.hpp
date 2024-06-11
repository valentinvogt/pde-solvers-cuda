#ifndef COUPLED_FUNCTION_HPP_
#define COUPLED_FUNCTION_HPP_

#include "zisa/memory/array_view_decl.hpp"
#include <zisa/memory/array.hpp>
/*
Function class for coupled function.

  function returns a vector f_1, f_2, ... , f_n_coupled
  where f_n = sum_(i, j, k = 0)^max_pot scalings_(n * (i + j * max_pot +  k *
max_pot^2...) + (n-1)) x^i y^j z^k... for example in 2d and max_pot = 2 f_1(x,
y) = scaling_(0) + scaling(2) * x + scaling(4) * y + scaling(6) * x * y

*/

template <typename Scalar> class CoupledFunction {
public:
  CoupledFunction() = delete;
  CoupledFunction(zisa::array_const_view<Scalar, 1> scalings, int n_coupled,
                  int max_pot)
      : scalings_(zisa::shape_t<1>(n_coupled *
                                   (int)std::pow<int>(max_pot, n_coupled))),
        n_coupled_(n_coupled), max_pot_(max_pot) {
    assert(scalings.size() == scalings_.size());
    zisa::copy(scalings_, scalings);
  }
#if CUDA_AVAILABLE

  template <typename ARRAY>
  inline __host__ __device__ void
  operator()(zisa::array_const_view<Scalar, 1> x, ARRAY result_values) {
    for (int i = 0; i < n_coupled_; i++) {
      result_values[i] = 0;
    }

    // printf("ncoupled in func: %i\n", n_coupled_);
    for (int i = 0; i < std::pow(max_pot_, n_coupled_); i++) {
      Scalar pot = 1.;
      int max_pot_pow_j = 1;
      for (int j = 0; j < n_coupled_; j++) {
        pot *= std::pow(x(j), (int)(i / max_pot_pow_j) % max_pot_);
        max_pot_pow_j *= max_pot_;
      }
      for (int k = 0; k < n_coupled_; k++) {
        result_values[k] += scalings_.const_view()(n_coupled_ * i + k) * pot;
      }
    }
  }

#else

  template <typename ARRAY>
  inline void operator()(zisa::array_const_view<Scalar, 1> x,
                         ARRAY result_values) {
    for (int i = 0; i < n_coupled_; i++) {
      result_values[i] = 0;
    }

    for (int i = 0; i < std::pow(max_pot_, n_coupled_); i++) {
      Scalar pot = 1.;
      int max_pot_pow_j = 1;
      for (int j = 0; j < n_coupled_; j++) {
        pot *= std::pow(x(j), (int)(i / max_pot_pow_j) % max_pot_);
        max_pot_pow_j *= max_pot_;
      }
      for (int k = 0; k < n_coupled_; k++) {
        result_values[k] += scalings_(n_coupled_ * i + k) * pot;
      }
    }
  }

#endif // CUDA_AVAILABLE

private:
  zisa::array<Scalar, 1> scalings_;
  const int n_coupled_;
  const int max_pot_;
};

#endif // COUPLED_FUNCTION_HPP_
