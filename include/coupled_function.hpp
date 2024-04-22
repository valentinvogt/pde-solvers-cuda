
#ifndef GENERIC_FUNCTION_HPP_
#define GENERIC_FUNCTION_HPP_

#if CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif
#include <zisa/memory/array.hpp>

#define CREATE_ARGUMENTS_REC(1, VALUE) VALUE
#define CREATE_ARGUMENTS_REC(SIZE, VALUE)                                      \
  VALUE, CREATE_ARGUMENTS_REC(SIZE - 1, VALUE)

// currently implemented for polynoms of max degree 2
#include <cmath>

template <typename Scalar, int n_coupled, int max_pot> class CoupledFunction {
public:
  CoupledFunction() {}
  CoupledFunction(zisa::array_const_view<Scalar, n_coupled> scalings)
      : scalings_(
            zisa::shape_t<n_coupled>(CREATE_ARGUMENTS_REC(n_coupled, max_pot),
                                     scalings.memory_location())) {
    zisa::copy(scalings_, scalings);
  }
#if CUDA_AVAILABLE
  // input: array of size n_coupled, representing all values of one position
  // output: Scalar, f(x, y, z)
  inline __host__ __device__ Scalar
  operator()(zisa::array_const_view<Scalar, 1> x) {
    return const_val_ + lin_val_ * x + quad_val_ * x * x +
           exp_scale_val_ * std::exp(exp_pot_val_ * x);
  }
#else
  // input: array of size n_coupled, representing all values of one position
  // output: Scalar, f(x, y, z)
  inline Scalar operator()(zisa::array_const_view<Scalar, 1> x) {
    return const_val_ + lin_val_ * x + quad_val_ * x * x +
           exp_scale_val_ * std::exp(exp_pot_val_ * x);
  }
#endif

  void set_const(Scalar value) { const_val_ = value; }

  void set_lin(Scalar value) { lin_val_ = value; }
  void set_quad(Scalar value) { quad_val_ = value; }

  void set_exp(Scalar scale, Scalar pot) {
    exp_scale_val_ = scale;
    exp_pot_val_ = pot;
  }

private:
  // function returns sum_(i, j, k = 0)^max_pot scalings_(i, j, k) x^i y^j z^k
  // for example in 2d and max_pot = 2
  // f(x, y) = scaling_(0, 0) + scaling(0, 1) * y + scaling(1, 0) * x +
  // scaling(1, 1) * x * y
  zisa::array<Scalar, n_coupled> scalings_;
};

#endif // GENERIC_FUNCTION_HPP_
