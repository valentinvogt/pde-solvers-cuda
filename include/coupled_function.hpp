
#ifndef COUPLED_FUNCTION_HPP_
#define COUPLED_FUNCTION_HPP_

#include "zisa/memory/device_type.hpp"
#if CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif
#include <zisa/memory/array.hpp>
#include <zisa/memory/shape.hpp>

// currently implemented for polynoms of max degree 2
#include <cmath>

template <typename Scalar, int n_coupled, int max_pot> class CoupledFunction {
public:
  CoupledFunction() {};
  CoupledFunction(zisa::array_const_view<Scalar, 1> scalings)
      : scalings_(zisa::shape_t<1>((int)std::pow<int>(max_pot, n_coupled)), scalings.memory_location()),
        memory_location_(scalings.memory_location()){
    zisa::copy(scalings_, scalings);
  }

  CoupledFunction(const CoupledFunction &other) 
  : scalings_(zisa::shape_t<1>(other.scalings_.size()), other.memory_location_){
    zisa::copy(scalings_, other.scalings_);
  }
  
  CoupledFunction operator=(const CoupledFunction &other) {
    CoupledFunction tmp(scalings_.const_view());
    return tmp;
  }

#if CUDA_AVAILABLE
  // input: array of size n_coupled, representing all values of one position
  // output: Scalar, f(x, y, z)
  inline __host__ __device__ Scalar
  operator()(zisa::array_const_view<Scalar, 1> x,
                           int n_values_left = n_coupled, int curr_pos = 0) {
    Scalar result = 0.;
    Scalar curr_pot = 1.;
    // base case
    if (n_values_left == 1) {
      for (int i = 0; i < max_pot; i++) {
        result += curr_pot * scalings_(curr_pos + i);
        curr_pot *= x(0);
      }
      return result;
    }
    int block_size = std::pow(max_pot, n_values_left - 1);
    for (int i = 0; i < max_pot; i++) {
      result += this->operator()(zisa::detail::slice(x, 1, n_values_left),
                                 n_values_left - 1, curr_pos + i * block_size) *
                curr_pot;
      curr_pot *= x(0);
    }
    return result;
}
#else

  // input: array of size n_coupled, representing all values of one position
  // output: Scalar, f(x, y, z)
  inline Scalar operator()(zisa::array_const_view<Scalar, 1> x,
                           int n_values_left = n_coupled, int curr_pos = 0) {
    Scalar result = 0.;
    Scalar curr_pot = 1.;
    // base case
    if (n_values_left == 1) {
      for (int i = 0; i < max_pot; i++) {
        result += curr_pot * scalings_(curr_pos + i);
        curr_pot *= x(0);
      }
      return result;
    }
    int block_size = std::pow(max_pot, n_values_left - 1);
    for (int i = 0; i < max_pot; i++) {
      result += this->operator()(zisa::detail::slice(x, 1, n_values_left),
                                 n_values_left - 1, curr_pos + i * block_size) *
                curr_pot;
      curr_pot *= x(0);
    }
    return result;
  }
#endif
  // function overloaded such that you don't have to create an
  // array every time you call this function with n_coupled == 1
  // could be deleted later
  inline Scalar operator()(Scalar value)  {
    assert(n_coupled == 1);
    zisa::array<Scalar, 1> tmp(zisa::shape_t<1>(1), memory_location_);
    tmp(0) = value;
    return this->operator()(tmp.const_view());
  }

private:
  // function returns sum_(i, j, k = 0)^max_pot scalings_(i, j, k) x^i y^j z^k
  // for example in 2d and max_pot = 2
  // f(x, y) = scaling_(0, 0) + scaling(0, 1) * y + scaling(1, 0) * x +
  // scaling(1, 1) * x * y
  zisa::array<Scalar, 1> scalings_;
  zisa::device_type memory_location_ = zisa::device_type::cpu;
};

#endif // COUPLED_FUNCTION_HPP_
