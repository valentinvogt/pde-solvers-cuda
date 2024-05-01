
#ifndef COUPLED_FUNCTION_HPP_
#define COUPLED_FUNCTION_HPP_

#include "zisa/memory/device_type.hpp"
#if CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif
#include <zisa/memory/array.hpp>
#include <zisa/memory/shape.hpp>

#include <cmath>

template <typename Scalar>
zisa::array_const_view<Scalar, 1>
slice_arr(const zisa::array_const_view<Scalar, 1> &arr, int begin, int size) {
  zisa::shape_t<1> sub_shape(size);

  auto ptr = arr.raw() + begin;
  zisa::device_type memory_location = arr.memory_location();
  return {sub_shape, ptr, memory_location};
}

template <typename Scalar, int n_coupled, int max_pot> class CoupledFunction {
public:
  CoupledFunction(){};
  CoupledFunction(zisa::array_const_view<Scalar, 1> scalings)
      : scalings_(zisa::shape_t<1>(n_coupled *
                                   (int)std::pow<int>(max_pot, n_coupled)),
                  scalings.memory_location()),
        memory_location_(scalings.memory_location()) {
    zisa::copy(scalings_, scalings);
  }

  CoupledFunction(const CoupledFunction &other)
      : scalings_(zisa::shape_t<1>(other.scalings_.size()),
                  other.memory_location_),
        memory_location_(other.memory_location_) {
    zisa::copy(scalings_, other.scalings_);
  }

#if CUDA_AVAILABLE
  // input: array of size n_coupled, representing all values of one position
  // output: Scalar, f(x, y, z)
  // make shure that if you have memory_location_ == cuda to only call this
  // function from cuda kernels
  inline __device__ void operator()(zisa::array_const_view<Scalar, 1> x,
                                    Scalar result_values[n_coupled],
                                    int n_values_left = n_coupled,
                                    int curr_scalings_pos = 0) {

    if (memory_location_ == zisa::device_type::cpu) {
      assert(x.memory_location() == zisa::device_type::cpu);
    } else if (memory_location_ == zisa::device_type::cuda) {
      assert(x.memory_location() == zisa::device_type::cuda);
    }

    // assert(scalings_.const_view().memory_location() == memory_location_);
    assert(x.size() > 0);
    assert(n_values_left <= n_coupled);
    assert(x.size() > n_coupled - n_values_left);
    // set array to zero
    for (int i = 0; i < n_coupled; i++) {
      result_values[i] = 0;
    }
    // Scalar result = 0.;
    Scalar curr_pot = 1.;
    // base case
    if (n_values_left == 1) {
      for (int i = 0; i < max_pot; i++) {
        assert(scalings_.size() > curr_scalings_pos + i);
        for (int j = 0; j < n_coupled; j++) {
          result_values[j] +=
              curr_pot * scalings_(curr_scalings_pos + i * n_coupled + j);
        }
        curr_pot *= x(n_coupled - n_values_left);
      }
      return;
    }
    int block_size = n_coupled * std::pow(max_pot, n_values_left - 1);
    for (int i = 0; i < max_pot; i++) {
      Scalar rec_res[n_coupled];
      this->operator()(x, rec_res, n_values_left - 1,
                       curr_scalings_pos + i * block_size);
      for (int j = 0; j < n_coupled; j++) {
        result_values[j] += rec_res[j] * curr_pot;
      }
      assert(n_coupled - n_values_left >= 0 &&
             n_coupled - n_values_left < x.size());
      curr_pot *= x(n_coupled - n_values_left);
    }
    return;
  }
  // function overloaded such that you don't have to create an
  // array every time you call this function with n_coupled == 1
  // could be deleted later
  inline __host__ __device__ Scalar operator()(Scalar value) {
    assert(n_coupled == 1);
    assert(memory_location_ == zisa::device_type::cpu);
    zisa::array<Scalar, 1> tmp(zisa::shape_t<1>(1), memory_location_);
    if (memory_location_ == zisa::device_type::cpu) {
      tmp(0) = value;
      return (this->operator()(tmp.const_view()))[0];
    } else if (memory_location_ == zisa::device_type::cuda) {
      zisa::array<Scalar, 1> tmp_cpu(zisa::shape_t<1>(1),
                                     zisa::device_type::cpu);
      tmp_cpu(0) = value;
      zisa::copy(tmp, tmp_cpu);
    }
    Scalar result_values[n_coupled];
    this->operator()(tmp.const_view());
    return result_values[0];
  }
#else
  // input: array of size n_coupled, representing all values of one position
  // output: array, f1(x, y, z), f2(x, y, z), f3(x, y, z) of size n_coupled
  inline zisa::array<Scalar, 1> operator()(zisa::array_const_view<Scalar, 1> x,
                                           int n_values_left = n_coupled,
                                           int curr_scalings_pos = 0) {

    assert(memory_location_ == zisa::device_type::cpu);
    assert(x.memory_location() == memory_location_);
    assert(scalings_.const_view().memory_location() == memory_location_);
    assert(x.size() > 0);
    assert(n_values_left <= n_coupled);
    assert(x.size() > n_coupled - n_values_left);
    zisa::shape_t<1> shape(n_coupled);
    zisa::array<Scalar, 1> result_values(shape, memory_location_);
    // set array to zero
    // TODO: make this more efficient
    for (int i = 0; i < n_coupled; i++) {
      result_values(i) = 0;
    }
    // Scalar result = 0.;
    Scalar curr_pot = 1.;
    // base case
    if (n_values_left == 1) {
      for (int i = 0; i < max_pot; i++) {
        assert(scalings_.size() > curr_scalings_pos + i);
        for (int j = 0; j < n_coupled; j++) {
          result_values(j) +=
              curr_pot * scalings_(curr_scalings_pos + i * n_coupled + j);
        }
        curr_pot *= x(n_coupled - n_values_left);
      }
      return result_values;
    }
    int block_size = n_coupled * std::pow(max_pot, n_values_left - 1);
    for (int i = 0; i < max_pot; i++) {
      zisa::array<Scalar, 1> rec_res = this->operator()(
          x, n_values_left - 1, curr_scalings_pos + i * block_size);
      for (int j = 0; j < n_coupled; j++) {
        result_values(j) += rec_res(j) * curr_pot;
      }
      assert(n_coupled - n_values_left >= 0 &&
             n_coupled - n_values_left < x.size());
      curr_pot *= x(n_coupled - n_values_left);
    }
    return result_values;
  }
  // function overloaded such that you don't have to create an
  // array every time you call this function with n_coupled == 1
  // could be deleted later
  inline Scalar operator()(Scalar value) {
    assert(n_coupled == 1);
    assert(memory_location_ == zisa::device_type::cpu);
    zisa::array<Scalar, 1> tmp(zisa::shape_t<1>(1), memory_location_);
    tmp(0) = value;
    return (this->operator()(tmp.const_view()))(0);
  }
#endif // CUDA_AVAILABLE

private:
  // function returns sum_(i, j, k = 0)^max_pot scalings_(i, j, k) x^i y^j z^k
  // for example in 2d and max_pot = 2
  // f(x, y) = scaling_(0, 0) + scaling(0, 1) * y + scaling(1, 0) * x +
  // scaling(1, 1) * x * y
  zisa::array<Scalar, 1> scalings_;
  zisa::device_type memory_location_ = zisa::device_type::cpu;
};

#endif // COUPLED_FUNCTION_HPP_
