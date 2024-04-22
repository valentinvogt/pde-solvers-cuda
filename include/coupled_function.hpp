
#ifndef COUPLED_FUNCTION_HPP_
#define COUPLED_FUNCTION_HPP_

#if CUDA_AVAILABLE
#include <cuda_runtime.h>
#endif
#include <zisa/memory/array.hpp>

// currently implemented for polynoms of max degree 2
#include <cmath>

// TODO: shape_t size > 3 ?
template <int n> zisa::shape_t<n> create_shape(int value) {
  std::cout << "more than 3 coupled functions not allowed yet!" << std::endl;
  return zisa::shape_t<n>(value);
}

template <> zisa::shape_t<1> create_shape<1>(int value) {
  return zisa::shape_t<1>(value);
}
template <> zisa::shape_t<2> create_shape<2>(int value) {
  return zisa::shape_t<2>(value, value);
}
template <> zisa::shape_t<3> create_shape<3>(int value) {
  return zisa::shape_t<3>(value, value, value);
}

template <typename Scalar, int n_coupled, int max_pot> class CoupledFunction {
public:
  CoupledFunction() {}
  CoupledFunction(zisa::array_const_view<Scalar, n_coupled> scalings)
      : scalings_(create_shape<n_coupled>(max_pot),
                  scalings.memory_location()) {
    zisa::copy(scalings_, scalings);
  }
#if CUDA_AVAILABLE
  // input: array of size n_coupled, representing all values of one position
  // output: Scalar, f(x, y, z)
  inline __host__ __device__ Scalar
  operator()(zisa::array_const_view<Scalar, 1> x) {
    // TODO
    // implemented for n_coupled <= 3
    Scalar result = 0;
    if (n_coupled == 1) {
      for (int i = 0; i < max_pot; i++) {
        if (scalings_(i) != 0) {
          result += scalings_(i) * std::pow(x(0), i);
        }
      }
    } else if (n_coupled == 2) {
      for (int i = 0; i < max_pot; i++) {
        for (int j = 0; j < max_pot; j++) {
          if (scalings_(i, j) != 0) {
            result += scalings_(i, j) * std::pow(x(0), i) * std::pow(x(1), j);
          }
        }
      }
    } else if (n_coupled == 3) {
      for (int i = 0; i < max_pot; i++) {
        for (int j = 0; j < max_pot; j++) {
          for (int k = 0; k < max_pot; k++) {
            if (scalings_(i, j, k) != 0) {
              result += scalings_(i, j, k) * std::pow(x(0), i) * std::pow(x(1), j) * std::pow(x(2), j);
            }
          }
        }
      }
    } else {
      std::cout << "operator+ not implented for n_coupled > 3 in coupled_function" << std::endl;
    }
    return result;
  }
#else
  // input: array of size n_coupled, representing all values of one position
  // output: Scalar, f(x, y, z)
  inline Scalar operator()(zisa::array_const_view<Scalar, 1> x) {
    // implemented for n_coupled <= 3
    Scalar result = 0;
    if (n_coupled == 1) {
      for (int i = 0; i < max_pot; i++) {
        if (scalings_(i) != 0) {
          result += scalings_(i) * std::pow(x(0), i);
        }
      }
    } else if (n_coupled == 2) {
      for (int i = 0; i < max_pot; i++) {
        for (int j = 0; j < max_pot; j++) {
          if (scalings_(i, j) != 0) {
            result += scalings_(i, j) * std::pow(x(0), i) * std::pow(x(1), j);
          }
        }
      }
    } else if (n_coupled == 3) {
      for (int i = 0; i < max_pot; i++) {
        for (int j = 0; j < max_pot; j++) {
          for (int k = 0; k < max_pot; k++) {
            if (scalings_(i, j, k) != 0) {
              result += scalings_(i, j, k) * std::pow(x(0), i) * std::pow(x(1), j) * std::pow(x(2), j);
            }
          }
        }
      }
    } else {
      std::cout << "operator+ not implented for n_coupled > 3 in coupled_function" << std::endl;
    }
    return result;
  }
#endif

private:
  // function returns sum_(i, j, k = 0)^max_pot scalings_(i, j, k) x^i y^j z^k
  // for example in 2d and max_pot = 2
  // f(x, y) = scaling_(0, 0) + scaling(0, 1) * y + scaling(1, 0) * x +
  // scaling(1, 1) * x * y
  zisa::array<Scalar, n_coupled> scalings_;
};

#endif // COUPLED_FUNCTION_HPP_
