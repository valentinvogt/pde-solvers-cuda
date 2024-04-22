
#include <coupled_function.hpp>
#include <gtest/gtest.h>
#include <zisa/memory/array.hpp>
#include <zisa/memory/device_type.hpp>

namespace CoupledFunctionTests {
TEST(CoupledFunctionTests, TEST_ZERO) {
  zisa::array<float, 1> function_scalings(zisa::shape_t<1>(9),
                                          zisa::device_type::cpu);
  for (int i = 0; i < 9; i++) {
    function_scalings(i) = 0.;
  }
  CoupledFunction<float, 2, 3> func(function_scalings.const_view());
  float tol = 1e-10;
  zisa::array<float, 1> values(zisa::shape_t<1>(2), zisa::device_type::cpu);
  values(0) = 1.;
  values(1) = 0.5;

  ASSERT_NEAR(0.0, func(values.const_view()), tol);
#if CUDA_AVAILABLE
  zisa::array<float, 1> function_scalings_cuda(zisa::shape_t<1>(9),
                                               zisa::device_type::cuda);
  zisa::copy(function_scalings_cuda, function_scalings);
  CoupledFunction<float, 2, 3> func_cuda(function_scalings_cuda.const_view());
  zisa::array<float, 1> values_cuda(zisa::shape_t<1>(2),
                                    zisa::device_type::cuda);
  values_cuda(0) = 1.;
  values_cuda(1) = 0.5;
  ASSERT_NEAR(0.0, func_cuda(values_cuda.const_view()), tol);

#endif
}

TEST(CoupledFunctionTests, TEST_CONSTANT) {
  zisa::array<float, 1> function_scalings(zisa::shape_t<1>(9),
                                          zisa::device_type::cpu);
  for (int i = 0; i < 9; i++) {
    function_scalings(i) = 0;
  }
  function_scalings(0) = 2.3;

  CoupledFunction<float, 2, 3> func(function_scalings.const_view());
  float tol = 1e-5;
  zisa::array<float, 1> values(zisa::shape_t<1>(2), zisa::device_type::cpu);
  values(0) = 1.;
  values(1) = 0.5;

  ASSERT_NEAR(2.3, func(values.const_view()), tol);

  values(0) = 0.;
  values(1) = 0.;
  ASSERT_NEAR(2.3, func(values.const_view()), tol);
#if CUDA_AVAILABLE
  zisa::array<float, 1> function_scalings_cuda(zisa::shape_t<1>(9),
                                               zisa::device_type::cuda);
  zisa::copy(function_scalings_cuda, function_scalings);
  CoupledFunction<float, 2, 3> func_cuda(function_scalings_cuda.const_view());
  zisa::array<float, 1> values_cuda(zisa::shape_t<1>(2),
                                    zisa::device_type::cuda);
  values_cuda(0) = 1.;
  values_cuda(1) = 0.5;
  ASSERT_NEAR(0.0, func_cuda(values_cuda.const_view()), tol);

  values_cuda(0) = 0.;
  values_cuda(1) = 0.;
  ASSERT_NEAR(0.0, func_cuda(values_cuda.const_view()), tol);

#endif
}

TEST(CoupledFunctionTests, TEST_3_COUPLED_LIINEAR) {
  zisa::array<float, 1> function_scalings(zisa::shape_t<1>(8),
                                          zisa::device_type::cpu);

  // f(x, y, z) = 1 + 2x + y + z + 3xy + 2xz + yz + 4xyz
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        function_scalings(4 * i + 2 * j + k) = i + i * j + i * j * k + 1;
      }
    }
  }

  CoupledFunction<float, 3, 2> func(function_scalings.const_view());
  float tol = 1e-5;
  zisa::array<float, 1> values(zisa::shape_t<1>(3), zisa::device_type::cpu);
  values(0) = 1.;
  values(1) = 1.;
  values(2) = 1.;

  ASSERT_NEAR(15.0, func(values.const_view()), tol);

  values(0) = 0.;
  ASSERT_NEAR(4.0, func(values.const_view()), tol);

  values(1) = 0.;
  ASSERT_NEAR(2.0, func(values.const_view()), tol);

  values(2) = -1.;
  ASSERT_NEAR(0.0, func(values.const_view()), tol);

  values(0) = 0.3;
  values(1) = 0.6;
  values(2) = 0.7;
  // f(0.3, 0.6, 0.7) = 1 + 2*0.3 + 0.6 + 0.7 + 3*0.3*0.6 + 2*0.3*0.7 + 0.6*0.7
  // + 4*0.3*0.6*0.7 = 4.784
  ASSERT_NEAR(4.784, func(values.const_view()), tol);
#if CUDA_AVAILABLE
  zisa::array<float, 1> function_scalings_cuda(zisa::shape_t<1>(9),
                                               zisa::device_type::cuda);
  zisa::copy(function_scalings_cuda, function_scalings);
  CoupledFunction<float, 2, 3> func_cuda(function_scalings_cuda.const_view());
  zisa::array<float, 1> values_cuda(zisa::shape_t<1>(3),
                                    zisa::device_type::cuda);
  values_cuda(0) = 1.;
  values_cuda(1) = 1.;
  values_cuda(2) = 1.;
  ASSERT_NEAR(15.0, func_cuda(values_cuda.const_view()), tol);

  values_cuda(0) = 0.;
  ASSERT_NEAR(4.0, func_cuda(values_cuda.const_view()), tol);

  values_cuda(1) = 0.;
  ASSERT_NEAR(2.0, func_cuda(values_cuda.const_view()), tol);

  values_cuda(2) = -1.;
  ASSERT_NEAR(0.0, func_cuda(values_cuda.const_view()), tol);

  values_cuda(0) = 0.3;
  values_cuda(1) = 0.6;
  values_cuda(2) = 0.7;
  ASSERT_NEAR(4.784, func_cuda(values_cuda.const_view()), tol);

#endif
}

} // namespace CoupledFunctionTests