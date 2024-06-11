
#include <coupled_function_2.hpp>
#include <gtest/gtest.h>
#include <zisa/memory/array.hpp>
#include <zisa/memory/array_view_decl.hpp>
#include <zisa/memory/device_type.hpp>

namespace CoupledFunctionTests {
TEST(CoupledFunctionTests, TEST_ZERO) {
  zisa::array<float, 1> function_scalings(zisa::shape_t<1>(9 * 2),
                                          zisa::device_type::cpu);
  for (int i = 0; i < 9 * 2; i++) {
    function_scalings(i) = 0.;
  }
  CoupledFunction2<float> func(function_scalings.const_view(), 2, 3);
  float tol = 1e-10;
  zisa::array<float, 1> values(zisa::shape_t<1>(2), zisa::device_type::cpu);
  values(0) = 1.;
  values(1) = 0.5;
  float result[2];
  result[0] = -1.;
  result[1] = -2.;
  func(values, result);
  ASSERT_NEAR(0.0, result[0], tol);
}

TEST(CoupledFunctionTests, TEST_CONSTANT) {
  zisa::array<float, 1> function_scalings(zisa::shape_t<1>(9 * 2),
                                          zisa::device_type::cpu);
  for (int i = 0; i < 9 * 2; i++) {
    function_scalings(i) = 0;
  }
  function_scalings(0) = 2.3;
  function_scalings(1) = 2.3;

  CoupledFunction2<float> func(function_scalings.const_view(), 2, 3);

  float tol = 1e-5;
  zisa::array<float, 1> values(zisa::shape_t<1>(2), zisa::device_type::cpu);
  values(0) = 1.;
  values(1) = 0.5;
  float results[2];
  func(values.const_view(), results);

  ASSERT_NEAR(2.3, results[0], tol);

  values(0) = 0.;
  values(1) = 0.;
  func(values.const_view(), results);
  ASSERT_NEAR(2.3, results[1], tol);

}

TEST(CoupledFunctionTests, TEST_3_COUPLED_LINEAR) {
  zisa::array<float, 1> function_scalings(zisa::shape_t<1>(8 * 3),
                                          zisa::device_type::cpu);

  // f(x, y, z) = 1 + x + y + xy + 2z + 2xz + 3yz + 4xyz
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        function_scalings(3 * (4 * i + 2 * j + k)) = i + i * j + i * j * k + 1;
        function_scalings(3 * (4 * i + 2 * j + k) + 1) =
            i + i * j + i * j * k + 1;
        function_scalings(3 * (4 * i + 2 * j + k) + 2) =
            i + i * j + i * j * k + 1;
      }
    }
  }

  CoupledFunction2<float> func(function_scalings.const_view(), 3, 2);

  float tol = 1e-5;
  zisa::array<float, 1> values(zisa::shape_t<1>(3), zisa::device_type::cpu);
  values(0) = 1.;
  values(1) = 1.;
  values(2) = 1.;

  float results[3];
  func(values.const_view(), results);
  ASSERT_NEAR(15.0, results[1], tol);

  values(0) = 0.;
  func(values.const_view(), results);
  ASSERT_NEAR(7.0, results[0], tol);


  values(1) = 0.;
  func(values.const_view(), results);
  ASSERT_NEAR(3.0, results[1], tol);

  values(2) = -1.;
  func(values.const_view(), results);
  ASSERT_NEAR(-1, results[0], tol);

  values(0) = 0.3;
  values(1) = 0.6;
  values(2) = 0.7;
  // f(0.3, 0.6, 0.7) = 1 + 0.3 + 0.6 + 0.3*0.6 + 2*0.7 + 2*0.3*0.7 + 3*0.6*0.7 + 4*0.3*0.6*0.7 = 5.664
  func(values.const_view(), results);
  ASSERT_NEAR(5.664, results[0], tol);

}

TEST(CoupledFunctionTests, TEST_FROM_ARRAY) {
  zisa::array<float, 1> function_scalings(zisa::shape_t<1>(8 * 3),
                                          zisa::device_type::cpu);
  // f_1(x, y, z) = 1 + x + y + xy + 2z + 2xz + 3yz + 4xyz
  // f_2(x, y, z) = 1 + x + y + xy + 2z + 2xz + 3yz + 4xyz
  // f_3(x, y, z) = 0
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        function_scalings(3 * (4 * i + 2 * j + k)) = i + i * j + i * j * k + 1;
        function_scalings(3 * (4 * i + 2 * j + k) + 1) =
            i + i * j + i * j * k + 1;
        function_scalings(3 * (4 * i + 2 * j + k) + 2) = 0.;
      }
    }
  }
  CoupledFunction2<float> func(function_scalings.const_view(), 3, 2);
  float tol = 1e-5;
  zisa::array<float, 2> values(zisa::shape_t<2>(2, 6), zisa::device_type::cpu);
  values(0, 0) = 1.;
  values(0, 1) = 1.;
  values(0, 2) = 1.;
  values(0, 3) = 0.;
  values(0, 4) = 1.;
  values(0, 5) = 1.;
  values(1, 0) = 1.;
  values(1, 1) = 0.;
  values(1, 2) = -1.;
  values(1, 3) = 0.3;
  values(1, 4) = 0.6;
  values(1, 5) = 0.7;
  zisa::array_const_view<float, 2> values_const = values.const_view();
  zisa::array_const_view<float, 1> curr_values{
      zisa::shape_t<1>(3), &values_const(0, 0), zisa::device_type::cpu};
  float results[3];
  func(curr_values, results);
  ASSERT_NEAR(15.0, results[1], tol);
  func(zisa::array_const_view<float, 1>{zisa::shape_t<1>(3),
                                        &values_const(0, 3),
                                        zisa::device_type::cpu},
       results);
  ASSERT_NEAR(7.0, results[0], tol);

  func(zisa::array_const_view<float, 1>{zisa::shape_t<1>(3),
                                        &values_const(1, 0),
                                        zisa::device_type::cpu},
       results);
  ASSERT_NEAR(-2.0, results[1], tol);
  func(zisa::array_const_view<float, 1>{zisa::shape_t<1>(3),
                                        &values_const(1, 3),
                                        zisa::device_type::cpu},
       results);
  ASSERT_NEAR(5.664, results[0], tol);
  func(zisa::array_const_view<float, 1>{zisa::shape_t<1>(3),
                                        &values_const(1, 3),
                                        zisa::device_type::cpu},
       results);
  ASSERT_NEAR(0., results[2], tol);
}

TEST(CoupledFunctionTests, TestCoupled2) {

  zisa::array<float, 1> function_scalings(zisa::shape_t<1>(9 * 2),
                                          zisa::device_type::cpu);
  for (int i = 0; i < 9 * 2; i++) {
    function_scalings(i) = 0.;
  }
  CoupledFunction2<float> func(function_scalings.const_view(), 2, 3);
  float tol = 1e-10;
  zisa::array<float, 1> values(zisa::shape_t<1>(2), zisa::device_type::cpu);
  values(0) = 1.;
  values(1) = 0.5;
  zisa::array<float, 1> result(zisa::shape_t<1>(2), zisa::device_type::cpu);
  result(0) = -1.;
  result(1) = -2.;
  func(values, result.view());
  ASSERT_NEAR(0.0, result[0], tol);
}

TEST(CoupledFunctionTests, TEST_FROM_ARRAY2) {
  zisa::array<float, 1> function_scalings(zisa::shape_t<1>(8 * 3),
                                          zisa::device_type::cpu);
  // f(x) = 1 + x + y + xy + 2z + 2xz + 3 yz + 4xyz
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        function_scalings(3 * (4 * i + 2 * j + k)) = i + i * j + i * j * k + 1;
        function_scalings(3 * (4 * i + 2 * j + k) + 1) =
            i + i * j + i * j * k + 1;
        function_scalings(3 * (4 * i + 2 * j + k) + 2) = 0.;
      }
    }
  }
  CoupledFunction2<float> func(function_scalings.const_view(), 3, 2);
  float tol = 1e-5;
  zisa::array<float, 2> values(zisa::shape_t<2>(2, 6), zisa::device_type::cpu);
  values(0, 0) = 1.;
  values(0, 1) = 1.;
  values(0, 2) = 1.;
  values(0, 3) = 0.;
  values(0, 4) = 1.;
  values(0, 5) = 1.;
  values(1, 0) = 1.;
  values(1, 1) = 0.;
  values(1, 2) = -1.;
  values(1, 3) = 0.3;
  values(1, 4) = 0.6;
  values(1, 5) = 0.7;
  zisa::array_const_view<float, 2> values_const = values.const_view();
  zisa::array_const_view<float, 1> curr_values{
      zisa::shape_t<1>(3), &values_const(0, 0), zisa::device_type::cpu};
  zisa::array<float, 1> results(zisa::shape_t<1>(3));
  func(curr_values, results.view());
  ASSERT_NEAR(15.0, results[1], tol);
  func(zisa::array_const_view<float, 1>{zisa::shape_t<1>(3),
                                        &values_const(0, 3),
                                        zisa::device_type::cpu},
       results.view());
  ASSERT_NEAR(7., results[0], tol);

  func(zisa::array_const_view<float, 1>{zisa::shape_t<1>(3),
                                        &values_const(1, 0),
                                        zisa::device_type::cpu},
       results.view());
  ASSERT_NEAR(-2., results[1], tol);
  func(zisa::array_const_view<float, 1>{zisa::shape_t<1>(3),
                                        &values_const(1, 3),
                                        zisa::device_type::cpu},
       results.view());
  ASSERT_NEAR(5.664, results[0], tol);
  func(zisa::array_const_view<float, 1>{zisa::shape_t<1>(3),
                                        &values_const(1, 3),
                                        zisa::device_type::cpu},
       results.view());
  ASSERT_NEAR(0., results[2], tol);
}

} // namespace CoupledFunctionTests