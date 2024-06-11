#include "pde_base.hpp"
#include "zisa/memory/device_type.hpp"
#include <coupled_function.hpp>
#include <gtest/gtest.h>
#include <pde_heat.hpp>
#include <zisa/memory/array.hpp>

// TODO: add tests for neumann and periodic bc, larger and nonsymmetric grids
//       add tests for sigma != constant (how to get solution)

namespace HeatEquationTests {

// helper function which creates simple data array where all values are set to
// value, if CUDA_AVAILABLE on gpu, else on cpu
template <typename Scalar>
inline zisa::array<Scalar, 2>
create_value_data(int x_size, int y_size, Scalar value,
                  zisa::device_type memory_location) {
  zisa::array<Scalar, 2> data(zisa::shape_t<2>(x_size, y_size),
                              zisa::device_type::cpu);
  for (int i = 0; i < x_size; i++) {
    for (int j = 0; j < y_size; j++) {
      data(i, j) = value;
    }
  }
  if (memory_location == zisa::device_type::cpu) {
    return data;
  }
#if CUDA_AVAILABLE
  else if (memory_location == zisa::device_type::cuda) {
    zisa::array<Scalar, 2> data_gpu(zisa::shape_t<2>(x_size, y_size),
                                    zisa::device_type::cuda);
    zisa::copy(data_gpu, data);
    return data_gpu;
  }
#endif
  else {
    std::cout << "device type not supported yet in test_heat_equation"
              << std::endl;
    exit(-1);
  }
}
// helper function which creates simple data array where all arr(i, j) = i*j
template <typename Scalar>
inline zisa::array<Scalar, 2>
create_simple_data(int x_size, int y_size, zisa::device_type memory_location) {
  zisa::array<Scalar, 2> data(zisa::shape_t<2>(x_size, y_size),
                              zisa::device_type::cpu);
  for (int i = 0; i < x_size; i++) {
    for (int j = 0; j < y_size; j++) {
      data(i, j) = i * j;
    }
  }
  if (memory_location == zisa::device_type::cpu) {
    return data;
  }
#if CUDA_AVAILABLE
  else if (memory_location == zisa::device_type::cuda) {
    zisa::array<Scalar, 2> data_gpu(zisa::shape_t<2>(x_size, y_size),
                                    zisa::device_type::cuda);
    zisa::copy(data_gpu, data);
    return data_gpu;
  }
#endif
  else {
    std::cout << "device type not supported yet in test_heat_equation"
              << std::endl;
    exit(-1);
  }
}

// u(x, y, 0) = 0, f = 0, sigma = 0 => u(x, y, t) = 0
TEST(HeatEquationTests, TEST_ZERO) {
  const int array_size = 10; // 2 border values included
#if CUDA_AVAILABLE
  const zisa::device_type memory_location = zisa::device_type::cuda;
#else
  const zisa::device_type memory_location = zisa::device_type::cpu;
#endif
  zisa::array<float, 2> data =
      create_value_data<float>(array_size, array_size, 0., memory_location);

  zisa::array<float, 2> sigma_values = create_value_data<float>(
      2 * array_size - 3, array_size - 1, 1., memory_location);
  // f == 0 everywhere
  zisa::array<float, 1> function_scalings(zisa::shape_t<1>(1), memory_location);
#if CUDA_AVAILABLE
  zisa::array<float, 1> tmp_cpu(zisa::shape_t<1>(1), zisa::device_type::cpu);
  tmp_cpu(0) = 0.;
  zisa::copy(function_scalings, tmp_cpu);
#else
  function_scalings(0) = 0.;
#endif
  CoupledFunction<float> func(function_scalings.const_view(), 1, 1);

  PDEHeat<1, float, CoupledFunction<float>> pde(
      8, 8, memory_location, BoundaryCondition::Dirichlet, func, 0.1, 0.1);

  pde.read_values(data.const_view(), sigma_values.const_view(),
                  data.const_view());
  for (int i = 0; i < 1000; i++) {
    pde.apply(0.1);
  }
#if CUDA_AVAILABLE
  zisa::array_const_view<float, 2> result_gpu = pde.get_data();
  zisa::array<float, 2> result(result_gpu.shape());
  zisa::copy(result, result_gpu);
#else
  zisa::array_const_view<float, 2> result = pde.get_data();
#endif
  float tol = 1e-10;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      ASSERT_NEAR(0.0, result(i, j), tol);
    }
  }
}

// u(x, y, 0) != 0, f = 0, sigma = 0 => u(x, y, t) = u(x, y, 0)
TEST(HeatEquationTests, TEST_U_CONSTANT) {
  const int array_size = 10; // 2 border values included
#if CUDA_AVAILABLE
  const zisa::device_type memory_location = zisa::device_type::cuda;
#else
  const zisa::device_type memory_location = zisa::device_type::cpu;
#endif

  zisa::array<float, 2> data =
      create_simple_data<float>(array_size, array_size, memory_location);

  // if sigma == 0, then du == 0 everywhere => u is constant
  zisa::array<float, 2> sigma_values = create_value_data<float>(
      2 * array_size - 3, array_size - 1, 0., memory_location);

  // f == 0 everywhere
  zisa::array<float, 1> function_scalings(zisa::shape_t<1>(3), memory_location);
#if CUDA_AVAILABLE
  zisa::array<float, 1> tmp_cpu(zisa::shape_t<1>(3), zisa::device_type::cpu);
  tmp_cpu(0) = 0.;
  tmp_cpu(1) = 0.;
  tmp_cpu(2) = 0.;
  zisa::copy(function_scalings, tmp_cpu);
#else
  function_scalings(0) = 0.;
  function_scalings(1) = 0.;
  function_scalings(2) = 0.;
#endif
  CoupledFunction<float> func(function_scalings.const_view(), 1, 3);

  PDEHeat<1, float, CoupledFunction<float>> pde(
      8, 8, memory_location, BoundaryCondition::Dirichlet, func, 0.1, 0.1);

  pde.read_values(data.const_view(), sigma_values.const_view(),
                  data.const_view());
  for (int i = 0; i < 1000; i++) {
    pde.apply(0.1);
  }

#if CUDA_AVAILABLE
  zisa::array_const_view<float, 2> result_gpu = pde.get_data();
  zisa::array<float, 2> result(result_gpu.shape());
  zisa::copy(result, result_gpu);
  zisa::array<float, 2> data_cpu =
      create_simple_data<float>(array_size, array_size, zisa::device_type::cpu);
#else
  zisa::array_const_view<float, 2> result = pde.get_data();
#endif

  float tol = 1e-10;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
#if CUDA_AVAILABLE
      ASSERT_NEAR(data_cpu(i, j), result(i, j), tol);
#else
      ASSERT_NEAR(data(i, j), result(i, j), tol);
#endif
    }
  }
}

// u(x, y, 0) != 0, sigma = 0, f = const => du = f => u(x, y, t) = u(x, y, 0) +
// f * t
TEST(HeatEquationTests, TEST_F_CONSTANT) {
  const int array_size = 10; // 2 border values included
#if CUDA_AVAILABLE
  const zisa::device_type memory_location = zisa::device_type::cuda;
#else
  const zisa::device_type memory_location = zisa::device_type::cpu;
#endif

  auto data =
      create_simple_data<float>(array_size, array_size, memory_location);

  zisa::array<float, 2> sigma_values = create_value_data<float>(
      2 * array_size - 3, array_size - 1, 0., memory_location);

  zisa::array<float, 1> function_scalings(zisa::shape_t<1>(3), memory_location);
#if CUDA_AVAILABLE
  zisa::array<float, 1> tmp_cpu(zisa::shape_t<1>(3), zisa::device_type::cpu);
  tmp_cpu(0) = 0.5;
  tmp_cpu(1) = 0.;
  tmp_cpu(2) = 0.;
  zisa::copy(function_scalings, tmp_cpu);
#else
  function_scalings(0) = 0.5;
  function_scalings(1) = 0.;
  function_scalings(2) = 0.;
#endif
  CoupledFunction<float> func(function_scalings.const_view(), 1, 3);
  PDEHeat<1, float, CoupledFunction<float>> pde(
      8, 8, memory_location, BoundaryCondition::Dirichlet, func, 0.1, 0.1);

  pde.read_values(data.const_view(), sigma_values.const_view(),
                  data.const_view());
  // pde.print();
  for (int i = 0; i < 200; i++) {
    pde.apply(0.1);
  }
  // pde.print();

#if CUDA_AVAILABLE
  zisa::array_const_view<float, 2> result_gpu = pde.get_data();
  zisa::array<float, 2> result(result_gpu.shape());
  zisa::copy(result, result_gpu);
  zisa::array<float, 2> data_cpu =
      create_simple_data<float>(array_size, array_size, zisa::device_type::cpu);
#else
  zisa::array_const_view<float, 2> result = pde.get_data();
#endif

  float tol = 1e-3;
  // values on boundary do not change because of dirichlet bc
  for (int i = 1; i < 9; i++) {
    for (int j = 1; j < 9; j++) {
#if CUDA_AVAILABLE
      ASSERT_NEAR(data_cpu(i, j) + 10, result(i, j), tol);
#else
      ASSERT_NEAR(data(i, j) + 10, result(i, j), tol);
#endif
    }
  }
}

// u(x, y, 0) != 0, sigma = 0, f(x) = a*x => du = a*u => u(x, y, t) = u(x, y, 0)
// * exp(a * t)

// this only works for very small times because
// the boundary stays constant but the inner values increases, which
// leads to a huge 2nd derivative in the corner values
// note that using float or double does not increase or decrease the error,
// it's because of algorithmic instabilities...could be prevented by reducing dx
// and dy

// this error should be resulved when using sigma=0 but nan * 0 != 0
// you could prevent it by using a max_value in convolve_sigma_add_f?
TEST(HeatEquationTests, TEST_F_LINEAR) {
  const int array_size = 10; // 2 border values included
#if CUDA_AVAILABLE
  const zisa::device_type memory_location = zisa::device_type::cuda;
#else
  const zisa::device_type memory_location = zisa::device_type::cpu;
#endif

  zisa::array<float, 2> data =
      create_simple_data<float>(array_size, array_size, memory_location);

  zisa::array<float, 2> sigma_values = create_value_data<float>(
      2 * array_size - 3, array_size - 1, 0., memory_location);

  zisa::array<float, 1> function_scalings(zisa::shape_t<1>(4), memory_location);
#if CUDA_AVAILABLE
  zisa::array<float, 1> tmp_cpu(zisa::shape_t<1>(4), zisa::device_type::cpu);
  tmp_cpu(0) = 0.;
  tmp_cpu(1) = 0.5;
  tmp_cpu(2) = 0.;
  tmp_cpu(3) = 0.;
  zisa::copy(function_scalings, tmp_cpu);
#else
  function_scalings(0) = 0.;
  function_scalings(1) = 0.5;
  function_scalings(2) = 0.;
  function_scalings(3) = 0.;
#endif
  CoupledFunction<float> func(function_scalings.const_view(), 1, 4);

  PDEHeat<1, float, CoupledFunction<float>> pde(
      8, 8, memory_location, BoundaryCondition::Dirichlet, func, 0.1, 0.1);

  pde.read_values(data.const_view(), sigma_values.const_view(),
                  data.const_view());
  // t = 0.2
  for (int i = 0; i < 20; i++) {
    pde.apply(0.01);
  }

#if CUDA_AVAILABLE
  zisa::array_const_view<float, 2> result_gpu = pde.get_data();
  zisa::array<float, 2> result(result_gpu.shape());
  zisa::copy(result, result_gpu);
  zisa::array<float, 2> data_cpu =
      create_simple_data<float>(array_size, array_size, zisa::device_type::cpu);
#else
  zisa::array_const_view<float, 2> result = pde.get_data();
#endif

  float tol = 1e-1;
  // values on boundary do not change because of dirichlet bc
  for (int i = 1; i < 9; i++) {
    for (int j = 1; j < 9; j++) {
#if CUDA_AVAILABLE
      ASSERT_NEAR(data_cpu(i, j) * std::exp(0.1), result(i, j), tol);
#else
      ASSERT_NEAR(data(i, j) * std::exp(0.1), result(i, j), tol);
#endif
    }
  }
}
} // namespace HeatEquationTests