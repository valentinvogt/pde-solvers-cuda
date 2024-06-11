#include "pde_base.hpp"
#include "zisa/memory/device_type.hpp"
#include <coupled_function.hpp>
#include <gtest/gtest.h>
#include <pde_wave.hpp>
#include <zisa/memory/array.hpp>

// TODO: add tests for neumann and periodic bc, larger and nonsymmetric grids
//       add tests for sigma != constant (how to get solution)

// TODO: implement for coupled function class
namespace WaveEquationTests {

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
    std::cout << "device type not supported yet in test_wave_equation"
              << std::endl;
    exit(-1);
  }
}
// helper function which creates simple data array where arr(i, j) = i*j
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
    std::cout << "device type not supported yet in test_wave_equation"
              << std::endl;
    exit(-1);
  }
}

void test_zero_helper(zisa::array_const_view<float, 2> data,
                      zisa::array_const_view<float, 2> sigma,
                      zisa::device_type memory_location,
                      CoupledFunction<float, 1, 1> func, BoundaryCondition bc) {
  PDEWave<1, float, CoupledFunction<float, 1, 1>> pde(8, 8, memory_location, bc,
                                                      func, 0.1, 0.1);
  pde.read_values(data, sigma, data, data);
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

// u(x, y, 0) = 0, f = 0, du = 0, sigma = 0 => u(x, y, t) = 0
TEST(WaveEquationTests, TEST_ZERO) {
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
  zisa::array<float, 1> function_scalings(zisa::shape_t<1>(1),
                                          zisa::device_type::cpu);
  function_scalings(0) = 0.;
#if CUDA_AVAILABLE
  zisa::array<float, 1> function_scalings_cuda(zisa::shape_t<1>(1),
                                               zisa::device_type::cuda);
  zisa::copy(function_scalings_cuda, function_scalings);
  CoupledFunction<float, 1, 1> func(function_scalings_cuda);
#else
  CoupledFunction<float, 1, 1> func(function_scalings);
#endif
  test_zero_helper(data.const_view(), sigma_values.const_view(),
                   memory_location, func, BoundaryCondition::Dirichlet);
  test_zero_helper(data.const_view(), sigma_values.const_view(),
                   memory_location, func, BoundaryCondition::Neumann);
  test_zero_helper(data.const_view(), sigma_values.const_view(),
                   memory_location, func, BoundaryCondition::Periodic);
}
void test_constant_helper(zisa::array_const_view<float, 2> data,
                          zisa::array_const_view<float, 2> sigma,
                          zisa::array_const_view<float, 2> deriv_data,
                          zisa::device_type memory_location,
                          CoupledFunction<float, 1, 1> func,
                          BoundaryCondition bc) {
  PDEWave<1, float, CoupledFunction<float, 1, 1>> pde(8, 8, memory_location, bc,
                                                      func, 0.1, 0.1);

  pde.read_values(data, sigma, data, deriv_data);
  for (int i = 0; i < 1000; i++) {
    pde.apply(0.1);
  }

#if CUDA_AVAILABLE
  zisa::array_const_view<float, 2> result_gpu = pde.get_data();
  zisa::array<float, 2> result(result_gpu.shape());
  zisa::copy(result, result_gpu);
  zisa::array<float, 2> data_cpu =
      create_simple_data<float>(10, 10, zisa::device_type::cpu);
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

// u(x, y, 0) != 0, f = 0, du = 0 sigma = 0 => u(x, y, t) = u(x, y, 0)
TEST(WaveEquationTests, TEST_U_CONSTANT) {
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

  zisa::array<float, 2> deriv_data =
      create_value_data<float>(array_size, array_size, 0., memory_location);

  // f == 0 everywhere

  zisa::array<float, 1> function_scalings(zisa::shape_t<1>(1),
                                          zisa::device_type::cpu);
  function_scalings(0) = 0.;
#if CUDA_AVAILABLE
  zisa::array<float, 1> function_scalings_cuda(zisa::shape_t<1>(1),
                                               zisa::device_type::cuda);
  zisa::copy(function_scalings_cuda, function_scalings);
  CoupledFunction<float, 1, 1> func(function_scalings_cuda);
#else
  CoupledFunction<float, 1, 1> func(function_scalings);
#endif

  test_constant_helper(data.const_view(), sigma_values.const_view(),
                       deriv_data.const_view(), memory_location, func,
                       BoundaryCondition::Dirichlet);
  test_constant_helper(data.const_view(), sigma_values.const_view(),
                       deriv_data.const_view(), memory_location, func,
                       BoundaryCondition::Neumann);
}

void test_linear_helper(zisa::array_const_view<float, 2> data,
                        zisa::array_const_view<float, 2> sigma,
                        zisa::array_const_view<float, 2> deriv_data,
                        zisa::device_type memory_location,
                        CoupledFunction<float, 1, 1> func,
                        BoundaryCondition bc) {
  PDEWave<1, float, CoupledFunction<float, 1, 1>> pde(98, 98, memory_location, bc,
                                                      func, 0.01, 0.01);

  pde.read_values(data, sigma, data, deriv_data);
  // apply for 10 seconds
  for (int i = 0; i < 10000; i++) {
    pde.apply(0.001);
  }

#if CUDA_AVAILABLE
  zisa::array_const_view<float, 2> result_gpu = pde.get_data();
  zisa::array<float, 2> result(result_gpu.shape());
  zisa::copy(result, result_gpu);
  zisa::array<float, 2> data_cpu =
      create_simple_data<float>(100, 100, zisa::device_type::cpu);
#else
  zisa::array_const_view<float, 2> result = pde.get_data();
#endif

  float tol = 1e-1;
  // do not test on boundary because dirichlet bc enforced
  for (int i = 1; i < 98; i++) {
    for (int j = 1; j < 98; j++) {
#if CUDA_AVAILABLE
      ASSERT_NEAR(data_cpu(i, j) + 100., result(i, j), tol);
#else
      ASSERT_NEAR(data(i, j) + 100., result(i, j), tol);
#endif
    }
  }
}
// u(x, y, 0) != 0, f = a, du = 0, sigma = 0 => u(x, y, t) = u(x, y, 0) +
// t^2/2 a
TEST(WaveEquationTests, TestConstantF) {
  const int array_size = 100; // 2 border values included
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

  zisa::array<float, 2> deriv_data =
      create_value_data<float>(array_size, array_size, 0., memory_location);

  // f == a everywhere
  zisa::array<float, 1> function_scalings(zisa::shape_t<1>(1),
                                          zisa::device_type::cpu);
  function_scalings(0) = 2.;

#if CUDA_AVAILABLE
  zisa::array<float, 1> function_scalings_cuda(zisa::shape_t<1>(4),
                                               zisa::device_type::cuda);
  zisa::copy(function_scalings_cuda, function_scalings);
  CoupledFunction<float, 1, 1> func(function_scalings_cuda);
#else
  CoupledFunction<float, 1, 1> func(function_scalings);
#endif

  test_linear_helper(data.const_view(), sigma_values.const_view(),
                     deriv_data.const_view(), memory_location, func,
                     BoundaryCondition::Dirichlet);
}

} // namespace WaveEquationTests