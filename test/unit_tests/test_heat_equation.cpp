#include "pde_base.hpp"
#include "zisa/memory/device_type.hpp"
#include <generic_function.hpp>
#include <gtest/gtest.h>
#include <pde_heat.hpp>
#include <zisa/memory/array.hpp>

namespace HeatEquationTests {

// creates simple data array where all values are set to value,
// if CUDA_AVAILABLE on cudagpu, else on cpu
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
  GenericFunction<float> func;

  PDEHeat<float, GenericFunction<float>> pde(
      8, 8, memory_location, BoundaryCondition::Dirichlet, func, 0.1, 0.1);
  pde.read_values(data.const_view(), sigma_values.const_view(),
                  data.const_view());
  for (int i = 0; i < 1000; i++) {
    pde.apply(0.1);
  }
  zisa::array_const_view<float, 2> result = pde.get_data();
  float tol = 1e-10;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      ASSERT_NEAR(0.0, result(i, j), tol);
    }
  }
}

// sigma = 0, f = 0 => u(x, y, t) = u(x, y, 0)
TEST(HeatEquationTests, TEST_U_CONSTANT) {
  const int array_size = 10; // 2 border values included
#if CUDA_AVAILABLE
  const zisa::device_type memory_location = zisa::device_type::cuda;
#else
  const zisa::device_type memory_location = zisa::device_type::cpu;
#endif

  zisa::array<float, 2> data(zisa::shape_t<2>(array_size, array_size),
                             memory_location);
#if CUDA_AVAILABLE
  zisa::array<float, 2> data_tmp(zisa::shape_t<2>(array_size, array_size),
                                 zisa::device_type::cpu);
#endif

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
#if CUDA_AVAILABLE
      data_tmp(i, j) = i * j;
#else
      data(i, j) = i * j;
#endif
    }
  }

#if CUDA_AVAILABLE
  zisa::copy(data, data_tmp);
#endif

  // if sigma == 0, then du == 0 everywhere => u is constant
  zisa::array<float, 2> sigma_values = create_value_data<float>(
      2 * array_size - 3, array_size - 1, 0., memory_location);

  // f == 0 everywhere
  GenericFunction<float> func;

  PDEHeat<float, GenericFunction<float>> pde(8, 8, memory_location,
                                             BoundaryCondition::Dirichlet, func,
                                             0.1, 0.1);

  pde.read_values(data.const_view(), sigma_values.const_view(),
                  data.const_view());
  for (int i = 0; i < 1000; i++) {
    pde.apply(0.1);
  }
  zisa::array_const_view<float, 2> result = pde.get_data();
  float tol = 1e-10;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      ASSERT_NEAR(data(i, j), result(i, j), tol);
    }
  }
}

// TODO: sigma = 0, f = const => du = f => u(x, y, t) = u(x, y, 0) + f * t
TEST(HeatEquationTests, TEST_F_CONSTANT) {
  const int array_size = 10; // 2 border values included
#if CUDA_AVAILABLE
  const zisa::device_type memory_location = zisa::device_type::cuda;
#else
  const zisa::device_type memory_location = zisa::device_type::cpu;
#endif

  zisa::array<float, 2> data(zisa::shape_t<2>(array_size, array_size),
                             memory_location);
#if CUDA_AVAILABLE
  zisa::array<float, 2> data_tmp(zisa::shape_t<2>(array_size, array_size),
                                 zisa::device_type::cpu);
#endif

  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
#if CUDA_AVAILABLE
      data_tmp(i, j) = i * j;
#else
      data(i, j) = i * j;
#endif
    }
  }

#if CUDA_AVAILABLE
  zisa::copy(data, data_tmp);
#endif
  zisa::array<float, 2> sigma_values = create_value_data<float>(
      2 * array_size - 3, array_size - 1, 0., memory_location);

  GenericFunction<float> func;
  func.set_const(1.);
  PDEHeat<float, GenericFunction<float>> pde(8, 8, memory_location,
                                             BoundaryCondition::Dirichlet, func,
                                             0.1, 0.1);

  pde.read_values(data.const_view(), sigma_values.const_view(),
                  data.const_view());
  for (int i = 0; i < 100; i++) {
    pde.apply(0.1);
  }
  zisa::array_const_view<float, 2> result = pde.get_data();
  float tol = 1e-3;
  // values on boundary do not change because of dirichlet bc
  for (int i = 1; i < 9; i++) {
    for (int j = 1; j < 9; j++) {
      ASSERT_NEAR(data(i, j) + 10, result(i, j), tol);
    }
  }
}
} // namespace HeatEquationTests