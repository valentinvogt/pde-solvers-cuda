#include "pde_base.hpp"
#include <generic_function.hpp>
#include <gtest/gtest.h>
#include <pde_heat.hpp>
#include <zisa/memory/array.hpp>

namespace HeatEquationTests {
// u(x, y, 0) = 0, f = 0, sigma = 0 => u(x, y, t) = 0
TEST(HeatEquationTests, TEST_ZERO) {
  int array_size = 10;
  zisa::array<float, 2> data(zisa::shape_t<2>(10, 10));
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      data(i, j) = 0.;
    }
  }

  zisa::array<float, 2> sigma_values(zisa::shape_t<2>(17, 9));
  for (int i = 0; i < 17; i++) {
    for (int j = 0; j < 9; j++) {
      sigma_values(i, j) = 1.;
    }
  }

  // f == 0 everywhere
  GenericFunction<float> func;
#if CUDA_AVAILABLE
  PDEHeat<float, GenericFunction<float>> pde(8, 8, zisa::device_type::cuda,
                                             BoundaryCondition::Dirichlet, func,
                                             0.1, 0.1);
#else
  PDEHeat<float, GenericFunction<float>> pde(8, 8, zisa::device_type::cpu,
                                             BoundaryCondition::Dirichlet, func,
                                             0.1, 0.1);
#endif
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
  int array_size = 10;
  zisa::array<float, 2> data(zisa::shape_t<2>(10, 10));
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      data(i, j) = i * j;
    }
  }

  // if sigma == 0, then du == 0 everywhere => u is constant
  zisa::array<float, 2> sigma_values(zisa::shape_t<2>(17, 9));
  for (int i = 0; i < 17; i++) {
    for (int j = 0; j < 9; j++) {
      sigma_values(i, j) = 0.;
    }
  }

  // f == 0 everywhere
  GenericFunction<float> func;
#if CUDA_AVAILABLE
  PDEHeat<float, GenericFunction<float>> pde(8, 8, zisa::device_type::cuda,
                                             BoundaryCondition::Dirichlet, func,
                                             0.1, 0.1);
#else
  PDEHeat<float, GenericFunction<float>> pde(8, 8, zisa::device_type::cpu,
                                             BoundaryCondition::Dirichlet, func,
                                             0.1, 0.1);
#endif
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
  int array_size = 10;
  zisa::array<float, 2> data(zisa::shape_t<2>(10, 10));
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      data(i, j) = i * j;
    }
  }

  zisa::array<float, 2> sigma_values(zisa::shape_t<2>(17, 9));
  for (int i = 0; i < 17; i++) {
    for (int j = 0; j < 9; j++) {
      sigma_values(i, j) = 0.;
    }
  }

  GenericFunction<float> func;
  func.set_const(1.);
#if CUDA_AVAILABLE
  PDEHeat<float, GenericFunction<float>> pde(8, 8, zisa::device_type::cuda,
                                             BoundaryCondition::Dirichlet, func,
                                             0.1, 0.1);
#else
  PDEHeat<float, GenericFunction<float>> pde(8, 8, zisa::device_type::cpu,
                                             BoundaryCondition::Dirichlet, func,
                                             0.1, 0.1);
#endif
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