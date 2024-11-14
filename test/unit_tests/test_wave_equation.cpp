#include "pde_base.hpp"
#include "run_from_netcdf.hpp"
#include "zisa/memory/device_type.hpp"
#include <coupled_function.hpp>
#include <gtest/gtest.h>
#include <pde_wave.hpp>
#include <zisa/memory/array.hpp>

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
                      CoupledFunction<float> func, BoundaryCondition bc) {
  PDEWave<1, float, CoupledFunction<float>> pde(8, 8, memory_location, bc, func,
                                                0.1, 0.1, 1.0, 1.0);
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
  CoupledFunction<float> func(function_scalings_cuda, 1, 1);
#else
  CoupledFunction<float> func(function_scalings, 1, 1);
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
                          CoupledFunction<float> func, BoundaryCondition bc) {

  PDEWave<1, float, CoupledFunction<float>> pde(8, 8, memory_location, bc, func,
                                                0.1, 0.1, 1.0, 1.0);

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
  CoupledFunction<float> func(function_scalings_cuda, 1, 1);
#else
  CoupledFunction<float> func(function_scalings, 1, 1);
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
                        CoupledFunction<float> func, BoundaryCondition bc) {
  PDEWave<1, float, CoupledFunction<float>> pde(98, 98, memory_location, bc,
                                                func, 0.01, 0.01, 1.0, 1.0
);

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
  zisa::array<float, 1> function_scalings_cuda(zisa::shape_t<1>(1),
                                               zisa::device_type::cuda);
  zisa::copy(function_scalings_cuda, function_scalings);
  CoupledFunction<float> func(function_scalings_cuda, 1, 1);
#else
  CoupledFunction<float> func(function_scalings, 1, 1);
#endif

  test_linear_helper(data.const_view(), sigma_values.const_view(),
                     deriv_data.const_view(), memory_location, func,
                     BoundaryCondition::Dirichlet);
}

void check_results(int nx, int ny) {
  int ncid;
  ASSERT_TRUE(nc_open("data/test_wave_out.nc", NC_NOWRITE, &ncid) == NC_NOERR);
  int type_of_equation;
  ASSERT_TRUE(nc_get_att(ncid, NC_GLOBAL, "type_of_equation",
                         &type_of_equation) == NC_NOERR);
  ASSERT_TRUE(type_of_equation == 1);

  int n_members;
  ASSERT_TRUE(nc_get_att(ncid, NC_GLOBAL, "n_members", &n_members) == NC_NOERR);
  ASSERT_TRUE(n_members == 2);

  int x_size, y_size;
  ASSERT_TRUE(nc_get_att(ncid, NC_GLOBAL, "n_x", &x_size) == NC_NOERR);
  ASSERT_TRUE(nc_get_att(ncid, NC_GLOBAL, "n_y", &y_size) == NC_NOERR);
  ASSERT_TRUE(x_size == nx);
  ASSERT_TRUE(y_size == ny);

  zisa::array<float, 2> final_value(zisa::shape_t<2>(nx + 2, 3 * (ny + 2)));
  int varid;
  ASSERT_TRUE(nc_inq_varid(ncid, "data", &varid) == NC_NOERR);
  size_t startp[4] = {0, 2, 0, 0};
  size_t countp[4] = {1, 1, (size_t)nx + 2, (size_t)3 * (ny + 2)};
  ASSERT_TRUE(nc_get_vara(ncid, varid, startp, countp,
                          final_value.view().raw()) == NC_NOERR);

  // ensure correct final function values of first member
  float tol = 1e-1;
  float sol_2 = 1.5 * std::exp(2.) + 0.5 * std::exp(-2.) - 6.;
  for (int i = 0; i < nx + 2; i++) {
    for (int j = 0; j < ny + 2; j++) {
      //  f_1(x, y, z) = 1 => x(t) = t^2/2, x'(0) = 0
      ASSERT_NEAR(final_value(i, 3 * j), 2., tol);
      // f_2(x, y, z) = 2x + y = t^2 + y => y(t) = 1.5exp(t) + 0.5exp(-t) - t^2
      // - 2, y'(0) = 1
      ASSERT_NEAR(final_value(i, 3 * j + 1), sol_2, tol);
      // f_3(x, y, z) = 6x = 3t^2 => z(t) = t^4/4, z'(0) = 0
      ASSERT_NEAR(final_value(i, 3 * j + 2), 4., tol);
    }
  }
  startp[0] = 1;
  ASSERT_TRUE(nc_get_vara(ncid, varid, startp, countp,
                          final_value.view().raw()) == NC_NOERR);
  float sol_3 = 0.75 * std::exp(2.) + 0.25 * std::exp(-2.);
  for (int i = 0; i < nx + 2; i++) {
    for (int j = 0; j < ny + 2; j++) {
      // x(t) = 0.5 + t
      ASSERT_NEAR(final_value(i, 3 * j), 2.5, tol);
      // z(t) = 0.75 exp(t) + 0.25 exp(-t)
      ASSERT_NEAR(final_value(i, 3 * j + 2), sol_3, tol);
    }
  }
  // test periodicity
  for (int i = 0; i < ny + 2; i++) {
    // upper boundary
    ASSERT_NEAR(final_value(1, 3 * i + 1), final_value(nx - 1, 3 * i + 1), tol);
    // lower boundary
    ASSERT_NEAR(final_value(nx - 2, 3 * i + 1), final_value(0, 3 * i + 1), tol);
  }
  for (int i = 1; i < nx + 1; i++) {
    // left boundary
    ASSERT_NEAR(final_value(i, 4), final_value(i, 3 * (ny + 2) - 2), tol);
    // right boundary
    ASSERT_NEAR(final_value(i, 3 * ny + 1), final_value(i, 1), tol);
  }
}

// This test is designed to test if the implementatino
// can handle more than one member and more than one coupled functions
// simultaneously and if the periodic boundary conditions are right.
TEST(WaveEquationTests, TEST_FROM_NC) {
  ASSERT_TRUE(
      std::system("python scripts/create_test_input_wave_periodic.py") == 0);

  int nx = 64, ny = 64;
  NetCDFPDEReader reader("data/test_wave.nc");

  zisa::array<float, 2> init_data_1(zisa::shape_t<2>(nx + 2, 3 * (ny + 2)),
                                    zisa::device_type::cpu);
  reader.write_variable_of_member_to_array(
      "initial_data", init_data_1.view().raw(), 0, nx + 2, 3 * (ny + 2));

  // ensure correct initial conditions for member 1
  float tol = 1e-5;
  for (int i = 0; i < nx + 2; i++) {
    for (int j = 0; j < 3 * (ny + 2); j++) {
      ASSERT_NEAR(init_data_1(i, j), 0, tol);
    }
  }

  zisa::array<float, 2> init_data_2(zisa::shape_t<2>(nx + 2, 3 * (ny + 2)),
                                    zisa::device_type::cpu);
  reader.write_variable_of_member_to_array(
      "initial_data", init_data_2.view().raw(), 1, nx + 2, 3 * (ny + 2));

  // ensure correct initial conditions for member 2
  for (int i = 0; i < nx + 2; i++) {
    for (int j = 0; j < (ny + 2); j++) {
      ASSERT_NEAR(init_data_2(i, 3 * j), 0.5, tol);
      ASSERT_NEAR(init_data_2(i, 3 * j + 2), 1., tol);
    }
  }

  run_simulation<float>(reader, zisa::device_type::cpu);
  check_results(nx, ny);

// calculate on cuda
#if CUDA_AVAILABLE
  std::remove("data/test_wave_out.nc");
  run_simulation<float>(reader, zisa::device_type::cuda);
  check_results(nx, ny);
#endif // CUDA_AVAILABLE
}
} // namespace WaveEquationTests