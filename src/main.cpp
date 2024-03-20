#include "zisa/io/hdf5_serial_writer.hpp"
#include "zisa/io/hierarchical_file.hpp"
#include "zisa/io/hierarchical_writer.hpp"
#include <filesystem>
#include <iostream>
#include <pde_heat.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/device_type.hpp>

void add_bc_values_file(zisa::HierarchicalWriter &writer) {
  zisa::array<float, 2> data(zisa::shape_t<2>(10, 10));
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      data(i, j) = 1.;
    }
  }
  zisa::save(writer, data, "bc");
}

void add_initial_data_file(zisa::HierarchicalWriter &writer) {
  zisa::array<float, 2> data(zisa::shape_t<2>(10, 10));
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      data(i, j) = i * j + j;
    }
  }
  zisa::save(writer, data, "initial_data");
}

void add_sigma_file(zisa::HierarchicalWriter &writer) {
  zisa::array<float, 2> data(zisa::shape_t<2>(10, 10));
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      data(i, j) = 0.2 * (i % 3) + 0.3 * (j % 2);
    }
  }
  zisa::save(writer, data, "sigma");
}

void add_simple_nc_file() {

  // check if it has already been created
  if (std::filesystem::exists("data/simple_data.nc"))
    return;
  zisa::HDF5SerialWriter serial_writer("data/simple_data.nc");
  add_initial_data_file(serial_writer);
  add_bc_values_file(serial_writer);
  add_sigma_file(serial_writer);
}

enum BoundaryCondition { Dirichlet, Neumann, Periodic };

int main() {
  add_simple_nc_file();

  zisa::array<float, 2> heat_kernel(zisa::shape_t<2>(3, 3));
  float scalar = 0.1; // k / dt^2
  heat_kernel(0, 0) = 0;
  heat_kernel(0, 1) = scalar;
  heat_kernel(0, 2) = 0;
  heat_kernel(1, 0) = scalar;
  heat_kernel(1, 1) = 1 - 4 * scalar;
  heat_kernel(1, 2) = scalar;
  heat_kernel(2, 0) = 0;
  heat_kernel(2, 1) = scalar;
  heat_kernel(2, 2) = 0;

  BoundaryCondition bc = BoundaryCondition::Periodic;

  auto func = [](double x) -> double { return x; };
// construct a pde of the heat equation with Dirichlet boundary conditions
#if CUDA_AVAILABLE
  zisa::array<float, 2> heat_kernel_gpu(zisa::shape_t<2>(3, 3),
                                        zisa::device_type::cuda);
  zisa::copy(heat_kernel_gpu, heat_kernel);
  std::cout << "case_gpu" << std::endl;

  PDEHeat<float, BoundaryCondition, decltype> pde(8, 8, zisa::device_type::cuda, bc, func);
#else
  std::cout << "case_cpu" << std::endl;
  PDEHeat<float, BoundaryCondition, decltype(func)> pde(8, 8, zisa::device_type::cpu, bc, func);
#endif

  pde.read_values("data/simple_data.nc");
  pde.print();

  pde.apply();
  pde.print();
  for (int i = 0; i < 1000; i++) {
    pde.apply();
  }
  pde.print();

  return 0;
}
