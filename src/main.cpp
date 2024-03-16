#include "zisa/io/hdf5_serial_writer.hpp"
#include "zisa/io/hierarchical_file.hpp"
#include <iostream>
#include <pde_base.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/device_type.hpp>

void add_bc_values_file() {
  zisa::HDF5SerialWriter serial_writer("data/bc_8_8.nc");
  serial_writer.open_group("group_1");
  float data[10][10];
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      data[i][j] = 0.;
    }
  }
  std::size_t dims[2] = {10, 10};
  serial_writer.write_array(data, zisa::erase_data_type<float>(), "data_1", 2, dims);
  serial_writer.close_group();
  
}

void add_initial_data_file(){
  zisa::HDF5SerialWriter serial_writer("data/data_8_8.nc");
  serial_writer.open_group("group_1");
  float data[10][10];
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      data[i][j] = i * j + j;
    }
  }
  std::size_t dims[2] = {10, 10};
  serial_writer.write_array(data, zisa::erase_data_type<float>(), "data_1", 2, dims);
  serial_writer.close_group();
}

// Dirichlet BC means currently that f(x) = 0 forall x on boundary
// Neumann BC means currently that f'(x) = 0 forall x on boundary
// => f(x) = f(x + dt)
//TODO: add Dirichlet and Neumann BC for different values or functions
enum BoundaryCondition { Dirichlet, Neumann, Periodic };

int main() {

  // add_initial_data_file();
  // add_bc_values_file();

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

  BoundaryCondition bc = BoundaryCondition::Dirichlet;

  //construct a pde of the heat equation with Dirichlet boundary conditions
  #if CUDA_AVAILABLE
  zisa::array<float, 2> heat_kernel_gpu(zisa::shape_t<2>(3, 3),
                                        zisa::device_type::cuda);
  zisa::copy(heat_kernel_gpu, heat_kernel); std::cout << "case_gpu" << std::endl;

  PDEBase<float, BoundaryCondition> pde(8, 8, heat_kernel_gpu, bc);
  #else
  std::cout << "case_cpu" << std::endl;
  PDEBase<float, BoundaryCondition> pde(8, 8, heat_kernel, bc);
  #endif

  pde.read_initial_data("data/data_8_8.nc", "group_1", "data_1");
  pde.read_bc_values("data/bc_8_8.nc", "group_1", "data_1");
  pde.print();

  pde.apply();
  pde.print();

  for (int i = 0; i < 10; i++) {
    pde.apply();
    pde.print();
  }
  pde.print();

  return 0;
}
