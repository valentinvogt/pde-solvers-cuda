#include "zisa/io/hdf5_serial_writer.hpp"
#include "zisa/io/hierarchical_file.hpp"
#include <iostream>
#include <pde_base.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/device_type.hpp>
#include <heat_kernel.hpp>


void add_initial_data_file(){
  zisa::HDF5SerialWriter serial_writer("data/data_8_8.nc");
  serial_writer.open_group("group_1");
  float data[8][8];
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      data[i][j] = i * j + j;
    }
  }
  std::size_t dims[2] = {8, 8};
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
  BoundaryCondition bc = BoundaryCondition::Neumann;

  //construct a pde of the heat equation with Dirichlet boundary conditions
  #if CUDA_AVAILABLE
  const HeatKernel<float> heat_kernel_gpu(1., 0.1, ziza::device_type::cuda);
  std::cout << "case_gpu" << std::endl;
  PDEBase pde(8, 8, heat_kernel_gpu, bc);
  #else
  const HeatKernel<float> heat_kernel_cpu(1., 0.1, zisa::device_type::cpu);
  std::cout << "case_cpu" << std::endl;
  heat_kernel_cpu.print();
  PDEBase pde(8, 8, heat_kernel_cpu, bc);
  #endif

  pde.read_initial_data("data/data_8_8.nc", "group_1", "data_1");
  pde.print();

  pde.apply();
  pde.print();
  return 0;
}
