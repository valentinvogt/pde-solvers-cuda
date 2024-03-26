#include <generic_function.hpp>
#include <iostream>
#include <pde_heat.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/device_type.hpp>
#include <chrono>
#include "helpers_main.hpp"


enum BoundaryCondition { Dirichlet, Neumann, Periodic };

void small_example() {
  BoundaryCondition bc = BoundaryCondition::Periodic;
  GenericFunction<float> func;
#if CUDA_AVAILABLE
  std::cout << "case_gpu" << std::endl;

  PDEHeat<float, BoundaryCondition, GenericFunction<float>> pde(
      8, 8, zisa::device_type::cuda, bc, func);
#else
  std::cout << "case_cpu" << std::endl;
  PDEHeat<float, BoundaryCondition, GenericFunction<float>> pde(
      8, 8, zisa::device_type::cpu, bc, func);
#endif
  pde.read_values("data/simple_data.nc");
  auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < 1000; i++) {
    pde.apply(0.001);
  } 
  auto end = std::chrono::steady_clock::now();
  std::cout << "time for 1000 iterations on cpu is " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
  pde.print();
  
}

int main() {
  add_simple_nc_file();
  add_medium_nc_file();

  small_example();

//   BoundaryCondition bc = BoundaryCondition::Periodic;

//   // construct a pde of the heat equation with Dirichlet boundary conditions
//   GenericFunction<float> func;
// #if CUDA_AVAILABLE
//   std::cout << "case_gpu" << std::endl;

//   PDEHeat<float, BoundaryCondition, GenericFunction<float>> pde(
//       100, 100, zisa::device_type::cuda, bc, func);
// #else
//   std::cout << "case_cpu" << std::endl;
//   PDEHeat<float, BoundaryCondition, GenericFunction<float>> pde(
//       100, 100, zisa::device_type::cpu, bc, func);
// #endif

//   pde.read_values("data/data_100_100.nc");
//   pde.print();

//   auto begin = std::chrono::steady_clock::now();
//   for (int i = 0; i < 1000; i++) {
//     pde.apply(0.01);
//   }
//   auto end = std::chrono::steady_clock::now();
//   std::cout << "time for 1000 iterations on cuda is " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

  
//   PDEHeat<float, BoundaryCondition, GenericFunction<float>> pde_cpu(
//     100, 100, zisa::device_type::cpu, bc, func);
//   pde_cpu.read_values("data/data_100_100.nc");
//   begin = std::chrono::steady_clock::now();
//   for (int i = 0; i < 1000; i++) {
//     pde_cpu.apply(0.01);
//   }
//   end = std::chrono::steady_clock::now();
//   std::cout << "time for 1000 iterations on cpu is " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
//   // pde.print();

  return 0;
}
