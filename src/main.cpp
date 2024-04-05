#include "helpers_main.hpp"
#include <chrono>
#include <generic_function.hpp>
#include <io/netcdf_writer.hpp>
#include <iostream>
#include <pde_heat.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/device_type.hpp>

void small_example() {
  BoundaryCondition bc = BoundaryCondition::Periodic;
  GenericFunction<float> func;
#if CUDA_AVAILABLE
  std::cout << "case_gpu" << std::endl;

  PDEHeat<float, GenericFunction<float>> pde(
      8, 8, zisa::device_type::cuda, bc, func, 0.1, 0.1);
#else
  std::cout << "case_cpu" << std::endl;
  PDEHeat<float, GenericFunction<float>> pde(
      8, 8, zisa::device_type::cpu, bc, func, 0.1, 0.1);
#endif
  pde.read_values("data/simple_data.nc");

  NetCDFPDEWriter<float> writer(3, 1, 1, 10, 0., 1., 10, 0., 1.,
                                "out/result01.nc");
  auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < 1000; i++) {
    pde.apply_with_snapshots(1., 1000, 3, writer);
  }
  auto end = std::chrono::steady_clock::now();
  std::cout << "time for 1000 iterations on cpu is "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     begin)
                   .count()
            << std::endl;
  pde.print();
}

int main() {
  add_simple_nc_file();
  add_medium_nc_file();

  small_example();

  return 0;
}
