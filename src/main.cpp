#include "helpers_main.hpp"
#include "io/netcdf_writer.hpp"
#include <chrono>
#include <coupled_function.hpp>
#include <iostream>
#include <pde_heat.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/device_type.hpp>

// void small_example() {
//   BoundaryCondition bc = BoundaryCondition::Periodic;
//   zisa::array<float, 1> function_scalings(zisa::shape_t<1>(16),
//                                           zisa::device_type::cpu);
//   for (int i = 0; i < 16; i++) {
//     function_scalings(i) = 0;
//   }
//   CoupledFunction<float> func_coupled_cpu(function_scalings.const_view(), 3,
//   2);
// #if CUDA_AVAILABLE
//   std::cout << "case_gpu" << std::endl;
//   zisa::array<float, 1> function_scalings_cuda(zisa::shape_t<1>(16),
//                                                zisa::device_type::cuda);
//   zisa::copy(function_scalings_cuda, function_scalings);
//   CoupledFunction<float> func_coupled_cuda(
//       function_scalings_cuda.const_view(), 3, 2);

//   PDEHeat<3, float, CoupledFunction<float>> pde(
//       8, 8, zisa::device_type::cuda, bc, func_coupled_cuda, 0.1, 0.1);
// #else
//   std::cout << "case_cpu" << std::endl;
//   PDEHeat<3, float, CoupledFunction<float>> pde(
//       8, 8, zisa::device_type::cpu, bc, func_coupled_cpu, 0.1, 0.1);
// #endif
//   pde.read_values("data/simple_data.nc");

//   NetCDFPDEWriter<float> writer(3, 1., 1, 10, 1, 10, 1, 1, 1, 1, 1, 1,
//   "output.nc"); auto begin = std::chrono::steady_clock::now(); for (int i =
//   0; i < 1000; i++) {
//     pde.apply_with_snapshots(1., 1000, 3, writer);
//   }
//   auto end = std::chrono::steady_clock::now();
//   std::cout << "time for 1000 iterations on cpu is "
//             << std::chrono::duration_cast<std::chrono::microseconds>(end -
//                                                                      begin)
//                    .count()
//             << std::endl;
//   pde.print();
// }

int main() {
  //   add_simple_nc_file();
  //   add_medium_nc_file();

  //   small_example();

  //   return 0;
}
