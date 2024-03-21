#include "zisa/io/hdf5_serial_writer.hpp"
#include "zisa/io/hierarchical_file.hpp"
#include "zisa/io/hierarchical_writer.hpp"
#include <filesystem>
#include <iostream>
#include <pde_heat.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/device_type.hpp>
#include <generic_function.hpp>

template <typename Scalar>
class MyFunConst: public GenericFunction<float> {
public:
  MyFunConst(Scalar x): x_(x) {}
#if CUDA_AVAILABLE
  inline __host__ __device__ Scalar operator()(Scalar x) {
    return x_;
  }

#else
  inline Scalar operator()(Scalar x){
    return x_;
  }
#endif


  
private:
  Scalar x_;
};

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

/* add sigma values already stored on half gridpoint.
   Store them in an array of size 2n-3 x m-1

   x_00              x_01              x_02              x_03
                       |                 |
                     sigma_00          sigma_01
                       |                 |
   x_10 - sigma_10 - x_11 - sigma_11 - x_12 - sigma_12 - x_13
                       |                 |
                     sigma_20          sigma_21
                       |                 |
   x_20              x_21              x_22              x_23

                                ||
                                \/

                sigma_00, sigma_01, 0
                sigma_10, sigma_11, sigma_12
                sigma_20, sigma_21, 0

    note that in this toy example only x_11 and x_12 are not on the boundary
*/
void add_sigma_file(zisa::HierarchicalWriter &writer) {
  zisa::array<float, 2> data(zisa::shape_t<2>(17, 9));
  for (int i = 0; i < 17; i++) {
    for (int j = 0; j < 9; j++) {
      data(i, j) = 1.;
    }
  }
  zisa::save(writer, data, "sigma");
}

void add_simple_nc_file() {
  // check if it has already been created
  if (std::filesystem::exists("data/simple_data.nc"))
    return;
  // TODO:
  zisa::HDF5SerialWriter serial_writer("data/simple_data.nc");
  add_initial_data_file(serial_writer);
  add_bc_values_file(serial_writer);
  add_sigma_file(serial_writer);
}

enum BoundaryCondition { Dirichlet, Neumann, Periodic };

int main() {
  add_simple_nc_file();

  BoundaryCondition bc = BoundaryCondition::Periodic;

  auto func_cpu = [](float /*x*/) -> float { return 0; };
// construct a pde of the heat equation with Dirichlet boundary conditions
  MyFunConst<float> func(0.);
#if CUDA_AVAILABLE
  std::cout << "case_gpu" << std::endl;

  PDEHeat<float, BoundaryCondition, MyFunConst<float>> pde(
      8, 8, zisa::device_type::cuda, bc, func_cuda);
#else
  std::cout << "case_cpu" << std::endl;
  PDEHeat<float, BoundaryCondition, MyFunConst<float>> pde(
      8, 8, zisa::device_type::cpu, bc, func);
#endif

  pde.read_values("data/simple_data.nc");
  pde.print();

  pde.apply(0.1);
  pde.print();
  for (int i = 0; i < 1000; i++) {
    pde.apply(0.1);
  }
  pde.print();

  return 0;
}
