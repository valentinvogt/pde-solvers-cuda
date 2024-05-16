#include "io/netcdf_reader.hpp"
#include "io/netcdf_writer.hpp"
#include "pde_base.hpp"
#include <coupled_function_2.hpp>
#include <iostream>
#include <netcdf.h>
#include <pde_heat.hpp>
#include <pde_wave.hpp>
#include <string>
#include <zisa/memory/array.hpp>

// TODO:
// read initial data, apply steps and save them to a output file
template <typename PDE, typename Scalar> void calculate_and_save_snapshots(PDE pde, const NetCDFPDEReader &reader) {
  pde.read_initial_data_from_netcdf(reader);
  pde.print();
  NetCDFPDEWriter<Scalar> writer(3, 1., 1, 10, 0., 1., 10, 0., 1., "data/out.nc");
  pde.apply_with_snapshots(1., 100, 3, writer);
}

template <typename Scalar> void run_simulation(const NetCDFPDEReader &reader) {
  int n_coupled = reader.get_n_coupled();
  int coupled_order = reader.get_coupled_function_order();
  zisa::array<Scalar, 1> function_scalings(
      zisa::shape_t<1>(n_coupled * coupled_order), zisa::device_type::cpu);
  reader.write_variable_to_array("function_scalings",
                                 function_scalings.view().raw());
  CoupledFunction2<Scalar> func_coupled(function_scalings.const_view(),
                                        n_coupled, coupled_order);
  BoundaryCondition bc;
  int boundary_value = reader.get_boundary_value();
  if (boundary_value == 0) {
    bc = BoundaryCondition::Dirichlet;
  } else if (boundary_value == 1) {
    bc = BoundaryCondition::Neumann;
  } else if (boundary_value == 2) {
    bc = BoundaryCondition::Periodic;
  } else {
    std::cout << "boundary condition not in range! " << std::endl;
    exit(-1);
  }

  // 0->Heat, 1->Wave
  int pde_type = reader.get_equation_type();
  if (pde_type == 0) {

    PDEHeat<1, Scalar, CoupledFunction2<Scalar>> pde(
        reader.get_x_size(), reader.get_y_size(), zisa::device_type::cpu, bc,
        func_coupled, reader.get_x_length() / reader.get_x_size(),
        reader.get_x_length() / reader.get_x_size());
    calculate_and_save_snapshots<PDEHeat<1, Scalar, CoupledFunction2<Scalar>>, Scalar>(pde, reader);

  } else if (pde_type == 1) {

    PDEWave<1, Scalar, CoupledFunction2<Scalar>> pde(
        reader.get_x_size(), reader.get_y_size(), zisa::device_type::cpu, bc,
        func_coupled, reader.get_x_length() / reader.get_x_size(),
        reader.get_x_length() / reader.get_x_size());
    calculate_and_save_snapshots<PDEWave<1, Scalar, CoupledFunction2<Scalar>>, Scalar>(pde, reader);

  } else {
    std::cout << "pde type " << pde_type << " not implemented yet! "
              << std::endl;
    exit(-1);
  }
}

// TODO: check if all sizes work (+- 1 boundary or no boundary)
int main(int argc, char **argv) {
  std::string filename;
  std::cout << "input filename to read: ";
  std::cin >> filename;
  NetCDFPDEReader reader(filename);
  int scalar_type = reader.get_scalar_type();
  if (scalar_type == 0) {
    run_simulation<float>(reader);
  } else if (scalar_type == 1) {
    run_simulation<double>(reader);
  } else {
    std::cout << "Only double or float (scalar_type {0,1}) allowed"
              << std::endl;
    return -1;
  }

  return 0;
}
