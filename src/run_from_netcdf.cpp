#include "io/netcdf_reader.hpp"
#include "io/netcdf_writer2.hpp"
#include "pde_base.hpp"
#include <coupled_function_2.hpp>
#include <iostream>
#include <netcdf.h>
#include <pde_heat.hpp>
#include <pde_wave.hpp>
#include <string>
#include <zisa/memory/array.hpp>

#define INSTANCIATE_PDE_AND_CALCULATE(PDE_TYPE, N_COUPLED)                     \
  case N_COUPLED:                                                              \
    calculate_and_save_snapshots<                                              \
        PDE_TYPE<N_COUPLED, Scalar, CoupledFunction2<Scalar>>, Scalar>(        \
        std::move(PDE_TYPE<N_COUPLED, Scalar, CoupledFunction2<Scalar>>(       \
            reader.get_x_size(), reader.get_y_size(), zisa::device_type::cpu,  \
            bc, func_coupled, reader.get_x_length() / reader.get_x_size(),     \
            reader.get_x_length() / reader.get_x_size())),                     \
        reader);                                                               \
    break;

// TODO:
// read initial data, apply steps and save them to a output file
template <typename PDE, typename Scalar>
inline void calculate_and_save_snapshots(PDE pde,
                                         const NetCDFPDEReader &reader) {

  reader.get_number_snapshots();
  reader.get_file_to_save_output();

  NetCDFPDEWriter2<Scalar> writer(
      reader.get_number_snapshots(), reader.get_final_time(),
      reader.get_n_members(), reader.get_x_size(),
      reader.get_x_length(), reader.get_y_size(),
      reader.get_y_length(),
      reader.get_equation_type(), reader.get_boundary_value(),
      reader.get_n_coupled(), reader.get_coupled_function_order(),
      reader.get_number_snapshots(), reader.get_file_to_save_output());
  for (int memb = 0; memb < reader.get_n_members(); memb++) {
    pde.read_initial_data_from_netcdf(reader, memb);
    pde.apply_with_snapshots(reader.get_final_time(),
                             reader.get_number_timesteps(),
                             reader.get_number_snapshots(), writer, memb);
  }
}

template <typename Scalar> void run_simulation(const NetCDFPDEReader &reader) {
  int n_coupled = reader.get_n_coupled();
  int coupled_order = reader.get_coupled_function_order();
  zisa::array<Scalar, 1> function_scalings(
      zisa::shape_t<1>(n_coupled * coupled_order), zisa::device_type::cpu);
  reader.write_whole_variable_to_array("function_scalings",
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
    switch (reader.get_n_coupled()) {
      INSTANCIATE_PDE_AND_CALCULATE(PDEHeat, 1)
      INSTANCIATE_PDE_AND_CALCULATE(PDEHeat, 2)
      INSTANCIATE_PDE_AND_CALCULATE(PDEHeat, 3)
      INSTANCIATE_PDE_AND_CALCULATE(PDEHeat, 4)
      INSTANCIATE_PDE_AND_CALCULATE(PDEHeat, 5)
    default:
      std::cout << "only implemented for n_coupled <= 5 yet" << std::endl;
      exit(-1);
    }
  } else if (pde_type == 1) {
    switch (reader.get_n_coupled()) {
      INSTANCIATE_PDE_AND_CALCULATE(PDEWave, 1)
      INSTANCIATE_PDE_AND_CALCULATE(PDEWave, 2)
      INSTANCIATE_PDE_AND_CALCULATE(PDEWave, 3)
      INSTANCIATE_PDE_AND_CALCULATE(PDEWave, 4)
      INSTANCIATE_PDE_AND_CALCULATE(PDEWave, 5)
    default:
      std::cout << "only implemented for n_coupled <= 5 yet" << std::endl;
      exit(-1);
    }
  } else {
    std::cout << "pde type " << pde_type << " not implemented yet! "
              << std::endl;
    exit(-1);
  }
}

// TODO: check if all sizes work (+- 1 boundary or no boundary)
int main(int argc, char **argv) {
  std::string filename;
  if (argc == 1) {
    std::cout << "input filename to read: ";
    std::cin >> filename;
  } else {
    filename = argv[1];
  }
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

