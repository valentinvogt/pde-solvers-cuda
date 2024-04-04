#ifndef HELPERS_MAIN_HPP_
#define HELPERS_MAIN_HPP_

#include "zisa/io/hdf5_serial_writer.hpp"
#include "zisa/io/hierarchical_file.hpp"
#include "zisa/io/hierarchical_writer.hpp"
#include <filesystem>
#include <zisa/memory/array.hpp>

inline void add_bc_values_file(zisa::HierarchicalWriter &writer) {
  zisa::array<float, 2> data(zisa::shape_t<2>(10, 10));
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      data(i, j) = 1.;
    }
  }
  zisa::save(writer, data, "bc");
}

inline void add_medium_bc_values_file(zisa::HierarchicalWriter &writer) {
  zisa::array<float, 2> data(zisa::shape_t<2>(102, 102));
  for (int i = 0; i < 102; i++) {
    for (int j = 0; j < 102; j++) {
      data(i, j) = 1.;
    }
  }
  zisa::save(writer, data, "bc");
}

inline void add_initial_data_file(zisa::HierarchicalWriter &writer) {
  zisa::array<float, 2> data(zisa::shape_t<2>(10, 10));
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      data(i, j) = i * j + j;
    }
  }
  zisa::save(writer, data, "initial_data");
}

inline void add_medium_initial_data_file(zisa::HierarchicalWriter &writer) {
  zisa::array<float, 2> data(zisa::shape_t<2>(102, 102));
  for (int i = 0; i < 102; i++) {
    for (int j = 0; j < 102; j++) {
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
inline void add_sigma_file(zisa::HierarchicalWriter &writer) {
  zisa::array<float, 2> data(zisa::shape_t<2>(17, 9));
  for (int i = 0; i < 17; i++) {
    for (int j = 0; j < 9; j++) {
      data(i, j) = 1.;
    }
  }
  zisa::save(writer, data, "sigma");
}

inline void add_medium_sigma_file(zisa::HierarchicalWriter &writer) {
  zisa::array<float, 2> data(zisa::shape_t<2>(197, 99));
  for (int i = 0; i < 197; i++) {
    for (int j = 0; j < 99; j++) {
      data(i, j) = 1.;
    }
  }
  zisa::save(writer, data, "sigma");
}

inline void add_simple_nc_file() {
  // check if it has already been created
  if (std::filesystem::exists("data/simple_data.nc"))
    return;
  // TODO:
  zisa::HDF5SerialWriter serial_writer("data/simple_data.nc");
  add_initial_data_file(serial_writer);
  add_bc_values_file(serial_writer);
  add_sigma_file(serial_writer);
}

inline void add_medium_nc_file() {
  // check if it has already been created
  if (std::filesystem::exists("data/data_100_100.nc"))
    return;
  // TODO:
  zisa::HDF5SerialWriter serial_writer("data/data_100_100.nc");
  add_medium_initial_data_file(serial_writer);
  add_medium_bc_values_file(serial_writer);
  add_medium_sigma_file(serial_writer);
}

#endif // HELPERS_MAIN_HPP_
