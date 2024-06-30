#ifndef NETCDF_WRITER2_HPP_
#define NETCDF_WRITER2_HPP_

#include "netcdf_reader.hpp"
#include <chrono>
#include <iostream>
#include <netcdf.h>
#include <string>
#include <zisa/memory/array.hpp>
#include <zisa/memory/array_view_decl.hpp>

#define DURATION(a)                                                            \
  std::chrono::duration_cast<std::chrono::milliseconds>(a).count()
#define NOW std::chrono::high_resolution_clock::now()

template <typename Scalar> class NetCDFPDEWriter {
public:
  // initializes reader and alredy saves member, time, n_x and n_y values
  // make shure the folder of filename exists, for example if the file
  // has to be savet at out/result.nc, make sure that the out folder is created
  NetCDFPDEWriter(int n_snapshots, Scalar T, int n_members, int n_x,
                  Scalar x_length, int n_y, Scalar y_length,
                  int type_of_equation, int boundary_value_type, int n_coupled,
                  int coupled_function_order, int number_snapshots,
                  char *filename)
      : n_snapshots_(n_snapshots), final_time_(T), n_members_(n_members),
        n_x_(n_x), x_length_(x_length), n_y_(n_y), y_length_(y_length),
        n_coupled_(n_coupled), filename_(filename) {

    const int scalar_type = std::is_same<Scalar, double>::value ? 1 : 0;

    check(nc_create(filename, NC_NETCDF4 | NC_CLOBBER, &ncid_));

    check(nc_put_att_int(ncid_, NC_GLOBAL, "type_of_equation", NC_INT, 1,
                         &type_of_equation));
    check(nc_put_att_int(ncid_, NC_GLOBAL, "n_members", NC_INT, 1, &n_members));
    check(nc_put_att_int(ncid_, NC_GLOBAL, "n_x", NC_INT, 1, &n_x));
    check(nc_put_att_int(ncid_, NC_GLOBAL, "n_y", NC_INT, 1, &n_y));

    if constexpr (scalar_type == 0) {
      check(nc_put_att_float(ncid_, NC_GLOBAL, "x_length", NC_FLOAT, 1,
                             &x_length));
      check(nc_put_att_float(ncid_, NC_GLOBAL, "y_length", NC_FLOAT, 1,
                             &y_length));
      check(nc_put_att_float(ncid_, NC_GLOBAL, "final_time", NC_FLOAT, 1, &T));
    } else if constexpr (scalar_type == 1) {
      check(nc_put_att_double(ncid_, NC_GLOBAL, "x_length", NC_DOUBLE, 1,
                              &x_length));
      check(nc_put_att_double(ncid_, NC_GLOBAL, "y_length", NC_DOUBLE, 1,
                              &y_length));
      check(
          nc_put_att_double(ncid_, NC_GLOBAL, "final_time", NC_DOUBLE, 1, &T));
    } else {
      std::cout << "only float and double implemented in netcdf_writer"
                << std::endl;
      exit(-1);
    }

    check(nc_put_att_int(ncid_, NC_GLOBAL, "boundary_value_type", NC_INT, 1,
                         &boundary_value_type));
    check(nc_put_att_int(ncid_, NC_GLOBAL, "scalar_type", NC_INT, 1,
                         &scalar_type));

    check(nc_put_att_int(ncid_, NC_GLOBAL, "n_coupled", NC_INT, 1, &n_coupled));
    check(nc_put_att_int(ncid_, NC_GLOBAL, "coupled_function_order", NC_INT, 1,
                         &coupled_function_order));
    check(nc_put_att_int(ncid_, NC_GLOBAL, "number_snapshots", NC_INT, 1,
                         &number_snapshots));

    // create dimiensions for variables
    int dims_data[4];
    check(nc_def_dim(ncid_, "n_members", n_members_, &dims_data[0]));
    check(nc_def_dim(ncid_, "n_snapshots", n_snapshots_, &dims_data[1]));
    check(nc_def_dim(ncid_, "x_size_and_boundary", n_x + 2, &dims_data[2]));
    check(nc_def_dim(ncid_, "n_coupled_and_x_size_and_boundary",
                     n_coupled * (n_y + 2), &dims_data[3]));

    size_t chunk_sizes[4] = {1, 1, (size_t)n_x + 2,
                             (size_t)n_coupled * (n_y + 2)};

    if constexpr (scalar_type == 0) {
      check(nc_def_var(ncid_, "data", NC_FLOAT, 4, dims_data, &varid_data_));
    } else {
      check(nc_def_var(ncid_, "data", NC_DOUBLE, 4, dims_data, &varid_data_));
    }
    check(nc_def_var_chunking(ncid_, varid_data_, NC_CHUNKED, chunk_sizes));

    // TODO: create variables for sigma_values and function_scalings
  }
  ~NetCDFPDEWriter() { nc_close(ncid_); }

  void save_snapshot(int member, int snapshot_number,
                     zisa::array_const_view<Scalar, 2> data) {
    // auto start = NOW;
    size_t offsets[4] = {(size_t)member, (size_t)snapshot_number, 0, 0};
    size_t counts[4] = {1, 1, (size_t)n_x_ + 2,
                        (size_t)n_coupled_ * (n_y_ + 2)};

    assert(n_x_ + 2 == data.shape(0));
    assert(n_coupled_ * (n_y_ + 2) == data.shape(1));
    if (data.memory_location() == zisa::device_type::cpu) {
      if constexpr (std::is_same<Scalar, float>::value) {
        nc_put_vara_float(ncid_, varid_data_, offsets, counts, &data[0]);
      } else {
        nc_put_vara_double(ncid_, varid_data_, offsets, counts, &data[0]);
      }
    }
#if CUDA_AVAILABLE
    else if (data.memory_location() == zisa::device_type::cuda) {
      zisa::array<Scalar, 2> tmp(
          zisa::shape_t<2>(data.shape()[0], data.shape()[1]),
          zisa::device_type::cpu);
      zisa::copy(tmp, data);

      if constexpr (std::is_same<Scalar, float>::value) {
        nc_put_vara_float(ncid_, varid_data_, offsets, counts,
                          &(tmp.const_view()[0]));
      } else {
        nc_put_vara_double(ncid_, varid_data_, offsets, counts,
                           &(tmp.const_view()[0]));
      }

    }
#endif
    else {
      std::cout << "error in writer, unknown memory location!\n";
      exit(-1);
    }

    // auto end = NOW;
    // std::cout << "time to save snapshot: " << DURATION(end - start) << " ms"
    // << std::endl;
  }

private:
  int ncid_;
  int n_snapshots_;
  Scalar final_time_;
  int n_members_;

  int n_x_;
  Scalar x_length_;
  int n_y_;
  Scalar y_length_;

  int varid_data_;

  int n_coupled_;

  std::string filename_;
};

#undef DURATION
#endif // NETCDF_WRITER2_HPP_
