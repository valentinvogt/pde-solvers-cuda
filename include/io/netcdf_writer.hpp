#ifndef NETCDF_WRITER_HPP_
#define NETCDF_WRITER_HPP_

#include "zisa/io/netcdf_serial_writer.hpp"
#include "zisa/memory/array_view_decl.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <zisa/io/hierarchical_file.hpp>
#include <zisa/memory/array.hpp>

// change this one later
inline zisa::NetCDFSerialWriter DummyWriter() {
  std::vector<std::tuple<std::string, std::size_t>> dims;
  std::vector<
      std::tuple<std::string, std::vector<std::string>, zisa::ErasedDataType>>
      vars;
  zisa::NetCDFFileStructure file_structure(dims, vars);

  return zisa::NetCDFSerialWriter("dummy", file_structure);
}

template <typename Scalar> class NetCDFPDEWriter {
public:
  // initializes reader and alredy saves member, time, n_x and n_y values
  NetCDFPDEWriter(unsigned int n_snapshots, Scalar T, unsigned int n_members,
                  unsigned int n_x, Scalar x_begin, Scalar x_end,
                  unsigned int n_y, Scalar y_begin, Scalar y_end,
                  const std::string &filename)
      : /* used for compilation (no default constructor), should be changed
           later*/
        writer_(DummyWriter()) {

    std::vector<std::tuple<std::string, std::size_t>> dims = {
        {"member", n_members}, {"time", n_snapshots}, {"x", n_x}, {"y", n_y}};

    std::vector<
        std::tuple<std::string, std::vector<std::string>, zisa::ErasedDataType>>
        vars;
    std::vector<std::string> member_var{"member"};
    vars.emplace_back("member", member_var, zisa::erase_data_type<int>());

    std::vector<std::string> time_var{"time"};
    vars.emplace_back("time", time_var, zisa::erase_data_type<Scalar>());

    std::vector<std::string> x_pos_var{"x"};
    vars.emplace_back("x_pos", x_pos_var, zisa::erase_data_type<Scalar>());

    std::vector<std::string> y_pos_var{"y"};
    vars.emplace_back("y_pos", y_pos_var, zisa::erase_data_type<Scalar>());

    // create a variable for every snapshot
    // for example, snapshot of member 2 at 3'rd timestep is called m_2_t_3
    for (int member = 0; member < n_members; member++) {
      for (int snapshot = 0; snapshot < n_snapshots; snapshot++) {
        std::string append =
            "m_" + std::to_string(member) + "_s_" + std::to_string(snapshot);
        std::vector<std::string> function_value_var{append};
        vars.emplace_back(append, function_value_var,
                          zisa::erase_data_type<Scalar>());
        dims.emplace_back(append, n_x * n_y);
      }
    }

    zisa::NetCDFFileStructure file_structure(dims, vars);
    writer_ = zisa::NetCDFSerialWriter(filename, file_structure);
    std::cout << "writer created" << std::endl;

    // already save member, time and position arrays
    zisa::array<int, 1> members((zisa::shape_t<1>(n_members)));
    for (int i = 0; i < n_members; i++) {
      members(i) = i;
    }
    zisa::save(writer_, members.const_view(), "member");

    // assume that snapshots are distributed equally, starting at 0 to T
    // there are n_snapshots snapshots, including 0 and T
    // maybe change this to accept a vector with values
    zisa::array<Scalar, 1> time((zisa::shape_t<1>(n_snapshots)));
    for (int i = 0; i < n_snapshots; i++) {
      time(i) = i * T / (Scalar)(n_snapshots - 1);
    }
    zisa::save(writer_, time.const_view(), "time");

    zisa::array<Scalar, 1> x_values((zisa::shape_t<1>(n_x)));
    for (int i = 0; i < n_x; i++) {
      x_values(i) = x_begin + ((x_end - x_begin) * i) / (n_x - 1);
    }
    zisa::save(writer_, x_values.const_view(), "x_pos");

    zisa::array<Scalar, 1> y_values((zisa::shape_t<1>(n_y)));
    for (int i = 0; i < n_y; i++) {
      y_values(i) = y_begin + ((y_end - y_begin) * i) / (n_y - 1);
    }
    zisa::save(writer_, y_values.const_view(), "y_pos");
  }

  void save_snapshot(int member, int snapshot_number,
                     zisa::array_const_view<Scalar, 2> data) {
    std::string snapshot_string =
        "m_" + std::to_string(member) + "_s_" + std::to_string(snapshot_number);
    zisa::save(writer_, data, snapshot_string);
  }

private:
  zisa::NetCDFSerialWriter writer_;
};

#endif // NETCDF_WRITER_HPP_
