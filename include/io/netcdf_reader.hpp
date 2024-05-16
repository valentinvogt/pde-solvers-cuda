#ifndef NETCDF_READER_HPP_
#define NETCDF_READER_HPP_

#include <iostream>
#include <netcdf.h>
#include <string>

class NetCDFPDEReader {
public:
  NetCDFPDEReader(std::string filename) : filename_(filename) {
    if (nc_open(filename.c_str(), NC_NOWRITE, &ncid_) != NC_NOERR) {
      std::cout << "error occured in opening netCDF file with path " << filename
                << std::endl;
      exit(-1);
    }

    if (nc_get_att(ncid_, NC_GLOBAL, "scalar_type", &scalar_type_) !=
        NC_NOERR) {
      std::cout << "error occured in detecting scalar type" << std::endl;
      exit(-1);
    }

    if (nc_get_att(ncid_, NC_GLOBAL, "x_size", &x_size_) != NC_NOERR) {
      std::cout << "error occured in reading x_size" << std::endl;
    }

    if (nc_get_att(ncid_, NC_GLOBAL, "y_size", &y_size_) != NC_NOERR) {
      std::cout << "error occured in reading y_size" << std::endl;
    }

    if (nc_get_att(ncid_, NC_GLOBAL, "x_length", &x_length_) != NC_NOERR) {
      std::cout << "error occured in reading x_length" << std::endl;
    }

    if (nc_get_att(ncid_, NC_GLOBAL, "y_length", &y_length_) != NC_NOERR) {
      std::cout << "error occured in reading y_length" << std::endl;
    }

    if (nc_get_att(ncid_, NC_GLOBAL, "boundary_value_type",
                   &boundary_value_type_) != NC_NOERR) {
      std::cout << "error occured in reading y_size" << std::endl;
    }

    if (nc_get_att(ncid_, NC_GLOBAL, "n_coupled", &n_coupled_) != NC_NOERR) {
      std::cout << "error occured in reading n_coupled" << std::endl;
    }

    if (nc_get_att(ncid_, NC_GLOBAL, "coupled_function_order",
                   &coupled_function_order_) != NC_NOERR) {
      std::cout << "error occured in reading coupled_function_order"
                << std::endl;
    }
  }
  NetCDFPDEReader() = delete;
  ~NetCDFPDEReader() { nc_close(ncid_); }

  // getters
  // 0->HeatEquation, 1->WaveEquation
  int get_equation_type() const { return scalar_type_; }
  int get_x_size() const { return x_size_; }
  int get_y_size() const { return y_size_; }
  double get_x_length() const { return x_length_; }
  double get_y_length() const { return y_length_; }
  // 0->Dirichlet, 1->Neumann, 2->Periodic
  int get_boundary_value() const { return boundary_value_type_; }
  // 0->float, 1->double
  int get_scalar_type() const { return scalar_type_; }
  int get_n_coupled() const { return n_coupled_; }
  int get_coupled_function_order() const { return coupled_function_order_; }

  int write_variable_to_array (std::string varname, void *arr_ptr) const {
    if (scalar_type_ == 0) {
      return write_variable_to_array_generic(varname, (float *)arr_ptr);
    } else if (scalar_type_ == 1) {
      return write_variable_to_array_generic(varname, (double *)arr_ptr);
    } else {
      std::cout
          << "error in write variable, scalar_type_ is not defined properly"
          << std::endl;
      return -1;
    }
  };

private:
  // TODO:
  template <typename Scalar>
  int write_variable_to_array_generic(std::string varname, Scalar *arr_ptr) const {

    int varid;
    if (nc_inq_varid(ncid_, varname.c_str(), &varid) != NC_NOERR) {
      std::cout << "error in getting varid of " << varname << std::endl;
      return -1;
    }
    // TODO: this could be faster, for example use nc_get_var_float
    if (nc_get_var(ncid_, varid, arr_ptr) != NC_NOERR) {
       std::cout << "error in getting data of " << varname << std::endl; 
      return -1;
    }
    return 0;
  }

  std::string filename_;
  int ncid_;
  int scalar_type_; // 0->float, 1->double
  int x_size_;
  int y_size_;
  double x_length_;
  double y_length_;
  int boundary_value_type_; // 0->Dirichlet, 1->Neumann, 2->PeriodicDirichlet,
                            // 1->Neumann, 2->Periodic
  int n_coupled_;
  int coupled_function_order_;
};

#endif // NETCDF_READER_HPP_
