#ifndef NC_WRITER_HPP_
#define NC_WRITER_HPP_

#include <netcdf.h>
#include <string>
#include <iostream>


template <typename Scalar>
int get_array_from_nc_file(std::string filename, int data_length, Scalar *data) {
  int ncid = -1;
  auto error_message = nc_open(filename.c_str(), NC_NOWRITE, &ncid);
  if (error_message != NC_NOERR) {
    std::cout << "error in opening file" << std::endl;
    std::cout << nc_strerror(error_message) << std::endl;
    return -1;
  }

  int varid = -1;
  error_message = nc_inq_varid(ncid, "var", &varid);
  if (error_message != NC_NOERR) {
    std::cout << "error in getting variable id" << std::endl;
    std::cout << nc_strerror(error_message) << std::endl;
    nc_close(ncid);
    return -1;
  }

  error_message = nc_get_var(ncid, varid, data);
  if (error_message != NC_NOERR) {
    std::cout << "error in getting data" << std::endl;
    std::cout << nc_strerror(error_message) << std::endl;
    nc_close(ncid);
    return -1;
  }
  
  error_message = nc_close(ncid);
  if (error_message != NC_NOERR) {
    std::cout << "error in closing file" << std::endl;
    std::cout << nc_strerror(error_message) << std::endl;
    return -1;
  }

  
  return 0;
}


template <typename Scalar>
int store_array_to_nc_file(std::string filename, int data_length, Scalar *data) {
  
  int ncid = -1;
  auto error_message = nc_create(filename.c_str(), NC_CLOBBER, &ncid);
  if (error_message != NC_NOERR) {
    std::cout << "error in creating file" << std::endl;
    std::cout << nc_strerror(error_message) << std::endl;
    return -1;
  }

  int dimid = -1;
  error_message = nc_def_dim(ncid, "dim", data_length, &dimid);
  if (error_message != NC_NOERR) {
    std::cout << "error in defining dimensions" << std::endl;
    std::cout << nc_strerror(error_message) << std::endl;
    return -1;
  }

  int varid = -1;
  error_message = nc_def_var(ncid, "var", NC_FLOAT, 1, &dimid, &varid);
  if (error_message != NC_NOERR) {
    std::cout << "error in defining variables" << std::endl;
    std::cout << nc_strerror(error_message) << std::endl;
    return -1;
  }

  error_message = nc_enddef(ncid);
  if (error_message != NC_NOERR) {
    std::cout << "error in enddef" << std::endl;
    std::cout << nc_strerror(error_message) << std::endl;
    return -1;
  }

  error_message = nc_put_var(ncid, varid, data);
  if (error_message != NC_NOERR) {
    std::cout << "error in putting variables to file" << std::endl;
    std::cout << nc_strerror(error_message) << std::endl;
    return -1;
  }

  error_message = nc_close(ncid);
  if (error_message != NC_NOERR) {
    std::cout << "error in closing file" << std::endl;
    std::cout << nc_strerror(error_message) << std::endl;
    return -1;
  }
  return 0;
}

#endif //NC_WRITER_HPP_

