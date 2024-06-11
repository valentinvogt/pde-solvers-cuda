#ifndef PDE_WAVE_HPP_
#define PDE_WAVE_HPP_

// TODO: add initial derivative data, correct apply function

#include "io/netcdf_reader.hpp"
#include "periodic_bc.hpp"
#include "zisa/io/hdf5_writer.hpp"
#include <pde_base.hpp>

template <int n_coupled, typename Scalar, typename Function>
class PDEWave : public virtual PDEBase<n_coupled, Scalar> {
public:
  // TODO: add derivative
  PDEWave(unsigned Nx, unsigned Ny, const zisa::device_type memory_location,
          BoundaryCondition bc, Function f, Scalar dx, Scalar dy)
      : PDEBase<n_coupled, Scalar>(Nx, Ny, memory_location, bc, dx, dy),
        func_(f),
        deriv_data_(zisa::shape_t<2>(Nx + 2, Ny + 2), memory_location) {}

  void apply(Scalar dt) override {
    if (!this->ready_) {
      std::cerr << "Wave solver is not ready yet! Read data first" << std::endl;
      return;
    }

    zisa::array<Scalar, 2> second_deriv(this->data_.shape(),
                                        this->data_.device());
    const Scalar del_x_2 = 1. / (this->dx_ * this->dy_);
    convolve_sigma_add_f<n_coupled>(
        second_deriv.view(), this->data_.const_view(),
        this->sigma_values_.const_view(), del_x_2, func_);

    // euler update of derivative
    add_arrays_interior<n_coupled>(this->deriv_data_.view(),
                                   second_deriv.const_view(), dt);

    // euler update of data
    add_arrays_interior<n_coupled>(this->data_.view(),
                                   this->deriv_data_.const_view(), dt);
    PDEBase<n_coupled, Scalar>::add_bc();
  }

  void read_values(const std::string &filename,
                   const std::string &tag_data = "initial_data",
                   const std::string &tag_sigma = "sigma",
                   const std::string &tag_initial_derivative = "init_deriv") {
    zisa::HDF5SerialReader reader(filename);

    read_data(reader, this->data_, tag_data);
    read_data(reader, this->sigma_values_, tag_sigma);
    read_data(reader, this->deriv_data_, tag_initial_derivative);

    if (this->bc_ == BoundaryCondition::Neumann) {
      zisa::copy(this->bc_neumann_values_, this->deriv_data_);
    } else if (this->bc_ == BoundaryCondition::Dirichlet) {
      // do noching as long as data on boundary does not change
    } else if (this->bc_ == BoundaryCondition::Periodic) {
      periodic_bc<n_coupled, Scalar>(this->data_.view());
    }
    this->ready_ = true;
  }

  void read_values(zisa::array_const_view<Scalar, 2> data,
                   zisa::array_const_view<Scalar, 2> sigma,
                   zisa::array_const_view<Scalar, 2> bc,
                   zisa::array_const_view<Scalar, 2> initial_derivative) {
    zisa::copy(this->data_, data);
    zisa::copy(this->sigma_values_, sigma);
    zisa::copy(this->deriv_data_, initial_derivative);
    if (this->bc_ == BoundaryCondition::Neumann) {
      zisa::copy(this->bc_neumann_values_, initial_derivative);
    } else if (this->bc_ == BoundaryCondition::Dirichlet) {
      // do noching as long as data on boundary does not change
    } else if (this->bc_ == BoundaryCondition::Periodic) {
      periodic_bc<n_coupled, Scalar>(this->data_.view());
    }
    this->ready_ = true;
  }

  void read_initial_data_from_netcdf(const NetCDFPDEReader &reader,
                                     int member) {
    if (reader.write_variable_of_member_to_array(
            "initial_data", this->data_.view().raw(), member,
            this->data_.shape()[0], this->data_.shape()[1]) != 0) {
      std::cout << "error occured in writing initial data from netcdf to array!"
                << std::endl;
      exit(-1);
    }
    if (reader.write_variable_of_member_to_array(
            "sigma_values", this->sigma_values_.view().raw(), member,
            this->sigma_values_.shape()[0],
            this->sigma_values_.shape()[1]) != 0) {
      std::cout << "error occured in writing sigma values from netcdf to array!"
                << std::endl;
      exit(-1);
    }
    if (reader.write_variable_of_member_to_array(
            "bc_neumann_values", this->bc_neumann_values_.view().raw(), member,
            this->bc_neumann_values_.shape()[0],
            this->bc_neumann_values_.shape()[1]) != 0) {
      std::cout << "error occured in writing deriv data from netcdf to array!"
                << std::endl;
      exit(-1);
    }
    if (this->bc_ == BoundaryCondition::Neumann) {
      if (reader.write_variable_of_member_to_array(
              "bc_neumann_values", this->bc_neumann_values_.view().raw(),
              member, this->bc_neumann_values_.shape()[0],
              this->bc_neumann_values_.shape()[1]) != 0) {
        std::cout << "error occured in writing bc neumann values from netcdf"
                     "to array!"
                  << std::endl;
        exit(-1);
      }
    } else if (this->bc_ == BoundaryCondition::Dirichlet) {
      // do noching as long as data on boundary does not change
    } else if (this->bc_ == BoundaryCondition::Periodic) {
      periodic_bc<n_coupled, Scalar>(this->data_.view());
    }
    this->ready_ = true;
  }
  void print_deriv() {
    std::cout << "deriv: " << std::endl;
    print_matrix(this->deriv_data_.const_view());
  }

protected:
  Function func_;
  // add
  zisa::array<Scalar, 2> deriv_data_;
};

#endif // PDE_WAVE_HPP_
