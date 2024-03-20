#ifndef PDE_BASE_HPP_
#define PDE_BASE_HPP_

#include <convolution.hpp>
#include <dirichlet_bc.hpp>
#include <neumann_bc.hpp>
#include <periodic_bc.hpp>
#include <zisa/io/file_manipulation.hpp>
#include <zisa/io/hdf5_serial_writer.hpp>
#include <zisa/io/hierarchical_reader.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/array_traits.hpp>
#include <zisa/memory/device_type.hpp>
#include <zisa/memory/memory_location.hpp>
#include <zisa/memory/shape.hpp>
#include <convolve_sigma_add_f.hpp>
#include <helpers.hpp>

template <typename Scalar, typename BoundaryCondition> class PDEBase {
public:

  PDEBase(unsigned Nx, unsigned Ny,
          const zisa::device_type memory_location, BoundaryCondition bc)
      : data_(zisa::shape_t<2>(Nx + 2, Ny + 2), memory_location),
        bc_values_(zisa::shape_t<2>(Nx + 2, Ny + 2), memory_location),
        sigma_values_vertical_(zisa::shape_t<2>(Nx + 1, Ny), memory_location),
        sigma_values_horizontal_(zisa::shape_t<2>(Nx, Ny + 1), memory_location),
        memory_location_(memory_location),
        bc_(bc) {}

  void read_values(const std::string &filename,
                   const std::string &tag_data = "initial_data",
                   const std::string &tag_sigma = "sigma",
                   const std::string &tag_bc = "bc") {
    zisa::HDF5SerialReader reader(filename);
    read_data(reader, data_, tag_data);
    zisa::array<Scalar, 2> sigma_tmp(data_.shape());
    read_data(reader, sigma_tmp, tag_sigma);
    construct_sigmas(sigma_tmp);

    if (bc_ == BoundaryCondition::Neumann) {
      read_data(reader, bc_values_, tag_bc);
    } else if (bc_ == BoundaryCondition::Dirichlet) {
      zisa::copy(bc_values_, data_);
    } else if (bc_ == BoundaryCondition::Periodic) {
      add_bc();
    }
    ready_ = true;
    std::cout << "initial data, sigma and boundary conditions read!"
              << std::endl;
  }

  virtual void apply() = 0;

  // remove those later
  unsigned num_ghost_cells(unsigned dir) { return 1; }
  unsigned num_ghost_cells_x() { return 1; }
  unsigned num_ghost_cells_y() { return 1; }

  // for testing/debugging
  void print() {
    std::cout << "data has size x: " << data_.shape(0)
              << ", y: " << data_.shape(1) << std::endl;

    std::cout << "data:" << std::endl;
    print_matrix(data_.const_view());
    std::cout << "bc values:" << std::endl;
    print_matrix(bc_values_.const_view());
    std::cout << "sigma values vertical:" << std::endl;
    print_matrix(sigma_values_vertical_.const_view());
    std::cout << "sigma values horizontal:" << std::endl;
    print_matrix(sigma_values_horizontal_.const_view());
  }

protected:
  void add_bc() {
    if (bc_ == BoundaryCondition::Dirichlet) {
      dirichlet_bc<Scalar>(data_.view(), bc_values_.const_view());
    } else if (bc_ == BoundaryCondition::Neumann) {
      // TODO: change dt
      Scalar dt = 0.1;
      neumann_bc(data_.view(), bc_values_.const_view(), dt);
    } else if (bc_ == BoundaryCondition::Periodic) {
      periodic_bc(data_.view());
    } else {
      std::cout << "boundary condition not implemented yet!" << std::endl;
    }
  }


  void construct_sigmas(zisa::array<Scalar, 2> &sigma_tmp) {
    // TODO: optimize
#if CUDA_AVAILABLE
    zisa::array<Scalar, 2> vertical_tmp(sigma_values_vertical_.shape());
    zisa::array<Scalar, 2> horizontal_tmp(sigma_values_horizontal_.shape());
    zisa::array<Scalar, 2> tmp(data_.shape(), data_.device());
    for (int x_idx = 0; x_idx < sigma_tmp.shape(0) - 1; x_idx++) {
      for (int y_idx = 0; y_idx < sigma_tmp.shape(1) - 1; y_idx++) {
        if (y_idx < sigma_tmp.shape(1) - 2) {
          vertical_tmp(x_idx, y_idx) =
              (sigma_tmp(x_idx, y_idx + 1) + sigma_tmp(x_idx + 1, y_idx + 1)) *
              .5;
        }
        if (x_idx < sigma_tmp.shape(0) - 2) {
          horizontal_tmp(x_idx, y_idx) =
              (sigma_tmp(x_idx + 1, y_idx) + sigma_tmp(x_idx + 1, y_idx + 1)) *
              .5;
        }
      }
    }
    zisa::copy(sigma_values_vertical_, vertical_tmp);
    zisa::copy(sigma_values_horizontal_, horizontal_tmp);
#else
    for (int x_idx = 0; x_idx < sigma_tmp.shape(0) - 1; x_idx++) {
      for (int y_idx = 0; y_idx < sigma_tmp.shape(1) - 1; y_idx++) {
        if (y_idx < sigma_tmp.shape(1) - 2) {
          sigma_values_vertical_(x_idx, y_idx) =
              (sigma_tmp(x_idx, y_idx + 1) + sigma_tmp(x_idx + 1, y_idx + 1)) *
              .5;
        }
        if (x_idx < sigma_tmp.shape(0) - 2) {
          sigma_values_horizontal_(x_idx, y_idx) =
              (sigma_tmp(x_idx + 1, y_idx) + sigma_tmp(x_idx + 1, y_idx + 1)) *
              .5;
        }
      }
    }
#endif // CUDA_AVAILABLE
  }

  zisa::array<Scalar, 2> data_;
  zisa::array<Scalar, 2> bc_values_;
  zisa::array<Scalar, 2> sigma_values_vertical_;
  zisa::array<Scalar, 2> sigma_values_horizontal_;

  const BoundaryCondition bc_;
  const zisa::device_type memory_location_;
  bool ready_ = false;
};

#endif // PDE_BASE_HPP_
