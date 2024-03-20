#ifndef PDE_BASE_HPP_
#define PDE_BASE_HPP_

#include "zisa/io/file_manipulation.hpp"
#include "zisa/io/hierarchical_reader.hpp"
#include "zisa/memory/array_traits.hpp"
#include "zisa/memory/memory_location.hpp"
#include "zisa/memory/shape.hpp"
#include <convolution.hpp>
#include <dirichlet_bc.hpp>
#include <neumann_bc.hpp>
#include <periodic_bc.hpp>
#include <zisa/io/hdf5_serial_writer.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/device_type.hpp>

template <typename Scalar, typename BoundaryCondition> class PDEBase {
public:
  using scalar_t = Scalar;

  PDEBase(unsigned Nx, unsigned Ny,
          const zisa::array_const_view<Scalar, 2> &kernel, BoundaryCondition bc)
      : data_(zisa::shape_t<2>(Nx + 2 * (kernel.shape(0) / 2),
                               Ny + 2 * (kernel.shape(1) / 2)),
              kernel.memory_location()),
        bc_values_(zisa::shape_t<2>(Nx + 2 * (kernel.shape(0) / 2),
                                    Ny + 2 * (kernel.shape(1) / 2)),
                   kernel.memory_location()),
        sigma_values_vertical_(zisa::shape_t<2>(Nx + 1, Ny),
                               kernel.memory_location()),
        sigma_values_horizontal_(zisa::shape_t<2>(Nx, Ny + 1),
                                 kernel.memory_location()),
        kernel_(kernel), bc_(bc) {}

  void read_values(const std::string &filename,
                   const std::string &tag_data = "initial_data",
                   const std::string &tag_sigma = "sigma",
                   const std::string &tag_bc = "bc") {
    zisa::HDF5SerialReader reader(filename);
    read_data(reader, data_, tag_data);
    zisa::array<Scalar, 2> sigma_tmp(data_.shape(), kernel_.memory_location());
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

  void apply() {
    zisa::array<scalar_t, 2> tmp(data_.shape(), data_.device());
    convolve(tmp.view(), data_.const_view(), this->kernel_);
    if (bc_ == BoundaryCondition::Neumann) {
      // make shure that boundary values stay constant to later apply boundary
      // conditions (they where not copied in convolve)
      dirichlet_bc(tmp.view(), data_.const_view(), num_ghost_cells_x(),
                   num_ghost_cells_y(), kernel_.memory_location());
    }
    zisa::copy(data_, tmp);
    add_bc();
  }

  unsigned num_ghost_cells(unsigned dir) { return kernel_.shape(dir) / 2; }
  unsigned num_ghost_cells_x() { return num_ghost_cells(0); }
  unsigned num_ghost_cells_y() { return num_ghost_cells(1); }

  // for testing/debugging
  void print() {
    std::cout << "data has size x: " << data_.shape(0)
              << ", y: " << data_.shape(1) << std::endl;
    std::cout << "border sizes are x: " << num_ghost_cells_x()
              << ", y: " << num_ghost_cells_y() << std::endl;

    std::cout << "data:" << std::endl;
    // print_matrix(data_);
    std::cout << "bc values:" << std::endl;
    // print_matrix(bc_values_);
    std::cout << "sigma values vertical:" << std::endl;
    // print_matrix(sigma_values_vertical_);
    std::cout << "sigma values horizontal:" << std::endl;
    // print_matrix(sigma_values_horizontal_);
  }

protected:
  void add_bc() {
    if (bc_ == BoundaryCondition::Dirichlet) {
      dirichlet_bc<Scalar>(data_.view(), bc_values_.const_view(),
                           num_ghost_cells_x(), num_ghost_cells_y(),
                           kernel_.memory_location());
    } else if (bc_ == BoundaryCondition::Neumann) {
      // TODO: change dt
      neumann_bc(data_.view(), bc_values_.const_view(), num_ghost_cells_x(),
                 num_ghost_cells_y(), kernel_.memory_location(), 0.1);
    } else if (bc_ == BoundaryCondition::Periodic) {
      periodic_bc(data_.view(), num_ghost_cells_x(), num_ghost_cells_y(),
                  kernel_.memory_location());
    } else {
      std::cout << "boundary condition not implemented yet!" << std::endl;
    }
  }

  inline void read_data(zisa::HierarchicalReader &reader,
                        zisa::array<Scalar, 2> &data, const std::string &tag) {
#if CUDA_AVAILABLE
    zisa::array<float, 2> cpu_data(data.shape());
    zisa::load_impl<Scalar, 2>(reader, cpu_data, tag,
                               zisa::default_dispatch_tag{});
    zisa::copy(data, cpu_data);
#else
    zisa::load_impl(reader, data, tag, zisa::bool_dispatch_tag{});
#endif // CUDA_AVAILABLE
  }

  inline void print_matrix(const zisa::array_const_view<Scalar, 2> &array) {
#if CUDA_AVAILABLE
    zisa::array<float, 2> cpu_data(array.shape());
    zisa::copy(cpu_data, array);
    for (int i = 0; i < array.shape(0); i++) {
      for (int j = 0; j < array.shape(1); j++) {
        std::cout << cpu_data(i, j) << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
#else
    for (int i = 0; i < array.shape(0); i++) {
      for (int j = 0; j < array.shape(1); j++) {
        std::cout << array(i, j) << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
#endif
  }

  void construct_sigmas(zisa::array<Scalar, 2> &sigma_tmp) {
    // TODO: optimize
#if CUDA_AVAILABLE
    zisa::array<Scalar, 2> vertical_tmp(sigma_values_vertical_.shape());
    zisa::array<Scalar, 2> horizontal_tmp(sigma_values_horizontal_.shape());
    for (int x_idx = 0; x_idx < sigma_tmp.shape(0) - 1; x_idx++) {
      for (int y_idx = 0; y_idx < sigma_tmp.shape(1) - 1; y_idx++) {
        if (y_idx < sigma_tmp.shape(1) - 2) {
          vertical_tmp(x_idx, y_idx) =
              (sigma_tmp(x_idx, y_idx + 1) + sigma_tmp(x_idx + 1, y_idx + 1)) *
              .5;
        }
        if (x_idx < sigma_tmp.shape(0) - 2) {
          vertical_tmp(x_idx, y_idx) =
              (sigma_tmp(x_idx + 1, y_idx) + sigma_tmp(x_idx + 1, y_idx)) * .5;
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
              (sigma_tmp(x_idx + 1, y_idx) + sigma_tmp(x_idx + 1, y_idx + 1)) * .5;
        }
      }
    }

#endif
  }

  zisa::array<Scalar, 2> data_;
  const zisa::array_const_view<Scalar, 2> kernel_;
  const BoundaryCondition bc_;
  zisa::array<Scalar, 2> bc_values_;
  zisa::array<Scalar, 2> sigma_values_vertical_;
  zisa::array<Scalar, 2> sigma_values_horizontal_;
  bool ready_ = false;
};

#endif // PDE_BASE_HPP_
