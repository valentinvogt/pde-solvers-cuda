#ifndef PDE_BASE_HPP_
#define PDE_BASE_HPP_

#include <convolution.hpp>
#include <convolve_sigma_add_f.hpp>
#include <helpers.hpp>
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

template <typename Scalar, typename BoundaryCondition> class PDEBase {
public:
  // note here that Nx and Ny denote the size INSIDE the boundary WITHOUT the
  // boundary
  PDEBase(unsigned Nx, unsigned Ny, const zisa::device_type memory_location,
          BoundaryCondition bc)
      : data_(zisa::shape_t<2>(Nx + 2, Ny + 2), memory_location),
        bc_neumann_values_(zisa::shape_t<2>(Nx + 2, Ny + 2), memory_location),
        sigma_values_(zisa::shape_t<2>(2 * Nx + 1, Ny + 1), memory_location),
        memory_location_(memory_location), bc_(bc) {}

  virtual void read_values(const std::string &filename,
                           const std::string &tag_data = "initial_data",
                           const std::string &tag_sigma = "sigma",
                           const std::string &tag_bc = "bc") = 0;

  virtual void apply(Scalar dt) = 0;

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
    print_matrix(bc_neumann_values_.const_view());
    std::cout << "sigma values:" << std::endl;
    print_matrix(sigma_values_.const_view());
  }

protected:
  void add_bc() {
    if (bc_ == BoundaryCondition::Dirichlet) {
      // do nothing as long as data on boundary does not change
      // dirichlet_bc(data_.view(), bc_neumann_values_.const_view());
    } else if (bc_ == BoundaryCondition::Neumann) {
      // TODO: change dt
      Scalar dt = 0.1;
      neumann_bc(data_.view(), bc_neumann_values_.const_view(), dt);
    } else if (bc_ == BoundaryCondition::Periodic) {
      periodic_bc(data_.view());
    } else {
      std::cout << "boundary condition not implemented yet!" << std::endl;
    }
  }

  zisa::array<Scalar, 2> data_;
  zisa::array<Scalar, 2> bc_neumann_values_;
  zisa::array<Scalar, 2> sigma_values_;

  const BoundaryCondition bc_;
  const zisa::device_type memory_location_;
  bool ready_ = false;
};

#endif // PDE_BASE_HPP_
