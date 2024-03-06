#ifndef PDE_BASE_HPP_
#define PDE_BASE_HPP_

#include <convolution.hpp>
#include <zisa/io/hdf5_serial_writer.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/device_type.hpp>

template <typename Scalar, typename BoundaryCondition>
class PDEBase {
public:
  using scalar_t = Scalar;

  PDEBase(unsigned Nx, unsigned Ny,
          const zisa::array_const_view<Scalar, 2> &kernel, BoundaryCondition bc)
      : data_(zisa::shape_t<2>(Nx + 2 * (kernel.shape(0) / 2), Ny + 2 * (kernel.shape(1) / 2)),
              kernel.memory_location()),
        kernel_(kernel), bc_(bc) { }

  // make shure that the file exists with the right group name and tag,
  // otherwise this will crash Additionally, the data has to be stored as a
  // 2-dimensional array with the right amount of entries
  void read_initial_data(const std::string &filename,
                         const std::string &group_name,
                         const std::string &tag) {
    unsigned x_disp = num_ghost_cells_x();
    unsigned y_disp = num_ghost_cells_y();

    unsigned Nx = data_.shape(0) - 2 * num_ghost_cells_x();
    unsigned Ny = data_.shape(1) - 2 * num_ghost_cells_y();

    // read data from file
    Scalar return_data[Nx][Ny];
    zisa::HDF5SerialReader serial_reader(filename);
    serial_reader.open_group(group_name);
    serial_reader.read_array(return_data, zisa::erase_data_type<Scalar>(), tag);

    // copy return_data to data_
    // TODO: Optimize
    for (int i = 0; i < Nx; i++) {
      for (int j = 0; j < Ny; j++) {
        data_(x_disp + i, y_disp + j) = return_data[i][j];
      }
    }
    add_bc();
  }

  void apply() {
    zisa::array<scalar_t, 2> tmp(data_.shape(), data_.device());
    convolve(tmp.view(), data_.const_view(), this->kernel_);
    zisa::copy(data_, tmp);
    add_bc();
  }

  unsigned num_ghost_cells(unsigned dir) { return kernel_.shape(dir) / 2; }
  unsigned num_ghost_cells_x() { return num_ghost_cells(0); }
  unsigned num_ghost_cells_y() { return num_ghost_cells(1); }

  // for testing/debugging
  void print() {
    int x_size = data_.shape(0);
    int y_size = data_.shape(1);
    std::cout << "data has size x: " << x_size << ", y: " << y_size
              << std::endl;
    std::cout << " border sizes are x: " << num_ghost_cells_x()
              << ", y: " << num_ghost_cells_y() << std::endl;
    for (int i = 0; i < x_size; i++) {
      for (int j = 0; j < y_size; j++) {
        std::cout << data_(i, j) << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

protected:

  void add_bc() {
    if (bc_ == BoundaryCondition::Dirichlet) {
      add_dirichlet_bc();
    } else if (bc_ == BoundaryCondition::Neumann) {
      add_neumann_bc();
      // TODO: add boundary conditions
    } else if (bc_ == BoundaryCondition::Periodic) {
      // TODO: add boundary conditions
    } else {
      std::cout << "boundary condition not implemented yet!" << std::endl;
    }
  }

  // adds dirichlet boundary conditions.
  // Note that this only has to be done once at the beginning,
  // the boundary data will not change during the algorithm
  void add_dirichlet_bc() {
    // assumption: f(x) = 0 at boundary
    Scalar value = 0;
    // TODO: add other values, optimize
    int x_length = data_.shape(0);
    int y_length = data_.shape(1);

    int x_disp = num_ghost_cells_x();
    int y_disp = num_ghost_cells_y();

    // add boundary condition on left and right boundary
    for (int x_idx = 0; x_idx < x_length; x_idx++) {
      for (int y_idx = 0; y_idx < y_disp; y_idx++) {
        data_(x_idx, y_idx) = value;
        data_(x_idx, y_length - 1 - y_idx) = value;
      }
    }

    // add boundary on top and botton without corners
    for (int x_idx = 0; x_idx < x_disp; x_idx++) {
      for (int y_idx = y_disp; y_idx < y_length - y_disp; y_idx++) {
        data_(x_idx, y_idx) = value;
        data_(x_length - 1 - x_idx, y_idx) = value;
      }
    }
  }


  void add_neumann_bc() {
    // assumption: f'(x) = 0 at boundary
    //TODO: do it for other values, optimize it
    int x_length = data_.shape(0);
    int y_length = data_.shape(1);

    int x_disp = num_ghost_cells_x();
    int y_disp = num_ghost_cells_y();
    Scalar value;

    for (int x_idx = x_disp; x_idx < x_length - x_disp; x_idx++) {
      // left cols without corners
      value = data_(x_idx, y_disp);
      for (int y_idx = 0; y_idx < y_disp; y_idx++) {
        data_(x_idx, y_idx) = value;
      }
      // right cols without corners
      value = data_(x_idx, y_length - y_disp - 1);
      for (int y_idx = 0; y_idx < y_disp; y_idx++) {
        data_(x_idx, y_length - 1 - y_idx) = value;
      }
    }

    for (int y_idx = y_disp; y_idx < y_length - y_disp; y_idx++) {
      // top cols without corners
      value = data_(x_disp, y_idx);
      for (int x_idx = 0; x_idx < x_disp; x_idx++) {
        data_(x_idx, y_idx) = value;
      }

      // bottom cols without corners
      value = data_(x_length - 1 - x_disp, y_idx);
      for (int x_idx = 0; x_idx < x_disp; x_idx++) {
        data_(x_length - x_idx - 1, y_idx) = value;
      }

    }

    // top left corners
    Scalar value_tl = data_(x_disp, y_disp);
    Scalar value_tr = data_(x_length - x_disp - 1, y_disp);
    Scalar value_bl = data_(x_disp, y_length - y_disp - 1);
    Scalar value_br = data_(x_length - x_disp - 1, y_length - y_disp - 1);
    for (int x_idx = 0; x_idx < x_disp; x_idx++) {
      for (int y_idx = 0; y_idx < y_disp; y_idx++) {
        data_(x_idx, y_idx) = value_tl;
        data_(x_length - x_idx - 1, y_idx) = value_tr;
        data_(x_idx, y_length - y_idx - 1) = value_bl;
        data_(x_length - x_idx - 1, y_length - y_idx - 1) = value_br;
      }
    }
  }

  zisa::array<Scalar, 2> data_;
  const zisa::array_const_view<Scalar, 2> kernel_;
  const BoundaryCondition bc_;
};

#endif // PDE_BASE_HPP_
