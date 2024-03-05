#ifndef PDE_BASE_HPP_
#define PDE_BASE_HPP_

#include <convolution.hpp>
#include <zisa/memory/array.hpp>
#include <kernel_base.hpp>

template <typename Scalar, typename BoundaryCondition, int rows, int cols>
class PDEBase {
public:
  using scalar_t = Scalar;

  PDEBase(unsigned Nx, unsigned Ny,
          const KernelBase<Scalar, rows, cols> &kernel, BoundaryCondition BC)
      : data_(zisa::shape_t<2>(Nx + 2 * (rows / 2),
                               Ny + 2 * (cols / 2)),
              kernel.memory_location()),
        kernel_(kernel) {}

  void apply() {
    zisa::array<scalar_t, 2> tmp(data_.shape(), data_.device());
    convolve(tmp.view(), data_.const_view(), this->kernel_);
    zisa::copy(data_, tmp);
    // TODO apply BC
  }

  unsigned num_ghost_cells(unsigned dir) { return kernel_.shape(dir) / 2; }
  unsigned num_ghost_cells_x() { return num_ghost_cells(0); }
  unsigned num_ghost_cells_y() { return num_ghost_cells(1); }

  // TODO
  void read_initial_conditions(std::ifstream &input) {
    // where do i have to be able to read initial conditions from?
    // a file? in which format?
  }
  

protected:
  zisa::array<scalar_t, 2> data_;
  const KernelBase<Scalar, rows, cols> kernel_;
};

#endif // PDE_BASE_HPP_
