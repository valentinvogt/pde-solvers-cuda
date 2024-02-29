#ifndef PDE_BASE_HPP_
#define PDE_BASE_HPP_

#include <convolution.hpp>
#include <zisa/memory/array.hpp>

template <typename Scalar, typename BoundaryCondition> class PDEBase {
public:
  using scalar_t = Scalar;

  PDEBase(unsigned Nx, unsigned Ny,
          const zisa::array_const_view<scalar_t, 2> &kernel, BoundaryCondition BC)
      : data_(zisa::shape_t<2>(Nx + 2 * (kernel.shape(0) / 2),
                               Ny + 2 * (kernel.shape(1) / 2)),
              kernel.memory_location()),
        kernel_(kernel.shape(), kernel.memory_location()) {
    zisa::copy(kernel_, kernel);
  }

  void apply() {
    zisa::array<scalar_t, 2> tmp(data_.shape(), data_.device());
    convolve(tmp.view(), data_.const_view(), kernel_.const_view());
    zisa::copy(data_, tmp);
    // TODO apply BC
  }

  unsigned num_ghost_cells(unsigned dir) { return kernel_.shape(dir) / 2; }
  unsigned num_ghost_cells_x() { return num_ghost_cells(0); }
  unsigned num_ghost_cells_y() { return num_ghost_cells(1); }

  // TODO
  void read_initial_conditions(/* datatype */) {
    // where do i have to be able to read initial conditions from?
    // a file? in which format?
  }
  

protected:
  zisa::array<scalar_t, 2> data_;
  zisa::array<scalar_t, 2> kernel_;
};

#endif // PDE_BASE_HPP_
