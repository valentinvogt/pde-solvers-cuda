#ifndef PDE_WAVE_HPP_
#define PDE_WAVE_HPP_

// TODO: add initial derivative data, correct apply function

#include <pde_base.hpp>

template <typename Scalar, typename BoundaryCondition, typename Function>
class PDEWave : public virtual PDEBase<Scalar, BoundaryCondition> {
public:
  // TODO: add derivative
  PDEWave(unsigned Nx, unsigned Ny, const zisa::device_type memory_location,
          BoundaryCondition bc, Function f)
      : PDEBase<Scalar, BoundaryCondition>(Nx, Ny, memory_location, bc),
        func_(f) {}

  void apply(Scalar dt) override {

    zisa::array<Scalar, 2> tmp(this->data_.shape(), this->data_.device());
    // TODO: add cuda implementation, handle 1/dx^2, add f
    convolve_sigma_add_f(tmp.view(), this->data_.const_view(),
                         this->sigma_values_.const_view(), 0.01, func_);
    // TODO: add
    add_arrays(this->data_.view(), tmp.const_view());
    PDEBase<Scalar, BoundaryCondition>::add_bc();
  }

protected:
  Function func_;
  // add
  zisa::array<Scalar, 2> deriv;
};

#endif // PDE_WAVE_HPP_
