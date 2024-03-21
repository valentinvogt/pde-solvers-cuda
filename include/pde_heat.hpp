#ifndef PDE_HEAT_HPP_
#define PDE_HEAT_HPP_

#include "zisa/memory/device_type.hpp"
#include <pde_base.hpp>

template <typename Scalar, typename BoundaryCondition, typename Function>
class PDEHeat : public virtual PDEBase<Scalar, BoundaryCondition> {
public:
  PDEHeat(unsigned Nx, unsigned Ny, const zisa::device_type memory_location,
          BoundaryCondition bc, Function f)
      : PDEBase<Scalar, BoundaryCondition>(Nx, Ny, memory_location, bc),
        func_(f) {}

  void apply() override {
    if (!this->ready_) {
      std::cerr << "Heat solver is not ready yet! Read data first" << std::endl;
      return;
    }

    zisa::array<Scalar, 2> tmp(this->data_.shape(), this->data_.device());
    // TODO: add cuda implementation, handle 1/dx^2, add f
    convolve_sigma_add_f(tmp.view(), this->data_.const_view(),
                         this->sigma_values_vertical_.const_view(),
                         this->sigma_values_horizontal_.const_view(), 0.01,
                         func_);
    // TODO:
    add_arrays(this->data_.view(), tmp.const_view());
    PDEBase<Scalar, BoundaryCondition>::add_bc();
  }

protected:
  Function func_;
};

#endif // PDE_HEAT_HPP_
